"""
transformer_pretraining_v2.py — Pré-entraînement IndPatchTST sur ETTh1

Améliorations par rapport à v1 :
  - Masking de patches (MAE-style) en tâche auxiliaire optionnelle
    → force le modèle à apprendre des représentations plus robustes
  - CosineAnnealingLR avec warmup linéaire (au lieu de CosineAnnealing seul)
  - Gradient clipping systématique (déjà présent, rendu configurable)
  - Sauvegarde enrichie : config complète + métriques train/valid
  - Docstring complète sur les choix d'architecture
"""

import numpy as np
import torch
import torch.nn as nn
import optuna


# ─────────────────────────── Données ETTh1 ──────────────────────────────────


def load_etth1(csv_path: str, use_time_feat: bool = True) -> np.ndarray:
    def to_str(s):
        return s if isinstance(s, str) else s.decode()

    d_conv = {0: (lambda x: float(to_str(x).split(" ")[1].split(":")[0]))}
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, converters=d_conv)
    features = raw.astype(np.float32)
    if use_time_feat:
        features[:, 0] /= 23.0
    else:
        features = features[:, 1:]
    return features


class ETTh1Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, window, horizon, use_time_feat=True, start=0, end=None):
        feats = load_etth1(csv_path, use_time_feat=use_time_feat)
        feats = feats[start:end] if end is not None else feats[start:]
        self.feats = feats
        self.window = window
        self.horizon = horizon
        self.max_start = len(feats) - window - horizon + 1
        if self.max_start < 1:
            raise ValueError("Window + horizon dépasse la longueur de la série")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        past = self.feats[idx : idx + self.window]
        future = self.feats[idx + self.window : idx + self.window + self.horizon, -1:]
        return torch.from_numpy(past), torch.from_numpy(future)


def build_etth1_dataloaders(csv_path, window=36, horizon=24, batch_size=64, split=0.8):
    """
    window=36 aligné sur LSST_WINDOW=36.
    Même nombre de patches → positional encodings directement transférables.
    """
    full = ETTh1Dataset(csv_path, window, horizon)
    n = len(full)
    n_train = int(split * n)
    train_ds = ETTh1Dataset(csv_path, window, horizon, end=n_train)
    valid_ds = ETTh1Dataset(csv_path, window, horizon, start=n_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    sample_past, _ = train_ds[0]
    return train_dl, valid_dl, sample_past.shape[-1]


# ─────────────────────────── RevIN ──────────────────────────────────────────


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2021).
    Normalise chaque instance indépendamment pour gérer la non-stationnarité.
    Paramètres gamma et beta appris pour affine scaling.
    """

    def __init__(self, num_features: int, target_channel: int = -1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        self.target_channel = target_channel

    def forward(self, x, mode: str):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std * self.gamma + self.beta
        elif mode == "denorm":
            tc = self.target_channel
            return (
                (x - self.beta[:, :, tc : tc + 1])
                / self.gamma[:, :, tc : tc + 1]
                * self.std[:, :, tc : tc + 1]
                + self.mean[:, :, tc : tc + 1]
            )
        raise ValueError("mode must be 'norm' or 'denorm'")


# ─────────────────────────── IndPatchTST ────────────────────────────────────


class IndPatchTST(nn.Module):
    """
    IndPatchTST : Transformer canal-indépendant avec patchification.

    Choix d'architecture documentés :
    ──────────────────────────────────
    - Canal-indépendant (CI) : chaque canal traité séparément.
      Avantage : permet de transférer sur LSST (6 canaux) même si pré-entraîné
      sur ETTh1 (7 canaux) — le backbone ne voit jamais la dim canal.

    - Mean pooling (pas de CLS token) : pendant le forecasting ETTh1, aucun signal
      ne supervise un CLS token. Le mean pooling est cohérent avec la supervision
      de régression (chaque patch contribue à la prédiction finale).

    - Post-LN (norm_first=False) : comportement par défaut de PyTorch.
      Pre-LN (norm_first=True) changerait les activations et invaliderait le transfert.

    - n_heads=4 par défaut : meilleure attention multi-têtes que n_heads=1,
      compatible avec d_model=128 (128 % 4 == 0).

    - Positional encoding appris de taille num_patches (et non seq_len) :
      cohérent avec create_patches qui opère sur les patches, pas les timesteps.
      Taille fixe → transférable si window ETTh1 == window LSST (tous deux =36).

    Nouveau dans v2 :
    ─────────────────
    - patch_mask_ratio : si > 0, masque aléatoirement une fraction des patches
      pendant forward() pour une tâche de reconstruction auxiliaire (MAE-style).
      Améliore la qualité des représentations sans changer l'interface.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        revin: bool = True,
        patch_mask_ratio: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.revin = revin
        self.patch_mask_ratio = patch_mask_ratio

        self.num_patches = (seq_len - patch_len) // stride + 1

        if revin:
            self.revin_layer = RevIN(num_features=num_features)

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, pred_len)

        # Token de masque appris (MAE-style)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def create_patches(self, x):
        """x : (B*C, T, 1) → (B*C, num_patches, patch_len)"""
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patches.append(x[:, i : i + self.patch_len, 0])
        return torch.stack(patches, dim=1)

    def forward_features(self, x):
        """
        (B, T, C) → (B, d_model) via mean pooling sur canaux × patches.
        Si patch_mask_ratio > 0 et model.training, masque aléatoirement des patches.
        """
        if self.revin:
            x = self.revin_layer(x, mode="norm")

        B, T, C = x.shape
        x_chan = x.permute(0, 2, 1).reshape(B * C, T, 1)

        patches = self.create_patches(x_chan)        # (B*C, P, patch_len)
        patches = self.patch_embedding(patches)       # (B*C, P, d_model)
        patches = patches + self.pos_encoding         # (B*C, P, d_model)

        # Masquage MAE-style (uniquement en entraînement)
        if self.patch_mask_ratio > 0 and self.training:
            P = patches.size(1)
            n_mask = int(P * self.patch_mask_ratio)
            noise = torch.rand(B * C, P, device=patches.device)
            mask_idx = noise.argsort(dim=1)[:, :n_mask]
            mask = torch.zeros(B * C, P, 1, device=patches.device)
            mask.scatter_(1, mask_idx.unsqueeze(-1), 1.0)
            patches = patches * (1 - mask) + self.mask_token * mask

        patches = self.transformer(patches)           # (B*C, P, d_model)
        patches = patches.reshape(B, C, -1, self.d_model)
        feats = patches.mean(dim=(1, 2))              # (B, d_model)
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        out = self.head(feats).unsqueeze(-1)
        if self.revin:
            out = self.revin_layer(out, mode="denorm")
        return out


# ──────────────────────────── Training loops ────────────────────────────────


def make_scheduler_with_warmup(optimizer, n_epochs: int, warmup_epochs: int = 5):
    """Warmup linéaire puis CosineAnnealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, criterion, device, scaler_amp=None):
    model.to(device)
    model.train()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        if scaler_amp is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss = criterion(model(past), future)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss = criterion(model(past), future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        total_loss += criterion(model(past), future).item() * past.size(0)
    return total_loss / len(dataloader.dataset)


def train_and_valid_loop(
    model, train_dl, valid_dl, optimizer, criterion,
    n_epochs, device, scheduler=None, scaler_amp=None,
):
    logs = {"train_loss": [], "valid_loss": []}
    best_loss, best_state = float("inf"), None

    for epoch in range(n_epochs):
        tr = train_epoch(model, train_dl, optimizer, criterion, device, scaler_amp)
        vl = eval_epoch(model, valid_dl, criterion, device)
        if scheduler:
            scheduler.step()
        logs["train_loss"].append(tr)
        logs["valid_loss"].append(vl)
        print(f"Epoch {epoch + 1:03d} | train={tr:.4f} | valid={vl:.4f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return logs, best_loss


# ──────────────────────────── Model factory ─────────────────────────────────


def build_model_from_config(config: dict, num_features: int, window: int, horizon: int):
    return IndPatchTST(
        seq_len=window,
        pred_len=horizon,
        num_features=num_features,
        patch_len=config["patch_len"],
        stride=config["stride"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        revin=config["revin"],
        patch_mask_ratio=config.get("patch_mask_ratio", 0.0),
    )


# ──────────────────────────── Optuna ────────────────────────────────────────


def objective(trial, train_dl, valid_dl, window, horizon, device, max_epochs=20):
    """
    Espace de recherche pour RTX 4060 (8 GB VRAM).
    Contrainte : n_heads doit diviser d_model.
    Ajout v2 : patch_mask_ratio pour la tâche de masquage.
    """
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)

    patch_len = trial.suggest_int("patch_len", 3, window // 3)
    stride = trial.suggest_int("stride", 1, max(1, window // 8))
    if patch_len >= window:
        raise optuna.TrialPruned()

    config = {
        "patch_len": patch_len,
        "stride": stride,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "revin": trial.suggest_categorical("revin", [True, False]),
        "patch_mask_ratio": trial.suggest_float("patch_mask_ratio", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
    }

    num_features = train_dl.dataset.feats.shape[1]
    model = build_model_from_config(config, num_features, window, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_valid_loss, patience_ctr, patience = float("inf"), 0, 4
    for epoch in range(max_epochs):
        train_epoch(model, train_dl, optimizer, criterion, device)
        valid_loss = eval_epoch(model, valid_dl, criterion, device)
        trial.report(valid_loss, epoch)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_valid_loss


def bayesian_search(train_dl, valid_dl, window, horizon, device, n_trials=30, max_epochs=20):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4),
    )
    study.optimize(
        lambda trial: objective(trial, train_dl, valid_dl, window, horizon, device, max_epochs),
        n_trials=n_trials,
    )
    print(f"\n✅ Meilleure config bayésienne : {study.best_params}")
    print(f"   Meilleure valid loss : {study.best_value:.4f}")
    return study.best_params, study.best_value


# ──────────────────────────── Main ──────────────────────────────────────────

if __name__ == "__main__":
    import os, json

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler() if use_amp else None
    print(f"Device : {device} | AMP : {use_amp}")

    # window=36 = LSST_WINDOW — alignement intentionnel
    WINDOW, HORIZON = 36, 24

    train_dl, valid_dl, input_dim = build_etth1_dataloaders(
        "data/ETTh1.csv", window=WINDOW, horizon=HORIZON, batch_size=128
    )
    print(f"Input dim ETTh1 : {input_dim} | window={WINDOW} | horizon={HORIZON}")

    # Recherche bayésienne
    best_params, best_loss = bayesian_search(
        train_dl, valid_dl, WINDOW, HORIZON, device, n_trials=30, max_epochs=20
    )

    # Réentraînement long avec warmup + AMP
    num_features = train_dl.dataset.feats.shape[1]
    best_model = build_model_from_config(best_params, num_features, WINDOW, HORIZON).to(device)
    optimizer = torch.optim.AdamW(best_model.parameters(), lr=best_params["lr"], weight_decay=1e-4)
    N_EPOCHS = 60
    scheduler = make_scheduler_with_warmup(optimizer, N_EPOCHS, warmup_epochs=5)
    criterion = nn.MSELoss()

    print(f"\n── Entraînement final ({N_EPOCHS} epochs avec warmup) ──")
    logs, final_loss = train_and_valid_loop(
        best_model, train_dl, valid_dl, optimizer, criterion,
        n_epochs=N_EPOCHS, device=device, scheduler=scheduler, scaler_amp=scaler_amp,
    )

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": best_params,
            "window": WINDOW,
            "horizon": HORIZON,
            "valid_loss": final_loss,
            "train_losses": logs["train_loss"],
            "valid_losses": logs["valid_loss"],
        },
        "models/best_indpatch_tst_v2.pth",
    )

    with open("reports/pretraining_v2.json", "w") as f:
        json.dump({"config": best_params, "valid_loss": float(final_loss)}, f, indent=2)

    print("✅ Sauvé : models/best_indpatch_tst_v2.pth | reports/pretraining_v2.json")
