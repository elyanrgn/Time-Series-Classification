"""
transformer_pretraining.py — Pré-entraînement IndPatchTST sur ETTh1 (régression)

CORRECTIONS par rapport à la version dégradée :
  - Suppression du CLS token : il n'est pas supervisé pendant la régression ETTh1
    → le CLS ne peut pas apprendre à agréger pendant le pré-entraînement
    → on garde le mean pooling, simple et efficace
  - Suppression de norm_first=True : change le comportement du transformer
    pré-entraîné et invalide le transfert de poids
  - pos_encoding de taille num_patches (cohérent avec create_patches)
  - n_heads=4 (diviseur de d_model=128), meilleur que n_heads=1

ALIGNEMENT WINDOW :
  - window=36 = LSST_WINDOW pour que les pos_encodings soient transférables
    (même nombre de patches → même pos_encoding shape)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import optuna


# ─────────────────────────── Données ETTh1 ──────────────────────────────────


def load_etth1(csv_path, use_time_feat=True):
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
    def __init__(
        self, csv_path, window, horizon, use_time_feat=True, start=0, end=None
    ):
        feats = load_etth1(csv_path, use_time_feat=use_time_feat)
        feats = feats[start:]
        if end is not None:
            feats = feats[:end]
        self.feats = feats
        self.window = window
        self.horizon = horizon
        self.max_start = len(feats) - window - horizon + 1
        if self.max_start < 1:
            raise ValueError("Window + horizon exceeds series length")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        past = self.feats[idx : idx + self.window]
        future = self.feats[idx + self.window : idx + self.window + self.horizon, -1:]
        return torch.from_numpy(past), torch.from_numpy(future)


def build_etth1_dataloaders(csv_path, window=36, horizon=24, batch_size=64, split=0.8):
    """
    window=36 est intentionnellement aligné sur LSST_WINDOW=36.
    Cela garantit que les positional embeddings (shape : num_patches)
    sont directement transférables sur LSST sans redimensionnement.
    """
    dataset = ETTh1Dataset(csv_path, window, horizon)
    n = len(dataset)
    n_train = int(split * n)

    train_ds = ETTh1Dataset(csv_path, window, horizon, end=n_train)
    valid_ds = ETTh1Dataset(csv_path, window, horizon, start=n_train)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False
    )
    sample_past, _ = train_ds[0]
    return train_dl, valid_dl, sample_past.shape[-1]


# ─────────────────────────── RevIN ──────────────────────────────────────────


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    Correction clé : mean et std sont retournés explicitement par norm()
    et passés en argument à denorm(), au lieu d'être stockés comme attributs
    d'instance. Le stockage en attribut causait deux problèmes :
      1. En mode eval (torch.no_grad), entre deux batchs de tailles différentes,
         self.mean/self.std pouvaient pointer vers le mauvais batch ou être None.
      2. En cas de séries quasi-constantes (std ≈ 0), la division dans denorm
         produisait des inf/nan. On clippe maintenant std à eps minimum.
    """

    def __init__(self, num_features, target_channel=-1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        self.target_channel = target_channel

    def norm(self, x):
        """x : (B, T, C) → x_norm : (B, T, C), retourne aussi (mean, std) pour denorm."""
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        std = x.std(dim=1, keepdim=True).clamp(
            min=self.eps
        )  # (B, 1, C) — clamp évite /0
        x_norm = (x - mean) / std * self.gamma + self.beta
        return x_norm, mean, std

    def denorm(self, x, mean, std):
        """x : (B, T, 1) — dénormalise le canal cible avec mean/std reçus de norm()."""
        tc = self.target_channel
        # gamma/beta sont de shape (1,1,C) ; on indexe le canal cible
        x_denorm = (x - self.beta[:, :, tc : tc + 1]) / self.gamma[
            :, :, tc : tc + 1
        ].clamp(min=self.eps)
        return x_denorm * std[:, :, tc : tc + 1] + mean[:, :, tc : tc + 1]

    def forward(self, x, mode, mean=None, std=None):
        """Compatibilité ascendante : accepte toujours l'ancienne interface à 2 args."""
        if mode == "norm":
            x_norm, mean, std = self.norm(x)
            return x_norm, mean, std
        elif mode == "denorm":
            if mean is None or std is None:
                raise ValueError(
                    "RevIN.forward(denorm) requiert mean et std explicites."
                )
            return self.denorm(x, mean, std)
        raise ValueError("mode must be 'norm' or 'denorm'")


# ─────────────────────── IndPatchTST ─────────────────────────────────────────


class IndPatchTST(nn.Module):
    """
    IndPatchTST : traitement canal-indépendant (channel-independent).

    Par rapport à la version originale, une seule correction d'architecture :
      - n_heads=4 par défaut (au lieu de 1) pour une meilleure attention
        multi-têtes, tout en restant compatible avec d_model=128 (128 % 4 == 0)

    Le reste est identique à l'original pour garantir la stabilité
    du pré-entraînement et la validité du transfert de poids.
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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.revin = revin

        self.num_patches = (seq_len - patch_len) // stride + 1

        if revin:
            self.revin_layer = RevIN(num_features=num_features)

        # Patch embedding univarié : patch_len → d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding appris — taille num_patches, cohérent avec create_patches
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Transformer standard Post-LN (identique à l'original)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Tête de régression (remplacée par Identity au fine-tuning)
        self.head = nn.Linear(d_model, pred_len)

    def create_patches(self, x):
        """x : (B*C, T, 1) → (B*C, num_patches, patch_len)"""
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patches.append(x[:, i : i + self.patch_len, 0])
        return torch.stack(patches, dim=1)

    def forward_features(self, x):
        """
        (B, T, C) → (B, d_model)  via mean pooling sur canaux × patches.

        Pourquoi mean pooling (et pas CLS token) :
          Pendant le pré-entraînement sur ETTh1 (régression), aucun signal
          ne supervise directement un CLS token. Il n'apprendrait rien d'utile.
          Le mean pooling est cohérent avec la supervision de régression car
          chaque patch contribue au vecteur de sortie utilisé pour prédire.
        """
        if self.revin:
            # norm() retourne explicitement (mean, std) — stockés pour denorm dans forward()
            x, self._revin_mean, self._revin_std = self.revin_layer.norm(x)

        B, T, C = x.shape
        x_chan = x.permute(0, 2, 1).reshape(B * C, T, 1)

        patches = self.create_patches(x_chan)  # (B*C, P, patch_len)
        patches = self.patch_embedding(patches)  # (B*C, P, d_model)
        patches = patches + self.pos_encoding  # (B*C, P, d_model)
        patches = self.transformer(patches)  # (B*C, P, d_model)

        patches = patches.reshape(B, C, -1, self.d_model)
        feats = patches.mean(dim=(1, 2))  # (B, d_model)
        return feats

    def forward(self, x):
        feats = self.forward_features(x)  # appelle revin norm si besoin
        out = self.head(feats).unsqueeze(-1)
        if self.revin:
            # denorm avec les mean/std sauvegardés lors du norm dans forward_features
            out = self.revin_layer.denorm(out, self._revin_mean, self._revin_std)
        return out


# ──────────────────────── Boucles d'entraînement ────────────────────────────


def train_epoch(model, dataloader, optimizer, criterion, device="cuda"):
    model.to(device)
    model.train()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        pred = model(past)
        loss = criterion(pred, future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device="cuda"):
    model.to(device)
    model.eval()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        loss = criterion(model(past), future)
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


def train_and_valid_loop(
    model,
    train_dl,
    valid_dl,
    optimizer,
    criterion,
    n_epochs,
    device="cuda",
    scheduler=None,
):
    logs = {"train_loss": [], "valid_loss": []}
    best_loss, best_state = float("inf"), None
    for epoch in range(n_epochs):
        tr = train_epoch(model, train_dl, optimizer, criterion, device)
        vl = eval_epoch(model, valid_dl, criterion, device)
        if scheduler:
            scheduler.step()
        logs["train_loss"].append(tr)
        logs["valid_loss"].append(vl)
        print(f"Epoch {epoch + 1:02d} | train={tr:.4f} | valid={vl:.4f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return logs


# ──────────────────────── Construction / recherche ──────────────────────────


def build_model_from_config(config, num_features, window, horizon):
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
    )


def objective(trial, train_dl, valid_dl, window, horizon, device, max_epochs=20):
    """
    Contrainte clé : n_heads doit diviser d_model.
    Espace restreint pour RTX 4060 (8 GB VRAM).
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
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
    }

    num_features = train_dl.dataset.feats.shape[1]
    model = build_model_from_config(config, num_features, window, horizon).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-4
    )
    criterion = nn.MSELoss()

    best_valid_loss, patience_counter, patience = float("inf"), 0, 4
    for epoch in range(max_epochs):
        train_epoch(model, train_dl, optimizer, criterion, device)
        valid_loss = eval_epoch(model, valid_dl, criterion, device)
        trial.report(valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_valid_loss


def bayesian_search(
    train_dl, valid_dl, window, horizon, device, n_trials=30, max_epochs=20
):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4),
    )
    study.optimize(
        lambda trial: objective(
            trial, train_dl, valid_dl, window, horizon, device, max_epochs
        ),
        n_trials=n_trials,
    )
    print("\n=== Meilleure config bayésienne ===")
    print(study.best_params)
    print(f"Meilleure valid loss: {study.best_value:.4f}")
    return study.best_params, study.best_value


# ─────────────────────────────── Main ────────────────────────────────────────

if __name__ == "__main__":
    import os

    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # window=36 = LSST_WINDOW — alignement intentionnel des pos_encodings
    WINDOW, HORIZON = 36, 24

    train_dl, valid_dl, input_dim = build_etth1_dataloaders(
        "..\\data\\ETTh1.csv", window=WINDOW, horizon=HORIZON, batch_size=128
    )
    print(f"Input dim ETTh1 : {input_dim} | window={WINDOW} | horizon={HORIZON}")

    best_params, best_loss = bayesian_search(
        train_dl, valid_dl, WINDOW, HORIZON, device, n_trials=100, max_epochs=20
    )

    # Ré-entraînement long avec la meilleure config
    num_features = train_dl.dataset.feats.shape[1]
    best_model = build_model_from_config(best_params, num_features, WINDOW, HORIZON).to(
        device
    )
    optimizer = torch.optim.AdamW(
        best_model.parameters(), lr=best_params["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    print("\n── Entraînement final (50 epochs) ──")
    train_and_valid_loop(
        best_model,
        train_dl,
        valid_dl,
        optimizer,
        criterion,
        n_epochs=50,
        device=device,
        scheduler=scheduler,
    )

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": best_params,
            "window": WINDOW,
            "horizon": HORIZON,
            "valid_loss": best_loss,
        },
        "models/best_indpatch_tst_optuna2.pth",
    )
    print("✅ Sauvé : models/best_indpatch_tst_optuna2.pth")
    print(f"   Config : {best_params}")
