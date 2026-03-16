"""
baseline_v2.py — CNN baseline pour la classification LSST

Améliorations par rapport à v1 :
  - Optuna bayésien (20 trials) au lieu de la gridsearch manuelle (4 configs)
    → exploration plus large et reproductible
  - CrossEntropyLoss avec class_weights (déséquilibre de classes)
  - CosineAnnealingLR avec warmup linéaire (même setup que PatchTST)
  - Rapport JSON automatique avec toutes les métriques
  - seed fixée pour reproductibilité complète
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import optuna
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader

from dataloader_v2 import prepare_lsst, build_dataloaders


# ──────────────────────────── Architecture ──────────────────────────────────


class CNNBaseline(nn.Module):
    """
    CNN 1D à 4 blocs convolutifs + global average pooling.

    Entrée : (B, T, C) → permutation → Conv1d sur (B, C, T)
    Sortie : logits (B, n_classes)
    """

    def __init__(
        self,
        n_features: int = 6,
        n_classes: int = 14,
        n_filters: list = None,
        kernel_size: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        if n_filters is None:
            n_filters = [64, 128, 256, 512]

        padding = kernel_size // 2
        layers = []
        in_ch = n_features
        for out_ch in n_filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_filters[-1], n_classes),
        )

    def forward(self, x):
        # x : (B, T, C) → (B, C, T) pour Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = self.pool(x)
        return self.classifier(x)


# ──────────────────────────── Training utils ────────────────────────────────


def make_scheduler_with_warmup(optimizer, n_epochs: int, warmup_epochs: int = 5):
    """
    Warmup linéaire sur warmup_epochs puis CosineAnnealing jusqu'à n_epochs.
    Même setup que IndPatchTST pour une comparaison équitable.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dl, optimizer, criterion, device, scaler_amp=None):
    model.train()
    total_loss = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler_amp is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dl.dataset)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    preds, labels = [], []
    for x, y in dl:
        logits = model(x.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    preds, labels = np.array(preds), np.array(labels)
    return (
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


def train_model(
    model,
    train_dl,
    val_dl,
    criterion,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 60,
    patience: int = 10,
    device=None,
    scaler_amp=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler_with_warmup(optimizer, n_epochs)

    best_f1, best_state, patience_ctr = -np.inf, None, 0

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler_amp)
        scheduler.step()
        val_acc, val_f1, _, _ = evaluate(model, val_dl, device)

        print(
            f"Epoch {epoch:03d} | loss={tr_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping epoch {epoch} (best val F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


# ──────────────────────────── Optuna search ─────────────────────────────────


def optuna_search(train_dl, val_dl, criterion, n_features, n_classes, device, n_trials=20):
    """
    Recherche bayésienne des hyperparamètres du CNN.
    Équivalent à ce que PatchTST fait avec Optuna sur ETTh1 → comparaison équitable.
    """

    def objective(trial):
        n_filters_base = trial.suggest_categorical("n_filters_base", [32, 64])
        n_layers = trial.suggest_int("n_layers", 2, 4)
        n_filters = [n_filters_base * (2 ** i) for i in range(n_layers)]

        model = CNNBaseline(
            n_features=n_features,
            n_classes=n_classes,
            n_filters=n_filters,
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            dropout=trial.suggest_float("dropout", 0.2, 0.6),
        ).to(device)

        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        _, best_f1 = train_model(
            model, train_dl, val_dl, criterion,
            lr=lr, weight_decay=wd,
            n_epochs=40, patience=8, device=device,
        )
        return best_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    # Réduit les logs Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    print(f"\n✅ Meilleure config CNN : {study.best_params}")
    print(f"   Meilleur val F1      : {study.best_value:.4f}")
    return study.best_params, study.best_value


# ──────────────────────────── Main ──────────────────────────────────────────

if __name__ == "__main__":
    from tslearn.datasets import UCR_UEA_datasets

    # Reproductibilité
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler() if use_amp else None
    print(f"Device : {device} | AMP : {use_amp}")

    # 1. Données
    print("\n── Chargement LSST ──")
    raw = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = raw.load_dataset("LSST")
    data = prepare_lsst(X_train, y_train, X_test, y_test, random_state=SEED)

    # class_weights sur GPU
    class_weights = data["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    train_dl, val_dl, test_dl = build_dataloaders(
        data, batch_size=32, augment_train=True
    )

    # 2. Recherche Optuna
    print("\n── Recherche Optuna CNN (20 trials) ──")
    best_params, _ = optuna_search(
        train_dl, val_dl, criterion,
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        device=device,
        n_trials=20,
    )

    # 3. Réentraînement final avec la meilleure config
    print("\n── Entraînement final CNN ──")
    nb = best_params["n_filters_base"]
    nl = best_params["n_layers"]
    best_model = CNNBaseline(
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        n_filters=[nb * (2 ** i) for i in range(nl)],
        kernel_size=best_params["kernel_size"],
        dropout=best_params["dropout"],
    )
    best_model, _ = train_model(
        best_model, train_dl, val_dl, criterion,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        n_epochs=80,
        patience=12,
        device=device,
        scaler_amp=scaler_amp,
    )

    # 4. Évaluation finale
    test_acc, test_f1, preds, labels = evaluate(best_model, test_dl, device)
    print(f"\n{'='*50}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test F1 macro : {test_f1:.4f}")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=[str(c) for c in data["le"].classes_]))

    # 5. Sauvegarde
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": best_params,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "le_classes": data["le"].classes_,
            "scaler_mean": data["scaler"].mean_,
            "scaler_std": data["scaler"].scale_,
        },
        "models/cnn_baseline_v2.pth",
    )

    report = {
        "model": "CNNBaseline_v2",
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1),
        "best_params": best_params,
    }
    with open("reports/cnn_baseline_v2.json", "w") as f:
        json.dump(report, f, indent=2)

    print("💾 Sauvegardé : models/cnn_baseline_v2.pth | reports/cnn_baseline_v2.json")
