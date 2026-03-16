"""
adapting_to_classification_v2.py — Fine-tuning IndPatchTST (ETTh1 → LSST)

Améliorations par rapport à v1 :
  - CrossEntropyLoss avec class_weights (cohérent avec baseline_v2)
  - Tête de classification MLP 2 couches + LayerNorm
  - Rapport JSON automatique par stratégie (acc, f1, std sur 15 runs)
  - Protocole de fine-tuning documenté et simplifié

Protocole de transfert (2 phases) :
  Phase 1 — Warmup (backbone gelé ou partiellement gelé) :
    Stabilise la tête avant de toucher au backbone.
  Phase 2 — Dégel total avec LR différenciés (lr_backbone ≪ lr_head) :
    Adapte le backbone sans catastrophic forgetting.

Stratégies comparées :
  A — From scratch     : aucun poids pré-entraîné (contrôle)
  B — Head Only        : backbone complètement gelé
  C — Late Encoders    : 2 dernières couches + tête dégelées
  D — Full Fine-tune   : tout dégelé avec LR différenciés
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import optuna

from transformer_pretraining_v2 import IndPatchTST
from dataloader_v2 import prepare_lsst, build_dataloaders


# ──────────────────────────── Classifier ────────────────────────────────────


class IndPatchTSTClassifier(nn.Module):
    """
    Backbone IndPatchTST + tête MLP pour la classification.

    Tête v2 : LayerNorm → Linear(d_model, hidden_dim) → GELU → Dropout → Linear → n_classes
    Deux couches offrent plus de capacité qu'une seule Linear, avec régularisation forte.

    RevIN désactivé pour la classification : la normalisation StandardScaler
    est déjà appliquée en amont sur toutes les séries.
    """

    def __init__(
        self,
        window: int,
        num_features: int,
        num_classes: int,
        backbone_config: dict,
        pretrained_path: str = None,
        hidden_dim: int = 128,
        dropout_clf: float = 0.4,
    ):
        super().__init__()

        # Backbone
        self.backbone = IndPatchTST(
            seq_len=window,
            pred_len=1,
            num_features=num_features,
            **{k: v for k, v in backbone_config.items() if k not in ("lr",)},
        )

        # Chargement des poids pré-entraînés (partiel : shape-compatible only)
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            own = self.backbone.state_dict()
            compatible = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
            own.update(compatible)
            self.backbone.load_state_dict(own, strict=False)
            pct = 100 * len(compatible) / max(len(state), 1)
            print(f"  ✅ Transfert : {len(compatible)}/{len(state)} poids ({pct:.0f}%)")
        else:
            print("  ⚡ From scratch")

        # Désactiver RevIN (normalisation faite en amont)
        self.backbone.revin = False
        self.backbone.head = nn.Identity()

        d = backbone_config["d_model"]

        # Tête MLP v2 : 2 couches pour plus de capacité
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_clf),
            nn.Linear(hidden_dim, num_classes),
        )

        # Groupes de freeze pour le dégel progressif
        n_layers = backbone_config.get("n_layers", 4)
        mid = max(1, n_layers // 2)
        self.group_enc_late = nn.ModuleList(
            self.backbone.transformer.layers[mid:]
        )

    def forward(self, x):
        return self.classifier(self.backbone.forward_features(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_late(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.group_enc_late.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def count_trainable(self, label=""):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [{label}] {trainable:,}/{total:,} ({100*trainable/total:.1f}%) params entraînables")


# ──────────────────────────── Training loops ────────────────────────────────


def make_scheduler(optimizer, n_epochs: int, warmup: int = 3):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, n_epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dl, optimizer, criterion, device, augment=False, scaler_amp=None):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        if augment:
            x = x + torch.randn_like(x) * 0.02
            x = x * torch.empty(x.size(0), 1, 1).uniform_(0.9, 1.1).to(x.device)
        optimizer.zero_grad()
        if scaler_amp is not None:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)


@torch.no_grad()
def eval_epoch(model, dl, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)


def train_loop(
    model, train_dl, val_dl, optimizer, criterion, n_epochs, device,
    scheduler=None, augment=False, patience=10, scaler_amp=None,
):
    best_acc, best_state, patience_ctr = -1.0, None, 0
    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, device, augment, scaler_amp)
        vl_loss, vl_acc = eval_epoch(model, val_dl, criterion, device)
        if scheduler:
            scheduler.step()
        print(
            f"    Ep {epoch+1:02d}/{n_epochs} | "
            f"TrL={tr_loss:.4f} TrA={tr_acc:.3f} | VlL={vl_loss:.4f} VlA={vl_acc:.3f}"
        )
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    ⏹ Early stopping (best val acc={best_acc:.3f})")
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_acc


@torch.no_grad()
def final_evaluate(model, test_dl, device):
    model.eval()
    preds, labels = [], []
    for x, y in test_dl:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    preds, labels = np.array(preds), np.array(labels)
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro"), preds, labels


# ──────────────────────────── Optuna head tuning ────────────────────────────


def tune_head(train_dl, val_dl, backbone_config, n_features, n_classes, device,
              pretrained_path, scaler_amp, n_trials=20):
    """
    Cherche les meilleurs hyperparamètres de la tête de classification.
    Backbone config fixée (optimisée sur ETTh1).
    """
    WINDOW = backbone_config.get("seq_len", 36)

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.6)
        lr_head = trial.suggest_float("lr_head", 1e-4, 5e-3, log=True)
        lr_bb = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

        model = IndPatchTSTClassifier(
            WINDOW, n_features, n_classes, backbone_config,
            pretrained_path=pretrained_path,
            hidden_dim=hidden_dim, dropout_clf=dropout_clf,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Phase 1 : tête seule
        model.freeze_backbone()
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=wd)
        sch = make_scheduler(opt, 15, warmup=3)
        best_acc = 0.0
        for ep in range(15):
            train_epoch(model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp)
            sch.step()
            _, vl_acc = eval_epoch(model, val_dl, criterion, device)
            trial.report(vl_acc, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
            best_acc = max(best_acc, vl_acc)

        # Phase 2 : dégel total court
        model.unfreeze_all()
        opt2 = torch.optim.AdamW(
            [{"params": model.backbone.parameters(), "lr": lr_bb},
             {"params": model.classifier.parameters(), "lr": lr_head * 0.1}],
            weight_decay=wd,
        )
        sch2 = make_scheduler(opt2, 8, warmup=2)
        for ep in range(8):
            train_epoch(model, train_dl, opt2, criterion, device, augment=False, scaler_amp=scaler_amp)
            sch2.step()
            _, vl_acc = eval_epoch(model, val_dl, criterion, device)
            best_acc = max(best_acc, vl_acc)

        return best_acc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"\n✅ Meilleurs hyperparamètres tête : {study.best_params}")
    return study.best_params


# ──────────────────────────── Experiment runner ─────────────────────────────


def run_experiment(
    seed, X_train, y_train_enc, X_test, y_test_enc,
    backbone_config, head_params, n_classes, n_features,
    window, device, class_weights, pretrained_path, scaler_amp,
):
    """Lance les 4 stratégies de fine-tuning pour un seed donné."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.1
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=seed
    )

    def make_dl(X, y, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
            batch_size=32, shuffle=shuffle, drop_last=shuffle,
        )

    train_dl = make_dl(X_tr, y_tr, shuffle=True)
    val_dl = make_dl(X_val, y_val, shuffle=False)
    test_dl = make_dl(X_test, y_test_enc, shuffle=False)

    lr_h = head_params["lr_head"]
    lr_bb = head_params["lr_backbone"]
    wd = head_params["weight_decay"]
    hd = head_params["hidden_dim"]
    do = head_params["dropout_clf"]

    strategies = {
        "A_scratch":   ("From scratch",              None,            "scratch",  60),
        "B_head_only": ("Head only",                 pretrained_path, "head",     35),
        "C_late_enc":  ("Late encoders + head",      pretrained_path, "late",     35),
        "D_full":      ("Full fine-tune (LR diff.)", pretrained_path, "full",     35),
    }

    results = {}
    for key, (label, ppath, strategy, n_ep) in strategies.items():
        print(f"\n{'─'*55}\n  Stratégie : {label}\n{'─'*55}")

        model = IndPatchTSTClassifier(
            window, n_features, n_classes, backbone_config,
            pretrained_path=ppath, hidden_dim=hd, dropout_clf=do,
        ).to(device)

        # ── Phase 1 : warmup ────────────────────────────────────────────────
        if strategy == "head":
            model.freeze_backbone()
            model.count_trainable("Head only")
            opt1 = torch.optim.AdamW(model.classifier.parameters(), lr=lr_h, weight_decay=wd)
        elif strategy == "late":
            model.unfreeze_late()
            model.count_trainable("Late enc")
            opt1 = torch.optim.AdamW(
                [{"params": model.group_enc_late.parameters(), "lr": lr_bb},
                 {"params": model.classifier.parameters(), "lr": lr_h}],
                weight_decay=wd,
            )
        elif strategy == "full":
            model.unfreeze_all()
            model.count_trainable("Full")
            opt1 = torch.optim.AdamW(
                [{"params": model.backbone.parameters(), "lr": lr_bb},
                 {"params": model.classifier.parameters(), "lr": lr_h}],
                weight_decay=wd,
            )
        else:  # scratch
            model.unfreeze_all()
            model.count_trainable("Scratch")
            opt1 = torch.optim.AdamW(model.parameters(), lr=lr_h, weight_decay=wd)

        sch1 = make_scheduler(opt1, n_ep, warmup=5)
        print(f"\n  ▶ Phase 1 — {n_ep} epochs")
        train_loop(
            model, train_dl, val_dl, opt1, criterion, n_ep, device,
            scheduler=sch1, augment=(strategy not in ("full", "scratch")),
            patience=12, scaler_amp=scaler_amp,
        )

        # ── Phase 2 : dégel total (sauf scratch et full) ─────────────────────
        if strategy in ("head", "late"):
            n_ep2 = 25
            print(f"\n  ▶ Phase 2 — dégel total ({n_ep2} epochs, lr_bb={lr_bb:.2e})")
            model.unfreeze_all()
            opt2 = torch.optim.AdamW(
                [{"params": model.backbone.parameters(), "lr": lr_bb},
                 {"params": model.classifier.parameters(), "lr": lr_h * 0.1}],
                weight_decay=wd,
            )
            sch2 = make_scheduler(opt2, n_ep2, warmup=3)
            train_loop(
                model, train_dl, val_dl, opt2, criterion, n_ep2, device,
                scheduler=sch2, augment=False, patience=10, scaler_amp=scaler_amp,
            )

        acc, f1, preds, labels = final_evaluate(model, test_dl, device)
        results[key] = {"label": label, "acc": acc, "f1": f1}
        print(f"\n  ✅ [{key}] Acc={acc:.4f} | F1={f1:.4f}")

    return results


def run_statistics(n_runs, base_seed, **kwargs):
    all_results = defaultdict(lambda: {"acc": [], "f1": []})
    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n{'#'*60}\n  RUN {i+1}/{n_runs}  (seed={seed})\n{'#'*60}")
        for key, metrics in run_experiment(seed=seed, **kwargs).items():
            all_results[key]["acc"].append(metrics["acc"])
            all_results[key]["f1"].append(metrics["f1"])
            all_results[key]["label"] = metrics["label"]
    return all_results


def print_and_save_statistics(all_results, baseline_acc=None, output_path=None):
    print(f"\n{'='*72}")
    print(f"{'RÉSULTATS STATISTIQUES':^72}")
    print(f"{'='*72}")
    print(f"{'Stratégie':<30} {'Acc μ±σ':^18} {'F1 μ±σ':^18} {'vs baseline':^10}")
    print(f"{'─'*72}")

    summary = {}
    for key, data in all_results.items():
        accs = np.array(data["acc"])
        f1s = np.array(data["f1"])
        summary[key] = {
            "label": data["label"],
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
            "acc_min": float(accs.min()), "acc_max": float(accs.max()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        }
        delta = f" {accs.mean() - baseline_acc:+.3f}" if baseline_acc else "  N/A "
        print(
            f"{data['label']:<30} "
            f"{accs.mean():.3f}±{accs.std():.3f}   "
            f"{f1s.mean():.3f}±{f1s.std():.3f}   "
            f"{delta}"
        )

    best = max(summary, key=lambda k: summary[k]["acc_mean"])
    print(f"\n🏆 Meilleure stratégie : {summary[best]['label']}")
    print(f"   Acc = {summary[best]['acc_mean']:.3f} ± {summary[best]['acc_std']:.3f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Rapport sauvegardé : {output_path}")

    return summary


# ──────────────────────────── Main ──────────────────────────────────────────

# Configuration backbone (à remplacer par study.best_params après pretraining_v2.py)
BACKBONE_CONFIG = {
    "d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 512,
    "dropout": 0.1, "revin": False,
    "patch_len": 6, "stride": 3,
    "patch_mask_ratio": 0.0,
}

if __name__ == "__main__":
    from tslearn.datasets import UCR_UEA_datasets

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler() if use_amp else None
    WINDOW = 36
    PRETRAINED_PATH = "models/best_indpatch_tst_v2.pth"

    # 1. Données
    print("\n── Chargement LSST ──")
    raw = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = raw.load_dataset("LSST")
    data = prepare_lsst(X_train, y_train, X_test, y_test, random_state=42)

    class_weights = data["class_weights"]
    criterion_optuna = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.1
    )

    # 2. Split fixe pour Optuna
    X_tr0, X_val0, y_tr0, y_val0 = train_test_split(
        data["X_tr"], data["y_tr"], test_size=0.2, stratify=data["y_tr"], random_state=42
    )
    train_dl0 = DataLoader(
        TensorDataset(torch.from_numpy(X_tr0).float(), torch.from_numpy(y_tr0).long()),
        batch_size=32, shuffle=True, drop_last=True,
    )
    val_dl0 = DataLoader(
        TensorDataset(torch.from_numpy(X_val0).float(), torch.from_numpy(y_val0).long()),
        batch_size=32, shuffle=False,
    )

    # 3. Tuning de la tête
    print("\n── Optuna : tuning de la tête (20 trials) ──")
    BACKBONE_CONFIG["seq_len"] = WINDOW
    head_params = tune_head(
        train_dl0, val_dl0, BACKBONE_CONFIG,
        n_features=data["n_features"], n_classes=data["n_classes"],
        device=device, pretrained_path=PRETRAINED_PATH,
        scaler_amp=scaler_amp, n_trials=20,
    )
    print(f"✅ head_params : {head_params}")

    # 4. 15 runs statistiques
    print("\n── 15 runs statistiques ──")
    all_results = run_statistics(
        n_runs=15, base_seed=0,
        X_train=data["X_tr"], y_train_enc=data["y_tr"],
        X_test=data["X_test"], y_test_enc=data["y_test"],
        backbone_config=BACKBONE_CONFIG, head_params=head_params,
        n_classes=data["n_classes"], n_features=data["n_features"],
        window=WINDOW, device=device,
        class_weights=class_weights,
        pretrained_path=PRETRAINED_PATH,
        scaler_amp=scaler_amp,
    )

    summary = print_and_save_statistics(
        all_results,
        output_path="reports/transfer_statistics_v2.json",
    )

    torch.save(
        {
            "summary": summary,
            "all_results": dict(all_results),
            "backbone_config": BACKBONE_CONFIG,
            "head_params": head_params,
            "scaler_mean": data["scaler"].mean_,
            "scaler_std": data["scaler"].scale_,
            "le_classes": data["le"].classes_,
        },
        "models/lsst_statistics_v2.pth",
    )
    print("💾 Sauvé : models/lsst_statistics_v2.pth")
