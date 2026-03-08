"""
adapting_to_classification.py — Fine-tuning IndPatchTST (ETTh1→LSST)

CORRECTIONS par rapport à la version dégradée (acc=0.039)
══════════════════════════════════════════════════════════
Diagnostic : les résultats from-scratch à 0.039 (pire que random=0.071 sur 14 classes)
indiquaient que le modèle n'apprenait pas du tout, pas simplement un mauvais transfer.

Causes identifiées et corrigées :

1. AUGMENTATION TROP AGRESSIVE
   - channel_dropout + noise_std=0.03 + scale [0.85, 1.15] sur des séquences
     courtes (T=36) corrompait les signaux astronomiques multivarés
   → Retour à jitter léger (noise_std=0.02) + scaling [0.9, 1.1] comme l'original

2. BATCH SIZE TROP GRAND (64 → 32)
   - LSST train ≈ 3200 samples. Avec batch=64, seulement ~50 steps/epoch,
     pas assez de gradient updates pour converger
   → Retour à batch_size=32 (original)

3. HEAD TROP PROFONDE
   - 3 couches cachées sur un petit dataset (3200 samples, 14 classes)
     → sur-paramétrage, gradient instable
   → Retour à 1 couche cachée avec dropout fort (comme l'original)
     mais avec la taille hidden_dim tunée par Optuna

4. SUPPRESSION DU CLASS WEIGHTS
   - Bien que LSST soit légèrement déséquilibré, les class_weights peuvent
     déstabiliser l'entraînement si les poids sont mal calibrés
   → Conservé mais avec label_smoothing=0.1 uniquement (pas de class weights)
     sauf si le déséquilibre est > 5:1 (à vérifier avec les données)

5. SCHEDULER CosineAnnealingWarmRestarts
   - Les restarts réinitialisent le LR trop souvent sur des entraînements courts
     (30-40 epochs) et empêchent la convergence
   → Retour à CosineAnnealingLR (original)

AMÉLIORATIONS CONSERVÉES (ne causent pas de dégradation) :
  - Alignement window=36 avec LSST
  - n_heads=4 dans le backbone (vs 1)
  - Mixed Precision AMP (vitesse, pas d'impact sur accuracy)
  - Justification documentée des stratégies de freeze
  - Progressive unfreezing (phases 1+2)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tslearn.datasets import UCR_UEA_datasets
from transformer_pretraining import IndPatchTST


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════


def augment_batch(x, noise_std=0.02):
    """
    Jitter gaussien léger + scaling aléatoire.
    Identique à l'original : simple et efficace pour LSST.
    noise_std=0.02 et scale [0.9, 1.1] sont calibrés pour ne pas dénaturer
    les motifs temporels sur des séquences courtes (T=36).
    """
    noise = torch.randn_like(x) * noise_std
    scale = torch.empty(x.size(0), 1, 1).uniform_(0.9, 1.1).to(x.device)
    return x * scale + noise


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE CLASSIFICATEUR
# ══════════════════════════════════════════════════════════════════════════════


class IndPatchTSTClassifier(nn.Module):
    """
    Wrapper de classification autour d'un backbone IndPatchTST pré-entraîné.

    Tête : LayerNorm → Linear(d_model, hidden_dim) → GELU → Dropout → Linear(hidden_dim, n_classes)
    Identique à l'original, avec hidden_dim et dropout tunés par Optuna.

    Groupes de freeze (du plus gelé au moins gelé) :
      group_embedding  : couche de projection des patches (features très bas-niveau)
      group_enc_early  : premières couches transformer (features structurelles)
      group_enc_late   : dernières couches transformer (features sémantiques)
      classifier       : tête de classification (toujours entraînable)
    """

    def __init__(
        self, window, num_features, num_classes, config, pretrained_model_path=None
    ):
        super().__init__()

        patch_len = config["patch_len"]
        stride = config["stride"]
        num_patches = (window - patch_len) // stride + 1
        assert num_patches > 0, (
            f"patch_len={patch_len} incompatible avec window={window}"
        )
        print(
            f"  ✅ {num_patches} patches | window={window} patch={patch_len} stride={stride}"
        )

        # ── Backbone ────────────────────────────────────────────────────────
        self.backbone = IndPatchTST(
            seq_len=window,
            pred_len=1,
            num_features=num_features,
            **config,
        )

        if pretrained_model_path is not None:
            checkpoint = torch.load(
                pretrained_model_path, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model_dict = self.backbone.state_dict()
            pretrained_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict, strict=False)
            pct = 100 * len(pretrained_dict) / max(len(state_dict), 1)
            print(
                f"  ✅ Chargé {len(pretrained_dict)}/{len(state_dict)} poids ({pct:.0f}%)"
            )
        else:
            print("  ⚡ From scratch (aucun poids pré-entraîné)")

        # Désactiver RevIN pour la classification (normalisation faite en amont)
        if self.backbone.revin:
            self.backbone.revin_layer = nn.Identity()
            self.backbone.revin = False

        self.backbone.head = nn.Identity()

        d = config["d_model"]

        # ── Tête de classification ───────────────────────────────────────────
        # 1 couche cachée avec régularisation forte : adapté à LSST (~3200 samples)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

        # ── Groupes de freeze ────────────────────────────────────────────────
        n_layers = config.get("n_layers", 4)
        mid = max(1, n_layers // 2)
        self.group_embedding = nn.ModuleList([self.backbone.patch_embedding])
        self.group_enc_early = nn.ModuleList(
            [self.backbone.transformer.layers[i] for i in range(mid)]
        )
        self.group_enc_late = nn.ModuleList(
            [self.backbone.transformer.layers[i] for i in range(mid, n_layers)]
        )

    def forward(self, x):
        return self.classifier(self.backbone.forward_features(x))

    # ── Méthodes de freeze ──────────────────────────────────────────────────

    def freeze_all_backbone(self):
        """
        Stratégie B — Head Only.

        Backbone complètement gelé, seule la tête s'entraîne.
        Avantage : rapide, aucun risque de catastrophic forgetting.
        Limite : si les features ETTh1 ne sont pas pertinentes pour LSST,
                 la tête seule ne peut pas compenser.
        Quand l'utiliser : dataset cible très petit, ou domaines très proches.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True
        self._print_trainable("Head Only")

    def unfreeze_late_encoders(self):
        """
        Stratégie C — Late Encoders + Head.

        Les 2 dernières couches du transformer + tête sont dégelées.
        Ces couches encodent des features de haut niveau (patterns, classes)
        qui doivent être adaptées au domaine cible (astronomie vs météo).
        Les premières couches (features bas-niveau : patches, fréquences)
        restent gelées car elles sont plus génériques.
        Quand l'utiliser : domaines modérément différents, dataset cible moyen.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.group_enc_late.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True
        self._print_trainable("Late Encoders + Head")

    def unfreeze_all(self):
        """
        Stratégie D — Full Fine-tune.

        Tout le modèle est dégelé.
        Capacité maximale d'adaptation au domaine cible.
        Risque : catastrophic forgetting si lr_backbone trop grand.
        → Utiliser LR différenciés : lr_backbone ≪ lr_head.
        Quand l'utiliser : domaines éloignés, dataset cible suffisamment grand.
        """
        for p in self.parameters():
            p.requires_grad = True
        self._print_trainable("Full Fine-tune")

    def _print_trainable(self, strategy_name=""):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"  [{strategy_name}] {trainable:,}/{total:,} "
            f"({100 * trainable / total:.1f}%) params entraînables"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BOUCLES D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════


def train_epoch(
    model, dl, optimizer, criterion, device, augment=False, scaler_amp=None
):
    model.train()
    total_loss, correct = 0.0, 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        if augment:
            x = augment_batch(x)

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
    model,
    train_dl,
    val_dl,
    optimizer,
    criterion,
    n_epochs,
    device,
    scheduler=None,
    augment=False,
    patience=8,
    scaler_amp=None,
):
    logs = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    best_val_acc, patience_ctr = -1.0, 0
    best_state = None

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_dl, optimizer, criterion, device, augment, scaler_amp
        )
        vl_loss, vl_acc = eval_epoch(model, val_dl, criterion, device)

        if scheduler:
            scheduler.step()

        logs["train_loss"].append(tr_loss)
        logs["train_acc"].append(tr_acc)
        logs["valid_loss"].append(vl_loss)
        logs["valid_acc"].append(vl_acc)
        print(
            f"    Epoch {epoch + 1:02d}/{n_epochs} | "
            f"TrL={tr_loss:.4f} TrA={tr_acc:.3f} | "
            f"VlL={vl_loss:.4f} VlA={vl_acc:.3f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    ⏹ Early stopping | meilleur VlA={best_val_acc:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    ✅ Meilleur modèle restauré (VlA={best_val_acc:.3f})")

    return logs


@torch.no_grad()
def evaluate(model, test_dl, device):
    model.eval()
    all_pred, all_y = [], []
    for x, y in test_dl:
        all_pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
        all_y.extend(y.numpy())
    all_pred, all_y = np.array(all_pred), np.array(all_y)
    return accuracy_score(all_y, all_pred), f1_score(all_y, all_pred, average="macro")


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA — Tuning de la tête de classification
# ══════════════════════════════════════════════════════════════════════════════


def objective_clf(
    trial,
    train_dl,
    val_dl,
    device,
    window,
    n_features,
    n_classes,
    backbone_config,
    scaler_amp,
):
    """
    Architecture backbone fixée (optimisée sur ETTh1).
    Seuls les hyperparamètres de la tête et du fine-tuning sont tunés.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_head = trial.suggest_float("lr_head", 1e-4, 5e-3, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = IndPatchTSTClassifier(
        window,
        n_features,
        n_classes,
        backbone_config,
        pretrained_model_path="models/best_indpatch_tst_optuna.pth",
    ).to(device)

    d = backbone_config["d_model"]
    model.classifier = nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_clf),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    # Phase 1 : head only (10 epochs)
    model.freeze_all_backbone()
    opt1 = torch.optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=wd)
    for epoch in range(10):
        train_epoch(
            model,
            train_dl,
            opt1,
            criterion,
            device,
            augment=True,
            scaler_amp=scaler_amp,
        )
        _, vl_acc = eval_epoch(model, val_dl, criterion, device)
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)

    # Phase 2 : full fine-tune court (5 epochs)
    model.unfreeze_all()
    opt2 = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head * 0.1},
        ],
        weight_decay=wd,
    )
    for epoch in range(5):
        train_epoch(
            model,
            train_dl,
            opt2,
            criterion,
            device,
            augment=False,
            scaler_amp=scaler_amp,
        )
        _, vl_acc = eval_epoch(model, val_dl, criterion, device)
        best_val_acc = max(best_val_acc, vl_acc)

    return best_val_acc


# ══════════════════════════════════════════════════════════════════════════════
#  EXPÉRIENCE UNIQUE
# ══════════════════════════════════════════════════════════════════════════════


def run_single_experiment(
    seed,
    X_train,
    y_train_enc,
    X_test,
    y_test_enc,
    backbone_config,
    best_head_params,
    n_classes,
    n_features,
    LSST_WINDOW,
    device,
    scaler_amp,
):
    """
    Un run complet : 4 stratégies de fine-tuning comparées.

    Protocole en 2 phases (progressive unfreezing) :
    ─────────────────────────────────────────────────
    Phase 1 — Warmup : entraîne uniquement les couches autorisées par la stratégie.
               But : stabiliser la tête avant de toucher au backbone.
    Phase 2 — Unfreeze total avec LR différenciés (lr_bb ≪ lr_head).
               But : adapter le backbone sans détruire les features pré-apprises.
               Appliquée seulement pour head_only et late_enc
               (full_tune fait déjà le dégel total en phase 1).

    Choix de la meilleure stratégie :
    ──────────────────────────────────
    Sélectionnée par Optuna sur un split fixe (seed=42) puis validée
    statistiquement sur 15 runs avec des seeds variées (0..14).
    La stratégie avec la meilleure acc_mean sur 15 runs est retenue.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=seed
    )

    # batch_size=32 : ~100 steps/epoch sur LSST train (~3200 samples)
    # → assez de gradient updates pour converger sur des entraînements courts
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
        batch_size=32,
        shuffle=False,
    )
    test_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test_enc).long()
        ),
        batch_size=32,
        shuffle=False,
    )

    lr_h = best_head_params["lr_head"]
    lr_bb = best_head_params["lr_backbone"]
    wd = best_head_params["weight_decay"]

    run_results = {}

    strategies = {
        "A_scratch": ("From Scratch (baseline)", None, "scratch", 60),
        "B_head_only": (
            "Head Only",
            "models/best_indpatch_tst_optuna.pth",
            "head_only",
            30,
        ),
        "C_late_enc": (
            "Late Encoders + Head",
            "models/best_indpatch_tst_optuna.pth",
            "late_enc",
            30,
        ),
        "D_full_tune": (
            "Full Fine-tune (lr différenciés)",
            "models/best_indpatch_tst_optuna.pth",
            "full",
            30,
        ),
    }

    for key, (label, pretrained_path, strategy, n_ep) in strategies.items():
        print(f"\n{'─' * 55}")
        print(f"  Stratégie : {label}")
        print(f"{'─' * 55}")

        model = IndPatchTSTClassifier(
            LSST_WINDOW,
            n_features,
            n_classes,
            backbone_config,
            pretrained_model_path=pretrained_path,
        ).to(device)

        # Tête avec hyperparamètres Optuna
        d = backbone_config["d_model"]
        hd = best_head_params["hidden_dim"]
        do = best_head_params["dropout_clf"]
        model.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hd),
            nn.GELU(),
            nn.Dropout(do),
            nn.Linear(hd, n_classes),
        ).to(device)

        # ── Phase 1 : Warmup ────────────────────────────────────────────────
        print(f"\n  ▶ Phase 1 : Warmup ({n_ep} epochs)")
        if strategy == "head_only":
            model.freeze_all_backbone()
            opt1 = torch.optim.AdamW(
                model.classifier.parameters(), lr=lr_h, weight_decay=wd
            )
        elif strategy == "late_enc":
            model.unfreeze_late_encoders()
            opt1 = torch.optim.AdamW(
                [
                    {"params": model.group_enc_late.parameters(), "lr": lr_bb},
                    {"params": model.classifier.parameters(), "lr": lr_h},
                ],
                weight_decay=wd,
            )
        elif strategy == "full":
            model.unfreeze_all()
            opt1 = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": lr_bb},
                    {"params": model.classifier.parameters(), "lr": lr_h},
                ],
                weight_decay=wd,
            )
        else:  # scratch
            model.unfreeze_all()
            opt1 = torch.optim.AdamW(model.parameters(), lr=lr_h, weight_decay=wd)

        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=n_ep)
        train_loop(
            model,
            train_dl,
            val_dl,
            opt1,
            criterion,
            n_ep,
            device,
            scheduler=sch1,
            augment=(strategy not in ("full", "scratch")),
            patience=12,
            scaler_amp=scaler_amp,
        )

        # ── Phase 2 : Progressive unfreeze ──────────────────────────────────
        if strategy in ("head_only", "late_enc"):
            n_ep2 = 20
            print(f"\n  ▶ Phase 2 : Dégel total ({n_ep2} epochs, lr_bb={lr_bb:.2e})")
            model.unfreeze_all()
            opt2 = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": lr_bb},
                    {"params": model.classifier.parameters(), "lr": lr_h * 0.1},
                ],
                weight_decay=wd,
            )
            sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=n_ep2)
            train_loop(
                model,
                train_dl,
                val_dl,
                opt2,
                criterion,
                n_ep2,
                device,
                scheduler=sch2,
                augment=False,
                patience=8,
                scaler_amp=scaler_amp,
            )

        acc, f1 = evaluate(model, test_dl, device)
        run_results[key] = {"label": label, "acc": acc, "f1": f1}
        print(f"\n  ✅ [{key}] Acc={acc:.4f} | F1={f1:.4f}")

    return run_results


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTIQUES MULTI-RUNS
# ══════════════════════════════════════════════════════════════════════════════


def run_statistics(n_runs=15, base_seed=0, **kwargs):
    all_results = defaultdict(lambda: {"acc": [], "f1": []})
    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n{'#' * 60}")
        print(f"  RUN {i + 1}/{n_runs}  (seed={seed})")
        print(f"{'#' * 60}")
        for key, metrics in run_single_experiment(seed=seed, **kwargs).items():
            all_results[key]["acc"].append(metrics["acc"])
            all_results[key]["f1"].append(metrics["f1"])
            all_results[key]["label"] = metrics["label"]
    return all_results


def print_statistics(all_results, baseline=0.40):
    print("\n")
    print(
        "╔══════════════════════════════════════════════════════════════════════════╗"
    )
    print("║            RÉSULTATS STATISTIQUES (n=15 runs, seed=0..14)               ║")
    print(
        "╠══════════════════════════════╦══════════════════╦══════════════════╦═════╣"
    )
    print(
        "║ Stratégie                    ║ Acc (μ ± σ)      ║ F1  (μ ± σ)      ║ Δ   ║"
    )
    print(
        "╠══════════════════════════════╬══════════════════╬══════════════════╬═════╣"
    )

    summary = {}
    for key, data in all_results.items():
        accs = np.array(data["acc"])
        f1s = np.array(data["f1"])
        summary[key] = {
            "label": data["label"],
            "acc_mean": accs.mean(),
            "acc_std": accs.std(),
            "acc_min": accs.min(),
            "acc_max": accs.max(),
            "f1_mean": f1s.mean(),
            "f1_std": f1s.std(),
        }
        delta = accs.mean() - baseline
        sign = "+" if delta >= 0 else ""
        print(
            f"║ {data['label']:<28} ║ {accs.mean():.3f} ± {accs.std():.3f}  "
            f"║ {f1s.mean():.3f} ± {f1s.std():.3f}  ║{sign}{delta:+.3f}║"
        )

    print(
        "╠══════════════════════════════╩══════════════════╩══════════════════╩═════╣"
    )
    best_key = max(summary, key=lambda k: summary[k]["acc_mean"])
    s = summary[best_key]
    print(f"║ 🏆 Meilleur : {s['label']:<57}║")
    print(
        f"║    Acc = {s['acc_mean']:.3f} ± {s['acc_std']:.3f}"
        f"  [min={s['acc_min']:.3f}, max={s['acc_max']:.3f}]{'':>14}║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════════════════╝"
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG BACKBONE
# ══════════════════════════════════════════════════════════════════════════════

# Valeurs par défaut raisonnables pour LSST (window=36, patch_len=6, stride=3 → 11 patches)
# À remplacer par study.best_params après avoir lancé transformer_pretraining.py
BACKBONE_CONFIG = {
    "d_model": 128,
    "n_heads": 4,  # 4 têtes : meilleur que 1, compatible avec d_model=128
    "n_layers": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "revin": False,
    "patch_len": 6,
    "stride": 3,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import optuna
    import os

    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler() if use_amp else None
    print(f"Mixed Precision AMP : {'activé ⚡' if use_amp else 'désactivé'}")

    LSST_WINDOW = 36

    # ── Chargement LSST ──────────────────────────────────────────────────────
    print("\n── Chargement LSST ──")
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)
    n_features = X_train.shape[2]
    print(f"Classes : {n_classes} | Features : {n_features}")
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    # ── Padding/truncation ───────────────────────────────────────────────────
    def pad_truncate(X, tlen):
        out = []
        for arr in X:
            T = arr.shape[0]
            out.append(
                arr[:tlen]
                if T >= tlen
                else np.vstack(
                    [arr, np.zeros((tlen - T, arr.shape[1]), dtype=arr.dtype)]
                )
            )
        return np.stack(out)

    X_train = pad_truncate(X_train, LSST_WINDOW)
    X_test = pad_truncate(X_test, LSST_WINDOW)

    # ── Normalisation ────────────────────────────────────────────────────────
    B, T, C = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, C))
    X_train = (
        scaler.transform(X_train.reshape(-1, C)).reshape(B, T, C).astype(np.float32)
    )
    X_test = (
        scaler.transform(X_test.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)
    )
    print(f"✅ Normalisé | train{X_train.shape} test{X_test.shape}")

    # ── Split fixe pour Optuna ───────────────────────────────────────────────
    X_tr0, X_val0, y_tr0, y_val0 = train_test_split(
        X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42
    )
    train_dl0 = DataLoader(
        TensorDataset(torch.from_numpy(X_tr0).float(), torch.from_numpy(y_tr0).long()),
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    val_dl0 = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val0).float(), torch.from_numpy(y_val0).long()
        ),
        batch_size=32,
        shuffle=False,
    )

    # ── Optuna ───────────────────────────────────────────────────────────────
    print("\n── Recherche Optuna (hyperparamètres de la tête, 30 trials) ──")
    study_clf = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study_clf.optimize(
        lambda trial: objective_clf(
            trial,
            train_dl0,
            val_dl0,
            device,
            LSST_WINDOW,
            n_features,
            n_classes,
            BACKBONE_CONFIG,
            scaler_amp,
        ),
        n_trials=30,
    )
    best_head_params = study_clf.best_params
    print(f"\n✅ Meilleurs hyperparamètres tête : {best_head_params}")
    print(f"✅ Meilleure val acc (Optuna) : {study_clf.best_value:.4f}")

    # ── 15 runs statistiques ─────────────────────────────────────────────────
    print("\n── Lancement 15 runs statistiques ──")
    all_results = run_statistics(
        n_runs=15,
        base_seed=0,
        X_train=X_train,
        y_train_enc=y_train_enc,
        X_test=X_test,
        y_test_enc=y_test_enc,
        backbone_config=BACKBONE_CONFIG,
        best_head_params=best_head_params,
        n_classes=n_classes,
        n_features=n_features,
        LSST_WINDOW=LSST_WINDOW,
        device=device,
        scaler_amp=scaler_amp,
    )

    summary = print_statistics(all_results, baseline=0.40)

    torch.save(
        {
            "summary": summary,
            "all_results": dict(all_results),
            "backbone_config": BACKBONE_CONFIG,
            "head_params": best_head_params,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
            "le_classes": le.classes_,
        },
        "lsst_statistics_complete.pth",
    )
    print("\n💾 Sauvegardé : lsst_statistics_complete.pth")
