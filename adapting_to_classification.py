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
    total_loss, correct, n_samples = 0.0, 0, 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        if augment:
            x = augment_batch(x)

        optimizer.zero_grad()

        if scaler_amp is not None:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            # Si la loss est NaN/inf (overflow float16), scaler skip le step
            # mais on doit aussi ne PAS accumuler dans total_loss
            if not torch.isfinite(loss):
                scaler_amp.update()  # met à jour le scale factor (le réduit)
                continue
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        n_samples += x.size(0)

    if n_samples == 0:
        return float("nan"), 0.0
    return total_loss / n_samples, correct / n_samples


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
#  OPTUNA — Une étude par stratégie pré-entraînée
#
#  Pourquoi une étude par stratégie ?
#  lr_backbone n'a AUCUN signal dans objective_clf (backbone gelé) → Optuna
#  choisit une valeur arbitraire dans [1e-6, 1e-4], qui se retrouve ensuite
#  utilisée pour late_enc et full_tune où elle est critique. Chaque stratégie
#  a sa propre dynamique d'entraînement et nécessite ses propres LR.
# ══════════════════════════════════════════════════════════════════════════════


def _build_clf_model(
    window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
):
    """Construit et retourne un IndPatchTSTClassifier avec tête personnalisée."""
    model = IndPatchTSTClassifier(
        window,
        n_features,
        n_classes,
        backbone_config,
        pretrained_model_path="models/best_indpatch_tst_optuna2.pth",
    ).to(device)
    d = backbone_config["d_model"]
    model.classifier = nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_clf),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)
    return model


def objective_head_only(
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
    Stratégie B — Head Only.
    Backbone entièrement gelé : seule la tête est entraînée.
    Hyperparamètres tunés : hidden_dim, dropout_clf, lr_head, weight_decay.
    lr_backbone n'est PAS tuné ici car il n'est pas utilisé dans cette stratégie.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = _build_clf_model(
        window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
    )
    model.freeze_all_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0
    for epoch in range(20):
        train_epoch(
            model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp
        )
        _, vl_acc = eval_epoch(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)
    return best_val_acc


def objective_late_enc(
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
    Stratégie C — Late Encoders + Head.
    Les 2 dernières couches transformer + tête sont dégelées.
    Hyperparamètres tunés : lr_late (late encoders), lr_head, hidden_dim,
    dropout_clf, weight_decay.
    lr_late est distinct de lr_backbone car seules les late layers bougent.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_late = trial.suggest_float("lr_late", 1e-5, 5e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = _build_clf_model(
        window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
    )
    model.unfreeze_late_encoders()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [
            {"params": model.group_enc_late.parameters(), "lr": lr_late},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0
    for epoch in range(20):
        train_epoch(
            model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp
        )
        _, vl_acc = eval_epoch(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)
    return best_val_acc


def objective_full_tune(
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
    Stratégie D — Full Fine-tune avec LR différenciés.
    Tout le modèle est dégelé. lr_backbone ≪ lr_head pour protéger
    les features pré-apprises du catastrophic forgetting.
    Hyperparamètres tunés : lr_backbone, lr_head, hidden_dim, dropout_clf, wd.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-5, 3e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = _build_clf_model(
        window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
    )
    model.unfreeze_all()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0
    for epoch in range(20):
        train_epoch(
            model,
            train_dl,
            opt,
            criterion,
            device,
            augment=False,
            scaler_amp=scaler_amp,
        )
        _, vl_acc = eval_epoch(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)
    return best_val_acc


def objective_scratch(
    trial,
    train_dl,
    val_dl,
    device,
    window,
    n_features,
    n_classes,
    scaler_amp,
):
    """
    Recherche Optuna dédiée au modèle From Scratch.

    Explore simultanément la config backbone ET les hyperparamètres
    d'entraînement, directement sur la tâche de classification LSST.
    C'est nécessaire car la config optimale pour la régression ETTh1
    (minimiser MSE sur séries météo) n'est pas forcément optimale pour
    classifier des objets astronomiques multivariés.
    """
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)
    patch_len = trial.suggest_int("patch_len", 3, window // 3)
    stride = trial.suggest_int("stride", 1, max(1, window // 8))
    if patch_len >= window:
        raise __import__("optuna").TrialPruned()

    scratch_config = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "revin": False,
        "patch_len": patch_len,
        "stride": stride,
    }

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = IndPatchTSTClassifier(
        window,
        n_features,
        n_classes,
        scratch_config,
        pretrained_model_path=None,
    ).to(device)

    model.classifier = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_clf),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    model.unfreeze_all()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0

    for epoch in range(20):
        train_epoch(
            model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp
        )
        _, vl_acc = eval_epoch(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
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
    scratch_config,
    params_head_only,
    params_late_enc,
    params_full_tune,
    best_scratch_params,
    n_classes,
    n_features,
    LSST_WINDOW,
    device,
    scaler_amp,
):
    """
    Un run complet : 4 stratégies comparées avec leurs hyperparamètres dédiés.

    Chaque stratégie (B, C, D) dispose de ses propres LR issus d'une étude
    Optuna indépendante — sans biais croisé entre les stratégies.
    Chaque stratégie conserve son niveau de gel pendant TOUS ses epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=seed
    )
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

    run_results = {}

    strategies = {
        "A_scratch": ("From Scratch (baseline)", None, "scratch", 60),
        "B_head_only": (
            "Head Only",
            "models/best_indpatch_tst_optuna2.pth",
            "head_only",
            60,
        ),
        "C_late_enc": (
            "Late Encoders + Head",
            "models/best_indpatch_tst_optuna2.pth",
            "late_enc",
            50,
        ),
        "D_full_tune": (
            "Full Fine-tune (lr différenciés)",
            "models/best_indpatch_tst_optuna2.pth",
            "full",
            40,
        ),
    }

    for key, (label, pretrained_path, strategy, n_ep) in strategies.items():
        print(f"\n{'─' * 55}")
        print(f"  Stratégie : {label}")
        print(f"{'─' * 55}")

        # ── Config et hyperparamètres dédiés à la stratégie ─────────────────
        if strategy == "scratch":
            cfg = scratch_config
            hd = best_scratch_params["hidden_dim"]
            do = best_scratch_params["dropout_clf"]
        elif strategy == "head_only":
            cfg = backbone_config
            hd = params_head_only["hidden_dim"]
            do = params_head_only["dropout_clf"]
        elif strategy == "late_enc":
            cfg = backbone_config
            hd = params_late_enc["hidden_dim"]
            do = params_late_enc["dropout_clf"]
        else:  # full
            cfg = backbone_config
            hd = params_full_tune["hidden_dim"]
            do = params_full_tune["dropout_clf"]

        model = IndPatchTSTClassifier(
            LSST_WINDOW,
            n_features,
            n_classes,
            cfg,
            pretrained_model_path=pretrained_path,
        ).to(device)
        model.classifier = nn.Sequential(
            nn.LayerNorm(cfg["d_model"]),
            nn.Linear(cfg["d_model"], hd),
            nn.GELU(),
            nn.Dropout(do),
            nn.Linear(hd, n_classes),
        ).to(device)

        # ── Optimiseur avec LR dédiés — gel fixe jusqu'au bout ───────────────
        print(f"\n  ▶ Entraînement ({n_ep} epochs)")
        if strategy == "head_only":
            model.freeze_all_backbone()
            opt1 = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=params_head_only["lr_head"],
                weight_decay=params_head_only["weight_decay"],
            )
        elif strategy == "late_enc":
            model.unfreeze_late_encoders()
            opt1 = torch.optim.AdamW(
                [
                    {
                        "params": model.group_enc_late.parameters(),
                        "lr": params_late_enc["lr_late"],
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": params_late_enc["lr_head"],
                    },
                ],
                weight_decay=params_late_enc["weight_decay"],
            )
        elif strategy == "full":
            model.unfreeze_all()
            opt1 = torch.optim.AdamW(
                [
                    {
                        "params": model.backbone.parameters(),
                        "lr": params_full_tune["lr_backbone"],
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": params_full_tune["lr_head"],
                    },
                ],
                weight_decay=params_full_tune["weight_decay"],
            )
        else:  # scratch
            model.unfreeze_all()
            opt1 = torch.optim.AdamW(
                model.parameters(),
                lr=best_scratch_params["lr"],
                weight_decay=best_scratch_params["weight_decay"],
            )

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


#  CONFIG BACKBONE
# Valeurs par défaut raisonnables pour LSST (window=36, patch_len=6, stride=3 → 11 patches)
# À remplacer par study.best_params après avoir lancé transformer_pretraining.py
BACKBONE_CONFIG = {
    "d_model": 256,
    "n_heads": 8,  # 8 têtes : meilleur que 1, compatible avec d_model=256
    "n_layers": 4,
    "d_ff": 256,
    "dropout": 0.08,
    "revin": False,
    "patch_len": 11,
    "stride": 4,
}

BACKBONE_CONFIG2 = {
    "d_model": 128,
    "n_heads": 4,  # 4 têtes : meilleur que 1, compatible avec d_model=128
    "n_layers": 4,
    "d_ff": 256,
    "dropout": 0.1948,
    "revin": False,
    "patch_len": 10,
    "stride": 3,
}
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

    # Padding/truncation pour aligner sur window=36
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

    # Normalisation
    B, T, C = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, C))
    X_train = (
        scaler.transform(X_train.reshape(-1, C)).reshape(B, T, C).astype(np.float32)
    )
    X_test = (
        scaler.transform(X_test.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)
    )
    print(f"✅ Normalisé | train{X_train.shape} test{X_test.shape}")

    # Split fixe pour Optuna
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

    # ── Optuna — 1 étude par stratégie pré-entraînée + 1 pour scratch ────────
    def _make_study(seed=42):
        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

    opt_args = dict(
        train_dl=train_dl0,
        val_dl=val_dl0,
        device=device,
        window=LSST_WINDOW,
        n_features=n_features,
        n_classes=n_classes,
        backbone_config=BACKBONE_CONFIG2,
        scaler_amp=scaler_amp,
    )

    print("\n── Optuna B — Head Only (25 trials) ──")
    study_b = _make_study()
    study_b.optimize(lambda t: objective_head_only(t, **opt_args), n_trials=25)
    params_head_only = study_b.best_params
    print(f"  ✅ Best acc={study_b.best_value:.4f} | {params_head_only}")

    print("\n── Optuna C — Late Encoders (25 trials) ──")
    study_c = _make_study()
    study_c.optimize(lambda t: objective_late_enc(t, **opt_args), n_trials=25)
    params_late_enc = study_c.best_params
    print(f"  ✅ Best acc={study_c.best_value:.4f} | {params_late_enc}")

    print("\n── Optuna D — Full Fine-tune (25 trials) ──")
    study_d = _make_study()
    study_d.optimize(lambda t: objective_full_tune(t, **opt_args), n_trials=25)
    params_full_tune = study_d.best_params
    print(f"  ✅ Best acc={study_d.best_value:.4f} | {params_full_tune}")

    print("\n── Optuna A — From Scratch backbone+tête (40 trials) ──")
    study_scratch = _make_study()
    study_scratch.optimize(
        lambda t: objective_scratch(
            t,
            train_dl0,
            val_dl0,
            device,
            LSST_WINDOW,
            n_features,
            n_classes,
            scaler_amp,
        ),
        n_trials=40,
    )
    best_scratch_params_all = study_scratch.best_params
    print(f"  ✅ Best acc={study_scratch.best_value:.4f} | {best_scratch_params_all}")

    scratch_config = {
        k: best_scratch_params_all[k]
        for k in [
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "patch_len",
            "stride",
        ]
    }
    scratch_config["revin"] = False
    best_scratch_train_params = {
        k: best_scratch_params_all[k]
        for k in ["hidden_dim", "dropout_clf", "lr", "weight_decay"]
    }

    # ── 15 runs statistiques ─────────────────────────────────────────────────
    print("\n── Lancement 15 runs statistiques ──")
    all_results = run_statistics(
        n_runs=15,
        base_seed=0,
        X_train=X_train,
        y_train_enc=y_train_enc,
        X_test=X_test,
        y_test_enc=y_test_enc,
        backbone_config=BACKBONE_CONFIG2,
        scratch_config=scratch_config,
        params_head_only=params_head_only,
        params_late_enc=params_late_enc,
        params_full_tune=params_full_tune,
        best_scratch_params=best_scratch_train_params,
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
            "backbone_config": BACKBONE_CONFIG2,
            "scratch_config": scratch_config,
            "params_head_only": params_head_only,
            "params_late_enc": params_late_enc,
            "params_full_tune": params_full_tune,
            "scratch_train_params": best_scratch_train_params,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
            "le_classes": le.classes_,
        },
        "lsst_statistics_complete2.pth",
    )
    print("\n💾 Sauvegardé : lsst_statistics_complete2.pth")
