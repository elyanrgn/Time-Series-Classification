import importlib.util
from pathlib import Path
import sys

# Fallback: if package not installed in this kernel, add repo root to sys.path
if importlib.util.find_spec("src") is None:
    # Resolve repo root from this file location (src/models/indpatchtst.py -> repo root)
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


import torch.nn as nn
from src.models.indpatchtst import IndPatchTST
import torch


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

    def __init__(self, window, num_features, num_classes, config, pretrained_model_path=None):
        super().__init__()

        patch_len = config["patch_len"]
        stride = config["stride"]
        num_patches = (window - patch_len) // stride + 1
        assert num_patches > 0, f"patch_len={patch_len} incompatible avec window={window}"
        print(f"  ✅ {num_patches} patches | window={window} patch={patch_len} stride={stride}")

        # ── Backbone ────────────────────────────────────────────────────────
        self.backbone = IndPatchTST(
            seq_len=window,
            pred_len=1,
            num_features=num_features,
            **config,
        )

        if pretrained_model_path is not None:
            checkpoint = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
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
            print(f"  ✅ Chargé {len(pretrained_dict)}/{len(state_dict)} poids ({pct:.0f}%)")
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
