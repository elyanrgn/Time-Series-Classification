import torch
import torch.nn as nn

from src.models.indpatchtst_classifier import IndPatchTSTClassifier

DEFAULT_PRETRAINED_PATH = "artifacts/models/best_indpatch_tst_optuna.pth"


def augment_batch(x, noise_std=0.02):
    """
    Jitter gaussien léger + scaling aléatoire.
    Identique à l'original : simple et efficace pour LSST.
    """
    noise = torch.randn_like(x) * noise_std
    scale = torch.empty(x.size(0), 1, 1).uniform_(0.9, 1.1).to(x.device)
    return x * scale + noise


def build_clf_model(
    window,
    n_features,
    n_classes,
    backbone_config,
    hidden_dim,
    dropout_clf,
    device,
    pretrained_model_path=DEFAULT_PRETRAINED_PATH,
):
    """Construit et retourne un IndPatchTSTClassifier avec tête personnalisée."""
    model = IndPatchTSTClassifier(
        window,
        n_features,
        n_classes,
        backbone_config,
        pretrained_model_path=pretrained_model_path,
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


# Backward-compat alias used by optuna_search
_build_clf_model = build_clf_model
