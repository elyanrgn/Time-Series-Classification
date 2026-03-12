import importlib.util
import sys
from pathlib import Path

# Fallback: if package not installed in this kernel, add repo root to sys.path
if importlib.util.find_spec("src") is None:
    # Resolve repo root from this file location (src/models/indpatchtst.py -> repo root)
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


import os

import torch.nn as nn


# MODÈLE CNN BASELINE
class CNNBaseline(nn.Module):
    def __init__(
        self,
        n_features=6,
        n_classes=14,
        n_filters1=64,
        n_filters2=128,
        n_filters3=256,
        n_filters4=512,
        kernel_size=3,
        dropout=0.5,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(n_features, n_filters1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filters1),
            nn.ReLU(),
            nn.Conv1d(n_filters1, n_filters2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filters2),
            nn.ReLU(),
            nn.Conv1d(n_filters2, n_filters3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filters3),
            nn.ReLU(),
            nn.Conv1d(n_filters3, n_filters4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filters4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_filters4, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        return self.net(x)


# === MAIN ===
if __name__ == "__main__":
    from src.training.trainer_cnn import run_statistics_cnn

    os.makedirs("models", exist_ok=True)
    run_statistics_cnn(n_runs=15)
