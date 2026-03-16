"""
dataloader_v2.py — Chargement et préparation des données LSST

Améliorations par rapport à v1 :
  - LabelEncoder sklearn partagé (fit sur train, transform sur val/test)
  - Stratification garantie dans le split train/val
  - StandardScaler fit uniquement sur train (pas de fuite de données)
  - Augmentation temporelle optionnelle intégrée (jitter, scaling, time masking)
  - class_weights calculés et exposés pour la CrossEntropyLoss
  - num_workers=0 par défaut (compatible Windows et environnements sans fork)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────── Dataset ────────────────────────────────────────


class LSSTDataset(Dataset):
    """
    Dataset LSST avec augmentation optionnelle.

    Args:
        X : (N, T, C) float32
        y : (N,) int64
        augment : active jitter + scaling + time masking pendant __getitem__
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (T, C)
        if self.augment:
            x = _augment(x)
        return x, self.y[idx]


def _augment(x: torch.Tensor) -> torch.Tensor:
    """
    Augmentations légères calibrées pour LSST (T=36, C=6) :
      - Jitter gaussien (std=0.02) : bruit de mesure réaliste
      - Scaling aléatoire [0.9, 1.1] : variation d'amplitude
      - Time masking (1 segment de 3 pas au max) : robustesse aux données manquantes
    Ces amplitudes sont volontairement conservatrices pour ne pas dénaturer
    les patterns astronomiques multivarié sur des séquences courtes.
    """
    # Jitter
    x = x + torch.randn_like(x) * 0.02
    # Amplitude scaling (par canal)
    scale = torch.empty(1, x.size(1)).uniform_(0.9, 1.1)
    x = x * scale
    # Time masking : masque aléatoire d'une fenêtre de 1–3 pas
    T = x.size(0)
    mask_len = torch.randint(1, 4, ()).item()
    mask_start = torch.randint(0, T - mask_len + 1, ()).item()
    x[mask_start : mask_start + mask_len] = 0.0
    return x


# ─────────────────────────── Préparation ────────────────────────────────────


def prepare_lsst(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
    target_len: int = 36,
):
    """
    Pipeline complet : encode → pad/tronque → normalise → split stratifié.

    Retourne un dict avec toutes les données et les outils de prétraitement.
    Le StandardScaler est fitté UNIQUEMENT sur X_train pour éviter toute fuite.

    Args:
        X_train_raw, y_train_raw : données brutes tslearn (y peut être des strings)
        X_test_raw, y_test_raw   : données de test brutes
        val_size                  : fraction de validation (défaut 0.2)
        random_state              : seed pour la reproductibilité
        target_len                : longueur temporelle cible (36 pour LSST)

    Returns:
        dict avec clés :
          X_tr, y_tr, X_val, y_val, X_test, y_test (np.ndarray float32/int64)
          le (LabelEncoder), scaler (StandardScaler)
          n_classes, n_features
          class_weights (torch.Tensor) pour CrossEntropyLoss
    """
    # 1. Encodage des labels (str → int)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw).astype(np.int64)
    y_test_enc = le.transform(y_test_raw).astype(np.int64)
    n_classes = len(le.classes_)
    n_features = X_train_raw.shape[2]

    # 2. Padding / troncature sur la dimension temporelle
    X_train_raw = _pad_or_truncate(X_train_raw, target_len)
    X_test_raw = _pad_or_truncate(X_test_raw, target_len)

    # 3. Split stratifié train / val (AVANT la normalisation pour cohérence)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw,
        y_train_enc,
        test_size=val_size,
        stratify=y_train_enc,
        random_state=random_state,
    )

    # 4. Normalisation : fit sur train uniquement → transform sur val et test
    N_tr, T, C = X_tr.shape
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.reshape(-1, C)).reshape(N_tr, T, C).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)

    # 5. Class weights (inverse de la fréquence) pour CrossEntropyLoss
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    class_weights = torch.from_numpy(1.0 / (counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * n_classes  # normalisation

    print(f"✅ Préparation LSST terminée")
    print(f"   Train : {X_tr.shape} | Val : {X_val.shape} | Test : {X_test.shape}")
    print(f"   Classes : {n_classes} | Features : {n_features}")
    _print_class_balance(y_tr, n_classes, "Train")

    return dict(
        X_tr=X_tr, y_tr=y_tr,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test_enc,
        le=le, scaler=scaler,
        n_classes=n_classes, n_features=n_features,
        class_weights=class_weights,
    )


def _pad_or_truncate(X: np.ndarray, target_len: int) -> np.ndarray:
    """Pad avec des zéros ou tronque chaque série à target_len."""
    N, T, C = X.shape
    if T == target_len:
        return X
    out = np.zeros((N, target_len, C), dtype=X.dtype)
    copy_len = min(T, target_len)
    out[:, :copy_len, :] = X[:, :copy_len, :]
    return out


def _print_class_balance(y: np.ndarray, n_classes: int, split_name: str):
    counts = np.bincount(y, minlength=n_classes)
    ratio = counts.max() / (counts.min() + 1e-6)
    print(f"   Balance {split_name}: min={counts.min()} max={counts.max()} ratio={ratio:.1f}x")
    if ratio > 5:
        print(f"   ⚠️  Déséquilibre important ({ratio:.1f}x) → class_weights activés")


# ─────────────────────────── DataLoaders ────────────────────────────────────


def build_dataloaders(
    data: dict,
    batch_size: int = 32,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    num_workers: int = 0,
):
    """
    Construit train / val / test DataLoaders à partir du dict retourné par prepare_lsst.

    Args:
        data              : dict retourné par prepare_lsst
        batch_size        : taille des mini-batches
        augment_train     : active l'augmentation sur le train
        use_weighted_sampler : échantillonnage pondéré (alternative aux class_weights)
        num_workers       : workers de chargement (0 = main process, sûr partout)

    Returns:
        train_dl, val_dl, test_dl
    """
    train_ds = LSSTDataset(data["X_tr"], data["y_tr"], augment=augment_train)
    val_ds = LSSTDataset(data["X_val"], data["y_val"], augment=False)
    test_ds = LSSTDataset(data["X_test"], data["y_test"], augment=False)

    sampler = None
    if use_weighted_sampler:
        # WeightedRandomSampler : alternative aux class_weights dans la loss
        sample_weights = data["class_weights"][data["y_tr"]]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dl, val_dl, test_dl


# ─────────────────────────── Test rapide ────────────────────────────────────

if __name__ == "__main__":
    from tslearn.datasets import UCR_UEA_datasets

    print("Chargement LSST...")
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    data = prepare_lsst(X_train, y_train, X_test, y_test)
    train_dl, val_dl, test_dl = build_dataloaders(data, batch_size=32, augment_train=True)

    for x, y in train_dl:
        print(f"Batch X: {x.shape} | Batch y: {y.shape} | y range [{y.min()}, {y.max()}]")
        break

    print(f"Class weights: {data['class_weights']}")
    print("✅ dataloader_v2.py OK")
