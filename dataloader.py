import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LSSTClassificationDataset(Dataset):
    """Windowed multivariate time series classification dataset for LSST."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, window: int = None, stride: int = 1
    ):
        """
        Args:
            X: (n_samples, n_timesteps=36, n_features=6) - LSST data
            y: (n_samples,) - integer labels
            window: int, optional - fenêtre glissante sur time dim (None = full sequence)
            stride: int - pas pour fenêtres
        """
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()
        self.n_samples, self.n_timesteps, self.n_features = self.X.shape

        if window is None:
            window = self.n_timesteps
        self.window = window
        self.stride = stride

        # Nombre de fenêtres possibles par sample
        self.n_windows_per_sample = max(1, (self.n_timesteps - window) // stride + 1)
        self.total_len = self.n_samples * self.n_windows_per_sample

        if self.total_len == 0:
            raise ValueError("Window + stride trop grands pour la longueur des séries.")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        sample_idx = idx // self.n_windows_per_sample
        window_idx = idx % self.n_windows_per_sample

        start_t = window_idx * self.stride
        end_t = start_t + self.window

        past = self.X[sample_idx, start_t:end_t, :]  # (window, 6)
        label = self.y[sample_idx]
        return past, label


def build_lsst_dataloader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window: int = None,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Create train/val/test DataLoaders for LSST classification.

    Args:
        X_train, y_train, X_test, y_test:  tslearn UCR_UEA_datasets
        window, stride: for sliding window (None=full seq)
        batch_size, shuffle, num_workers: std DataLoader params

    Returns:
        train_dl, val_dl, test_dl
    """
    # CONVERSION LABELS STR → INT (fix pour tslearn)
    le = {}  # Label encoder simple

    def encode_labels(y):
        y_int = np.zeros(len(y), dtype=np.int64)
        for i, label in enumerate(np.unique(y)):
            le[label] = i
            y_int[y == label] = i
        return y_int, le

    y_train_int, le = encode_labels(y_train)
    y_val_int, _ = encode_labels(y_train)  # Même mapping
    y_test_int, _ = encode_labels(y_test)

    # Split train en train/val (80/20)
    n_train = len(X_train)
    n_val = int(0.2 * n_train)

    train_X = X_train[: n_train - n_val]
    train_y = y_train_int[: n_train - n_val]
    val_X = X_train[n_train - n_val :]
    val_y = y_val_int[n_train - n_val :]
    test_dataset = LSSTClassificationDataset(X_test, y_test_int, window, stride)

    train_dataset = LSSTClassificationDataset(train_X, train_y, window, stride)
    val_dataset = LSSTClassificationDataset(val_X, val_y, window, stride)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    from tslearn.datasets import UCR_UEA_datasets

    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    train_dl, val_dl, test_dl = build_lsst_dataloader(
        X_train, y_train, X_test, y_test, window=12, stride=6, batch_size=64
    )

    # Vérifions les shapes
    for x_batch, y_batch in train_dl:
        print(f"Batch X shape: {x_batch.shape}, Batch y shape: {y_batch.shape}")
        break
