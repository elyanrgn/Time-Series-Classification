import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torch.utils.data import DataLoader


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
        padding = kernel_size // 2  # pour garder la longueur temporelle

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
            nn.AdaptiveAvgPool1d(1),  # global average pooling sur le temps
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_filters4, n_classes),
        )

    def forward(self, x):
        # x: (batch, T, C) → permute pour Conv1d: (batch, C, T)
        x = x.permute(0, 2, 1)
        return self.net(x)  # logits (batch, n_classes)


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train_one_model(
    model,
    train_loader,
    val_loader,
    max_epochs=50,
    patience=5,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Early stopping sur F1 de validation
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping à l'epoch {epoch} (best val F1 = {best_val_f1:.4f})"
                )
                break

    # Recharge le meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_f1


def hyperparam_search(train_loader, val_loader, n_features, n_classes, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = [
        {
            "n_filters1": 32,
            "n_filters2": 64,
            "n_filters3": 128,
            "n_filters4": 256,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "n_filters1": 64,
            "n_filters2": 128,
            "n_filters3": 256,
            "n_filters4": 512,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "n_filters1": 64,
            "n_filters2": 128,
            "n_filters3": 256,
            "n_filters4": 512,
            "dropout": 0.3,
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
        {
            "n_filters1": 64,
            "n_filters2": 128,
            "n_filters3": 256,
            "n_filters4": 512,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 5e-4,
        },
    ]

    best_config = None
    best_model = None
    best_val_f1 = -np.inf

    for i, cfg in enumerate(search_space, 1):
        print(f"\n=== Config {i}/{len(search_space)}: {cfg} ===")
        model = CNNBaseline(
            n_features=n_features,
            n_classes=n_classes,
            n_filters1=cfg["n_filters1"],
            n_filters2=cfg["n_filters2"],
            n_filters3=cfg["n_filters3"],
            n_filters4=cfg["n_filters4"],
            dropout=cfg["dropout"],
        )

        model, val_f1 = train_one_model(
            model,
            train_loader,
            val_loader,
            max_epochs=50,
            patience=7,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=device,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_model = model

    print(f"\nMeilleure config: {best_config} avec val F1 = {best_val_f1:.4f}")
    return best_model, best_config, best_val_f1


if __name__ == "__main__":
    import numpy as np
    from tslearn.datasets import UCR_UEA_datasets
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import TensorDataset, DataLoader
    from dataloader import build_lsst_dataloader

    # 1) Charger LSST
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset(
        "LSST"
    )  # X: (N, T, C), y: strings
    # train_dl, val_dl, test_dl = build_lsst_dataloader( ##Très peu efficace, à revoir
    #    X_train, y_train, X_test, y_test,batch_size=64,shuffle=False, drop_last=False)
    # 2) Encoder les labels (string -> int 0..C-1)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    n_features = X_train.shape[2]

    # 3) Construire TensorDatasets + DataLoaders (split train/val 80/20)
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train_enc).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test_enc).long()

    n_train = len(X_train_t)
    n_val = int(0.2 * n_train)

    X_tr, y_tr = X_train_t[: n_train - n_val], y_train_t[: n_train - n_val]
    X_val, y_val = X_train_t[n_train - n_val :], y_train_t[n_train - n_val :]

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 4) Hyperparam search + early stopping
    best_model, best_cfg, best_val_f1 = hyperparam_search(
        train_dl, val_dl, n_features=n_features, n_classes=n_classes, device=device
    )

    # 5) Évaluation finale sur test
    test_acc, test_f1 = evaluate(best_model, test_dl, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
