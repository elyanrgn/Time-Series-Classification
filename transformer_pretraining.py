import numpy
import torch
from torch.utils.data import Dataset, DataLoader
import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Reuse ETTh1Dataset from Lab 2 without scaling (we will use RevIn normalization layers)
def load_etth1(csv_path, use_time_feat=True):
    def to_str(str_or_bytes):
        if isinstance(str_or_bytes, str):
            return str_or_bytes
        else:
            return str_or_bytes.decode()

    d_conv = {0: (lambda x: float(to_str(x).split(" ")[1].split(":")[0]))}
    raw = numpy.loadtxt(csv_path, delimiter=",", skiprows=1, converters=d_conv)
    features = raw.astype(numpy.float32)
    if use_time_feat:
        features[:, 0] /= 23.0
    else:
        features = features[:, 1:]
    return features


class ETTh1Dataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_path, window, horizon, use_time_feat=True, start=0, end=None
    ):
        feats = load_etth1(csv_path, use_time_feat=use_time_feat)
        feats = feats[start:]
        if end is not None:
            feats = feats[:end]
        self.feats = feats
        self.window = window
        self.horizon = horizon
        self.max_start = len(feats) - window - horizon + 1
        if self.max_start < 1:
            raise ValueError("Window + horizon exceeds series length")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        past = self.feats[idx : idx + self.window]
        future = self.feats[idx + self.window : idx + self.window + self.horizon, -1:]
        return torch.from_numpy(past), torch.from_numpy(future)


def build_etth1_dataloaders(csv_path, window=96, horizon=24, batch_size=64, split=0.8):
    dataset = ETTh1Dataset(csv_path, window, horizon)
    n = len(dataset)
    n_train = int(split * n)

    train_ds = ETTh1Dataset(csv_path, window, horizon, end=n_train)
    valid_ds = ETTh1Dataset(csv_path, window, horizon, start=n_train)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False
    )

    # Get input dimension
    sample_past, _ = train_ds[0]
    input_dim = sample_past.shape[-1]

    return train_dl, valid_dl, input_dim


class RevIN(nn.Module):
    def __init__(self, num_features, target_channel, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        self.target_channel = target_channel

    def forward(self, x, mode):
        if mode == "norm":
            # x is (B,T,c)
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.std = torch.std(x, dim=1, keepdim=True) + self.eps
            x_norm = (x - self.mean) / self.std
            x_scaled = x_norm * self.gamma + self.beta
            return x_scaled
        elif mode == "denorm":
            x_denorm = (x - self.beta[self.target_channel]) / self.gamma[
                self.target_channel
            ]
            x_rescaled = (
                x_denorm * self.std[self.target_channel]
                + self.mean[self.target_channel]
            )
            return x_rescaled
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")


class PatchTST(nn.Module):
    """Patch-based Time Series Transformer for univariate forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        """
        Args:
            seq_len: input sequence length (window size)
            pred_len: prediction horizon length
            num_features: number of input channels
            patch_len: length of each patch
            stride: stride for patch creation
            d_model: model dimension
            n_heads: number of attention heads
            n_layers: number of transformer layers
            d_ff: feedforward dimension
            dropout: dropout rate
            revin: whether to use RevIN
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.revin = revin

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # RevIN
        if revin:
            self.revin_layer = RevIN(num_features=num_features, target_channel=-1)

        # Patch embedding: project patches to d_model
        self.patch_embedding = nn.Linear(patch_len * num_features, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.head = nn.Linear(d_model, pred_len)

        self.flatten = nn.Flatten()

    def create_patches(self, x):
        """
        Create patches from input sequence.
        Args:
            x: (batch, seq_len, 1) for univariate
        Returns:
            patches: (batch, num_patches, patch_len)
        """
        B = x.shape[0]
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :]  # (batch, patch_len, c)
            patches.append(patch.view(B, -1))  # (batch, patch_len * c)
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * c)
        return patches

    def forward(self, x):
        # X: (B,T,C)
        if self.revin:
            x = self.revin_layer(x, mode="norm")
        patches = self.create_patches(x)
        patches = self.patch_embedding(patches)
        patches = patches + self.pos_encoding
        patches = self.transformer(patches)
        x = self.head(patches.mean(dim=1))[:, :, None]
        if self.revin:
            x = self.revin_layer(x, mode="denorm")
        return x


class IndPatchTST(nn.Module):
    """Patch-based Time Series Transformer for univariate forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        """
        Args:
            seq_len: input sequence length (window size)
            pred_len: prediction horizon length
            num_features: number of input channels
            patch_len: length of each patch
            stride: stride for patch creation
            d_model: model dimension
            n_heads: number of attention heads
            n_layers: number of transformer layers
            d_ff: feedforward dimension
            dropout: dropout rate
            revin: whether to use RevIN
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.revin = revin

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # RevIN
        if revin:
            self.revin_layer = RevIN(num_features=num_features, target_channel=-1)

        # Patch embedding: project patches to d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.head = nn.Linear(d_model, pred_len)

        self.flatten = nn.Flatten()

    def create_patches(self, x):
        """
        Create patches from input sequence.
        Args:
            x: (batch, seq_len, 1) for univariate
        Returns:
            patches: (batch, num_patches, patch_len)
        """
        B = x.shape[0]
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :]  # (batch, patch_len, c)
            patches.append(patch.view(B, -1))  # (batch, patch_len * c)
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * c)
        return patches

    def forward(self, x):
        # X: (B,T,C)
        if self.revin:
            x = self.revin_layer(x, mode="norm")
        B, T, C = x.shape
        x = x.permute((0, 2, 1))  # B,C,T
        x = x.reshape(B * C, T, 1)
        patches = self.create_patches(x)
        patches = self.patch_embedding(patches)
        patches = patches + self.pos_encoding
        patches = self.transformer(patches)  # (B*C,P,model_dim)
        patches = patches.reshape(B, C, -1, self.d_model)
        x = self.head(patches.mean(dim=(1, 2)))[:, :, None]  # Y true (B,H,1)
        if self.revin:
            x = self.revin_layer(x, mode="denorm")
        return x


def train_epoch(model, dataloader, optimizer, criterion, device="cuda"):
    model.to(device)
    model.train()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        pred = model(past)  # pred: (batch, horizon, 1)
        loss = criterion(pred, future)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device="cuda"):
    model.to(device)
    model.eval()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        pred = model(past)
        loss = criterion(pred, future)
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


def train_and_valid_loop(
    model, train_dl, valid_dl, optimizer, criterion, n_epochs, device="cuda"
):
    logs = {"train_loss": [], "valid_loss": []}
    print(model.__class__.__name__)
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device=device)
        logs["train_loss"].append(train_loss)
        valid_loss = eval_epoch(model, valid_dl, criterion, device=device)
        logs["valid_loss"].append(valid_loss)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | valid={valid_loss:.4f}")
    return logs


def save_forecast_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = 96
    horizon = 24
    train_dl, valid_dl, input_dim = build_etth1_dataloaders(
        "..\\data\\ETTh1.csv", window=window, horizon=horizon
    )
    print(f"Input dimension: {input_dim}")
    model1 = PatchTST(
        window,
        horizon,
        num_features=train_dl.dataset.feats.shape[1],
        patch_len=1,
        stride=1,
        revin=True,
    )
    optimizer = torch.optim.Adam(model1.parameters(), 1e-3)
    criterion = nn.MSELoss()
    train_and_valid_loop(
        model=model1,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=5,
        device=device,
    )
    save_forecast_model(model1, "models\\patchtst_etth1.pth")
