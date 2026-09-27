"""3D Poisson dataset I/O used by the data-efficiency experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _as_fields(array: np.ndarray, grid_size: int | None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if tensor.dim() == 2:
        n = grid_size or int(round(tensor.shape[1] ** (1.0 / 3.0)))
        if n**3 != tensor.shape[1]:
            raise ValueError(f"Cannot reshape flattened fields of length {tensor.shape[1]} into a cube.")
        tensor = tensor.reshape(tensor.shape[0], n, n, n)
    if tensor.dim() != 4:
        raise ValueError(f"Expected fields of shape (N, n, n, n), got {tuple(tensor.shape)}.")
    return tensor


def load_poisson3d(
    path: str | Path,
    train_size: int,
    val_size: int = 200,
    grid_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load forcing/solution pairs and split into train/test sets.

    The paper uses the first ``train_size`` samples for training and the last
    ``val_size`` samples for testing (Appendix F.2.3: 256 train + 200 test).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as handle:
            x = _as_fields(handle["X"][:], grid_size)
            y = _as_fields(handle["Y"][:], grid_size)
    elif suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path, engine="pyarrow")
        x = _as_fields(np.array(list(frame["X"].values)), grid_size)
        y = _as_fields(np.array(list(frame["Y"].values)), grid_size)
    elif suffix == ".npz":
        archive = np.load(path)
        x = _as_fields(archive["X"], grid_size)
        y = _as_fields(archive["Y"], grid_size)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if train_size + val_size > x.shape[0]:
        raise ValueError(
            f"Need {train_size + val_size} samples, but the dataset only contains {x.shape[0]}."
        )

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[-val_size:], y[-val_size:]
    return x_train, y_train, x_val, y_val


def make_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, float]:
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    scale = float(y_train.abs().max().clamp_min(1e-8))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=min(batch_size, len(x_val)),
        shuffle=False,
    )
    return train_loader, val_loader, scale
