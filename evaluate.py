#!/usr/bin/env python3
"""Evaluate a trained Neural-HSS checkpoint on the 3D Poisson test set."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from neural_hss.config import Config
from neural_hss.data import load_poisson3d, make_dataloaders
from neural_hss.metrics import RelativeL2
from neural_hss.model import NeuralHSS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Neural-HSS on 3D Poisson.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/poisson3d.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--plot", type=str, default=None, help="Optional path for a mid-slice figure.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.device is not None:
        config.device = args.device

    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    scale = float(ckpt.get("scale", 1.0))

    x_train, y_train, x_val, y_val = load_poisson3d(
        config.data_path,
        train_size=int(config.train_size),
        val_size=int(config.get("val_size", 200)),
        grid_size=int(config.grid_size),
    )
    _, val_loader, _ = make_dataloaders(
        x_train, y_train, x_val, y_val, batch_size=int(config.batch_size), device=device
    )

    model = NeuralHSS(
        size=int(config.grid_size),
        ndim=int(config.get("ndim", 3)),
        depth=int(config.depth),
        levels=int(config.levels),
        rank=int(config.rank),
        outer_rank=int(config.outer_rank),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mse = nn.MSELoss()
    rel_l2 = RelativeL2()
    total_mse = 0.0
    total_rel = 0.0
    last_pred = last_target = None
    for inputs, targets in val_loader:
        preds = model(inputs) * scale
        total_mse += mse(preds, targets).item()
        total_rel += rel_l2(preds, targets).item()
        last_pred, last_target = preds, targets

    n_batches = max(len(val_loader), 1)
    print(f"test MSE:          {total_mse / n_batches:.6e}")
    print(f"test relative L2:  {total_rel / n_batches:.6e}")

    if args.plot and last_pred is not None:
        mid = last_target.shape[-1] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        panels = [
            (last_target[0, mid].cpu(), "Ground truth"),
            (last_pred[0, mid].cpu(), "Neural-HSS"),
            ((last_pred[0, mid] - last_target[0, mid]).abs().cpu(), "Absolute error"),
        ]
        for ax, (image, title) in zip(axes, panels):
            im = ax.imshow(image, cmap="magma")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
