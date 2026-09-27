#!/usr/bin/env python3
"""Train Neural-HSS on the 3D Poisson data-efficiency experiment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neural_hss.config import Config
from neural_hss.data import load_poisson3d, make_dataloaders
from neural_hss.metrics import RelativeL2
from neural_hss.model import NeuralHSS3D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Neural-HSS on 3D Poisson.")
    parser.add_argument("--config", type=str, default="configs/poisson3d.yaml")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader, scale: float, mse: nn.Module, rel_l2: RelativeL2) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_rel = 0.0
    for inputs, targets in loader:
        preds = model(inputs) * scale
        total_mse += mse(preds, targets).item()
        total_rel += rel_l2(preds, targets).item()
    n_batches = max(len(loader), 1)
    return total_mse / n_batches, total_rel / n_batches


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.device is not None:
        config.device = args.device
    if args.train_size is not None:
        config.train_size = args.train_size
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    set_seed(int(config.seed))
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    x_train, y_train, x_val, y_val = load_poisson3d(
        config.data_path,
        train_size=int(config.train_size),
        val_size=int(config.get("val_size", 200)),
        grid_size=int(config.grid_size),
    )
    train_loader, val_loader, scale = make_dataloaders(
        x_train, y_train, x_val, y_val, batch_size=int(config.batch_size), device=device
    )
    print(f"Train samples: {len(x_train)}  |  Test samples: {len(x_val)}  |  scale={scale:.4e}")

    model = NeuralHSS3D(
        grid_size=int(config.grid_size),
        depth=int(config.depth),
        levels=int(config.levels),
        rank=int(config.rank),
        outer_rank=int(config.outer_rank),
    ).to(device)
    print(f"Neural-HSS parameters: {model.num_parameters:,}")

    use_wandb = bool(config.get("use_wandb", False))
    if use_wandb:
        import wandb

        wandb.init(project=config.get("project", "neural-hss"), config=config.__dict__)
        wandb.watch(model)

    mse = nn.MSELoss()
    rel_l2 = RelativeL2()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=tuple(config.get("betas", (0.9, 0.99))),
    )
    max_steps = int(config.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=float(config.learning_rate_min),
    )
    slope_reg = float(config.get("slope_reg", 1e-8))
    grad_clip = float(config.get("grad_clip", 1.0))
    eval_every = int(config.get("eval_every", 200))

    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_rel = float("inf")
    step = 0
    model.train()

    while step < max_steps:
        for inputs, targets in train_loader:
            targets_scaled = targets / scale
            preds = model(inputs)
            loss = mse(preds, targets_scaled) + 0.5 * slope_reg * model.slope_regularizer()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 50 == 0 or step == 1:
                print(f"step {step}/{max_steps}  train_mse={loss.item():.6e}  lr={scheduler.get_last_lr()[0]:.2e}")

            if use_wandb:
                wandb.log({"train/loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0], "step": step})

            if step % eval_every == 0 or step == max_steps:
                val_mse, val_rel = evaluate(model, val_loader, scale, mse, rel_l2)
                print(f"  val_mse={val_mse:.6e}  val_rel_l2={val_rel:.6e}")
                if use_wandb:
                    wandb.log({"val/mse": val_mse, "val/relative_l2": val_rel, "step": step})
                if val_rel < best_rel:
                    best_rel = val_rel
                    ckpt = checkpoint_dir / f"neural_hss_n{config.train_size}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config.__dict__,
                            "scale": scale,
                            "val_relative_l2": val_rel,
                            "step": step,
                        },
                        ckpt,
                    )
                    print(f"  saved {ckpt}")
                model.train()

            if step >= max_steps:
                break

    print(f"Training complete. Best relative L2: {best_rel:.6e}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
