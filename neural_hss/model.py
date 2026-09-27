"""Neural-HSS architecture for 3D PDEs."""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_hss.layers import HSS3DLayer, TrainableLeakyReLU


class NeuralHSS3D(nn.Module):
    """Neural-HSS model for three-dimensional fields.

    The network is a stack of 3D HSS layers, each followed by a trainable
    LeakyReLU. Depth 1 with grid size 128 is the configuration used for the
    3D Poisson data-efficiency experiment (Appendix M, Table 9).
    """

    def __init__(
        self,
        grid_size: int,
        depth: int = 1,
        levels: int = 2,
        rank: int = 4,
        outer_rank: int = 2,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        width = grid_size if hidden_size is None else hidden_size
        sizes = [grid_size] + [width] * (depth - 1) + [grid_size]

        blocks: list[nn.Module] = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            blocks.append(HSS3DLayer(in_size, out_size, levels, rank, outer_rank))
            blocks.append(TrainableLeakyReLU())
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def slope_regularizer(self) -> torch.Tensor:
        """``sum_i (alpha_i - 1)^2`` from Theorem 2.3."""
        penalty = next(self.parameters()).new_zeros(())
        for module in self.modules():
            if isinstance(module, TrainableLeakyReLU):
                penalty = penalty + (module.slope - 1.0) ** 2
        return penalty

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
