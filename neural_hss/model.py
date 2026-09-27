"""Neural-HSS architecture for fields of any spatial dimension."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from neural_hss.layers import ChannelMix, HSSLayer, TrainableLeakyReLU, _as_sizes


class NeuralHSS(nn.Module):
    """Neural-HSS model for 1D, 2D, 3D, or higher-dimensional fields.

    Each block is an HSS layer (Eq. 1 for ``ndim > 1``) followed by a
    trainable LeakyReLU. Optional channel lift / projection follows
    Appendix E when ``in_channels`` or ``out_channels`` is greater than 1.

    Examples
    --------
    >>> NeuralHSS(size=256, ndim=1, depth=3, levels=3, rank=2)
    >>> NeuralHSS(size=64, ndim=2, depth=3, levels=2, rank=2, outer_rank=8)
    >>> NeuralHSS(size=128, ndim=3, depth=1, levels=2, rank=4, outer_rank=2)
    """

    def __init__(
        self,
        size: int | Sequence[int],
        ndim: int = 1,
        depth: int = 1,
        levels: int = 2,
        rank: int = 2,
        outer_rank: int = 1,
        hidden_size: int | Sequence[int] | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if ndim < 1:
            raise ValueError("ndim must be at least 1.")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be at least 1.")

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_sizes = _as_sizes(size, ndim)
        width = self.in_sizes if hidden_size is None else _as_sizes(hidden_size, ndim)
        out_sizes = self.in_sizes
        sizes = [self.in_sizes] + [width] * (depth - 1) + [out_sizes]

        self.lift = ChannelMix(in_channels, 1) if in_channels > 1 else None
        blocks: list[nn.Module] = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            blocks.append(
                HSSLayer(
                    in_size=in_size,
                    out_size=out_size,
                    levels=levels,
                    rank=rank,
                    ndim=ndim,
                    outer_rank=outer_rank,
                )
            )
            blocks.append(TrainableLeakyReLU())
        self.layers = nn.Sequential(*blocks)
        self.project = ChannelMix(1, out_channels) if out_channels > 1 else None

    def _as_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == self.ndim + 1:
            return x.unsqueeze(1)
        if x.dim() == self.ndim + 2:
            return x
        raise ValueError(
            f"Expected (batch, *spatial) or (batch, channels, *spatial) with ndim={self.ndim}, "
            f"got {tuple(x.shape)}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_channels(x)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}.")

        if self.lift is not None:
            x = self.lift(x)
        x = self.layers(x.squeeze(1))
        if self.project is not None:
            return self.project(x.unsqueeze(1))
        return x

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
