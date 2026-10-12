"""HSS layers used by Neural-HSS.

The 1D factor is a hierarchical off-diagonal low-rank (HSS-style) matrix.
For m > 1 the layer follows Eq. (1): a CP expansion of modal products
with independent 1D HSS factors along each spatial axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class TrainableLeakyReLU(nn.Module):
    """LeakyReLU with a learnable slope ``alpha``.

    For linear PDEs the slope is driven toward 1 (identity) by the training
    regularizer, which is the setting used in Theorem 2.3.
    """

    def __init__(self, init_slope: float = 0.0) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(init_slope)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.slope * x)


class HSSNode(nn.Module):
    """Recursive HSS / HODLR block.

    Leaf nodes store a dense block. Interior nodes store two diagonal children
    and rank-``r`` factors for the off-diagonal couplings.
    """

    def __init__(self, rows: int, cols: int, levels: int, rank: int) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.levels = levels
        self.rank = rank

        if levels == 0:
            self.is_leaf = True
            self.full = nn.Parameter(torch.empty(rows, cols))
        else:
            self.is_leaf = False
            r1, r2 = rows // 2, rows - rows // 2
            c1, c2 = cols // 2, cols - cols // 2
            self.r1, self.r2, self.c1, self.c2 = r1, r2, c1, c2
            self.child1 = HSSNode(r1, c1, levels - 1, rank)
            self.child2 = HSSNode(r2, c2, levels - 1, rank)
            self.U12 = nn.Parameter(torch.empty(r1, rank))
            self.V12 = nn.Parameter(torch.empty(c2, rank))
            self.U21 = nn.Parameter(torch.empty(r2, rank))
            self.V21 = nn.Parameter(torch.empty(c1, rank))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.is_leaf:
            nn.init.orthogonal_(self.full)
            return
        for factor in (self.U12, self.V12, self.U21, self.V21):
            nn.init.orthogonal_(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the HSS matrix to ``x``.

        Parameters
        ----------
        x:
            ``(cols,)`` or ``(batch, cols)``.
        """
        batched = x.dim() == 2
        if self.is_leaf:
            return x @ self.full.T if batched else self.full @ x

        if batched:
            x1, x2 = x[:, : self.c1], x[:, self.c1 :]
        else:
            x1, x2 = x[: self.c1], x[self.c1 :]

        y1 = self.child1(x1)
        y2 = self.child2(x2)

        if batched:
            y1 = y1 + (x2 @ self.V12) @ self.U12.T
            y2 = y2 + (x1 @ self.V21) @ self.U21.T
            return torch.cat([y1, y2], dim=1)

        y1 = y1 + self.U12 @ (self.V12.T @ x2)
        y2 = y2 + self.U21 @ (self.V21.T @ x1)
        return torch.cat([y1, y2], dim=0)


def _as_sizes(size: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(size, int):
        return (size,) * ndim
    sizes = tuple(int(s) for s in size)
    if len(sizes) != ndim:
        raise ValueError(f"Expected {ndim} sizes, got {sizes}.")
    return sizes


def _mode_product(x: torch.Tensor, hss: HSSNode, axis: int) -> torch.Tensor:
    """Apply a 1D HSS matrix along one spatial axis of a batched field."""
    x_perm = x.movedim(axis, -1)
    leading = x_perm.shape[:-1]
    y = hss(x_perm.reshape(-1, x_perm.shape[-1]))
    return y.reshape(*leading, -1).movedim(-1, axis)


class ChannelMix(nn.Module):
    """Pointwise channel mixing, used as a lift or projection (Appendix E)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,bc...->bo...", self.weight, x) + self.bias.view(
            (1, -1) + (1,) * (x.dim() - 2)
        )


class HSSLayer(nn.Module):
    """HSS layer in any spatial dimension through CP expansion approximation.

    * ``ndim = 1``: a single HSS matrix, as in the one-dimensional model.
    * ``ndim > 1``: the m-dimensional layer from Eq. (1),

      ``H(Z) = sum_k Z ×_1 W_1^{(k)} ×_2 ... ×_m W_m^{(k)}``

      with outer CP-rank ``outer_rank``.
    """

    def __init__(
        self,
        in_size: int | Sequence[int],
        out_size: int | Sequence[int],
        levels: int,
        rank: int,
        ndim: int = 1,
        outer_rank: int = 1,
    ) -> None:
        super().__init__()
        if ndim < 1:
            raise ValueError("ndim must be at least 1.")
        if levels < 0:
            raise ValueError("levels must be non-negative.")

        self.ndim = ndim
        self.in_sizes = _as_sizes(in_size, ndim)
        self.out_sizes = _as_sizes(out_size, ndim)
        self.outer_rank = 1 if ndim == 1 else outer_rank

        for size in self.in_sizes + self.out_sizes:
            if size % (2**levels) != 0:
                raise ValueError("Each spatial size must be divisible by 2^levels.")

        if ndim == 1:
            self.hss = HSSNode(self.out_sizes[0], self.in_sizes[0], levels, rank)
            self.factors = None
        else:
            self.hss = None
            self.factors = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            HSSNode(self.out_sizes[axis], self.in_sizes[axis], levels, rank)
                            for axis in range(ndim)
                        ]
                    )
                    for _ in range(self.outer_rank)
                ]
            )

        self.bias = nn.Parameter(torch.zeros(self.out_sizes[0]))

    def _bias_view(self) -> torch.Tensor:
        return self.bias.view((1, self.out_sizes[0]) + (1,) * (self.ndim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != self.ndim + 1:
            raise ValueError(
                f"Expected input of shape (batch, {', '.join(['D'] * self.ndim)}), got {tuple(x.shape)}."
            )
        for axis, size in enumerate(self.in_sizes):
            if x.shape[axis + 1] != size:
                raise ValueError(
                    f"Expected size {size} along spatial axis {axis}, got {x.shape[axis + 1]}."
                )

        if self.ndim == 1:
            return self.hss(x) + self._bias_view()

        y = x.new_zeros((x.shape[0],) + self.out_sizes)
        for factors in self.factors:
            term = x
            for axis, weight in enumerate(factors):
                term = _mode_product(term, weight, axis=axis + 1)
            y = y + term
        return y + self._bias_view()
