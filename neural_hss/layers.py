"""HSS layers used by Neural-HSS.

The 1D factor is a hierarchical off-diagonal low-rank (HSS-style) matrix.
The 3D layer follows Eq. (1) of the paper: a CP expansion of modal products
with 1D HSS factors along each spatial axis.
"""

from __future__ import annotations

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


def _mode_product(x: torch.Tensor, hss: HSSNode, axis: int) -> torch.Tensor:
    """Apply a 1D HSS matrix along one spatial axis of a 3D field."""
    x_perm = x.movedim(axis, -1)
    leading = x_perm.shape[:-1]
    y = hss(x_perm.reshape(-1, x_perm.shape[-1]))
    return y.reshape(*leading, -1).movedim(-1, axis)


class HSS3DLayer(nn.Module):
    """Three-dimensional HSS layer (paper Eq. 1).

    Maps a field ``Z`` of shape ``(batch, d, d, d)`` by

    ``H(Z) = sum_k Z ×_1 W_1^{(k)} ×_2 W_2^{(k)} ×_3 W_3^{(k)}``

    where each ``W_j^{(k)}`` is a 1D HSS matrix and ``k = 1, ..., r_out``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        levels: int,
        rank: int,
        outer_rank: int,
    ) -> None:
        super().__init__()
        if levels < 0:
            raise ValueError("levels must be non-negative.")
        if in_size % (2**levels) != 0 or out_size % (2**levels) != 0:
            raise ValueError("in_size and out_size must be divisible by 2^levels.")

        self.in_size = in_size
        self.out_size = out_size
        self.outer_rank = outer_rank
        self.factors = nn.ModuleList(
            [
                nn.ModuleList(
                    [HSSNode(out_size, in_size, levels, rank) for _ in range(3)]
                )
                for _ in range(outer_rank)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, D, D, D), got {tuple(x.shape)}.")
        if x.shape[-1] != self.in_size:
            raise ValueError(f"Expected last spatial size {self.in_size}, got {x.shape[-1]}.")

        y = x.new_zeros(x.shape[0], self.out_size, self.out_size, self.out_size)
        for w1, w2, w3 in self.factors:
            term = _mode_product(x, w3, axis=3)
            term = _mode_product(term, w2, axis=2)
            term = _mode_product(term, w1, axis=1)
            y = y + term
        return y + self.bias.view(1, -1, 1, 1)
