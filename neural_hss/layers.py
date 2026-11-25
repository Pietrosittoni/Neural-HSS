"""HSS layers used by Neural-HSS.

These reproduce the layers of the code behind the paper's experiments
(``HSS_Linear`` of ``models/oned/hss_mat.py`` and ``HSS2D_outer_product`` /
``HSS3D_outer_product`` of ``models/{twod,threed}/hss_nd_mat.py``), with the
same parameters, initialization and state-dict keys.

The 1D factor is a hierarchical off-diagonal low-rank (HODLR) matrix: dense
leaves and independent rank-``r`` factors for the two off-diagonal blocks of
every split. For m > 1 the layer is the CP expansion of Eq. (1) with one HSS
factor per CP term, shared by every spatial axis:

    H(Z) = sum_k Z x_1 W^(k) x_2 W^(k) ... x_m W^(k) + bias,

where the bias varies along spatial axis m - 2 (``bias.view(1, -1, 1)`` in the
original code: axis 0 in 2D, axis 1 in 3D). The 1D layer has no bias.
"""

from __future__ import annotations

import operator

import torch
import torch.nn as nn

INITS = ("orthogonal", "normal")


class TrainableLeakyReLU(nn.Module):
    """LeakyReLU with a learnable slope ``alpha`` (a scalar).

    For linear PDEs the slope is driven toward 1 (identity) by the training
    regularizer, which is the setting used in Theorem 2.3. The original code
    starts it at 0 in the 2D and 3D models and at 1 in the 1D model.
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

    ``init`` is ``"orthogonal"`` (the 2D and 3D models) or ``"normal"``
    (N(0, 0.01^2), the 1D model). As in the original code, every node fills
    its parameters with ``torch.randn`` and then re-initializes its whole
    subtree after its children have done the same, so a given seed produces
    the original weights.
    """

    def __init__(self, rows: int, cols: int, levels: int, rank: int, init: str = "orthogonal") -> None:
        super().__init__()
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}, got {init!r}.")
        self.rows = rows
        self.cols = cols
        self.levels = levels
        self.rank = rank
        self.init_scheme = init

        if levels == 0:
            self.is_leaf = True
            self.full = nn.Parameter(torch.randn(rows, cols))
        else:
            self.is_leaf = False
            r1, r2 = rows // 2, rows - rows // 2
            c1, c2 = cols // 2, cols - cols // 2
            self.r1, self.r2, self.c1, self.c2 = r1, r2, c1, c2
            self.child1 = HSSNode(r1, c1, levels - 1, rank, init)
            self.child2 = HSSNode(r2, c2, levels - 1, rank, init)
            self.U12 = nn.Parameter(torch.randn(r1, rank))
            self.V12 = nn.Parameter(torch.randn(c2, rank))
            self.U21 = nn.Parameter(torch.randn(r2, rank))
            self.V21 = nn.Parameter(torch.randn(c1, rank))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for p in self.parameters():  # the whole subtree, as in the original code
            if self.init_scheme == "orthogonal":
                nn.init.orthogonal_(p)
            else:
                nn.init.normal_(p, std=1e-2)

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


def _as_size(size: int) -> int:
    try:
        return operator.index(size)
    except TypeError:
        raise TypeError(
            f"Spatial sizes must be integers, got {size!r}: every axis has the same size, since "
            "each CP term applies one HSS factor along all of them."
        ) from None


def _mode_product(x: torch.Tensor, hss: HSSNode, axis: int) -> torch.Tensor:
    """Apply a 1D HSS matrix along one spatial axis of a batched field."""
    x_perm = x.movedim(axis, -1)
    leading = x_perm.shape[:-1]
    y = hss(x_perm.reshape(-1, x_perm.shape[-1]))
    return y.reshape(*leading, -1).movedim(-1, axis)


class ChannelMix(nn.Module):
    """Pointwise channel mixing, used as a lift or projection for multi-channel fields.

    Not part of the single-channel models of the paper's Poisson experiments.
    """

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

    * ``ndim = 1``: a single HSS matrix without bias (``HSS_Linear``), with the
      1D initialization N(0, 0.01^2).
    * ``ndim > 1``: the m-dimensional layer from Eq. (1) with outer CP-rank
      ``outer_rank``,

      ``H(Z) = sum_k Z ×_1 W^{(k)} ×_2 ... ×_m W^{(k)} + bias``,

      one orthogonally initialized HSS factor per term, shared by every axis,
      and a bias along spatial axis ``ndim - 2`` (``HSS2D_outer_product`` and
      ``HSS3D_outer_product``).
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
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
        in_size, out_size = _as_size(in_size), _as_size(out_size)

        self.ndim = ndim
        self.in_sizes = (in_size,) * ndim
        self.out_sizes = (out_size,) * ndim

        if ndim == 1:
            if in_size % (2**levels) or out_size % (2**levels):
                raise ValueError("1D sizes must be divisible by 2^levels.")
            self.outer_rank = 1
            self.root = HSSNode(out_size, in_size, levels, rank, init="normal")
            self.hss = None
            self.register_parameter("bias", None)
        else:
            if outer_rank < 1:
                raise ValueError("outer_rank must be at least 1.")
            self.outer_rank = outer_rank
            self.root = None
            # One node per term, listed once per axis (the same module, as `[HSSNode(...)] * m`
            # in the original code), so the parameters and state-dict keys are the original ones.
            self.hss = nn.ModuleList(
                [nn.ModuleList([HSSNode(out_size, in_size, levels, rank)] * ndim) for _ in range(outer_rank)]
            )
            self.bias = nn.Parameter(torch.zeros(out_size))
        # Axis order of the original code (2D: 0, 1; 3D: 2, 1, 0). The products commute, so it
        # only fixes the rounding, which then matches the original code.
        self._axis_order = (0, 1) if ndim == 2 else tuple(range(ndim - 1, -1, -1))

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
            return self.root(x)

        y = x.new_zeros((x.shape[0],) + self.out_sizes)
        for factors in self.hss:
            term = x
            for axis in self._axis_order:
                term = _mode_product(term, factors[axis], axis=axis + 1)
            y = y + term
        return y + self.bias.view(1, -1, 1)  # broadcasts along spatial axis ndim - 2
