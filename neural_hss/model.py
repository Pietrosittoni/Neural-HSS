"""Neural-HSS architecture for fields of any spatial dimension."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from neural_hss.layers import ChannelMix, HSSLayer, TrainableLeakyReLU, _as_size

# "torch": reference PyTorch layers. "cuda": fused CUDA kernels with FP32-level
# accuracy (3xTF32 tensor cores). "cuda-tf32": same kernels, single TF32 pass.
# The CUDA backends cover any ndim on sm_80+ GPUs; see neural_hss/cuda_backbone.py.
BACKENDS = ("torch", "cuda", "cuda-tf32")


class NeuralHSS(nn.Module):
    """Neural-HSS model for 1D, 2D, 3D, or higher-dimensional fields.

    The model of the paper's experiments (``HSS_MLP``, ``HSS_MLP2D`` and
    ``HSS_MLP3D`` of the original code, with ``hidden_sizes = [hidden_size] *
    depth``): ``depth + 1`` HSS layers (Eq. 1 for ``ndim > 1``) with a
    trainable LeakyReLU after every layer except the last. Parameters,
    initialization (for a given seed) and state-dict keys are those of the
    original code, so its checkpoints load directly. An optional pointwise
    channel lift / projection is added when ``in_channels`` or
    ``out_channels`` is greater than 1.

    ``backend="cuda"`` runs each HSS layer and its activation with the fused
    CUDA kernels; parameters and checkpoints are the same for every backend.
    Layers with a spatial size that is not a multiple of 4 keep the PyTorch
    implementation (with a warning).

    Examples
    --------
    >>> NeuralHSS(size=256, ndim=1, depth=3, levels=3, rank=2)  # 1D heat, 45,059 parameters
    >>> NeuralHSS(size=64, ndim=2, depth=2, levels=2, rank=2, outer_rank=8)  # 2D Poisson, 37,058
    >>> NeuralHSS(size=128, ndim=3, depth=1, levels=2, rank=4, outer_rank=2)  # 3D Poisson, 24,833
    """

    def __init__(
        self,
        size: int,
        ndim: int = 1,
        depth: int = 1,
        levels: int = 2,
        rank: int = 2,
        outer_rank: int = 1,
        hidden_size: int | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth (the number of hidden layers) must be non-negative.")
        if ndim < 1:
            raise ValueError("ndim must be at least 1.")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be at least 1.")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
        self.backend = backend

        self.ndim = ndim
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        size = _as_size(size)
        width = size if hidden_size is None else _as_size(hidden_size)
        self.in_sizes = (size,) * ndim
        sizes = [size] + [width] * depth + [size]
        init_slope = 1.0 if ndim == 1 else 0.0  # as in the original 1D / 2D and 3D code

        self.lift = ChannelMix(in_channels, 1) if in_channels > 1 else None
        blocks: list[nn.Module] = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
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
            if i < depth:  # no activation after the output layer
                blocks.append(TrainableLeakyReLU(init_slope=init_slope))
        self.layers = nn.Sequential(*blocks)
        self.project = ChannelMix(1, out_channels) if out_channels > 1 else None
        self._cuda_ok: list[bool] = []
        if backend != "torch":
            from neural_hss import cuda_backbone

            reasons = [cuda_backbone.unsupported_reason(b) for b in self.layers if isinstance(b, HSSLayer)]
            self._cuda_ok = [r is None for r in reasons]
            bad = [r for r in reasons if r is not None]
            if bad:
                warnings.warn(
                    f"{len(bad)} of {len(reasons)} HSS layers keep the PyTorch implementation: {bad[0]}.",
                    stacklevel=2,
                )

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
        x = self._backbone(x.squeeze(1))
        if self.project is not None:
            return self.project(x.unsqueeze(1))
        return x

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "torch" or not x.is_cuda or x.dtype != torch.float32:
            return self.layers(x)
        from neural_hss import cuda_backbone

        if not cuda_backbone.is_supported(x.device):
            if not getattr(self, "_warned_fallback", False):
                warnings.warn("The CUDA backbone needs an sm_80+ GPU; falling back to the PyTorch backbone.")
                self._warned_fallback = True
            return self.layers(x)
        precision = "tf32" if self.backend == "cuda-tf32" else "fp32"
        blocks = list(self.layers)
        # [HSS, act, HSS, act, ..., HSS]: the output layer has no activation
        for use_cuda, layer, act in zip(self._cuda_ok, blocks[0::2], blocks[1::2] + [None]):
            if use_cuda:
                x = cuda_backbone.layer_forward(x, layer, act, precision)
            else:
                x = layer(x) if act is None else act(layer(x))
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
