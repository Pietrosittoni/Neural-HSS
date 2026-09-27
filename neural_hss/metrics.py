"""Training loss and evaluation metrics from Appendix G."""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeL2(nn.Module):
    """Batch-mean relative L2 error (Appendix G, Eq. 2)."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, predictions.dim()))
        diff = torch.linalg.vector_norm(predictions - targets, ord=2, dim=reduce_dims)
        denom = torch.linalg.vector_norm(targets, ord=2, dim=reduce_dims).clamp_min(self.epsilon)
        return (diff / denom).mean()
