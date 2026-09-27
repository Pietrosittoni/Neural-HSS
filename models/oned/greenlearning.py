import torch
import torch.nn as nn
import numpy as np
# Rational activation
class Rational(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeffs = nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([
            [1.1915, 0.0],
            [1.5957, 2.383],
            [0.5, 0.0],
            [0.0218, 1.0]
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        return PQ[..., 0] / PQ[..., 1]


# Neural operator model
class GreenLearning(nn.Module):
    def __init__(self, s, embed_dim=64, depth=4):
        """ls
        s: number of discretization points
        embed_dim: hidden size of each MLP
        depth: number of rational layers per row
        """
        super().__init__()
        self.s = s

        # MLP to learn G(x_i, x_j)
        dims = [2] + [embed_dim] * depth + [1]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.rationals = nn.ModuleList([Rational() for _ in range(depth)])

        # Fixed 1D grid (s points)
        grid = torch.linspace(0, 1, s + 2)[1:-1]  # exclude boundaries
        self.register_buffer('grid', grid)

    def forward(self, f):
        """
        f: tensor of shape (batch_size, s)
        Returns: u of shape (batch_size, s)
        """
        _, s = f.shape

        # Build (s^2, 2) pairs of (x_i, x_j)
        x_i = self.grid.repeat(self.s)          # (s * s,)
        x_j = self.grid.repeat_interleave(self.s)  # (s * s,)
        input_coords = torch.stack([x_i, x_j], dim=1)  # (s^2, 2)

        # Pass through MLP to compute G(x_i, x_j)
        x = input_coords
        for lin, act in zip(self.linears[:-1], self.rationals):
            x = act(lin(x))
        G_entries = self.linears[-1](x).squeeze(-1)  # (s^2,)

        # Form full operator matrix G (s x s)
        G = G_entries.view(self.s, self.s)  # (s, s)
        
        # Apply operator to f: u = G @ f.T for each sample
        u = torch.matmul(f, G.T)  # (batch_size, s)
        return u
