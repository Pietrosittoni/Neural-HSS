import torch
import torch.nn as nn

# class Rational(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.coeffs = nn.Parameter(torch.Tensor(4, 2))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.coeffs.data = torch.Tensor([
#             [1.1915, 0.0],
#             [1.5957, 2.383],
#             [0.5, 0.0],
#             [0.0218, 1.0]
#         ])

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         self.coeffs.data[0,1].zero_()
#         exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
#         X = torch.pow(input.unsqueeze(-1), exp)
#         PQ = X @ self.coeffs
#         return PQ[..., 0] / PQ[..., 1]

class Rational(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeffs = nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()
        self.relu = nn.ReLU()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([
            [1.1915, 0.0],
            [1.5957, 2.383],
            [0.5, 0.0],
            [0.0218, 1.0]
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(input)

class GreenLearning2D(nn.Module):
    def __init__(self, s, embed_dim=64, depth=4):
        """
        s: number of discretization points per axis (total grid: s x s)
        embed_dim: hidden size of each MLP
        depth: number of rational layers per row
        """
        super().__init__()
        self.s = s

        # MLP to learn G((x_i, y_i), (x_j, y_j))
        dims = [4] + [embed_dim] * depth + [1]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.rationals = nn.ModuleList([Rational() for _ in range(depth)])

        # Fixed 2D grid (s x s points)
        grid_1d = torch.linspace(0, 1, s + 2)[1:-1]  # exclude boundaries
        grid_x, grid_y = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

    def forward(self, f):
        """
        f: tensor of shape (batch_size, s, s)
        Returns: u of shape (batch_size, s, s)
        """
        batch_size, s1, s2 = f.shape
        assert s1 == self.s and s2 == self.s

        # Build (s^4, 4) pairs: (x_i, y_i, x_j, y_j)
        grid_points = torch.stack([self.grid_x.flatten(), self.grid_y.flatten()], dim=1)  # (s*s, 2)
        x_i = grid_points.repeat(self.s * self.s, 1)  # (s*s*s*s, 2)
        x_j = grid_points.repeat_interleave(self.s * self.s, dim=0)  # (s*s*s*s, 2)
        input_coords = torch.cat([x_i, x_j], dim=1)  # (s*s*s*s, 4)

        # Pass through MLP to compute G((x_i, y_i), (x_j, y_j))
        x = input_coords
        for lin, act in zip(self.linears[:-1], self.rationals):
            x = act(lin(x))
        G_entries = self.linears[-1](x).squeeze(-1)  # (s*s*s*s,)

        # Form full operator matrix G (s*s, s*s)
        G = G_entries.view(self.s * self.s, self.s * self.s)  # (s*s, s*s)

        # Flatten input f to (batch_size, s*s)
        f_flat = f.view(batch_size, -1)  # (batch_size, s*s)

        # Apply operator to f: u = G @ f.T for each sample
        u_flat = torch.matmul(f_flat, G.T)  # (batch_size, s*s)
        u = u_flat.view(batch_size, self.s, self.s)  # (batch_size, s, s)
        return u