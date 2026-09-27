import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepONet1D(nn.Module):
    def __init__(self, n, branch_hidden_dim, trunk_hidden_dim, embedding_dim):
        super().__init__()

        self.n = n
        x_grid = np.linspace(0, 1, n, endpoint=False)
        self.register_buffer('x_grid', torch.tensor(x_grid, dtype=torch.float32).unsqueeze(-1))

        # Branch net (takes discretized f as input)
        self.branch = nn.Sequential(
            nn.Linear(n, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, embedding_dim)
        )

        # Trunk net (takes coordinates y as input)
        self.trunk = nn.Sequential(
            nn.Linear(1, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, embedding_dim)
        )

    def forward(self, f):
        # f: (batch, n) where n is the number of discretization points
        batch_size = f.shape[0]

        # Get trunk evaluation points from internal x_grid
        y = self.x_grid  # (n, 1)

        # Get embeddings from branch and trunk nets
        branch_out = self.branch(f)        # (batch, embed_dim)
        trunk_out = self.trunk(y)          # (n, embed_dim)

        # Outer product: (batch, n)
        out = torch.einsum('be,ne->bn', branch_out, trunk_out)

        return out  # (batch, n)



