import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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

class GreenLearning3D(nn.Module):
    def __init__(self, s, embed_dim=64, depth=4):
        """
        s: number of discretization points per axis (total grid: s x s x s)
        embed_dim: hidden size of each MLP
        depth: number of rational layers per row
        """
        super().__init__()
        self.s = s

        # MLP to learn G((x_i, y_i, z_i), (x_j, y_j, z_j))
        dims = [6] + [embed_dim] * depth + [1]  # 6 inputs: (x_i, y_i, z_i, x_j, y_j, z_j)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.rationals = nn.ModuleList([Rational() for _ in range(depth)])

        # Fixed 3D grid (s x s x s points)
        grid_1d = torch.linspace(0, 1, s + 2)[1:-1]  # exclude boundaries
        grid_x, grid_y, grid_z = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)
        self.register_buffer('grid_z', grid_z)

    def forward(self, f):
        """
        f: tensor of shape (batch_size, s, s, s)
        Returns: u of shape (batch_size, s, s, s)
        """
        batch_size, s1, s2, s3 = f.shape
        assert s1 == self.s and s2 == self.s and s3 == self.s

        # Build (s^6, 6) pairs: (x_i, y_i, z_i, x_j, y_j, z_j)
        grid_points = torch.stack([self.grid_x.flatten(), self.grid_y.flatten(), self.grid_z.flatten()], dim=1)  # (s*s*s, 3)
        s_cubed = self.s ** 3
        x_i = grid_points.repeat(s_cubed, 1)  # (s^3*s^3, 3)
        x_j = grid_points.repeat_interleave(s_cubed, dim=0)  # (s^3*s^3, 3)
        input_coords = torch.cat([x_i, x_j], dim=1)  # (s^3*s^3, 6)

        # Pass through MLP to compute G((x_i, y_i, z_i), (x_j, y_j, z_j))
        x = input_coords
        for lin, act in zip(self.linears[:-1], self.rationals):
            x = act(lin(x))
        G_entries = self.linears[-1](x).squeeze(-1)  # (s^3*s^3,)

        # Form full operator matrix G (s^3, s^3)
        G = G_entries.view(s_cubed, s_cubed)  # (s^3, s^3)

        # Flatten input f to (batch_size, s^3)
        f_flat = f.view(batch_size, -1)  # (batch_size, s^3)

        # Apply operator to f: u = G @ f.T for each sample
        u_flat = torch.matmul(f_flat, G.T)  # (batch_size, s^3)
        u = u_flat.view(batch_size, self.s, self.s, self.s)  # (batch_size, s, s, s)
        return u


def test_greenlearning3d():
    try:
        # Try to import performance measurement utilities
        from net_utils import get_flops, count_params
        has_utils = True
    except ImportError:
        print("Note: net_utils module not found. Performance metrics will be limited.")
        has_utils = False
        
        # Define a simple function to count parameters
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create a GreenLearning3D model - using smaller s for testing
    s = 32 # discretization points per axis (reduced size to avoid memory issues)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL = GreenLearning3D(s=s, embed_dim=1, depth=1).to(device)
    
    # Create random input data
    X = torch.randn(1, s, s, s).to(device)  # batch of 4, s×s×s input
    
    # Run model and measure performance
    print(f"GreenLearning3D model for {s}×{s}×{s} grid")
    print(f"Model parameters: {count_params(MODEL)}")
    
    output = MODEL(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    
    if has_utils:
        print(f"FLOPs: {get_flops(MODEL, X)}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"Memory peak: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    
    # Test with different batch size
    X_large = torch.randn(8, s, s, s).to(device)  # larger batch
    output_large = MODEL(X_large)
    print(f"\nLarger batch test - shape: {output_large.shape}")
    
    if torch.cuda.is_available():
        print(f"Memory peak: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    test_greenlearning3d()