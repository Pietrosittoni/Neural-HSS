import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepONet3D(nn.Module):
    def __init__(self, n, branch_hidden_dim, trunk_hidden_dim, embedding_dim):
        """
        n: number of discretization points per axis (total grid: n x n x n)
        """
        super().__init__()

        self.n = n
        
        # 3D grid points (excluding endpoint for consistency)
        x_grid = np.linspace(0, 1, n, endpoint=False)
        xx, yy, zz = np.meshgrid(x_grid, x_grid, x_grid, indexing='ij')
        coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (n*n*n, 3)
        self.register_buffer('xyz_grid', torch.tensor(coords, dtype=torch.float32))  # (n*n*n, 3)

        # Branch net (takes discretized f as input, flattened)
        self.branch = nn.Sequential(
            nn.Linear(n * n * n, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, embedding_dim)
        )

        # Trunk net (takes 3D coordinates as input)
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, embedding_dim)
        )

    def forward(self, f):
        # f: (batch, n, n, n)
        batch_size = f.shape[0]
        
        # Flatten input for branch net
        f_flat = f.reshape(batch_size, self.n * self.n * self.n)  # (batch, n*n*n)
        
        # Get embeddings from branch and trunk nets
        branch_out = self.branch(f_flat)        # (batch, embed_dim)
        trunk_out = self.trunk(self.xyz_grid)   # (n*n*n, embed_dim)
        
        # Outer product and reshape to (batch, n, n, n)
        out = torch.einsum('be,me->bm', branch_out, trunk_out)  # (batch, n*n*n)
        out = out.reshape(batch_size, self.n, self.n, self.n)   # (batch, n, n, n)
        
        return out  # (batch, n, n, n)


#def test_deeponet3d():
#    # Check for CUDA availability and select device
#    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    print(f"Using device: {device}")
#    
#    # Initialize CUDA if available
#    if device.type == 'cuda':
#        torch.cuda.init()
#        torch.cuda.empty_cache()
#        torch.cuda.reset_peak_memory_stats(device)
#    
#    model = DeepONet3D(n=32, branch_hidden_dim=64, trunk_hidden_dim=64, embedding_dim=32).to(device)
#    x = torch.randn(16, 32, 32, 32).to(device)  # batch of 5, 16x16x16 input grid
#    y = model(x)
#    
#    print(f"Input shape: {x.shape}")
#    print(f"Output shape: {y.shape}")
#    
#    # Measure memory peak if using CUDA
#    if device.type == 'cuda':
#        memory_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # Convert to MB
#        print(f"Memory peak: {memory_peak:.2f} MB")
#    
#    return y
#
## Run test function when script is executed directly
#if __name__ == "__main__":
#    test_deeponet3d()