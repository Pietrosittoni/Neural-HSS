import torch
import torch.nn as nn
import numpy as np

class Embed(torch.nn.Module):
    def __init__(self,input_size: int,in_channels:int,out_channels : int, hidden_size: int,compressed = True):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.compressed = compressed
        if not self.compressed:
            self.mixing_tensor = torch.nn.Parameter( torch.empty( self.out_channels, self.hidden_size,self.hidden_size,self.input_size,self.input_size,self.in_channels ) )
        else:
            self.rank = min([self.hidden_size,self.input_size])
            self.mixing_tensors = torch.nn.ParameterList(
                                                                [
                                                                torch.nn.Parameter(torch.empty(self.rank)), 
                                                                 torch.nn.Parameter(torch.empty(self.out_channels,self.rank)),
                                                                torch.nn.Parameter(torch.empty(self.in_channels,self.rank)),
                                                                 torch.nn.Parameter(torch.empty(self.hidden_size,self.rank)),
                                                                 torch.nn.Parameter(torch.empty(self.hidden_size,self.rank)),
                                                                torch.nn.Parameter(torch.empty(self.input_size,self.rank)),
                                                                torch.nn.Parameter(torch.empty(self.input_size,self.rank)),
                                                              ]
                                                                )
                                                        
        self.reset_parameter()

    @torch.no_grad()
    def reset_parameter(self):
        for p in self.parameters():
            if len(p.shape)>=2:
                nn.init.orthogonal_(p.data)
            else:
                nn.init.uniform_(p.data)

    def forward(self,x):
        if not self.compressed:
            out = x
            if len(x.shape)<4:
                out = torch.unsqueeze(x,1)
            out =  torch.einsum( 'bcxy,ohkxyc->bohk',out,self.mixing_tensor )
            if self.out_channels == 1:
                out = torch.squeeze(out)
            return out
        else:
            return self.forward_compressed(x)
    
    def forward_compressed(self,x):
        out = x
        if len(x.shape)<4:
            out = torch.unsqueeze(x,1)
        out =  torch.einsum( 'bcxy,r,or,cr,hr,kr,xr,yr->bohk',out,*(self.mixing_tensors) )
        if self.out_channels == 1:
            out = torch.squeeze(out)
        return out

class DeepONet2D(nn.Module):
    def __init__(self, n, branch_hidden_dim, trunk_hidden_dim, embedding_dim):
        """
        n: number of discretization points per axis (total grid: n x n)
        """
        super().__init__()
        self.n = n
        # 2D grid points (excluding endpoint for consistency)
        x_grid = np.linspace(0, 1, n, endpoint=False)
        xx, yy = np.meshgrid(x_grid, x_grid, indexing='ij')
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (n*n, 2)
        self.register_buffer('xy_grid', torch.tensor(coords, dtype=torch.float32))  # (n*n, 2)

        # Branch net (takes discretized f as input, flattened)
        self.branch = nn.Sequential(
            nn.Linear(n * n, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, embedding_dim)
        )

        # Trunk net (takes 2D coordinates as input)
        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, embedding_dim)
        )

    def forward(self, f):
        # f: (batch, n, n)
        batch_size = f.shape[0]
        n = self.n

        # Flatten input for branch net
        f_flat = f.view(batch_size, n * n)  # (batch, n*n)

        # Get embeddings from branch and trunk nets
        branch_out = self.branch(f_flat)         # (batch, embed_dim)
        trunk_out = self.trunk(self.xy_grid)     # (n*n, embed_dim)

        # Outer product and reshape to (batch, n, n)
        out = torch.einsum('be,me->bm', branch_out, trunk_out)  # (batch, n*n)
        out = out.view(batch_size, n, n)  # (batch, n, n)

        return out
    


class DeepONet2D_2Dout(nn.Module):
    def __init__(self, n, branch_hidden_dim, trunk_hidden_dim, embedding_dim):
        """
        n: number of discretization points per axis (total grid: n x n)
        """
        super().__init__()
        self.n = n
        # 2D grid points (excluding endpoint for consistency)
        x_grid = np.linspace(0, 1, n, endpoint=False)
        xx, yy = np.meshgrid(x_grid, x_grid, indexing='ij')
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (n*n, 2)
        self.register_buffer('xy_grid', torch.tensor(coords, dtype=torch.float32))  # (n*n, 2)

        self.embed = torch.nn.Sequential(
            Embed(n,4,1,n),
            nn.GELU(),
        )

        # Branch net (takes discretized f as input, flattened)
        self.branch = nn.Sequential(
            nn.Linear(n * n, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, embedding_dim)
        )

        # Trunk net (takes 2D coordinates as input)
        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, embedding_dim),
            nn.GELU()
        )
        self.lift = Embed(n,1,2,n)

    def forward(self, f):
        # f: (batch, n, n)
        batch_size = f.shape[0]
        n = self.n

        f_flat  = self.embed(f)
        # Flatten input for branch net
        f_flat = f_flat.reshape(batch_size, n * n)  # (batch, n*n)

        # Get embeddings from branch and trunk nets
        branch_out = self.branch(f_flat)         # (batch, embed_dim)
        trunk_out = self.trunk(self.xy_grid)     # (n*n, embed_dim)

        # Outer product and reshape to (batch, n, n)
        out = torch.einsum('be,me->bm', branch_out, trunk_out)  # (batch, n*n)
        out = out.view(batch_size, n, n)  # (batch, n, n)
        out = nn.GELU()(out)
        out = self.lift(out)

        return out
    

def test_2d():
    MODEL = DeepONet2D_2Dout(n=128, branch_hidden_dim=64,trunk_hidden_dim=128,embedding_dim=128)
    X = torch.randn(10, 4, 128, 128)  # batch of 2, 4x4 input
    print(f'test forward {MODEL(X).shape}')

# test_2d()