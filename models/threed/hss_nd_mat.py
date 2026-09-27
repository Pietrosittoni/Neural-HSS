import torch
import torch.nn as nn
from functools import reduce
import operator,math
from einops import rearrange
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TrainableLeakyReLU(torch.nn.Module):
    def __init__(self, in_features = None , init_slope=0.):
        super(TrainableLeakyReLU, self).__init__()
        # Initialize the slope as a trainable parameter.
        self.slope = torch.nn.Parameter(torch.tensor(init_slope, dtype=torch.float),requires_grad=True)

    def forward(self, x):
        # Use torch.where to apply the leaky operation without using built-in LeakyReLU.
        return torch.where(x >= 0, x, self.slope * x)
    
#########################

class HSSNode(nn.Module):
    """
    Hierarchically Semi-Separable node (rectangular) with nn.Parameters for low-rank factors
    or full block at leaf.
    Splits both row and column dimensions at each level.
    """
    def __init__(self, rows: int, cols: int, levels: int, rank: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        # Leaf: store full matrix parameter
        if levels == 0:
            self.is_leaf = True
            self.full = nn.Parameter(torch.randn(rows, cols))
        else:
            self.is_leaf = False
            # Split dimensions
            r1 = rows // 2
            r2 = rows - r1
            c1 = cols // 2
            c2 = cols - c1
            # Children: top-left and bottom-right diagonal blocks
            self.child1 = HSSNode(r1, c1, levels-1, rank)
            self.child2 = HSSNode(r2, c2, levels-1, rank)
            # Low-rank factors for off-diagonals
            # M12: block rows [0:r1], cols [c1:] of size (r1, c2)
            self.U12 = nn.Parameter(torch.randn(r1, rank))  # (r1, rk)
            self.V12 = nn.Parameter(torch.randn(c2, rank))  # (c2, rk)
            # M21: block rows [r1:], cols [0:c1] of size (r2, c1)
            self.U21 = nn.Parameter(torch.randn(r2, rank))  # (r2, rk)
            self.V21 = nn.Parameter(torch.randn(c1, rank))  # (c1, rk)
            # store splits
            self.r1, self.r2, self.c1, self.c2 = r1, r2, c1, c2
        self.rank = rank
        self.levels = levels
        
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for p in self.parameters():
            if len(p.shape)==2:
                # torch.nn.init.normal_(p.data)
                torch.nn.init.orthogonal_(p.data)
                # torch.nn.init.kaiming_normal_(p.data)
                # p.div_(self.rank*self.levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute y = M * x via recursive HSS mat-vec.
        x: shape (cols,) or (batch, cols)
        Returns y of shape (rows,) or (batch, rows)
        """
        batched = (x.dim() == 2)
        if self.is_leaf:
            # full matrix multiply: (batch, rows, cols) @ (batch, cols, 1)
            if batched:
                return torch.matmul(x, self.full.t())
            else:
                return self.full @ x

        # split input vector by col dimension
        if batched:
            x1, x2 = x[:, :self.c1], x[:, self.c1:]
        else:
            x1, x2 = x[:self.c1], x[self.c1:]

        # diagonal block mat-vecs
        y1 = self.child1(x1)  # shape (r1) or (batch, r1)
        y2 = self.child2(x2)  # shape (r2) or (batch, r2)

        # off-diagonal low-rank contributions
        if batched:
            # x2 @ V12 -> (batch, rk)
            a12 = torch.matmul(x2, self.V12)
            # x1 @ V21 -> (batch, rk)
            a21 = torch.matmul(x1, self.V21)
            # add U12 a12 and U21 a21
            y1 = y1 + torch.matmul(a12, self.U12.t())
            y2 = y2 + torch.matmul(a21, self.U21.t())
        else:
            a12 = self.V12.t() @ x2  # (rk,)
            a21 = self.V21.t() @ x1  # (rk,)
            y1 = y1 + self.U12 @ a12
            y2 = y2 + self.U21 @ a21

        # concatenate by row dimension
        if batched:
            return torch.cat([y1, y2], dim=1)
        else:
            return torch.cat([y1, y2], dim=0)


class HSS3D_outer_product(nn.Module):
    """
    HSS outer product layer: computes outer product of two vectors.
    """
    def __init__(self, in_size: int, out_size: int, levels: int, rank: int,outer_rank :int ):
        super().__init__()
        self.outer_rank = outer_rank
        model_dim = 3
        self.hss = nn.ModuleList([nn.ModuleList([HSSNode(out_size, in_size, levels, rank)]*model_dim) for _ in range(outer_rank)])
        self.bias = nn.Parameter(torch.zeros(out_size), requires_grad=True)
        self.out_size = out_size
        self.in_size = in_size

    def forward(self, x) -> torch.Tensor:
        """
        Computes outer product y = x1 @ x2.T using HSS.
        x1: shape (..., in_size)
        x2: shape (..., in_size)
        Returns y of shape (..., out_size, in_size)
        """
        y = torch.zeros((x.shape[0],self.out_size,self.out_size,self.out_size),device=x.device)
        for h in self.hss:
            y_1 = rearrange(x,'b n m k -> (b n m) k')
            y_1 = h[0](y_1)
            y_2 = rearrange(y_1,'(b n m) k -> (b n k) m',b=x.shape[0],m=x.shape[2],n = x.shape[1], k = self.out_size)
            y_2 = h[1](y_2)
            y_3 = rearrange(y_2,'(b n k) m -> (b m k) n',b=x.shape[0],m=self.out_size,n = x.shape[1],k = self.out_size)
            y_3 = h[1](y_3)
            y += rearrange(y_3,'(b m k) n -> b n m k',b=x.shape[0],n=self.out_size,m = self.out_size,k  =self.out_size)
        # reshape to output shape
        return y + self.bias.view(1, -1, 1)  # add bias to each output vector
    

class HSS_MLP3D(torch.nn.Module):
    """
    HSS MLP with multiple layers of HSS matrices.
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, levels: int, rank: int,outer_rank :int ):
        super().__init__()
        self.layers = nn.Sequential()
        # Input layer
        self.layers.append(HSS3D_outer_product(input_size,hidden_sizes[0], levels, rank,outer_rank=outer_rank))
        self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[0]))
        # Hidden layers
        for i,_ in enumerate(hidden_sizes[0:len(hidden_sizes)-1]):
            self.layers.append(HSS3D_outer_product(hidden_sizes[i],hidden_sizes[i+1], levels, rank,outer_rank=outer_rank))
            self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[i+1]))
        # Output layer
        self.layers.append(HSS3D_outer_product(hidden_sizes[-1],output_size, levels, rank,outer_rank=outer_rank))
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    


def test():
    # import matplotlib.pyplot as plt
    from net_utils import get_flops, count_params
    # MODEL = HSS2D_outer_product(   in_size=64, out_size=64, levels=2, rank=2, outer_rank=10)
    MODEL = HSS_MLP3D(input_size=128, hidden_sizes=[128, 128], output_size=128, levels=2, rank=2, outer_rank=5).to('cuda:0')
    X = torch.randn(32, 128, 128, 128).to('cuda:0')  # batch of 2, 4x4 input
    print("Model params:", count_params(MODEL))
    print(f'test forward {MODEL(X).shape}')
    print(f'flops forward: {get_flops(MODEL, X)}')
    print(f'Memory peak: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB')


#test()