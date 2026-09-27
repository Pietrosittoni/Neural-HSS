import torch
import torch.nn as nn
from functools import reduce
import operator


class TrainableLeakyReLU(torch.nn.Module):
    def __init__(self, in_features = None , init_slope=1.):
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
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
                    if len(p.shape)==2:
                        # torch.nn.init.kaiming_uniform_(p.data,a = 1.)
                        torch.nn.init.normal_(p.data,std = 1e-2)

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

class HSS_Linear(nn.Module):
    """
    Rectangular HSS matrix with shape (rows, cols), depth, and rank.
    """
    def __init__(self, in_shape: int,out_shape: int, levels: int, rank: int):
        super().__init__()
        # require dims divisible by 2^levels
        assert out_shape % (2**levels) == 0, "rows must be divisible by 2^levels"
        assert in_shape % (2**levels) == 0, "cols must be divisible by 2^levels"
        self.root = HSSNode(out_shape, in_shape, levels, rank)
        self.out_shape = out_shape
        self.cols = in_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform mat-vec: x shape (..., cols), returns (..., rows).
        """
        assert x.shape[-1] == self.cols, f"Expected last dim {self.cols}, got {x.shape[-1]}"
        return self.root(x)
    


class HSS_MLP(torch.nn.Module):
    """
    HSS MLP with multiple layers of HSS matrices.
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, levels: int, rank: int):
        super().__init__()
        self.layers = nn.Sequential()
        # Input layer
        self.layers.append(HSS_Linear(input_size,hidden_sizes[0], levels, rank))
        self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[0]))
        # Hidden layers
        for i,_ in enumerate(hidden_sizes[0:len(hidden_sizes)-1]):
            self.layers.append(HSS_Linear(hidden_sizes[i],hidden_sizes[i+1], levels, rank))
            self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[i+1]))
        # Output layer
        self.layers.append(HSS_Linear(hidden_sizes[-1],output_size, levels, rank))
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)