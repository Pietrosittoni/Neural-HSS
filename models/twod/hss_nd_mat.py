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



######## ND version

class HSSNodeTensor(nn.Module):
    """
    General N-mode HSS node: represents a linear map from tensor of shape in_dims to out_dims.
    Recursively splits the first mode of both input and output.
    """
    def __init__(self, in_dims, out_dims, levels, rank):
        super().__init__()
        self.in_dims = list(in_dims)
        self.out_dims = list(out_dims)
        self.levels = levels
        self.rank = rank
        self.in_size = reduce(operator.mul, self.in_dims, 1)
        self.out_size = reduce(operator.mul, self.out_dims, 1)

        if levels == 0:
            # leaf: full dense block
            self.is_leaf = True
            self.W = nn.Parameter(torch.randn(self.out_size, self.in_size))
        else:
            self.is_leaf = False
            # split first mode of input and output
            i = self.in_dims[0]
            o = self.out_dims[0]
            i1, i2 = i // 2, i - i // 2
            o1, o2 = o // 2, o - o // 2
            # build dims lists for children
            in1 = [i1] + self.in_dims[1:]
            in2 = [i2] + self.in_dims[1:]
            out1 = [o1] + self.out_dims[1:]
            out2 = [o2] + self.out_dims[1:]
            # children nodes for diagonal blocks
            self.child11 = HSSNodeTensor(in1, out1, levels-1, rank)
            self.child22 = HSSNodeTensor(in2, out2, levels-1, rank)
            # sizes of flattened subspaces
            in1_size = reduce(operator.mul, in1, 1)
            in2_size = reduce(operator.mul, in2, 1)
            out1_size = reduce(operator.mul, out1, 1)
            out2_size = reduce(operator.mul, out2, 1)
            # low-rank factors for off-diagonals
            # top-right block mapping in2 -> out1
            self.U12 = nn.Parameter(torch.randn(out1_size, rank))  # (out1, r)
            self.V12 = nn.Parameter(torch.randn(in2_size, rank))   # (in2, r)
            # bottom-left block mapping in1 -> out2
            self.U21 = nn.Parameter(torch.randn(out2_size, rank))  # (out2, r)
            self.V21 = nn.Parameter(torch.randn(in1_size, rank))   # (in1, r)
            # store split dims
            self.i1, self.i2, self.o1, self.o2 = i1, i2, o1, o2

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for p in self.parameters():
            if len(p.shape)==2:
                torch.nn.init.normal_(p.data,std = 0.0001)

    def forward(self, x):
        """
        x: tensor of shape (..., *in_dims)
        returns y: tensor of shape (..., *out_dims)
        """
        # flatten input to (..., in_size)
        orig_shape = x.shape[:-len(self.in_dims)]
        v = x.reshape(*orig_shape, self.in_size)
        batched = v.dim() > 1
        if self.is_leaf:
            # apply full weight: (..., out_size) = (..., in_size) @ (in_size, out_size).T
            y = torch.matmul(v, self.W.t())
        else:
            # split along last dim
            v1, v2 = torch.split(v, [reduce(operator.mul, [self.i1]+self.in_dims[1:],1),
                                     reduce(operator.mul, [self.i2]+self.in_dims[1:],1)], dim=-1)
            # diag block applications
            y1 = self.child11(v1.reshape(*orig_shape, *([self.i1] + self.in_dims[1:])))
            y2 = self.child22(v2.reshape(*orig_shape, *([self.i2] + self.in_dims[1:])))
            # flatten children outputs
            y1_flat = y1.reshape(*orig_shape, reduce(operator.mul, [self.o1]+self.out_dims[1:],1))
            y2_flat = y2.reshape(*orig_shape, reduce(operator.mul, [self.o2]+self.out_dims[1:],1))
            # low-rank contributions
            a12 = torch.matmul(v2, self.V12)  # (..., rank)
            a21 = torch.matmul(v1, self.V21)
            y1_flat = y1_flat + torch.matmul(a12, self.U12.t())
            y2_flat = y2_flat + torch.matmul(a21, self.U21.t())
            # concat flats
            y = torch.cat([y1_flat, y2_flat], dim=-1)
        # reshape to output dims
        return y.reshape(*orig_shape, *self.out_dims)

class HSS_ND_Linear(nn.Module):
    """
    Hierarchically Semi-Separable operator on N-mode tensors.
    in_dims, out_dims: tuples of length N
    levels: recursion depth
    rank: off-diagonal rank
    """
    def __init__(self, in_dims, out_dims, levels, rank):
        super().__init__()
        assert len(in_dims) == len(out_dims), "Input and output dims must match in length"
        self.node = HSSNodeTensor(in_dims, out_dims, levels, rank)
        self.in_dims = in_dims
        self.out_dims = out_dims

    def forward(self, X):
        """
        X: tensor of shape (..., *in_dims)
        returns Y: tensor of shape (..., *out_dims)
        """
        assert X.shape[-len(self.in_dims):] == tuple(self.in_dims), \
            f"Expected input shape (*, {self.in_dims}), got {X.shape}"
        return self.node(X)


# class HSS_MLP2D(torch.nn.Module):
#     """
#     HSS MLP with multiple layers of HSS matrices.
#     """
#     def __init__(self, input_size: int, hidden_sizes: list, output_size: int, levels: int, rank: int):
#         super().__init__()
#         self.layers = nn.Sequential()
#         # Input layer
#         self.layers.append(HSS_ND_Linear([input_size]*2,[hidden_sizes[0]]*2, levels, rank))
#         self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[0]))
#         self.layers.append(nn.LayerNorm(hidden_sizes[0]))
#         # Hidden layers
#         for i,_ in enumerate(hidden_sizes[0:len(hidden_sizes)-1]):
#             self.layers.append(HSS_ND_Linear([hidden_sizes[i]]*2,[hidden_sizes[i+1]]*2, levels, rank))
#             self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[i+1]))
#             self.layers.append(nn.LayerNorm(hidden_sizes[i+1],))
#         # Output layer
#         self.layers.append(HSS_ND_Linear([hidden_sizes[-1]]*2,[output_size]*2, levels, rank))
#         self.output_size = output_size

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layers(x)


class HSS2D_outer_product(nn.Module):
    """
    HSS outer product layer: computes outer product of two vectors.
    """
    def __init__(self, in_size: int, out_size: int, levels: int, rank: int,outer_rank :int ):
        super().__init__()
        self.outer_rank = outer_rank
        self.hss = nn.ModuleList([nn.ModuleList([HSSNode(out_size, in_size, levels, rank)]*2) for _ in range(outer_rank)])
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
        y = torch.zeros((x.shape[0],self.out_size,self.out_size),device=x.device)
        for h in self.hss:
            y_1 = rearrange(x,'b n m -> (b m) n')
            y_1 = h[0](y_1)
            y_2 = rearrange(y_1,'(b m) n -> (b n) m',b=x.shape[0],m=x.shape[2],n = self.out_size)
            y_2 = h[1](y_2)
            y += rearrange(y_2,'(b n) m -> b n m',b=x.shape[0],n=self.out_size)
        # reshape to output shape
        return y + self.bias.view(1, -1, 1)  # add bias to each output vector
    

class HSS_MLP2D(torch.nn.Module):
    """
    HSS MLP with multiple layers of HSS matrices.
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, levels: int, rank: int,outer_rank :int ):
        super().__init__()
        self.layers = nn.Sequential()
        # Input layer
        self.layers.append(HSS2D_outer_product(input_size,hidden_sizes[0], levels, rank,outer_rank=outer_rank))
        self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[0]))
        # Hidden layers
        for i,_ in enumerate(hidden_sizes[0:len(hidden_sizes)-1]):
            self.layers.append(HSS2D_outer_product(hidden_sizes[i],hidden_sizes[i+1], levels, rank,outer_rank=outer_rank))
            self.layers.append(TrainableLeakyReLU(in_features=hidden_sizes[i+1]))
        # Output layer
        self.layers.append(HSS2D_outer_product(hidden_sizes[-1],output_size, levels, rank,outer_rank=outer_rank))
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

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
            self.rank = int(min([self.hidden_size,self.input_size])*0.5)
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
                # nn.init.orthogonal_(p.data)
                nn.init.kaiming_normal_(p.data)
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


class HSS_MLP2D_2Dout(torch.nn.Module):
    """
    HSS MLP with multiple layers of HSS matrices.
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,in_channels:int,out_channels, levels: int, rank: int,outer_rank :int,compressed = True ):
        super().__init__()
        self.layers = nn.Sequential()
        h_channels = 6
        # self.layers.extend([nn.Conv2d(in_channels, h_channels, kernel_size=1, bias=True),TrainableLeakyReLU(in_features = input_size)])
        # self.layers.extend([Embed( input_size,h_channels,1,input_size,compressed=compressed),TrainableLeakyReLU(in_features = input_size)])
        self.project = torch.nn.Sequential(*[
            nn.Conv2d(in_channels, h_channels, kernel_size=1, bias=True),torch.nn.GELU(), #TrainableLeakyReLU(in_features = input_size),
            Embed( input_size,h_channels,1,input_size,compressed=compressed),torch.nn.GELU() #TrainableLeakyReLU(in_features = input_size)
        ])
        # Input layer
        self.layers.append(HSS2D_outer_product(input_size,hidden_sizes[0], levels, rank,outer_rank=outer_rank))
        self.layers.append(torch.nn.GELU())#TrainableLeakyReLU(in_features=hidden_sizes[0]))
        # self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        # Hidden layers
        for i,_ in enumerate(hidden_sizes[0:len(hidden_sizes)-1]):
            self.layers.append(HSS2D_outer_product(hidden_sizes[i],hidden_sizes[i+1], levels, rank,outer_rank=outer_rank))
            self.layers.append(torch.nn.GELU())#TrainableLeakyReLU(in_features=hidden_sizes[i+1]))
            # self.layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
        # Output layer
        self.layers.extend([HSS2D_outer_product(hidden_sizes[-1],output_size, levels, rank,outer_rank=outer_rank),torch.nn.GELU()])#TrainableLeakyReLU(in_features=output_size)])
        # self.layers.append(Embed( output_size,1,out_channels,output_size,compressed=compressed))
        self.lift = Embed( output_size,1,out_channels,output_size,compressed=compressed)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.layers(x)
        out = self.project(x)
        # print("Input shape:", out.shape)
        # for l in self.layers:
        #         out = l(out) if isinstance(l,TrainableLeakyReLU) else l(out)+out
        out = self.layers(out)
        out = self.lift(out)
        return out


def test():
    # import matplotlib.pyplot as plt
    from utils import get_flops, count_params
    # MODEL = HSS2D_outer_product(   in_size=64, out_size=64, levels=2, rank=2, outer_rank=10)
    MODEL = HSS_MLP2D(input_size=64, hidden_sizes=[64, 64], output_size=64, levels=2, rank=2, outer_rank=5)
    X = torch.randn(10, 64, 64)  # batch of 2, 4x4 input
    print("Model params:", count_params(MODEL))
    print(f'test forward {MODEL(X).shape}')
    print(f'flops forward: {get_flops(MODEL, X)}')


def test_2d():
    # MODEL = HSS_MLP2D_2Dout(input_size=128, hidden_sizes=[64, 64],in_channels = 4,out_channels=2, output_size=128, levels=2, rank=2, outer_rank=5,compressed = True)
    from utils import get_flops, count_params
    MODEL = Embed(128,4,1,128)
    X = torch.randn(10, 4, 128, 128)  # batch of 2, 4x4 input
    print("Model params:", count_params(MODEL))
    print(f'test forward {MODEL(X).shape}')
    print(f'flops forward: {get_flops(MODEL, X)}')

# test()
# test_2d()

