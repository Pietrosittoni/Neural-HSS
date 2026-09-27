import torch
import torch.nn as nn
from utils.modules import TrainableLeakyReLU
import torch.multiprocessing as mp
from opt_einsum import contract
from functorch import vmap



# Speeduped version of GroupedLinear 1.4 x faster
class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias=False):
        """
        Splits the input (of size in_features) into num_groups groups,
        each of size in_features/num_groups, and applies an independent
        linear map to produce outputs of total size out_features.
        """
        super().__init__()
        assert in_features % num_groups == 0, "in_features must be divisible by num_groups"
        assert out_features % num_groups == 0, "out_features must be divisible by num_groups"
        self.num_groups = num_groups
        self.in_per_group = in_features // num_groups
        self.out_per_group = out_features // num_groups
        
        self.linears = nn.ModuleList([
            nn.Linear(self.in_per_group, self.out_per_group, bias=bias)
            for _ in range(num_groups)
        ])
        self.weight  = torch.nn.Parameter(torch.stack([lin.weight for lin in self.linears], dim=0))  # Shape: (num_groups, out_per_group, in_per_group)
        del self.linears
        if bias:
            self.bias = torch.nn.Parameter(torch.stack([lin.bias for lin in self.linears], dim=0))  # Shape: (num_groups, out_per_group)
        else:
            self.bias = None
    def forward(self, x):
        # x has shape (batch_size, N * in_per_group)
        batch_size = x.shape[0]
        # reshape to (batch_size, N, in_per_group)
        x_grouped = x.view(batch_size, self.num_groups, self.in_per_group)
        # perform batched matrix multiplication:
        # weight is (N, out_features, in_per_group)
        # result: (batch_size, N, out_features)
        y_grouped = torch.einsum('bni,noi->bno', x_grouped, self.weight)
        # add bias: bias is (N, out_features)
        if self.bias is not None:
            y_grouped += self.bias.unsqueeze(0)  # broadcasting bias to match batch size
        # Finally, reshape back to (batch_size, N*out_features)
        y_grouped = y_grouped.reshape(batch_size, -1)
        return y_grouped

        
class SubNet(nn.Module):
    def __init__(self, i, M, t, r,act = TrainableLeakyReLU):
        """
        i: index of this sub-network (1 <= i <= M+1)
        M: such that the input has (2^M)*t units
        t, r: parameters (independent) chosen so that (2^M)*t and (2^M)*(2r) match the data dimensions
        """
        super().__init__()
        layers = []
        n_hidden = M - i + 1  # number of hidden layers
        if n_hidden > 0:
            # Input to first hidden layer
            in_dim = (2**M) * t
            hid1 = (2**i) * r
            # Group the first hidden layer into groups of 2*r units.
            num_groups = 2**(i - 1)
            layers.append(GroupedLinear(in_dim, hid1, num_groups))
            # layers.append(nn.LeakyReLU(negative_slope=1/2))
            layers.append(act(hid1))
            # Hidden-to-hidden connections
            for j in range(1, n_hidden):
                in_dim = (2**(i+j-1)) * r
                out_dim = (2**(i+j)) * r
                num_groups = 2**(i+j-1)  # source is split into groups of r units
                layers.append(GroupedLinear(in_dim, out_dim, num_groups))
                # layers.append(nn.LeakyReLU(negative_slope=1/2))
                layers.append(act(out_dim))
            
            # Last hidden layer to output
            in_dim = (2**M) * r
            out_dim = (2**M) * (2 * r)  # final output dimension of this sub-network
            num_groups = 2**M  # source groups of r and target groups of 2r
            layers.append(GroupedLinear(in_dim, out_dim, num_groups))
        else:
            # For i = M+1: Direct mapping from input to output.
            in_dim = (2**M) * t
            out_dim = (2**M) * (2 * r)
            num_groups = 2**M  # group output into groups of 2r
            layers.append(GroupedLinear(in_dim, out_dim, num_groups))
        
        self.net = nn.Sequential(*layers)
        # self.reset_parameters()

    def reset_parameters(self):
        for mod in self.net:
            if isinstance(mod,GroupedLinear):
                for p in mod.parameters():
                    if len(p.shape)==2:
                        torch.nn.init.kaiming_uniform_(p.data,a = 1.)
        
    def forward(self, x):
        return self.net(x)

class OverallNet(nn.Module):
    def __init__(self, M, t, r, act  = TrainableLeakyReLU):
        """
        Constructs the overall network so that:
          - Input dimension = (2^M)*t 
          - Output dimension = (2^M)*(2r)
        """
        super().__init__()
        self.subnets = nn.ModuleList([SubNet(i, M, t, r, act = act) for i in range(1, M + 2)])

    def forward(self, x):
        outputs = torch.stack([subnet(x.clone()) for subnet in self.subnets], dim=0)  # Shape: [num_subnets, ...]    #### 2X speedup forward
        return outputs.sum(dim=0)
    
class OverallNet_deep(nn.Module):
    def __init__(self, depth, M, t, r, act  = TrainableLeakyReLU):
        """
        Constructs the overall network so that:
          - Input dimension = (2^M)*t 
          - Output dimension = (2^M)*(2r)
        """
        super().__init__()
        self.depth = depth
        self.overall_nets = nn.ModuleList([OverallNet(M, t, r, act = act) for _ in range(depth)])


    def forward(self, x):
        for overall_net in self.overall_nets:
            x = overall_net(x)
        return x

def apply_f_along_axis(X, axis, f):
    """
    Permutes X so that the given axis is moved to the last dimension.
    Then, flattens all other dimensions into a batch dimension,
    applies f, and reshapes the output back.
    
    Parameters:
      X (Tensor): Input tensor of arbitrary shape.
      axis (int): The axis to treat as the "feature" dimension.
      f (callable): A function or model that expects input shape (batch, features).
      
    Returns:
      Tensor: The output, reshaped to match the original indices (except for the feature dimension,
      which now becomes the output feature dimension of f).
    """
    # 1. Create the permutation:
    # All dimensions except the target axis, followed by the axis itself.
    permute_order = [i for i in range(X.dim()) if i != axis] + [axis]
    X_perm = X.permute(*permute_order)
    
    # 2. Flatten all dimensions except the last one into one batch dimension.
    # new_feature_dim is the size of the axis we care about.
    new_feature_dim = X_perm.size(-1)
    batch_size = X_perm.numel() // new_feature_dim  # total number of samples when flattened
    X_flat = X_perm.reshape(batch_size, new_feature_dim)
    
    # 3. Apply the model/function f.
    output_flat = f(X_flat)
    
    # 4. Reshape back.
    # First, get the output feature size (could be different from new_feature_dim).
    output_feature_dim = output_flat.size(-1)
    
    # The new shape for the batch dimensions is the original X_perm shape except the last dim.
    batch_shape = X_perm.shape[:-1]
    # Append the new output feature dimension.
    new_shape = batch_shape + (output_feature_dim,)
    
    output = output_flat.reshape(new_shape)
    return output

class HSS_approximate_Nd(nn.Module):
    def __init__(self,spatial_dim,outer_rank, M, t, r, act = TrainableLeakyReLU):
        """
        Constructs the overall network so that:
          - Input dimension = (2^M)*t 
          - Output dimension = (2^M)*(2r)
        """
        super().__init__()
        self.outer_rank = outer_rank
        self.onedim_HSS = [nn.ModuleList([OverallNet(M, t, r, act = act) for _ in range(spatial_dim)]) for _ in range(self.outer_rank)]
        self.spatial_dim = spatial_dim

    def forward(self, x):
        
        ### x is of shape (batch_size, n_1, n_2, ..., n_spatial_dim)
        ### We need to apply the same HSS model to each spatial dimension.
        out = 0
        for one_d_HSS in self.onedim_HSS:
            
            # Apply the HSS model to each spatial dimension
            # einsum('b...i->bi...', x) will permute the dimensions of x
            # and apply the HSS model to each spatial dimension.
            partial_out = x.clone()

            for d in range(1,self.spatial_dim):
                partial_out = apply_f_along_axis(partial_out,d,one_d_HSS[d])
            out += partial_out

        return out


def test_nd():
    device = torch.device('cuda:0')
    total_cols = 128*2
    half = total_cols // 2  # both input and target dimensions
    M = 5
    t = half // (2**M)      # Must be an integer
    r = half // (2**(M+1))    # Must be an integer
    X = torch.randn((20,128,128,128))
    model = HSS_approximate_Nd(3, 2,M, t, r, TrainableLeakyReLU).to(device)

    print(model(X).shape)
        

#### In progress to Speedup even more
# from functorch import make_functional_with_buffers
# import torch.nn as nn
# from functorch import combine_state_for_ensemble

# class HSS_ensembled(nn.Module):
#     def __init__(self, Overall_net):
#         super().__init__()
#         # Create a ModuleList of subnets. They all have the same architecture.
#         self.subnets = Overall_net.subnets
        
#         # Convert each subnet into a functional version
#         # self.fmodels, params, buffers = zip(*[make_functional_with_buffers(subnet) for subnet in self.subnets])
#         # Store the parameters and buffers for later use.
#         # All subnets have the same architecture so we can use the first functional model for all.
#         self.f_model,self.ensemble_params,buffers = combine_state_for_ensemble(self.subnets)
#         # Assume buffers are identical across subnets, take the first.
#         self.ensemble_buffers = buffers[0]

#     def forward(self, x):
#         # The forward pass will be handled with vmap over the batched parameters.
#         # We assume each subnet's forward is the same, so we select the first functional model.
#         outputs = vmap(lambda params: self.fmodels[0](params, self.ensemble_buffers, x))(
#             self.ensemble_params
#         )
#         # For example, summing the outputs along the ensemble dimension
#         return outputs.sum(dim=0)

# # Helper to combine parameter trees (this method is provided by functorch)
