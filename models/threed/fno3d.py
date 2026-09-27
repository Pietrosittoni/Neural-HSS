import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 3d fourier layers

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes, hidden_channels, in_channels, out_channels, hidden_layers):
        super(FNO3d, self).__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.modes3 = modes
        self.width = hidden_channels
        self.layer = hidden_layers

        self.p = nn.Linear(in_channels, self.width)# input channel: the solution of the first n timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.q = nn.Linear(self.width, out_channels)
        self.conv_layers = nn.ModuleList([SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(self.layer)])
        self.mlp_layers = nn.ModuleList([nn.Conv3d(self.width, self.width, kernel_size=1) for _ in range(self.layer)])
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        for i in range(self.layer):
            x1 = self.conv_layers[i](x)
            x2 = self.mlp_layers[i](x)
            x = x1 + x2
            x = self.activation(x)

        x = x.permute(0, 2, 3, 4, 1) 


        x = self.q(x)
        x = x.squeeze(-1)
        return x


#def test():
#    # import matplotlib.pyplot as plt
#    import sys
#    sys.path.append('/home/_/hss_learning')
#    from models.net_utils import get_flops, count_params
#    
#    # Define FNO3d model with appropriate parameters
#    # modes=8, hidden_channels=32, in_channels=1, out_channels=1, depth=4
#    MODEL = FNO3d(modes=16, hidden_channels=32, in_channels=1, out_channels=1, depth=4).to('cuda:1')
#    
#    # Create input tensor with appropriate shape (batch, x, y, z)
#    X = torch.randn(16, 32, 32, 32).to('cuda:1')
#    
#    print("Model params:", count_params(MODEL))
#    print(f'test forward {MODEL(X).shape}')
#    flops, _ = get_flops(MODEL, X)
#    print(f'flops forward: {flops}')
#    print(f'Memory peak: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB')
#
#
#if __name__ == "__main__":
#    test()