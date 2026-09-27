import torch
from torch import nn

class SpectralConv1d(nn.Module):
    '''FFT, linear transform, and Inverse FFT
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes (int): Number of Fourier modes
    [paper](https://arxiv.org/abs/2010.08895)
    '''
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        std = torch.sqrt(torch.tensor(2.0 / in_channels, dtype=torch.float32))
        self.weights = nn.Parameter(
            torch.empty(in_channels, out_channels, modes, 2, dtype=torch.float32).normal_(0, std)
        )



    def batchmul1d(self, input, weights):
        ## (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix, iox -> box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes] = self.batchmul1d(x_ft[:, :, : self.modes], torch.view_as_complex(self.weights))

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
class FourierLayer1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, bias: bool=True):
        super().__init__()

        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.activation = nn.GELU()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        x = x1 + x2
        x = self.activation(x)
        return x

class FNO1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, modes: int, depth: int=4, bias: bool=True):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=bias)

        layers = []
        for _ in range(depth):
            layers.append(FourierLayer1D(hidden_channels, hidden_channels, modes, bias=bias))
        self.layers = nn.ModuleList(layers)

        self.proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.transpose(-2,-1)
        x = x.unsqueeze(1)
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)
            
        x = self.proj(x)
        #x = x.transpose(-2,-1)
        x = x.squeeze(1)
        return x