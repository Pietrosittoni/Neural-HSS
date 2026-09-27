import torch
from torch import nn

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
                nn.init.kaiming_uniform_(p.data)
            else:
                nn.init.uniform_(p.data)

    def forward(self,x):
        if not self.compressed:
            out = x
            if len(x.shape)<4:
                out = torch.unsqueeze(x,1)
            out =  torch.einsum( 'bxyc,ohkxyc->bhko',out,self.mixing_tensor )
            if self.out_channels == 1:
                out = torch.squeeze(out)
            return out
        else:
            return self.forward_compressed(x)
    
    def forward_compressed(self,x):
        out = x
        if len(x.shape)<4:
            out = torch.unsqueeze(x,1)
        out =  torch.einsum( 'bxyc,r,or,cr,hr,kr,xr,yr->bhko',out,*(self.mixing_tensors) )
        if self.out_channels == 1:
            out = torch.squeeze(out)
        return out

class SpectralConv2d(nn.Module):
    '''2D FFT, linear transform, and Inverse FFT
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes (int): Number of Fourier modes
    '''
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        std = torch.sqrt(torch.tensor(2.0 / in_channels, dtype=torch.float32))
        self.weights = nn.Parameter(
            torch.empty(in_channels, out_channels, modes, modes, 2, dtype=torch.float32).normal_(0, std)
        )
        def get_flops_fft_ifft2(n):
            return 2*in_channels * n * torch.log2(torch.tensor(n))
        self.get_flops_fft_ifft = get_flops_fft_ifft2
        self.flops = None

    def batchmul2d(self, input, weights):
        ## (batch, in_channel, x, y), (in_channel, out_channel, modes_x, modes_y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy, ioyx -> boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.fft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(2),
            x.size(3),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :self.modes, :self.modes] = self.batchmul2d(x_ft[:, :, :self.modes, :self.modes], torch.view_as_complex(self.weights))

        # Return to physical space
        x = torch.fft.ifft2(out_ft)
        if self.flops is None:
            self.flops = self.get_flops_fft_ifft(x.size(-1)*x.size(-2))
        return x.real

class FourierLayer2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, bias: bool=True):
        super().__init__()

        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.activation = nn.GELU()
        self.flops = None

    def forward(self, x):
        x1 = self.spectral_conv(x)
        if self.flops is None:
            self.flops = self.spectral_conv.flops
        x2 = self.pointwise_conv(x)
        x = x1 + x2
        x = self.activation(x)
        return x

class FNO2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, modes: int, depth: int=4, bias: bool=True):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias)

        layers = []
        for _ in range(depth):
            layers.append(FourierLayer2D(hidden_channels, hidden_channels, modes, bias=bias))
        self.layers = nn.ModuleList(layers)

        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extra_flops = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extra_flops = 0
        if self.in_channels == 1:
            x = x.unsqueeze(1)
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)
            if hasattr(layer, 'flops') and layer.flops is not None:
                self.extra_flops += layer.flops
            
        x = self.proj(x)
        if self.out_channels == 1:
            x = x.squeeze(1)
        return x

    
def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def test():
    import matplotlib.pyplot as plt
    n_params = []
    for n in range(1,5):
        model = FNO2D(1,1,64,16,depth = n)
        n_params.append(count_params(model))

    plt.plot(list(range(1,5)),n_params)
    plt.xlabel('depth')
    plt.ylabel('number of parameters')
    plt.title('HSS MLP2D')
    plt.savefig('fno2d_params.pdf', dpi=300)
    plt.show()


def test_2d():
    MODEL = FNO2D(4,2,10,4)
    X = torch.randn(10, 4, 128, 128)  # batch of 2, 4x4 input
    print(f'test forward {MODEL(X).shape}')
# test()

# test_2d()