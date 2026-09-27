import torch.nn as nn
import torch


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
    

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias)
        
        self.activation = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv2(h)
        h = h + self.shortcut(x)
        h = self.activation(h)
        return h

class ResNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3, depth=4, bias=True):
        super().__init__()

        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias)

        layers = []
        for _ in range(depth):
            layers.append(ResidualBlock2D(hidden_channels, hidden_channels, kernel_size=kernel_size, bias=bias))
        self.layers = nn.ModuleList(layers)

        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        if self.in_channels == 1:
            x = x.unsqueeze(1)
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)
        if self.out_channels == 1:
            x = x.squeeze(1)
        return x

    


# def test_2d():
#     MODEL = ResNet2D(4,2,10)
#     X = torch.randn(10, 4, 128, 128)  # batch of 2, 4x4 input
#     print(f'test forward {MODEL(X).shape}')
# test()

# test_2d()