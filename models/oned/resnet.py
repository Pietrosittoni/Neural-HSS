import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1)/2) , padding_mode='circular', bias=bias)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1)/2) , padding_mode='circular', bias=bias)
        
        self.activation = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv2(h)
        h = h + self.shortcut(x)
        h = self.activation(h)
        return h

class ResNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3, depth=4, bias=True):
        super().__init__()

        self.lift = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=bias)

        layers = []
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_channels, hidden_channels, kernel_size=kernel_size, bias=bias))
        self.layers = nn.ModuleList(layers)

        self.proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        #x = x.transpose(-2,-1)
        x = x.unsqueeze(1)
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)
        #x = x.transpose(-2,-1)
        x = x.squeeze(1)

        return x