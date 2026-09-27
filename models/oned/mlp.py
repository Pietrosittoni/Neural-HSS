import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, depth=4):
        super().__init__()

        self.lift = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(depth)])
        #self.activation = nn.GELU()
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.proj(x)
        x = x.squeeze(-1)
        return x