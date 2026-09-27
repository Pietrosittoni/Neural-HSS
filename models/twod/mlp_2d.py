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



class MLP2D(nn.Module):
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
    
    