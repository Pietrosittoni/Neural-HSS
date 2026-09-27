import torch 

class Trainable_Leaky_ReLU_ne(torch.nn.Module):

    def __init__(self,in_features, init_slope=1.):
        super(Trainable_Leaky_ReLU_ne, self).__init__()
        # Initialize the slope as a trainable parameter.
        self.in_features = in_features
        self.slope = torch.nn.Parameter(torch.tensor([init_slope]*self.in_features, dtype=torch.float),requires_grad=True)

    def forward(self, x):
        # Use torch.where to apply the leaky operation without using built-in LeakyReLU.
        return torch.where(x >= 0, x, self.slope * x)
    

class TrainableLeakyReLU(torch.nn.Module):
    def __init__(self, in_features = None , init_slope=1.):
        super(TrainableLeakyReLU, self).__init__()
        # Initialize the slope as a trainable parameter.
        self.slope = torch.nn.Parameter(torch.tensor(init_slope, dtype=torch.float),requires_grad=True)

    def forward(self, x):
        # Use torch.where to apply the leaky operation without using built-in LeakyReLU.
        return torch.where(x >= 0, x, self.slope * x)