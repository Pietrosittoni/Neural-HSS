from .oned.hss_mat import HSS_MLP, HSS_Linear
from .oned.mlp import MLP
from .oned.resnet import ResNet1D
from .oned.fno_1d import FNO1D
from .oned.hss import OverallNet 
from .oned.greenlearning import GreenLearning
from .oned.deeponet import DeepONet1D
from .twod.fno_2d import FNO2D
from .twod.resnet_2d import ResNet2D
from .twod.greenlearning_2d import GreenLearning2D
from .twod.deeponet_2d import DeepONet2D,DeepONet2D_2Dout
from .twod.hss_nd_mat import HSS_MLP2D,HSS_MLP2D_2Dout
from .twod.mlp_2d import MLP2D
from .threed.hss_nd_mat import HSS_MLP3D
from .threed.fno3d import FNO3d
from .threed.resnet3d import ResNet3D
from .threed.greenlearning3d import GreenLearning3D
from .threed.deeponet3d import DeepONet3D

# Dictionary mapping model names to their classes
MODEL_REGISTRY = {
    'hss_mlp': HSS_MLP,
    'hss_linear': HSS_Linear,
    'hss': OverallNet, 
    'mlp': MLP,
    'resnet': ResNet1D,
    'greenlearning': GreenLearning,
    'fno_1d': FNO1D,
    'deeponet': DeepONet1D,
    'fno_2d': FNO2D,
    'resnet_2d': ResNet2D,
    'greenlearning_2d': GreenLearning2D,
    'deeponet_2d': DeepONet2D,
    'hss_mlp_2d': HSS_MLP2D,
    'mlp_2d':MLP2D,
    'hss_mlp_2d_2dout':HSS_MLP2D_2Dout,
    'deeponet_2d_2dout':DeepONet2D_2Dout,
    'hss_mlp_3d':HSS_MLP3D,
    'fno_3d':FNO3d,
    'resnet_3d':ResNet3D,
    'greenlearning_3d':GreenLearning3D,
    'deeponet_3d':DeepONet3D
}

__all__ = ['MODEL_REGISTRY'] 