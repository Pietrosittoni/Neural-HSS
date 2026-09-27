import argparse
import os
import random  # Add random module
import torch
import wandb
import numpy as np
from torch.utils.data import TensorDataset
from models import MODEL_REGISTRY  # Import model registry
from utils.config import Config
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.train_eval import train, valid_steady
from utils.read_steady import read_steady_data
from models.utils import get_flops

def set_seed(seed):
    """Set all seeds for reproducibility"""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # CUDNN
    torch.backends.cudnn.benchmark = False  # CUDNN
    


class RelativeMSE(nn.Module):
    """
    Computes the relative L2 loss:
    Loss = mean_over_batch( ||pred_i - y_i||_2 / ||y_i||_2 )
    
    Where ||.||_2 is the L2 norm (Euclidean norm)
    """
    def __init__(self, epsilon: float = 1e-8):
        super(RelativeMSE, self).__init__()
        self.epsilon = epsilon  # Small constant to avoid division by zero
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate L2 norm of difference for each sample
        # Assuming first dimension is batch dimension
        diff_norm = torch.norm(predictions - targets, p=2, dim=-1)**2
        
        # Calculate L2 norm of targets for each sample
        target_norm = torch.norm(targets, p=2, dim=-1)**2+ self.epsilon
        
        # Calculate relative L2 error for each sample
        relative_l2 = diff_norm / target_norm
        
        # Average over the batch
        return torch.mean(relative_l2)



'''
Main
'''
def main(config: str):
    # Initialize wandb for experiment tracking
    model_ids = {'hss_mlp': 1,
    'hss_linear': 2,
    'hss': 3, 
    'mlp': 4,
    'resnet': 5,
    'greenlearning': 6,
    'fno_1d': 7,
    'deeponet': 8,
    'fno_2d': 9,
    'resnet_2d': 10,
    'greenlearning_2d': 11,
    'deeponet_2d': 12,
    'hss_mlp_2d': 13,
    'mlp_2d':14 }

    setattr(config,'model_id', model_ids[config.model])
    wandb.login()
    wandb.init(
        project=config.project,
        config=config.__dict__,
        tags=config.tags,
        # entity="_",
    )


    # Set random seeds for reproducibility
    set_seed(config.seed)

    # Set up device (GPU/CPU)
    device = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.')

    train_dataloader, valid_dataloader, scale  = read_steady_data(config,device)
    wandb.log({'scale': scale})
    
    # Initialize model based on configuration
    if config.model not in MODEL_REGISTRY:
        raise ValueError(f"Model {config.model} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[config.model]
    model = model_class(
        **{k: v for k, v in config.__dict__.items() if k in model_class.__init__.__code__.co_varnames}
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    flops = get_flops(model, input_size=(1, config.input_dim))
    wandb.log({'number_of_parameters': total_params,'flops': flops})
    print(f"Total number of parameters: {total_params},flops: {flops}")

    # Enable model parameter tracking in wandb
    wandb.watch(model)

    # Initialize loss function (Mean Squared Error) for training
    loss_fn = RelativeMSE()
    loss_fn_val = RelativeMSE()
    
    # Initialize AdamW optimizer with configured learning rate
    # AdamW is a variant of Adam that implements weight decay correction
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    iters = len(train_dataloader) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=config.learning_rate_min)

    # Training loop
    print('Training...')
    for epoch in range(config.epochs):
        print(f'Epoch: {epoch + 1}/{config.epochs}')

        train(train_dataloader, model, loss_fn, optimizer, scheduler,scale)
        
        # Run validation if validation data is provided

        valid_steady(valid_dataloader, model, loss_fn_val,if_plot=False,scale=scale)

    print('Training complete.')

    valid_steady(valid_dataloader, model, loss_fn_val,if_plot=True,scale=scale)


    # Save model and scaler
    model_path = config.folder_model_path + f'/ models/{wandb.run.name}'
    os.makedirs(model_path, exist_ok=True)

    torch.save(model.state_dict(), f'{model_path}/model.pt')

    print(f'Model saved as {model_path}/model.pt.')

    # Clean up wandb
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--device', type=int, help='Override GPU device number.')
    args = parser.parse_args()

    # Correct way to load config as an object
    config = Config.from_yaml(args.config)  # or use the constructor as shown above

    if args.device is not None:
        config.device = args.device

    main(config)