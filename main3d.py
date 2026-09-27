import argparse
import os
import random
import torch
import wandb
import numpy as np
from torch.utils.data import TensorDataset
from models import MODEL_REGISTRY  # <-- 2D model registry
from utils.config import Config
import torch.nn as nn
from utils.train_eval import train, valid_steady
from utils.read_pois_3d import read_data3d  # <-- 2D data loader
from models.utils import get_flops

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# class RelativeMSE(nn.Module):
#     def __init__(self, epsilon: float = 1e-3):
#         super().__init__()
#         self.epsilon = epsilon
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         diff_norm = torch.norm(predictions - targets, p='fro', dim=(-2, -1))**2
#         target_norm = torch.norm(targets, p='fro', dim=(-2, -1))**2 + self.epsilon
#         relative_l2 = diff_norm / target_norm
#         return torch.mean(relative_l2)

class RelativeMSE(nn.Module):
    """
    Computes the relative L2 loss:
    Loss = mean_over_batch( ||pred_b - y_b||_F / (||y_b||_F + eps) )
    Where ||.||_F is the Frobenius norm over all non-batch dims.
    """
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions, targets: (b, c, h, h)
        diff_norm = torch.norm(predictions - targets, p='fro', dim=(2,3))  # (b,)
        diff_norm = torch.norm(diff_norm,dim = 1 )**2
        target_norm = torch.norm(targets, p='fro', dim=(2,3)) + self.epsilon  # (b,)
        target_norm = torch.norm(target_norm,dim = 1)**2
        relative_l2 = diff_norm / target_norm  # (b,)
        return relative_l2.mean()  # scalar

def main(config: str):
    wandb.login()
    wandb.init(
        project=config.project,
        config=config.__dict__,
        tags=config.tags,
        # entity="_",
    )

    set_seed(config.seed)
    device = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.')
    for size in [16,32,64,128,256,512,768,900]:

        train_dataloader, valid_dataloader, scale = read_data3d(config, device, size)
        # scale = 1.
        wandb.log({'scale': scale})

        if config.model not in MODEL_REGISTRY:
            raise ValueError(f"Model {config.model} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")

        print(f'using model {config.model}.')
        model_class = MODEL_REGISTRY[config.model]
        model = model_class(
            **{k: v for k, v in config.__dict__.items() if k in model_class.__init__.__code__.co_varnames}
        ).to(device)
        print({k: v for k, v in config.__dict__.items() if k in model_class.__init__.__code__.co_varnames})

        total_params = sum(p.numel() for p in model.parameters())
        #flops = get_flops(model, input_size=(1, config.input_dim))
        wandb.log({'number_of_parameters': total_params})
        print(f"Total number of parameters: {total_params}")

        wandb.watch(model)

     
        MSE = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
        # optimizer = SOAP(model.parameters(),lr = config.learning_rate,weight_decay=config.weight_decay)
        iters = len(train_dataloader) * config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=config.learning_rate_min)

        print('Training...')
        for epoch in range(config.epochs):
            print(f'Epoch: {epoch + 1}/{config.epochs}')
            train(train_dataloader, model, MSE, optimizer, scheduler, scale = scale)
            #valid_steady(valid_dataloader, model, MSE, if_plot=False, scale=scale)
        valid_steady(valid_dataloader, model, MSE, if_plot=False, scale=scale)

    print('Training complete.')
    model_path = config.folder_model_path + f'/models/{wandb.run.name}'
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f'{model_path}/model.pt')
    print(f'Model saved as {model_path}/model.pt.')
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--device', type=int, help='Override GPU device number.')
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    if args.device is not None:
        config.device = args.device
    main(config)