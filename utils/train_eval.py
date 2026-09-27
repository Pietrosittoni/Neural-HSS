import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch.nn as nn

# class RelativeL2Loss(nn.Module):
#     """
#     Computes the relative L2 loss:
#     Loss = mean_over_batch( ||pred_i - y_i||_2 / ||y_i||_2 )
    
#     Where ||.||_2 is the L2 norm (Euclidean norm)
#     """
#     def __init__(self, epsilon: float = 1e-8):
#         super(RelativeL2Loss, self).__init__()
#         self.epsilon = epsilon  # Small constant to avoid division by zero
    
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         # Calculate L2 norm of difference for each sample
#         # Assuming first dimension is batch dimension
#         diff_norm = torch.norm(predictions - targets, p=2, dim=-1)
        
#         # Calculate L2 norm of targets for each sample
#         target_norm = torch.norm(targets, p=2, dim=-1)+ self.epsilon
        
#         # Calculate relative L2 error for each sample
#         relative_l2 = diff_norm / target_norm
        
#         # Average over the batch
#         return torch.mean(relative_l2)


class RelativeL2Loss(nn.Module):
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
        diff_norm = torch.norm(diff_norm,dim = 1 )
        target_norm = torch.norm(targets, p='fro', dim=(2,3)) + self.epsilon  # (b,)
        target_norm = torch.norm(target_norm,dim = 1)
        relative_l2 = diff_norm / target_norm  # (b,)
        return relative_l2.mean()  # scalar
    

def get_grad_norms(model):
    """
    Computes the L2 norm of gradients for each parameter in the model.
    Returns a dictionary of norms and the total norm.
    """
    grad_norms = {}
    total = 0.0
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm(2).item()
            grad_norms[param_name] = norm
            total += norm ** 2
    return grad_norms, total

def train(data_loader, model, criterion, optim, lr_scheduler,scale=1.0):
    """
    Executes one epoch of training.
    """
    model.train()
    for _, (inputs, targets) in enumerate(data_loader):
        targets = targets/scale

        outputs = model(inputs)/scale
        loss = criterion(outputs, targets)

        optim.zero_grad()
        reg = 0.0
        for n,param in model.named_parameters():
            if "slope" in n and param.requires_grad:
                reg += (param-1.0)**2
        reg*= 1e-8
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional: gradient clipping
        optim.step()
        lr_scheduler.step()

        _, grad_total = get_grad_norms(model)
        wandb.log({'gradient_norm': grad_total})
        wandb.log({'learning_rate': lr_scheduler.get_last_lr()[0], 'train/loss': loss.item()})

def valid_steady(data_loader, model, criterion, if_plot=False,scale=1.0):
    """
    Evaluates the model on the validation set.
    If if_plot is True, logs a plot of predictions vs ground truth.
    """
    model.eval()
    relative_l2_loss = RelativeL2Loss()
    total_loss = 0.0
    total_relative_l2 = 0.0
    predictions, targets = [], []
    
    with torch.inference_mode():
        for batch_inputs, batch_targets in data_loader:
            batch_outputs = model(batch_inputs)#*scale
            total_loss += criterion(batch_outputs, batch_targets)
            total_relative_l2 += relative_l2_loss(batch_outputs, batch_targets)
            predictions.append(batch_outputs)
            targets.append(batch_targets)
        avg_loss = total_loss / len(data_loader)
        avg_relative_l2 = total_relative_l2 / len(data_loader)

    if if_plot:
        # Concatenate all predictions and targets for plotting
        preds = torch.cat(predictions, dim=0)
        targs = torch.cat(targets, dim=0)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for idx, sample_idx in enumerate(range(4)):
            ax = axes[idx // 2, idx % 2]
            x_vals = np.arange(len(targs[sample_idx]))
            ax.plot(x_vals, targs[sample_idx].cpu().numpy(), '--', color='#1f77b4', label="Ground Truth")
            ax.plot(x_vals, preds[sample_idx].cpu().numpy(), '--', color='#ff7f0e', label="Prediction")
            ax.set_title(f"Sample {sample_idx+1} Prediction vs Ground Truth", fontsize=14)
            ax.set_xlabel("Position", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(frameon=True, fontsize=12)
        plt.tight_layout()
        wandb.log({"validation/plot": wandb.Image(fig)})

    wandb.log({'validation/mse': avg_loss, 'validation/relative_l2': avg_relative_l2})
    print(f'validation/mse: {avg_loss:.7f}')
    print(f'validation/relative_l2: {avg_relative_l2:.7f}')
