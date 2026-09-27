import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

@torch.no_grad
def get_linear_operator(model,n):

    device = list(model.parameters())[0].device
    operator = model(torch.eye(n,device = device))

    return operator

@torch.no_grad
def plot_operators(model,X,Y,savename = 'operators'):

    W_star,_,_,_ = torch.linalg.lstsq(X,Y)
    W_star = W_star.cpu().detach().numpy()

    W_hat = get_linear_operator(model,X.shape[1]).cpu().detach().numpy()

    relative_operator_error = np.linalg.norm(W_hat-W_star,ord = 2)/np.linalg.norm(W_star,ord = 2)

    plt.figure(figsize=(30,30))
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    im1 = ax1.imshow(W_star,cmap='jet', aspect='equal')
    im2= ax2.imshow(W_hat,cmap='jet', aspect='equal')
    im3 = ax3.imshow(np.abs(W_hat-W_star),cmap='jet', aspect='equal',norm=LogNorm())
    ax1.set_title(f'real solution operator for {savename}')
    fig.colorbar(im1)
    fig.colorbar(im2)
    fig.colorbar(im3)
    ax2.set_title('estimated solution operator')
    ax3.set_title(f'difference between two operators (rel err. {relative_operator_error})')
    fig.tight_layout()
    plt.savefig(f'./figures/{savename}.png',bbox_inches='tight')
    return relative_operator_error