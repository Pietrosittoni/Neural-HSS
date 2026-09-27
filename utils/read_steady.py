import torch
from torch.utils.data import TensorDataset
import pandas as pd
import torch
import numpy as np

def read_data(config):
    """
    Reads the dataset from config.data_path, downsamples X and Y to the specified spatial_resolution,
    and splits into train and validation sets according to config.train_size.

    Args:
        config: An object with 'data_path' (str) and 'train_size' (int) attributes.
        spatial_resolution: The number of spatial points to keep (int).

    Returns:
        X_train, Y_train: Training tensors of shape (train_size, spatial_resolution)
        X_val, Y_val: Validation tensors of shape (val_size, spatial_resolution)
    """
    # Load the dataset
    df = pd.read_parquet(config.data_path, engine="pyarrow")
    X = torch.from_numpy(np.array(list(df['X'].values))).to(torch.float32)
    Y = torch.from_numpy(np.array(list(df['Y'].values))).to(torch.float32)
    # Downsample: select spatial_resolution evenly spaced indices
    n_points = X.shape[1]
    if config.spatial_resolution > n_points:
        raise ValueError("spatial_resolution cannot be greater than the number of points in the data.")
    indices = torch.linspace(0, n_points - 1, config.spatial_resolution).long()
    X = X[:, indices]
    Y = Y[:, indices]

    # Split into train and validation sets
    train_size = config.train_size
    val_size = 1000
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_val = X[-val_size:]
    Y_val = Y[-val_size:]
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    return X_train, Y_train, X_val, Y_val

def scaler(data):
    return torch.max(torch.abs(data)).item()

def read_steady_data(config,device):
    # Set up data paths
    train_x, train_y, valid_x, valid_y = read_data(config)
    train_data_input, train_data_output = train_x.to(device), train_y.to(device)
    train_dataloader = torch.utils.data.DataLoader(TensorDataset(train_data_input, train_data_output), 
                                                 batch_size=config.batch_size, 
                                                 shuffle=True)
    
    scale = scaler(train_data_output)
    
    valid_x, valid_y = valid_x.to(device), valid_y.to(device)
    valid_dataloader = torch.utils.data.DataLoader(TensorDataset(valid_x, valid_y), 
                                                 batch_size=config.batch_size, 
                                                 shuffle=False)
    
    return train_dataloader, valid_dataloader, scale