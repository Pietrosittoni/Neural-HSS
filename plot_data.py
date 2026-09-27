import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(data_path,spatial_resolution,train_size):
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
    df = pd.read_parquet(data_path, engine="pyarrow")
    X = torch.tensor(list(df['X'].values),dtype=torch.float32)
    Y = torch.tensor(list(df['Y'].values),dtype=torch.float32)

    # Downsample: select spatial_resolution evenly spaced indices
    n_points = X.shape[1]
    if spatial_resolution > n_points:
        raise ValueError("spatial_resolution cannot be greater than the number of points in the data.")
    indices = torch.linspace(0, n_points - 1, spatial_resolution).long()
    X = X[:, indices]
    Y = Y[:, indices]

    # Split into train and validation sets

    val_size = int(train_size * 0.15)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_val = X[train_size:train_size + val_size]
    Y_val = Y[train_size:train_size + val_size]

    return X_train, Y_train, X_val, Y_val


def plot_data(X_train, Y_train, X_val, Y_val, num_samples=5, domain=(-2, 2)):
    """
    Plots a few random samples from the training and validation sets, using the physical spatial grid for the x-axis.
    """
    plt.figure(figsize=(12, 6))
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_points = X_train.shape[1]
    x_axis = np.linspace(domain[0], domain[1], n_points, endpoint=False)

    # Plot random samples from training set
    train_indices = np.random.choice(n_train, size=min(num_samples, n_train), replace=False)
    for idx in train_indices:
        plt.plot(x_axis, X_train[idx], color='blue', alpha=0.4, linestyle='-', label='Input train' if idx == train_indices[0] else "")
        plt.plot(x_axis, Y_train[idx], color='red', alpha=0.7, linestyle='-', label='Output train' if idx == train_indices[0] else "")

    # Plot random samples from validation set
    val_indices = np.random.choice(n_val, size=min(num_samples, n_val), replace=False)
    for idx in val_indices:
        plt.plot(x_axis, X_val[idx], color='cyan', alpha=0.4, linestyle='--', label='Input val' if idx == val_indices[0] else "")
        plt.plot(x_axis, Y_val[idx], color='orange', alpha=0.7, linestyle='--', label='Output val' if idx == val_indices[0] else "")

    plt.xlabel('Spatial Coordinate')
    plt.ylabel('Value')
    plt.title('Random Samples from Train and Validation Sets')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('data_2.png')
    plt.show()


if __name__ == "__main__":
    data_path = "/home/_/data02/data_hss/dataset_1DPoisson_res1024_N3000.parquet"
    spatial_resolution = 1024
    train_size = 500
    X_train, Y_train, X_val, Y_val = read_data(data_path,spatial_resolution,train_size)
    plot_data(X_train,Y_train,X_val,Y_val)