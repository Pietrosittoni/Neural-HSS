import torch
from torch.utils.data import TensorDataset
import pandas as pd
import torch
import numpy as np
from einops import rearrange

def read_data2d(config):
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
    dim_data = int(np.sqrt(X.shape[1]))  # Assuming X and Y are square matrices
    X,Y = X.reshape(X.shape[0], dim_data,dim_data), Y.reshape(Y.shape[0], dim_data,dim_data)  # Flatten the 2D data
    # Downsample: select spatial_resolution evenly spaced indices
    n_points = X.shape[1]
    if config.spatial_resolution > n_points:
        raise ValueError("spatial_resolution cannot be greater than the number of points in the data.")
    indices_1d = torch.linspace(0, n_points - 1, config.spatial_resolution).long()
    grid_x, grid_y = torch.meshgrid(indices_1d, indices_1d, indexing='ij')
    X = X[:, grid_x,grid_y]
    Y = Y[:, grid_x,grid_y]

    # Split into train and validation sets
    train_size = config.train_size
    val_size = 1000
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_val = X[-val_size:]
    Y_val = Y[-val_size:]

    return X_train, Y_train, X_val, Y_val

def scaler(data):
    return torch.mean(torch.abs(data)).item()

def read_steady_data2d(config,device):

    if 'gray_scott' not in config.data_path.lower():
        
        # Set up data paths
        train_x, train_y, valid_x, valid_y = read_data2d(config)
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
    
    else:

        return read_thewell(config,device)


def read_thewell(config,device):
    import os,h5py
    output_dir = "./data_hss/"
    paths = ['train','valid','test']
    data = {}
    for p in paths:
        filename = f"gray_scott_reaction_diffusion_{p}.h5"
        output_path = os.path.join(output_dir, filename)
        with h5py.File(output_path, 'r') as hf:
            X = torch.tensor(hf['X'][:],device = device)  # numpy array
            X = rearrange(X,'b x y c -> b c x y')
            Y = torch.tensor(hf['Y'][:],device = device)
            Y = rearrange(Y,'b x y c -> b c x y')
        data[p] = (X,Y)

    train_size = config.train_size
    
    data['valtest'] = (torch.cat([data['valid'][0],data['test'][0]],dim = 0),torch.cat([data['valid'][1],data['test'][1]],dim = 0))
    train_data_input, train_data_output = data['train']
    train_data_input,train_data_output = train_data_input[:train_size],train_data_output[:train_size]
    mean_in,std_in = torch.mean(train_data_input,dim = 0),torch.std(train_data_input,dim = 0)
    mean_out,std_out = torch.mean(train_data_output,dim = 0),torch.std(train_data_output,dim = 0)
    train_dataloader = torch.utils.data.DataLoader(NormalizedTensorDataset(train_data_input, train_data_output,mean=(mean_in,mean_out),std = (std_in,std_out)),
                                                    batch_size=config.batch_size, 
                                                    shuffle=True
                                                    )
    scale = scaler(train_data_output)

    valid_x, valid_y = data['valtest']
    valid_dataloader = torch.utils.data.DataLoader(NormalizedTensorDataset(valid_x, valid_y,mean=mean_in,std = std_in), 
                                                    batch_size=config.batch_size, 
                                                    shuffle=False)
    
    return train_dataloader , valid_dataloader, scale
            

class NormalizedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, mean, std):
        self.X = X
        self.Y = Y
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # x = (self.X[idx] - self.mean[0]) / (self.std[0] + 1e-8)
        # y = (self.Y[idx] - self.mean[1]) / (self.std[1] + 1e-8)
        x,y = self.X[idx],self.Y[idx]
        return x, y


def prepare_thewell(config = None,device = torch.device('cpu')):
    output_dir = "./data_hss/"
    from the_well.data import WellDataset
    import os,h5py
    data = {}
    batch_size = 64
    paths = ['train','valid','test']
    for p in paths:
        dataset = WellDataset(
        well_base_path=f"./data_hss/data_hss/datasets/",
        well_dataset_name="gray_scott_reaction_diffusion",
        well_split_name = p,
        full_trajectory_mode = True,
        use_normalization=False,
        max_rollout_steps=1000
        )
        train_data = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        for i,batch in enumerate(train_data):
            # print(batch.keys())
            print(f'batch {i}/{len(train_data)}...')
            # print(batch['input_fields'].shape,batch['output_fields'].shape)
            field_in = batch['input_fields'][:,0,:,:,:].to(device)
            Fk = torch.ones(size = (field_in.shape[0],128,128,2),device = device)
            # print(batch['constant_scalars'].shape)
            for b in range(field_in.shape[0]):
                Fk[b,:,:,0].mul_(batch['constant_scalars'][b,0])
                Fk[b,:,:,1].mul_(batch['constant_scalars'][b,1])
            data_in = torch.cat([field_in,Fk],dim = 3)
            data_out = batch['output_fields'][:,-1,:,:,:].to(device)
            try:
                train_dataset_input = torch.cat([train_dataset_input,data_in ],dim = 0)
                train_dataset_output = torch.cat([train_dataset_output,data_out],dim = 0)
                # break
            except:
                train_dataset_input = data_in
                train_dataset_output = data_out
            

        print(f'finished! dataset shape in-out {train_dataset_input.shape,train_dataset_output.shape}')
        # data[p] = (train_dataset_input,train_dataset_output)
        filename = f"gray_scott_reaction_diffusion_{p}"
        output_path = os.path.join(output_dir, filename)

        with h5py.File(output_path + '.h5', 'w') as hf:
            hf.create_dataset('X', data=train_dataset_input.cpu().numpy())
            hf.create_dataset('Y', data=train_dataset_output.cpu().numpy())

        print(f"Dataset saved to '{output_path}.h5' (HDF5 format, X and Y as separate datasets)")
        del train_dataset_input 
        del train_dataset_output

# prepare_thewell()



def smoothen(path = './data_hss/dataset_2DKS_res64_N6000.parquet'):

    import torchvision.transforms as transforms
    df = pd.read_parquet(path, engine="pyarrow")
    X = np.array(list(df['X'].values))
    Y = np.array(list(df['Y'].values))
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    X = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))
    # Y = Y.reshape(Y.shape[0], int(np.sqrt(Y.shape[1])), int(np.sqrt(Y.shape[1])))
    smooth = transforms.GaussianBlur((3,3),sigma = 1)
    X = smooth(torch.tensor(X,device = 'cuda:2')).cpu().numpy()
    X = X.reshape(X.shape[0], 64*64)
    df = pd.DataFrame({'X': list(X), 'Y': list(Y)})
    df.to_parquet('./data_hss/dataset_2DKS_res64_N6000_smooth.parquet', engine='pyarrow')


# smoothen()
    

