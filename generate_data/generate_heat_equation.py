import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import argparse
import torch
import os
import pandas as pd

parser = argparse.ArgumentParser(prog='HSS-learning')
parser.add_argument("--n_samples", type=int, default=20000, help='Number of samples to produce')
parser.add_argument("--n_grid_pts", type=int, default=1024, help='Number of grid points (resolution)')
parser.add_argument("--T_max", type=float, default=5.0, help='Maximum time horizon for the heat equation')
parser.add_argument("--dt_out", type=float, default=0.2, help='Output time step for downsampling the solution')
parser.add_argument("--n_fourier_features", type=int, default=20, help='Number of Fourier features for the input data')
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(42)

n = args.n_grid_pts
N_modes_fourier = args.n_fourier_features
T_max = args.T_max
n_samples = args.n_samples
dt_out = args.dt_out

x = np.linspace(0, 1, n)

# Create fourth-order finite difference matrix for 1D Heat Equation
h = 1.0 / n
main_diag = -30.0 * np.ones(n)
off1_diag = 16.0 * np.ones(n-1)
off2_diag = -1.0 * np.ones(n-2)
A = diags(
    [off2_diag, off1_diag, main_diag, off1_diag, off2_diag],
    offsets=[-2, -1, 0, 1, 2],
    shape=(n, n)
).toarray() / (12 * h**2)
# Dirichlet boundary conditions: set first two and last two rows to identity
A[0, :] = 0; A[0, 0] = 1
A[1, :] = 0; A[1, 1] = 1
A[-2, :] = 0; A[-2, -2] = 1
A[-1, :] = 0; A[-1, -1] = 1

# Time points for output (downsampled)
t_eval = np.arange(0, T_max + dt_out, dt_out)
num_time_steps = len(t_eval)

# Generate random input data (Fourier expansion)
X = np.zeros((n_samples, n))
trajectories = []
for i in range(n_samples):
    coeffs = np.random.rand(N_modes_fourier)
    X[i, :] = np.sum([coeffs[k] * np.sin(2 * (k+1) * np.pi * x) for k in range(N_modes_fourier)], axis=0)

# Compute the full trajectory for each sample
for i in range(n_samples):
    u_0 = X[i, :]
    solution = solve_ivp(fun=lambda t, v: A @ v, y0=u_0, method='RK45', t_span=(0, T_max), t_eval=t_eval)
    trajectories.append(solution.y.T)  # shape: (num_time_steps, n)

trajectories = np.array(trajectories)  # shape: (n_samples, num_time_steps, n)

# Construct the output directory and filename
output_dir = "/home/_/data02/data_hss"
os.makedirs(output_dir, exist_ok=True)
filename = f"dataset_1DHeat_trajectory_res{n}_N{n_samples}_T{T_max}_dt{dt_out}.parquet"
output_path = os.path.join(output_dir, filename)

# Store the full trajectory for each sample in a DataFrame
# Each row is a numpy array of shape (num_time_steps, n)
df = pd.DataFrame({
    'trajectory': [traj.tolist() for traj in trajectories]
})
df.to_parquet(output_path, engine="pyarrow")

print(f"Dataset saved to '{output_path}' (Parquet format, each row is a trajectory of shape (num_time_steps, n_grid_pts))")