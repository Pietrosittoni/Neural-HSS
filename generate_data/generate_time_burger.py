import numpy as np
from scipy.integrate import solve_ivp
import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
                        prog='HSS-learning')

parser.add_argument("--n_data", dest="n_data", type=int, default=1000,help = 'Number of samples to produce')
parser.add_argument("--n_grid_pts", dest="n_grid_pts", type=int, default=128)
parser.add_argument("--nu", dest="nu", type=float, default=1e-4,help = 'Diffusion coefficient')
parser.add_argument("--dt", dest="dt", type=float, default=1e-4,help = 'Time step for the heat equation')
parser.add_argument("--n_fourier_features", dest="n_fourier_features", type=int, default=20,help = 'Number of Fourier features for the input data')

args = parser.parse_args()

# Parameters and Grid Setup
num_samples = args.n_data  # number of samples (rows in X)
n = args.n_grid_pts  # number of grid points (including boundaries)
nu = args.nu  # viscosity (diffusion) coefficient

x = np.linspace(0, 1, n)  # uniform grid on [0,1]
dx = x[1] - x[0]  # grid spacing
dt = args.dt

# Newton Solver parameters
tol = 1e-6  # tolerance for Newton convergence
max_iter = 50  # maximum number of Newton iterations
N_modes_fourier = args.n_fourier_features

def burgers_rhs(t, u):
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2  # second derivative
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)        # first derivative
    rhs = -nu * u_xx + u * u_x
    
    # # Enforce boundary conditions (u[0] = u[-1] = 0)
    rhs[0] = 0
    rhs[-1] = 0
    return rhs

# Preallocate Matrices for Data
# X = np.random.rand(num_samples, n)  # each row is a random f vector (right-hand side)
X = np.linalg.qr(np.random.rand(num_samples, N_modes_fourier))[0]@np.vstack( [np.sin(2 * k * np.pi * x) for k in range(N_modes_fourier)] )   ## generate data with random fourier expansion
Y = np.zeros((num_samples, n))  # to store the computed solutions

# Loop Over Each Sample and Solve the Nonlinear System
for sample in range(num_samples):
    u_0 = X[sample, :]
    
    solution = solve_ivp(
        fun=burgers_rhs,
        y0=u_0,
        method='RK45',
        t_span=(0, dt),
        t_eval=[dt]
    )
    # Store the computed solution in Y
    u_1 = solution.y
    Y[sample, :] = u_1[:, -1]  # reshape solution

# Export the Data to CSV
X,Y = torch.tensor(X), torch.tensor(Y)

torch.save(torch.hstack([X,Y]), './data/dataset_1DtimeBurger.pt')

plt.plot(X[0,:])
plt.plot(Y[0,:])
plt.title("Brger next step")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend(['u(x,0)', 'u(x,dt)']) 
plt.savefig('testing_figures/test_timeburger.png')