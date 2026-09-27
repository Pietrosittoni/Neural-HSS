import numpy as np

import argparse

parser = argparse.ArgumentParser(
                        prog='HSS-learning')

parser.add_argument("--n_data", dest="n_data", type=int, default=1000,help = 'Number of samples to produce')
parser.add_argument("--n_grid_pts", dest="n_grid_pts", type=int, default=128)
parser.add_argument("--nu", dest="nu", type=float, default=1e-2,help = 'Diffusion coefficient')

args = parser.parse_args()

# Parameters and Grid Setup
num_samples = args.n_data  # number of samples (rows in X)
n = args.n_grid_pts  # number of grid points (including boundaries)
nu = args.nu  # viscosity (diffusion) coefficient

x = np.linspace(0, 1, n)  # uniform grid on [0,1]
h = x[1] - x[0]  # grid spacing

# Newton Solver parameters
tol = 1e-6  # tolerance for Newton convergence
max_iter = 50  # maximum number of Newton iterations
N_modes_fourier = 20

# Preallocate Matrices for Data
# X = np.random.rand(num_samples, n)  # each row is a random f vector (right-hand side)
X = np.random.rand(num_samples, n)@np.hstack( [np.sin(2 * k * np.pi * x) for k in range(N_modes_fourier)] ) # each row is a random f vector (right-hand side)
Y = np.zeros((num_samples, n))  # to store the computed solutions

# Loop Over Each Sample and Solve the Nonlinear System
for sample in range(num_samples):
    f = X[sample, :].copy()
    f[0] = 0
    f[-1] = 0
    
    # Initial guess for u: using zeros (satisfies BC)
    u = np.zeros(n)
    
    for _ in range(max_iter):
        # Assemble F(u)
        F = np.zeros(n)
        F[0] = u[0]
        F[-1] = u[-1]
        
        for i in range(1, n - 1):
            u_xx = (u[i - 1] - 2 * u[i] + u[i + 1]) / h**2
            u_x = (u[i + 1] - u[i - 1]) / (2 * h)
            F[i] = -nu * u_xx + u[i] * u_x - f[i]
        
        # Check convergence
        if np.linalg.norm(F, np.inf) < tol:
            break
        
        # Assemble the Jacobian matrix J
        J = np.zeros((n, n))
        J[0, 0] = 1
        J[-1, -1] = 1
        
        for i in range(1, n - 1):
            J[i, i - 1] = -nu / h**2 - u[i] / (2 * h)
            J[i, i] = 2 * nu / h**2 + (u[i + 1] - u[i - 1]) / (2 * h)
            J[i, i + 1] = -nu / h**2 + u[i] / (2 * h)
        
        # Solve for Newton update: J * delta = -F
        delta = np.linalg.solve(J, -F)
        
        # Update u
        u += delta
        
        # Check for small update size as convergence
        if np.linalg.norm(delta, np.inf) < tol:
            break
    
    # Store the computed solution in Y
    Y[sample, :] = u

# Export the Data to CSV
np.savetxt("./data/dataset_1DBurger_p.csv", np.hstack((X, Y)), delimiter=",")

print("Dataset saved to 'dataset_1DBurger.csv'")