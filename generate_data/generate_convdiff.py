import numpy as np
import scipy.linalg as la
import argparse

parser = argparse.ArgumentParser(
                        prog='HSS-learning')

parser.add_argument("--n_data", dest="n_data", type=int, default=1000,help = 'Number of samples to produce')
parser.add_argument("--n_grid_pts", dest="n_grid_pts", type=int, default=128)
parser.add_argument("--D", dest="D", type=float, default=1e-2,help = 'Diffusion coefficient')
parser.add_argument("--c", dest="c", type=float, default=1.,help = 'Convection coefficient')

args = parser.parse_args()

# Parameters and Grid Setup
num_samples = args.n_data  # number of samples (rows in X)
n = args.n_grid_pts  # number of grid points (including boundaries)
D = args.D  # diffusion coefficient
c = args.c  # convection coefficient
N_modes_fourier = 20

# Create grid on [0,1]
x = np.linspace(0, 1, n)  # grid points
h = x[1] - x[0]  # uniform grid spacing

# Build Finite Difference Matrix A
A = np.zeros((n, n))
A[0, 0] = 1  # u(0) = 0
A[-1, -1] = 1  # u(1) = 0

# Fill in the interior rows using central differences
for i in range(1, n - 1):
    A[i, i - 1] = -D / h**2 - c / (2 * h)
    A[i, i] = 2 * D / h**2
    A[i, i + 1] = -D / h**2 + c / (2 * h)

# Perform LU factorization of A
P, L, U = la.lu(A)

# Generate Data and Solve for Each Sample
# X = np.random.rand(num_samples, n)
X = np.random.rand(num_samples, n)@np.hstack( [np.sin(2 * k * np.pi * x) for k in range(N_modes_fourier)] )
Y = np.zeros((num_samples, n))

for i in range(num_samples):
    f = X[i, :].copy()  # Copy to avoid modifying X
    f[0] = 0  # Dirichlet BC at x=0
    f[-1] = 0  # Dirichlet BC at x=1
    
    # Solve A * u = f using the LU factorization
    u = la.solve(U, la.solve(L, P @ f))
    
    # Save the solution
    Y[i, :] = u

# Export the Data to CSV
np.savetxt("./data/dataset_1DConvDiff_p.csv", np.hstack((X, Y)), delimiter=",",)