import numpy as np
from scipy.sparse import diags, kron, eye, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import os
import argparse
import pandas as pd


def sample_fourier_2d(n, N_modes_fourier=20):
    """
    Sample a 2D function as a sum of random Fourier sine modes on [0,1]^2,
    with coefficients in [0, 1].
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    f = np.zeros((n, n))
    sum_coeffs = 0.0
    for kx in range(1, N_modes_fourier + 1):
        for ky in range(1, N_modes_fourier + 1):
            coeff = np.random.uniform(-1/(N_modes_fourier),1/(N_modes_fourier))
            sum_coeffs += coeff**2
            f += coeff * np.sin(2 * np.pi * kx * xx) * np.sin(2 * np.pi * ky * yy)
    # f /= np.linalg.norm(f)  # Normalize to have unit L2 norm
    return f


def apply_dirichlet_bc(L, f_flat, n):
    """
    Modify sparse matrix L and RHS f_flat to enforce u=0 on the boundary of an n x n grid.
    """
    # Identify boundary node indices
    boundary = set()
    # bottom and top rows
    boundary.update(range(0, n))
    boundary.update(range(n*(n-1), n*n))
    # left and right columns
    boundary.update(range(0, n*n, n))
    boundary.update(range(n-1, n*n, n))

    for idx in boundary:
        # zero out row
        L[idx, :] = 0
        # set diagonal to 1
        L[idx, idx] = 1.0
        # zero RHS
        f_flat[idx] = 0.0

    return L, f_flat


def generate_dataset_2d(n, n_samples, N_modes_fourier=20):
    """
    Generate dataset of random Fourier input functions and solutions to 2D Poisson problem
    using 2nd-order finite difference (5-point Laplacian) with Dirichlet BCs.
    """
    # grid spacing
    h = 1.0/(n-1)

    # 1D Laplacian (5-point)
    main_diag = -4.0 * np.ones(n)
    off_diag = np.ones(n-1)
    L1D = diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(n, n))
    I = eye(n)
    L2D = kron(I, L1D) + kron(L1D, I)
    L2D = csc_matrix(L2D / (h**2))

    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))

    for i in range(n_samples):
        f = sample_fourier_2d(n, N_modes_fourier)
        # enforce zero Dirichlet at boundary in RHS
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0

        f_flat = f.flatten(order='C')
        # copy matrix to modify BCs
        Lmod = L2D.copy().tolil()
        Lmod, f_flat = apply_dirichlet_bc(Lmod, f_flat, n)
        Lmod = csc_matrix(Lmod)

        u_flat = spsolve(Lmod, f_flat)
        u = u_flat.reshape((n, n), order='C')

        X[i] = f
        Y[i] = u

    # flatten for output
    X = X.reshape(n_samples, n*n)
    Y = Y.reshape(n_samples, n*n)
    return X, Y


def laplacian_2d_4th_order(n, h):
    """
    Construct the 2D 4th-order finite difference Laplacian operator with Dirichlet BCs.
    Returns a sparse matrix of shape (n*n, n*n).
    """
    N = n * n
    L = lil_matrix((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Boundary: identity for Dirichlet
            if i < 2 or i >= n-2 or j < 2 or j >= n-2:
                L[idx, idx] = 1.0
            else:
                # 4th-order stencil
                L[idx, idx]       = -60.0
                L[idx, idx-2*n]   = -1.0
                L[idx, idx-n]     = 16.0
                L[idx, idx+n]     = 16.0
                L[idx, idx+2*n]   = -1.0
                L[idx, idx-2]     = -1.0
                L[idx, idx-1]     = 16.0
                L[idx, idx+1]     = 16.0
                L[idx, idx+2]     = -1.0
    return csc_matrix(L / (12 * h**2))


def generate_dataset_2d_4th(n, n_samples, N_modes_fourier=20):
    """
    Generate dataset of random Fourier input functions and solutions to 2D Poisson problem
    using 4th-order finite difference (9-point Laplacian) with Dirichlet BCs.
    """
    h = 1.0/(n-1)
    L2D_4 = laplacian_2d_4th_order(n, h)

    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))

    for i in range(n_samples):
        f = sample_fourier_2d(n, np.random.randint(1, N_modes_fourier+1))
        # enforce zero Dirichlet at boundary
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0

        f_flat = f.flatten(order='C')
        # solve
        u_flat = spsolve(L2D_4, f_flat/ h)
        u = u_flat.reshape((n, n), order='C')

        X[i] = f/h
        Y[i] = u

    X = X.reshape(n_samples, n*n)
    Y = Y.reshape(n_samples, n*n)
    return X, Y


def main():
    parser = argparse.ArgumentParser(prog='HSS-learning-2D')
    parser.add_argument('--n_grid_pts', type=int, default=64, help='Number of grid points per axis')
    parser.add_argument('--n_samples', type=int, default=4200, help='Number of samples to produce')
    parser.add_argument('--N_modes_fourier', type=int, default=15, help='Number of Fourier sine modes')
    parser.add_argument('--output_dir', type=str, default='./data_hss', help='Output directory for dataset')
    parser.add_argument('--order', type=int, choices=[2,4], default=4, help='FD order: 2 or 4')
    args = parser.parse_args()

    np.random.seed(42)
    if args.order == 2:
        X, Y = generate_dataset_2d(args.n_grid_pts, args.n_samples, args.N_modes_fourier)
    else:
        X, Y = generate_dataset_2d_4th(args.n_grid_pts, args.n_samples, args.N_modes_fourier)

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dataset_2DPoisson_order{args.order}_res{args.n_grid_pts}_N{args.n_samples}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    df = pd.DataFrame({'X': list(X), 'Y': list(Y)})
    df.to_parquet(output_path, engine='pyarrow')
    print(f"Dataset saved to '{output_path}'")


if __name__ == '__main__':
    main()
