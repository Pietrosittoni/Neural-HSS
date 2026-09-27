import numpy as np
from scipy.sparse import diags, kron, eye, csc_matrix,lil_matrix
from scipy.sparse.linalg import spsolve
import os
import argparse
import pandas as pd


def sample_fourier_2d(n, N_modes_fourier=20):
    """
    Sample a 2D function as a sum of random Fourier sine modes, with coefficients in [0, 1].
    """
    x = np.linspace(0, 1, n, endpoint=False)
    y = np.linspace(0, 1, n, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    f = np.zeros((n, n))
    for kx in range(1, N_modes_fourier+1):
        for ky in range(1, N_modes_fourier+1):
            coeff = np.random.uniform(0,1)
            f += coeff * np.sin(2 * np.pi * kx * xx) * np.sin(2 * np.pi * ky * yy)*(xx**2+yy**2)
    return f

def generate_dataset_2d(n, n_samples, N_modes_fourier=20):
    """
    Generate dataset of random Fourier input functions and solutions to 2D Poisson problem
    using finite difference (5-point Laplacian).
    """
    h = 1.0 / n
    # 1D Laplacian
    main_diag = -4.0 * np.ones(n)
    off_diag = np.ones(n-1)
    diagonals = [main_diag, off_diag, off_diag]
    L1D = diags(diagonals, [0, -1, 1], shape=(n, n))
    I = eye(n)
    # 2D Laplacian with Dirichlet BCs
    L2D = kron(I, L1D) + kron(L1D, I)
    L2D = L2D / (h**2)
    L2D = csc_matrix(L2D)

    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))

    for i in range(n_samples):
        f = sample_fourier_2d(n, N_modes_fourier)
        # Dirichlet BCs: zero out boundary
        # f[0, :] = 0
        # f[-1, :] = 0
        # f[:, 0] = 0
        # f[:, -1] = 0
        X[i, :, :] = f
        u = spsolve(L2D, f.flatten())
        Y[i, :, :] = u.reshape(n, n)
    X = X.reshape(n_samples, n*n)
    Y = Y.reshape(n_samples, n*n)

    return X, Y


def laplacian_2d_4th_order(n, h):
    N = n * n
    L = lil_matrix((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Boundary: set identity
            if i < 2 or i >= n-2 or j < 2 or j >= n-2:
                L[idx, idx] = 1.0
            else:
                # Center
                L[idx, idx] = -60.0
                # x-direction
                L[idx, idx - 2*n] = -1.0
                L[idx, idx - 1*n] = 16.0
                L[idx, idx + 1*n] = 16.0
                L[idx, idx + 2*n] = -1.0
                # y-direction
                L[idx, idx - 2] = -1.0
                L[idx, idx - 1] = 16.0
                L[idx, idx + 1] = 16.0
                L[idx, idx + 2] = -1.0
    L = L / (12 * h**2)
    return csc_matrix(L)

# def laplacian_2d_4th_order(n, h):
#     """
#     Construct the 2D 4th-order finite difference Laplacian operator with Dirichlet BCs.
#     Returns a sparse matrix of shape (n*n, n*n).
#     """
#     # 1D 4th-order Laplacian stencil: [-1, 16, -30, 16, -1] / (12 h^2)
#     main_diag = -30.0 * np.ones(n)
#     off1_diag = 16.0 * np.ones(n-1)
#     off2_diag = -1.0 * np.ones(n-2)
#     diagonals = [main_diag, off1_diag, off1_diag, off2_diag, off2_diag]
#     offsets = [0, -1, 1, -2, 2]
#     L1D = diags(diagonals, offsets, shape=(n, n))
#     I = eye(n)
#     # 2D Laplacian
#     L2D = kron(I, L1D) + kron(L1D, I)
#     L2D = L2D / (12 * h**2)
#     return csc_matrix(L2D)

def generate_dataset_2d_4th(n, n_samples, N_modes_fourier=20):
    """
    Generate dataset of random Fourier input functions and solutions to 2D Poisson problem
    using 4th-order finite difference (9-point Laplacian).
    """
    h = 1.0 / n
    L2D = laplacian_2d_4th_order(n, h)

    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))


    for i in range(n_samples):
        f = sample_fourier_2d(n, np.random.choice(range(1, N_modes_fourier+1)))
        # Dirichlet BCs: zero out boundary
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0
        X[i, :, :] = f
        u = spsolve(L2D, f.reshape(n*n))
        # plt.imshow(u, cmap='viridis')
        # plt.savefig(f'output_{i}.png', dpi=300)
        # plt.close()
        Y[i, :, :] = u.reshape(n, n)

    X = X.reshape(n_samples, n*n)
    Y = Y.reshape(n_samples, n*n)

    return X, Y

# You can use your existing sample_fourier_2d function.

def main():
    parser = argparse.ArgumentParser(prog='HSS-learning-2D')
    parser.add_argument('--n_grid_pts', type=int, default=64, help='Number of grid points per axis (resolution)')
    parser.add_argument('--n_samples', type=int, default=4200, help='Number of samples to produce')
    parser.add_argument('--N_modes_fourier', type=int, default=10, help='Number of Fourier sine modes')
    parser.add_argument('--output_dir', type=str, default="./data_hss", help='Output directory for dataset')
    args = parser.parse_args()

    np.random.seed(42)

    # X, Y = generate_dataset_2d(
    #     n=args.n_grid_pts,
    #     n_samples=args.n_samples,
    #     N_modes_fourier=args.N_modes_fourier
    # )

    X,Y = generate_dataset_2d_4th(
        n=args.n_grid_pts,
        n_samples=args.n_samples,
        N_modes_fourier=args.N_modes_fourier
    )

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dataset_2DPoisson_res{args.n_grid_pts}_N{args.n_samples}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    df = pd.DataFrame({
        'X': list(X),
        'Y': list(Y)
    })
    df.to_parquet(output_path, engine="pyarrow")

    print(f"Dataset saved to '{output_path}' (Parquet format, X and Y as separate columns)")

if __name__ == '__main__':
    main()