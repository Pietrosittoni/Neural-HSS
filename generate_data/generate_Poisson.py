import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_banded as solve
from scipy.fftpack import dct, idct
import os
import argparse
import pandas as pd

def sample_fourier_1d(n, N_modes_fourier=20):
    """
    Sample a 1D function as a sum of random Fourier sine modes, with coefficients in [0, 1].
    """
    x_grid = np.linspace(0, 1, n, endpoint=False)
    coeffs = np.random.uniform(0, 1, N_modes_fourier)
    f = np.sum([coeffs[k] * np.sin(2 * (k+1) * np.pi * x_grid) for k in range(N_modes_fourier)], axis=0)
    return f

def generate_dataset_1d(n, n_samples, N_modes_fourier=20):
    """
    Generate dataset of random Fourier input functions and solutions to 1D Poisson problem
    using efficient banded solver (4th-order finite difference).
    """
    x_grid = np.linspace(-10, 10, n, endpoint=False)
    h = x_grid[1] - x_grid[0]

    # 4th-order finite difference coefficients for second derivative
    main_diag = -30.0 * np.ones(n)
    upper1_diag = 16.0 * np.ones(n - 1)
    upper2_diag = -1.0 * np.ones(n - 2)
    lower1_diag = 16.0 * np.ones(n - 1)
    lower2_diag = -1.0 * np.ones(n - 2)

    # Dirichlet boundary conditions: set first two and last two rows to identity
    main_diag[0:2] = 1.0
    main_diag[-2:] = 1.0
    upper1_diag[0:2] = 0.0
    upper2_diag[0:2] = 0.0
    lower1_diag[-2:] = 0.0
    lower2_diag[-2:] = 0.0

    # Construct banded matrix for solve_banded (5 bands: 2 upper, main, 2 lower)
    ab = np.zeros((5, n))
    ab[0, 2:] = upper2_diag      # 2nd upper diagonal
    ab[1, 1:] = upper1_diag      # 1st upper diagonal
    ab[2, :] = main_diag         # main diagonal
    ab[3, :-1] = lower1_diag     # 1st lower diagonal
    ab[4, :-2] = lower2_diag     # 2nd lower diagonal
    ab /= 12 * h ** 2

    # For Dirichlet BCs, set the first two and last two rows to identity
    ab[0, :2] = 0.0
    ab[1, :2] = 0.0
    ab[2, :2] = 1.0
    ab[3, :2] = 0.0
    ab[4, :2] = 0.0
    ab[0, -2:] = 0.0
    ab[1, -2:] = 0.0
    ab[2, -2:] = 1.0
    ab[3, -2:] = 0.0
    ab[4, -2:] = 0.0

    # Allocate arrays
    X = np.zeros((n_samples, n))
    Y = np.zeros((n_samples, n))

    for i in range(n_samples):
        f = sample_fourier_1d(n, N_modes_fourier)
        f[0] = 0
        f[1] = 0
        f[-2] = 0
        f[-1] = 0
        X[i, :] = f
        Y[i, :] = solve((2, 2), ab, f)  # 2 upper, 2 lower

    return X, Y

def main():
    parser = argparse.ArgumentParser(prog='HSS-learning')
    parser.add_argument('--n_grid_pts', type=int, default=1024, help='Number of grid points (resolution)')
    parser.add_argument('--n_samples', type=int, default=3000, help='Number of samples to produce')
    parser.add_argument('--N_modes_fourier', type=int, default=20, help='Number of Fourier sine modes')
    parser.add_argument('--output_dir', type=str, default="./data_hss", help='Output directory for dataset')
    args = parser.parse_args()

    np.random.seed(42)

    X, Y = generate_dataset_1d(
        n=args.n_grid_pts,
        n_samples=args.n_samples,
        N_modes_fourier=args.N_modes_fourier
    )

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dataset_DPoisson_res{args.n_grid_pts}_N{args.n_samples}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    df = pd.DataFrame({
        'X': list(X),
        'Y': list(Y)
    })
    df.to_parquet(output_path, engine="pyarrow")

    print(f"Dataset saved to '{output_path}' (Parquet format, X and Y as separate columns)")

if __name__ == '__main__':
    main()
