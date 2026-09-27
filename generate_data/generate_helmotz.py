import numpy as np
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def sample_fourier_2d(n, N_modes_fourier=20, domain=((0, 1), (0, 1))):
    """
    Sample a 2D periodic function from random Fourier modes.
    
    Parameters:
    -----------
    n : int
        Number of grid points per dimension
    N_modes_fourier : int
        Maximum number of Fourier modes to use
    domain : tuple of tuples
        ((x_min, x_max), (y_min, y_max)) - Domain boundaries
    """
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    
    x = np.linspace(x_min, x_max, n, endpoint=False)
    y = np.linspace(y_min, y_max, n, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    f = np.zeros((n, n))
    
    # Scale coordinates to [0, 1] so Fourier phases are periodic.
    xx_scaled = (xx - x_min) / (x_max - x_min)
    yy_scaled = (yy - y_min) / (y_max - y_min)
    
    for kx in range(1, N_modes_fourier + 1):
        for ky in range(1, N_modes_fourier + 1):
            coeff = np.random.uniform(0, 1)
            phase = np.random.uniform(0, 2 * np.pi)
            theta = 2 * np.pi * (kx * xx_scaled + ky * yy_scaled) + phase
            f += coeff * np.sin(theta)
    return f


def periodic_second_derivative_1d_4th(n, h):
    """
    Construct the 1D 4th-order periodic second-derivative matrix.
    """
    D2 = diags(
        diagonals=[-1.0, 16.0, -30.0, 16.0, -1.0],
        offsets=[-2, -1, 0, 1, 2],
        shape=(n, n),
        format='lil'
    )

    # Periodic wrap-around couplings for +/-1 and +/-2 neighbors.
    D2[0, n - 1] = 16.0
    D2[n - 1, 0] = 16.0
    D2[0, n - 2] = -1.0
    D2[1, n - 1] = -1.0
    D2[n - 2, 0] = -1.0
    D2[n - 1, 1] = -1.0

    return (D2 / (12 * h**2)).tocsc()


def laplacian_2d_4th_order(n, hx, hy):
    """
    Construct the 2D 4th-order finite-difference Laplacian with periodic BCs.
    """
    # 1D periodic 4th-order second derivative:
    # ( -u_{i-2} + 16u_{i-1} - 30u_i + 16u_{i+1} - u_{i+2} ) / (12 h^2)
    D2x = periodic_second_derivative_1d_4th(n, hx)
    D2y = periodic_second_derivative_1d_4th(n, hy)
    I = eye(n, format='csc')

    # 2D Laplacian on a tensor-product periodic grid.
    L = kron(I, D2x, format='csc') + kron(D2y, I, format='csc')
    return csc_matrix(L)


def helmholtz_operator_2d_4th(n, hx, hy, k):
    """
    Construct the 4th-order Helmholtz operator with periodic BCs: (-Δ - k^2 I)
    """
    L = laplacian_2d_4th_order(n, hx, hy)
    I = eye(n * n, format='csc')
    A = -L - (k**2) * I
    return A


def generate_helmholtz_dataset_2d(n, n_samples, k, N_modes_fourier=20, domain=((0, 1), (0, 1))):
    """
    Generate dataset of random Fourier input functions and solutions
    to the 2D high-frequency Helmholtz problem with periodic BCs.
    
    Parameters:
    -----------
    n : int
        Number of grid points per dimension
    n_samples : int
        Number of samples to generate
    k : float
        Wavenumber (frequency)
    N_modes_fourier : int
        Maximum number of Fourier modes to use
    domain : tuple of tuples
        ((x_min, x_max), (y_min, y_max)) - Domain boundaries
    """
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    
    # Grid spacing in each axis.
    hx = (x_max - x_min) / n
    hy = (y_max - y_min) / n
    
    A = helmholtz_operator_2d_4th(n, hx, hy, k)

    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))

    for i in range(n_samples):
        f = sample_fourier_2d(n, np.random.choice(range(1, N_modes_fourier + 1)), domain)

        X[i, :, :] = f

        u = spsolve(A, f.reshape(n * n))
        Y[i, :, :] = u.reshape(n, n)

        # Plot first sample
        if i == 0:
            plt.figure(figsize=(10, 4))
            
            # Create coordinate grids for plotting
            x = np.linspace(x_min, x_max, n, endpoint=False)
            y = np.linspace(y_min, y_max, n, endpoint=False)
            
            plt.subplot(1, 2, 1)
            plt.imshow(f, cmap='viridis', extent=[x_min, x_max, y_min, y_max], origin='lower')
            plt.title(f"Input $f(x,y)$ (k={k})")
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(Y[i], cmap='viridis', extent=[x_min, x_max, y_min, y_max], origin='lower')
            plt.title("Solution $u(x,y)$")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f"helmholtz_sample_k{k}.png", dpi=300)
            plt.close()

    X = X.reshape(n_samples, n * n)
    Y = Y.reshape(n_samples, n * n)

    return X, Y


def main():
    parser = argparse.ArgumentParser(prog='Helmholtz-2D-dataset')
    parser.add_argument('--n_grid_pts', type=int, default=64, help='Grid resolution per axis')
    parser.add_argument('--n_samples', type=int, default=2500, help='Number of samples')
    parser.add_argument('--k', type=float, default=7.0, help='Wavenumber (frequency)')
    parser.add_argument('--N_modes_fourier', type=int, default=5, help='Number of Fourier sine modes')
    parser.add_argument('--x_min', type=float, default=-0.0, help='Minimum x-coordinate of domain')
    parser.add_argument('--x_max', type=float, default=1.0, help='Maximum x-coordinate of domain')
    parser.add_argument('--y_min', type=float, default=-0.0, help='Minimum y-coordinate of domain')
    parser.add_argument('--y_max', type=float, default=1.0, help='Maximum y-coordinate of domain')
    parser.add_argument('--output_dir', type=str, default="/home/sittoni/data02/HSS_learning/data_helmholtz_128", help='Output directory')
    args = parser.parse_args()

    #np.random.seed(42)

    # Define domain
    domain = ((args.x_min, args.x_max), (args.y_min, args.y_max))
    
    # Add domain info to filename
    domain_str = f"_domain{args.x_min}-{args.x_max}x{args.y_min}-{args.y_max}"
    
    X, Y = generate_helmholtz_dataset_2d(
        n=args.n_grid_pts,
        n_samples=args.n_samples,
        k=args.k,
        N_modes_fourier=args.N_modes_fourier,
        domain=domain
    )

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dataset_2DHelmholtz_res{args.n_grid_pts}_N{args.n_samples}_k{args.k}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    df = pd.DataFrame({
        'X': list(X),
        'Y': list(Y)
    })
    df.to_parquet(output_path, engine="pyarrow")

    print(f"✅ Dataset saved to '{output_path}'")
    print(f"📊 Preview image saved as 'helmholtz_sample_k{args.k}.png'")


if __name__ == '__main__':
    main()
