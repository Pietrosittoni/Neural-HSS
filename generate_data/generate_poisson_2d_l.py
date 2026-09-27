import numpy as np
from scipy.sparse import diags, kron, eye, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

def l_shape_mask(n):
    """
    Return a boolean mask of shape (n, n) where True indicates points inside the L-shaped domain.
    We consider an L-shape formed by removing the top-right quadrant.
    """
    mask = np.ones((n, n), dtype=bool)
    mask[n//2:, n//2:] = False  # Remove top-right quadrant when viewed with origin='lower'
    return mask

def laplacian_2d_4th_order_lshape(n, h):
    """
    Construct the 4th-order finite difference Laplacian for an L-shaped domain.
    Returns a sparse matrix of size (N_active, N_active) where N_active is the number of points in the domain.
    """
    mask = l_shape_mask(n)
    indices = np.argwhere(mask)
    N_active = len(indices)
    
    idx_map = -np.ones((n, n), dtype=int)
    for k, (i, j) in enumerate(indices):
        idx_map[i, j] = k

    L = lil_matrix((N_active, N_active))
    
    for k, (i, j) in enumerate(indices):
        # Center
        L[k, k] = -60.0
        # Neighbor offsets for 4th-order stencil
        neighbors = [
            (-2,0,-1), (-1,0,16), (1,0,16), (2,0,-1),  # x-direction
            (0,-2,-1), (0,-1,16), (0,1,16), (0,2,-1)   # y-direction
        ]
        for di, dj, val in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and mask[ni, nj]:
                L[k, idx_map[ni, nj]] = val
            # else: outside domain -> Dirichlet zero
    L = L / (12 * h**2)
    return csc_matrix(L)

def generate_dataset_2d_4th_lshape(n, n_samples, N_modes_fourier=20):
    """
    Generate dataset for L-shaped domain using 4th-order finite difference Laplacian.
    """
    h = 1.0 / n
    mask = l_shape_mask(n)
    L = laplacian_2d_4th_order_lshape(n, h)

    indices = np.argwhere(mask)
    N_active = len(indices)

    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))

    for i in range(n_samples):
        f_full = sample_fourier_2d(n, np.random.choice(range(1, N_modes_fourier+1)))
        f_full[~mask] = 0
        f_active = f_full[mask]
        u_active = spsolve(L, f_active)
        u_full = np.zeros((n, n))
        u_full[mask] = u_active
        X[i, :, :] = f_full
        Y[i, :, :] = u_full
        
        # Plot first sample
        if i == 0:
            plt.figure(figsize=(10, 4))
            
            # Create coordinate grids for plotting
            x = np.linspace(0, 1, n, endpoint=False)
            y = np.linspace(0, 1, n, endpoint=False)
            
            plt.subplot(1, 2, 1)
            plt.imshow(f_full, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
            plt.title("Input $f(x,y)$ (L-shape domain)")
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(u_full, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
            plt.title("Solution $u(x,y)$")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig("poisson_lshape_sample.png", dpi=300)
            plt.close()

    return X.reshape(n_samples, n*n), Y.reshape(n_samples, n*n)

def main():
    parser = argparse.ArgumentParser(prog='HSS-learning-2D-Lshape')
    parser.add_argument('--n_grid_pts', type=int, default=64, help='Number of grid points per axis (resolution)')
    parser.add_argument('--n_samples', type=int, default=2500, help='Number of samples to produce')
    parser.add_argument('--N_modes_fourier', type=int, default=10, help='Number of Fourier sine modes')
    parser.add_argument('--output_dir', type=str, default="/home/sittoni/data02/HSS_learning/data_hss_lshape", help='Output directory for dataset')
    args = parser.parse_args()

    #np.random.seed(42)

    X, Y = generate_dataset_2d_4th_lshape(
        n=args.n_grid_pts,
        n_samples=args.n_samples,
        N_modes_fourier=args.N_modes_fourier
    )

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dataset_2DPoisson_Lshape_res{args.n_grid_pts}_N{args.n_samples}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    df = pd.DataFrame({
        'X': list(X),
        'Y': list(Y)
    })
    df.to_parquet(output_path, engine="pyarrow")

    print(f"✅ Dataset saved to '{output_path}' (Parquet format, X and Y as separate columns)")
    print(f"📊 Preview image saved as 'poisson_lshape_sample.png'")

if __name__ == '__main__':
    main()
