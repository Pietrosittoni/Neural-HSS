from posix import truncate
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import os
import argparse
import pandas as pd
import time

# Check if CUDA is available and import CuPy if it is
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve
    CUDA_AVAILABLE = True
    print("CUDA support enabled with CuPy")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA support not available. Using CPU version.")


def sample_fourier_3d(n, N_modes_fourier=10):
    """
    Sample a 3D function as a sum of random Fourier sine modes, with coefficients in [0, 1].
    Scales the output to have reasonable magnitude.
    """
    x = np.linspace(0, 1, n, endpoint=False)
    y = np.linspace(0, 1, n, endpoint=False)
    z = np.linspace(0, 1, n, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    f = np.zeros((n, n, n))
    
    # Use fewer combinations to make computation faster
    num_terms = min(20, N_modes_fourier)
    for _ in range(num_terms):
        kx = np.random.randint(1, N_modes_fourier+1)
        ky = np.random.randint(1, N_modes_fourier+1)
        kz = np.random.randint(1, N_modes_fourier+1)
        coeff = np.random.uniform(0, 1)
        f += coeff * np.sin(kx * np.pi * xx) * np.sin(ky * np.pi * yy) * np.sin(kz * np.pi * zz)
    
    # Scale to have reasonable magnitude (around 1.0)
    f_max = np.max(np.abs(f))
    if f_max > 0:
        f = f / f_max #* 10.0
    
    return f


def laplacian_3d(n, h):
    """
    Construct the 3D finite difference Laplacian operator with Dirichlet BCs.
    Uses standard 7-point stencil.
    Returns a sparse matrix of shape (n³, n³).
    """
    # 1D Laplacian
    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n-1)
    diagonals = [main_diag, off_diag, off_diag]
    L1D = diags(diagonals, [0, -1, 1], shape=(n, n))
    
    # 3D Laplacian
    I = eye(n)
    L3D = kron(kron(I, I), L1D) + kron(kron(I, L1D), I) + kron(kron(L1D, I), I)
    L3D = L3D / (h**2)
    
    return csc_matrix(L3D)


def generate_dataset_3d(n, n_samples, N_modes_fourier=10, use_cuda=False, plot_sample=False, output_dir=None):
    """
    Generate dataset of random Fourier input functions and solutions to 3D Poisson problem
    using finite difference (7-point Laplacian).
    """
    start_time = time.time()
    print(f"Generating {n_samples} samples with resolution {n}x{n}x{n}...")
    h = 1.0 / n
    L3D = laplacian_3d(n, h)
    
    X = np.zeros((n_samples, n, n, n))
    Y = np.zeros((n_samples, n, n, n))
    
    if use_cuda and CUDA_AVAILABLE:
        # Convert to CuPy for GPU acceleration
        L3D_gpu = cp_csr_matrix(L3D)
        print("Using GPU acceleration")
    else:
        print("Using CPU solver")
    
    for i in range(n_samples):
        if (i+1) % 10 == 0 or i == 0:
            print(f"Processing sample {i+1}/{n_samples}")
            
        f = sample_fourier_3d(n, N_modes_fourier)
        
        # Apply Dirichlet BCs: zero out boundary
        f[0, :, :] = 0
        f[-1, :, :] = 0
        f[:, 0, :] = 0
        f[:, -1, :] = 0
        f[:, :, 0] = 0
        f[:, :, -1] = 0
        
        X[i, :, :, :] = f
        
        # Solve Poisson equation
        if use_cuda and CUDA_AVAILABLE:
            f_gpu = cp.array(f.reshape(n**3))
            u_gpu = cp_spsolve(L3D_gpu, f_gpu)
            u = cp.asnumpy(u_gpu)
        else:
            u = spsolve(L3D, f.reshape(n**3))
            
        Y[i, :, :, :] = u.reshape(n, n, n)
        
        # Plot the first sample to verify
        if i == 0 and plot_sample:
            plot_3d_sample(f, u.reshape(n, n, n), n, output_dir=output_dir)
    
    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    
    X = X.reshape(n_samples, n**3)
    Y = Y.reshape(n_samples, n**3)
    
    return X, Y


def laplacian_3d_higher_order(n, h):
    """
    Construct a higher-order (19-point) finite difference 3D Laplacian operator.
    This approximates the Laplacian with higher accuracy but is more computationally expensive.
    """
    N = n**3
    L = lil_matrix((N, N))
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = i * n**2 + j * n + k
                
                # Boundary: set identity (Dirichlet BC)
                if i == 0 or i == n-1 or j == 0 or j == n-1 or k == 0 or k == n-1:
                    L[idx, idx] = 1.0
                else:
                    # Center point
                    L[idx, idx] = -6.0
                    
                    # Six face neighbors (standard 7-point stencil components)
                    if i > 0:
                        L[idx, (i-1) * n**2 + j * n + k] = 1.0
                    if i < n-1:
                        L[idx, (i+1) * n**2 + j * n + k] = 1.0
                    if j > 0:
                        L[idx, i * n**2 + (j-1) * n + k] = 1.0
                    if j < n-1:
                        L[idx, i * n**2 + (j+1) * n + k] = 1.0
                    if k > 0:
                        L[idx, i * n**2 + j * n + (k-1)] = 1.0
                    if k < n-1:
                        L[idx, i * n**2 + j * n + (k+1)] = 1.0
                    
                    # Twelve edge neighbors (to improve accuracy)
                    # Adding these with weight 0.5 improves the stencil accuracy
                    if i > 0 and j > 0:
                        L[idx, (i-1) * n**2 + (j-1) * n + k] = 0.5
                    if i > 0 and j < n-1:
                        L[idx, (i-1) * n**2 + (j+1) * n + k] = 0.5
                    if i < n-1 and j > 0:
                        L[idx, (i+1) * n**2 + (j-1) * n + k] = 0.5
                    if i < n-1 and j < n-1:
                        L[idx, (i+1) * n**2 + (j+1) * n + k] = 0.5
                    
                    if i > 0 and k > 0:
                        L[idx, (i-1) * n**2 + j * n + (k-1)] = 0.5
                    if i > 0 and k < n-1:
                        L[idx, (i-1) * n**2 + j * n + (k+1)] = 0.5
                    if i < n-1 and k > 0:
                        L[idx, (i+1) * n**2 + j * n + (k-1)] = 0.5
                    if i < n-1 and k < n-1:
                        L[idx, (i+1) * n**2 + j * n + (k+1)] = 0.5
                    
                    if j > 0 and k > 0:
                        L[idx, i * n**2 + (j-1) * n + (k-1)] = 0.5
                    if j > 0 and k < n-1:
                        L[idx, i * n**2 + (j-1) * n + (k+1)] = 0.5
                    if j < n-1 and k > 0:
                        L[idx, i * n**2 + (j+1) * n + (k-1)] = 0.5
                    if j < n-1 and k < n-1:
                        L[idx, i * n**2 + (j+1) * n + (k+1)] = 0.5
                    
                    # Adjust center coefficient to maintain consistency
                    L[idx, idx] = -8.0 - 6.0
    
    # Scale by appropriate factor
    L = L / h**2
    return csc_matrix(L)


def generate_dataset_3d_higher_order(n, n_samples, N_modes_fourier=10, use_cuda=False, plot_sample=False, output_dir=None):
    """
    Generate dataset using higher-order (19-point) 3D Laplacian for better accuracy.
    """
    start_time = time.time()
    print(f"Generating {n_samples} samples with resolution {n}x{n}x{n} using higher-order Laplacian...")
    h = 1.0 / n
    L3D = laplacian_3d_higher_order(n, h)
    
    X = np.zeros((n_samples, n, n, n))
    Y = np.zeros((n_samples, n, n, n))
    
    if use_cuda and CUDA_AVAILABLE:
        # Convert to CuPy for GPU acceleration
        L3D_gpu = cp_csr_matrix(L3D)
        print("Using GPU acceleration")
    else:
        print("Using CPU solver")
    
    for i in range(n_samples):
        if (i+1) % 10 == 0 or i == 0:
            print(f"Processing sample {i+1}/{n_samples}")
            
        f = sample_fourier_3d(n, np.random.choice(range(1, N_modes_fourier+1)))
        
        # Apply Dirichlet BCs: zero out boundary
        f[0, :, :] = 0
        f[-1, :, :] = 0
        f[:, 0, :] = 0
        f[:, -1, :] = 0
        f[:, :, 0] = 0
        f[:, :, -1] = 0
        
        X[i, :, :, :] = f
        
        # Solve Poisson equation
        if use_cuda and CUDA_AVAILABLE:
            f_gpu = cp.array(f.reshape(n**3))
            u_gpu = cp_spsolve(L3D_gpu, f_gpu)
            u = cp.asnumpy(u_gpu)
        else:
            u = spsolve(L3D, f.reshape(n**3))
            
        Y[i, :, :, :] = u.reshape(n, n, n)
        
        # Plot the first sample to verify
        if i == 0 and plot_sample:
            print(f"Plotting sample {i+1}/{n_samples}")
            plot_3d_sample(f, u.reshape(n, n, n), n, output_dir=output_dir)
    
    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    
    X = X.reshape(n_samples, n**3)
    Y = Y.reshape(n_samples, n**3)
    
    return X, Y


def plot_3d_sample(input_func, solution, n, output_dir=None):
    """Plot slices of a 3D input function and its solution"""
    # Create a figure with slices through the volume
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    print(input_func.shape)
    print(solution.shape)
    print(n)
    
    # Middle slices through each dimension
    mid_x, mid_y, mid_z = n//2, n//2, n//2
    
    # Input function slices
    im1 = axs[0, 0].imshow(input_func[mid_x, :, :], cmap='viridis')
    axs[0, 0].set_title(f'Input: X-slice at x={mid_x}')
    plt.colorbar(im1, ax=axs[0, 0])
    
    im2 = axs[0, 1].imshow(input_func[:, mid_y, :], cmap='viridis')
    axs[0, 1].set_title(f'Input: Y-slice at y={mid_y}')
    plt.colorbar(im2, ax=axs[0, 1])
    
    im3 = axs[0, 2].imshow(input_func[:, :, mid_z], cmap='viridis')
    axs[0, 2].set_title(f'Input: Z-slice at z={mid_z}')
    plt.colorbar(im3, ax=axs[0, 2])
    
    # Solution slices
    im4 = axs[1, 0].imshow(solution[mid_x, :, :], cmap='plasma')
    axs[1, 0].set_title(f'Solution: X-slice at x={mid_x}')
    plt.colorbar(im4, ax=axs[1, 0])
    
    im5 = axs[1, 1].imshow(solution[:, mid_y, :], cmap='plasma')
    axs[1, 1].set_title(f'Solution: Y-slice at y={mid_y}')
    plt.colorbar(im5, ax=axs[1, 1])
    
    im6 = axs[1, 2].imshow(solution[:, :, mid_z], cmap='plasma')
    axs[1, 2].set_title(f'Solution: Z-slice at z={mid_z}')
    plt.colorbar(im6, ax=axs[1, 2])
    
    plt.tight_layout()
    if output_dir:
        output_file = os.path.join(output_dir, 'poisson_3d_sample.png')
    else:
        output_file = 'poisson_3d_sample.png'
    plt.savefig("poisson_3d_sample.png", dpi=150)
    print(f"Sample visualization saved to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(prog='HSS-learning-3D')
    parser.add_argument('--n_grid_pts', type=int, default=32, help='Number of grid points per axis (resolution)')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to produce')
    parser.add_argument('--N_modes_fourier', type=int, default=10, help='Number of Fourier sine modes')
    parser.add_argument('--higher_order', default=True, help='Use higher-order Laplacian')
    parser.add_argument('--use_cuda', default=False, help='Use CUDA acceleration if available')
    parser.add_argument('--plot_sample', default=True, help='Plot first sample to verify correctness')
    parser.add_argument('--output_dir', type=str, default="/home/_/data02/HSS_learning", help='Output directory for dataset')
    args = parser.parse_args()
    
    np.random.seed(42)
    
    if args.higher_order:
        X, Y = generate_dataset_3d_higher_order(
            n=args.n_grid_pts,
            n_samples=args.n_samples,
            N_modes_fourier=args.N_modes_fourier,
            use_cuda=args.use_cuda,
            plot_sample=args.plot_sample,
            output_dir=args.output_dir
        )
    else:
        X, Y = generate_dataset_3d(
            n=args.n_grid_pts,
            n_samples=args.n_samples,
            N_modes_fourier=args.N_modes_fourier,
            use_cuda=args.use_cuda,
            plot_sample=args.plot_sample,
            output_dir=args.output_dir
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    order_str = "_higher_order" if args.higher_order else ""
    filename = f"dataset_3DPoisson.parquet"
    output_path = os.path.join(args.output_dir, filename)
    
    df = pd.DataFrame({
        'X': list(X),
        'Y': list(Y)
    })
    df.to_parquet(output_path, engine="pyarrow")
    
    print(f"Dataset saved to '{output_path}' (Parquet format, X and Y as separate columns)")


if __name__ == '__main__':
    main()
