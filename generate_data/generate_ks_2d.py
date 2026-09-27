from pde import PDE,ScalarField,UnitGrid
import numpy as np
import os
import argparse
import pandas as pd


def sample_fourier_2d(n, N_modes_fourier=10):
    grid = UnitGrid([n, n])
    state = ScalarField.random_colored(grid,exponent=-1.5,scale = 0.5)
    return state


import numba as nb

from pde import PDEBase, ScalarField, UnitGrid


# class KuramotoSivashinskyPDE(PDEBase):
#     """Implementation of the normalized Kuramoto–Sivashinsky equation."""

#     def __init__(self, bc="auto_periodic_neumann"):
#         super().__init__()
#         self.bc = bc

#     def evolution_rate(self, state, t=0):
#         """Implement the python version of the evolution equation."""
#         state_lap = state.laplace(bc=self.bc)
#         state_lap2 = state_lap.laplace(bc=self.bc)
#         state_grad_sq = state.gradient_squared(bc=self.bc)
#         return -state_grad_sq / 2 - state_lap - state_lap2

#     def _make_pde_rhs_numba(self, state):
#         """Nunmba-compiled implementation of the PDE."""
#         gradient_squared = state.grid.make_operator("gradient_squared", bc=self.bc)
#         laplace = state.grid.make_operator("laplace", bc=self.bc)

#         @nb.njit
#         def pde_rhs(data, t):
#             return -0.5 * gradient_squared(data) - laplace(data + laplace(data))

#         return pde_rhs


class KuramotoSivashinskyPDE(PDEBase):
    """Implementation of the normalized Kuramoto–Sivashinsky equation."""

    def __init__(self, bc="auto_periodic_neumann",nu = 0.01):
        super().__init__()
        self.bc = bc
        self.nu = nu

    def evolution_rate(self, state, t=0):
        """Implement the python version of the evolution equation."""
        state_lap = state.laplace(bc="auto_periodic_neumann")
        state_lap2 = state_lap.laplace(bc="auto_periodic_neumann")
        state_grad = state.gradient(bc="auto_periodic_neumann")
        return -(1-self.nu)*state_grad.to_scalar("squared_sum") - self.nu*state_lap - (1-self.nu)*state_lap2


def ks_steady_2d_linear_solver(f, nu=0.01):
    """
    Solve steady-state linearized 2D Burgers' equation:
        nu * (u_xx + u_yy) = f
    with Dirichlet BCs (u=0 at boundary).
    """
    state = f
    # eq = PDE({"u": f"-gradient_squared(u) / 2 - {nu} * laplace(u + laplace(u))"},)  # define the pde
    # eq = PDE({"u": f"u*d_dx(u) + u*d_dy(u) - {nu} * laplace(u)"})
    eq = KuramotoSivashinskyPDE(nu = nu)
    u = eq.solve(state,t_range=20,adaptive = True)
    return u.data

def generate_dataset_2d_ks(n, n_samples, N_modes_fourier=10, nu=0.01):
    X = np.zeros((n_samples, n, n))
    Y = np.zeros((n_samples, n, n))
    i = 0
    while i < n_samples:
        try:
            print(f"generating sample {i}...")
            f = sample_fourier_2d(n, N_modes_fourier)
            u = ks_steady_2d_linear_solver(f, nu=nu)
            X[i] = f.data
            Y[i] = u
            i+=1
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}, retrying...")
            continue
    X = X.reshape(n_samples, n*n)
    Y = Y.reshape(n_samples, n*n)
    return X, Y

def main():
    parser = argparse.ArgumentParser(prog='HSS-learning-2D-Kuramoto-Sivashinsky')
    parser.add_argument('--n_grid_pts', type=int, default=64, help='Number of grid points per axis')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to produce')
    parser.add_argument('--N_modes_fourier', type=int, default=10, help='Number of Fourier sine modes')
    parser.add_argument('--nu', type=float, default=0.001, help='Viscosity')
    parser.add_argument('--output_dir', type=str, default='./data_hss', help='Output directory for dataset')
    args = parser.parse_args()

    np.random.seed(42)
    X, Y = generate_dataset_2d_ks(args.n_grid_pts, args.n_samples, args.N_modes_fourier, nu=args.nu)

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dataset_2DKS_res{args.n_grid_pts}_N{args.n_samples}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    df = pd.DataFrame({'X': list(X), 'Y': list(Y)})
    df.to_parquet(output_path, engine='pyarrow')
    print(f"Dataset saved to '{output_path}'")

if __name__ == '__main__':
    main()