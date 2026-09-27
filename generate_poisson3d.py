#!/usr/bin/env python3
"""Generate the 3D Poisson dataset from Appendix F.2.3.

The unit cube is discretized on an n^3 grid. Forcing terms are random
Fourier sine series; solutions of -Δu = f with homogeneous Dirichlet
boundary conditions are obtained from a 19-point finite-difference stencil.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve


def sample_forcing(n: int, n_modes: int = 20, k_min: int = 3, k_max: int = 23) -> np.ndarray:
    """Random Fourier sine series on [0, 1]^3, zero on the boundary."""
    axis = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.zeros((n, n, n), dtype=np.float64)
    for _ in range(n_modes):
        kx, ky, kz = np.random.randint(k_min, k_max + 1, size=3)
        coeff = np.random.uniform(0.0, 1.0)
        field += coeff * np.sin(kx * np.pi * xx) * np.sin(ky * np.pi * yy) * np.sin(kz * np.pi * zz)

    peak = np.max(np.abs(field))
    if peak > 0:
        field /= peak
    field[0] = field[-1] = 0.0
    field[:, 0] = field[:, -1] = 0.0
    field[:, :, 0] = field[:, :, -1] = 0.0
    return field


def laplacian_19point(n: int) -> csc_matrix:
    """Higher-order 19-point 3D Laplacian with Dirichlet identity on the boundary."""
    h = 1.0 / n
    neighbor = diags([np.ones(n - 1), np.ones(n - 1)], offsets=[-1, 1], shape=(n, n))
    identity = eye(n)

    faces = (
        kron(kron(identity, identity), neighbor)
        + kron(kron(identity, neighbor), identity)
        + kron(kron(neighbor, identity), identity)
    )
    edges = (
        kron(kron(identity, neighbor), neighbor)
        + kron(kron(neighbor, identity), neighbor)
        + kron(kron(neighbor, neighbor), identity)
    )
    operator = (faces + 0.5 * edges) / (h**2)
    operator = operator.tolil()
    operator.setdiag(0.0)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = i * n * n + j * n + k
                if i in (0, n - 1) or j in (0, n - 1) or k in (0, n - 1):
                    operator.rows[idx] = [idx]
                    operator.data[idx] = [1.0]
                else:
                    operator[idx, idx] = -14.0 / (h**2)
    return operator.tocsc()


def generate_dataset(n: int, n_samples: int, n_modes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    print(f"Assembling 19-point Laplacian on a {n}x{n}x{n} grid...")
    operator = laplacian_19point(n)
    forcing = np.zeros((n_samples, n, n, n), dtype=np.float32)
    solution = np.zeros((n_samples, n, n, n), dtype=np.float32)

    start = time.time()
    for i in range(n_samples):
        rhs = sample_forcing(n, n_modes=n_modes)
        u = spsolve(operator, rhs.reshape(-1)).reshape(n, n, n)
        forcing[i] = rhs.astype(np.float32)
        solution[i] = u.astype(np.float32)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  sample {i + 1}/{n_samples}")
    print(f"Finished in {time.time() - start:.1f}s")
    return forcing, solution


def save_dataset(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "w") as handle:
            handle.create_dataset("X", data=x, compression="gzip")
            handle.create_dataset("Y", data=y, compression="gzip")
            handle.attrs["grid_size"] = x.shape[-1]
    elif suffix == ".npz":
        np.savez_compressed(path, X=x, Y=y)
    else:
        raise ValueError("Use a .h5 or .npz output path.")
    print(f"Saved {x.shape[0]} samples to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Neural-HSS 3D Poisson dataset.")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid points per axis (paper: 128).")
    parser.add_argument("--n-train", type=int, default=256, help="Training samples.")
    parser.add_argument("--n-test", type=int, default=200, help="Test samples.")
    parser.add_argument("--n-modes", type=int, default=20, help="Random Fourier modes per sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/poisson3d.h5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y = generate_dataset(
        n=args.grid_size,
        n_samples=args.n_train + args.n_test,
        n_modes=args.n_modes,
        seed=args.seed,
    )
    save_dataset(Path(args.output), x, y)


if __name__ == "__main__":
    main()
