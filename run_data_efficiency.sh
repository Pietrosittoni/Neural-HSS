#!/usr/bin/env bash
# Reproduce Figure 5 (left): 3D Poisson data-efficiency sweep.
set -euo pipefail

CONFIG="${1:-configs/poisson3d.yaml}"

for n in 16 32 64 128 256; do
  echo "=== Neural-HSS, train_size=${n} ==="
  python train.py --config "${CONFIG}" --train-size "${n}"
done
