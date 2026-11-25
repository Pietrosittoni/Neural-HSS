# Neural-HSS

Official implementation of **Neural-HSS: Hierarchical Semi-Separable Neural PDE Solver**.

[Paper](https://openreview.net/pdf?id=ySwE6xymYR) · ICML 2026

This repository contains the **Neural-HSS** architecture and the **3D Poisson data-efficiency experiment** from the paper: learning the solution operator of

$$-\Delta u = f, \qquad u|_{\partial\Omega}=0$$

on a $128\times128\times128$ grid ($\approx 2\times 10^6$ points).

<p align="center">
  <img src="https://img.shields.io/badge/ICML-2026-0A3D62" alt="ICML 2026"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/>
</p>

## Overview

Neural-HSS injects the Hierarchical Semi-Separable (HSS) structure of elliptic Green’s functions into a neural PDE solver. The same `NeuralHSS` model covers one-dimensional HSS layers and the *m*-dimensional CP expansion (paper Eq. 1):

$$H_\theta(Z)=\sum_{k=1}^{r_{\mathrm{out}}} Z \bigtimes_{j=1}^{m} W_j^{(k)}, \qquad W_j^{(k)}\in\mathrm{HSS}(r), \qquad \theta = (W_j^{(k)})_{j,k}$$

```python
from neural_hss import NeuralHSS

# 1D
model = NeuralHSS(size=256, ndim=1, depth=3, levels=3, rank=2)

# 2D
model = NeuralHSS(size=64, ndim=2, depth=3, levels=2, rank=2, outer_rank=8)

# 3D Poisson experiment (Appendix M, Table 9)
model = NeuralHSS(size=128, ndim=3, depth=1, levels=2, rank=4, outer_rank=2)
```

This repository ships the **3D Poisson data-efficiency experiment** from Section 3.1: **16 to 256** training samples and **16k** optimization steps.

## Installation

```bash
git clone https://github.com/Pietrosittoni/Neural-HSS.git
cd Neural-HSS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Generate the dataset from Appendix F.2.3 (256 train + 200 test, 19-point Laplacian, random Fourier forcing):

```bash
python generate_poisson3d.py \
  --grid-size 128 \
  --n-train 256 \
  --n-test 200 \
  --output data/poisson3d.h5
```

A smaller grid is useful for a sanity check:

```bash
python generate_poisson3d.py --grid-size 32 --n-train 32 --n-test 16 --output data/poisson3d_32.h5
```

## Training

Train Neural-HSS with the paper configuration (Table 9):

```bash
python train.py --config configs/poisson3d.yaml
```

Override the training-set size for the data-efficiency sweep in Figure 5:

```bash
python train.py --config configs/poisson3d.yaml --train-size 16
python train.py --config configs/poisson3d.yaml --train-size 32
python train.py --config configs/poisson3d.yaml --train-size 64
python train.py --config configs/poisson3d.yaml --train-size 128
python train.py --config configs/poisson3d.yaml --train-size 256
```

or run the full sweep:

```bash
bash run_data_efficiency.sh
```

Optional [Weights & Biases](https://wandb.ai) logging is enabled by setting `use_wandb: true` in the config.

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/neural_hss_n256.pt \
  --config configs/poisson3d.yaml \
  --plot outputs/slice.png
```

Reports test MSE and relative $L^2$ error (Appendix G).

## Fast CUDA backbone (optional)

> **Note:** the CUDA backend is a recent addition to this repository and is not part of the original paper. The paper's results were obtained with the PyTorch implementation (`backend: torch`, the default).

The backbone (every HSS layer and its activation) can run on fused CUDA kernels instead of the reference PyTorch layers, in any number of dimensions. Set it in the config, on the command line (`--backend cuda` for `train.py` and `evaluate.py`), or in the constructor; parameters and checkpoints are identical across backends.

```yaml
backend: cuda        # torch (default) | cuda | cuda-tf32
```

```python
model = NeuralHSS(size=128, ndim=3, depth=1, levels=2, rank=4, outer_rank=2, backend="cuda")
```

How it works (`neural_hss/csrc/hss_cuda.cu`, `neural_hss/cuda_backbone.py`): the HSS factors are assembled into dense matrices in one kernel launch (large blocks are split over many thread blocks), and every mode product of Eq. (1), forward and backward, runs as a strided GEMM on tensor cores that reads and writes the activations in place, with no permute copies. The CP terms share launches, bias and activation are fused into the last GEMM, and long reductions (weight gradients, small batches) are split across the GPU. `cuda` splits each operand into two TF32 parts (3xTF32) and sums every 32-wide slice of the reduction in FP32 outside the tensor cores, whose accumulator rounds toward zero and would otherwise drift with the reduction length. `cuda-tf32` uses a single TF32 pass.

Time per training step (forward + MSE + backward + clipping + AdamW, back to back) / inference, in ms, on an A100 80GB PCIe (`torch` with PyTorch's default FP32 matmuls). The paper models are those of the `NeuralHSS` docstring (1D: depth 3, levels 3, rank 2; 2D: depth 2, levels 2, rank 2, $r_{\mathrm{out}}=8$; 3D: depth 1, levels 2, rank 4, $r_{\mathrm{out}}=2$); the other rows use the same layers at other sizes:

| Configuration | `torch` | `cuda` | `cuda-tf32` |
| --- | --- | --- | --- |
| 1D heat (paper model), $N=256$, batch 16 | 17.6 / 3.73 | 6.20 / 0.52 (2.8× / 7.2×) | 6.16 / 0.52 (2.9× / 7.2×) |
| 1D, $N=4096$, paper 1D layers, batch 64 | 21.9 / 4.16 | 7.17 / 1.22 (3.1× / 3.4×) | 7.05 / 1.09 (3.1× / 3.8×) |
| 2D Poisson (paper model), $64^2$, batch 16 | 104 / 21.8 | 13.3 / 0.67 (7.9× / 33×) | 13.0 / 0.65 (8.0× / 33×) |
| 2D, $256^2$, paper 2D layers with $r_{\mathrm{out}}=4$, batch 16 | 53.0 / 10.7 | 7.88 / 0.70 (6.7× / 15×) | 7.88 / 0.56 (6.7× / 19×) |
| 3D, $64^3$, paper 3D layers, batch 16 | 29.1 / 6.08 | 4.02 / 0.65 (7.2× / 9.4×) | 4.02 / 0.41 (7.2× / 15×) |
| **3D Poisson, $128^3$, paper configuration, batch 16** | 84.9 / 26.7 | **12.1 / 4.13 (7.0× / 6.5×)** | **7.67 / 2.58 (11× / 10×)** |
| 4D, $32^4$, depth 1, levels 2, rank 2, $r_{\mathrm{out}}=2$, batch 4 | 43.2 / 9.43 | 4.89 / 1.24 (8.8× / 7.6×) | 4.85 / 0.79 (8.9× / 12×) |

At $128^3$ the GPU is busy 95% of the step and the GEMMs run at about 70% of the card's sustained TF32 rate. In the small 1D and 2D models the CUDA training step is mostly CPU time spent on the hundreds of small HSS parameter tensors (autograd, `clip_grad_norm_`, AdamW), which costs the same with either backend; `torch.optim.AdamW(..., fused=True)` cuts the optimizer part by about 3×.

## Paper settings (3D Poisson)

| Setting | Value |
| --- | --- |
| Grid | $128^3$ |
| Train / test | 16–256 / 200 |
| Depth | 1 |
| Levels | 2 |
| Rank | 4 |
| Outer rank $r_{\mathrm{out}}$ | 2 |
| Activation | Trainable LeakyReLU |
| Optimizer | AdamW, $(\beta_1,\beta_2)=(0.9,0.99)$ |
| Learning rate | $5\cdot 10^{-3}\to 1\cdot 10^{-3}$ (cosine) |
| Weight decay | 0 |
| Gradient clip | 1 |
| Steps | 16k |
| Batch size | 16 |

## Repository layout

```
neural_hss/              Neural-HSS model, data loader, metrics
neural_hss/csrc/         CUDA kernels for the fast backbone
tests/                   CUDA backbone vs. an FP64 run of the PyTorch reference
configs/poisson3d.yaml   Paper hyperparameters
generate_poisson3d.py    Dataset generation (Appendix F.2.3)
train.py                 Training entry point
evaluate.py              Test-set evaluation
```

## Citation

```bibtex
@inproceedings{sittoni2026neuralhss,
  title     = {Neural-HSS: Hierarchical Semi-Separable Neural PDE Solver},
  author    = {Sittoni, Pietro and Zangrando, Emanuele and Casulli, Angelo Alberto and Guglielmi, Nicola and Tudisco, Francesco},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
