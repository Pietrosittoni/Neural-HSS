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

Neural-HSS injects the Hierarchical Semi-Separable (HSS) structure of elliptic Green’s functions into a neural PDE solver. In 3D, each layer is a CP expansion of modal products of one-dimensional HSS factors (paper Eq. 1):

$$H_\theta(Z)=\sum_{k=1}^{r_{\mathrm{out}}} Z \bigtimes_{j=1}^{3} W_j^{(k)}, \qquad W_j^{(k)}\in\mathrm{HSS}(r).$$

The 3D experiment in Section 3.1 trains this model with **16 to 256** samples and **16k** optimization steps (Appendix M, Table 9).

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
