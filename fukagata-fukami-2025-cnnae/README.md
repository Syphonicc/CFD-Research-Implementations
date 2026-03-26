# Nonlinear Flow Field Compression via CNN-Autoencoder
### Reproduction of Fukagata & Fukami (2025)

> **Paper:** *Compressing Fluid Flows with Nonlinear Machine Learning: Mode Decomposition, Latent Modeling, and Flow Control*
> **Authors:** Koji Fukagata (Keio University), Kai Fukami (Tohoku University)
> **Preprint:** [arXiv:2505.00343](https://arxiv.org/abs/2505.00343)
> **Original Code:** [github.com/kfukami/CNNAE_Practice](https://github.com/kfukami/CNNAE_Practice)

---

## Overview

This reproduction implements and validates the core result from Section 2.5 of Fukagata & Fukami (2025): a quantitative comparison between **linear POD (Proper Orthogonal Decomposition)** and a **nonlinear CNN-based Autoencoder (CNN-AE)** for compressing fluid flow fields.

The central claim of the paper is that nonlinear autoencoders outperform linear methods at low latent dimensions by exploiting the nonlinear manifold structure of fluid flow data. This reproduction tests that claim across a latent dimension sweep from 1 to 16, using DNS data of a **laminar cylinder wake at Re_D = 100**.

The work was conducted as part of a research study under the guidance of **Dr. Sachidananda Behera (IIT Hyderabad)**, who requested a formal comparison between nonlinear autoencoder modes and POD modes.

---

## What Was Reproduced

- **CNN-AE architecture** from Table 1 of the paper: 6-layer convolutional encoder with MaxPooling, MLP bottleneck, symmetric decoder with Upsampling layers
- **POD implementation** using snapshot SVD on the same flow field dataset
- **Latent dimension scaling study** across n_ξ ∈ {1, 2, 4, 8, 16}
- **L2 reconstruction error** comparison between POD and CNN-AE at each latent dimension
- **Flow field visualizations** showing original DNS, POD modes, and CNN-AE reconstruction

---

## Key Results

### Reconstruction Error: POD vs CNN-AE

| Latent Dimension (n_ξ) | POD L2 Error | CNN-AE L2 Error | CNN-AE Advantage |
|:---:|:---:|:---:|:---:|
| 1 | 0.7319 | 0.2148 | **3.4× better** |
| 2 | 0.2300 | 0.0490 | **4.7× better** |
| 4 | 0.1291 | 0.0554 | **2.3× better** |
| 8 | 0.0167 | 0.0426 | POD better |
| 16 | 0.0012 | 0.0409 | POD better |

**Key finding:** CNN-AE substantially outperforms POD at low latent dimensions (n_ξ ≤ 4), confirming the paper's central claim. At higher dimensions (n_ξ ≥ 8), POD's linear projection converges faster — consistent with the theoretical argument that POD is optimal among *linear* methods.

The crossover point near n_ξ = 4–8 reflects the practical regime where nonlinear compression provides the most value: when you need the most compact possible representation of a flow field.

---

## Visualizations

### CNN-AE Flow Field Reconstruction

![CNN-AE Reconstruction](results/reconstruction_comparison.png)
*Left: DNS (ground truth) cylinder wake at Re_D = 100. Right: CNN-AE reconstruction...*

### POD Mode Decomposition (k=2)
![POD Modes](results/pic1.png)
*Original fluctuation field alongside 2-mode POD reconstruction...*

### Nonlinear vs Linear Compression Efficiency
![Scaling Study](results/pic2.png)
*L2 reconstruction error vs latent dimension on log scale...*

---

## Implementation Notes

**Base:** Built on the architecture and dataset from [kfukami/CNNAE_Practice](https://github.com/kfukami/CNNAE_Practice)

**Modifications made:**
- Added training checkpoints to handle session crashes (training run on Kaggle GPU)
- Implemented POD from scratch via snapshot SVD for direct comparison
- Built automated latent dimension sweep loop (n_ξ = 1, 2, 4, 8, 16)
- Added L2 error logging and comparison table generation

**Environment:** Kaggle GPU (Tesla P100), TensorFlow/Keras, NumPy, Matplotlib

**Dataset:** Laminar cylinder wake DNS at Re_D = 100 (provided in original repo)

---

## Architecture Summary

```
Encoder:
  Input (384, 192, 2)
  -> Conv2D(16) -> MaxPool -> Conv2D(8) -> MaxPool
  -> Conv2D(8) -> MaxPool -> Conv2D(8) -> MaxPool
  -> Conv2D(4) -> MaxPool -> Conv2D(4) -> MaxPool
  -> Flatten -> Dense(n_ξ)   [latent vector]

Decoder:
  Dense(n_ξ) -> Reshape(6,3,4)
  -> Upsample -> Conv2D(4) -> Upsample -> Conv2D(8)
  -> Upsample -> Conv2D(8) -> Upsample -> Conv2D(8)
  -> Upsample -> Conv2D(16) -> Upsample -> Conv2D(2)
  Output (384, 192, 2)
```

Loss: Mean Squared Error | Optimizer: Adam | Filter size: 3×3

---

## Theoretical Context

POD solves:
```
Γ* = argmin_Γ ||q' - Γᵀ Γ q'||₂
```

CNN-AE solves:
```
Γ* = argmin_Γ ||q' - φ_d Γᵀ φ_e Γ q'||₂
```

The two are **identical when activation functions are linear** (φ_e = φ_d = 1). Nonlinear activations (ReLU, tanh) allow the autoencoder to map data onto a curved manifold rather than a flat hyperplane providing superior compression when the flow's intrinsic dynamics are nonlinear, which for most turbulent and unsteady flows, they are.

---

## Connection to Ongoing Research

This reproduction was the starting point for a formal comparison study requested by Dr. Sachidananda Behera (IIT Hyderabad, Dept. of Mechanical Engineering) examining how nonlinear autoencoder modes relate to POD modes specifically, the paper's result that a single nonlinear AE mode can encode information equivalent to multiple POD modes.

---

## Citation

```bibtex
@article{fukagata2025compressing,
  title={Compressing fluid flows with nonlinear machine learning: 
         mode decomposition, latent modeling, and flow control},
  author={Fukagata, Koji and Fukami, Kai},
  journal={arXiv preprint arXiv:2505.00343},
  year={2025}
}
```

---

## Repository Structure

```
CFD-Research-Implementations/
└── fukagata-fukami-2025-cnnae/
    ├── README.md
    ├── cnnae_reproduction.ipynb     # Main training notebook (Kaggle)
    ├── pod_comparison.ipynb         # POD implementation and comparison
    ├── data/                        # Cylinder wake DNS snapshots
    ├── results/
    │   ├── reconstruction_comparison.png
    │   ├── pic1.png                 # POD mode visualization
    │   ├── pic2.png                 # Scaling study plot
        └── data.txt                 # Numerical error table
    
```
