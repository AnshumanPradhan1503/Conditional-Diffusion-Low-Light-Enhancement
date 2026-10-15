# Conditional Diffusion for Low-Light Image Enhancement

**An Experimental Study of Conditioning Strategies**

This research project investigates conditional Denoising Diffusion Probabilistic Models (DDPMs) for low-light image enhancement. It explores whether spatial information, illumination-aware features, and alternative U-Net architectures can improve the reconstruction of images captured under poor lighting conditions.

The project includes a baseline diffusion model, multiple conditioning and architecture variants, quantitative evaluation, and diagnostic experiments.

## Research Objectives

- Implement a conditional DDPM with a U-Net noise-prediction network.
- Establish a baseline for low-light image enhancement.
- Investigate spatial conditioning and illumination-aware conditioning strategies.
- Compare alternative U-Net architectures and training objectives.
- Evaluate enhancement quality using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).
- Compare diffusion-based enhancement with a conventional gamma-correction baseline.
- Analyze the limitations and trade-offs of the investigated approaches.

## Methodology

The project follows an experimental workflow:

1. **Dataset preparation:** Work with paired low-light and normal-light images from the LOLv2-real dataset.
2. **Preprocessing:** Prepare paired images, resize inputs, apply transformations, and handle image normalization.
3. **Baseline implementation:** Train a conditional diffusion model using a U-Net architecture.
4. **Conditioning experiments:** Investigate spatial conditioning and illumination-aware conditioning.
5. **Architecture experiments:** Evaluate alternative U-Net designs, including multiscale, gated, and adaptive spatial variants.
6. **Evaluation:** Measure image quality using PSNR and SSIM and compare the results with a gamma-correction baseline.
7. **Diagnostics:** Analyze conditioning behavior, reconstruction quality, sampling trajectories, and other experimental factors.

## Model Configuration

The primary baseline configuration uses the following settings:

| Parameter | Value |
|---|---|
| Dataset | LOLv2-real |
| Input image size | 256 × 256 |
| Model | Conditional DDPM with U-Net |
| Diffusion timesteps | 1,000 |
| Beta schedule range | 0.0001–0.02 |
| Base channels | 32 |
| Time embedding dimension | 128 |
| Batch size | 2 |
| Training epochs | 20 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Random seed | 42 |
| Training device | CPU |
| Baseline model parameters | 1,162,947 |

These values describe the primary baseline configuration; experimental variants may use different settings.

## Experimental Variants

The repository contains implementations and experiments for several approaches:

- **Baseline DDPM:** Conditional diffusion with a U-Net backbone.
- **Spatial conditioning:** Investigates spatial information as an additional conditioning signal.
- **Illumination-aware conditioning:** Explores conditioning based on illumination-related information.
- **Illumination-Aware V2:** Investigates a revised illumination-aware architecture.
- **Multiscale U-Net:** Explores feature processing at multiple spatial scales.
- **Multiscale gated U-Net:** Investigates gated feature contributions across scales.
- **Adaptive spatial U-Net:** Explores adaptive spatial conditioning.
- **Alternative objective experiments:** Investigate different reconstruction and diffusion objectives.

These are experimental approaches under comparison, not claims that every variant improves performance.

## Quantitative Results

The following results were obtained from the project's reported test evaluations. PSNR is reported in decibels (dB); higher PSNR and SSIM values generally indicate better reconstruction quality.

| Method | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| Low-light input | 9.7660 | 0.1911 |
| Gamma correction (γ = 0.4) | 19.1928 | 0.6775 |
| Original DDPM control — spatial experiment | 15.0334 | 0.4857 |
| Spatial conditioning | 14.9617 | 0.4967 |
| Original DDPM control — V2 experiment | 15.0114 | 0.4832 |
| Illumination-Aware V2 | 13.2734 | 0.4979 |

**Important:** The spatial and V2 results use their respective experimental controls and should be interpreted within those comparisons.

### Interpretation

The reported diffusion-based methods improve on the unprocessed low-light input. However, gamma correction with γ = 0.4 achieves higher aggregate PSNR and SSIM than the listed diffusion results.

Spatial conditioning produces a slightly higher SSIM than its corresponding original DDPM control, while its PSNR is slightly lower. Illumination-Aware V2 also produces a higher SSIM than its corresponding control, but its PSNR is lower.

These findings indicate that the investigated conditioning strategies do not consistently improve all quantitative metrics. They motivate further investigation into conditioning design, reconstruction objectives, perceptual quality, and the relationship between quantitative metrics and visual appearance.

## Repository Structure

```text
Conditional-Diffusion-Low-Light-Enhancement/
├── .gitignore
├── README.md
├── requirements.txt
├── analyze_dataset.py
├── eda_images.py
├── statistical_eda.py
├── docs/
│   ├── Phase_1_Summary.md
│   ├── Phase_2_Summary.md
│   ├── Phase_3_Summary.md
│   ├── Phase_4_Summary.md
│   ├── Phase_5_summary.md
│   └── Phase_6_Summary.md
├── documentation/
│   ├── literature_review.md
│   ├── references.md
│   └── research_gap.md
├── sandbox/
└── src/
    ├── configs/
    ├── diffusion/
    ├── models/
    ├── preprocessing/
    ├── train.py
    ├── evaluate.py
    ├── generate.py
    └── split.py
```

The tree highlights selected project components rather than listing every file. Additional training, evaluation, analysis, and diagnostic scripts may also be present.

## Dataset

This project uses the **LOLv2-real** low-light image enhancement dataset.

The dataset is not included in this GitHub repository. Obtain it from its authorized distribution source and organize it according to the paths and paired-image structure expected by the project's configuration and data-loading code.

Dataset access, directory layout, and split configuration must be verified before running the training pipeline.

## Installation and Usage

The project was developed using Python 3.11 and PyTorch, with the primary baseline trained on a CPU.

### Prerequisites

- Python 3.11
- Git
- A compatible CPU build of PyTorch and torchvision
- The packages listed in `requirements.txt`
- The LOLv2-real dataset, obtained separately

Training diffusion models can be computationally expensive, especially on a CPU.

### 1. Clone the repository

Open PowerShell and run:

```powershell
git clone https://github.com/AnshumanPradhan1503/Conditional-Diffusion-Low-Light-Enhancement.git
cd Conditional-Diffusion-Low-Light-Enhancement
```

### 2. Create an isolated Python environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3. Install CPU-compatible PyTorch

The following command requests CPU wheels for the pinned PyTorch and torchvision versions:

```powershell
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install the remaining dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Verify the installation

```powershell
python -c "import torch, torchvision, numpy, PIL, pandas, matplotlib, scipy, skimage, tqdm; print('Dependencies imported successfully'); print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pip check
```

A successful import check confirms that the listed packages load in the active environment. A clean `pip check` indicates that pip has not detected broken declared dependencies in that environment.

### 6. Prepare the dataset

Obtain the LOLv2-real dataset from its authorized distribution source. Configure the dataset and split paths to match the current data-loading code before running experiments.

### 7. Run an experiment

Run the appropriate training, evaluation, or image-generation script from the repository root. Consult the source code and configuration files for the exact supported commands and required arguments.

Exact execution commands should be confirmed against the current scripts before reproducing an experiment.

## Reproducibility and Limitations

- The dataset, trained checkpoints, and generated results are excluded from the repository to keep the source-code upload manageable.
- Training diffusion models can be computationally expensive, especially on a CPU.
- Experimental variants may use different configurations and should be compared using consistent evaluation protocols.
- PSNR and SSIM do not fully capture perceptual quality, naturalness, color fidelity, or downstream usefulness.
- The reported results should be interpreted as experimental findings, not evidence that diffusion-based enhancement universally outperforms conventional methods.
- Reproducing the reported metrics requires matching the dataset split, preprocessing, model configuration, checkpoint, sampling procedure, and evaluation protocol.

## Documentation

The `docs/` directory contains phase-wise project summaries. The `documentation/` directory contains the literature review, references, and research-gap discussion.

## Future Work

Potential directions for further experiments include:

- Improving conditioning mechanisms and feature fusion.
- Investigating reconstruction objectives and sampling strategies.
- Evaluating perceptual quality alongside PSNR and SSIM.
- Conducting more controlled ablation studies.
- Improving computational efficiency and reproducibility.
- Comparing methods across additional datasets and lighting conditions.

## Author

**Anshuman Pradhan**

GitHub: [@AnshumanPradhan1503](https://github.com/AnshumanPradhan1503)

## Research Integrity

This repository documents an experimental investigation into conditional diffusion for low-light image enhancement. Results are presented with their limitations, including comparisons against a conventional gamma-correction baseline.
