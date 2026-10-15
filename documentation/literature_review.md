# Literature Review

## 1. Overview

Low-light image enhancement aims to convert an underexposed image into a
clearer and more visually useful image while retaining scene structure,
texture, and realistic color. The task is difficult because low
illumination often occurs together with noise, low contrast, color
shifts, and loss of fine detail. The ten papers reviewed here show
several main directions: Retinex-based decomposition, unpaired or
zero-reference learning, normalizing flows, transformer architectures,
diffusion models, and more realistic evaluation datasets.

This review is based on the ten papers selected for the project. The
descriptions below summarize their stated methods and contributions. The
limitations are framed cautiously as considerations relevant to this
project, rather than claims that the original authors' methods fail in
every setting.

## 2. Retinex decomposition

### \[1\] Deep Retinex Decomposition for Low-Light Enhancement (2018)

Wei et al. introduce Retinex-Net and a paired low-/normal-light dataset.
The method decomposes an image into reflectance and illumination
components, then enhances illumination while applying denoising to
reflectance. Its important contribution is to make a classical Retinex
idea trainable with deep networks and paired examples.

**Relevance:** It establishes illumination decomposition as a useful way
to represent low-light degradation.

**Consideration:** A decomposition-based system depends on the quality
of the estimated illumination and reflectance. This motivates testing
whether explicit illumination information is helpful when conditioning a
diffusion model.

### \[5\] Retinex Image Enhancement Based on Sequential Decomposition With a Plug-and-Play Framework (2024)

Wu et al. develop a sequential Retinex decomposition framework and
incorporate a CNN-based denoiser in a plug-and-play process to estimate
reflectance. The enhanced result combines the illumination and
reflectance components, including gamma correction. The work emphasizes
the relationship between enhancement, denoising, and interpretable
decomposition.

**Relevance:** It supports the idea that illumination and reflectance
can provide structured information for enhancement.

**Consideration:** A multi-stage or plug-and-play pipeline is different
from a single conditional diffusion model. The project therefore studies
conditioning inside a common generative framework rather than
reproducing this pipeline.

## 3. Learning without paired supervision or reference images

### \[2\] EnlightenGAN (2021)

Jiang et al. propose an unsupervised GAN-based approach that does not
require paired low-/normal-light images. Its design includes
global-local discrimination, self-regularized perceptual losses, and
attention mechanisms. This addresses the practical difficulty of
collecting perfectly aligned paired images.

**Relevance:** It demonstrates that low-light enhancement can be learned
under less restrictive supervision.

**Consideration:** GAN-based unpaired learning and paired conditional
diffusion have different training objectives and failure modes. This
paper is useful background, but it does not directly answer which
conditioning strategy works best within the project's DDPM.

### \[3\] Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE, 2020)

Guo et al. formulate enhancement as estimating image-specific curves
that adjust pixel intensities. The method uses carefully designed
non-reference losses and does not need paired or unpaired normal-light
targets during training.

**Relevance:** Zero-DCE provides an example of a lightweight enhancement
method based on direct intensity transformation rather than image
generation.

**Consideration:** A curve-based method has a different modeling
objective from a diffusion model. Its existence also means that a
diffusion model should not be assumed to outperform simpler methods
without experimental evidence.

## 4. Normalizing flows and transformers

### \[4\] Low-Light Image Enhancement with Normalizing Flow (2022)

Wang et al. apply normalizing flows to low-light image enhancement.
Flow-based models learn an invertible mapping and can model image
distributions through a likelihood-based formulation.

**Relevance:** The paper broadens the generative-model background beyond
GANs and diffusion.

**Consideration:** Normalizing flows and diffusion models have different
sampling, likelihood, and architecture characteristics. The paper does
not isolate the effect of spatial versus illumination-aware conditioning
in the project's DDPM.

### \[6\] Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method (LLFormer, 2023)

Wang et al. introduce a benchmark with 4K and 8K low-light images and
propose LLFormer, a transformer-based enhancement method using
axis-based multi-head self-attention and cross-layer attention fusion.
The work addresses both high-resolution enhancement and evaluation.

**Relevance:** It highlights that model performance should be considered
alongside image resolution, efficiency, and downstream usefulness.

**Consideration:** The project's LOLv2-real experiments use a different
dataset and a conditional diffusion architecture, so LLFormer results
are not a direct apples-to-apples comparison.

### \[7\] Retinexformer (2023)

Cai et al. propose a one-stage Retinex-based framework that estimates
illumination and uses an Illumination-Guided Transformer to model
interactions across regions with different lighting conditions. It aims
to address degradations present in dark images and those introduced
during brightening.

**Relevance:** Retinexformer is particularly relevant because it
explicitly uses illumination information to guide enhancement.

**Consideration:** Illumination guidance is already an established idea.
The project's contribution should therefore be described as an
experimental comparison of conditioning designs in its own diffusion
setup, not as the first use of illumination guidance for low-light
enhancement.

## 5. Diffusion-based enhancement

### \[8\] Diff-Retinex (2023)

Yi et al. combine Retinex decomposition with conditional diffusion.
Their approach uses a Retinex Transformer decomposition network and
multiple generative diffusion paths to address illumination, noise,
color deviation, and loss of scene content.

**Relevance:** This is one of the closest papers to the project because
it already combines Retinex-style physical structure and diffusion-based
image generation.

**Consideration:** Because Diff-Retinex already exists, the project
should not claim that applying diffusion to Retinex decomposition is
itself novel. Instead, it examines controlled alternatives for
conditioning a smaller conditional DDPM and reports their measured
effects.

### \[10\] Diffusion-Based Low-Light Image Enhancement with Color and Luminance Priors (2026 preprint)

Fu et al. describe a conditional diffusion framework with a Structured
Control Embedding Module. It extracts multiple controls, including
illumination, illumination-invariant features, shadow priors, and
color-invariant cues, to guide a U-Net diffusion model.

**Relevance:** This paper is very close to the project's interest in
structured conditioning and indicates that color and illumination priors
are active research directions.

**Consideration:** It is a recent preprint in the reviewed version. Its
claims should be described as the authors' reported findings, not as
independently reproduced results. It also means the project's novelty
must be narrow and precise: the contribution is the specific controlled
experiment performed in this project, not the general idea of
conditioning diffusion with illumination or color priors.

## 6. Evaluation in realistic settings

### \[9\] LoLI-Street: Benchmarking Low-Light Image Enhancement and Beyond (ACCV 2024)

Islam et al. introduce a street-scene benchmark and discuss the
difficulty of enhancing real-world low-light street images, particularly
for computer-vision tasks such as autonomous driving. The paper
emphasizes the need for datasets that reflect realistic street
conditions.

**Relevance:** It reminds us that image quality on one paired benchmark
does not guarantee robustness in all real-world scenes or improvement in
downstream tasks.

**Consideration:** The current project evaluates LOLv2-real, so its
conclusions should be limited to that evaluation setting unless
additional datasets or downstream tasks are tested.

## 7. Synthesis

Across the ten papers, three points are especially relevant.

1.  **Illumination information is useful but not automatically
    sufficient.** Retinex-Net, Retinexformer, and Diff-Retinex all use
    explicit structure related to illumination or reflectance.
2.  **Generative models are already part of low-light enhancement
    research.** Normalizing flows and diffusion-based methods show that
    the field extends beyond direct pixel transformations, but more
    complex models are not guaranteed to outperform simpler baselines in
    every experiment.
3.  **Evaluation must be careful and realistic.** Dataset choice, paired
    references, color fidelity, structural similarity, and real-world
    generalization all affect the conclusions that can be drawn.

These observations motivate a controlled comparison of conditioning
strategies while holding the main diffusion framework and evaluation
protocol as constant as practical.
