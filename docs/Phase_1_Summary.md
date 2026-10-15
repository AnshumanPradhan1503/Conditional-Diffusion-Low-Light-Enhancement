# PHASE 1 — RESEARCH PROPOSAL AND LITERATURE REVIEW

## Project Title

**Conditional Diffusion for Low-Light Image Enhancement: An Experimental Study of Conditioning Strategies**

---

## 1. Introduction

Low-light image enhancement is an important problem in computer vision and digital image processing. Images captured under insufficient illumination often suffer from low brightness, poor contrast, loss of fine details, color distortion, and increased visibility of noise.

These problems affect the visual quality of photographs and can also reduce the reliability of downstream computer vision applications such as object detection, image recognition, surveillance, and scene understanding.

Traditional enhancement techniques, including histogram equalization, gamma correction, and Retinex-based methods, attempt to improve visibility by modifying image intensity, contrast, or illumination. Although these methods can produce useful enhancement, they may also introduce overexposure, amplify noise, produce unnatural colors, or fail to recover fine details.

Deep learning approaches have introduced more advanced methods for low-light enhancement. Convolutional neural networks, generative adversarial networks, transformers, and generative models have been investigated to improve image restoration.

Among these approaches, diffusion models have attracted research interest because they learn image distributions through a gradual noise addition and denoising process. Conditional diffusion models can use a low-light image as additional information to guide the generation of an enhanced image.

This project investigates conditional diffusion for low-light image enhancement, with particular emphasis on how different conditioning strategies influence restoration quality.

---

## 2. Problem Statement

Low-light photographs frequently contain insufficient illumination, degraded contrast, color inaccuracies, and loss of structural details.

Although diffusion models provide a generative framework for image restoration, their performance depends on how effectively the conditioning information is incorporated into the denoising network.

A basic conditional diffusion model uses the low-light image as a condition while learning to reconstruct its corresponding normal-light image. However, the conditioning information may not be sufficiently utilized at different spatial resolutions or in representing illumination variations.

Consequently, the enhanced images may exhibit blur, color artifacts, incorrect luminance, or loss of fine details.

The problem addressed in this research is:

**To investigate whether different conditioning strategies in a conditional denoising diffusion probabilistic model can improve the structural and photometric quality of low-light image enhancement, and to analyze the trade-offs between pixel-level fidelity and structural similarity.**

The study focuses on experimental comparison rather than assuming that increasing model complexity or introducing additional conditioning information will necessarily improve enhancement quality.

---

## 3. Aim of the Project

The primary aim is to develop and experimentally investigate a conditional diffusion-based low-light image enhancement system using paired low-light and normal-light images.

The research will examine the influence of conditioning strategies on image restoration quality, structural similarity, color consistency, and performance on severely underexposed images.

---

## 4. Research Objectives

The main objectives are:

1. To study existing low-light image enhancement methods, including Retinex-based approaches, CNNs, GANs, transformers, and generative diffusion models.

2. To develop a conditional denoising diffusion probabilistic model (DDPM) for low-light image enhancement.

3. To use low-light images as conditioning inputs and corresponding normal-light images as diffusion targets.

4. To investigate different conditioning strategies, including spatial conditioning and illumination-aware conditioning.

5. To compare the investigated models using quantitative image-quality metrics, particularly Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

6. To analyze enhancement behavior on images with different levels of illumination.

7. To investigate color, luminance, structural, and perceptual artifacts in the generated images.

8. To compare the diffusion-based methods against a fixed classical gamma-correction reference.

9. To identify the limitations and research opportunities associated with conditional diffusion-based low-light enhancement.

---

## 5. Research Questions

The project is guided by the following research questions:

### RQ1 — Conditional Diffusion

Can a conditional DDPM learn to enhance low-light images using paired low-light and normal-light training data?

### RQ2 — Conditioning Strategies

How does the method of incorporating low-light conditioning information into a diffusion U-Net affect image restoration quality?

### RQ3 — Spatial Conditioning

Does injecting conditioning information at multiple spatial resolutions improve structural similarity or restoration quality compared with the original conditional U-Net?

### RQ4 — Illumination-Aware Conditioning

Can explicitly providing illumination-related information improve enhancement performance, particularly for severely underexposed images?

### RQ5 — Evaluation Trade-offs

How do PSNR and SSIM respond to different conditioning strategies, and do improvements in structural similarity correspond to improvements in pixel-level fidelity?

---

## 6. Scope of the Research

The research focuses on paired, real-world low-light image enhancement using conditional diffusion models.

The scope includes:

- Conditional denoising diffusion probabilistic models.
- Low-light image conditioning.
- Spatial and illumination-aware conditioning strategies.
- Image restoration at a fixed spatial resolution.
- Quantitative evaluation using PSNR and SSIM.
- Per-image and illumination-dependent performance analysis.
- Visual inspection of image quality and artifacts.
- Comparison with a fixed gamma-correction reference.

The experimental study uses the LOLv2-real dataset.

The research does not claim that diffusion models are inherently superior to classical enhancement methods. It investigates their behavior and the effects of conditioning strategies under the selected experimental configuration.

---

## 7. Literature Review

The literature review covers ten research papers related to low-light image enhancement, generative restoration, and diffusion-based methods.

### 7.1 Deep Retinex Decomposition for Low-Light Enhancement

The Retinex-Net approach models an image as the combination of reflectance and illumination components.

The method uses decomposition and enhancement networks to estimate reflectance and illumination, improve illumination, and reduce noise.

Its importance to this project is the explicit representation of illumination and reflectance, which motivates investigating illumination-aware conditioning in a diffusion framework.

### 7.2 EnlightenGAN: Deep Light Enhancement without Paired Supervision

EnlightenGAN investigates low-light enhancement using an unpaired generative adversarial network.

The method addresses the difficulty of obtaining perfectly aligned low-light and normal-light image pairs and uses global and local discrimination mechanisms to guide enhancement.

This work provides a generative-learning reference for low-light enhancement and highlights the importance of realistic image appearance.

### 7.3 Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

Zero-DCE estimates image-specific enhancement curves using a deep network.

It relies on carefully designed non-reference losses rather than requiring paired normal-light ground-truth images.

The paper is relevant as a lightweight enhancement approach and as a reference for understanding the limitations and potential of traditional intensity-based enhancement.

### 7.4 Low-Light Image Enhancement with Normalizing Flow (LLFlow)

LLFlow formulates low-light enhancement using conditional normalizing flows.

The model learns a conditional distribution of normal-light images given low-light inputs, allowing it to represent multiple plausible enhanced results.

This work is relevant because it investigates generative restoration and the uncertainty associated with recovering normal-light images from degraded inputs.

### 7.5 Retinex Image Enhancement Based on Sequential Decomposition With Plug-and-Play Framework

This work investigates Retinex-based enhancement using sequential decomposition and a plug-and-play framework.

It provides another example of incorporating image-formation assumptions and illumination-related information into low-light restoration.

The work motivates studying whether explicit illumination information can be incorporated into a learned generative restoration model.

### 7.6 LLFormer: Low-Light Image Enhancement with Transformer-Based Architecture

LLFormer investigates transformer-based low-light enhancement, including efficient attention mechanisms and feature interactions for high-resolution images.

Its focus on spatial information and long-range feature relationships is relevant to the design of conditioning mechanisms within an encoder-decoder restoration architecture.

The work also highlights the importance of evaluating enhancement methods on images with different spatial resolutions.

### 7.7 Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement

Retinexformer combines Retinex-based image modeling with a transformer architecture.

It investigates illumination-guided feature processing to improve low-light image restoration.

Its relevance to this project lies in the integration of illumination information with learned feature representations, which provides motivation for illumination-aware conditioning in diffusion models.

### 7.8 Diff-Retinex: Rethinking Low-light Image Enhancement with a Generative Diffusion Model

Diff-Retinex investigates low-light image enhancement using a generative diffusion framework combined with Retinex-based decomposition.

The approach incorporates illumination and reflectance information into diffusion-based image restoration.

This paper is particularly relevant because it establishes that diffusion-based low-light enhancement and Retinex-related conditioning have already been investigated.

Therefore, the present project does not claim to be the first application of diffusion models or illumination priors to low-light enhancement.

Instead, it focuses on a controlled experimental study of selected conditioning strategies within its own conditional DDPM framework.

### 7.9 LoLI-Street: Benchmarking Low-Light Image Enhancement and Beyond

LoLI-Street introduces a low-light street-scene benchmark and investigates enhancement methods in relation to downstream computer vision tasks.

Its relevance is the importance of evaluating enhancement beyond visual brightness alone.

The paper motivates the need to examine structural preservation and the potential relationship between enhancement quality and downstream scene understanding.

### 7.10 Diffusion-Based Low-Light Image Enhancement with Color and Luminance Priors

This 2026 arXiv preprint investigates diffusion-based enhancement using explicit color and luminance priors.

Its structured conditioning mechanism is closely related to the present project's investigation of illumination-aware conditioning.

Because this work is a recent preprint, its claims should be interpreted according to its publication status and reported experimental evidence.

It further demonstrates that color and illumination priors in diffusion-based low-light enhancement are an active research direction.

---

## 8. Research Gap

The reviewed literature establishes that low-light enhancement has been investigated using Retinex-based decomposition, CNNs, GANs, transformers, normalizing flows, and diffusion models.

Previous diffusion-based studies have also explored Retinex decomposition and explicit illumination, color, or luminance information.

Therefore, the research gap is not the absence of diffusion-based low-light enhancement or the absence of illumination-aware approaches.

The specific gap addressed by this project is the need for a controlled experimental investigation of how selected conditioning strategies affect restoration behavior within a conditional DDPM framework.

In particular, the study examines:

1. The effect of spatial conditioning on structural similarity and pixel-level fidelity.

2. The effect of a dedicated illumination-aware conditioning pathway.

3. The relationship between aggregate PSNR and SSIM and their per-image behavior.

4. Whether conditioning modifications provide measurable benefits on severely underexposed images.

5. The color and luminance errors that may accompany structural improvements.

The project aims to provide empirical evidence about the trade-offs introduced by these conditioning strategies under a consistent dataset and evaluation protocol.

---

## 9. Proposed Methodology

The proposed methodology follows a structured deep-learning workflow.

### Phase 1 — Problem Definition and Literature Review

Define the research problem, objectives, research questions, scope, and experimental direction.

Review existing literature on low-light enhancement and conditional generative models.

### Phase 2 — Dataset Collection and Analysis

Use the LOLv2-real dataset, which contains paired low-light and normal-light images.

Study the dataset structure, image characteristics, illumination distribution, and training, validation, and testing splits.

### Phase 3 — Data Preprocessing

Apply paired image preprocessing.

Prepare images at a fixed resolution, normalize pixel values, and ensure that corresponding low-light and normal-light images undergo consistent transformations.

### Phase 4 — Conditional Diffusion Model Development

Develop a conditional DDPM using a U-Net-based noise prediction architecture.

The low-light image is provided as conditioning information, while the normal-light image is modeled through the diffusion process.

Investigate the original conditional architecture and selected conditioning modifications.

### Phase 5 — Experimental Evaluation and Ablation

Evaluate the trained models using the official test set.

Compare the original conditional model with spatial and illumination-aware conditioning variants.

Use PSNR and SSIM as primary quantitative metrics, supported by visual and per-image analysis.

### Phase 6 — Final Model Selection and Research Interpretation

Consolidate the final experimental results.

Analyze the effects of conditioning strategies, identify performance trade-offs, document limitations, and prepare the research report and presentation.

---

## 10. Dataset

The project uses the LOLv2-real dataset for paired low-light image enhancement.

The dataset contains real low-light photographs and their corresponding normal-light reference images.

The dataset is suitable for supervised conditional restoration because each low-light image has a corresponding normal-light target.

The dataset is divided into training, validation, and testing subsets.

The official test set is kept separate from training and validation to support evaluation on unseen image pairs.

The experimental preprocessing and data-split details are documented in Phase 2.

---

## 11. Proposed Model Framework

The main model is a conditional denoising diffusion probabilistic model.

The framework consists of the following components:

1. **Low-light condition:** The degraded low-light image provides conditioning information to the denoising network.

2. **Forward diffusion:** Gaussian noise is progressively added to the normal-light target during training.

3. **Conditional U-Net:** A U-Net predicts the noise added to the target while using the low-light image as a condition.

4. **Reverse diffusion:** The model progressively removes noise to generate an enhanced image conditioned on the low-light input.

5. **Conditioning experiments:** Spatial and illumination-aware conditioning strategies are investigated to study their influence on restoration quality.

The principal training objective is noise prediction using a mean squared error loss.

The original conditional DDPM serves as the experimental control for the conditioning comparisons.

---

## 12. Evaluation Metrics

The primary quantitative evaluation metrics are:

### 12.1 Peak Signal-to-Noise Ratio (PSNR)

PSNR measures pixel-level reconstruction fidelity relative to the corresponding normal-light reference.

Higher PSNR indicates lower mean squared pixel error under the specified image representation.

### 12.2 Structural Similarity Index Measure (SSIM)

SSIM measures structural similarity between the enhanced image and the normal-light reference.

It considers image luminance, contrast, and structural information.

### 12.3 Additional Analysis

The quantitative evaluation is supplemented by:

- Per-image metric comparisons.
- Visual inspection of enhanced images.
- Analysis by input brightness.
- Color and luminance error analysis.
- Comparison with a fixed gamma-correction reference.

PSNR and SSIM are not treated as complete measures of perceptual quality or human preference.

---

## 13. Expected Outcomes

The expected outcomes of the research are:

1. A functioning conditional diffusion-based low-light enhancement pipeline.

2. An experimental comparison of the original conditional DDPM with selected conditioning strategies.

3. Quantitative measurements of PSNR and SSIM.

4. An analysis of how spatial and illumination-aware conditioning affect image restoration.

5. An investigation of performance on severely underexposed images.

6. Identification of structural, color, and luminance artifacts.

7. A documented experimental record that can support future research into conditional generative image restoration.

The project is intended to produce an empirical understanding of conditional diffusion-based enhancement rather than assume that the proposed architecture will outperform all existing methods.

---

## 14. Significance of the Research

Low-light image enhancement is relevant to computer vision applications that operate under insufficient illumination.

The project contributes an experimental investigation of conditional diffusion models and their conditioning mechanisms.

By comparing structural similarity, pixel-level fidelity, brightness-dependent behavior, and color errors, the study aims to provide a more detailed understanding of the strengths and limitations of diffusion-based enhancement.

The findings may help inform future work on more effective conditioning mechanisms, illumination representations, and restoration objectives.

---

## 15. Limitations and Scope of Conclusions

The conclusions of this study are limited by the selected dataset, computational resources, evaluation metrics, and experimental configurations.

The study uses LOLv2-real and does not establish generalization to every low-light dataset or imaging environment.

PSNR and SSIM do not fully measure perceptual quality, photographic realism, or human preference.

The classical comparison uses a fixed gamma-correction configuration and should not be interpreted as an exhaustive comparison with all traditional enhancement methods.

The study also does not claim that the investigated diffusion model is superior to state-of-the-art enhancement systems.

Its primary contribution is a controlled experimental investigation of selected conditioning strategies and their measured trade-offs.

---

## 16. Phase 1 Conclusion

Phase 1 established the research direction for conditional diffusion-based low-light image enhancement.

The literature review identified existing approaches based on Retinex decomposition, GANs, CNNs, transformers, normalizing flows, and diffusion models.

The review also established that diffusion-based enhancement and illumination-aware conditioning have already been investigated in previous work.

Consequently, the project is framed as an experimental study of conditioning strategies rather than a claim of introducing the first diffusion-based low-light enhancement method.

The research plan defines the use of LOLv2-real, a conditional DDPM, selected conditioning experiments, and quantitative evaluation using PSNR and SSIM.

This foundation supports the subsequent implementation, experimental analysis, final research report, and presentation.

**PHASE 1 — RESEARCH PROPOSAL AND LITERATURE REVIEW: COMPLETED**