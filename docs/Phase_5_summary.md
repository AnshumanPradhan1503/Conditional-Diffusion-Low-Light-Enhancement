# PHASE 5 — EXPERIMENTAL EVALUATION, ABLATION & RESEARCH ANALYSIS

## Project

**Diffusion Models for Low-Light Enhancement**

### Problem Statement

Develop a denoising diffusion probabilistic model pipeline to restore clarity and detail to landscape photographs captured in low-light environments.

---

## 1. Phase 5 Objective

The objective of Phase 5 was to experimentally evaluate the conditional diffusion-based low-light enhancement pipeline and investigate whether different architectural and training modifications could improve restoration quality.

The experiments were designed around the established conditional DDPM framework:

* Low-light image as the conditioning input.
* Normal-light image as the diffusion target.
* Conditional U-Net noise prediction.
* 1000-step diffusion schedule.
* 256 × 256 image processing.
* LOLv2-real official test set.
* PSNR and SSIM as primary quantitative metrics.
* A fixed gamma-enhancement method as a classical reference.
* Controlled experiments with isolated checkpoints and result directories.

The goal was not simply to maximize a single metric, but to determine how different conditioning and architectural strategies affect low-light restoration.

---

# 2. Experimental Control

The original conditional DDPM was maintained as the primary control model.

### Original control

* Architecture: Conditional U-Net
* Base channels: 32
* Parameters: **1,162,947**
* Diffusion timesteps: **1000**
* Image size: **256 × 256**
* Best validation checkpoint: **Epoch 11**
* Official test set: **100 image pairs**
* Random seed: **42**

Experimental variants were stored in separate checkpoint and result directories.

The original checkpoint remained separately stored at:

`checkpoints/best_model.pt`

Experimental checkpoints were stored under their respective experiment directories.

This isolation prevented experimental models from replacing the original control checkpoint.

---

# 3. Classical Reference

A fixed gamma correction baseline with:

**gamma = 0.4**

was evaluated on the same official test set.

Controlled gamma results:

* Mean PSNR: **19.192762 dB**
* Mean SSIM: **0.677547**

The gamma method was used as a classical reference rather than as the target of the research.

The results show that the tested diffusion system did not outperform this simple reference on aggregate PSNR or SSIM.

This observation is important for interpreting the practical performance of the current implementation.

---

# 4. Original Conditional DDPM

The original conditional DDPM established that the diffusion pipeline can perform meaningful low-light enhancement.

An earlier official evaluation reported:

* Mean low-light PSNR: **9.7660 dB**
* Mean low-light SSIM: **0.191124**
* Diffusion PSNR: **15.1016 dB**
* Diffusion SSIM: **0.485159**

This corresponds to:

* PSNR improvement: **+5.3356 dB**
* SSIM improvement: **+0.294035**

The model improved PSNR on **95/100** images and SSIM on **100/100** images in that evaluation.

Subsequent controlled comparisons used the reproduced epoch-11 control checkpoint and produced approximately **15 dB PSNR and 0.48 SSIM**, depending on the controlled evaluation run.

Therefore, the original conditional DDPM provides a valid functional research baseline for the subsequent ablation experiments.

---

# 5. Capacity Experiment

## 48-Channel Conditional U-Net

The base channel capacity was increased from 32 to 48.

### Model

* Parameters: **2,527,395**
* Training duration: short controlled ablation
* Separate checkpoint and evaluation directory

### Test results

* Mean PSNR: **14.8713 dB**
* Mean SSIM: **0.175927**

Compared with the low-light input:

* PSNR improvement: **+5.1053 dB**
* SSIM change: **−0.015197**

The increased model capacity did not produce a corresponding improvement in restoration quality.

This experiment indicates that simply increasing the number of feature channels is not sufficient to improve the current conditional diffusion formulation.

Because this was a shorter ablation rather than a full 20-epoch experiment, it should not be treated as directly equivalent in training duration to the full architectural experiments.

---

# 6. X0 Objective Experiment

An additional x0 reconstruction objective was investigated alongside the diffusion objective.

### Controlled evaluation

* Original control PSNR: **15.033402 dB**
* Original control SSIM: **0.485660**
* X0-objective PSNR: **16.011893 dB**
* X0-objective SSIM: **0.299780**

The X0 objective improved PSNR but substantially reduced SSIM.

Direct comparison:

* Original PSNR wins: **40/100**
* X0 PSNR wins: **60/100**
* Original SSIM wins: **100/100**
* X0 SSIM wins: **0/100**

Therefore, the additional objective produced a metric trade-off rather than a consistent improvement.

It demonstrates that optimizing for improved pixel-level reconstruction does not necessarily produce better structural similarity.

This experiment was retained as an ablation rather than selected as the final training objective.

---

# 7. Naive Multi-Scale Conditioning

A multi-scale conditioning architecture was introduced to provide the low-light condition at multiple spatial resolutions.

### Model

* Parameters: **1,198,979**

### Results

* Mean PSNR: **8.817100 dB**
* Mean SSIM: **0.418816**

Compared with the original control:

* PSNR difference: **−6.216301 dB**
* SSIM difference: **−0.066845**
* Multi-scale PSNR wins: **0/100**
* Original PSNR wins: **100/100**
* Multi-scale SSIM wins: **1/100**
* Original SSIM wins: **99/100**

The naive multi-scale design therefore performed substantially worse than the original conditional U-Net.

This experiment demonstrated that providing conditioning information at multiple resolutions is not automatically beneficial. The manner in which the conditioning information is integrated into the U-Net is important.

---

# 8. Gated Multi-Scale Conditioning

A gated version of the multi-scale architecture was subsequently evaluated.

### Model

* Parameters: **1,198,982**

### Results

* Mean PSNR: **12.561047 dB**
* Mean SSIM: **0.414056**

Compared with the original control:

* PSNR difference: **−2.472355 dB**
* SSIM difference: **−0.071604**
* Gated PSNR wins: **3/100**
* Original PSNR wins: **97/100**
* Gated SSIM wins: **0/100**
* Original SSIM wins: **100/100**

Compared with naive multi-scale:

* PSNR improvement: **+3.743946 dB**
* Gated PSNR wins: **100/100**

The gating mechanism therefore substantially recovered performance relative to the naive multi-scale design, but it did not recover the performance of the original control.

This suggests that controlling the strength of multi-scale conditioning is useful, but the tested gated formulation was still not an improvement over the original architecture.

---

# 9. Spatial Conditioning

Spatial conditioning was introduced to inject the low-light condition into multiple stages of the U-Net while preserving the main backbone structure.

### Model

* Parameters: **1,246,275**

### Results

* Mean PSNR: **14.961661 dB**
* Mean SSIM: **0.496689**

Compared with the original control:

* PSNR difference: **−0.071741 dB**
* SSIM difference: **+0.011029**
* Spatial PSNR wins: **34/100**
* Original PSNR wins: **66/100**
* Spatial SSIM wins: **59/100**
* Original SSIM wins: **41/100**

This experiment produced one of the clearest structural improvements.

Although average PSNR was slightly lower, SSIM improved and the spatial model won SSIM on 59 of the 100 test images.

Further analysis of severely dark images showed a stronger benefit from spatial conditioning in that subset.

Therefore, spatial conditioning provided evidence that the location and resolution at which low-light information is introduced can influence restoration behavior.

---

# 10. Adaptive Spatial Conditioning

Adaptive spatial conditioning was developed to allow the strength of spatial conditioning to vary according to the input.

### Model

* Parameters: **1,249,507**

### Results

* Mean PSNR: **14.357167 dB**
* Mean SSIM: **0.497548**

Compared with the original control:

* PSNR difference: **−0.678012 dB**
* SSIM difference: **+0.011797**
* Adaptive PSNR wins: **26/100**
* Original PSNR wins: **74/100**
* Adaptive SSIM wins: **55/100**
* Original SSIM wins: **45/100**

The adaptive model produced a small additional SSIM improvement over the spatial model, but at the cost of a larger PSNR reduction.

Analysis of the learned gates showed that the conditioning strength was input-dependent, but the variation was relatively small.

Therefore, adaptive spatial conditioning demonstrated learned adaptive behavior but did not provide an overall improvement sufficient to replace the original control.

---

# 11. Illumination-Aware V1

An illumination-aware architecture was introduced using a separate illumination pathway.

The first version substantially altered the underlying network structure.

### Model

* Parameters: **926,979**

### Results

* Mean PSNR: **8.961483 dB**
* Mean SSIM: **0.022517**
* PSNR difference vs original: **−6.073696 dB**
* SSIM difference vs original: **−0.463233**
* Illumination-aware PSNR wins: **0/100**
* Illumination-aware SSIM wins: **0/100**

The model performed poorly.

However, this result cannot be interpreted as evidence that illumination-aware conditioning itself is ineffective because the V1 architecture differed substantially from the original backbone and had significantly fewer parameters.

Therefore, V1 is treated as an architecture-confounded experiment.

No further V1 training was performed.

---

# 12. Illumination-Aware V2

A second illumination-aware architecture was developed to provide a fairer comparison.

The V2 design preserved the original backbone structure while adding a separate low-frequency illumination pathway.

### Model

* Parameters: **1,239,811**
* Best checkpoint: **Epoch 14**
* Best validation loss: **0.020194557225466637**
* Official test images: **100**

The illumination pathway produced features at:

* 256 × 256
* 128 × 128
* 64 × 64

The pathway was therefore active and learned non-trivial representations.

---

# 13. Illumination-Aware V2 Controlled Evaluation

Final controlled evaluation:

### Gamma

* PSNR: **19.192762 dB**
* SSIM: **0.677547**

### Original DDPM

* PSNR: **15.011430 dB**
* SSIM: **0.483198**

### Illumination-Aware V2

* PSNR: **13.273388 dB**
* SSIM: **0.497885**

### V2 − Original

* PSNR: **−1.738042 dB**
* SSIM: **+0.014686**

Per-image:

* V2 PSNR wins: **10/100**
* V2 SSIM wins: **62/100**

The result demonstrates a clear metric trade-off.

V2 improved structural similarity for a substantial portion of the test set but produced lower aggregate pixel-level fidelity according to PSNR.

Therefore, V2 was not selected as a replacement for the original control.

---

# 14. Illumination V2 Brightness Analysis

Brightness-dependent analysis was performed over all 100 official test images.

Results:

* PSNR-delta correlation with brightness: **−0.107745**
* SSIM-delta correlation with brightness: **−0.518254**

The darkest quartile showed:

* Original PSNR: **14.848072 dB**
* V2 PSNR: **13.491151 dB**
* PSNR delta: **−1.356921 dB**
* Original SSIM: **0.419724**
* V2 SSIM: **0.481896**
* SSIM delta: **+0.062172**
* PSNR wins: **4/25**
* SSIM wins: **25/25**

The results indicate that the structural benefit of the illumination-aware formulation was particularly visible in darker images, while PSNR remained lower.

This supports the interpretation that the illumination pathway can influence structural restoration without necessarily improving pixel-level fidelity.

---

# 15. Illumination V2 Visual Analysis

Controlled visual comparisons showed several recurring behaviors.

The original DDPM frequently produced:

* blurred structures
* purple/blue color tendencies
* reduced fine detail

The V2 model frequently produced:

* sharper or more recognizable structural details
* improved structural appearance in some cases
* substantial warm/orange/red color artifacts in other cases
* occasional green or warm color shifts

Thus, visual inspection supported the quantitative observation that V2 could improve structural characteristics while introducing photometric errors.

The controlled visual outputs were preserved rather than regenerated so that the analysis remained tied to the same evaluation run.

---

# 16. Illumination V2 Color / Photometric Analysis

A color analysis was performed on the existing controlled comparison visuals.

Mean absolute channel errors relative to the normal-light target were:

### V2

* R error: **0.295474**
* G error: **0.139449**
* B error: **0.049135**
* Luminance error: **0.175797**
* Mean RGB-channel error: **0.161353**

### Original

* R error: **0.107431**
* G error: **0.060045**
* B error: **0.200475**
* Luminance error: **0.066931**
* Mean RGB-channel error: **0.122650**

The V2 model therefore showed larger average RGB and luminance errors on these inspected cases.

The most notable pattern was a systematic warm/red-channel deviation.

This provides a plausible explanation for why V2 could achieve higher SSIM while simultaneously producing lower PSNR.

---

# 17. Illumination V2 Contribution Analysis

The illumination pathway was inspected directly without modifying or retraining the checkpoint.

Mean illumination-to-backbone activation ratios were approximately:

| Resolution | Mean Ratio |
| ---------- | ---------: |
| 256        | **13.40%** |
| 128        |  **5.05%** |
| 64         | **26.16%** |

The largest relative contribution occurred at the 64-resolution stage.

The contribution also varied with diffusion timestep at the 128 and 64 stages, while the 256-stage contribution remained relatively stable.

These ratios represent activation-magnitude comparisons only.

They **must not be interpreted as causal percentages of the final prediction**.

The analysis confirms that the illumination pathway was active and learned a measurable role, but its learned contribution did not translate into improved aggregate restoration fidelity.

---

# 18. Consolidated Experimental Results

| Experiment                | Parameters | Test PSNR | Test SSIM | Main Observation                              |
| ------------------------- | ---------: | --------: | --------: | --------------------------------------------- |
| Original Conditional DDPM |  1,162,947 |  ~15.0 dB |     ~0.48 | Functional control                            |
| 48-channel capacity       |  2,527,395 |   14.8713 |  0.175927 | More capacity did not improve overall quality |
| X0 objective              |          — | 16.011893 |  0.299780 | PSNR increased but SSIM degraded              |
| Naive multi-scale         |  1,198,979 |  8.817100 |  0.418816 | Poor and unstable                             |
| Gated multi-scale         |  1,198,982 | 12.561047 |  0.414056 | Recovered some performance                    |
| Spatial conditioning      |  1,246,275 | 14.961661 |  0.496689 | Improved SSIM; useful spatial behavior        |
| Adaptive spatial          |  1,249,507 | 14.357167 |  0.497548 | Slight additional SSIM gain, lower PSNR       |
| Illumination V1           |    926,979 |  8.961483 |  0.022517 | Failed; architecture-confounded               |
| Illumination V2           |  1,239,811 | 13.273388 |  0.497885 | SSIM improvement, PSNR reduction              |
| Gamma 0.4                 |          — | 19.192762 |  0.677547 | Classical reference                           |

**Note:** Some experiment summaries were produced in earlier controlled evaluation runs using the reproduced original control with 15.033402 dB / 0.485660 SSIM, while the final V2 controlled evaluation reports 15.011430 dB / 0.483198 SSIM for the same epoch-11 control checkpoint. These values are not silently mixed. Direct V2 comparisons use the V2 controlled evaluation values.

---

# 19. Overall Experimental Interpretation

The experiments demonstrate that the conditional DDPM is capable of performing meaningful low-light enhancement on the LOLv2-real dataset.

However, the experiments also show that increasing architectural complexity does not automatically improve restoration quality.

Three major observations emerged.

### 19.1 Conditioning strategy matters

Naive multi-scale conditioning substantially degraded performance.

Gating partially recovered the degradation, while spatial conditioning produced a more favorable structural result.

This indicates that how the low-light image is injected into the diffusion U-Net is important.

### 19.2 Pixel fidelity and structural similarity can move in different directions

The X0 objective improved PSNR while strongly reducing SSIM.

Spatial and illumination-aware conditioning showed the opposite type of trade-off in several cases: structural similarity improved while PSNR decreased.

Therefore, a single metric does not completely characterize low-light restoration quality.

### 19.3 Illumination information was learned but did not produce aggregate improvement

The V2 illumination pathway was active and made measurable contributions, particularly at the 64-resolution stage.

It also produced higher SSIM on 62/100 images.

Nevertheless, aggregate PSNR decreased by 1.738042 dB relative to the controlled original model.

The tested illumination-aware formulation therefore did not improve overall restoration fidelity despite demonstrating meaningful learned behavior.

---

# 20. Final Phase 5 Conclusion

The Phase 5 experiments establish that a conditional diffusion model can be used for low-light image enhancement, but the tested architectural modifications do not automatically produce better overall restoration quality.

The original conditional DDPM provides a valid research control and demonstrates substantial improvement over the raw low-light images.

Spatial conditioning produced the clearest structural benefit among the tested architectural modifications, particularly for severely dark images, although its average PSNR remained slightly below the control.

The illumination-aware V2 architecture demonstrated that a dedicated illumination pathway can learn non-trivial representations and improve SSIM for a substantial subset of images. However, it also introduced color and luminance errors and reduced aggregate PSNR.

The experiments therefore support the following research conclusion:

> **Conditional diffusion can perform low-light enhancement, but architectural modifications to the conditioning pathway do not automatically improve overall restoration fidelity. Spatial conditioning showed the clearest structural benefit, particularly for severely dark images, while illumination-aware conditioning increased structural similarity for many images but introduced substantial pixel-level and photometric errors.**

The tested gamma correction reference achieved higher aggregate PSNR and SSIM than the diffusion models. This is reported as an experimental reference result rather than as a replacement for the research objective.

---

# 21. Phase 5 Research Significance

Phase 5 provides an empirical foundation for the subsequent research discussion.

The experiments demonstrate:

1. A working conditional DDPM low-light enhancement pipeline.
2. Controlled comparison against a fixed classical enhancement reference.
3. Multiple architecture and objective ablations.
4. Quantitative PSNR and SSIM evaluation.
5. Per-image comparison of competing conditioning strategies.
6. Brightness-dependent analysis.
7. Visual analysis of restoration behavior.
8. Color and photometric error analysis.
9. Internal illumination-pathway contribution analysis.
10. Isolated checkpoints and reproducible experiment directories.

The results also identify several research limitations and future directions without requiring additional model experimentation in the current project.

---

# 22. Phase 5 Completion Status

### Experimental work

**COMPLETED**

### Controlled evaluations

**COMPLETED**

### Ablation experiments

**COMPLETED**

### Illumination-aware diagnostics

**COMPLETED**

### Checkpoint isolation verification

**COMPLETED**

### Result-file inventory verification

**COMPLETED**

### Numerical consolidation

**COMPLETED**

### Model experimentation

**FROZEN**

No additional architecture should be trained as part of the current Phase 5 work unless a genuine verification error is discovered.

---

# 23. Phase 5 Final Status

**PHASE 5 — COMPLETED**

The project can now proceed from experimental development toward final research documentation, interpretation, and project closure.
