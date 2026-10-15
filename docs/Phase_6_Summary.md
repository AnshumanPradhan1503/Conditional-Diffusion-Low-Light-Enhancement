# Phase 6 — Final Model Selection, Evaluation & Research Interpretation Summary

## 1. Phase Objective

Phase 6 was conducted to finalize the experimental interpretation of the low-light enhancement project after completion of Phase 5.

The objectives were:

* identify and lock the final experimental results;
* verify the final checkpoints and model configurations;
* establish a consistent final evaluation record;
* interpret the effects of spatial and illumination-aware conditioning;
* document the relationship between PSNR and SSIM;
* establish the role of the classical gamma-correction reference;
* identify limitations and threats to validity;
* establish a reproducible final research record.

No further model experimentation was performed during this phase.

---

## 2. Final Models Selected for Documentation

Three model configurations were retained as the principal experimental models:

### 2.1 Original Conditional DDPM

The original conditional diffusion model serves as the primary control architecture.

* Parameters: **1,162,947**
* Best checkpoint: `checkpoints/best_model.pt`
* Best checkpoint epoch: **11**

The model uses the low-light image as the conditioning input while the normal-light target is modeled through the diffusion process.

### 2.2 Spatial Conditional U-Net

The spatial-conditioning model introduces conditioning information at multiple spatial stages of the U-Net.

* Parameters: **1,246,275**
* Checkpoint: `checkpoints/spatial/best_model.pt`

Its purpose was to investigate whether providing the low-light condition at multiple spatial resolutions could improve restoration.

### 2.3 Illumination-Aware V2

Illumination-Aware V2 introduces a separate low-frequency illumination pathway while preserving the original backbone structure more closely than the earlier V1 implementation.

* Parameters: **1,239,811**
* Checkpoint: `checkpoints/illumination_aware_v2/best_model.pt`
* Best checkpoint epoch: **14**
* Best validation loss: **0.020194557225466637**

The illumination pathway was independently verified to be active and non-trivial.

---

## 3. Final Evaluation Protocol

The final evaluation record used the LOLv2-real official test set.

### Evaluation configuration

* Dataset: **LOLv2-real**
* Official test images: **100**
* Image size: **256×256**
* Diffusion timesteps: **1000**
* Random seed: **42**
* Device: **CPU**
* Classical reference: **gamma correction, γ=0.4**

The official test set remained separate from training and validation data.

The final controlled evaluations were retained as paired comparisons so that each conditioning experiment was compared against its corresponding original control under its own documented sampling protocol.

---

## 4. Final Quantitative Results

| Method                             | PSNR (dB) |     SSIM |
| ---------------------------------- | --------: | -------: |
| Low-light input                    |  9.766036 | 0.191124 |
| Gamma γ=0.4                        | 19.192762 | 0.677547 |
| Original DDPM — spatial experiment | 15.033402 | 0.485660 |
| Spatial conditioning               | 14.961661 | 0.496689 |
| Original DDPM — V2 experiment      | 15.011430 | 0.483198 |
| Illumination-Aware V2              | 13.273388 | 0.497885 |

The original DDPM substantially improves over the raw low-light input in the evaluated metrics.

The gamma-correction reference produces higher aggregate PSNR and SSIM than the investigated diffusion models under the selected γ=0.4 configuration.

---

## 5. Spatial Conditioning Findings

The spatial-conditioning experiment produced:

* Original control:

  * PSNR: **15.033402 dB**
  * SSIM: **0.485660**
* Spatial conditioning:

  * PSNR: **14.961661 dB**
  * SSIM: **0.496689**

Difference:

* PSNR: **−0.071741 dB**
* SSIM: **+0.011029**

Per-image comparison:

* Spatial PSNR wins: **34/100**
* Original PSNR wins: **66/100**
* Spatial SSIM wins: **59/100**
* Original SSIM wins: **41/100**

The severe-darkness analysis showed a more meaningful benefit from spatial conditioning for very dark images.

Therefore, spatial conditioning should not be described as an overall improvement in restoration fidelity. Instead, it produced a measurable trade-off between pixel-level and structural similarity.

---

## 6. Illumination-Aware V2 Findings

Illumination-Aware V2 produced:

* Original control:

  * PSNR: **15.011430 dB**
  * SSIM: **0.483198**
* V2:

  * PSNR: **13.273388 dB**
  * SSIM: **0.497885**

Difference:

* PSNR: **−1.738042 dB**
* SSIM: **+0.014686**

Per-image comparison:

* V2 PSNR wins: **10/100**
* V2 SSIM wins: **62/100**

The contribution analysis confirmed that the illumination pathway was active.

Its relative contribution was strongest at the 64×64 resolution stage, where the mean illumination-to-backbone activation RMS ratio was approximately **0.262**.

These measurements demonstrate that the pathway learned non-trivial representations. However, activation contribution does not constitute a causal percentage of final prediction quality.

The controlled evaluation showed that the learned illumination pathway did not improve aggregate PSNR.

---

## 7. PSNR–SSIM Trade-off

An important finding across the experiments is that PSNR and SSIM do not always move in the same direction.

For example, some V2 images showed reduced PSNR while simultaneously showing substantially higher SSIM.

One representative case produced:

* PSNR change: **−2.016515 dB**
* SSIM change: **+0.072877**

This indicates that some generated images can become structurally more similar to the reference while still exhibiting larger pixel-level or photometric differences.

Visual analysis supported this interpretation. Some illumination-aware outputs showed sharper or more recognizable structural content while also exhibiting warm, orange, red, or green color artifacts.

Therefore, neither PSNR nor SSIM alone provides a complete description of restoration quality.

---

## 8. Classical Gamma Reference

The fixed gamma-correction reference with γ=0.4 achieved:

* PSNR: **19.192762 dB**
* SSIM: **0.677547**

This was higher than the evaluated diffusion configurations on the aggregate PSNR and SSIM metrics.

This result demonstrates that a more complex generative model does not automatically outperform a simple photometric enhancement method according to conventional image-similarity metrics.

The gamma experiment should nevertheless be interpreted as a reference configuration rather than proof that γ=0.4 represents every possible classical enhancement method or an optimal classical configuration.

---

## 9. Interpretation of Capacity and Objective Experiments

The capacity experiment increased the model capacity substantially but did not produce a corresponding improvement in restoration quality.

This indicates that simply increasing model capacity is not sufficient to guarantee improved low-light enhancement.

The X0-objective experiment showed an increase in PSNR but a substantial reduction in SSIM. Because it was conducted as a shorter ablation, it should be treated as exploratory evidence rather than as a definitive comparison with the full primary experiments.

Together, these experiments support the broader observation that changing model capacity or optimization objectives can alter different aspects of image quality without producing uniformly better restoration.

---

## 10. Interpretation of Illumination-Aware V1

The first illumination-aware implementation changed more of the backbone architecture than originally intended and substantially reduced the parameter count.

Because of this architectural confound, its poor evaluation result cannot be used as isolated evidence that illumination-aware conditioning itself is ineffective.

The later Illumination-Aware V2 experiment was designed to provide a fairer architectural comparison.

Therefore, V1 is retained as an experimental development result but is not used as the principal evidence against illumination-aware conditioning.

---

## 11. Overall Research Interpretation

The experiments demonstrate that the conditional DDPM learned a meaningful low-light enhancement mapping on the LOLv2-real dataset.

However, the results also demonstrate that architectural modifications to the conditioning pathway do not automatically improve aggregate restoration fidelity.

The main findings are:

1. The original conditional DDPM substantially improves over the raw low-light input.
2. Increasing model capacity alone did not guarantee improvement.
3. Spatial conditioning produced a small structural-similarity benefit while slightly reducing average PSNR.
4. Spatial conditioning showed more useful behavior on severely dark images.
5. Illumination-Aware V2 increased SSIM for many images but reduced aggregate PSNR.
6. Illumination-Aware V2 learned a non-trivial illumination representation, but this did not translate into better aggregate restoration fidelity.
7. Visual and photometric analysis identified color-related artifacts in illumination-aware outputs.
8. Gamma correction with γ=0.4 remained a strong classical reference under the evaluated PSNR and SSIM metrics.
9. PSNR and SSIM sometimes disagree, demonstrating the importance of interpreting multiple metrics together.

---

## 12. Final Scientific Conclusion

The final conclusion of the experimental study is:

> **Conditional diffusion can perform low-light image enhancement, but architectural modifications to the conditioning pathway do not consistently improve aggregate restoration fidelity. Spatial conditioning produced a modest structural benefit, particularly for severely dark images, while illumination-aware conditioning increased structural similarity for many images but introduced substantial pixel-level and photometric errors. The classical gamma baseline remained a strong reference under the evaluated PSNR and SSIM metrics.**

The project therefore does not claim that the investigated diffusion model is superior to classical enhancement methods.

Instead, the project provides a controlled investigation of conditional diffusion-based low-light enhancement and the effects of different conditioning strategies.

---

## 13. Limitations

The study has the following limitations:

### Dataset limitation

The experiments were conducted on LOLv2-real. Therefore, the findings should not automatically be generalized to all low-light photography.

### Computational limitation

The project was developed and trained locally on CPU hardware. This restricted the breadth of model-scale exploration and hyperparameter search.

### Metric limitation

PSNR and SSIM do not completely represent perceptual quality, photographic realism, semantic usefulness, human preference, or natural color appearance.

### Classical-baseline limitation

Only a fixed gamma configuration of γ=0.4 was used as the classical reference.

### Stochastic-sampling limitation

Diffusion sampling is stochastic. Different random-number-generation protocols can produce different aggregate results, which makes the sampling protocol an important part of reproducibility.

### Ablation limitation

Some experiments were exploratory or shorter-duration ablations and should not be interpreted as exhaustive evidence against the investigated alternatives.

---

## 14. Threats to Validity

The principal threats to validity are:

* limited dataset diversity;
* limited official test-set size;
* CPU-based computational constraints;
* incomplete coverage of perceptual evaluation metrics;
* stochastic diffusion sampling;
* use of a single fixed gamma reference;
* limited hyperparameter and architecture search;
* architectural confounding in the first illumination-aware V1 experiment.

These limitations define the scope of the conclusions rather than invalidate the experimental findings.

---

## 15. Reproducibility Record

The final experimental record includes:

* frozen original control checkpoint;
* spatial-conditioning checkpoint;
* illumination-aware V2 checkpoint;
* documented model parameter counts;
* official LOLv2-real test-set evaluation;
* fixed image size;
* fixed diffusion schedule;
* documented random seed;
* controlled evaluation summaries;
* per-image metric CSV files;
* selected visual comparison cases;
* brightness analysis;
* color/photometric analysis;
* illumination contribution analysis;
* documented limitations and interpretation.

The original control checkpoint remained protected during the final evaluation and verification process.

---

## 16. Final Phase 6 Status

### Completed

* Final model selection
* Final checkpoint verification
* Final parameter-count verification
* Final evaluation metadata verification
* Final quantitative verification
* Final visual-case verification
* Research interpretation
* Limitations analysis
* Threats-to-validity analysis
* Reproducibility record

### Experimental status

**Model experimentation is frozen.**

No additional architecture search or retraining is required for completion of the experimental phase.

### Phase 6 conclusion

**Phase 6 — Final Model Selection, Evaluation & Research Interpretation: COMPLETE.**

The project now has a locked experimental record that can be used for the subsequent final documentation, research-paper writing, presentation preparation, and project closure stages.
