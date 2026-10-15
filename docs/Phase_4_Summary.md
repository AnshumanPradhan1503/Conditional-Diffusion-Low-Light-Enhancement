# Phase 4 — Diffusion Model Development and Experimental Analysis

## 1. Phase Overview

Phase 4 focused on developing the core conditional diffusion model for the low-light image enhancement project and experimentally analyzing its behavior.

The objective of this phase was to move from the preprocessing pipeline developed in Phase 3 to an actual diffusion-based image restoration system.

The model was designed as a **conditional Denoising Diffusion Probabilistic Model (DDPM)** in which:

* The low-light image acts as the conditioning image.
* The normal-light image acts as the target image.
* Gaussian noise is progressively added to the normal-light target during forward diffusion.
* A conditional U-Net learns to predict the noise added at each diffusion timestep.
* During inference, the model performs reverse diffusion while using the low-light image as a condition.
* The final generated image is intended to approximate the corresponding normal-light reference.

The phase also included controlled experiments investigating:

1. The diffusion noise schedule.
2. Forward diffusion.
3. Reverse diffusion.
4. Conditional U-Net architecture.
5. The epsilon-prediction training objective.
6. Training configuration.
7. Full model training.
8. Image generation.
9. Diffusion diagnostics.
10. Quantitative evaluation.
11. Classical gamma-correction baseline comparison.
12. Model-capacity analysis.
13. Training-objective analysis.
14. An ablation experiment using an additional X0 reconstruction loss.

---

# 2. Conditional Diffusion Formulation

## 2.1 Problem Formulation

The project uses paired low-light and normal-light images.

Let:

* `x` = low-light input image
* `y0` = corresponding normal-light target
* `yt` = noisy version of the normal-light target at timestep `t`
* `epsilon` = Gaussian noise
* `epsilon_theta` = noise predicted by the neural network

The model learns:

```text
epsilon_theta(yt, x, t)
```

where the network receives:

1. The noisy target image.
2. The original low-light image as conditioning information.
3. The diffusion timestep.

The purpose of the conditional model is to learn how the low-light observation can guide restoration toward the corresponding normal-light appearance.

---

# 3. Conditional U-Net Design

The main model uses a conditional U-Net architecture.

## 3.1 Input Configuration

The model receives:

```text
Noisy target image : 3 channels
Low-light condition: 3 channels
--------------------------------
Total input        : 6 channels
```

The network predicts:

```text
Predicted noise: 3 channels
```

The images are processed at:

```text
256 × 256
```

with RGB channels.

## 3.2 Main Architecture

The U-Net contains:

* Sinusoidal timestep embedding.
* Residual blocks.
* Group Normalization.
* SiLU activation.
* Downsampling layers.
* Bottleneck processing.
* Upsampling layers.
* Skip connections.
* Timestep conditioning through learned projections.

The primary model configuration is:

```text
Base channels        : 32
Time embedding       : 128
Image channels       : 3
Condition channels   : 3
Parameters            : 1,162,947
```

The architecture was verified using forward and backward passes.

Verification confirmed:

* Correct output shape.
* Finite forward outputs.
* Gradient propagation.
* Finite gradients.

---

# 4. Diffusion Noise Schedule

A linear beta schedule was implemented.

Configuration:

```text
Diffusion timesteps : 1000
Beta start          : 0.0001
Beta end            : 0.02
```

The schedule produces:

```text
alpha_t = 1 - beta_t
```

and:

```text
alpha_bar_t = cumulative product(alpha_t)
```

Important values obtained during verification:

```text
beta[0]       = 0.0001
beta[499]     ≈ 0.010040
beta[999]     = 0.0200

alpha[0]      = 0.9999
alpha[999]    = 0.9800

alpha_bar[0]   ≈ 0.9999
alpha_bar[499] ≈ 0.078587
alpha_bar[999] ≈ 0.000040
```

This schedule allows the target image to transition gradually from a nearly clean image to an approximately Gaussian-noise distribution.

The schedule implementation passed numerical verification.

---

# 5. Forward Diffusion

The forward diffusion process was implemented using the standard DDPM formulation.

The noisy target is generated from:

```text
y_t =
sqrt(alpha_bar_t) * y_0
+
sqrt(1 - alpha_bar_t) * epsilon
```

where:

```text
epsilon ~ N(0, I)
```

The implementation supports:

* Random Gaussian noise generation.
* User-supplied noise.
* Batch-specific timesteps.
* Correct broadcasting of diffusion coefficients.

Verification was performed at several timesteps.

Observed behavior:

```text
t = 0
→ image remains almost clean

t = 100
→ moderate noise

t = 500
→ strong degradation

t = 999
→ image approaches a noise-dominated representation
```

The forward diffusion implementation passed:

* Shape verification.
* Determinism checks using supplied noise.
* Numerical stability checks.
* No NaN/Inf checks.

---

# 6. Reverse Diffusion

The reverse diffusion process reconstructs an image from noise while conditioning on the low-light input.

The implementation contains:

* Predicted X0 reconstruction.
* Posterior mean calculation.
* Posterior variance calculation.
* Single reverse sampling step.
* Complete 1000-step reverse diffusion.

The sampler also supports:

```text
clip_x_start = True
```

which constrains reconstructed X0 values to:

```text
[-1, 1]
```

This matches the image normalization used in the preprocessing pipeline.

The reverse sampler was additionally modified to accept an optional random generator. This allowed controlled experiments in which different models received exactly the same random noise.

This became important for the later objective comparison experiment.

---

# 7. Diffusion Training Objective

The original model uses the standard epsilon-prediction objective.

During training:

1. A normal-light target image is selected.
2. A random timestep is sampled.
3. Gaussian noise is generated.
4. Forward diffusion produces a noisy target.
5. The conditional U-Net receives:

   * noisy target,
   * low-light image,
   * timestep.
6. The network predicts the noise.
7. Mean squared error is calculated between:

   * true noise,
   * predicted noise.

The training loss is therefore:

```text
L_epsilon =
MSE(epsilon_predicted, epsilon_true)
```

The initial model was intentionally kept simple by using pure epsilon MSE before introducing experimental modifications.

---

# 8. Training Configuration

The primary training configuration was:

```text
Model                 : Conditional U-Net
Base channels         : 32
Parameters            : 1,162,947

Image size            : 256 × 256
Batch size            : 2
Timesteps             : 1000

Beta start            : 0.0001
Beta end              : 0.02

Learning rate         : 1e-4
Weight decay          : 1e-4

Optimizer             : AdamW
Gradient clipping     : 1.0

Epochs                : 20

Validation frequency  : Every epoch
Checkpoint frequency  : Every 5 epochs

Device                : CPU
Random seed            : 42
```

The training pipeline also supports:

* Checkpoint saving.
* Latest-model checkpoint.
* Best-model checkpoint.
* Periodic checkpoints.
* Training history.
* Resume training.
* RNG-state restoration.
* Training-loss plots.

---

# 9. Training Verification

Before full training, a smoke test was performed.

The smoke test successfully executed:

```text
Training step
Validation step
Loss calculation
Checkpoint generation
```

This confirmed that the complete training pipeline was operational before beginning the expensive full training process.

---

# 10. Full 20-Epoch Training

The main 32-channel model was trained for 20 epochs.

The training results were:

|  Epoch | Training Loss | Validation Loss |
| -----: | ------------: | --------------: |
|      1 |      0.278084 |        0.077413 |
|      2 |      0.070854 |        0.078937 |
|      3 |      0.060467 |        0.085614 |
|      4 |      0.049886 |        0.062185 |
|      5 |      0.057181 |        0.067020 |
|      6 |      0.044614 |        0.049394 |
|      7 |      0.036016 |        0.051509 |
|      8 |      0.039096 |        0.040654 |
|      9 |      0.036551 |        0.039863 |
|     10 |      0.029224 |        0.027245 |
| **11** |  **0.029231** |    **0.022894** |
|     12 |      0.031169 |        0.029724 |
|     13 |      0.030125 |        0.040232 |
|     14 |      0.025903 |        0.038617 |
|     15 |      0.021710 |        0.039950 |
|     16 |      0.028208 |        0.030928 |
|     17 |      0.023659 |        0.029753 |
|     18 |      0.026761 |        0.038198 |
|     19 |      0.024578 |        0.024690 |
|     20 |      0.022141 |        0.031918 |

The best validation loss was:

```text
0.022894
```

at:

```text
Epoch 11
```

Therefore, the primary model used for subsequent evaluation was:

```text
checkpoints/best_model.pt
```

which corresponds to epoch 11.

The final epoch-20 model was retained separately as:

```text
checkpoints/latest_model.pt
```

---

# 11. Training Behavior

The training curve showed a substantial reduction in training loss during the first several epochs.

Validation loss also decreased overall, although it fluctuated between epochs.

The best validation performance occurred at epoch 11.

After epoch 11, validation loss increased and fluctuated rather than producing a consistent improvement.

This supports the use of the best-validation checkpoint instead of automatically using the final epoch.

Importantly, the training loss alone was not treated as sufficient evidence of image-enhancement quality. Image generation and independent PSNR/SSIM evaluation were therefore performed.

---

# 12. Actual Image Generation

The trained model was used to generate images from the official LOLv2-real test set.

Generation pipeline:

```text
Low-light test image
        ↓
Deterministic center crop
        ↓
Normalization to [-1,1]
        ↓
Conditional U-Net
        ↓
1000-step reverse diffusion
        ↓
Generated enhanced image
        ↓
RGB PNG
```

The following outputs were generated:

```text
results/generated/low_000.png
results/generated/generated_000.png
results/generated/high_000.png
```

The generated image demonstrated that the model had learned meaningful scene structure.

However, visual artifacts were clearly present, including:

* Blur.
* Color distortion.
* Purple/blue/red artifacts.
* Loss of fine texture.

Therefore, the model was functioning as an enhancement system, but image quality was still limited.

---

# 13. Diffusion Diagnostic Analysis

A timestep-wise diagnostic was performed to understand model behavior at different diffusion stages.

For one diagnostic experiment:

```text
t = 0
noise MSE = 1.194216
x0 MSE    = 0.000119

t = 100
noise MSE = 0.138188
x0 MSE    = 0.016183

t = 500
noise MSE = 0.045710
x0 MSE    = 0.364584

t = 999
noise MSE = 0.045286
x0 MSE    = 1.082504
```

The high X0 error at large timesteps does not automatically indicate a broken sampler.

When alpha-bar becomes very small, reconstructing X0 from predicted noise becomes highly sensitive to prediction errors.

Therefore, the diagnostic demonstrated an important characteristic of the epsilon-prediction formulation:

> Noise prediction can become highly accurate at large timesteps while direct X0 reconstruction becomes increasingly sensitive to small epsilon-prediction errors.

---

# 14. Controlled Reverse-Diffusion Reconstruction

A controlled reverse-diffusion experiment was also performed.

A known noisy target was created from an actual normal-light image and then passed through the complete reverse process.

The final reconstruction produced:

```text
Final reconstruction MSE:
0.071897
```

The reconstruction preserved recognizable scene structure, although it still contained:

* Blur.
* Color artifacts.
* Imperfect fine details.

This experiment provided evidence that:

1. Forward diffusion was functioning.
2. Reverse diffusion equations were functioning.
3. The trained model was capable of meaningful noise prediction.
4. The conditional information was being used.
5. The main limitation was image quality rather than an obvious catastrophic implementation failure.

---

# 15. Quantitative Evaluation

The official test set contains:

```text
100 paired images
```

The primary model was evaluated against the original low-light images.

Metrics:

* PSNR.
* SSIM.

---

## 15.1 Single-Image Evaluation

For the first evaluated image:

### Low-light baseline

```text
PSNR : 7.4184 dB
SSIM : 0.156276
```

### Diffusion model

```text
PSNR : 17.1403 dB
SSIM : 0.543021
```

Improvement:

```text
PSNR : +9.7219 dB
SSIM : +0.386744
```

This demonstrated that the trained diffusion model could produce a substantial improvement over the original low-light image on an individual test case.

---

# 16. Full 100-Image Evaluation

The complete official test set was evaluated.

The results were:

| Metric        | Low-Light Input | Diffusion Model |
| ------------- | --------------: | --------------: |
| **Mean PSNR** |       9.7660 dB |  **15.1016 dB** |
| **Mean SSIM** |        0.191124 |    **0.485159** |

Mean improvements:

```text
PSNR : +5.3356 dB
SSIM : +0.294035
```

The diffusion model improved:

```text
PSNR : 95 / 100 images
SSIM : 100 / 100 images
```

Therefore, the model demonstrated measurable enhancement capability across the majority of the official test set.

---

# 17. Test-Set PSNR Analysis

For the original low-light inputs:

```text
Mean   : 9.766036 dB
Median : 9.707196 dB
Std    : 1.956409
Min    : 4.989546 dB
Max    : 13.482776 dB
```

For the diffusion model:

```text
Mean   : 15.101587 dB
Median : 14.981737 dB
Std    : 1.670767
Min    : 11.348531 dB
Max    : 19.037417 dB
```

PSNR improvement statistics:

```text
Mean   : +5.335550 dB
Median : +5.415445 dB
Std    : 3.220541 dB
Min    : -1.360332 dB
Max    : +11.731500 dB
```

The model produced a substantial average PSNR improvement, although performance varied between individual images.

---

# 18. Test-Set SSIM Analysis

For the low-light images:

```text
Mean   : 0.191124
Median : 0.171134
Std    : 0.108111
```

For the diffusion model:

```text
Mean   : 0.485159
Median : 0.489408
Std    : 0.071347
```

SSIM improvement:

```text
Mean : +0.294035
```

SSIM improved for:

```text
100 / 100 images
```

This was particularly important because it showed that the model consistently improved structural similarity relative to the original low-light inputs.

---

# 19. Diffusion Model Failure Cases

Although the diffusion model improved most test images, PSNR decreased for five images.

The five PSNR failures were:

```text
low00763
low00724
low00725
low00769
low00780
```

Their PSNR improvements ranged from approximately:

```text
-1.36 dB to -0.54 dB
```

However, these images still showed positive SSIM improvements.

This indicates that failure in PSNR did not necessarily mean complete failure in structural restoration.

---

# 20. Brightness and Model Performance

Brightness analysis was performed on the 100 low-light test images using the same 256×256 center crop.

Statistics:

```text
Mean brightness : 13.66 / 255
Minimum         : 3.33 / 255
Maximum         : 33.19 / 255
```

Correlation between input brightness and PSNR improvement:

```text
r ≈ +0.108
```

This is a very weak relationship.

Correlation between brightness and SSIM improvement:

```text
r ≈ -0.614
```

This indicates a moderate negative relationship in this dataset: darker inputs tended to experience larger SSIM improvements.

However, correlation does not establish causation.

Therefore, the experiment does not support the conclusion that darker images automatically produce better PSNR performance.

---

# 21. Classical Gamma-Correction Baseline

To establish a meaningful non-deep-learning baseline, gamma correction was evaluated.

Candidate gamma values were:

```text
0.4
0.5
0.6
0.7
0.8
```

The validation set was used to select the gamma value in order to avoid test-set leakage.

The best validation value was:

```text
gamma = 0.4
```

Validation performance:

```text
Mean PSNR : 12.6594 dB
Mean SSIM : 0.424038
```

Gamma = 0.4 was therefore fixed before evaluation on the official test set.

---

# 22. Gamma-Correction Test Results

On the official 100-image test set:

### Low-light baseline

```text
PSNR : 9.7660 dB
SSIM : 0.191124
```

### Gamma correction

```text
PSNR : 19.1928 dB
SSIM : 0.677547
```

Improvement:

```text
PSNR : +9.4267 dB
SSIM : +0.486424
```

Gamma correction improved:

```text
PSNR : 100 / 100
SSIM : 100 / 100
```

---

# 23. Diffusion vs Gamma Baseline

The comparison revealed an important result.

| Method           |   Mean PSNR |    Mean SSIM |
| ---------------- | ----------: | -----------: |
| Low-light input  |      9.7660 |     0.191124 |
| Diffusion        |     15.1016 |     0.485159 |
| Gamma correction | **19.1928** | **0.677547** |

Gamma correction was the stronger overall method on this dataset under the current experimental configuration.

Across the 100 test images:

```text
PSNR:
Gamma wins      : 70 / 100
Diffusion wins  : 30 / 100

SSIM:
Gamma wins      : 97 / 100
Diffusion wins  : 3 / 100
```

Therefore:

> The diffusion model should not be claimed to outperform gamma correction overall.

However, the diffusion model was competitive on a subset of particularly challenging dark images.

For example, several images showed substantially larger PSNR gains from diffusion than gamma correction.

This indicates that the diffusion model has potential beyond simple brightness correction, but its current implementation does not yet outperform the classical baseline consistently.

---

# 24. Visual Comparison with Gamma Correction

Visual inspection of the generated results showed:

### Gamma correction

Generally produced:

* Cleaner appearance.
* Stronger similarity to reference images.
* Better color consistency.
* Less visible artifacting.

### Diffusion model

Successfully recovered:

* Scene structure.
* Overall brightness.
* Some hidden visual information.

But frequently exhibited:

* Purple/blue/red mottling.
* Blur.
* Color inaccuracies.
* Loss of fine texture.

This visual evidence was consistent with the quantitative PSNR/SSIM comparison.

---

# 25. Model Capacity Experiment

The original U-Net was evaluated for architectural capacity.

Three configurations were considered:

```text
Base 32 channels : 1,162,947 parameters
Base 48 channels : 2,527,395 parameters
Base 64 channels : 4,424,835 parameters
```

A controlled 48-channel experiment was performed.

The 48-channel model passed architectural verification.

Configuration:

```text
Parameters : 2,527,395
Epochs     : 3
Batch size : 2
```

Training results:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |      0.187358 |    **0.059414** |
|     2 |      0.059225 |        0.059567 |
|     3 |      0.046445 |        0.067611 |

The best validation loss was:

```text
0.059414
```

at epoch 1.

---

# 26. 48-Channel Model Evaluation

The 48-channel model was evaluated on the same official test set.

Results:

```text
Mean PSNR : 14.8713 dB
Mean SSIM : 0.175927
```

Compared with the original 32-channel model:

```text
32-channel:
PSNR = 15.1016 dB
SSIM = 0.485159

48-channel:
PSNR = 14.8713 dB
SSIM = 0.175927
```

Difference:

```text
PSNR : -0.2303 dB
SSIM : -0.309232
```

Therefore:

> Simply increasing U-Net capacity did not improve enhancement quality under the current training configuration.

The 32-channel model was consequently retained as the main model.

This experiment also showed that increasing parameter count alone is not a sufficient solution to the observed image-quality limitations.

---

# 27. Timestep-Wise Training Objective Analysis

The original epsilon-only model was analyzed at different diffusion timesteps using the validation set.

Results:

| Timestep | Alpha-bar | Epsilon MSE |
| -------: | --------: | ----------: |
|        0 |  0.999900 |    1.013962 |
|      100 |  0.895141 |    0.069930 |
|      250 |  0.521423 |    0.021974 |
|      500 |  0.077797 |    0.007473 |
|      750 |  0.003300 |    0.006065 |
|      999 |  0.000040 |    0.006232 |

The epsilon prediction error decreased dramatically as timestep increased.

The model therefore learned noise prediction more easily at high-noise timesteps than near the clean-image region.

---

# 28. Timestep-Wise X0 Reconstruction Analysis

The predicted noise was converted back into an estimated clean image.

Results:

| Timestep | Epsilon MSE |   X0 MSE |
| -------: | ----------: | -------: |
|        0 |    1.013962 | 0.000101 |
|      100 |    0.069930 | 0.008160 |
|      250 |    0.021974 | 0.020003 |
|      500 |    0.007473 | 0.079128 |
|      750 |    0.006065 | 0.548711 |
|      999 |    0.006232 | 0.999441 |

A clear pattern appeared:

```text
As timestep increases:

epsilon prediction error
        ↓

X0 reconstruction error
        ↑
```

The increase in X0 error at large timesteps is expected because X0 reconstruction involves division by the square root of alpha-bar.

When alpha-bar is extremely small, even a small epsilon prediction error can become amplified during X0 reconstruction.

This analysis motivated the investigation of a direct X0 reconstruction loss.

---

# 29. Improved X0-Weighted Training Objective

An experimental objective was introduced:

```text
L_total =
L_epsilon
+
0.1 × L_X0
```

where:

```text
L_epsilon = MSE(predicted noise, true noise)

L_X0 = MSE(predicted X0, normal-light target)
```

The predicted X0 was clipped to:

```text
[-1, 1]
```

before calculating the reconstruction loss.

The implementation was verified independently.

Example verification:

```text
Total loss   : 1.176396
Epsilon loss : 1.087599
X0 loss      : 0.887972
```

The calculated total loss exactly matched:

```text
1.087599 + 0.1 × 0.887972
= 1.176396
```

Gradient propagation and numerical stability were also verified.

---

# 30. X0-Objective Training Experiment

A controlled experiment was performed using:

```text
Base channels : 32
X0 weight     : 0.1
Epochs        : 3
Batch size    : 2
```

Training results:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |      0.337795 |        0.130626 |
|     2 |      0.116335 |    **0.116501** |
|     3 |      0.099749 |        0.118770 |

Best validation loss:

```text
0.116501
```

at epoch 2.

The X0-objective experiment was therefore retained as an ablation study rather than replacing the main model.

---

# 31. Controlled 100-Image X0 Objective Comparison

A controlled evaluation was performed between:

```text
Original epsilon-only model
vs.
X0-objective model
```

The evaluation used:

* The same official 100 test images.
* The same preprocessing.
* The same metric implementation.
* The same fixed random seed.
* Separate generators initialized with the same seed.
* The same reverse-diffusion procedure.

This made the comparison more controlled than the earlier independent evaluations.

---

# 32. X0 Objective Results

The controlled results were:

| Metric        | Original ε-only |     X0 Objective |
| ------------- | --------------: | ---------------: |
| **Mean PSNR** |    15.033402 dB | **16.011893 dB** |
| Median PSNR   |    14.904627 dB | **16.404109 dB** |
| **Mean SSIM** |    **0.485660** |         0.299780 |
| Median SSIM   |    **0.488710** |         0.301061 |

Differences:

```text
PSNR:
X0 − Original = +0.978491 dB

SSIM:
X0 − Original = -0.185880
```

---

# 33. Per-Image X0 Comparison

For PSNR:

```text
X0 wins       : 60 / 100
Original wins : 40 / 100
```

Therefore, the X0 objective achieved higher PSNR on 60% of the test images.

However, for SSIM:

```text
X0 wins       : 0 / 100
Original wins : 100 / 100
```

This is a particularly strong result.

The X0 objective did not outperform the original epsilon-only model on SSIM for a single test image.

---

# 34. X0 Objective Variability

The standard deviation of PSNR was:

```text
Original : 1.682738
X0       : 2.411499
```

Therefore, the X0 model had greater variability in PSNR performance.

The X0 objective produced very strong PSNR gains on some images.

Largest observed PSNR improvements included approximately:

```text
+8.13 dB
+8.13 dB
+7.21 dB
+7.14 dB
+6.97 dB
```

However, it also produced substantial PSNR losses on some images:

```text
-6.48 dB
-6.47 dB
-6.47 dB
-5.51 dB
-5.50 dB
```

This confirms that the X0 objective does not simply provide a universally better reconstruction.

---

# 35. PSNR–SSIM Trade-Off

The X0 experiment demonstrates a clear trade-off.

The X0 objective:

```text
Improves:
Pixel-level reconstruction / PSNR
```

but:

```text
Degrades:
Structural similarity / SSIM
```

The mean results are:

```text
Original:
PSNR = 15.0334
SSIM = 0.485660

X0:
PSNR = 16.0119
SSIM = 0.299780
```

Therefore, the X0 objective is not considered an overall improvement.

The experiment demonstrates that optimizing directly toward X0 reconstruction can improve one quantitative metric while damaging another important image-quality property.

---

# 36. Final Model Selection

Based on all experiments, the project retains the following primary model:

```text
Conditional DDPM
+
32-channel U-Net
+
epsilon-prediction MSE objective
```

Primary checkpoint:

```text
checkpoints/best_model.pt
```

Reason for selection:

1. Strong structural similarity.
2. SSIM improvement on 100/100 test images relative to the low-light baseline.
3. Better SSIM than the X0-objective model.
4. More stable PSNR behavior than the X0 model.
5. Increasing U-Net capacity did not improve results.
6. The original objective provides the best balance among the tested diffusion configurations.

The X0-objective model remains an important **ablation experiment**.

---

# 37. Important Overall Finding

The experiments reveal three important conclusions.

### Finding 1 — Diffusion enhancement works

The conditional diffusion model learned to transform low-light inputs into images with substantially improved:

* brightness,
* scene structure,
* PSNR,
* SSIM.

Compared with the low-light baseline, the primary diffusion model improved mean PSNR by approximately:

```text
+5.34 dB
```

and mean SSIM by:

```text
+0.294
```

under the original 100-image evaluation.

---

### Finding 2 — Diffusion does not yet beat the classical baseline

Gamma correction with gamma = 0.4 achieved:

```text
PSNR = 19.1928 dB
SSIM = 0.677547
```

compared with:

```text
Diffusion:
PSNR = 15.1016 dB
SSIM = 0.485159
```

Therefore, the current diffusion model is not the best-performing enhancement method on the LOLv2-real test set.

This is an important experimental result rather than a failure of the project.

---

### Finding 3 — Training-objective choice matters

The X0-objective experiment showed:

```text
PSNR:
X0 objective > epsilon-only objective

SSIM:
epsilon-only objective > X0 objective
```

The difference was substantial enough that the original epsilon-only objective was retained.

This demonstrates that simply adding an additional reconstruction loss does not automatically improve overall diffusion image quality.

---

# 38. Phase 4 Limitations

The following limitations were identified:

## 38.1 Limited Dataset Size

The training set contains only:

```text
689 paired images
```

before the train/validation split.

This is relatively small for training a diffusion model.

---

## 38.2 Limited Model Capacity

The main U-Net contains:

```text
1,162,947 parameters
```

which is relatively small compared with many modern image-generation models.

However, increasing capacity to 48 channels did not improve performance in the controlled experiment.

---

## 38.3 Shallow Conditional Architecture

The current condition is primarily introduced through concatenation of:

```text
Noisy target + low-light condition
```

This provides conditioning information but does not use sophisticated multi-scale feature injection.

---

## 38.4 Computational Constraint

Training was performed on CPU.

The 32-channel 20-epoch training required substantial computation time.

Therefore, experimentation was performed carefully and controlled to avoid unnecessary training runs.

---

## 38.5 Image-Quality Artifacts

Visual inspection revealed:

* Color artifacts.
* Purple/blue/red mottling.
* Blur.
* Fine-detail loss.

These limitations explain why the diffusion model remains behind the gamma baseline in overall PSNR and SSIM.

---

# 39. Phase 4 Outputs

The major outputs produced during this phase include:

```text
checkpoints/
├── best_model.pt
├── latest_model.pt
├── checkpoint_epoch_005.pt
├── checkpoint_epoch_010.pt
├── checkpoint_epoch_015.pt
└── checkpoint_epoch_020.pt
```

Training results:

```text
results/
├── training_history.pt
├── training_history.csv
└── training_loss_curve.png
```

Generated images:

```text
results/generated/
```

Test evaluation:

```text
results/test_evaluation/
├── test_metrics.csv
├── summary.txt
└── visual_results/
```

Gamma baseline:

```text
results/baseline/
├── validation_gamma_search.csv
├── test_metrics_gamma.csv
├── summary_gamma.txt
└── gamma_visual_results/
```

Capacity experiment:

```text
checkpoints/capacity_48/
results/capacity_48/
```

Timestep analysis:

```text
results/timestep_analysis/
├── timestep_loss.csv
└── summary.txt
```

X0 reconstruction analysis:

```text
results/x0_reconstruction_analysis/
├── x0_timestep_analysis.csv
└── summary.txt
```

X0 objective experiment:

```text
checkpoints/objective_x0/
results/objective_x0/
```

Controlled objective comparison:

```text
results/objective_comparison/
├── test_metrics.csv
├── summary.txt
└── visual_results/
```

Objective analysis:

```text
results/objective_comparison_analysis/
├── analysis_metrics.csv
├── summary.txt
├── psnr_distribution.png
├── ssim_distribution.png
├── psnr_difference.png
└── ssim_difference.png
```

---

# 40. Phase 4 Final Conclusion

Phase 4 successfully transformed the preprocessing pipeline into a functioning conditional diffusion-based low-light image enhancement system.

The implementation of the DDPM components—including the noise schedule, forward diffusion, reverse diffusion, conditional U-Net, training objective, training pipeline, checkpointing, image generation, and quantitative evaluation—was verified through dedicated experiments.

The primary 32-channel conditional diffusion model demonstrated meaningful enhancement capability on the official LOLv2-real test set.

Under the original evaluation:

```text
Low-light:
PSNR = 9.7660 dB
SSIM = 0.191124

Diffusion:
PSNR = 15.1016 dB
SSIM = 0.485159
```

The model improved:

```text
PSNR on 95/100 images
SSIM on 100/100 images
```

However, gamma correction with gamma = 0.4 performed better overall:

```text
Gamma:
PSNR = 19.1928 dB
SSIM = 0.677547
```

Therefore, the current diffusion implementation should be regarded as a **working research prototype rather than the final highest-performing enhancement method**.

The capacity experiment demonstrated that simply increasing U-Net width did not solve the problem.

The training-objective experiment further demonstrated that adding an X0 reconstruction loss improved PSNR but substantially reduced SSIM:

```text
Original:
15.0334 dB PSNR / 0.485660 SSIM

X0 objective:
16.0119 dB PSNR / 0.299780 SSIM
```

Consequently, the original epsilon-only 32-channel model was retained as the primary model, while the X0 objective was retained as an ablation study.

The findings from Phase 4 establish a strong experimental foundation for subsequent improvements. In particular, the remaining gap between the diffusion model and the gamma baseline suggests that future work should focus on improving conditional information flow, image reconstruction quality, color consistency, and fine-detail preservation rather than simply increasing model size.

**Phase 4 Status: COMPLETED**
