# Research Gap and Project Positioning

## 1. What existing work already covers

The reviewed literature already includes: - Deep Retinex decomposition
and illumination enhancement \[1\], \[5\]. - Unpaired GAN-based
enhancement \[2\]. - Zero-reference curve estimation \[3\]. -
Normalizing-flow-based enhancement \[4\]. - Transformer-based
enhancement and high-resolution benchmarks \[6\], \[7\]. - Retinex
decomposition combined with conditional diffusion \[8\]. - A
street-scene benchmark focused on realistic conditions \[9\]. - Recent
conditional diffusion guided by color, luminance, and other structured
priors in the reviewed arXiv preprint \[10\].

Therefore, the broad claims "use AI for low-light enhancement," "use
illumination guidance," or "use diffusion for low-light enhancement" are
not new by themselves.

## 2. Narrow, defensible gap

The project investigates a narrower experimental question:

**Within the project's conditional DDPM implementation and LOLv2-real
evaluation setup, how do alternative conditioning strategies affect
low-light enhancement quality?**

The completed experiments compare the base conditional diffusion model
with spatial conditioning and an illumination-aware V2 design. They also
include a fixed gamma-correction baseline. This gives the project a
controlled, implementation-specific comparison rather than a claim to
introduce an entirely new family of low-light enhancement methods.

## 3. What the completed experiments show

The project's frozen test-set results are:

  Method                            Mean PSNR (dB)   Mean SSIM
  ------------------------------- ---------------- -----------
  Low-light input                         9.766036    0.191124
  Gamma correction, gamma = 0.4          19.192762    0.677547
  Spatial conditioning                   14.961661    0.496689
  Illumination-aware V2                  13.273388    0.497885

The project records also note that the spatial and V2 experiments used
experiment-specific control values. When reporting changes relative to
controls, use each variant's matched control and do not mix those
control values.

The results support a balanced conclusion: - Both conditioning variants
have higher SSIM than the low-light input. - Gamma correction performs
best on the reported aggregate PSNR and SSIM values. - Spatial
conditioning has higher PSNR than illumination-aware V2 in the reported
table, while V2 has marginally higher SSIM. - The experiments do not
establish that either conditioning design is universally better, or that
diffusion is better than classical enhancement.

## 4. Suggested research-gap paragraph for the report

Although prior work has explored Retinex decomposition,
illumination-guided transformers, and conditional diffusion for
low-light image enhancement, the effect of alternative conditioning
designs depends on the model, dataset, training procedure, and
evaluation protocol. This project therefore conducts an experimental
comparison of spatial and illumination-aware conditioning strategies
within a common conditional DDPM framework on LOLv2-real. The study
evaluates PSNR and SSIM and interprets the results alongside a fixed
gamma-correction baseline. Its contribution is an
implementation-specific empirical analysis of the trade-offs observed
between the tested conditioning designs, rather than a claim that
diffusion-based or illumination-guided enhancement is new.

## 5. Limitations to state clearly

-   The evaluation is based on the project's selected LOLv2-real test
    set.
-   The reported gamma baseline uses one validation-selected fixed gamma
    value; it is not an exhaustive comparison against every classical or
    state-of-the-art method.
-   PSNR and SSIM do not fully capture perceptual quality, natural
    color, or downstream task performance.
-   The project has observed color artifacts and detail loss in
    generated images; these should be discussed, not hidden.
-   Results from papers using different datasets, resolutions, model
    sizes, or evaluation protocols should not be directly ranked against
    the project's numbers.

## 6. Final positioning

Present this as a **controlled experimental study of conditioning
strategies in a conditional diffusion model**. Avoid saying that the
project proposes the first diffusion model for low-light enhancement,
the first illumination-guided method, or a state-of-the-art method.
Those claims are not supported by the reviewed literature or the current
experimental results.
