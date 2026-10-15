from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.diffusion.forward import q_sample
from src.preprocessing.illumination_proxy import (
    illumination_proxy_from_normalized_rgb,
)


def illumination_aware_training_loss(
    model,
    low_light: Tensor,
    normal_light: Tensor,
    alpha_bars: Tensor,
) -> Tensor:
    """
    Compute the DDPM noise-prediction training loss for the
    illumination-aware conditional U-Net V2.

    Training procedure is the same as the original
    diffusion_training_loss, with one additional conditioning
    input: a low-frequency illumination proxy derived from
    the low-light RGB image.

    Steps:
        1. Sample a random diffusion timestep.
        2. Generate Gaussian noise.
        3. Add noise to the normal-light target.
        4. Compute the low-frequency illumination proxy.
        5. Predict the noise using the V2 conditional U-Net.
        6. Compute MSE between actual and predicted noise.

    The illumination proxy is derived dynamically from the
    low-light input and is not a separate learned target.
    """

    batch_size = normal_light.shape[0]
    device = normal_light.device

    # ---------------------------------------------------------
    # Sample random diffusion timestep
    # ---------------------------------------------------------

    timesteps = torch.randint(
        low=0,
        high=len(alpha_bars),
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    # ---------------------------------------------------------
    # Forward diffusion
    # ---------------------------------------------------------

    noisy_target, noise = q_sample(
        x_start=normal_light,
        timesteps=timesteps,
        alpha_bars=alpha_bars,
    )

    # ---------------------------------------------------------
    # Compute low-frequency illumination proxy
    #
    # low_light is already normalized to [-1, 1].
    # The proxy function converts it to [0, 1], computes
    # luminance, and applies low-frequency Gaussian smoothing.
    # ---------------------------------------------------------

    illumination = illumination_proxy_from_normalized_rgb(
        low_light
    )

    # ---------------------------------------------------------
    # Predict noise
    # ---------------------------------------------------------

    predicted_noise = model(
        noisy_image=noisy_target,
        condition=low_light,
        timesteps=timesteps,
        illumination=illumination,
    )

    # ---------------------------------------------------------
    # Standard epsilon MSE
    # ---------------------------------------------------------

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    return loss