from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.diffusion.forward import q_sample


def diffusion_training_loss(
    model,
    low_light: Tensor,
    normal_light: Tensor,
    alpha_bars: Tensor,
) -> Tensor:
    """
    Compute the DDPM noise-prediction training loss.

    Steps:
        1. Sample a random diffusion timestep.
        2. Generate Gaussian noise.
        3. Add noise to the normal-light target.
        4. Predict the noise using the conditional U-Net.
        5. Compute MSE between actual and predicted noise.
    """

    batch_size = normal_light.shape[0]
    device = normal_light.device

    timesteps = torch.randint(
        low=0,
        high=len(alpha_bars),
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    noisy_target, noise = q_sample(
        x_start=normal_light,
        timesteps=timesteps,
        alpha_bars=alpha_bars,
    )

    predicted_noise = model(
        noisy_image=noisy_target,
        condition=low_light,
        timesteps=timesteps,
    )

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    return loss