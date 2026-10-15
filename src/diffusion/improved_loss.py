from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.diffusion.forward import q_sample
from src.diffusion.reverse import predict_x_start_from_noise


def improved_diffusion_training_loss(
    model,
    low_light: Tensor,
    normal_light: Tensor,
    alpha_bars: Tensor,
    x0_weight: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Improved diffusion training objective.

    The objective combines:

        1. Standard epsilon prediction loss
        2. Clean-image x0 reconstruction loss

    Total loss:

        L = L_epsilon + x0_weight * L_x0

    The original diffusion loss remains unchanged.
    """

    batch_size = normal_light.shape[0]
    device = normal_light.device

    # --------------------------------------------------------
    # 1. Sample random diffusion timestep
    # --------------------------------------------------------

    timesteps = torch.randint(
        low=0,
        high=len(alpha_bars),
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # 2. Forward diffusion
    # --------------------------------------------------------

    noisy_target, noise = q_sample(
        x_start=normal_light,
        timesteps=timesteps,
        alpha_bars=alpha_bars,
    )

    # --------------------------------------------------------
    # 3. Predict noise
    # --------------------------------------------------------

    predicted_noise = model(
        noisy_image=noisy_target,
        condition=low_light,
        timesteps=timesteps,
    )

    # --------------------------------------------------------
    # 4. Standard epsilon prediction loss
    # --------------------------------------------------------

    epsilon_loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    # --------------------------------------------------------
    # 5. Reconstruct x0 from predicted noise
    # --------------------------------------------------------

    predicted_x0 = predict_x_start_from_noise(
        x_t=noisy_target,
        timesteps=timesteps,
        predicted_noise=predicted_noise,
        alpha_bars=alpha_bars,
    )

    # Keep reconstructed image inside the valid
    # normalized image range.
    predicted_x0 = predicted_x0.clamp(
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # 6. x0 reconstruction loss
    # --------------------------------------------------------

    x0_loss = F.mse_loss(
        predicted_x0,
        normal_light,
    )

    # --------------------------------------------------------
    # 7. Combined objective
    # --------------------------------------------------------

    total_loss = (
        epsilon_loss
        + x0_weight * x0_loss
    )

    return (
        total_loss,
        epsilon_loss,
        x0_loss,
    )