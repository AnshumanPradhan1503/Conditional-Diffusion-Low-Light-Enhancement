from __future__ import annotations

import torch
from torch import Tensor


def extract(
    values: Tensor,
    timesteps: Tensor,
    x_shape: tuple[int, ...],
) -> Tensor:
    """
    Extract values corresponding to the selected timesteps
    and reshape them for broadcasting over image tensors.
    """

    batch_size = timesteps.shape[0]

    out = values.gather(0, timesteps)

    return out.reshape(
        batch_size,
        *((1,) * (len(x_shape) - 1)),
    )


def q_sample(
    x_start: Tensor,
    timesteps: Tensor,
    alpha_bars: Tensor,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Perform the forward diffusion process.

    Equation:

        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * epsilon

    Args:
        x_start:
            Clean normal-light image tensor [B, C, H, W].

        timesteps:
            Diffusion timestep for each image [B].

        alpha_bars:
            Precomputed cumulative alpha values.

        noise:
            Optional Gaussian noise. If None, generated automatically.

    Returns:
        x_t:
            Noisy image at timestep t.

        noise:
            Gaussian noise that was added.
    """

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_bar = torch.sqrt(
        extract(alpha_bars, timesteps, x_start.shape)
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - extract(alpha_bars, timesteps, x_start.shape)
    )

    x_t = (
        sqrt_alpha_bar * x_start
        + sqrt_one_minus_alpha_bar * noise
    )

    return x_t, noise