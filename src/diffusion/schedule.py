from __future__ import annotations

import torch


def linear_beta_schedule(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Create a linear beta schedule for DDPM.

    Args:
        timesteps: Number of diffusion timesteps.
        beta_start: Initial noise variance.
        beta_end: Final noise variance.

    Returns:
        Tensor containing beta_t for every timestep.
    """

    if timesteps <= 0:
        raise ValueError("timesteps must be greater than 0")

    if beta_start <= 0:
        raise ValueError("beta_start must be greater than 0")

    if beta_end <= beta_start:
        raise ValueError("beta_end must be greater than beta_start")

    betas = torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        dtype=torch.float32,
    )

    return betas


def get_diffusion_parameters(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
):
    """
    Calculate the main DDPM schedule parameters.
    """

    betas = linear_beta_schedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bars