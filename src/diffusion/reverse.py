from __future__ import annotations

import torch
from torch import Tensor

from src.diffusion.forward import extract


def predict_x_start_from_noise(
    x_t: Tensor,
    timesteps: Tensor,
    predicted_noise: Tensor,
    alpha_bars: Tensor,
) -> Tensor:
    """
    Recover an estimate of the original clean image x_0
    from x_t and predicted noise.
    """

    sqrt_alpha_bar = torch.sqrt(
        extract(alpha_bars, timesteps, x_t.shape)
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - extract(alpha_bars, timesteps, x_t.shape)
    )

    x_start = (
        x_t - sqrt_one_minus_alpha_bar * predicted_noise
    ) / sqrt_alpha_bar

    return x_start


def posterior_mean(
    x_start: Tensor,
    x_t: Tensor,
    timesteps: Tensor,
    betas: Tensor,
    alphas: Tensor,
    alpha_bars: Tensor,
) -> Tensor:
    """
    Compute the mean of:

        q(x_{t-1} | x_t, x_0)
    """

    alpha_bar_t = extract(
        alpha_bars,
        timesteps,
        x_t.shape,
    )

    alpha_bar_prev = extract(
        torch.cat(
            [
                torch.ones(
                    1,
                    device=alpha_bars.device,
                ),
                alpha_bars[:-1],
            ]
        ),
        timesteps,
        x_t.shape,
    )

    beta_t = extract(
        betas,
        timesteps,
        x_t.shape,
    )

    alpha_t = extract(
        alphas,
        timesteps,
        x_t.shape,
    )

    coef_x_start = (
        torch.sqrt(alpha_bar_prev) * beta_t
    ) / (1.0 - alpha_bar_t)

    coef_x_t = (
        torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_prev)
    ) / (1.0 - alpha_bar_t)

    mean = (
        coef_x_start * x_start
        + coef_x_t * x_t
    )

    return mean


def posterior_variance(
    timesteps: Tensor,
    betas: Tensor,
    alpha_bars: Tensor,
) -> Tensor:
    """
    Compute the variance of:

        q(x_{t-1} | x_t, x_0)
    """

    alpha_bar_prev = torch.cat(
        [
            torch.ones(
                1,
                device=alpha_bars.device,
            ),
            alpha_bars[:-1],
        ]
    )

    alpha_bar_t = extract(
        alpha_bars,
        timesteps,
        (timesteps.shape[0], 1, 1, 1),
    )

    alpha_bar_prev_t = extract(
        alpha_bar_prev,
        timesteps,
        (timesteps.shape[0], 1, 1, 1),
    )

    beta_t = extract(
        betas,
        timesteps,
        (timesteps.shape[0], 1, 1, 1),
    )

    variance = (
        beta_t
        * (1.0 - alpha_bar_prev_t)
        / (1.0 - alpha_bar_t)
    )

    return variance


def p_sample(
    x_t: Tensor,
    timesteps: Tensor,
    predicted_noise: Tensor,
    betas: Tensor,
    alphas: Tensor,
    alpha_bars: Tensor,
    clip_x_start: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Perform one reverse-diffusion step:

        x_t -> x_{t-1}

    If clip_x_start is True, the predicted clean image
    is constrained to the valid training range [-1, 1].
    """

    x_start = predict_x_start_from_noise(
        x_t=x_t,
        timesteps=timesteps,
        predicted_noise=predicted_noise,
        alpha_bars=alpha_bars,
    )

    if clip_x_start:
        x_start = x_start.clamp(-1.0, 1.0)

    mean = posterior_mean(
        x_start=x_start,
        x_t=x_t,
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )

    variance = posterior_variance(
        timesteps=timesteps,
        betas=betas,
        alpha_bars=alpha_bars,
    )

    noise = torch.randn(
        x_t.shape,
        device=x_t.device,
        dtype=x_t.dtype,
        generator=generator,
    )

    nonzero_mask = (
        (timesteps != 0)
        .float()
        .reshape(
            timesteps.shape[0],
            *((1,) * (len(x_t.shape) - 1)),
        )
    )

    sample = (
        mean
        + nonzero_mask
        * torch.sqrt(variance)
        * noise
    )

    return sample


def reverse_diffusion(
    shape: tuple[int, ...],
    condition: Tensor,
    predicted_noise_fn,
    betas: Tensor,
    alphas: Tensor,
    alpha_bars: Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Complete conditional reverse-diffusion sampling loop.

    Starts from Gaussian noise x_T and repeatedly applies
    the reverse process until x_0.

    The low-light image is supplied as the condition at
    every denoising timestep.
    """

    x_t = torch.randn(
        shape,
        device=device,
        generator=generator,
    )

    batch_size = shape[0]

    if condition.shape != shape:
        raise ValueError(
            f"Condition shape {condition.shape} "
            f"does not match sampling shape {shape}."
        )

    condition = condition.to(device)

    for timestep in reversed(range(len(betas))):

        t = torch.full(
            (batch_size,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        predicted_noise = predicted_noise_fn(
            x_t,
            condition,
            t,
        )

        x_t = p_sample(
            x_t=x_t,
            timesteps=t,
            predicted_noise=predicted_noise,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            generator=generator,
        )

    return x_t
