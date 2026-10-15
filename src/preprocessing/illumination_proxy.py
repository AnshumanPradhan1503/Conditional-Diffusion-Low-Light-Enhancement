from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_kernel_2d(
    kernel_size: int = 15,
    sigma: float = 3.0,
) -> torch.Tensor:
    """
    Create a normalized 2D Gaussian kernel.

    Returns:
        Tensor of shape [1, 1, kernel_size, kernel_size].
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    radius = kernel_size // 2

    coords = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
    )

    gaussian_1d = torch.exp(
        -(coords ** 2) / (2 * sigma ** 2)
    )

    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    kernel_2d = gaussian_1d[:, None] * gaussian_1d[None, :]

    kernel_2d = kernel_2d / kernel_2d.sum()

    return kernel_2d.unsqueeze(0).unsqueeze(0)


def low_frequency_illumination(
    luminance: torch.Tensor,
    kernel_size: int = 15,
    sigma: float = 3.0,
) -> torch.Tensor:
    """
    Compute a low-frequency illumination proxy by
    Gaussian smoothing of a luminance map.

    Accepted inputs:
        [H, W]
        [1, H, W]
        [B, 1, H, W]

    Returns:
        [B, 1, H, W]

    Note:
        This is a low-frequency illumination proxy,
        not a ground-truth physical illumination map.
    """

    if luminance.ndim == 2:
        luminance_batch = luminance.unsqueeze(0).unsqueeze(0)

    elif luminance.ndim == 3:
        if luminance.shape[0] != 1:
            raise ValueError(
                "3D luminance input must have shape [1, H, W]."
            )

        luminance_batch = luminance.unsqueeze(0)

    elif luminance.ndim == 4:
        if luminance.shape[1] != 1:
            raise ValueError(
                "4D luminance input must have shape [B, 1, H, W]."
            )

        luminance_batch = luminance

    else:
        raise ValueError(
            "luminance must have shape [H, W], "
            "[1, H, W], or [B, 1, H, W]."
        )

    kernel = gaussian_kernel_2d(
        kernel_size=kernel_size,
        sigma=sigma,
    ).to(
        device=luminance_batch.device,
        dtype=luminance_batch.dtype,
    )

    padding = kernel_size // 2

    # Reflective padding prevents artificial dark borders.
    padded = F.pad(
        luminance_batch,
        (
            padding,
            padding,
            padding,
            padding,
        ),
        mode="reflect",
    )

    illumination = F.conv2d(
        padded,
        kernel,
        padding=0,
    )

    return illumination


def illumination_proxy_from_normalized_rgb(
    image: torch.Tensor,
    kernel_size: int = 15,
    sigma: float = 3.0,
) -> torch.Tensor:
    """
    Generate a low-frequency illumination proxy from
    normalized RGB image data in [-1, 1].

    Accepted inputs:
        [3, H, W]
        [B, 3, H, W]

    Returns:
        Single image:
            [1, H, W]

        Batch:
            [B, 1, H, W]
    """

    single_image = False

    if image.ndim == 3:
        if image.shape[0] != 3:
            raise ValueError(
                "3D RGB input must have shape [3, H, W]."
            )

        image_batch = image.unsqueeze(0)
        single_image = True

    elif image.ndim == 4:
        if image.shape[1] != 3:
            raise ValueError(
                "4D RGB input must have shape [B, 3, H, W]."
            )

        image_batch = image

    else:
        raise ValueError(
            "image must have shape [3, H, W] "
            "or [B, 3, H, W]."
        )

    # Convert normalized [-1, 1] RGB to [0, 1].
    image_01 = (
        (image_batch + 1.0) / 2.0
    ).clamp(0.0, 1.0)

    red = image_01[:, 0:1]
    green = image_01[:, 1:2]
    blue = image_01[:, 2:3]

    # Standard luminance approximation.
    luminance = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    illumination = low_frequency_illumination(
        luminance,
        kernel_size=kernel_size,
        sigma=sigma,
    )

    # Preserve the existing single-image API:
    # [1, 1, H, W] -> [1, H, W]
    if single_image:
        illumination = illumination.squeeze(0)

    return illumination