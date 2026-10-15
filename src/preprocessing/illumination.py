from __future__ import annotations

import torch


def rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor to a luminance/illumination representation.

    Expected input:
        Tensor with shape [3, H, W] or [B, 3, H, W]

    Expected value range:
        [0, 1]

    Output:
        Tensor with shape [1, H, W] or [B, 1, H, W]

    Luminance formula:
        Y = 0.299 R + 0.587 G + 0.114 B
    """

    if image.ndim == 3:
        if image.shape[0] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got shape {tuple(image.shape)}"
            )

        red = image[0:1]
        green = image[1:2]
        blue = image[2:3]

        luminance = (
            0.299 * red
            + 0.587 * green
            + 0.114 * blue
        )

        return luminance

    if image.ndim == 4:
        if image.shape[1] != 3:
            raise ValueError(
                f"Expected RGB batch with 3 channels, got shape {tuple(image.shape)}"
            )

        red = image[:, 0:1]
        green = image[:, 1:2]
        blue = image[:, 2:3]

        luminance = (
            0.299 * red
            + 0.587 * green
            + 0.114 * blue
        )

        return luminance

    raise ValueError(
        "Expected image tensor with shape [3, H, W] "
        "or [B, 3, H, W]."
    )


def normalized_to_01(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor from [-1, 1] to [0, 1].
    """

    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def illumination_from_normalized_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image normalized to [-1, 1] into a
    luminance/illumination representation in [0, 1].
    """

    image_01 = normalized_to_01(image)

    return rgb_to_luminance(image_01)