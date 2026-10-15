from __future__ import annotations

import random
from typing import Tuple

from PIL import Image
import torchvision.transforms.functional as TF


IMAGE_SIZE = 256


def paired_random_crop(
    low_image: Image.Image,
    high_image: Image.Image,
    size: int = IMAGE_SIZE,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the exact same random crop to a low-light/high-light pair.
    """

    if low_image.size != high_image.size:
        raise ValueError(
            f"Image size mismatch: "
            f"low={low_image.size}, high={high_image.size}"
        )

    width, height = low_image.size

    if width < size or height < size:
        raise ValueError(
            f"Image size {width}x{height} is smaller than crop size {size}."
        )

    max_left = width - size
    max_top = height - size

    left = random.randint(0, max_left)
    top = random.randint(0, max_top)

    low_crop = TF.crop(
        low_image,
        top=top,
        left=left,
        height=size,
        width=size,
    )

    high_crop = TF.crop(
        high_image,
        top=top,
        left=left,
        height=size,
        width=size,
    )

    return low_crop, high_crop


def paired_center_crop(
    low_image: Image.Image,
    high_image: Image.Image,
    size: int = IMAGE_SIZE,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the exact same center crop to a low-light/high-light pair.
    """

    if low_image.size != high_image.size:
        raise ValueError(
            f"Image size mismatch: "
            f"low={low_image.size}, high={high_image.size}"
        )

    width, height = low_image.size

    if width < size or height < size:
        raise ValueError(
            f"Image size {width}x{height} is smaller than crop size {size}."
        )

    left = (width - size) // 2
    top = (height - size) // 2

    low_crop = TF.crop(
        low_image,
        top=top,
        left=left,
        height=size,
        width=size,
    )

    high_crop = TF.crop(
        high_image,
        top=top,
        left=left,
        height=size,
        width=size,
    )

    return low_crop, high_crop


def paired_random_horizontal_flip(
    low_image: Image.Image,
    high_image: Image.Image,
    probability: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the same random horizontal flip to both images.
    """

    if random.random() < probability:
        low_image = TF.hflip(low_image)
        high_image = TF.hflip(high_image)

    return low_image, high_image