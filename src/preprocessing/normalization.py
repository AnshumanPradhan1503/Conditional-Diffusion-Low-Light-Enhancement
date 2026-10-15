from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL RGB image from uint8 [0, 255]
    to float32 [-1, 1].
    """

    # Convert PIL image to tensor.
    # Output shape: [C, H, W]
    # Initial range: [0, 255]
    tensor = TF.pil_to_tensor(image)

    # Convert uint8 -> float32
    tensor = tensor.float()

    # [0, 255] -> [0, 1]
    tensor = tensor / 255.0

    # [0, 1] -> [-1, 1]
    tensor = tensor * 2.0 - 1.0

    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor from [-1, 1] back to a PIL RGB image.
    """

    # [-1, 1] -> [0, 1]
    tensor = (tensor + 1.0) / 2.0

    # Keep values safely within [0, 1]
    tensor = tensor.clamp(0.0, 1.0)

    # [0, 1] -> [0, 255]
    tensor = tensor * 255.0

    tensor = tensor.byte()

    return TF.to_pil_image(tensor)