from pathlib import Path

import torch
from PIL import Image

from src.preprocessing.normalization import (
    image_to_tensor,
    tensor_to_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "LOLv2-real"
    / "train"
    / "low"
    / "low00009.png"
)


def main():
    print("=" * 60)
    print("TESTING IMAGE NORMALIZATION")
    print("=" * 60)

    image = Image.open(IMAGE_PATH).convert("RGB")

    print("\nOriginal image:")
    print(f"  Type   : {type(image).__name__}")
    print(f"  Mode   : {image.mode}")
    print(f"  Size   : {image.size}")

    tensor = image_to_tensor(image)

    print("\nNormalized tensor:")
    print(f"  Shape  : {tensor.shape}")
    print(f"  Dtype  : {tensor.dtype}")
    print(f"  Min    : {tensor.min().item():.6f}")
    print(f"  Max    : {tensor.max().item():.6f}")
    print(f"  Mean   : {tensor.mean().item():.6f}")

    # Verify properties
    assert tensor.dtype == torch.float32
    assert tensor.ndim == 3
    assert tensor.shape[0] == 3
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0

    print("\nNormalization checks:")
    print("  [PASS] dtype = float32")
    print("  [PASS] image has 3 channels")
    print("  [PASS] values are within [-1, 1]")

    # Convert back to image
    reconstructed = tensor_to_image(tensor)

    print("\nReverse conversion:")
    print(f"  Type   : {type(reconstructed).__name__}")
    print(f"  Mode   : {reconstructed.mode}")
    print(f"  Size   : {reconstructed.size}")

    assert reconstructed.mode == "RGB"
    assert reconstructed.size == image.size

    # Save verification image
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "normalization_reconstructed.png"
    reconstructed.save(output_path)

    print(f"\nReconstructed image saved to:")
    print(f"  {output_path}")

    print("\n" + "=" * 60)
    print("TASK 4 NORMALIZATION VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()