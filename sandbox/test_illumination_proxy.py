from __future__ import annotations

import torch

from src.preprocessing.illumination_proxy import (
    illumination_proxy_from_normalized_rgb,
    low_frequency_illumination,
)


def main() -> None:
    print("Testing low-frequency illumination proxy...")

    # ---------------------------------------------------------
    # Test 1: Single image
    # ---------------------------------------------------------
    image = torch.rand(3, 256, 256) * 2.0 - 1.0

    illumination = illumination_proxy_from_normalized_rgb(image)

    print("\nTest 1: Single normalized RGB image")
    print(f"Input shape:       {tuple(image.shape)}")
    print(f"Output shape:      {tuple(illumination.shape)}")
    print(f"Output min:        {illumination.min().item():.6f}")
    print(f"Output max:        {illumination.max().item():.6f}")

    assert illumination.shape == (1, 256, 256)
    assert illumination.min() >= 0.0
    assert illumination.max() <= 1.0

    # ---------------------------------------------------------
    # Test 2: Batch
    # ---------------------------------------------------------
    batch = torch.rand(2, 3, 256, 256) * 2.0 - 1.0

    batch_illumination = illumination_proxy_from_normalized_rgb(batch)

    print("\nTest 2: Batch")
    print(f"Input shape:       {tuple(batch.shape)}")
    print(f"Output shape:      {tuple(batch_illumination.shape)}")

    assert batch_illumination.shape == (2, 1, 256, 256)
    assert batch_illumination.min() >= 0.0
    assert batch_illumination.max() <= 1.0

    # ---------------------------------------------------------
    # Test 3: Constant image
    # ---------------------------------------------------------
    constant = torch.full((1, 32, 32), 0.4)

    constant_output = low_frequency_illumination(constant)

    print("\nTest 3: Constant luminance")
    print(
        f"Input mean:        {constant.mean().item():.6f}"
    )
    print(
        f"Output mean:       {constant_output.mean().item():.6f}"
    )

    # Gaussian filtering should preserve a constant signal.
    assert torch.allclose(
        constant_output,
        constant,
        atol=1e-5,
    )

    # ---------------------------------------------------------
    # Test 4: Output remains smooth
    # ---------------------------------------------------------
    random_luminance = torch.rand(1, 64, 64)

    smooth = low_frequency_illumination(
        random_luminance,
        kernel_size=15,
        sigma=3.0,
    )

    random_variance = random_luminance.var().item()
    smooth_variance = smooth.var().item()

    print("\nTest 4: Smoothing behavior")
    print(f"Random variance:  {random_variance:.6f}")
    print(f"Smooth variance:  {smooth_variance:.6f}")

    assert smooth_variance < random_variance

    print("\nAll illumination proxy tests passed.")


if __name__ == "__main__":
    main()