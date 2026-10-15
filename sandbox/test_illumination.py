from __future__ import annotations

import torch

from src.preprocessing.illumination import (
    illumination_from_normalized_rgb,
    rgb_to_luminance,
)


def main() -> None:
    print("Testing RGB → illumination representation...")

    # ---------------------------------------------------------
    # Test 1: Single RGB image
    # ---------------------------------------------------------
    image = torch.rand(3, 256, 256)

    illumination = rgb_to_luminance(image)

    print("\nTest 1: RGB image")
    print(f"Input shape:       {tuple(image.shape)}")
    print(f"Output shape:      {tuple(illumination.shape)}")
    print(f"Output min:        {illumination.min().item():.6f}")
    print(f"Output max:        {illumination.max().item():.6f}")

    assert illumination.shape == (1, 256, 256)
    assert illumination.min() >= 0.0
    assert illumination.max() <= 1.0

    # ---------------------------------------------------------
    # Test 2: Batch of RGB images
    # ---------------------------------------------------------
    batch = torch.rand(2, 3, 256, 256)

    batch_illumination = rgb_to_luminance(batch)

    print("\nTest 2: RGB batch")
    print(f"Input shape:       {tuple(batch.shape)}")
    print(f"Output shape:      {tuple(batch_illumination.shape)}")

    assert batch_illumination.shape == (2, 1, 256, 256)
    assert batch_illumination.min() >= 0.0
    assert batch_illumination.max() <= 1.0

    # ---------------------------------------------------------
    # Test 3: Existing [-1, 1] representation
    # ---------------------------------------------------------
    normalized_image = torch.rand(3, 256, 256) * 2.0 - 1.0

    illumination_from_normalized = (
        illumination_from_normalized_rgb(normalized_image)
    )

    print("\nTest 3: [-1, 1] RGB → illumination")
    print(f"Input shape:       {tuple(normalized_image.shape)}")
    print(f"Output shape:      {tuple(illumination_from_normalized.shape)}")
    print(
        f"Output min:        "
        f"{illumination_from_normalized.min().item():.6f}"
    )
    print(
        f"Output max:        "
        f"{illumination_from_normalized.max().item():.6f}"
    )

    assert illumination_from_normalized.shape == (1, 256, 256)
    assert illumination_from_normalized.min() >= 0.0
    assert illumination_from_normalized.max() <= 1.0

    # ---------------------------------------------------------
    # Test 4: Known values
    # ---------------------------------------------------------
    pure_red = torch.zeros(3, 1, 1)
    pure_red[0, 0, 0] = 1.0

    pure_green = torch.zeros(3, 1, 1)
    pure_green[1, 0, 0] = 1.0

    pure_blue = torch.zeros(3, 1, 1)
    pure_blue[2, 0, 0] = 1.0

    red_y = rgb_to_luminance(pure_red).item()
    green_y = rgb_to_luminance(pure_green).item()
    blue_y = rgb_to_luminance(pure_blue).item()

    print("\nTest 4: Known luminance coefficients")
    print(f"Pure red:          {red_y:.6f}")
    print(f"Pure green:        {green_y:.6f}")
    print(f"Pure blue:         {blue_y:.6f}")

    assert abs(red_y - 0.299) < 1e-6
    assert abs(green_y - 0.587) < 1e-6
    assert abs(blue_y - 0.114) < 1e-6

    print("\nAll illumination representation tests passed.")


if __name__ == "__main__":
    main()