from pathlib import Path

from PIL import Image

from src.preprocessing.paired_transforms import (
    paired_random_crop,
    paired_center_crop,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOW_PATH = (
    PROJECT_ROOT
    / "data"
    / "LOLv2-real"
    / "train"
    / "low"
    / "low00009.png"
)

HIGH_PATH = (
    PROJECT_ROOT
    / "data"
    / "LOLv2-real"
    / "train"
    / "high"
    / "normal00009.png"
)


def main():
    print("=" * 60)
    print("TESTING PAIRED CROPPING")
    print("=" * 60)

    low_image = Image.open(LOW_PATH).convert("RGB")
    high_image = Image.open(HIGH_PATH).convert("RGB")

    print(f"\nOriginal low-light size  : {low_image.size}")
    print(f"Original normal-light size: {high_image.size}")

    # Random crop
    low_random, high_random = paired_random_crop(
        low_image,
        high_image,
        size=256,
    )

    print("\nRandom crop:")
    print(f"Low-light crop size      : {low_random.size}")
    print(f"Normal-light crop size   : {high_random.size}")

    # Center crop
    low_center, high_center = paired_center_crop(
        low_image,
        high_image,
        size=256,
    )

    print("\nCenter crop:")
    print(f"Low-light crop size      : {low_center.size}")
    print(f"Normal-light crop size   : {high_center.size}")

    # Save temporary verification images
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    low_random.save(results_dir / "test_low_random_crop.png")
    high_random.save(results_dir / "test_high_random_crop.png")

    low_center.save(results_dir / "test_low_center_crop.png")
    high_center.save(results_dir / "test_high_center_crop.png")

    print("\nSaved verification images:")
    print("  results/test_low_random_crop.png")
    print("  results/test_high_random_crop.png")
    print("  results/test_low_center_crop.png")
    print("  results/test_high_center_crop.png")

    print("\n" + "=" * 60)
    print("TASK 3 CROPPING VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()