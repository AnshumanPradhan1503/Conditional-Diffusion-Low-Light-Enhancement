from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DATASET_ROOT = Path("data/LOLv2-real")


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def analyze_split(split):
    low_dir = DATASET_ROOT / split / "low"
    high_dir = DATASET_ROOT / split / "high"

    low_images = sorted(low_dir.glob("*.png"))
    high_images = sorted(high_dir.glob("*.png"))

    print(f"\n{'=' * 60}")
    print(f"{split.upper()} VISUAL EDA")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------
    # 1. Select representative image pairs
    # ---------------------------------------------------------

    # Six evenly distributed samples
    indices = np.linspace(0, len(low_images) - 1, 6, dtype=int)

    fig, axes = plt.subplots(6, 2, figsize=(12, 22))

    for row, index in enumerate(indices):

        low_path = low_images[index]
        high_path = high_images[index]

        low_img = load_image(low_path)
        high_img = load_image(high_path)

        axes[row, 0].imshow(low_img)
        axes[row, 0].set_title(f"Low-light: {low_path.name}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(high_img)
        axes[row, 1].set_title(f"Normal-light: {high_path.name}")
        axes[row, 1].axis("off")

    plt.suptitle(
        f"{split.capitalize()} Dataset - Low-Light vs Normal-Light",
        fontsize=16
    )

    plt.tight_layout()

    output_path = Path("results") / f"{split}_image_pairs.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nSaved image comparison to:")
    print(output_path)

    # ---------------------------------------------------------
    # 2. Brightness distribution
    # ---------------------------------------------------------

    low_brightness = []
    high_brightness = []

    for low_path, high_path in zip(low_images, high_images):

        low_img = load_image(low_path)
        high_img = load_image(high_path)

        low_gray = (
            0.299 * low_img[:, :, 0]
            + 0.587 * low_img[:, :, 1]
            + 0.114 * low_img[:, :, 2]
        )

        high_gray = (
            0.299 * high_img[:, :, 0]
            + 0.587 * high_img[:, :, 1]
            + 0.114 * high_img[:, :, 2]
        )

        low_brightness.append(low_gray.mean())
        high_brightness.append(high_gray.mean())

    low_brightness = np.array(low_brightness)
    high_brightness = np.array(high_brightness)

    # ---------------------------------------------------------
    # 3. Brightness histogram
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        low_brightness,
        bins=30,
        alpha=0.7,
        label="Low-light"
    )

    plt.hist(
        high_brightness,
        bins=30,
        alpha=0.7,
        label="Normal-light"
    )

    plt.xlabel("Average Image Brightness")
    plt.ylabel("Number of Images")
    plt.title(f"{split.capitalize()} Brightness Distribution")
    plt.legend()
    plt.grid(alpha=0.2)

    output_path = Path("results") / f"{split}_brightness_distribution.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved brightness histogram to:")
    print(output_path)

    # ---------------------------------------------------------
    # 4. Statistics
    # ---------------------------------------------------------

    print("\nBrightness Statistics")

    print(
        f"Low-light    → "
        f"Mean: {low_brightness.mean():.2f}, "
        f"Std: {low_brightness.std():.2f}, "
        f"Min: {low_brightness.min():.2f}, "
        f"Max: {low_brightness.max():.2f}"
    )

    print(
        f"Normal-light → "
        f"Mean: {high_brightness.mean():.2f}, "
        f"Std: {high_brightness.std():.2f}, "
        f"Min: {high_brightness.min():.2f}, "
        f"Max: {high_brightness.max():.2f}"
    )


if __name__ == "__main__":

    analyze_split("train")
    analyze_split("test")

    print("\n" + "=" * 60)
    print("VISUAL EDA COMPLETED")
    print("=" * 60)