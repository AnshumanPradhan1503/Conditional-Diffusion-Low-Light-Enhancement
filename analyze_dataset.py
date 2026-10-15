from pathlib import Path
from PIL import Image
import numpy as np


DATASET_ROOT = Path("data/LOLv2-real")


def analyze_split(split):
    low_dir = DATASET_ROOT / split / "low"
    high_dir = DATASET_ROOT / split / "high"

    low_images = sorted(low_dir.glob("*.png"))
    high_images = sorted(high_dir.glob("*.png"))

    print(f"\n{'=' * 50}")
    print(f"{split.upper()} DATASET")
    print(f"{'=' * 50}")

    print(f"Low-light images   : {len(low_images)}")
    print(f"Normal-light images: {len(high_images)}")

    if len(low_images) != len(high_images):
        print("WARNING: Image counts do not match!")

    dimensions = set()
    channels = set()
    modes = set()

    low_brightness = []
    high_brightness = []

    corrupted = []

    for low_path, high_path in zip(low_images, high_images):

        try:
            low_img = Image.open(low_path).convert("RGB")
            high_img = Image.open(high_path).convert("RGB")

            low_array = np.asarray(low_img)
            high_array = np.asarray(high_img)

            dimensions.add(low_img.size)
            dimensions.add(high_img.size)

            channels.add(low_array.shape[2])
            channels.add(high_array.shape[2])

            modes.add(low_img.mode)
            modes.add(high_img.mode)

            # Convert RGB to grayscale brightness
            low_gray = (
                0.299 * low_array[:, :, 0]
                + 0.587 * low_array[:, :, 1]
                + 0.114 * low_array[:, :, 2]
            )

            high_gray = (
                0.299 * high_array[:, :, 0]
                + 0.587 * high_array[:, :, 1]
                + 0.114 * high_array[:, :, 2]
            )

            low_brightness.append(low_gray.mean())
            high_brightness.append(high_gray.mean())

        except Exception as e:
            corrupted.append((low_path.name, str(e)))

    print(f"\nImage dimensions : {dimensions}")
    print(f"Channels         : {channels}")
    print(f"Image modes      : {modes}")

    print("\nBrightness:")
    print(f"Average low-light brightness    : {np.mean(low_brightness):.2f}")
    print(f"Average normal-light brightness : {np.mean(high_brightness):.2f}")

    print("\nBrightness range:")
    print(
        f"Low-light    : "
        f"{np.min(low_brightness):.2f} - "
        f"{np.max(low_brightness):.2f}"
    )

    print(
        f"Normal-light : "
        f"{np.min(high_brightness):.2f} - "
        f"{np.max(high_brightness):.2f}"
    )

    print(f"\nCorrupted pairs: {len(corrupted)}")

    if corrupted:
        for item in corrupted[:10]:
            print(item)


if __name__ == "__main__":
    analyze_split("train")
    analyze_split("test")