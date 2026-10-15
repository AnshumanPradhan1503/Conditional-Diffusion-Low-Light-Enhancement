from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATASET_ROOT = Path("data/LOLv2-real")
RESULTS_DIR = Path("results")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def calculate_statistics(low_img, high_img):
    # ---------------------------------------------------------
    # RGB channel statistics
    # ---------------------------------------------------------

    low_mean_rgb = low_img.mean(axis=(0, 1))
    high_mean_rgb = high_img.mean(axis=(0, 1))

    low_std_rgb = low_img.std(axis=(0, 1))
    high_std_rgb = high_img.std(axis=(0, 1))

    # ---------------------------------------------------------
    # Grayscale brightness
    # ---------------------------------------------------------

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

    low_brightness = low_gray.mean()
    high_brightness = high_gray.mean()

    # ---------------------------------------------------------
    # Very dark pixels
    # ---------------------------------------------------------

    low_dark_percentage = (low_gray < 10).mean() * 100
    high_dark_percentage = (high_gray < 10).mean() * 100

    # ---------------------------------------------------------
    # Bright / saturated pixels
    # ---------------------------------------------------------

    low_bright_percentage = (low_gray > 245).mean() * 100
    high_bright_percentage = (high_gray > 245).mean() * 100

    # ---------------------------------------------------------
    # Brightness improvement
    # ---------------------------------------------------------

    brightness_difference = high_brightness - low_brightness

    brightness_ratio = (
        high_brightness / low_brightness
        if low_brightness > 0
        else np.nan
    )

    return {
        "low_mean_r": low_mean_rgb[0],
        "low_mean_g": low_mean_rgb[1],
        "low_mean_b": low_mean_rgb[2],

        "high_mean_r": high_mean_rgb[0],
        "high_mean_g": high_mean_rgb[1],
        "high_mean_b": high_mean_rgb[2],

        "low_std_r": low_std_rgb[0],
        "low_std_g": low_std_rgb[1],
        "low_std_b": low_std_rgb[2],

        "high_std_r": high_std_rgb[0],
        "high_std_g": high_std_rgb[1],
        "high_std_b": high_std_rgb[2],

        "low_brightness": low_brightness,
        "high_brightness": high_brightness,

        "brightness_difference": brightness_difference,
        "brightness_ratio": brightness_ratio,

        "low_dark_percentage": low_dark_percentage,
        "high_dark_percentage": high_dark_percentage,

        "low_bright_percentage": low_bright_percentage,
        "high_bright_percentage": high_bright_percentage,
    }


def analyze_split(split):
    low_dir = DATASET_ROOT / split / "low"
    high_dir = DATASET_ROOT / split / "high"

    low_images = sorted(low_dir.glob("*.png"))
    high_images = sorted(high_dir.glob("*.png"))

    records = []

    print("\n" + "=" * 70)
    print(f"{split.upper()} STATISTICAL EDA")
    print("=" * 70)

    for low_path, high_path in zip(low_images, high_images):

        low_img = load_rgb(low_path)
        high_img = load_rgb(high_path)

        stats = calculate_statistics(low_img, high_img)

        stats["low_filename"] = low_path.name
        stats["high_filename"] = high_path.name

        records.append(stats)

    df = pd.DataFrame(records)

    # ---------------------------------------------------------
    # Overall statistics
    # ---------------------------------------------------------

    print("\nRGB CHANNEL MEANS")
    print("-" * 70)

    print(
        f"Low-light RGB mean    : "
        f"R={df['low_mean_r'].mean():.2f}, "
        f"G={df['low_mean_g'].mean():.2f}, "
        f"B={df['low_mean_b'].mean():.2f}"
    )

    print(
        f"Normal-light RGB mean : "
        f"R={df['high_mean_r'].mean():.2f}, "
        f"G={df['high_mean_g'].mean():.2f}, "
        f"B={df['high_mean_b'].mean():.2f}"
    )

    # ---------------------------------------------------------
    # RGB standard deviation
    # ---------------------------------------------------------

    print("\nRGB CHANNEL STANDARD DEVIATION")
    print("-" * 70)

    print(
        f"Low-light RGB std    : "
        f"R={df['low_std_r'].mean():.2f}, "
        f"G={df['low_std_g'].mean():.2f}, "
        f"B={df['low_std_b'].mean():.2f}"
    )

    print(
        f"Normal-light RGB std : "
        f"R={df['high_std_r'].mean():.2f}, "
        f"G={df['high_std_g'].mean():.2f}, "
        f"B={df['high_std_b'].mean():.2f}"
    )

    # ---------------------------------------------------------
    # Brightness
    # ---------------------------------------------------------

    print("\nBRIGHTNESS")
    print("-" * 70)

    print(
        f"Low-light mean brightness    : "
        f"{df['low_brightness'].mean():.2f}"
    )

    print(
        f"Normal-light mean brightness : "
        f"{df['high_brightness'].mean():.2f}"
    )

    print(
        f"Mean brightness difference   : "
        f"{df['brightness_difference'].mean():.2f}"
    )

    print(
        f"Mean brightness ratio        : "
        f"{df['brightness_ratio'].mean():.2f}x"
    )

    # ---------------------------------------------------------
    # Dark pixels
    # ---------------------------------------------------------

    print("\nVERY DARK PIXELS (< 10)")
    print("-" * 70)

    print(
        f"Low-light    : "
        f"{df['low_dark_percentage'].mean():.2f}%"
    )

    print(
        f"Normal-light : "
        f"{df['high_dark_percentage'].mean():.2f}%"
    )

    # ---------------------------------------------------------
    # Bright / saturated pixels
    # ---------------------------------------------------------

    print("\nVERY BRIGHT PIXELS (> 245)")
    print("-" * 70)

    print(
        f"Low-light    : "
        f"{df['low_bright_percentage'].mean():.2f}%"
    )

    print(
        f"Normal-light : "
        f"{df['high_bright_percentage'].mean():.2f}%"
    )

    # ---------------------------------------------------------
    # Most difficult images
    # ---------------------------------------------------------

    print("\nDARKEST LOW-LIGHT IMAGES")
    print("-" * 70)

    darkest = df.nsmallest(10, "low_brightness")

    for _, row in darkest.iterrows():
        print(
            f"{row['low_filename']} → "
            f"{row['low_brightness']:.2f}"
        )

    # ---------------------------------------------------------
    # Largest brightness improvements
    # ---------------------------------------------------------

    print("\nLARGEST BRIGHTNESS IMPROVEMENTS")
    print("-" * 70)

    largest_improvement = df.nlargest(
        10,
        "brightness_difference"
    )

    for _, row in largest_improvement.iterrows():
        print(
            f"{row['low_filename']} → "
            f"{row['brightness_difference']:.2f}"
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        RESULTS_DIR / f"{split}_statistical_analysis.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nSaved detailed statistics to:")
    print(output_file)

    return df


if __name__ == "__main__":

    train_df = analyze_split("train")
    test_df = analyze_split("test")

    print("\n" + "=" * 70)
    print("STATISTICAL EDA COMPLETED")
    print("=" * 70)