from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "val_pairs.txt"

OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "validation_gamma_search.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMAGE_SIZE = 256

# Gamma < 1 brightens a dark image.
GAMMA_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------
# Image utilities
# ---------------------------------------------------------

def center_crop(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Center-crop an image to size x size."""

    width, height = image.size

    if width < size or height < size:
        raise ValueError(
            f"Image size {image.size} is smaller than crop size {size}."
        )

    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    return image.crop((left, top, right, bottom))


def load_image(path: Path) -> np.ndarray:
    """Load RGB image and convert to float32 [0, 1]."""

    image = Image.open(path).convert("RGB")
    image = center_crop(image)

    array = np.asarray(image).astype(np.float32) / 255.0

    return array


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to an image in [0, 1]."""

    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0.")

    corrected = np.power(image, gamma)

    return np.clip(corrected, 0.0, 1.0)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:

    psnr = peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0,
    )

    ssim = structural_similarity(
        target,
        prediction,
        data_range=1.0,
        channel_axis=2,
    )

    return float(psnr), float(ssim)


# ---------------------------------------------------------
# Read validation pairs
# ---------------------------------------------------------

def read_pairs(split_file: Path) -> list[tuple[str, str]]:
    """Read low/high filename pairs from a split file."""

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    pairs = []

    with split_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid split line: {line}"
                )

            low_name = parts[0]
            high_name = parts[1]

            pairs.append((low_name, high_name))

    return pairs


# ---------------------------------------------------------
# Main validation experiment
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("TASK 14.1 — VALIDATION-BASED GAMMA SEARCH")
    print("=" * 60)

    print(f"Dataset: {DATA_ROOT}")
    print(f"Validation split: {SPLIT_FILE}")
    print(f"Candidate gamma values: {GAMMA_VALUES}")

    pairs = read_pairs(SPLIT_FILE)

    print(f"Validation pairs: {len(pairs)}")

    if len(pairs) != 138:
        raise ValueError(
            f"Expected 138 validation pairs, found {len(pairs)}."
        )

    results = []

    for gamma in GAMMA_VALUES:

        psnr_values = []
        ssim_values = []

        print()
        print(f"Testing gamma = {gamma}")

        for index, (low_name, high_name) in enumerate(pairs):

            low_path = DATA_ROOT / "train" / "low" / low_name
            high_path = DATA_ROOT / "train" / "high" / high_name

            if not low_path.exists():
                raise FileNotFoundError(
                    f"Low-light image not found: {low_path}"
                )

            if not high_path.exists():
                raise FileNotFoundError(
                    f"Normal-light image not found: {high_path}"
                )

            low_image = load_image(low_path)
            high_image = load_image(high_path)

            enhanced_image = gamma_correct(
                low_image,
                gamma,
            )

            psnr, ssim = calculate_metrics(
                enhanced_image,
                high_image,
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

        mean_psnr = float(np.mean(psnr_values))
        mean_ssim = float(np.mean(ssim_values))

        median_psnr = float(np.median(psnr_values))
        median_ssim = float(np.median(ssim_values))

        results.append(
            {
                "gamma": gamma,
                "mean_psnr": mean_psnr,
                "median_psnr": median_psnr,
                "mean_ssim": mean_ssim,
                "median_ssim": median_ssim,
            }
        )

        print(f"Mean PSNR   : {mean_psnr:.4f} dB")
        print(f"Median PSNR : {median_psnr:.4f} dB")
        print(f"Mean SSIM   : {mean_ssim:.6f}")
        print(f"Median SSIM : {median_ssim:.6f}")

    # -----------------------------------------------------
    # Select best gamma
    # -----------------------------------------------------

    best_by_psnr = max(
        results,
        key=lambda item: item["mean_psnr"],
    )

    best_by_ssim = max(
        results,
        key=lambda item: item["mean_ssim"],
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    import csv

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "gamma",
                "mean_psnr",
                "median_psnr",
                "mean_ssim",
                "median_ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("VALIDATION GAMMA SEARCH COMPLETED")
    print("=" * 60)

    print()
    print("Best gamma according to PSNR:")
    print(f"Gamma       : {best_by_psnr['gamma']}")
    print(f"Mean PSNR   : {best_by_psnr['mean_psnr']:.4f} dB")
    print(f"Mean SSIM   : {best_by_psnr['mean_ssim']:.6f}")

    print()
    print("Best gamma according to SSIM:")
    print(f"Gamma       : {best_by_ssim['gamma']}")
    print(f"Mean PSNR   : {best_by_ssim['mean_psnr']:.4f} dB")
    print(f"Mean SSIM   : {best_by_ssim['mean_ssim']:.6f}")

    print()
    print(f"Results saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()