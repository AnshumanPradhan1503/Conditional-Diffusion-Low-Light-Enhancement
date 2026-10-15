from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "test_pairs.txt"

OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline"
VISUAL_DIR = OUTPUT_DIR / "gamma_visual_results"

OUTPUT_CSV = OUTPUT_DIR / "test_metrics_gamma.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary_gamma.txt"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMAGE_SIZE = 256

# Frozen from validation search.
# DO NOT change this based on test results.
GAMMA = 0.4


# ---------------------------------------------------------
# Image utilities
# ---------------------------------------------------------

def center_crop(
    image: Image.Image,
    size: int = IMAGE_SIZE,
) -> Image.Image:
    """Center-crop image to size x size."""

    width, height = image.size

    if width < size or height < size:
        raise ValueError(
            f"Image size {image.size} is smaller than crop size {size}."
        )

    left = (width - size) // 2
    top = (height - size) // 2

    return image.crop(
        (
            left,
            top,
            left + size,
            top + size,
        )
    )


def load_image(path: Path) -> np.ndarray:
    """Load RGB image and convert to float32 [0, 1]."""

    image = Image.open(path).convert("RGB")
    image = center_crop(image)

    return np.asarray(image).astype(np.float32) / 255.0


def gamma_correct(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply gamma correction."""

    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0.")

    corrected = np.power(image, gamma)

    return np.clip(corrected, 0.0, 1.0)


def save_image(
    image: np.ndarray,
    path: Path,
) -> None:
    """Save [0,1] RGB image as PNG."""

    image_uint8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        image_uint8,
        mode="RGB",
    ).save(path)


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
# Split reader
# ---------------------------------------------------------

def read_pairs(
    split_file: Path,
) -> list[tuple[str, str]]:

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    pairs = []

    with split_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = [
                part.strip()
                for part in line.split(",")
            ]

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid split line: {line}"
                )

            pairs.append(
                (
                    parts[0],
                    parts[1],
                )
            )

    return pairs


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("TASK 14.2 — OFFICIAL TEST-SET GAMMA EVALUATION")
    print("=" * 60)

    print(f"Device: CPU")
    print(f"Evaluation set: official LOLv2-real test set")
    print(f"Frozen gamma: {GAMMA}")
    print(f"Test split: {SPLIT_FILE}")

    pairs = read_pairs(SPLIT_FILE)

    print(f"Test pairs: {len(pairs)}")

    if len(pairs) != 100:
        raise ValueError(
            f"Expected 100 official test pairs, found {len(pairs)}."
        )

    VISUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = []

    print()
    print("Starting gamma-correction evaluation...")
    print()

    for index, (low_name, high_name) in enumerate(pairs):

        low_path = DATA_ROOT / "test" / "low" / low_name
        high_path = DATA_ROOT / "test" / "high" / high_name

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

        # -------------------------------------------------
        # Original low-light baseline
        # -------------------------------------------------

        low_psnr, low_ssim = calculate_metrics(
            low_image,
            high_image,
        )

        # -------------------------------------------------
        # Gamma enhancement
        # -------------------------------------------------

        gamma_image = gamma_correct(
            low_image,
            GAMMA,
        )

        gamma_psnr, gamma_ssim = calculate_metrics(
            gamma_image,
            high_image,
        )

        psnr_improvement = (
            gamma_psnr - low_psnr
        )

        ssim_improvement = (
            gamma_ssim - low_ssim
        )

        metrics.append(
            {
                "index": index,
                "low_file": low_name,
                "high_file": high_name,
                "low_psnr": low_psnr,
                "low_ssim": low_ssim,
                "gamma_psnr": gamma_psnr,
                "gamma_ssim": gamma_ssim,
                "psnr_improvement": psnr_improvement,
                "ssim_improvement": ssim_improvement,
            }
        )

        # -------------------------------------------------
        # Save visual results
        # -------------------------------------------------

        save_image(
            low_image,
            VISUAL_DIR / f"low_{index:03d}.png",
        )

        save_image(
            gamma_image,
            VISUAL_DIR / f"gamma_{index:03d}.png",
        )

        save_image(
            high_image,
            VISUAL_DIR / f"high_{index:03d}.png",
        )

        if (index + 1) % 10 == 0:

            current_psnr = np.mean(
                [
                    item["gamma_psnr"]
                    for item in metrics
                ]
            )

            current_ssim = np.mean(
                [
                    item["gamma_ssim"]
                    for item in metrics
                ]
            )

            print(
                f"Completed {index + 1}/100 images | "
                f"Current mean PSNR: {current_psnr:.4f} dB | "
                f"Current mean SSIM: {current_ssim:.6f}"
            )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    low_psnr = np.array(
        [item["low_psnr"] for item in metrics]
    )

    low_ssim = np.array(
        [item["low_ssim"] for item in metrics]
    )

    gamma_psnr = np.array(
        [item["gamma_psnr"] for item in metrics]
    )

    gamma_ssim = np.array(
        [item["gamma_ssim"] for item in metrics]
    )

    psnr_improvement = np.array(
        [item["psnr_improvement"] for item in metrics]
    )

    ssim_improvement = np.array(
        [item["ssim_improvement"] for item in metrics]
    )

    mean_low_psnr = float(np.mean(low_psnr))
    median_low_psnr = float(np.median(low_psnr))

    mean_low_ssim = float(np.mean(low_ssim))
    median_low_ssim = float(np.median(low_ssim))

    mean_gamma_psnr = float(np.mean(gamma_psnr))
    median_gamma_psnr = float(np.median(gamma_psnr))

    mean_gamma_ssim = float(np.mean(gamma_ssim))
    median_gamma_ssim = float(np.median(gamma_ssim))

    mean_psnr_improvement = float(
        np.mean(psnr_improvement)
    )

    median_psnr_improvement = float(
        np.median(psnr_improvement)
    )

    mean_ssim_improvement = float(
        np.mean(ssim_improvement)
    )

    median_ssim_improvement = float(
        np.median(ssim_improvement)
    )

    psnr_improved_count = int(
        np.sum(psnr_improvement > 0)
    )

    ssim_improved_count = int(
        np.sum(ssim_improvement > 0)
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "low_file",
                "high_file",
                "low_psnr",
                "low_ssim",
                "gamma_psnr",
                "gamma_ssim",
                "psnr_improvement",
                "ssim_improvement",
            ],
        )

        writer.writeheader()
        writer.writerows(metrics)

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary = f"""
TASK 14.2 — OFFICIAL TEST-SET GAMMA EVALUATION

Dataset:
LOLv2-real official test set

Number of test pairs:
100

Gamma:
{GAMMA}

LOW-LIGHT BASELINE
Mean PSNR: {mean_low_psnr:.4f} dB
Median PSNR: {median_low_psnr:.4f} dB
Mean SSIM: {mean_low_ssim:.6f}
Median SSIM: {median_low_ssim:.6f}

GAMMA CORRECTION
Mean PSNR: {mean_gamma_psnr:.4f} dB
Median PSNR: {median_gamma_psnr:.4f} dB
Mean SSIM: {mean_gamma_ssim:.6f}
Median SSIM: {median_gamma_ssim:.6f}

IMPROVEMENT
Mean PSNR change: {mean_psnr_improvement:+.4f} dB
Median PSNR change: {median_psnr_improvement:+.4f} dB
Mean SSIM change: {mean_ssim_improvement:+.6f}
Median SSIM change: {median_ssim_improvement:+.6f}

PSNR improved on:
{psnr_improved_count}/100 images

SSIM improved on:
{ssim_improved_count}/100 images
""".strip()

    OUTPUT_SUMMARY.write_text(
        summary,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("TASK 14.2 OFFICIAL TEST EVALUATION COMPLETED")
    print("=" * 60)

    print()
    print("FROZEN GAMMA")
    print(f"Gamma: {GAMMA}")

    print()
    print("LOW-LIGHT BASELINE")
    print(f"Mean PSNR: {mean_low_psnr:.4f} dB")
    print(f"Mean SSIM: {mean_low_ssim:.6f}")

    print()
    print("GAMMA CORRECTION")
    print(f"Mean PSNR: {mean_gamma_psnr:.4f} dB")
    print(f"Mean SSIM: {mean_gamma_ssim:.6f}")

    print()
    print("IMPROVEMENT")
    print(
        f"Mean PSNR change: "
        f"{mean_psnr_improvement:+.4f} dB"
    )

    print(
        f"Mean SSIM change: "
        f"{mean_ssim_improvement:+.6f}"
    )

    print()
    print(
        f"PSNR improved on "
        f"{psnr_improved_count}/100 images"
    )

    print(
        f"SSIM improved on "
        f"{ssim_improved_count}/100 images"
    )

    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(f"Visual results: {VISUAL_DIR}")


if __name__ == "__main__":
    main()