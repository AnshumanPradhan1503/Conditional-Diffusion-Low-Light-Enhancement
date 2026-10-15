from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.configs import illumination_aware_v2_training_config as config


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_ROOT = (
    PROJECT_ROOT / "results" / "illumination_aware_v2_evaluation"
)

METRICS_FILE = EVALUATION_ROOT / "test_metrics.csv"

OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "illumination_aware_v2_diagnostics"
)

ANALYSIS_FILE = OUTPUT_ROOT / "per_image_analysis.csv"

SUMMARY_FILE = (
    OUTPUT_ROOT / "brightness_analysis_summary.txt"
)


# ------------------------------------------------------------------
# Brightness calculation
# ------------------------------------------------------------------

def calculate_mean_brightness(image_path: Path) -> float:
    """
    Calculate mean luminance brightness of an RGB image.

    Y = 0.299R + 0.587G + 0.114B

    Returns brightness in the original uint8 range [0, 255].
    """
    image = Image.open(image_path).convert("RGB")

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    luminance = (
        0.299 * image_array[:, :, 0]
        + 0.587 * image_array[:, :, 1]
        + 0.114 * image_array[:, :, 2]
    )

    return float(luminance.mean())


# ------------------------------------------------------------------
# Test image lookup
# ------------------------------------------------------------------

def find_test_low_image(image_name: str) -> Path:
    """
    Locate the corresponding low-light test image.
    """
    image_path = (
        config.DATA_ROOT
        / "test"
        / "low"
        / image_name
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Low-light image not found:\n{image_path}"
        )

    return image_path


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "Illumination-Aware V2 — "
        "Per-Image & Brightness Analysis"
    )
    print("=" * 70)

    # --------------------------------------------------------------
    # Check metrics file
    # --------------------------------------------------------------

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found:\n{METRICS_FILE}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading metrics:")
    print(METRICS_FILE)

    metrics = pd.read_csv(
        METRICS_FILE
    )

    print(
        f"\nNumber of test images: "
        f"{len(metrics)}"
    )

    # --------------------------------------------------------------
    # Verify actual CSV schema
    # --------------------------------------------------------------

    required_columns = {
        "index",
        "image",
        "gamma_psnr",
        "gamma_ssim",
        "original_psnr",
        "original_ssim",
        "v2_psnr",
        "v2_ssim",
        "v2_minus_original_psnr",
        "v2_minus_original_ssim",
    }

    missing_columns = (
        required_columns
        - set(metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns in "
            "test_metrics.csv:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    print("\nCSV schema verification: PASSED")

    # --------------------------------------------------------------
    # Independently calculate performance deltas
    # --------------------------------------------------------------

    metrics["psnr_delta"] = (
        metrics["v2_psnr"]
        - metrics["original_psnr"]
    )

    metrics["ssim_delta"] = (
        metrics["v2_ssim"]
        - metrics["original_ssim"]
    )

    # --------------------------------------------------------------
    # Verify existing evaluation deltas
    # --------------------------------------------------------------

    psnr_delta_difference = np.abs(
        metrics["psnr_delta"]
        - metrics["v2_minus_original_psnr"]
    )

    ssim_delta_difference = np.abs(
        metrics["ssim_delta"]
        - metrics["v2_minus_original_ssim"]
    )

    max_psnr_delta_difference = float(
        psnr_delta_difference.max()
    )

    max_ssim_delta_difference = float(
        ssim_delta_difference.max()
    )

    print(
        "\nDelta consistency check:"
    )

    print(
        f"  Maximum PSNR delta difference: "
        f"{max_psnr_delta_difference:.12f}"
    )

    print(
        f"  Maximum SSIM delta difference: "
        f"{max_ssim_delta_difference:.12f}"
    )

    if max_psnr_delta_difference > 1e-10:
        raise ValueError(
            "PSNR delta consistency check failed."
        )

    if max_ssim_delta_difference > 1e-10:
        raise ValueError(
            "SSIM delta consistency check failed."
        )

    print(
        "  Delta consistency check: PASSED"
    )

    # --------------------------------------------------------------
    # Calculate low-light brightness
    # --------------------------------------------------------------

    brightness_values = []

    print(
        "\nCalculating low-light image brightness..."
    )

    for index, row in metrics.iterrows():

        image_name = row["image"]

        image_path = find_test_low_image(
            image_name
        )

        brightness = calculate_mean_brightness(
            image_path
        )

        brightness_values.append(
            brightness
        )

        if (
            (index + 1) % 10 == 0
            or index == 0
        ):
            print(
                f"  Processed "
                f"{index + 1}/{len(metrics)} "
                f"images"
            )

    metrics["low_brightness"] = (
        brightness_values
    )

    # --------------------------------------------------------------
    # Create brightness quartiles
    # --------------------------------------------------------------

    metrics["brightness_group"] = pd.qcut(
        metrics["low_brightness"],
        q=4,
        labels=[
            "Q1_darkest",
            "Q2",
            "Q3",
            "Q4_brightest",
        ],
        duplicates="drop",
    )

    # --------------------------------------------------------------
    # Save per-image analysis
    # --------------------------------------------------------------

    preferred_columns = [
        "index",
        "image",
        "low_brightness",
        "brightness_group",
        "gamma_psnr",
        "gamma_ssim",
        "original_psnr",
        "v2_psnr",
        "psnr_delta",
        "original_ssim",
        "v2_ssim",
        "ssim_delta",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in metrics.columns
    ]

    metrics[
        existing_columns
    ].to_csv(
        ANALYSIS_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------------

    psnr_wins = int(
        (metrics["psnr_delta"] > 0).sum()
    )

    psnr_losses = int(
        (metrics["psnr_delta"] < 0).sum()
    )

    psnr_ties = int(
        (metrics["psnr_delta"] == 0).sum()
    )

    ssim_wins = int(
        (metrics["ssim_delta"] > 0).sum()
    )

    ssim_losses = int(
        (metrics["ssim_delta"] < 0).sum()
    )

    ssim_ties = int(
        (metrics["ssim_delta"] == 0).sum()
    )

    # --------------------------------------------------------------
    # Brightness / performance correlations
    # --------------------------------------------------------------

    brightness_psnr_corr = metrics[
        [
            "low_brightness",
            "psnr_delta",
        ]
    ].corr().iloc[0, 1]

    brightness_ssim_corr = metrics[
        [
            "low_brightness",
            "ssim_delta",
        ]
    ].corr().iloc[0, 1]

    # --------------------------------------------------------------
    # Brightness quartile analysis
    # --------------------------------------------------------------

    quartile_summary = (
        metrics
        .groupby(
            "brightness_group",
            observed=True,
        )
        .agg(
            image_count=(
                "image",
                "count",
            ),
            mean_brightness=(
                "low_brightness",
                "mean",
            ),
            mean_original_psnr=(
                "original_psnr",
                "mean",
            ),
            mean_v2_psnr=(
                "v2_psnr",
                "mean",
            ),
            mean_psnr_delta=(
                "psnr_delta",
                "mean",
            ),
            mean_original_ssim=(
                "original_ssim",
                "mean",
            ),
            mean_v2_ssim=(
                "v2_ssim",
                "mean",
            ),
            mean_ssim_delta=(
                "ssim_delta",
                "mean",
            ),
            psnr_wins=(
                "psnr_delta",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            ssim_wins=(
                "ssim_delta",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------------
    # Darkest images
    # --------------------------------------------------------------

    darkest = metrics.nsmallest(
        10,
        "low_brightness",
    )[
        [
            "image",
            "low_brightness",
            "original_psnr",
            "v2_psnr",
            "psnr_delta",
            "original_ssim",
            "v2_ssim",
            "ssim_delta",
        ]
    ]

    # --------------------------------------------------------------
    # Brightest images
    # --------------------------------------------------------------

    brightest = metrics.nlargest(
        10,
        "low_brightness",
    )[
        [
            "image",
            "low_brightness",
            "original_psnr",
            "v2_psnr",
            "psnr_delta",
            "original_ssim",
            "v2_ssim",
            "ssim_delta",
        ]
    ]

    # --------------------------------------------------------------
    # Largest PSNR improvements
    # --------------------------------------------------------------

    largest_psnr_improvements = metrics.nlargest(
        10,
        "psnr_delta",
    )[
        [
            "image",
            "low_brightness",
            "original_psnr",
            "v2_psnr",
            "psnr_delta",
            "original_ssim",
            "v2_ssim",
            "ssim_delta",
        ]
    ]

    # --------------------------------------------------------------
    # Largest PSNR degradations
    # --------------------------------------------------------------

    largest_psnr_degradations = metrics.nsmallest(
        10,
        "psnr_delta",
    )[
        [
            "image",
            "low_brightness",
            "original_psnr",
            "v2_psnr",
            "psnr_delta",
            "original_ssim",
            "v2_ssim",
            "ssim_delta",
        ]
    ]

    # --------------------------------------------------------------
    # Write summary
    # --------------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Illumination-Aware V2 — "
            "Per-Image & Brightness Analysis\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Number of test images: "
            f"{len(metrics)}\n\n"
        )

        # ----------------------------------------------------------
        # Overall performance
        # ----------------------------------------------------------

        file.write(
            "OVERALL PERFORMANCE DIFFERENCES\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"PSNR wins:   "
            f"{psnr_wins}/{len(metrics)}\n"
        )

        file.write(
            f"PSNR losses: "
            f"{psnr_losses}/{len(metrics)}\n"
        )

        file.write(
            f"PSNR ties:   "
            f"{psnr_ties}/{len(metrics)}\n\n"
        )

        file.write(
            f"SSIM wins:   "
            f"{ssim_wins}/{len(metrics)}\n"
        )

        file.write(
            f"SSIM losses: "
            f"{ssim_losses}/{len(metrics)}\n"
        )

        file.write(
            f"SSIM ties:   "
            f"{ssim_ties}/{len(metrics)}\n\n"
        )

        file.write(
            f"Mean PSNR delta: "
            f"{metrics['psnr_delta'].mean():.6f} dB\n"
        )

        file.write(
            f"Mean SSIM delta: "
            f"{metrics['ssim_delta'].mean():.6f}\n\n"
        )

        # ----------------------------------------------------------
        # Brightness statistics
        # ----------------------------------------------------------

        file.write(
            "BRIGHTNESS STATISTICS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Minimum brightness: "
            f"{metrics['low_brightness'].min():.6f}\n"
        )

        file.write(
            f"Maximum brightness: "
            f"{metrics['low_brightness'].max():.6f}\n"
        )

        file.write(
            f"Mean brightness: "
            f"{metrics['low_brightness'].mean():.6f}\n"
        )

        file.write(
            f"Median brightness: "
            f"{metrics['low_brightness'].median():.6f}\n\n"
        )

        # ----------------------------------------------------------
        # Correlations
        # ----------------------------------------------------------

        file.write(
            "BRIGHTNESS / PERFORMANCE CORRELATION\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Brightness vs PSNR delta correlation: "
            f"{brightness_psnr_corr:.6f}\n"
        )

        file.write(
            f"Brightness vs SSIM delta correlation: "
            f"{brightness_ssim_corr:.6f}\n\n"
        )

        # ----------------------------------------------------------
        # Quartile analysis
        # ----------------------------------------------------------

        file.write(
            "BRIGHTNESS QUARTILE ANALYSIS\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            quartile_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        file.write("\n\n")

        # ----------------------------------------------------------
        # Darkest images
        # ----------------------------------------------------------

        file.write(
            "10 DARKEST TEST IMAGES\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            darkest.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        file.write("\n\n")

        # ----------------------------------------------------------
        # Brightest images
        # ----------------------------------------------------------

        file.write(
            "10 BRIGHTEST TEST IMAGES\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            brightest.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        file.write("\n\n")

        # ----------------------------------------------------------
        # Largest improvements
        # ----------------------------------------------------------

        file.write(
            "10 LARGEST V2 PSNR IMPROVEMENTS\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            largest_psnr_improvements.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        file.write("\n\n")

        # ----------------------------------------------------------
        # Largest degradations
        # ----------------------------------------------------------

        file.write(
            "10 LARGEST V2 PSNR DEGRADATIONS\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            largest_psnr_degradations.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        file.write("\n\n")

    # --------------------------------------------------------------
    # Console output
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nPer-image analysis:"
    )

    print(
        ANALYSIS_FILE
    )

    print(
        f"\nSummary:"
    )

    print(
        SUMMARY_FILE
    )

    print("\nOverall:")

    print(
        f"  PSNR wins   : "
        f"{psnr_wins}/{len(metrics)}"
    )

    print(
        f"  PSNR losses : "
        f"{psnr_losses}/{len(metrics)}"
    )

    print(
        f"  SSIM wins   : "
        f"{ssim_wins}/{len(metrics)}"
    )

    print(
        f"  SSIM losses : "
        f"{ssim_losses}/{len(metrics)}"
    )

    print(
        f"\nMean PSNR delta: "
        f"{metrics['psnr_delta'].mean():.6f} dB"
    )

    print(
        f"Mean SSIM delta: "
        f"{metrics['ssim_delta'].mean():.6f}"
    )

    print(
        f"\nBrightness vs PSNR delta correlation: "
        f"{brightness_psnr_corr:.6f}"
    )

    print(
        f"Brightness vs SSIM delta correlation: "
        f"{brightness_ssim_corr:.6f}"
    )

    print("\nBrightness quartiles:")

    print(
        quartile_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
