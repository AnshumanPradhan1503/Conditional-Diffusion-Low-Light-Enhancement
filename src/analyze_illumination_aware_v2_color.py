from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "illumination_aware_v2_evaluation"
)

METRICS_FILE = EVALUATION_ROOT / "test_metrics.csv"
VISUALS_ROOT = EVALUATION_ROOT / "visuals"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "illumination_aware_v2_diagnostics"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

# Existing controlled visual comparisons.
# We intentionally analyze ONLY these existing files.
VISUAL_INDICES = [0, 10, 25, 50, 75, 99]

# Each comparison image contains five equal-width panels:
#
# 1. Low-light
# 2. Gamma 0.4
# 3. Original DDPM
# 4. Illumination V2
# 5. Normal-light GT
#
# The labels above are part of the existing evaluation output.
NUM_PANELS = 5


# ============================================================
# Utility functions
# ============================================================

def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load an image as float32 RGB in [0, 1].
    """
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def calculate_channel_statistics(
    image: np.ndarray,
) -> dict[str, float]:
    """
    Calculate RGB and luminance statistics.

    Luminance:
        Y = 0.299 R + 0.587 G + 0.114 B
    """

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    luminance = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    return {
        "mean_r": float(red.mean()),
        "mean_g": float(green.mean()),
        "mean_b": float(blue.mean()),
        "std_r": float(red.std()),
        "std_g": float(green.std()),
        "std_b": float(blue.std()),
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
    }


def calculate_color_ratios(
    image: np.ndarray,
) -> dict[str, float]:
    """
    Calculate simple channel relationships.

    These are diagnostic statistics only.
    They are not intended as color-science metrics.
    """

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    eps = 1e-8

    return {
        "r_minus_g": float((red - green).mean()),
        "g_minus_b": float((green - blue).mean()),
        "r_minus_b": float((red - blue).mean()),
        "r_to_g": float(red.mean() / (green.mean() + eps)),
        "g_to_b": float(green.mean() / (blue.mean() + eps)),
        "r_to_b": float(red.mean() / (blue.mean() + eps)),
    }


def crop_comparison_panels(
    image: np.ndarray,
) -> list[np.ndarray]:
    """
    Split an existing comparison image into its five equal-width
    image panels.

    The evaluation visual layout is:

        Low-light
        Gamma 0.4
        Original DDPM
        Illumination V2
        Normal-light GT

    Text labels are above the image panels, so the actual image
    region is cropped by removing the upper label area.
    """

    height, width, _ = image.shape

    # Existing comparison images contain a label/header area.
    # Detect the main image region by using the lower ~82%.
    #
    # This is intentionally conservative and applies only to the
    # already-generated comparison files.
    image_start = int(height * 0.14)

    image_region = image[image_start:, :, :]

    panel_width = image_region.shape[1] // NUM_PANELS

    panels = []

    for panel_index in range(NUM_PANELS):
        start_x = panel_index * panel_width

        if panel_index == NUM_PANELS - 1:
            end_x = image_region.shape[1]
        else:
            end_x = (panel_index + 1) * panel_width

        panel = image_region[:, start_x:end_x, :]
        panels.append(panel)

    return panels


# ============================================================
# Load metrics
# ============================================================

print("=" * 70)
print("Illumination-Aware V2 Color / Photometric Diagnostic")
print("=" * 70)

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Metrics file not found:\n{METRICS_FILE}"
    )

metrics = pd.read_csv(METRICS_FILE)

required_columns = {
    "index",
    "image",
    "original_psnr",
    "original_ssim",
    "v2_psnr",
    "v2_ssim",
    "v2_minus_original_psnr",
    "v2_minus_original_ssim",
}

missing_columns = required_columns - set(metrics.columns)

if missing_columns:
    raise ValueError(
        "Metrics CSV is missing required columns:\n"
        f"{sorted(missing_columns)}"
    )

print("\nMetrics CSV verification: PASSED")
print(f"Total metric rows: {len(metrics)}")


# ============================================================
# Verify existing visual files
# ============================================================

print("\nChecking existing controlled visual files...")

visual_files = {}

for index in VISUAL_INDICES:
    path = VISUALS_ROOT / f"comparison_{index:03d}.png"

    if not path.exists():
        print(f"WARNING: Missing visual: {path}")
        continue

    visual_files[index] = path

    print(
        f"  {path.name} "
        f"({path.stat().st_size:,} bytes)"
    )

print(
    f"\nExisting controlled visuals available: "
    f"{len(visual_files)}/{len(VISUAL_INDICES)}"
)


# ============================================================
# Analyze the six existing visual comparisons
# ============================================================

panel_names = [
    "low_light",
    "gamma",
    "original",
    "v2",
    "ground_truth",
]

visual_results = []

for index, visual_path in visual_files.items():

    print(
        f"\nAnalyzing visual comparison index {index}: "
        f"{visual_path.name}"
    )

    full_image = load_rgb_image(visual_path)

    panels = crop_comparison_panels(full_image)

    if len(panels) != NUM_PANELS:
        raise RuntimeError(
            f"Expected {NUM_PANELS} panels, "
            f"got {len(panels)}."
        )

    row = {
        "index": index,
    }

    metric_row = metrics.loc[
        metrics["index"] == index
    ]

    if len(metric_row) != 1:
        print(
            f"WARNING: Expected one metric row for index "
            f"{index}, found {len(metric_row)}."
        )
    else:
        metric_row = metric_row.iloc[0]

        row["image"] = metric_row["image"]

        row["original_psnr"] = float(
            metric_row["original_psnr"]
        )

        row["original_ssim"] = float(
            metric_row["original_ssim"]
        )

        row["v2_psnr"] = float(
            metric_row["v2_psnr"]
        )

        row["v2_ssim"] = float(
            metric_row["v2_ssim"]
        )

        row["v2_minus_original_psnr"] = float(
            metric_row["v2_minus_original_psnr"]
        )

        row["v2_minus_original_ssim"] = float(
            metric_row["v2_minus_original_ssim"]
        )

    panel_stats = {}

    for name, panel in zip(panel_names, panels):

        statistics = calculate_channel_statistics(panel)
        ratios = calculate_color_ratios(panel)

        for key, value in statistics.items():
            row[f"{name}_{key}"] = value

        for key, value in ratios.items():
            row[f"{name}_{key}"] = value

        panel_stats[name] = {
            **statistics,
            **ratios,
        }

    # --------------------------------------------------------
    # V2 versus GT
    # --------------------------------------------------------

    row["v2_gt_mean_r_error"] = abs(
        panel_stats["v2"]["mean_r"]
        - panel_stats["ground_truth"]["mean_r"]
    )

    row["v2_gt_mean_g_error"] = abs(
        panel_stats["v2"]["mean_g"]
        - panel_stats["ground_truth"]["mean_g"]
    )

    row["v2_gt_mean_b_error"] = abs(
        panel_stats["v2"]["mean_b"]
        - panel_stats["ground_truth"]["mean_b"]
    )

    row["v2_gt_mean_luminance_error"] = abs(
        panel_stats["v2"]["mean_luminance"]
        - panel_stats["ground_truth"]["mean_luminance"]
    )

    row["v2_gt_std_luminance_error"] = abs(
        panel_stats["v2"]["std_luminance"]
        - panel_stats["ground_truth"]["std_luminance"]
    )

    # --------------------------------------------------------
    # Original versus GT
    # --------------------------------------------------------

    row["original_gt_mean_r_error"] = abs(
        panel_stats["original"]["mean_r"]
        - panel_stats["ground_truth"]["mean_r"]
    )

    row["original_gt_mean_g_error"] = abs(
        panel_stats["original"]["mean_g"]
        - panel_stats["ground_truth"]["mean_g"]
    )

    row["original_gt_mean_b_error"] = abs(
        panel_stats["original"]["mean_b"]
        - panel_stats["ground_truth"]["mean_b"]
    )

    row["original_gt_mean_luminance_error"] = abs(
        panel_stats["original"]["mean_luminance"]
        - panel_stats["ground_truth"]["mean_luminance"]
    )

    row["original_gt_std_luminance_error"] = abs(
        panel_stats["original"]["std_luminance"]
        - panel_stats["ground_truth"]["std_luminance"]
    )

    visual_results.append(row)


# ============================================================
# Save visual color analysis
# ============================================================

visual_analysis = pd.DataFrame(visual_results)

visual_output = (
    OUTPUT_ROOT
    / "existing_visual_color_analysis.csv"
)

visual_analysis.to_csv(
    visual_output,
    index=False,
)

print(
    f"\nSaved visual color analysis:\n"
    f"{visual_output}"
)


# ============================================================
# Summary calculations
# ============================================================

summary_lines = []

summary_lines.append(
    "Illumination-Aware V2 Color / Photometric Diagnostic"
)

summary_lines.append("=" * 70)

summary_lines.append(
    ""
)

summary_lines.append(
    "IMPORTANT:"
)

summary_lines.append(
    "This diagnostic analyzes only the six existing controlled "
    "comparison images."
)

summary_lines.append(
    "It does NOT generate new diffusion samples."
)

summary_lines.append(
    "It does NOT modify or retrain any model."
)

summary_lines.append(
    ""
)

summary_lines.append(
    f"Visual comparisons analyzed: {len(visual_analysis)}"
)

summary_lines.append(
    f"Metric rows available: {len(metrics)}"
)


if len(visual_analysis) > 0:

    # --------------------------------------------------------
    # Mean channel errors
    # --------------------------------------------------------

    summary_lines.append("")
    summary_lines.append("V2 versus Ground Truth")
    summary_lines.append("-" * 70)

    for channel in ["r", "g", "b"]:

        column = f"v2_gt_mean_{channel}_error"

        mean_error = visual_analysis[column].mean()

        summary_lines.append(
            f"Mean absolute {channel.upper()} mean error: "
            f"{mean_error:.6f}"
        )

    luminance_error = visual_analysis[
        "v2_gt_mean_luminance_error"
    ].mean()

    summary_lines.append(
        f"Mean absolute luminance mean error: "
        f"{luminance_error:.6f}"
    )

    # --------------------------------------------------------
    # Original versus GT
    # --------------------------------------------------------

    summary_lines.append("")
    summary_lines.append("Original DDPM versus Ground Truth")
    summary_lines.append("-" * 70)

    for channel in ["r", "g", "b"]:

        column = f"original_gt_mean_{channel}_error"

        mean_error = visual_analysis[column].mean()

        summary_lines.append(
            f"Mean absolute {channel.upper()} mean error: "
            f"{mean_error:.6f}"
        )

    luminance_error = visual_analysis[
        "original_gt_mean_luminance_error"
    ].mean()

    summary_lines.append(
        f"Mean absolute luminance mean error: "
        f"{luminance_error:.6f}"
    )

    # --------------------------------------------------------
    # Compare color error
    # --------------------------------------------------------

    summary_lines.append("")
    summary_lines.append(
        "Mean channel-error comparison"
    )
    summary_lines.append("-" * 70)

    v2_rgb_error = visual_analysis[
        [
            "v2_gt_mean_r_error",
            "v2_gt_mean_g_error",
            "v2_gt_mean_b_error",
        ]
    ].mean(axis=1).mean()

    original_rgb_error = visual_analysis[
        [
            "original_gt_mean_r_error",
            "original_gt_mean_g_error",
            "original_gt_mean_b_error",
        ]
    ].mean(axis=1).mean()

    summary_lines.append(
        f"Original mean RGB-channel error: "
        f"{original_rgb_error:.6f}"
    )

    summary_lines.append(
        f"V2 mean RGB-channel error: "
        f"{v2_rgb_error:.6f}"
    )

    if v2_rgb_error > original_rgb_error:
        summary_lines.append(
            "Observation: V2 has larger mean RGB-channel "
            "error than Original on the analyzed visual cases."
        )
    else:
        summary_lines.append(
            "Observation: V2 does not have larger mean "
            "RGB-channel error than Original on the analyzed "
            "visual cases."
        )

    # --------------------------------------------------------
    # Luminance comparison
    # --------------------------------------------------------

    v2_luminance_error = visual_analysis[
        "v2_gt_mean_luminance_error"
    ].mean()

    original_luminance_error = visual_analysis[
        "original_gt_mean_luminance_error"
    ].mean()

    summary_lines.append("")
    summary_lines.append(
        "Luminance-error comparison"
    )
    summary_lines.append("-" * 70)

    summary_lines.append(
        f"Original mean luminance error: "
        f"{original_luminance_error:.6f}"
    )

    summary_lines.append(
        f"V2 mean luminance error: "
        f"{v2_luminance_error:.6f}"
    )

    if v2_luminance_error < original_luminance_error:
        summary_lines.append(
            "Observation: V2 has lower mean luminance "
            "error than Original on the analyzed visual cases."
        )
    else:
        summary_lines.append(
            "Observation: V2 does not have lower mean "
            "luminance error than Original on the analyzed "
            "visual cases."
        )


# ============================================================
# Save summary
# ============================================================

summary_output = (
    OUTPUT_ROOT
    / "color_analysis_summary.txt"
)

summary_output.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print(
    f"\nSaved summary:\n"
    f"{summary_output}"
)

print("\n" + "=" * 70)
print("Task 24.8.3 diagnostic completed.")
print("=" * 70)

print(
    "\nNo checkpoint was modified."
)

print(
    "No model was retrained."
)

print(
    "No new diffusion samples were generated."
)