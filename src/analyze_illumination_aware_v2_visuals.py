from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "illumination_aware_v2_evaluation"
)

METRICS_FILE = (
    EVALUATION_ROOT
    / "test_metrics.csv"
)

EXISTING_VISUALS_ROOT = (
    EVALUATION_ROOT
    / "visuals"
)

DIAGNOSTICS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "illumination_aware_v2_diagnostics"
)

SELECTED_ROOT = (
    DIAGNOSTICS_ROOT
    / "visual_cases"
)

CASE_TABLE_FILE = (
    DIAGNOSTICS_ROOT
    / "visual_case_selection.csv"
)

SUMMARY_FILE = (
    DIAGNOSTICS_ROOT
    / "visual_case_summary.txt"
)


# ------------------------------------------------------------------
# Required CSV columns
# ------------------------------------------------------------------

REQUIRED_COLUMNS = {
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


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def require_path(path: Path, description: str) -> None:
    """
    Verify that a required path exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def select_unique_images(
    dataframe: pd.DataFrame,
    count: int,
    sort_column: str,
    ascending: bool,
) -> pd.DataFrame:
    """
    Select up to `count` unique images according to a metric.
    """
    selected = (
        dataframe
        .sort_values(
            by=sort_column,
            ascending=ascending,
        )
        .drop_duplicates(
            subset=["image"]
        )
        .head(count)
        .copy()
    )

    return selected


def add_case_records(
    records: list[dict],
    dataframe: pd.DataFrame,
    category: str,
    count: int,
    sort_column: str,
    ascending: bool,
) -> None:
    """
    Add selected image records while preventing duplicates.
    """
    selected = select_unique_images(
        dataframe=dataframe,
        count=count,
        sort_column=sort_column,
        ascending=ascending,
    )

    for _, row in selected.iterrows():

        image_name = row["image"]

        if any(
            record["image"] == image_name
            for record in records
        ):
            continue

        records.append(
            {
                "category": category,
                "image": image_name,
                "index": int(row["index"]),
                "low_brightness": float(
                    row["low_brightness"]
                )
                if "low_brightness" in row
                else None,
                "original_psnr": float(
                    row["original_psnr"]
                ),
                "v2_psnr": float(
                    row["v2_psnr"]
                ),
                "psnr_delta": float(
                    row["v2_minus_original_psnr"]
                ),
                "original_ssim": float(
                    row["original_ssim"]
                ),
                "v2_ssim": float(
                    row["v2_ssim"]
                ),
                "ssim_delta": float(
                    row["v2_minus_original_ssim"]
                ),
            }
        )


def find_visual_files(
    image_name: str,
) -> list[Path]:
    """
    Find existing evaluation visual files corresponding
    to the test image.

    The evaluator's visual directory may contain different
    naming conventions, so search recursively using the
    image stem.
    """
    image_stem = Path(image_name).stem.lower()

    matches = []

    if not EXISTING_VISUALS_ROOT.exists():
        return matches

    for path in EXISTING_VISUALS_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if image_stem in path.stem.lower():
            matches.append(path)

    return sorted(matches)


def copy_case_visuals(
    image_name: str,
    category: str,
    case_number: int,
) -> list[Path]:
    """
    Copy existing evaluation visuals for one selected image
    into an organized diagnostic directory.
    """
    source_files = find_visual_files(
        image_name
    )

    case_directory = (
        SELECTED_ROOT
        / f"{case_number:02d}_{category}"
        / Path(image_name).stem
    )

    case_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_files = []

    for source_file in source_files:

        destination_file = (
            case_directory
            / source_file.name
        )

        shutil.copy2(
            source_file,
            destination_file,
        )

        copied_files.append(
            destination_file
        )

    return copied_files


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "Task 24.8.2 — "
        "Illumination-Aware V2 Visual Diagnosis"
    )
    print("=" * 70)

    # --------------------------------------------------------------
    # Verify required files/directories
    # --------------------------------------------------------------

    require_path(
        METRICS_FILE,
        "Evaluation metrics file",
    )

    require_path(
        EXISTING_VISUALS_ROOT,
        "Existing evaluation visuals directory",
    )

    # --------------------------------------------------------------
    # Load metrics
    # --------------------------------------------------------------

    print("\nLoading evaluation metrics:")
    print(METRICS_FILE)

    metrics = pd.read_csv(
        METRICS_FILE
    )

    print(
        f"\nNumber of test images: "
        f"{len(metrics)}"
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    print(
        "CSV schema verification: PASSED"
    )

    # --------------------------------------------------------------
    # Verify existing delta columns
    # --------------------------------------------------------------

    calculated_psnr_delta = (
        metrics["v2_psnr"]
        - metrics["original_psnr"]
    )

    calculated_ssim_delta = (
        metrics["v2_ssim"]
        - metrics["original_ssim"]
    )

    max_psnr_difference = (
        calculated_psnr_delta
        - metrics["v2_minus_original_psnr"]
    ).abs().max()

    max_ssim_difference = (
        calculated_ssim_delta
        - metrics["v2_minus_original_ssim"]
    ).abs().max()

    print("\nDelta verification:")

    print(
        f"  Maximum PSNR difference: "
        f"{max_psnr_difference:.12f}"
    )

    print(
        f"  Maximum SSIM difference: "
        f"{max_ssim_difference:.12f}"
    )

    if max_psnr_difference > 1e-10:
        raise ValueError(
            "PSNR delta verification failed."
        )

    if max_ssim_difference > 1e-10:
        raise ValueError(
            "SSIM delta verification failed."
        )

    print(
        "  Delta verification: PASSED"
    )

    # --------------------------------------------------------------
    # Load brightness analysis if available
    # --------------------------------------------------------------

    brightness_analysis_file = (
        DIAGNOSTICS_ROOT
        / "per_image_analysis.csv"
    )

    if brightness_analysis_file.exists():

        print(
            "\nLoading brightness analysis:"
        )

        print(
            brightness_analysis_file
        )

        brightness_data = pd.read_csv(
            brightness_analysis_file
        )

        if {
            "image",
            "low_brightness",
        }.issubset(
            brightness_data.columns
        ):

            metrics = metrics.merge(
                brightness_data[
                    [
                        "image",
                        "low_brightness",
                    ]
                ],
                on="image",
                how="left",
            )

            print(
                "Brightness data merged: PASSED"
            )

        else:

            print(
                "Brightness analysis exists, "
                "but required columns are missing."
            )

    else:

        print(
            "\nBrightness analysis file not found."
        )

        print(
            "Continuing without brightness values."
        )

    # --------------------------------------------------------------
    # Create diagnostic directories
    # --------------------------------------------------------------

    SELECTED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Select representative cases
    # --------------------------------------------------------------

    records: list[dict] = []

    # --------------------------------------------------------------
    # 1. Strongest V2 PSNR improvements
    # --------------------------------------------------------------

    add_case_records(
        records=records,
        dataframe=metrics,
        category="v2_psnr_best",
        count=3,
        sort_column="v2_minus_original_psnr",
        ascending=False,
    )

    # --------------------------------------------------------------
    # 2. Largest V2 PSNR degradations
    # --------------------------------------------------------------

    add_case_records(
        records=records,
        dataframe=metrics,
        category="v2_psnr_worst",
        count=3,
        sort_column="v2_minus_original_psnr",
        ascending=True,
    )

    # --------------------------------------------------------------
    # 3. Strongest V2 SSIM improvements
    # --------------------------------------------------------------

    add_case_records(
        records=records,
        dataframe=metrics,
        category="v2_ssim_best",
        count=3,
        sort_column="v2_minus_original_ssim",
        ascending=False,
    )

    # --------------------------------------------------------------
    # 4. Largest V2 SSIM degradations
    # --------------------------------------------------------------

    add_case_records(
        records=records,
        dataframe=metrics,
        category="v2_ssim_worst",
        count=3,
        sort_column="v2_minus_original_ssim",
        ascending=True,
    )

    # --------------------------------------------------------------
    # 5. Darkest images
    # --------------------------------------------------------------

    if "low_brightness" in metrics.columns:

        add_case_records(
            records=records,
            dataframe=metrics,
            category="darkest",
            count=3,
            sort_column="low_brightness",
            ascending=True,
        )

        # ----------------------------------------------------------
        # 6. Brightest images
        # ----------------------------------------------------------

        add_case_records(
            records=records,
            dataframe=metrics,
            category="brightest",
            count=3,
            sort_column="low_brightness",
            ascending=False,
        )

    # --------------------------------------------------------------
    # Create selection table
    # --------------------------------------------------------------

    selection_dataframe = pd.DataFrame(
        records
    )

    if selection_dataframe.empty:
        raise RuntimeError(
            "No visual cases were selected."
        )

    selection_dataframe.insert(
        0,
        "case_number",
        range(
            1,
            len(selection_dataframe) + 1,
        ),
    )

    selection_dataframe.to_csv(
        CASE_TABLE_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Copy existing visual files
    # --------------------------------------------------------------

    print(
        "\nSelected visual cases:"
    )

    visual_copy_results = []

    for _, record in selection_dataframe.iterrows():

        case_number = int(
            record["case_number"]
        )

        category = record["category"]

        image_name = record["image"]

        copied_files = copy_case_visuals(
            image_name=image_name,
            category=category,
            case_number=case_number,
        )

        visual_copy_results.append(
            {
                "case_number": case_number,
                "category": category,
                "image": image_name,
                "visual_files_found": len(
                    copied_files
                ),
            }
        )

        print(
            f"  Case {case_number:02d}: "
            f"{category:<20} "
            f"{image_name:<20} "
            f"visual files: "
            f"{len(copied_files)}"
        )

    # --------------------------------------------------------------
    # Write summary
    # --------------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Task 24.8.2 — "
            "Illumination-Aware V2 Visual Diagnosis\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Purpose\n"
        )

        file.write(
            "-------\n"
        )

        file.write(
            "Select representative test images for visual "
            "comparison between the existing low-light input, "
            "Original DDPM output, Illumination-Aware V2 output, "
            "and ground-truth normal-light image.\n\n"
        )

        file.write(
            "IMPORTANT\n"
        )

        file.write(
            "This script does not retrain a model, does not "
            "run diffusion sampling, and does not modify any "
            "checkpoint or existing evaluation result.\n\n"
        )

        file.write(
            "SOURCE METRICS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"{METRICS_FILE}\n\n"
        )

        file.write(
            "SOURCE VISUALS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"{EXISTING_VISUALS_ROOT}\n\n"
        )

        file.write(
            "SELECTED CASES\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            selection_dataframe.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            "VISUAL FILE SEARCH RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        visual_results_dataframe = pd.DataFrame(
            visual_copy_results
        )

        file.write(
            visual_results_dataframe.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            "INTERPRETATION GUIDE\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            "For each selected case, compare:\n"
        )

        file.write(
            "1. Low-light input\n"
        )

        file.write(
            "2. Original DDPM output\n"
        )

        file.write(
            "3. Illumination-Aware V2 output\n"
        )

        file.write(
            "4. Ground-truth normal-light image\n\n"
        )

        file.write(
            "Pay particular attention to:\n"
        )

        file.write(
            "- overall brightness\n"
        )

        file.write(
            "- color consistency\n"
        )

        file.write(
            "- edge and structural preservation\n"
        )

        file.write(
            "- fine texture\n"
        )

        file.write(
            "- noise and artifacts\n"
        )

        file.write(
            "- over-smoothing\n"
        )

        file.write(
            "- unnatural illumination\n"
        )

        file.write(
            "- differences between dark and bright scenes\n"
        )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TASK 24.8.2 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSelected cases: "
        f"{len(selection_dataframe)}"
    )

    print(
        "\nCase selection table:"
    )

    print(
        CASE_TABLE_FILE
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nOrganized visual cases:"
    )

    print(
        SELECTED_ROOT
    )

    print(
        "\nCase selection:"
    )

    print(
        selection_dataframe.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
