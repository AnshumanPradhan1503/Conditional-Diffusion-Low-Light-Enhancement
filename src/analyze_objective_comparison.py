
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "objective_comparison"
    / "test_metrics.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "objective_comparison_analysis"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("TASK 15.5.7 — OBJECTIVE COMPARISON ANALYSIS")
print("=" * 70)

print(f"Input file: {INPUT_CSV}")
print(f"Images found: {len(df)}")


# ---------------------------------------------------------
# Required columns
# ---------------------------------------------------------

required_columns = [
    "original_psnr",
    "original_ssim",
    "x0_psnr",
    "x0_ssim",
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    print("\n[ERROR] Missing columns:")
    for column in missing:
        print(f"  - {column}")

    print("\nAvailable columns:")
    for column in df.columns:
        print(f"  - {column}")

    raise ValueError("Required columns are missing from CSV.")


# ---------------------------------------------------------
# Per-image differences
# ---------------------------------------------------------

df["psnr_difference"] = (
    df["x0_psnr"] - df["original_psnr"]
)

df["ssim_difference"] = (
    df["x0_ssim"] - df["original_ssim"]
)


# ---------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------

original_psnr_mean = df["original_psnr"].mean()
x0_psnr_mean = df["x0_psnr"].mean()

original_psnr_median = df["original_psnr"].median()
x0_psnr_median = df["x0_psnr"].median()

original_psnr_std = df["original_psnr"].std()
x0_psnr_std = df["x0_psnr"].std()

original_ssim_mean = df["original_ssim"].mean()
x0_ssim_mean = df["x0_ssim"].mean()

original_ssim_median = df["original_ssim"].median()
x0_ssim_median = df["x0_ssim"].median()

original_ssim_std = df["original_ssim"].std()
x0_ssim_std = df["x0_ssim"].std()


psnr_difference_mean = df["psnr_difference"].mean()
psnr_difference_median = df["psnr_difference"].median()
psnr_difference_std = df["psnr_difference"].std()

ssim_difference_mean = df["ssim_difference"].mean()
ssim_difference_median = df["ssim_difference"].median()
ssim_difference_std = df["ssim_difference"].std()


# ---------------------------------------------------------
# Win counts
# ---------------------------------------------------------

psnr_x0_wins = (df["psnr_difference"] > 0).sum()
psnr_original_wins = (df["psnr_difference"] < 0).sum()
psnr_ties = (df["psnr_difference"] == 0).sum()

ssim_x0_wins = (df["ssim_difference"] > 0).sum()
ssim_original_wins = (df["ssim_difference"] < 0).sum()
ssim_ties = (df["ssim_difference"] == 0).sum()

total = len(df)


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("PSNR")
print("-" * 70)

print(f"Original mean   : {original_psnr_mean:.6f} dB")
print(f"X0 mean         : {x0_psnr_mean:.6f} dB")
print(f"Original median : {original_psnr_median:.6f} dB")
print(f"X0 median       : {x0_psnr_median:.6f} dB")
print(f"Original std    : {original_psnr_std:.6f}")
print(f"X0 std          : {x0_psnr_std:.6f}")

print(f"\nMean difference : {psnr_difference_mean:+.6f} dB")
print(f"Median difference: {psnr_difference_median:+.6f} dB")
print(f"Difference std  : {psnr_difference_std:.6f}")

print(f"\nX0 wins         : {psnr_x0_wins}/{total}")
print(f"Original wins   : {psnr_original_wins}/{total}")
print(f"Ties            : {psnr_ties}/{total}")


print("\n" + "-" * 70)
print("SSIM")
print("-" * 70)

print(f"Original mean   : {original_ssim_mean:.6f}")
print(f"X0 mean         : {x0_ssim_mean:.6f}")
print(f"Original median : {original_ssim_median:.6f}")
print(f"X0 median       : {x0_ssim_median:.6f}")
print(f"Original std    : {original_ssim_std:.6f}")
print(f"X0 std          : {x0_ssim_std:.6f}")

print(f"\nMean difference : {ssim_difference_mean:+.6f}")
print(f"Median difference: {ssim_difference_median:+.6f}")
print(f"Difference std  : {ssim_difference_std:.6f}")

print(f"\nX0 wins         : {ssim_x0_wins}/{total}")
print(f"Original wins   : {ssim_original_wins}/{total}")
print(f"Ties            : {ssim_ties}/{total}")


# ---------------------------------------------------------
# Percentage calculations
# ---------------------------------------------------------

psnr_x0_win_pct = 100 * psnr_x0_wins / total
psnr_original_win_pct = 100 * psnr_original_wins / total

ssim_x0_win_pct = 100 * ssim_x0_wins / total
ssim_original_win_pct = 100 * ssim_original_wins / total


print("\n" + "-" * 70)
print("WIN PERCENTAGES")
print("-" * 70)

print(f"PSNR X0 wins       : {psnr_x0_win_pct:.2f}%")
print(f"PSNR Original wins : {psnr_original_win_pct:.2f}%")

print(f"SSIM X0 wins       : {ssim_x0_win_pct:.2f}%")
print(f"SSIM Original wins : {ssim_original_win_pct:.2f}%")


# ---------------------------------------------------------
# Strongest PSNR improvements
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TOP 10 X0 PSNR IMPROVEMENTS")
print("-" * 70)

top_psnr = df.sort_values(
    "psnr_difference",
    ascending=False,
).head(10)

print(
    top_psnr[
        [
            "psnr_difference",
            "original_psnr",
            "x0_psnr",
        ]
    ].to_string()
)


# ---------------------------------------------------------
# Strongest PSNR losses
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TOP 10 X0 PSNR LOSSES")
print("-" * 70)

worst_psnr = df.sort_values(
    "psnr_difference",
    ascending=True,
).head(10)

print(
    worst_psnr[
        [
            "psnr_difference",
            "original_psnr",
            "x0_psnr",
        ]
    ].to_string()
)


# ---------------------------------------------------------
# Strongest SSIM improvements
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TOP 10 X0 SSIM IMPROVEMENTS")
print("-" * 70)

top_ssim = df.sort_values(
    "ssim_difference",
    ascending=False,
).head(10)

print(
    top_ssim[
        [
            "ssim_difference",
            "original_ssim",
            "x0_ssim",
        ]
    ].to_string()
)


# ---------------------------------------------------------
# Strongest SSIM losses
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TOP 10 X0 SSIM LOSSES")
print("-" * 70)

worst_ssim = df.sort_values(
    "ssim_difference",
    ascending=True,
).head(10)

print(
    worst_ssim[
        [
            "ssim_difference",
            "original_ssim",
            "x0_ssim",
        ]
    ].to_string()
)


# ---------------------------------------------------------
# Save complete analysis CSV
# ---------------------------------------------------------

analysis_csv = OUTPUT_ROOT / "analysis_metrics.csv"

df.to_csv(
    analysis_csv,
    index=False,
)

print(f"\nSaved detailed analysis: {analysis_csv}")


# ---------------------------------------------------------
# Save summary text
# ---------------------------------------------------------

summary_path = OUTPUT_ROOT / "summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("TASK 15.5.7 — OBJECTIVE COMPARISON ANALYSIS\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Images evaluated: {total}\n\n")

    file.write("PSNR\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Original mean: {original_psnr_mean:.6f} dB\n"
    )
    file.write(
        f"X0 mean: {x0_psnr_mean:.6f} dB\n"
    )
    file.write(
        f"Mean difference: {psnr_difference_mean:+.6f} dB\n"
    )
    file.write(
        f"X0 wins: {psnr_x0_wins}/{total} "
        f"({psnr_x0_win_pct:.2f}%)\n"
    )
    file.write(
        f"Original wins: {psnr_original_wins}/{total} "
        f"({psnr_original_win_pct:.2f}%)\n"
    )

    file.write("\nSSIM\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Original mean: {original_ssim_mean:.6f}\n"
    )
    file.write(
        f"X0 mean: {x0_ssim_mean:.6f}\n"
    )
    file.write(
        f"Mean difference: {ssim_difference_mean:+.6f}\n"
    )
    file.write(
        f"X0 wins: {ssim_x0_wins}/{total} "
        f"({ssim_x0_win_pct:.2f}%)\n"
    )
    file.write(
        f"Original wins: {ssim_original_wins}/{total} "
        f"({ssim_original_win_pct:.2f}%)\n"
    )

    file.write("\n\nTOP PSNR X0 IMPROVEMENTS\n")
    file.write("-" * 70 + "\n")
    file.write(
        top_psnr[
            [
                "psnr_difference",
                "original_psnr",
                "x0_psnr",
            ]
        ].to_string()
    )

    file.write("\n\nTOP PSNR X0 LOSSES\n")
    file.write("-" * 70 + "\n")
    file.write(
        worst_psnr[
            [
                "psnr_difference",
                "original_psnr",
                "x0_psnr",
            ]
        ].to_string()
    )

    file.write("\n\nTOP SSIM X0 IMPROVEMENTS\n")
    file.write("-" * 70 + "\n")
    file.write(
        top_ssim[
            [
                "ssim_difference",
                "original_ssim",
                "x0_ssim",
            ]
        ].to_string()
    )

    file.write("\n\nTOP SSIM X0 LOSSES\n")
    file.write("-" * 70 + "\n")
    file.write(
        worst_ssim[
            [
                "ssim_difference",
                "original_ssim",
                "x0_ssim",
            ]
        ].to_string()
    )


print(f"Saved summary: {summary_path}")


# ---------------------------------------------------------
# Plot 1 — PSNR distributions
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["original_psnr"],
    bins=15,
    alpha=0.6,
    label="Original 32-channel",
)

plt.hist(
    df["x0_psnr"],
    bins=15,
    alpha=0.6,
    label="X0 objective",
)

plt.xlabel("PSNR (dB)")
plt.ylabel("Number of images")
plt.title("PSNR Distribution")
plt.legend()
plt.tight_layout()

psnr_plot = OUTPUT_ROOT / "psnr_distribution.png"
plt.savefig(psnr_plot, dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 2 — SSIM distributions
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["original_ssim"],
    bins=15,
    alpha=0.6,
    label="Original 32-channel",
)

plt.hist(
    df["x0_ssim"],
    bins=15,
    alpha=0.6,
    label="X0 objective",
)

plt.xlabel("SSIM")
plt.ylabel("Number of images")
plt.title("SSIM Distribution")
plt.legend()
plt.tight_layout()

ssim_plot = OUTPUT_ROOT / "ssim_distribution.png"
plt.savefig(ssim_plot, dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 3 — Per-image PSNR difference
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, total + 1),
    df["psnr_difference"],
    marker=".",
    linewidth=1,
)

plt.axhline(
    0,
    linestyle="--",
)

plt.xlabel("Test image")
plt.ylabel("X0 PSNR - Original PSNR (dB)")
plt.title("Per-image PSNR Difference")
plt.tight_layout()

psnr_diff_plot = OUTPUT_ROOT / "psnr_difference.png"
plt.savefig(psnr_diff_plot, dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 4 — Per-image SSIM difference
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, total + 1),
    df["ssim_difference"],
    marker=".",
    linewidth=1,
)

plt.axhline(
    0,
    linestyle="--",
)

plt.xlabel("Test image")
plt.ylabel("X0 SSIM - Original SSIM")
plt.title("Per-image SSIM Difference")
plt.tight_layout()

ssim_diff_plot = OUTPUT_ROOT / "ssim_difference.png"
plt.savefig(ssim_diff_plot, dpi=150)
plt.close()


print("\nSaved plots:")
print(f"  {psnr_plot}")
print(f"  {ssim_plot}")
print(f"  {psnr_diff_plot}")
print(f"  {ssim_diff_plot}")

print("\n" + "=" * 70)
print("TASK 15.5.7 ANALYSIS COMPLETED")
print("=" * 70)

