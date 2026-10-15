from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.configs import adaptive_spatial_training_config as config
from src.models.adaptive_spatial_unet import AdaptiveSpatialConditionalUNet
from src.preprocessing.dataloader import create_dataloaders
from src.diffusion.schedule import get_diffusion_parameters


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(config.DEVICE)

CHECKPOINT_PATH = (
    config.PROJECT_ROOT
    / "checkpoints"
    / "adaptive_spatial"
    / "best_model.pt"
)

EXISTING_METRICS_PATH = (
    config.PROJECT_ROOT
    / "results"
    / "adaptive_spatial_evaluation"
    / "test_metrics.csv"
)

OUTPUT_ROOT = (
    config.PROJECT_ROOT
    / "results"
    / "adaptive_spatial_gate_analysis"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GATE_METRICS_PATH = OUTPUT_ROOT / "gate_metrics.csv"
BRIGHTNESS_SUMMARY_PATH = OUTPUT_ROOT / "gate_brightness_summary.csv"
CORRELATION_PATH = OUTPUT_ROOT / "gate_correlations.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.txt"

# Controlled diagnostic timestep.
DIAGNOSTIC_TIMESTEP = 500

# Fixed diagnostic seed.
SEED = config.RANDOM_SEED


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

def create_model(device: torch.device) -> AdaptiveSpatialConditionalUNet:
    model = AdaptiveSpatialConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ============================================================
# Dataset
# ============================================================

def create_test_dataset():
    """
    Load the official test dataset using the same project
    dataloader and split configuration used by evaluation.
    """

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    return test_loader.dataset


# ============================================================
# Helpers
# ============================================================

def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def calculate_brightness(image: torch.Tensor) -> float:
    """
    Input:
        image in [-1, 1], shape [3, H, W]

    Output:
        mean brightness on 0-255 scale.
    """

    image_01 = (image + 1.0) / 2.0

    image_01 = torch.clamp(
        image_01,
        0.0,
        1.0,
    )

    brightness = image_01.mean().item() * 255.0

    return brightness


def calculate_gate_statistics(
    gate: torch.Tensor,
) -> tuple[float, float]:
    """
    Returns:
        mean gate activation
        standard deviation of gate activation
    """

    gate_mean = gate.mean().item()
    gate_std = gate.std().item()

    return gate_mean, gate_std


# ============================================================
# Main diagnostic
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ADAPTIVE SPATIAL GATE-BEHAVIOR DIAGNOSTICS")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Diagnostic timestep: {DIAGNOSTIC_TIMESTEP}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Existing metrics: {EXISTING_METRICS_PATH}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print()

    set_seed(SEED)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading Adaptive Spatial U-Net...")

    model = create_model(DEVICE)

    print(
        f"Model parameters: {count_parameters(model):,}"
    )

    print("Model loaded successfully.")
    print()

    # --------------------------------------------------------
    # Load official test dataset
    # --------------------------------------------------------

    print("Loading official test dataset...")

    test_dataset = create_test_dataset()

    print(
        f"Test images: {len(test_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Load existing evaluation metrics
    # --------------------------------------------------------

    if not EXISTING_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Existing metrics file not found:\n"
            f"{EXISTING_METRICS_PATH}"
        )

    existing_metrics = pd.read_csv(
        EXISTING_METRICS_PATH
    )

    print(
        f"Existing metric rows: {len(existing_metrics)}"
    )

    print()

    # --------------------------------------------------------
    # Diffusion parameters
    # --------------------------------------------------------

    _, _, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    alpha_bar = alpha_bars[
        DIAGNOSTIC_TIMESTEP
    ].to(DEVICE)

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar
    )

    # --------------------------------------------------------
    # Diagnostic forward passes
    # --------------------------------------------------------

    records = []

    print("Collecting adaptive gate statistics...")
    print()

    for index in range(len(test_dataset)):

        low_light, normal_light = test_dataset[index]

        low_light_batch = (
            low_light
            .unsqueeze(0)
            .to(DEVICE)
        )

        normal_light_batch = (
            normal_light
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Deterministic noise
        # ----------------------------------------------------

        generator = torch.Generator(
            device=DEVICE
        )

        generator.manual_seed(
            SEED + index
        )

        noise = torch.randn(
            low_light_batch.shape,
            generator=generator,
            device=DEVICE,
            dtype=low_light_batch.dtype,
        )

        # ----------------------------------------------------
        # Create x_t at diagnostic timestep
        # ----------------------------------------------------

        timestep = torch.tensor(
            [DIAGNOSTIC_TIMESTEP],
            device=DEVICE,
            dtype=torch.long,
        )

        noisy_target = (
            sqrt_alpha_bar * normal_light_batch
            + sqrt_one_minus_alpha_bar * noise
        )

        # ----------------------------------------------------
        # Adaptive model forward pass
        # ----------------------------------------------------

        with torch.no_grad():

            _ = model(
                noisy_image=noisy_target,
                condition=low_light_batch,
                timesteps=timestep,
            )

        # ----------------------------------------------------
        # Read stored adaptive gates
        # ----------------------------------------------------

        gates = model.last_gates

        gate_256 = gates["256"]
        gate_128 = gates["128"]
        gate_64 = gates["64"]

        gate_256_mean, gate_256_std = (
            calculate_gate_statistics(gate_256)
        )

        gate_128_mean, gate_128_std = (
            calculate_gate_statistics(gate_128)
        )

        gate_64_mean, gate_64_std = (
            calculate_gate_statistics(gate_64)
        )

        brightness = calculate_brightness(
            low_light
        )

        low_name = Path(
            test_dataset.pairs[index][0]
        ).name

        records.append(
            {
                "index": index,
                "image": low_name,
                "brightness": brightness,

                "gate_256_mean": gate_256_mean,
                "gate_256_std": gate_256_std,

                "gate_128_mean": gate_128_mean,
                "gate_128_std": gate_128_std,

                "gate_64_mean": gate_64_mean,
                "gate_64_std": gate_64_std,
            }
        )

        if (index + 1) % 10 == 0:
            print(
                f"Processed {index + 1:3d} / "
                f"{len(test_dataset)}"
            )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    gate_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Merge existing evaluation metrics
    # --------------------------------------------------------

    image_column = None

    possible_image_columns = [
        "image",
        "low_image",
        "filename",
        "low_name",
    ]

    for column in possible_image_columns:
        if column in existing_metrics.columns:
            image_column = column
            break

    if image_column is not None:

        metrics_for_merge = existing_metrics.copy()

        if image_column != "image":
            metrics_for_merge = (
                metrics_for_merge.rename(
                    columns={
                        image_column: "image"
                    }
                )
            )

        useful_columns = [
            column
            for column in [
                "image",
                "original_psnr",
                "adaptive_psnr",
                "original_ssim",
                "adaptive_ssim",
                "delta_psnr",
                "delta_ssim",
            ]
            if column in metrics_for_merge.columns
        ]

        gate_df = gate_df.merge(
            metrics_for_merge[useful_columns],
            on="image",
            how="left",
        )

    # --------------------------------------------------------
    # Save gate metrics
    # --------------------------------------------------------

    gate_df.to_csv(
        GATE_METRICS_PATH,
        index=False,
    )

    print()
    print(
        f"Saved gate metrics:\n"
        f"{GATE_METRICS_PATH}"
    )

    # --------------------------------------------------------
    # Brightness quartiles
    # --------------------------------------------------------

    gate_df["brightness_quartile"] = pd.qcut(
        gate_df["brightness"],
        q=4,
        labels=[
            "Q1_darkest",
            "Q2",
            "Q3",
            "Q4_brightest",
        ],
    )

    brightness_summary = (
        gate_df
        .groupby(
            "brightness_quartile",
            observed=False,
        )
        .agg(
            count=("image", "count"),

            mean_brightness=(
                "brightness",
                "mean",
            ),

            gate_256_mean=(
                "gate_256_mean",
                "mean",
            ),

            gate_128_mean=(
                "gate_128_mean",
                "mean",
            ),

            gate_64_mean=(
                "gate_64_mean",
                "mean",
            ),

            gate_256_std=(
                "gate_256_mean",
                "std",
            ),

            gate_128_std=(
                "gate_128_mean",
                "std",
            ),

            gate_64_std=(
                "gate_64_mean",
                "std",
            ),
        )
        .reset_index()
    )

    brightness_summary.to_csv(
        BRIGHTNESS_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_records = []

    correlation_targets = [
        (
            "gate_256_mean",
            "brightness",
        ),
        (
            "gate_128_mean",
            "brightness",
        ),
        (
            "gate_64_mean",
            "brightness",
        ),
    ]

    for gate_column, target_column in correlation_targets:

        valid = gate_df[
            [gate_column, target_column]
        ].dropna()

        if len(valid) < 3:
            continue

        pearson_r, pearson_p = pearsonr(
            valid[gate_column],
            valid[target_column],
        )

        spearman_r, spearman_p = spearmanr(
            valid[gate_column],
            valid[target_column],
        )

        correlation_records.append(
            {
                "gate": gate_column,
                "target": target_column,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n": len(valid),
            }
        )

    # --------------------------------------------------------
    # Gate vs Adaptive improvement
    # --------------------------------------------------------

    for gate_column in [
        "gate_256_mean",
        "gate_128_mean",
        "gate_64_mean",
    ]:

        for target_column in [
            "delta_psnr",
            "delta_ssim",
        ]:

            if target_column not in gate_df.columns:
                continue

            valid = gate_df[
                [gate_column, target_column]
            ].dropna()

            if len(valid) < 3:
                continue

            pearson_r, pearson_p = pearsonr(
                valid[gate_column],
                valid[target_column],
            )

            spearman_r, spearman_p = spearmanr(
                valid[gate_column],
                valid[target_column],
            )

            correlation_records.append(
                {
                    "gate": gate_column,
                    "target": target_column,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n": len(valid),
                }
            )

    correlation_df = pd.DataFrame(
        correlation_records
    )

    correlation_df.to_csv(
        CORRELATION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ADAPTIVE SPATIAL GATE-BEHAVIOR DIAGNOSTICS\n"
        )

        file.write(
            "===========================================\n\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Seed: {SEED}\n"
        )

        file.write(
            f"Diagnostic timestep: "
            f"{DIAGNOSTIC_TIMESTEP}\n"
        )

        file.write(
            f"Test images: {len(gate_df)}\n"
        )

        file.write(
            f"Model parameters: "
            f"{count_parameters(model):,}\n\n"
        )

        file.write(
            "MEAN GATE ACTIVATIONS\n"
        )

        file.write(
            "---------------------\n"
        )

        for gate_column in [
            "gate_256_mean",
            "gate_128_mean",
            "gate_64_mean",
        ]:

            file.write(
                f"{gate_column}: "
                f"{gate_df[gate_column].mean():.6f}\n"
            )

        file.write("\n")

        file.write(
            "BRIGHTNESS QUARTILES\n"
        )

        file.write(
            "--------------------\n"
        )

        for _, row in brightness_summary.iterrows():

            file.write(
                f"{row['brightness_quartile']}: "
                f"n={int(row['count'])}, "
                f"brightness={row['mean_brightness']:.4f}, "
                f"gate256={row['gate_256_mean']:.6f}, "
                f"gate128={row['gate_128_mean']:.6f}, "
                f"gate64={row['gate_64_mean']:.6f}\n"
            )

        file.write("\n")

        file.write(
            "CORRELATIONS\n"
        )

        file.write(
            "------------\n"
        )

        for _, row in correlation_df.iterrows():

            file.write(
                f"{row['gate']} vs "
                f"{row['target']}: "
                f"Pearson r={row['pearson_r']:.6f}, "
                f"p={row['pearson_p']:.6g}; "
                f"Spearman r={row['spearman_r']:.6f}, "
                f"p={row['spearman_p']:.6g}; "
                f"n={int(row['n'])}\n"
            )

    print()
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Gate metrics:\n"
        f"{GATE_METRICS_PATH}"
    )

    print(
        f"Brightness summary:\n"
        f"{BRIGHTNESS_SUMMARY_PATH}"
    )

    print(
        f"Correlations:\n"
        f"{CORRELATION_PATH}"
    )

    print(
        f"Summary:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
