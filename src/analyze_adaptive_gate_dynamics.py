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
    / "adaptive_spatial_gate_dynamics"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

GATE_DYNAMICS_PATH = (
    OUTPUT_ROOT
    / "gate_dynamics.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.txt"
)

CORRELATION_PATH = (
    OUTPUT_ROOT
    / "gate_dynamics_correlations.csv"
)

# ------------------------------------------------------------
# Timesteps to inspect
# ------------------------------------------------------------

DIAGNOSTIC_TIMESTEPS = [
    100,
    250,
    500,
    750,
]

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

def create_model(
    device: torch.device,
) -> AdaptiveSpatialConditionalUNet:

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

        state_dict = checkpoint[
            "model_state_dict"
        ]

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

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    return test_loader.dataset


# ============================================================
# Helpers
# ============================================================

def count_parameters(
    model: torch.nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def calculate_brightness(
    image: torch.Tensor,
) -> float:
    """
    Input:
        image in [-1, 1]

    Output:
        mean brightness on 0-255 scale.
    """

    image_01 = (
        image + 1.0
    ) / 2.0

    image_01 = torch.clamp(
        image_01,
        0.0,
        1.0,
    )

    return (
        image_01.mean().item()
        * 255.0
    )


def get_gate_statistics(
    gate: torch.Tensor,
) -> tuple[float, float]:

    mean_value = (
        gate.mean().item()
    )

    std_value = (
        gate.std().item()
    )

    return mean_value, std_value


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ADAPTIVE SPATIAL GATE DYNAMICS")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Timesteps: {DIAGNOSTIC_TIMESTEPS}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Output directory: {OUTPUT_ROOT}"
    )

    print()

    set_seed(SEED)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading Adaptive Spatial U-Net..."
    )

    model = create_model(
        DEVICE
    )

    print(
        f"Model parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        "Model loaded successfully."
    )

    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "Loading official test dataset..."
    )

    test_dataset = (
        create_test_dataset()
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Load existing metrics
    # --------------------------------------------------------

    existing_metrics = None

    if EXISTING_METRICS_PATH.exists():

        existing_metrics = pd.read_csv(
            EXISTING_METRICS_PATH
        )

        print(
            f"Existing evaluation rows: "
            f"{len(existing_metrics)}"
        )

    else:

        print(
            "Warning: existing evaluation "
            "metrics were not found."
        )

    print()

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    _, _, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    # --------------------------------------------------------
    # Collect diagnostics
    # --------------------------------------------------------

    records = []

    print(
        "Collecting gate dynamics..."
    )

    print()

    for index in range(
        len(test_dataset)
    ):

        low_light, normal_light = (
            test_dataset[index]
        )

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

        brightness = (
            calculate_brightness(
                low_light
            )
        )

        image_name = Path(
            test_dataset
            .pairs[index][0]
        ).name

        # ----------------------------------------------------
        # One deterministic noise sample per image
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
        # Inspect each timestep
        # ----------------------------------------------------

        for timestep_value in (
            DIAGNOSTIC_TIMESTEPS
        ):

            timestep = torch.tensor(
                [timestep_value],
                device=DEVICE,
                dtype=torch.long,
            )

            alpha_bar = (
                alpha_bars[
                    timestep_value
                ].to(DEVICE)
            )

            sqrt_alpha_bar = torch.sqrt(
                alpha_bar
            )

            sqrt_one_minus_alpha_bar = (
                torch.sqrt(
                    1.0 - alpha_bar
                )
            )

            noisy_target = (
                sqrt_alpha_bar
                * normal_light_batch
                +
                sqrt_one_minus_alpha_bar
                * noise
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with torch.no_grad():

                _ = model(
                    noisy_image=noisy_target,
                    condition=low_light_batch,
                    timesteps=timestep,
                )

            # ------------------------------------------------
            # Read adaptive gates
            # ------------------------------------------------

            gates = model.last_gates

            gate_256 = gates["256"]
            gate_128 = gates["128"]
            gate_64 = gates["64"]

            gate_256_mean, gate_256_std = (
                get_gate_statistics(
                    gate_256
                )
            )

            gate_128_mean, gate_128_std = (
                get_gate_statistics(
                    gate_128
                )
            )

            gate_64_mean, gate_64_std = (
                get_gate_statistics(
                    gate_64
                )
            )

            records.append(
                {
                    "index": index,
                    "image": image_name,
                    "brightness": brightness,
                    "timestep": timestep_value,

                    "gate_256_mean":
                        gate_256_mean,

                    "gate_256_std":
                        gate_256_std,

                    "gate_128_mean":
                        gate_128_mean,

                    "gate_128_std":
                        gate_128_std,

                    "gate_64_mean":
                        gate_64_mean,

                    "gate_64_std":
                        gate_64_std,
                }
            )

        if (
            (index + 1) % 10
            == 0
        ):

            print(
                f"Processed "
                f"{index + 1:3d} / "
                f"{len(test_dataset)}"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    gate_df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Merge existing metrics
    # --------------------------------------------------------

    if existing_metrics is not None:

        image_column = None

        for column in [
            "image",
            "low_image",
            "filename",
            "low_name",
        ]:

            if column in (
                existing_metrics.columns
            ):

                image_column = column
                break

        if image_column is not None:

            metrics_for_merge = (
                existing_metrics.copy()
            )

            if image_column != "image":

                metrics_for_merge = (
                    metrics_for_merge.rename(
                        columns={
                            image_column:
                                "image"
                        }
                    )
                )

            useful_columns = [
                column
                for column in [
                    "image",
                    "delta_psnr",
                    "delta_ssim",
                ]
                if column in (
                    metrics_for_merge.columns
                )
            ]

            gate_df = gate_df.merge(
                metrics_for_merge[
                    useful_columns
                ],
                on="image",
                how="left",
            )

    # --------------------------------------------------------
    # Save raw dynamics
    # --------------------------------------------------------

    gate_df.to_csv(
        GATE_DYNAMICS_PATH,
        index=False,
    )

    print()

    print(
        f"Saved gate dynamics:\n"
        f"{GATE_DYNAMICS_PATH}"
    )

    # --------------------------------------------------------
    # Calculate timestep statistics
    # --------------------------------------------------------

    timestep_summary = (
        gate_df
        .groupby("timestep")
        .agg(
            count=("image", "count"),

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

    # --------------------------------------------------------
    # Correlations at every timestep
    # --------------------------------------------------------

    correlation_records = []

    for timestep_value in (
        DIAGNOSTIC_TIMESTEPS
    ):

        subset = gate_df[
            gate_df["timestep"]
            == timestep_value
        ]

        for gate_column in [
            "gate_256_mean",
            "gate_128_mean",
            "gate_64_mean",
        ]:

            valid = subset[
                [
                    gate_column,
                    "brightness",
                ]
            ].dropna()

            if len(valid) < 3:
                continue

            pearson_r, pearson_p = (
                pearsonr(
                    valid[gate_column],
                    valid["brightness"],
                )
            )

            spearman_r, spearman_p = (
                spearmanr(
                    valid[gate_column],
                    valid["brightness"],
                )
            )

            correlation_records.append(
                {
                    "timestep":
                        timestep_value,

                    "gate":
                        gate_column,

                    "target":
                        "brightness",

                    "pearson_r":
                        pearson_r,

                    "pearson_p":
                        pearson_p,

                    "spearman_r":
                        spearman_r,

                    "spearman_p":
                        spearman_p,

                    "n":
                        len(valid),
                }
            )

    # --------------------------------------------------------
    # Gate vs performance improvement
    # --------------------------------------------------------

    if (
        "delta_psnr"
        in gate_df.columns
    ):

        for timestep_value in (
            DIAGNOSTIC_TIMESTEPS
        ):

            subset = gate_df[
                gate_df["timestep"]
                == timestep_value
            ]

            for gate_column in [
                "gate_256_mean",
                "gate_128_mean",
                "gate_64_mean",
            ]:

                for target_column in [
                    "delta_psnr",
                    "delta_ssim",
                ]:

                    valid = subset[
                        [
                            gate_column,
                            target_column,
                        ]
                    ].dropna()

                    if len(valid) < 3:
                        continue

                    pearson_r, pearson_p = (
                        pearsonr(
                            valid[gate_column],
                            valid[target_column],
                        )
                    )

                    spearman_r, spearman_p = (
                        spearmanr(
                            valid[gate_column],
                            valid[target_column],
                        )
                    )

                    correlation_records.append(
                        {
                            "timestep":
                                timestep_value,

                            "gate":
                                gate_column,

                            "target":
                                target_column,

                            "pearson_r":
                                pearson_r,

                            "pearson_p":
                                pearson_p,

                            "spearman_r":
                                spearman_r,

                            "spearman_p":
                                spearman_p,

                            "n":
                                len(valid),
                        }
                    )

    correlation_df = (
        pd.DataFrame(
            correlation_records
        )
    )

    correlation_df.to_csv(
        CORRELATION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ADAPTIVE SPATIAL GATE DYNAMICS\n"
        )

        file.write(
            "==============================\n\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Seed: {SEED}\n"
        )

        file.write(
            f"Test images: "
            f"{len(test_dataset)}\n"
        )

        file.write(
            f"Model parameters: "
            f"{count_parameters(model):,}\n\n"
        )

        file.write(
            "TIMESTEP GATE MEANS\n"
        )

        file.write(
            "-------------------\n"
        )

        for _, row in (
            timestep_summary.iterrows()
        ):

            file.write(
                f"t={int(row['timestep'])}: "
                f"gate256="
                f"{row['gate_256_mean']:.6f}, "
                f"gate128="
                f"{row['gate_128_mean']:.6f}, "
                f"gate64="
                f"{row['gate_64_mean']:.6f}\n"
            )

        file.write("\n")

        file.write(
            "TIMESTEP GATE VARIATION\n"
        )

        file.write(
            "-----------------------\n"
        )

        for _, row in (
            timestep_summary.iterrows()
        ):

            file.write(
                f"t={int(row['timestep'])}: "
                f"std256="
                f"{row['gate_256_std']:.8f}, "
                f"std128="
                f"{row['gate_128_std']:.8f}, "
                f"std64="
                f"{row['gate_64_std']:.8f}\n"
            )

        file.write("\n")

        file.write(
            "CORRELATIONS\n"
        )

        file.write(
            "------------\n"
        )

        for _, row in (
            correlation_df.iterrows()
        ):

            file.write(
                f"t={int(row['timestep'])}, "
                f"{row['gate']} vs "
                f"{row['target']}: "
                f"Pearson r="
                f"{row['pearson_r']:.6f}, "
                f"p="
                f"{row['pearson_p']:.6g}; "
                f"Spearman r="
                f"{row['spearman_r']:.6f}, "
                f"p="
                f"{row['spearman_p']:.6g}; "
                f"n="
                f"{int(row['n'])}\n"
            )

    print()

    print("=" * 70)
    print("GATE DYNAMICS ANALYSIS COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Gate dynamics:\n"
        f"{GATE_DYNAMICS_PATH}"
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
