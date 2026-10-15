from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.configs import training_config as config
from src.diffusion.forward import q_sample
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL
# ============================================================

def load_model(device: torch.device) -> ConditionalUNet:

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint_path = (
        config.CHECKPOINT_ROOT / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# TIMESTEP ANALYSIS
# ============================================================

@torch.no_grad()
def analyze_timestep(
    model: ConditionalUNet,
    val_loader,
    alpha_bars: torch.Tensor,
    timestep_value: int,
    device: torch.device,
) -> tuple[float, int]:

    total_squared_error = 0.0
    total_elements = 0

    for low_light, normal_light in val_loader:

        low_light = low_light.to(device)
        normal_light = normal_light.to(device)

        batch_size = normal_light.shape[0]

        timesteps = torch.full(
            (batch_size,),
            timestep_value,
            device=device,
            dtype=torch.long,
        )

        # Add noise to the normal-light target.
        noisy_target, noise = q_sample(
            x_start=normal_light,
            timesteps=timesteps,
            alpha_bars=alpha_bars,
        )

        # Predict the noise.
        predicted_noise = model(
            noisy_image=noisy_target,
            condition=low_light,
            timesteps=timesteps,
        )

        # Sum squared error rather than averaging each batch.
        squared_error = F.mse_loss(
            predicted_noise,
            noise,
            reduction="sum",
        )

        total_squared_error += squared_error.item()
        total_elements += noise.numel()

    mean_mse = (
        total_squared_error
        / total_elements
    )

    return mean_mse, total_elements


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print(
        "PHASE 4 — TASK 15.5.1: "
        "TIMESTEP-WISE LOSS ANALYSIS"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        config.DEVICE
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Model base channels: "
        f"{config.BASE_CHANNELS}"
    )

    print(
        f"Diffusion timesteps: "
        f"{config.TIMESTEPS}"
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        config.RANDOM_SEED
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print(
        "\nCreating validation DataLoader..."
    )

    _, val_loader, _ = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading best 32-channel model..."
    )

    model = load_model(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

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

    alpha_bars = alpha_bars.to(
        device
    )

    # --------------------------------------------------------
    # Timesteps
    # --------------------------------------------------------

    timestep_values = [
        0,
        100,
        250,
        500,
        750,
        999,
    ]

    print(
        "\nTesting timesteps:"
    )

    print(
        timestep_values
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    results = []

    print()
    print("-" * 60)

    for timestep_value in timestep_values:

        print(
            f"Analyzing timestep "
            f"{timestep_value}..."
        )

        mse, element_count = analyze_timestep(
            model=model,
            val_loader=val_loader,
            alpha_bars=alpha_bars,
            timestep_value=timestep_value,
            device=device,
        )

        alpha_bar = (
            alpha_bars[timestep_value]
            .item()
        )

        results.append(
            {
                "timestep":
                    timestep_value,

                "alpha_bar":
                    alpha_bar,

                "noise_mse":
                    mse,
            }
        )

        print(
            f"t={timestep_value:3d} | "
            f"alpha_bar={alpha_bar:.8f} | "
            f"noise MSE={mse:.6f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = (
        config.RESULTS_ROOT
        / "timestep_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "timestep_loss.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestep",
                "alpha_bar",
                "noise_mse",
            ],
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "timestep":
                        row["timestep"],

                    "alpha_bar":
                        f'{row["alpha_bar"]:.10f}',

                    "noise_mse":
                        f'{row["noise_mse"]:.8f}',
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHASE 4 — TASK 15.5.1\n"
            "TIMESTEP-WISE DIFFUSION LOSS ANALYSIS\n\n"
        )

        file.write(
            f"Model parameters: "
            f"{parameter_count:,}\n"
        )

        file.write(
            f"Validation samples: "
            f"{len(val_loader.dataset)}\n"
        )

        file.write(
            f"Diffusion timesteps: "
            f"{config.TIMESTEPS}\n\n"
        )

        file.write(
            "RESULTS\n"
        )

        for row in results:

            file.write(
                f"t={row['timestep']:3d} | "
                f"alpha_bar="
                f"{row['alpha_bar']:.8f} | "
                f"noise MSE="
                f"{row['noise_mse']:.8f}\n"
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "TASK 15.5.1 ANALYSIS COMPLETED"
    )
    print("=" * 60)

    print(
        f"\nCSV: {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()