from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.configs import training_config as config
from src.diffusion.forward import q_sample
from src.diffusion.reverse import predict_x_start_from_noise
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
# LOAD MODEL
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
# ANALYZE ONE TIMESTEP
# ============================================================

@torch.no_grad()
def analyze_timestep(
    model,
    val_loader,
    alpha_bars,
    timestep_value,
    device,
):
    epsilon_errors = []
    x0_errors_unclipped = []
    x0_errors_clipped = []

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

        # ----------------------------------------------------
        # Forward diffusion
        # ----------------------------------------------------

        noisy_target, noise = q_sample(
            x_start=normal_light,
            timesteps=timesteps,
            alpha_bars=alpha_bars,
        )

        # ----------------------------------------------------
        # Predict noise
        # ----------------------------------------------------

        predicted_noise = model(
            noisy_image=noisy_target,
            condition=low_light,
            timesteps=timesteps,
        )

        # ----------------------------------------------------
        # Epsilon prediction error
        # ----------------------------------------------------

        epsilon_mse = (
            (predicted_noise - noise) ** 2
        ).mean(dim=(1, 2, 3))

        # ----------------------------------------------------
        # Reconstruct x0
        # ----------------------------------------------------

        predicted_x0 = predict_x_start_from_noise(
            x_t=noisy_target,
            timesteps=timesteps,
            predicted_noise=predicted_noise,
            alpha_bars=alpha_bars,
        )

        # Raw x0 reconstruction error
        x0_mse_unclipped = (
            (predicted_x0 - normal_light) ** 2
        ).mean(dim=(1, 2, 3))

        # Same clipping used by the reverse sampler
        predicted_x0_clipped = predicted_x0.clamp(
            -1.0,
            1.0,
        )

        x0_mse_clipped = (
            (predicted_x0_clipped - normal_light) ** 2
        ).mean(dim=(1, 2, 3))

        epsilon_errors.extend(
            epsilon_mse.cpu().tolist()
        )

        x0_errors_unclipped.extend(
            x0_mse_unclipped.cpu().tolist()
        )

        x0_errors_clipped.extend(
            x0_mse_clipped.cpu().tolist()
        )

    return {
        "epsilon_mean":
            float(np.mean(epsilon_errors)),

        "epsilon_std":
            float(np.std(epsilon_errors)),

        "x0_unclipped_mean":
            float(np.mean(x0_errors_unclipped)),

        "x0_unclipped_std":
            float(np.std(x0_errors_unclipped)),

        "x0_clipped_mean":
            float(np.mean(x0_errors_clipped)),

        "x0_clipped_std":
            float(np.std(x0_errors_clipped)),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print(
        "PHASE 4 — TASK 15.5.2: "
        "TIMESTEP-WISE x0 RECONSTRUCTION ANALYSIS"
    )
    print("=" * 65)

    device = torch.device(config.DEVICE)

    print(f"\nDevice: {device}")
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

    set_seed(config.RANDOM_SEED)

    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    print("\nCreating validation DataLoader...")

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

    print("\nLoading best 32-channel model...")

    model = load_model(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    _, _, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    alpha_bars = alpha_bars.to(device)

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
    print(timestep_values)

    results = []

    print("\n" + "-" * 65)

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    for timestep in timestep_values:

        print(
            f"Analyzing timestep {timestep}..."
        )

        result = analyze_timestep(
            model=model,
            val_loader=val_loader,
            alpha_bars=alpha_bars,
            timestep_value=timestep,
            device=device,
        )

        alpha_bar = (
            alpha_bars[timestep]
            .item()
        )

        result["timestep"] = timestep
        result["alpha_bar"] = alpha_bar

        results.append(result)

        print(
            f"t={timestep:3d} | "
            f"alpha_bar={alpha_bar:.8f} | "
            f"epsilon MSE="
            f"{result['epsilon_mean']:.6f} | "
            f"x0 MSE="
            f"{result['x0_clipped_mean']:.6f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = (
        config.RESULTS_ROOT
        / "x0_reconstruction_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "x0_timestep_analysis.csv"
    )

    fieldnames = [
        "timestep",
        "alpha_bar",
        "epsilon_mean",
        "epsilon_std",
        "x0_unclipped_mean",
        "x0_unclipped_std",
        "x0_clipped_mean",
        "x0_clipped_std",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                key: result[key]
                for key in fieldnames
            })

    # --------------------------------------------------------
    # Save summary
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
            "PHASE 4 — TASK 15.5.2\n"
            "TIMESTEP-WISE x0 RECONSTRUCTION ANALYSIS\n\n"
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

        for result in results:

            file.write(
                f"t={result['timestep']:3d} | "
                f"alpha_bar="
                f"{result['alpha_bar']:.8f} | "
                f"epsilon MSE="
                f"{result['epsilon_mean']:.8f} | "
                f"x0 clipped MSE="
                f"{result['x0_clipped_mean']:.8f} | "
                f"x0 raw MSE="
                f"{result['x0_unclipped_mean']:.8f}\n"
            )

    print("\n" + "=" * 65)
    print(
        "TASK 15.5.2 ANALYSIS COMPLETED"
    )
    print("=" * 65)

    print(
        f"\nCSV: {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()