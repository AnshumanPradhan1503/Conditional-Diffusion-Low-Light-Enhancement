from __future__ import annotations

from pathlib import Path
import csv
import random

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.configs import training_config as config
from src.configs import illumination_aware_v2_training_config
from src.diffusion.reverse import reverse_diffusion
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.models.illumination_aware_unet_v2 import (
    IlluminationAwareConditionalUNet,
)
from src.preprocessing.illumination_proxy import (
    illumination_proxy_from_normalized_rgb,
)
from src.preprocessing.paired_dataset import LOLv2PairedDataset


# ============================================================
# Evaluation settings
# ============================================================

SEED = 42
GAMMA = 0.4


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Image utilities
# ============================================================

def normalized_to_float(
    image: torch.Tensor,
) -> np.ndarray:
    """
    Convert normalized [-1, 1] RGB tensor to
    float HWC [0, 1].
    """
    image = image.detach().cpu().clamp(-1.0, 1.0)

    image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)

    return image.permute(1, 2, 0).numpy()


# ============================================================
# Gamma baseline
# ============================================================

def gamma_correct(
    image: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Apply gamma correction to an image in [-1, 1].

    gamma < 1 brightens the image.
    """
    image = image.clamp(-1.0, 1.0)

    image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)

    image = torch.pow(image, gamma)

    image = image * 2.0 - 1.0

    return image


# ============================================================
# Metrics
# ============================================================

def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Calculate PSNR for images represented in [-1, 1].
    """
    prediction = prediction.detach().cpu().clamp(
        -1.0,
        1.0,
    )

    target = target.detach().cpu().clamp(
        -1.0,
        1.0,
    )

    prediction = (prediction + 1.0) / 2.0
    target = (target + 1.0) / 2.0

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse == 0.0:
        return float("inf")

    return float(
        10.0 * np.log10(1.0 / mse)
    )


def calculate_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Calculate RGB SSIM.
    """
    prediction_np = normalized_to_float(
        prediction
    )

    target_np = normalized_to_float(
        target
    )

    return float(
        structural_similarity(
            target_np,
            prediction_np,
            channel_axis=2,
            data_range=1.0,
        )
    )


# ============================================================
# Load original control model
# ============================================================

def load_original_model(
    device: torch.device,
) -> tuple[ConditionalUNet, dict]:

    print("Loading original control model...")

    checkpoint_path = (
        config.CHECKPOINT_ROOT
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Original checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    return model, checkpoint


# ============================================================
# Load illumination-aware V2 model
# ============================================================

def load_v2_model(
    device: torch.device,
) -> tuple[
    IlluminationAwareConditionalUNet,
    dict,
]:

    print(
        "Loading illumination-aware V2 model..."
    )

    v2_config = (
        illumination_aware_v2_training_config
    )

    checkpoint_path = (
        v2_config.CHECKPOINT_ROOT
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = IlluminationAwareConditionalUNet(
        image_channels=v2_config.IMAGE_CHANNELS,
        condition_channels=v2_config.CONDITION_CHANNELS,
        base_channels=v2_config.BASE_CHANNELS,
        time_dim=v2_config.TIME_EMBEDDING_DIM,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"V2 checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    return model, checkpoint


# ============================================================
# Original DDPM sampling
# ============================================================

@torch.no_grad()
def sample_original_model(
    model: ConditionalUNet,
    low_light: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:

    def predicted_noise_fn(
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:

        return model(
            noisy_image=x_t,
            condition=condition,
            timesteps=timestep,
        )

    return reverse_diffusion(
        shape=low_light.shape,
        condition=low_light,
        predicted_noise_fn=predicted_noise_fn,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=generator,
    )


# ============================================================
# V2 DDPM sampling
# ============================================================

@torch.no_grad()
def sample_v2_model(
    model: IlluminationAwareConditionalUNet,
    low_light: torch.Tensor,
    illumination: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:

    def predicted_noise_fn(
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:

        return model(
            noisy_image=x_t,
            condition=condition,
            timesteps=timestep,
            illumination=illumination,
        )

    return reverse_diffusion(
        shape=low_light.shape,
        condition=low_light,
        predicted_noise_fn=predicted_noise_fn,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=generator,
    )


# ============================================================
# Save visual comparison
# ============================================================

def save_comparison_image(
    low_light: torch.Tensor,
    normal_light: torch.Tensor,
    gamma_result: torch.Tensor,
    original_result: torch.Tensor,
    v2_result: torch.Tensor,
    output_path: Path,
) -> None:

    import matplotlib.pyplot as plt

    low_np = normalized_to_float(
        low_light
    )

    gamma_np = normalized_to_float(
        gamma_result
    )

    original_np = normalized_to_float(
        original_result
    )

    v2_np = normalized_to_float(
        v2_result
    )

    target_np = normalized_to_float(
        normal_light
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    axes[0].imshow(low_np)
    axes[0].set_title("Low-light")

    axes[1].imshow(gamma_np)
    axes[1].set_title("Gamma 0.4")

    axes[2].imshow(original_np)
    axes[2].set_title("Original DDPM")

    axes[3].imshow(v2_np)
    axes[3].set_title("Illumination V2")

    axes[4].imshow(target_np)
    axes[4].set_title("Normal-light GT")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "TASK 24.7 — CONTROLLED V2 EVALUATION"
    )
    print("=" * 70)

    set_seed(SEED)

    device = torch.device(
        config.DEVICE
    )

    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(
        f"Image size: {config.IMAGE_SIZE}"
    )
    print(
        f"Diffusion steps: {config.TIMESTEPS}"
    )
    print(f"Gamma: {GAMMA}")
    print()

    # ========================================================
    # Load original control model
    # ========================================================

    original_model, original_checkpoint = (
        load_original_model(device)
    )

    print()

    # ========================================================
    # Load V2 model
    # ========================================================

    v2_model, v2_checkpoint = (
        load_v2_model(device)
    )

    print()

    # ========================================================
    # Create diffusion schedule
    # ========================================================

    print(
        "Creating diffusion schedule..."
    )

    betas, alphas, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    print(
        "Diffusion schedule created."
    )

    print()

    # ========================================================
    # Create official test dataset
    # ========================================================

    print(
        "Creating test dataset..."
    )

    test_dataset = LOLv2PairedDataset(
        data_root=config.DATA_ROOT,
        split_file=(
            config.SPLIT_ROOT
            / "test_pairs.txt"
        ),
        mode="test",
        image_size=config.IMAGE_SIZE,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print()

    # ========================================================
    # Results directories
    # ========================================================

    results_root = (
        config.RESULTS_ROOT
        / "illumination_aware_v2_evaluation"
    )

    visuals_root = (
        results_root / "visuals"
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    visuals_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Metric storage
    # ========================================================

    rows = []

    gamma_psnr_values = []
    gamma_ssim_values = []

    original_psnr_values = []
    original_ssim_values = []

    v2_psnr_values = []
    v2_ssim_values = []

    v2_psnr_wins = 0
    v2_ssim_wins = 0

    print(
        "Starting controlled evaluation..."
    )

    print()

    # ========================================================
    # Evaluate all official test images
    # ========================================================

    for index in range(
        len(test_dataset)
    ):

        # ----------------------------------------------------
        # Load exact test pair
        # ----------------------------------------------------

        low_light, normal_light = (
            test_dataset[index]
        )

        low_light = (
            low_light
            .unsqueeze(0)
            .to(device)
        )

        normal_light = (
            normal_light
            .unsqueeze(0)
            .to(device)
        )

        low_name = Path(
            test_dataset.pairs[index][0]
        ).name

        # ----------------------------------------------------
        # Deterministic generator for original model
        # ----------------------------------------------------

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + index
        )

        # ----------------------------------------------------
        # Gamma baseline
        # ----------------------------------------------------

        gamma_result = gamma_correct(
            low_light,
            GAMMA,
        )

        gamma_psnr = calculate_psnr(
            gamma_result[0],
            normal_light[0],
        )

        gamma_ssim = calculate_ssim(
            gamma_result[0],
            normal_light[0],
        )

        # ----------------------------------------------------
        # Original DDPM
        # ----------------------------------------------------

        original_result = (
            sample_original_model(
                model=original_model,
                low_light=low_light,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                device=device,
                generator=generator,
            )
        )

        original_psnr = calculate_psnr(
            original_result[0],
            normal_light[0],
        )

        original_ssim = calculate_ssim(
            original_result[0],
            normal_light[0],
        )

        # ----------------------------------------------------
        # V2 illumination proxy
        # ----------------------------------------------------

        illumination = (
            illumination_proxy_from_normalized_rgb(
                low_light
            )
        )

        # ----------------------------------------------------
        # V2 generator
        #
        # Same seed as original model.
        # This keeps the initial diffusion noise identical
        # for the original and V2 comparison.
        # ----------------------------------------------------

        generator_v2 = torch.Generator(
            device=device
        )

        generator_v2.manual_seed(
            SEED + index
        )

        # ----------------------------------------------------
        # V2 DDPM
        # ----------------------------------------------------

        v2_result = sample_v2_model(
            model=v2_model,
            low_light=low_light,
            illumination=illumination,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            device=device,
            generator=generator_v2,
        )

        v2_psnr = calculate_psnr(
            v2_result[0],
            normal_light[0],
        )

        v2_ssim = calculate_ssim(
            v2_result[0],
            normal_light[0],
        )

        # ----------------------------------------------------
        # V2 - Original
        # ----------------------------------------------------

        v2_minus_original_psnr = (
            v2_psnr
            - original_psnr
        )

        v2_minus_original_ssim = (
            v2_ssim
            - original_ssim
        )

        # ----------------------------------------------------
        # Per-image wins
        # ----------------------------------------------------

        if v2_psnr > original_psnr:
            v2_psnr_wins += 1

        if v2_ssim > original_ssim:
            v2_ssim_wins += 1

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        rows.append(
            {
                "index": index,
                "image": low_name,
                "gamma_psnr": gamma_psnr,
                "gamma_ssim": gamma_ssim,
                "original_psnr": original_psnr,
                "original_ssim": original_ssim,
                "v2_psnr": v2_psnr,
                "v2_ssim": v2_ssim,
                "v2_minus_original_psnr": (
                    v2_minus_original_psnr
                ),
                "v2_minus_original_ssim": (
                    v2_minus_original_ssim
                ),
            }
        )

        gamma_psnr_values.append(
            gamma_psnr
        )

        gamma_ssim_values.append(
            gamma_ssim
        )

        original_psnr_values.append(
            original_psnr
        )

        original_ssim_values.append(
            original_ssim
        )

        v2_psnr_values.append(
            v2_psnr
        )

        v2_ssim_values.append(
            v2_ssim
        )

        # ----------------------------------------------------
        # Selected visual outputs
        # ----------------------------------------------------

        if index in {
            0,
            10,
            25,
            50,
            75,
            99,
        }:

            comparison_path = (
                visuals_root
                / f"comparison_{index:03d}.png"
            )

            save_comparison_image(
                low_light[0],
                normal_light[0],
                gamma_result[0],
                original_result[0],
                v2_result[0],
                comparison_path,
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"[{index + 1:3d}/"
            f"{len(test_dataset):3d}] "
            f"{low_name} | "
            f"Gamma PSNR="
            f"{gamma_psnr:.4f} "
            f"SSIM="
            f"{gamma_ssim:.4f} | "
            f"Original PSNR="
            f"{original_psnr:.4f} "
            f"SSIM="
            f"{original_ssim:.4f} | "
            f"V2 PSNR="
            f"{v2_psnr:.4f} "
            f"SSIM="
            f"{v2_ssim:.4f}"
        )

    # ========================================================
    # Aggregate metrics
    # ========================================================

    gamma_psnr_mean = float(
        np.mean(gamma_psnr_values)
    )

    gamma_ssim_mean = float(
        np.mean(gamma_ssim_values)
    )

    original_psnr_mean = float(
        np.mean(original_psnr_values)
    )

    original_ssim_mean = float(
        np.mean(original_ssim_values)
    )

    v2_psnr_mean = float(
        np.mean(v2_psnr_values)
    )

    v2_ssim_mean = float(
        np.mean(v2_ssim_values)
    )

    v2_delta_psnr = (
        v2_psnr_mean
        - original_psnr_mean
    )

    v2_delta_ssim = (
        v2_ssim_mean
        - original_ssim_mean
    )

    # ========================================================
    # Save per-image CSV
    # ========================================================

    csv_path = (
        results_root
        / "test_metrics.csv"
    )

    fieldnames = [
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
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Save summary
    # ========================================================

    summary_path = (
        results_root
        / "summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "TASK 24.7 — CONTROLLED V2 EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Device: {device}\n"
        )

        file.write(
            f"Seed: {SEED}\n"
        )

        file.write(
            f"Image size: {config.IMAGE_SIZE}\n"
        )

        file.write(
            f"Diffusion steps: "
            f"{config.TIMESTEPS}\n"
        )

        file.write(
            f"Gamma: {GAMMA}\n\n"
        )

        file.write(
            f"Original checkpoint epoch: "
            f"{original_checkpoint['epoch']}\n"
        )

        file.write(
            f"V2 checkpoint epoch: "
            f"{v2_checkpoint['epoch']}\n\n"
        )

        file.write(
            "AGGREGATE RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Gamma PSNR: "
            f"{gamma_psnr_mean:.6f} dB\n"
        )

        file.write(
            f"Gamma SSIM: "
            f"{gamma_ssim_mean:.6f}\n"
        )

        file.write(
            f"Original PSNR: "
            f"{original_psnr_mean:.6f} dB\n"
        )

        file.write(
            f"Original SSIM: "
            f"{original_ssim_mean:.6f}\n"
        )

        file.write(
            f"V2 PSNR: "
            f"{v2_psnr_mean:.6f} dB\n"
        )

        file.write(
            f"V2 SSIM: "
            f"{v2_ssim_mean:.6f}\n"
        )

        file.write(
            f"V2 - Original PSNR: "
            f"{v2_delta_psnr:+.6f} dB\n"
        )

        file.write(
            f"V2 - Original SSIM: "
            f"{v2_delta_ssim:+.6f}\n"
        )

        file.write(
            f"V2 PSNR wins: "
            f"{v2_psnr_wins}/100\n"
        )

        file.write(
            f"V2 SSIM wins: "
            f"{v2_ssim_wins}/100\n"
        )

    # ========================================================
    # Final terminal summary
    # ========================================================

    print()

    print("=" * 70)
    print(
        "TASK 24.7 — FINAL RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Gamma PSNR: "
        f"{gamma_psnr_mean:.6f} dB"
    )

    print(
        f"Gamma SSIM: "
        f"{gamma_ssim_mean:.6f}"
    )

    print()

    print(
        f"Original PSNR: "
        f"{original_psnr_mean:.6f} dB"
    )

    print(
        f"Original SSIM: "
        f"{original_ssim_mean:.6f}"
    )

    print()

    print(
        f"V2 PSNR: "
        f"{v2_psnr_mean:.6f} dB"
    )

    print(
        f"V2 SSIM: "
        f"{v2_ssim_mean:.6f}"
    )

    print()

    print(
        f"V2 - Original PSNR: "
        f"{v2_delta_psnr:+.6f} dB"
    )

    print(
        f"V2 - Original SSIM: "
        f"{v2_delta_ssim:+.6f}"
    )

    print()

    print(
        f"V2 PSNR wins: "
        f"{v2_psnr_wins}/100"
    )

    print(
        f"V2 SSIM wins: "
        f"{v2_ssim_wins}/100"
    )

    print()

    print(
        "Metrics saved to:"
    )

    print(csv_path)

    print()

    print(
        "Summary saved to:"
    )

    print(summary_path)

    print()

    print(
        "Visuals saved to:"
    )

    print(visuals_root)

    print()

    print("=" * 70)
    print(
        "CONTROLLED V2 EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()