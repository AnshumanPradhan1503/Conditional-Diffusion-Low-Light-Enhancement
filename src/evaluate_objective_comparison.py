from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from src.configs import training_config as config
from src.diffusion.reverse import reverse_diffusion
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# IMAGE CONVERSION
# ============================================================

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)

    image = tensor.squeeze(0)
    image = image.permute(1, 2, 0)

    return image.numpy().astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0,
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=2,
        data_range=1.0,
    )

    return float(psnr), float(ssim)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ConditionalUNet:

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_image(
    model: ConditionalUNet,
    low_light: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:

    shape = low_light.shape

    def predicted_noise_fn(
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        return model(
            noisy_image=x_t,
            condition=condition,
            timesteps=timesteps,
        )

    generated = reverse_diffusion(
        shape=shape,
        condition=low_light,
        predicted_noise_fn=predicted_noise_fn,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=generator,
    )

    return generated


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print("TASK 15.5.6 — EVALUATION CONSISTENCY CHECK")
    print("=" * 60)

    device = torch.device(config.DEVICE)

    print(f"Device: {device}")
    print(f"Random seed: {SEED}")

    # --------------------------------------------------------
    # Set identical seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    print(f"Diffusion steps: {config.TIMESTEPS}")
    print(f"Image size: {config.IMAGE_SIZE}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    test_dataset = test_loader.dataset

    print(f"Official test images: {len(test_dataset)}")

    # --------------------------------------------------------
    # Check first test image
    # --------------------------------------------------------

    low_light, normal_light = test_dataset[0]

    low_light = low_light.unsqueeze(0).to(device)
    normal_light = normal_light.unsqueeze(0).to(device)

    print()
    print("FIRST TEST IMAGE")
    print(f"Low-light shape : {low_light.shape}")
    print(f"Normal shape    : {normal_light.shape}")

    print(
        f"Low-light range: "
        f"{low_light.min().item():.6f} to "
        f"{low_light.max().item():.6f}"
    )

    print(
        f"Normal range   : "
        f"{normal_light.min().item():.6f} to "
        f"{normal_light.max().item():.6f}"
    )

    # --------------------------------------------------------
    # Model paths
    # --------------------------------------------------------

    original_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "best_model.pt"
    )

    objective_x0_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "objective_x0"
        / "best_model.pt"
    )

    print()
    print("CHECKPOINTS")

    print(f"Original : {original_checkpoint}")
    print(f"x0       : {objective_x0_checkpoint}")

    if not original_checkpoint.exists():
        raise FileNotFoundError(
            f"Original checkpoint not found: "
            f"{original_checkpoint}"
        )

    if not objective_x0_checkpoint.exists():
        raise FileNotFoundError(
            f"x0 checkpoint not found: "
            f"{objective_x0_checkpoint}"
        )

    # --------------------------------------------------------
    # Load original model
    # --------------------------------------------------------

    print()
    print("Loading original 32-channel model...")

    original_model = load_model(
        original_checkpoint,
        device,
    )

    print("Original model loaded.")

    # --------------------------------------------------------
    # Generate original model
    # --------------------------------------------------------

    print()
    print("Generating ORIGINAL model output...")

    original_generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    original_generated = generate_image(
        model=original_model,
        low_light=low_light,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=original_generator,
    )

    print("Original generation completed.")

    # --------------------------------------------------------
    # Load x0 model
    # --------------------------------------------------------

    print()
    print("Loading x0-objective model...")

    objective_x0_model = load_model(
        objective_x0_checkpoint,
        device,
    )

    print("x0-objective model loaded.")

    # --------------------------------------------------------
    # Generate x0 model
    # --------------------------------------------------------

    print()
    print("Generating x0-objective output...")

    objective_x0_generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    objective_x0_generated = generate_image(
        model=objective_x0_model,
        low_light=low_light,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=objective_x0_generator,
    )

    print("x0 generation completed.")

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    low_np = tensor_to_numpy(low_light)
    reference_np = tensor_to_numpy(normal_light)

    original_np = tensor_to_numpy(
        original_generated
    )

    objective_x0_np = tensor_to_numpy(
        objective_x0_generated
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    low_psnr, low_ssim = calculate_metrics(
        low_np,
        reference_np,
    )

    original_psnr, original_ssim = calculate_metrics(
        original_np,
        reference_np,
    )

    objective_x0_psnr, objective_x0_ssim = calculate_metrics(
        objective_x0_np,
        reference_np,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONSISTENCY CHECK RESULTS")
    print("=" * 60)

    print()
    print("LOW-LIGHT BASELINE")
    print(f"PSNR: {low_psnr:.6f} dB")
    print(f"SSIM: {low_ssim:.6f}")

    print()
    print("ORIGINAL 32-CHANNEL MODEL")
    print(f"PSNR: {original_psnr:.6f} dB")
    print(f"SSIM: {original_ssim:.6f}")

    print()
    print("X0-OBJECTIVE MODEL")
    print(f"PSNR: {objective_x0_psnr:.6f} dB")
    print(f"SSIM: {objective_x0_ssim:.6f}")

    print()
    print("IMPROVEMENT OVER LOW-LIGHT")

    print(
        f"Original PSNR change: "
        f"{original_psnr - low_psnr:+.6f} dB"
    )

    print(
        f"Original SSIM change: "
        f"{original_ssim - low_ssim:+.6f}"
    )

    print(
        f"x0 PSNR change: "
        f"{objective_x0_psnr - low_psnr:+.6f} dB"
    )

    print(
        f"x0 SSIM change: "
        f"{objective_x0_ssim - low_ssim:+.6f}"
    )

    print()
    print("=" * 60)
    print("TASK 15.5.6 — ONE-IMAGE CHECK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
