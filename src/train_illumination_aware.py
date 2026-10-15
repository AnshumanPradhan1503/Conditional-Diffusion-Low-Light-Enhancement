from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.configs import illumination_aware_training_config as config
from src.diffusion.forward import q_sample
from src.diffusion.schedule import get_diffusion_parameters
from src.models.illumination_aware_unet import (
    IlluminationAwareConditionalUNet,
)
from src.preprocessing.dataloader import create_dataloaders
from src.preprocessing.illumination_proxy import (
    illumination_proxy_from_normalized_rgb,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL
# ============================================================

def create_model(device: torch.device):
    model = IlluminationAwareConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    )

    return model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

def create_diffusion_schedule(
    device: torch.device,
):
    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    return (
        betas.to(device),
        alphas.to(device),
        alpha_bars.to(device),
    )


# ============================================================
# ILLUMINATION-AWARE DIFFUSION LOSS
# ============================================================

def illumination_aware_training_loss(
    model,
    low_light: torch.Tensor,
    normal_light: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:

    batch_size = normal_light.shape[0]
    device = normal_light.device

    # Same timestep sampling as the validated DDPM loss.
    timesteps = torch.randint(
        low=0,
        high=len(alpha_bars),
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    # Same forward diffusion process.
    noisy_target, noise = q_sample(
        x_start=normal_light,
        timesteps=timesteps,
        alpha_bars=alpha_bars,
    )

    # Generate the low-frequency illumination proxy
    # directly from the normalized low-light condition.
    illumination = illumination_proxy_from_normalized_rgb(
        low_light
    )

    illumination = illumination.to(
        device=device,
        dtype=low_light.dtype,
    )

    # Predict the noise using the illumination-aware model.
    predicted_noise = model(
        noisy_image=noisy_target,
        condition=low_light,
        timesteps=timesteps,
        illumination=illumination,
    )

    loss = torch.nn.functional.mse_loss(
        predicted_noise,
        noise,
    )

    return loss


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    alpha_bars: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:

    model.train()

    total_loss = 0.0
    batch_count = 0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for batch_index, (
        low_light,
        normal_light,
    ) in enumerate(progress):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        low_light = low_light.to(device)
        normal_light = normal_light.to(device)

        optimizer.zero_grad(set_to_none=True)

        loss = illumination_aware_training_loss(
            model=model,
            low_light=low_light,
            normal_light=normal_light,
            alpha_bars=alpha_bars,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected: "
                f"{loss.item()}"
            )

        loss.backward()

        clip_grad_norm_(
            model.parameters(),
            max_norm=config.GRADIENT_CLIP_MAX_NORM,
        )

        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    if batch_count == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    return total_loss / batch_count


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    val_loader,
    alpha_bars: torch.Tensor,
    device: torch.device,
    max_batches: int | None = None,
) -> float:

    model.eval()

    total_loss = 0.0
    batch_count = 0

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    )

    for batch_index, (
        low_light,
        normal_light,
    ) in enumerate(progress):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        low_light = low_light.to(device)
        normal_light = normal_light.to(device)

        loss = illumination_aware_training_loss(
            model=model,
            low_light=low_light,
            normal_light=normal_light,
            alpha_bars=alpha_bars,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss detected: "
                f"{loss.item()}"
            )

        total_loss += loss.item()
        batch_count += 1

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    if batch_count == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    return total_loss / batch_count


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    history: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train illumination-aware "
            "conditional DDPM for low-light enhancement."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one epoch using a few batches "
            "and save to the smoke-test directory."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_arguments()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(config.RANDOM_SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(config.DEVICE)

    print("=" * 60)
    print(
        "TASK 24.4 — "
        "ILLUMINATION-AWARE TRAINING"
    )
    print("=" * 60)

    print(f"\nDevice: {device}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Diffusion timesteps: {config.TIMESTEPS}")

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    if args.smoke_test:

        print("\nMODE: SMOKE TEST")

        checkpoint_root = (
            config.CHECKPOINT_ROOT
            / "smoke_test"
        )

        results_root = (
            config.RESULTS_ROOT
            / "smoke_test"
        )

        epochs = 1
        max_train_batches = 3
        max_val_batches = 2

    else:

        print("\nMODE: FULL TRAINING")

        checkpoint_root = (
            config.CHECKPOINT_ROOT
        )

        results_root = (
            config.RESULTS_ROOT
        )

        epochs = config.EPOCHS
        max_train_batches = None
        max_val_batches = None

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nCreating DataLoaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            data_root=config.DATA_ROOT,
            split_root=config.SPLIT_ROOT,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating illumination-aware model...")

    model = create_model(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(model)

    print("Optimizer: AdamW")
    print(
        f"Learning rate: "
        f"{config.LEARNING_RATE}"
    )
    print(
        f"Weight decay: "
        f"{config.WEIGHT_DECAY}"
    )

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    _, _, alpha_bars = (
        create_diffusion_schedule(device)
    )

    print(
        f"Diffusion schedule: "
        f"{config.TIMESTEPS} steps"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):

        print("\n" + "-" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 60)

        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            alpha_bars=alpha_bars,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            alpha_bars=alpha_bars,
            device=device,
            max_batches=max_val_batches,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_seconds": epoch_time,
            }
        )

        print(
            f"\nTrain loss: {train_loss:.6f}"
        )

        print(
            f"Validation loss: {val_loss:.6f}"
        )

        print(
            f"Epoch time: {epoch_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                history=history,
                path=(
                    checkpoint_root
                    / "best_model.pt"
                ),
            )

            print(
                "Best model checkpoint saved."
            )

    # --------------------------------------------------------
    # Latest checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        train_loss=history[-1]["train_loss"],
        val_loss=history[-1]["val_loss"],
        best_val_loss=best_val_loss,
        history=history,
        path=(
            checkpoint_root
            / "latest_model.pt"
        ),
    )

    print(
        "\nLatest checkpoint saved."
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint directory: "
        f"{checkpoint_root}"
    )

    print(
        f"Results directory: "
        f"{results_root}"
    )


if __name__ == "__main__":
    main()