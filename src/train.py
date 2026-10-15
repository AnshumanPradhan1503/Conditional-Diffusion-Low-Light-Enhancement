from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.configs import training_config as config
from src.diffusion.loss import diffusion_training_loss
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    """Save random-number-generator states."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: dict) -> None:
    """Restore random-number-generator states."""

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# ============================================================
# MODEL
# ============================================================

def create_model(device: torch.device) -> ConditionalUNet:
    """Create and initialize the conditional U-Net."""

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    )

    model = model.to(device)

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model: ConditionalUNet,
) -> torch.optim.Optimizer:
    """Create the AdamW optimizer."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    return optimizer


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

def create_diffusion_schedule(
    device: torch.device,
):
    """Create diffusion schedule and move it to the selected device."""

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
# TRAINING
# ============================================================

def train_one_epoch(
    model: ConditionalUNet,
    train_loader,
    alpha_bars: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    batch_count = 0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for batch_index, (low_light, normal_light) in enumerate(progress):

        if max_batches is not None and batch_index >= max_batches:
            break

        low_light = low_light.to(device)
        normal_light = normal_light.to(device)

        optimizer.zero_grad(set_to_none=True)

        loss = diffusion_training_loss(
            model=model,
            low_light=low_light,
            normal_light=normal_light,
            alpha_bars=alpha_bars,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected: {loss.item()}"
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
    model: ConditionalUNet,
    val_loader,
    alpha_bars: torch.Tensor,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Calculate validation diffusion loss."""

    model.eval()

    total_loss = 0.0
    batch_count = 0

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    )

    for batch_index, (low_light, normal_light) in enumerate(progress):

        if max_batches is not None and batch_index >= max_batches:
            break

        low_light = low_light.to(device)
        normal_light = normal_light.to(device)

        loss = diffusion_training_loss(
            model=model,
            low_light=low_light,
            normal_light=normal_light,
            alpha_bars=alpha_bars,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss detected: {loss.item()}"
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
    model: ConditionalUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    history: list[dict],
    path: Path,
) -> None:
    """Save model, optimizer, history and training state."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "best_val_loss":
            best_val_loss,

        "history":
            history,

        "rng_state":
            get_rng_state(),

        "config": {
            "image_size":
                config.IMAGE_SIZE,

            "batch_size":
                config.BATCH_SIZE,

            "timesteps":
                config.TIMESTEPS,

            "beta_start":
                config.BETA_START,

            "beta_end":
                config.BETA_END,

            "base_channels":
                config.BASE_CHANNELS,

            "time_embedding_dim":
                config.TIME_EMBEDDING_DIM,

            "learning_rate":
                config.LEARNING_RATE,

            "weight_decay":
                config.WEIGHT_DECAY,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: Path,
    model: ConditionalUNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Load a checkpoint and return training state."""

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    print("\nLoading checkpoint...")
    print(f"Checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]

    train_loss = checkpoint.get(
        "train_loss",
        float("nan"),
    )

    val_loss = checkpoint.get(
        "val_loss",
        float("nan"),
    )

    history = checkpoint.get(
        "history",
        [],
    )

    # Recover the true best validation loss from training history.
    # This protects against older checkpoints where best_val_loss
    # was saved before the current epoch's best value was updated.
    historical_losses = [
        row["val_loss"]
        for row in history
        if "val_loss" in row
    ]

    if historical_losses:
        historical_best = min(
            historical_losses
        )
    else:
        historical_best = val_loss

    stored_best = checkpoint.get(
        "best_val_loss",
        val_loss,
    )

    if stored_best == float("inf"):
        best_val_loss = historical_best
    else:
        best_val_loss = min(
            stored_best,
            historical_best,
        )

    # Compatibility with the older Task 8 checkpoint.
    if not history:
        history = [
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        ]

    if "rng_state" in checkpoint:
        restore_rng_state(
            checkpoint["rng_state"]
        )

    print(
        f"Resuming after epoch {epoch}"
    )

    print(
        f"Previous train loss: {train_loss:.6f}"
    )

    print(
        f"Previous validation loss: {val_loss:.6f}"
    )

    print(
        f"Best validation loss: {best_val_loss:.6f}"
    )

    return (
        epoch,
        best_val_loss,
        history,
    )


# ============================================================
# HISTORY
# ============================================================

def save_history_csv(
    history: list[dict],
    path: Path,
) -> None:
    """Save training history as CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "epoch_time_seconds",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:
            writer.writerow(
                {
                    "epoch":
                        row["epoch"],

                    "train_loss":
                        f'{row["train_loss"]:.8f}',

                    "val_loss":
                        f'{row["val_loss"]:.8f}',

                    "epoch_time_seconds":
                        f'{row.get("epoch_time_seconds", 0.0):.2f}',
                }
            )


def save_loss_curve(
    history: list[dict],
    path: Path,
) -> None:
    """Save training and validation loss curve."""

    if not history:
        return

    epochs = [
        row["epoch"]
        for row in history
    ]

    train_losses = [
        row["train_loss"]
        for row in history
    ]

    val_losses = [
        row["val_loss"]
        for row in history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Diffusion MSE Loss")
    plt.title(
        "DDPM Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train conditional DDPM "
            "for low-light enhancement."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only a few batches and "
            "save outputs to smoke-test locations."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to checkpoint from which "
            "training should resume."
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

    set_seed(
        config.RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        config.DEVICE
    )

    print("=" * 60)
    print(
        "PHASE 4 — TASK 10: "
        "TRAINING MONITORING & RESUME"
    )
    print("=" * 60)

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Random seed: "
        f"{config.RANDOM_SEED}"
    )

    print(
        f"Batch size: "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Image size: "
        f"{config.IMAGE_SIZE}"
    )

    print(
        f"Diffusion timesteps: "
        f"{config.TIMESTEPS}"
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    if args.smoke_test:

        print("\nMODE: SMOKE TEST")

        print(
            "Only a few batches will be processed."
        )

        checkpoint_root = (
            config.CHECKPOINT_ROOT
            / "smoke_test"
        )

        results_root = (
            config.RESULTS_ROOT
            / "smoke_test"
        )

    else:

        print("\nMODE: FULL TRAINING")

        checkpoint_root = (
            config.CHECKPOINT_ROOT
        )

        results_root = (
            config.RESULTS_ROOT
        )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

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

    print(
        "\nCreating DataLoaders..."
    )

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

    print(
        "\nCreating model..."
    )

    model = create_model(
        device
    )

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

    optimizer = create_optimizer(
        model
    )

    print(
        "Optimizer: AdamW"
    )

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
        create_diffusion_schedule(
            device
        )
    )

    print(
        f"Diffusion schedule: "
        f"{config.TIMESTEPS} steps"
    )

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

    if args.smoke_test:

        epochs = 1

        max_train_batches = 3

        max_val_batches = 2

    else:

        epochs = config.EPOCHS

        max_train_batches = None

        max_val_batches = None

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    history: list[dict] = []

    start_epoch = 1

    if args.resume is not None:

        if args.smoke_test:
            raise ValueError(
                "--resume cannot be combined "
                "with --smoke-test."
            )

        (
            previous_epoch,
            best_val_loss,
            history,
        ) = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        start_epoch = (
            previous_epoch + 1
        )

        if start_epoch > epochs:

            print(
                "\nTraining is already complete "
                "for the configured number of epochs."
            )

            return

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            f"Epoch {epoch}/{epochs}"
        )

        print(
            "-" * 60
        )

        epoch_start_time = time.perf_counter()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            alpha_bars=alpha_bars,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            alpha_bars=alpha_bars,
            device=device,
            max_batches=max_val_batches,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "epoch_time_seconds":
                    epoch_time,
            }
        )

        print(
            f"\nTrain loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        latest_path = (
            checkpoint_root
            / "latest_model.pt"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            history=history,
            path=latest_path,
        )

        print(
            "[CHECKPOINT] Latest model saved: "
            f"{latest_path}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = (
                checkpoint_root
                / "best_model.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                history=history,
                path=best_path,
            )

            print(
                "[CHECKPOINT] New best model saved: "
                f"{best_path}"
            )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if (
            epoch
            % config.SAVE_EVERY_EPOCH
            == 0
        ):

            checkpoint_path = (
                checkpoint_root
                / (
                    f"checkpoint_epoch_"
                    f"{epoch:03d}.pt"
                )
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                history=history,
                path=checkpoint_path,
            )

            print(
                "[CHECKPOINT] Periodic checkpoint saved: "
                f"{checkpoint_path}"
            )

        # ----------------------------------------------------
        # Save history after every epoch
        # ----------------------------------------------------

        history_path = (
            results_root
            / "training_history.pt"
        )

        csv_path = (
            results_root
            / "training_history.csv"
        )

        curve_path = (
            results_root
            / "training_loss_curve.png"
        )

        torch.save(
            history,
            history_path,
        )

        save_history_csv(
            history=history,
            path=csv_path,
        )

        save_loss_curve(
            history=history,
            path=curve_path,
        )

        print(
            "[MONITORING] Training history updated."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "TASK 10 TRAINING RUN COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Training history: "
        f"{results_root / 'training_history.pt'}"
    )

    print(
        f"Training CSV: "
        f"{results_root / 'training_history.csv'}"
    )

    print(
        f"Loss curve: "
        f"{results_root / 'training_loss_curve.png'}"
    )


if __name__ == "__main__":
    main()