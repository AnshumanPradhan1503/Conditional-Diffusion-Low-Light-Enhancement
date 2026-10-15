from __future__ import annotations

from pathlib import Path

import torch

from src.configs import illumination_aware_training_config as illumination_config
from src.configs import training_config as original_config
from src.models.illumination_aware_unet import IlluminationAwareConditionalUNet
from src.models.unet import ConditionalUNet


EXPECTED_ILLUMINATION_PARAMS = 926_979
EXPECTED_ORIGINAL_PARAMS = 1_162_947
EXPECTED_BEST_EPOCH = 19
EXPECTED_BEST_VAL_LOSS = 0.265206


def count_parameters(model) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def verify_illumination_checkpoint(checkpoint_path: Path) -> None:
    print("\n" + "-" * 60)
    print(f"Checking: {checkpoint_path.name}")
    print("-" * 60)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "train_loss",
        "val_loss",
        "best_val_loss",
        "history",
    }

    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise RuntimeError(
            f"Missing checkpoint keys: {missing_keys}"
        )

    model = IlluminationAwareConditionalUNet(
        image_channels=illumination_config.IMAGE_CHANNELS,
        condition_channels=illumination_config.CONDITION_CHANNELS,
        base_channels=illumination_config.BASE_CHANNELS,
        time_dim=illumination_config.TIME_EMBEDDING_DIM,
    )

    actual_parameters = count_parameters(model)

    print(f"Model parameters: {actual_parameters:,}")

    if actual_parameters != EXPECTED_ILLUMINATION_PARAMS:
        raise RuntimeError(
            f"Unexpected parameter count: {actual_parameters:,} "
            f"(expected {EXPECTED_ILLUMINATION_PARAMS:,})"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # --------------------------------------------------------
    # Full-training best-checkpoint verification
    # --------------------------------------------------------

    if checkpoint_path.name == "best_model.pt":
        if checkpoint["epoch"] != EXPECTED_BEST_EPOCH:
            raise RuntimeError(
                f"Best checkpoint epoch is {checkpoint['epoch']} "
                f"(expected {EXPECTED_BEST_EPOCH})"
            )

        if abs(
            checkpoint["best_val_loss"]
            - EXPECTED_BEST_VAL_LOSS
        ) > 1e-5:
            raise RuntimeError(
                "Unexpected best validation loss: "
                f"{checkpoint['best_val_loss']}"
            )

        print("Best epoch verification: PASSED")
        print("Best validation-loss verification: PASSED")

    # --------------------------------------------------------
    # Forward-pass verification
    # --------------------------------------------------------

    torch.manual_seed(42)

    noisy_image = torch.randn(
        1, 3, 256, 256
    )

    condition = torch.randn(
        1, 3, 256, 256
    )

    illumination = torch.randn(
        1, 1, 256, 256
    )

    timesteps = torch.tensor(
        [500],
        dtype=torch.long,
    )

    with torch.no_grad():
        output = model(
            noisy_image=noisy_image,
            condition=condition,
            timesteps=timesteps,
            illumination=illumination,
        )

    expected_shape = (
        1,
        3,
        256,
        256,
    )

    print(
        f"Forward output shape: {tuple(output.shape)}"
    )

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected forward output shape: "
            f"{tuple(output.shape)}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Checkpoint produced non-finite output."
        )

    print("Forward-pass verification: PASSED")
    print("Checkpoint verification: PASSED")


def verify_original_control() -> None:
    print("\n" + "-" * 60)
    print("Checking original control checkpoint")
    print("-" * 60)

    original_checkpoint = (
        original_config.CHECKPOINT_ROOT
        / "best_model.pt"
    )

    if not original_checkpoint.exists():
        raise FileNotFoundError(
            "Original control checkpoint is missing."
        )

    model = ConditionalUNet(
        image_channels=original_config.IMAGE_CHANNELS,
        condition_channels=original_config.CONDITION_CHANNELS,
        base_channels=original_config.BASE_CHANNELS,
        time_dim=original_config.TIME_EMBEDDING_DIM,
    )

    actual_parameters = count_parameters(model)

    print(
        f"Original model parameters: "
        f"{actual_parameters:,}"
    )

    if actual_parameters != EXPECTED_ORIGINAL_PARAMS:
        raise RuntimeError(
            f"Original control model has unexpected "
            f"parameter count: {actual_parameters:,}"
        )

    checkpoint = torch.load(
        original_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Original checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Original validation loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    if checkpoint["epoch"] != 11:
        raise RuntimeError(
            "Original control checkpoint is not "
            "the expected epoch 11 checkpoint."
        )

    print(
        "Original control checkpoint: PASSED"
    )


def main() -> None:
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    illumination_root = (
        project_root
        / "checkpoints"
        / "illumination_aware"
    )

    print("=" * 60)
    print(
        "TASK 24.4C — FULL TRAINING ARTIFACT VERIFICATION"
    )
    print("=" * 60)

    print(
        "\nIllumination-aware checkpoint directory:"
    )
    print(illumination_root)

    # Full-training checkpoints
    best_checkpoint = (
        illumination_root / "best_model.pt"
    )

    latest_checkpoint = (
        illumination_root / "latest_model.pt"
    )

    verify_illumination_checkpoint(
        best_checkpoint
    )

    verify_illumination_checkpoint(
        latest_checkpoint
    )

    # Original control
    verify_original_control()

    print("\n" + "=" * 60)
    print(
        "ALL FULL-TRAINING ARTIFACT VERIFICATIONS PASSED"
    )
    print("=" * 60)

    print(
        "\nIllumination-aware best checkpoint:"
    )
    print(
        "Epoch 19 / validation loss 0.265206"
    )

    print(
        "\nOriginal control checkpoint:"
    )
    print(
        "Epoch 11 / validation loss 0.022893"
    )

    print(
        "\nBoth experiments are safely preserved."
    )


if __name__ == "__main__":
    main()