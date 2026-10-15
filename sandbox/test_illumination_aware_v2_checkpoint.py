from __future__ import annotations

import torch

from src.models.illumination_aware_unet_v2 import (
    IlluminationAwareConditionalUNet,
)
from src.configs import illumination_aware_v2_training_config as config


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 70)
    print("Illumination-Aware U-Net V2 - Full Checkpoint Verification")
    print("=" * 70)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # 1. Check checkpoint
    # ---------------------------------------------------------

    checkpoint_path = (
        config.CHECKPOINT_ROOT / "best_model.pt"
    )

    print("\n[1] Checkpoint path")
    print(checkpoint_path)

    assert checkpoint_path.exists(), (
        f"Checkpoint not found: {checkpoint_path}"
    )

    print("Checkpoint exists: PASSED")

    # ---------------------------------------------------------
    # 2. Load checkpoint
    # ---------------------------------------------------------

    print("\n[2] Loading checkpoint")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    print("Checkpoint loaded: PASSED")

    # ---------------------------------------------------------
    # 3. Verify checkpoint metadata
    # ---------------------------------------------------------

    print("\n[3] Checkpoint metadata")

    actual_epoch = checkpoint.get("epoch")
    actual_best_val_loss = checkpoint.get("best_val_loss")

    print(f"Epoch: {actual_epoch}")
    print(f"Best validation loss: {actual_best_val_loss}")

    expected_epoch = 14
    expected_best_val_loss = 0.020195

    assert actual_epoch == expected_epoch, (
        f"Expected best checkpoint from epoch {expected_epoch}, "
        f"but found epoch {actual_epoch}."
    )

    print("Best epoch metadata: PASSED")

    assert actual_best_val_loss is not None

    assert abs(
        actual_best_val_loss - expected_best_val_loss
    ) < 1e-5, (
        f"Unexpected best validation loss: "
        f"{actual_best_val_loss}"
    )

    print("Best validation loss metadata: PASSED")

    # ---------------------------------------------------------
    # 4. Create V2 model
    # ---------------------------------------------------------

    print("\n[4] Creating V2 model")

    model = IlluminationAwareConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    expected_parameters = 1_239_811
    actual_parameters = count_parameters(model)

    print(
        f"Expected parameters: {expected_parameters:,}"
    )

    print(
        f"Actual parameters:   {actual_parameters:,}"
    )

    assert actual_parameters == expected_parameters, (
        "Parameter count does not match the verified V2 architecture."
    )

    print("Parameter count: PASSED")

    # ---------------------------------------------------------
    # 5. Load state dictionary
    # ---------------------------------------------------------

    print("\n[5] Checking model state dictionary")

    assert "model_state_dict" in checkpoint, (
        "Checkpoint does not contain model_state_dict."
    )

    state_dict = checkpoint["model_state_dict"]

    print(
        f"Checkpoint tensors: {len(state_dict)}"
    )

    model.load_state_dict(state_dict)

    print("State dictionary compatibility: PASSED")

    # ---------------------------------------------------------
    # 6. Verify illumination pathway
    # ---------------------------------------------------------

    print("\n[6] Checking illumination pathway")

    illumination_keys = [
        key
        for key in state_dict.keys()
        if key.startswith("illumination_encoder.")
    ]

    print(
        f"Illumination pathway tensors: "
        f"{len(illumination_keys)}"
    )

    assert len(illumination_keys) == 30, (
        "Unexpected number of illumination pathway tensors."
    )

    required_keys = [
        "illumination_encoder.projection_256.weight",
        "illumination_encoder.projection_256.bias",
        "illumination_encoder.projection_128.weight",
        "illumination_encoder.projection_128.bias",
        "illumination_encoder.projection_64.weight",
        "illumination_encoder.projection_64.bias",
    ]

    for key in required_keys:
        assert key in state_dict, (
            f"Missing required illumination key: {key}"
        )

    print("Required illumination parameters: PASSED")
    print("Illumination pathway: PASSED")

    # ---------------------------------------------------------
    # 7. Verify illumination projection values
    # ---------------------------------------------------------

    print("\n[7] Checking illumination projections")

    projection_names = [
        "projection_256",
        "projection_128",
        "projection_64",
    ]

    for projection_name in projection_names:

        weight_key = (
            f"illumination_encoder."
            f"{projection_name}.weight"
        )

        bias_key = (
            f"illumination_encoder."
            f"{projection_name}.bias"
        )

        weight = state_dict[weight_key]
        bias = state_dict[bias_key]

        weight_max = (
            weight.detach()
            .abs()
            .max()
            .item()
        )

        bias_max = (
            bias.detach()
            .abs()
            .max()
            .item()
        )

        print(
            f"{projection_name}: "
            f"max|weight|={weight_max:.10f}, "
            f"max|bias|={bias_max:.10f}"
        )

        assert torch.isfinite(weight).all(), (
            f"Non-finite values in {weight_key}"
        )

        assert torch.isfinite(bias).all(), (
            f"Non-finite values in {bias_key}"
        )

    print("Projection values: PASSED")

    # ---------------------------------------------------------
    # 8. Check entire checkpoint for NaN/Inf
    # ---------------------------------------------------------

    print("\n[8] Checking checkpoint numerical stability")

    non_finite_keys = []

    for key, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            non_finite_keys.append(key)

    print(
        f"Non-finite tensors: {len(non_finite_keys)}"
    )

    assert len(non_finite_keys) == 0, (
        "Checkpoint contains NaN or Inf values."
    )

    print("Checkpoint numerical stability: PASSED")

    # ---------------------------------------------------------
    # 9. Forward-pass verification
    # ---------------------------------------------------------

    print("\n[9] Forward-pass verification")

    model.eval()

    torch.manual_seed(42)

    batch_size = 2
    image_size = config.IMAGE_SIZE

    noisy_image = torch.randn(
        batch_size,
        config.IMAGE_CHANNELS,
        image_size,
        image_size,
        device=device,
    )

    condition = torch.randn(
        batch_size,
        config.CONDITION_CHANNELS,
        image_size,
        image_size,
        device=device,
    )

    illumination = torch.rand(
        batch_size,
        1,
        image_size,
        image_size,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        output = model(
            noisy_image=noisy_image,
            condition=condition,
            timesteps=timesteps,
            illumination=illumination,
        )

    print(
        f"Output shape: {tuple(output.shape)}"
    )

    expected_shape = (
        batch_size,
        config.IMAGE_CHANNELS,
        image_size,
        image_size,
    )

    assert output.shape == expected_shape

    print("Output shape: PASSED")

    assert torch.isfinite(output).all()

    print("Output numerical stability: PASSED")

    # ---------------------------------------------------------
    # 10. Original control checkpoint protection
    # ---------------------------------------------------------

    print("\n[10] Original control checkpoint protection")

    original_checkpoint_path = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "best_model.pt"
    )

    print(
        f"Original checkpoint: "
        f"{original_checkpoint_path}"
    )

    assert original_checkpoint_path.exists(), (
        "Original control checkpoint is missing."
    )

    original_checkpoint = torch.load(
        original_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    print(
        f"Original checkpoint epoch: "
        f"{original_checkpoint.get('epoch')}"
    )

    original_model = torch.nn.Module()

    from src.models.unet import ConditionalUNet

    original_model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    original_parameter_count = count_parameters(
        original_model
    )

    expected_original_parameters = 1_162_947

    print(
        f"Original expected parameters: "
        f"{expected_original_parameters:,}"
    )

    print(
        f"Original actual parameters:   "
        f"{original_parameter_count:,}"
    )

    assert (
        original_parameter_count
        == expected_original_parameters
    )

    original_model.load_state_dict(
        original_checkpoint["model_state_dict"]
    )

    print("Original control checkpoint: PASSED")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL V2 FULL CHECKPOINT VERIFICATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
