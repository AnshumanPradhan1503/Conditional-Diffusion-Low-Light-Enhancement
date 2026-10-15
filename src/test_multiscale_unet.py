
from __future__ import annotations

import torch

from src.models.multiscale_unet import MultiScaleConditionalUNet


def main() -> None:
    print("=" * 70)
    print("TASK 18 — MULTI-SCALE U-NET VERIFICATION")
    print("=" * 70)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    batch_size = 2
    image_channels = 3
    image_size = 256
    base_channels = 32
    time_dim = 128

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = MultiScaleConditionalUNet(
        image_channels=image_channels,
        condition_channels=image_channels,
        base_channels=base_channels,
        time_dim=time_dim,
    ).to(device)

    model.train()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Model parameters: {parameter_count:,}")

    # ---------------------------------------------------------
    # Create test inputs
    # ---------------------------------------------------------

    noisy_image = torch.randn(
        batch_size,
        image_channels,
        image_size,
        image_size,
        device=device,
    )

    condition = torch.randn(
        batch_size,
        image_channels,
        image_size,
        image_size,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    print(f"Noisy image shape : {noisy_image.shape}")
    print(f"Condition shape   : {condition.shape}")
    print(f"Timesteps shape   : {timesteps.shape}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    output = model(
        noisy_image=noisy_image,
        condition=condition,
        timesteps=timesteps,
    )

    print(f"Output shape      : {output.shape}")

    # ---------------------------------------------------------
    # Output shape verification
    # ---------------------------------------------------------

    expected_shape = (
        batch_size,
        image_channels,
        image_size,
        image_size,
    )

    if output.shape != expected_shape:
        raise AssertionError(
            f"Expected output shape {expected_shape}, "
            f"got {output.shape}"
        )

    print("[PASS] Output shape")

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    if not torch.isfinite(output).all():
        raise AssertionError(
            "Model output contains NaN or Inf"
        )

    print("[PASS] Forward output finite")

    # ---------------------------------------------------------
    # Loss and backward pass
    # ---------------------------------------------------------

    target_noise = torch.randn_like(output)

    loss = torch.nn.functional.mse_loss(
        output,
        target_noise,
    )

    print(f"Test loss: {loss.item():.6f}")

    model.zero_grad(set_to_none=True)

    loss.backward()

    # ---------------------------------------------------------
    # Gradient verification
    # ---------------------------------------------------------

    trainable_tensors = 0
    tensors_with_gradients = 0
    nonfinite_gradients = 0

    for parameter in model.parameters():

        if parameter.requires_grad:
            trainable_tensors += 1

            if parameter.grad is not None:
                tensors_with_gradients += 1

                if not torch.isfinite(
                    parameter.grad
                ).all():
                    nonfinite_gradients += 1

    print(
        f"Trainable tensors: {trainable_tensors}"
    )

    print(
        f"Tensors with gradients: "
        f"{tensors_with_gradients}"
    )

    if tensors_with_gradients != trainable_tensors:
        raise AssertionError(
            "Not all trainable tensors received gradients"
        )

    print("[PASS] Gradients propagated")

    if nonfinite_gradients != 0:
        raise AssertionError(
            "Non-finite gradients detected"
        )

    print("[PASS] Gradients finite")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TASK 18 — VERIFICATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
