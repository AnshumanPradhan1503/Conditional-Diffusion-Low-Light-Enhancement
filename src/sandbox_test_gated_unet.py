from __future__ import annotations

import torch

from src.models.multiscale_gated_unet import (
    MultiScaleGatedConditionalUNet,
)


def main() -> None:
    print("=" * 70)
    print("MULTI-SCALE GATED U-NET VERIFICATION")
    print("=" * 70)

    device = torch.device("cpu")

    model = MultiScaleGatedConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=32,
        time_dim=128,
        gate_initial_value=0.1,
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters     : {parameter_count:,}")
    print(f"Trainable parameters : {trainable_parameter_count:,}")

    # ---------------------------------------------------------
    # Input tensors
    # ---------------------------------------------------------

    noisy_image = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    condition = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    prediction = model(
        noisy_image=noisy_image,
        condition=condition,
        timesteps=timesteps,
    )

    print(f"Input shape           : {tuple(noisy_image.shape)}")
    print(f"Condition shape       : {tuple(condition.shape)}")
    print(f"Output shape          : {tuple(prediction.shape)}")

    # ---------------------------------------------------------
    # Shape check
    # ---------------------------------------------------------

    expected_shape = (2, 3, 256, 256)

    if tuple(prediction.shape) == expected_shape:
        print("[PASS] Output shape")
    else:
        print("[FAIL] Output shape")

    # ---------------------------------------------------------
    # Finite check
    # ---------------------------------------------------------

    if torch.isfinite(prediction).all():
        print("[PASS] Output finite")
    else:
        print("[FAIL] Output contains NaN or Inf")

    # ---------------------------------------------------------
    # Gate values
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("INITIAL CONDITIONING GATES")
    print("-" * 70)

    print(
        f"Gate 256 : "
        f"{model.condition_gate_256.gate.item():.6f}"
    )

    print(
        f"Gate 128 : "
        f"{model.condition_gate_128.gate.item():.6f}"
    )

    print(
        f"Gate 64  : "
        f"{model.condition_gate_64.gate.item():.6f}"
    )

    # ---------------------------------------------------------
    # Gradient test
    # ---------------------------------------------------------

    target = torch.randn_like(prediction)

    loss = torch.mean(
        (prediction - target) ** 2
    )

    loss.backward()

    print()
    print("-" * 70)
    print("GRADIENT CHECK")
    print("-" * 70)

    tensors_with_gradients = 0
    tensors_with_finite_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:
            tensors_with_gradients += 1

            if torch.isfinite(parameter.grad).all():
                tensors_with_finite_gradients += 1

    total_trainable_tensors = sum(
        1
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable tensors            : "
        f"{total_trainable_tensors}"
    )

    print(
        f"Tensors with gradients       : "
        f"{tensors_with_gradients}"
    )

    print(
        f"Tensors with finite gradients: "
        f"{tensors_with_finite_gradients}"
    )

    if tensors_with_gradients == total_trainable_tensors:
        print("[PASS] Gradients propagated")
    else:
        print("[FAIL] Some tensors have no gradients")

    if tensors_with_finite_gradients == total_trainable_tensors:
        print("[PASS] Gradients finite")
    else:
        print("[FAIL] Some gradients contain NaN or Inf")

    print()
    print("-" * 70)
    print(f"Test loss: {loss.item():.6f}")
    print("-" * 70)

    print()
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
