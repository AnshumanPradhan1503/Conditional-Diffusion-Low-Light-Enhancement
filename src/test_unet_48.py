from __future__ import annotations

import torch

from src.models.unet import ConditionalUNet


def main() -> None:
    device = torch.device("cpu")

    print("TASK 15.4.1 — 48-CHANNEL U-NET VERIFICATION")
    print(f"Device: {device}")

    model = ConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=48,
        time_dim=128,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    # ---------------------------------------------------------
    # Forward-pass test
    # ---------------------------------------------------------

    noisy_image = torch.randn(
        2, 3, 256, 256,
        device=device,
        requires_grad=True,
    )

    condition = torch.randn(
        2, 3, 256, 256,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    output = model(
        noisy_image=noisy_image,
        condition=condition,
        timesteps=timesteps,
    )

    print(f"Input shape : {noisy_image.shape}")
    print(f"Output shape: {output.shape}")

    # ---------------------------------------------------------
    # Shape verification
    # ---------------------------------------------------------

    expected_shape = (2, 3, 256, 256)

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    print("[PASS] Output shape")

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Output contains NaN or Inf values."
        )

    print("[PASS] Forward output finite")

    # ---------------------------------------------------------
    # Backward-pass test
    # ---------------------------------------------------------

    loss = output.square().mean()

    print(f"Test loss: {loss.item():.6f}")

    loss.backward()

    # ---------------------------------------------------------
    # Gradient verification
    # ---------------------------------------------------------

    trainable_parameters = 0
    parameters_with_gradients = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters += 1

            if parameter.grad is not None:
                parameters_with_gradients += 1

                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(
                        "NaN or Inf detected in gradients."
                    )

    print(
        f"Trainable tensors: {trainable_parameters}"
    )

    print(
        f"Tensors with gradients: "
        f"{parameters_with_gradients}"
    )

    if trainable_parameters != parameters_with_gradients:
        raise RuntimeError(
            "Some trainable parameters did not receive gradients."
        )

    print("[PASS] Gradients propagated")
    print("[PASS] Gradients finite")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("TASK 15.4.1 VERIFICATION COMPLETED")


if __name__ == "__main__":
    main()