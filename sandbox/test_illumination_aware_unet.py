from __future__ import annotations

import torch

from src.models.illumination_aware_unet import (
    IlluminationAwareConditionalUNet,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("Testing Illumination-Aware Conditional U-Net...")
    print()

    device = torch.device("cpu")

    model = IlluminationAwareConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=32,
        time_dim=128,
    ).to(device)

    model.train()

    parameter_count = count_parameters(model)

    print(f"Parameter count: {parameter_count:,}")

    # -----------------------------------------------------
    # Test inputs
    # -----------------------------------------------------

    batch_size = 2

    noisy_image = torch.randn(
        batch_size,
        3,
        256,
        256,
        device=device,
    )

    condition = torch.randn(
        batch_size,
        3,
        256,
        256,
        device=device,
    )

    illumination = torch.rand(
        batch_size,
        1,
        256,
        256,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    print()
    print("Running forward pass...")

    output = model(
        noisy_image=noisy_image,
        condition=condition,
        timesteps=timesteps,
        illumination=illumination,
    )

    print(
        f"Input noisy image shape: "
        f"{tuple(noisy_image.shape)}"
    )

    print(
        f"Condition shape: "
        f"{tuple(condition.shape)}"
    )

    print(
        f"Illumination shape: "
        f"{tuple(illumination.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    # -----------------------------------------------------
    # Output shape check
    # -----------------------------------------------------

    assert output.shape == (
        2,
        3,
        256,
        256,
    ), (
        "Unexpected output shape: "
        f"{tuple(output.shape)}"
    )

    print("Output shape test: PASSED")

    # -----------------------------------------------------
    # Output numerical stability
    # -----------------------------------------------------

    assert torch.isfinite(output).all(), (
        "Output contains NaN or Inf values."
    )

    print("Output finite-value test: PASSED")

    # -----------------------------------------------------
    # Illumination encoder shape verification
    # -----------------------------------------------------

    with torch.no_grad():

        illumination_256, illumination_128, illumination_64 = (
            model.illumination_encoder(
                illumination
            )
        )

    print()
    print("Illumination feature shapes:")

    print(
        f"256x256: "
        f"{tuple(illumination_256.shape)}"
    )

    print(
        f"128x128: "
        f"{tuple(illumination_128.shape)}"
    )

    print(
        f"64x64:   "
        f"{tuple(illumination_64.shape)}"
    )

    assert illumination_256.shape == (
        2,
        32,
        256,
        256,
    )

    assert illumination_128.shape == (
        2,
        64,
        128,
        128,
    )

    assert illumination_64.shape == (
        2,
        128,
        64,
        64,
    )

    print("Illumination feature shape test: PASSED")

    # -----------------------------------------------------
    # Gradient test
    # -----------------------------------------------------

    print()
    print("Testing gradients...")

    target = torch.randn_like(output)

    loss = torch.nn.functional.mse_loss(
        output,
        target,
    )

    print(
        f"Test loss: {loss.item():.6f}"
    )

    loss.backward()

    finite_gradients = True
    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

            if not torch.isfinite(
                parameter.grad
            ).all():

                finite_gradients = False
                break

    print(
        f"Parameters with gradients: "
        f"{gradient_count}"
    )

    assert gradient_count > 0, (
        "No gradients were produced."
    )

    assert finite_gradients, (
        "NaN or Inf detected in gradients."
    )

    print("Gradient finite-value test: PASSED")

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print(
        "All illumination-aware U-Net "
        "tests passed."
    )


if __name__ == "__main__":
    main()