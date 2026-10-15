from __future__ import annotations

import torch

from src.models.illumination_aware_unet_v2 import (
    IlluminationAwareConditionalUNet,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 70)
    print("Illumination-Aware U-Net V2 - Architecture Verification")
    print("=" * 70)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    device = torch.device("cpu")

    batch_size = 2
    channels = 3
    image_size = 256
    time_dim = 128
    base_channels = 32

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = IlluminationAwareConditionalUNet(
        image_channels=channels,
        condition_channels=channels,
        base_channels=base_channels,
        time_dim=time_dim,
    ).to(device)

    model.train()

    print("\n[1] Model created")
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    parameters = count_parameters(model)

    print("\n[2] Parameter count")
    print(f"Trainable parameters: {parameters:,}")

    # ---------------------------------------------------------
    # Create dummy inputs
    # ---------------------------------------------------------

    torch.manual_seed(42)

    noisy_image = torch.randn(
        batch_size,
        channels,
        image_size,
        image_size,
        device=device,
    )

    condition = torch.randn(
        batch_size,
        channels,
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

    print("\n[3] Input shapes")
    print(f"Noisy image:   {tuple(noisy_image.shape)}")
    print(f"Condition:     {tuple(condition.shape)}")
    print(f"Illumination:  {tuple(illumination.shape)}")
    print(f"Timesteps:     {tuple(timesteps.shape)}")

    # ---------------------------------------------------------
    # Verify zero initialization of illumination projections
    # ---------------------------------------------------------

    print("\n[4] Illumination projection initialization")

    projections = {
        "256": model.illumination_encoder.projection_256,
        "128": model.illumination_encoder.projection_128,
        "64": model.illumination_encoder.projection_64,
    }

    for name, layer in projections.items():
        weight_max = layer.weight.detach().abs().max().item()
        bias_max = layer.bias.detach().abs().max().item()

        print(
            f"Projection {name}: "
            f"max|weight|={weight_max:.10f}, "
            f"max|bias|={bias_max:.10f}"
        )

        assert weight_max == 0.0, (
            f"Projection {name} weights are not zero initialized."
        )

        assert bias_max == 0.0, (
            f"Projection {name} bias is not zero initialized."
        )

    print("Zero initialization: PASSED")

    # ---------------------------------------------------------
    # Illumination encoder shape test
    # ---------------------------------------------------------

    print("\n[5] Illumination encoder shape test")

    with torch.no_grad():
        illumination_256, illumination_128, illumination_64 = (
            model.illumination_encoder(illumination)
        )

    print(
        f"Illumination 256: {tuple(illumination_256.shape)}"
    )

    print(
        f"Illumination 128: {tuple(illumination_128.shape)}"
    )

    print(
        f"Illumination 64:  {tuple(illumination_64.shape)}"
    )

    assert illumination_256.shape == (
        batch_size,
        base_channels,
        image_size,
        image_size,
    )

    assert illumination_128.shape == (
        batch_size,
        base_channels,
        image_size // 2,
        image_size // 2,
    )

    assert illumination_64.shape == (
        batch_size,
        base_channels * 2,
        image_size // 4,
        image_size // 4,
    )

    print("Illumination feature shapes: PASSED")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    print("\n[6] Forward pass")

    output = model(
        noisy_image=noisy_image,
        condition=condition,
        timesteps=timesteps,
        illumination=illumination,
    )

    print(f"Output shape: {tuple(output.shape)}")

    assert output.shape == (
        batch_size,
        channels,
        image_size,
        image_size,
    )

    print("Output shape: PASSED")

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    print("\n[7] Numerical stability")

    finite_output = torch.isfinite(output).all().item()

    print(f"Output finite: {finite_output}")

    assert finite_output, (
        "Model output contains NaN or Inf values."
    )

    print("Numerical stability: PASSED")

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    print("\n[8] Backward pass")

    target = torch.randn_like(output)

    loss = torch.mean(
        (output - target) ** 2
    )

    print(f"Test loss: {loss.item():.6f}")

    loss.backward()

    # ---------------------------------------------------------
    # Gradient verification
    # ---------------------------------------------------------

    print("\n[9] Gradient verification")

    total_gradients = 0
    finite_gradients = 0
    zero_gradients = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            total_gradients += 1

            if parameter.grad is not None:

                if torch.isfinite(parameter.grad).all():
                    finite_gradients += 1

                if torch.all(parameter.grad == 0):
                    zero_gradients += 1

    print(f"Trainable tensors: {total_gradients}")
    print(f"Finite gradients:  {finite_gradients}")
    print(f"Zero gradients:    {zero_gradients}")

    assert finite_gradients == total_gradients, (
        "Some gradients contain NaN or Inf values."
    )

    print("Gradient verification: PASSED")

    # ---------------------------------------------------------
    # Specifically verify illumination pathway gradients
    # ---------------------------------------------------------

    print("\n[10] Illumination pathway gradient verification")

    illumination_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if "illumination_encoder" in name
        and parameter.requires_grad
    ]

    illumination_with_grad = 0
    illumination_finite_grad = 0

    for name in illumination_gradient_names:

        parameter = dict(
            model.named_parameters()
        )[name]

        if parameter.grad is not None:

            illumination_with_grad += 1

            if torch.isfinite(parameter.grad).all():
                illumination_finite_grad += 1

    print(
        f"Illumination trainable tensors: "
        f"{len(illumination_gradient_names)}"
    )

    print(
        f"Illumination tensors with gradients: "
        f"{illumination_with_grad}"
    )

    print(
        f"Illumination tensors with finite gradients: "
        f"{illumination_finite_grad}"
    )

    assert illumination_with_grad > 0, (
        "No illumination pathway gradients detected."
    )

    assert (
        illumination_finite_grad
        == len(illumination_gradient_names)
    ), (
        "Some illumination pathway gradients are not finite."
    )

    print(
        "Illumination gradient verification: PASSED"
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL ILLUMINATION-AWARE U-NET V2 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()