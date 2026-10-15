from __future__ import annotations

import torch

from src.models.adaptive_spatial_unet import AdaptiveSpatialConditionalUNet


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 70)
    print("ADAPTIVE SPATIAL U-NET VERIFICATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(42)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = AdaptiveSpatialConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=32,
        time_dim=128,
    ).to(device)

    model.eval()

    parameter_count = count_parameters(model)

    print("\n[1] Model")
    print(f"Trainable parameters: {parameter_count:,}")

    # ---------------------------------------------------------
    # Dummy inputs
    # ---------------------------------------------------------

    batch_size = 2
    image_size = 256

    noisy_image = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    condition = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    print("\n[2] Input shapes")
    print(f"Noisy image: {tuple(noisy_image.shape)}")
    print(f"Condition:   {tuple(condition.shape)}")
    print(f"Timesteps:   {tuple(timesteps.shape)}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():
        output = model(
            noisy_image=noisy_image,
            condition=condition,
            timesteps=timesteps,
        )

    print("\n[3] Forward pass")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output finite: {torch.isfinite(output).all().item()}")

    # ---------------------------------------------------------
    # Shape verification
    # ---------------------------------------------------------

    expected_shape = (
        batch_size,
        3,
        image_size,
        image_size,
    )

    assert output.shape == expected_shape, (
        f"Unexpected output shape: {output.shape}; "
        f"expected {expected_shape}"
    )

    assert torch.isfinite(output).all(), (
        "Model output contains NaN or Inf values."
    )

    print("Shape verification: PASS")
    print("Finite-output verification: PASS")

    # ---------------------------------------------------------
    # Epsilon MSE
    # ---------------------------------------------------------

    target_noise = torch.randn_like(output)

    loss = torch.nn.functional.mse_loss(
        output,
        target_noise,
    )

    print("\n[4] Test loss")
    print(f"Epsilon MSE: {loss.item():.6f}")

    assert torch.isfinite(loss), (
        "Loss contains NaN or Inf."
    )

    print("Loss verification: PASS")

    # ---------------------------------------------------------
    # Gradient verification
    # ---------------------------------------------------------

    model.train()

    model.zero_grad(set_to_none=True)

    output_train = model(
        noisy_image=noisy_image,
        condition=condition,
        timesteps=timesteps,
    )

    train_loss = torch.nn.functional.mse_loss(
        output_train,
        target_noise,
    )

    train_loss.backward()

    print("\n[5] Gradient verification")

    trainable_tensors = 0
    tensors_with_gradients = 0
    tensors_with_finite_gradients = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        trainable_tensors += 1

        if parameter.grad is not None:
            tensors_with_gradients += 1

            if torch.isfinite(parameter.grad).all():
                tensors_with_finite_gradients += 1
            else:
                print(
                    f"Non-finite gradient detected: {name}"
                )

    print(f"Trainable tensors: {trainable_tensors}")
    print(f"Tensors with gradients: {tensors_with_gradients}")
    print(
        "Tensors with finite gradients: "
        f"{tensors_with_finite_gradients}"
    )

    assert trainable_tensors > 0
    assert tensors_with_gradients == trainable_tensors
    assert tensors_with_finite_gradients == trainable_tensors

    print("Gradient verification: PASS")

    # ---------------------------------------------------------
    # Adaptive gate verification
    # ---------------------------------------------------------

    print("\n[6] Adaptive gate verification")

    assert hasattr(model, "last_gates"), (
        "Model does not expose last_gates."
    )

    required_scales = {"256", "128", "64"}

    assert set(model.last_gates.keys()) == required_scales, (
        f"Unexpected gate scales: {model.last_gates.keys()}"
    )

    for scale in ("256", "128", "64"):

        gate = model.last_gates[scale]

        print(f"\nScale {scale}:")
        print(f"Gate shape: {tuple(gate.shape)}")
        print(
            f"Gate min: {gate.min().item():.6f}"
        )
        print(
            f"Gate max: {gate.max().item():.6f}"
        )
        print(
            f"Gate mean: {gate.mean().item():.6f}"
        )

        assert torch.isfinite(gate).all(), (
            f"Gate at scale {scale} contains NaN or Inf."
        )

        assert gate.min().item() >= 0.0, (
            f"Gate at scale {scale} is below 0."
        )

        assert gate.max().item() <= 1.0, (
            f"Gate at scale {scale} is above 1."
        )

        print(f"Scale {scale} gate range: PASS")

    # ---------------------------------------------------------
    # Gate gradient verification
    # ---------------------------------------------------------

    print("\n[7] Gate gradient verification")

    gate_gradient_parameters = []

    for name, parameter in model.named_parameters():

        if "adaptive_gate" in name:

            gate_gradient_parameters.append(
                (name, parameter)
            )

    print(
        "Adaptive gate parameter tensors: "
        f"{len(gate_gradient_parameters)}"
    )

    assert len(gate_gradient_parameters) > 0

    gate_parameters_with_gradients = 0

    for name, parameter in gate_gradient_parameters:

        if parameter.grad is not None:

            gate_parameters_with_gradients += 1

            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite gate gradient: {name}"
            )

    print(
        "Gate parameters with gradients: "
        f"{gate_parameters_with_gradients}"
    )

    assert (
        gate_parameters_with_gradients
        == len(gate_gradient_parameters)
    )

    print("Gate gradient verification: PASS")

    # ---------------------------------------------------------
    # Conditioning pathway gradient verification
    # ---------------------------------------------------------

    print("\n[8] Conditioning pathway verification")

    condition_keywords = (
        "condition_256",
        "condition_down_128",
        "condition_128",
        "condition_down_64",
        "condition_64",
    )

    condition_parameters = []

    for name, parameter in model.named_parameters():

        if any(
            keyword in name
            for keyword in condition_keywords
        ):
            condition_parameters.append(
                (name, parameter)
            )

    condition_with_gradients = sum(
        parameter.grad is not None
        for _, parameter in condition_parameters
    )

    condition_with_finite_gradients = sum(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for _, parameter in condition_parameters
    )

    print(
        "Conditioning parameter tensors: "
        f"{len(condition_parameters)}"
    )

    print(
        "Conditioning tensors with gradients: "
        f"{condition_with_gradients}"
    )

    print(
        "Conditioning tensors with finite gradients: "
        f"{condition_with_finite_gradients}"
    )

    assert len(condition_parameters) > 0

    assert (
        condition_with_gradients
        == len(condition_parameters)
    )

    assert (
        condition_with_finite_gradients
        == len(condition_parameters)
    )

    print("Conditioning pathway verification: PASS")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL ADAPTIVE SPATIAL U-NET CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
