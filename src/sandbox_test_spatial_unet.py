from __future__ import annotations

import torch

from src.models.spatial_unet import SpatialConditionalUNet
from src.configs import training_config
from src.diffusion.schedule import get_diffusion_parameters
from src.diffusion.forward import q_sample


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    device = torch.device(training_config.DEVICE)

    print("TASK 22.6.2 - SPATIAL CONDITIONING VERIFICATION")
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = SpatialConditionalUNet(
        image_channels=training_config.IMAGE_CHANNELS,
        condition_channels=training_config.CONDITION_CHANNELS,
        base_channels=training_config.BASE_CHANNELS,
        time_dim=training_config.TIME_EMBEDDING_DIM,
    ).to(device)

    model.train()

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ---------------------------------------------------------
    # Synthetic inputs
    # ---------------------------------------------------------
    batch_size = 2
    image_size = training_config.IMAGE_SIZE

    target = torch.randn(
        batch_size,
        training_config.IMAGE_CHANNELS,
        image_size,
        image_size,
        device=device,
    )

    condition = torch.randn(
        batch_size,
        training_config.CONDITION_CHANNELS,
        image_size,
        image_size,
        device=device,
    )

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------
    # Diffusion schedule
    # ---------------------------------------------------------
    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=training_config.TIMESTEPS,
        beta_start=training_config.BETA_START,
        beta_end=training_config.BETA_END,
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    # ---------------------------------------------------------
    # Forward diffusion
    # ---------------------------------------------------------
    noisy_target, noise = q_sample(
        x_start=target,
        timesteps=timesteps,
        alpha_bars=alpha_bars,
    )

    # ---------------------------------------------------------
    # Model forward
    # ---------------------------------------------------------
    prediction = model(
        noisy_target,
        condition,
        timesteps,
    )

    # ---------------------------------------------------------
    # Output verification
    # ---------------------------------------------------------
    print(
        f"Input noisy shape : "
        f"{tuple(noisy_target.shape)}"
    )

    print(
        f"Condition shape   : "
        f"{tuple(condition.shape)}"
    )

    print(
        f"Timestep shape    : "
        f"{tuple(timesteps.shape)}"
    )

    print(
        f"Output shape      : "
        f"{tuple(prediction.shape)}"
    )

    expected_shape = (
        batch_size,
        training_config.IMAGE_CHANNELS,
        image_size,
        image_size,
    )

    if tuple(prediction.shape) == expected_shape:
        print("[PASS] Output shape")
    else:
        print("[FAIL] Output shape")

    if torch.isfinite(prediction).all():
        print("[PASS] Output finite")
    else:
        print("[FAIL] Output contains NaN/Inf")

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------
    loss = torch.nn.functional.mse_loss(
        prediction,
        noise,
    )

    print(
        f"Test epsilon MSE: "
        f"{loss.item():.6f}"
    )

    if torch.isfinite(loss):
        print("[PASS] Loss finite")
    else:
        print("[FAIL] Loss contains NaN/Inf")

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------
    model.zero_grad(set_to_none=True)

    loss.backward()

    trainable_tensors = 0
    tensors_with_gradients = 0
    finite_gradients = True

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        trainable_tensors += 1

        if parameter.grad is not None:
            tensors_with_gradients += 1

            if not torch.isfinite(parameter.grad).all():
                finite_gradients = False

    print(
        f"Trainable tensors      : "
        f"{trainable_tensors}"
    )

    print(
        f"Tensors with gradients : "
        f"{tensors_with_gradients}"
    )

    if tensors_with_gradients == trainable_tensors:
        print("[PASS] Gradients propagated")
    else:
        print("[FAIL] Missing gradients")

    if finite_gradients:
        print("[PASS] Gradients finite")
    else:
        print("[FAIL] Non-finite gradients")

    # ---------------------------------------------------------
    # Verify spatial-conditioning layers receive gradients
    # ---------------------------------------------------------
    spatial_layers = [
        "condition_256",
        "condition_down_128",
        "condition_128",
        "condition_down_64",
        "condition_64",
    ]

    print(
        "\nSPATIAL CONDITIONING GRADIENT CHECK"
    )

    all_spatial_gradients = True

    for layer_name in spatial_layers:

        layer = getattr(model, layer_name)

        layer_has_gradient = False

        for parameter in layer.parameters():

            if parameter.grad is not None:
                layer_has_gradient = True

                if not torch.isfinite(
                    parameter.grad
                ).all():
                    all_spatial_gradients = False

        print(
            f"{layer_name:22s}: "
            f"{'PASS' if layer_has_gradient else 'FAIL'}"
        )

    if all_spatial_gradients:
        print(
            "[PASS] Spatial conditioning "
            "receives gradients"
        )
    else:
        print(
            "[FAIL] Spatial conditioning "
            "gradient problem"
        )


if __name__ == "__main__":
    main()