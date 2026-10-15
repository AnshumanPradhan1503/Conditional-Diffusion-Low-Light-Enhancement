from __future__ import annotations

import torch

from src.configs import training_config as config
from src.diffusion.improved_loss import (
    improved_diffusion_training_loss,
)
from src.diffusion.schedule import (
    get_diffusion_parameters,
)
from src.models.unet import ConditionalUNet


def main():

    print("=" * 60)
    print(
        "TASK 15.5.3 — IMPROVED LOSS VERIFICATION"
    )
    print("=" * 60)

    device = torch.device(
        config.DEVICE
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Synthetic batch
    # --------------------------------------------------------

    batch_size = 2

    low_light = torch.randn(
        batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )

    normal_light = torch.randn(
        batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )

    normal_light = normal_light.clamp(
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    _, _, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    alpha_bars = alpha_bars.to(
        device
    )

    # --------------------------------------------------------
    # Forward + loss
    # --------------------------------------------------------

    model.train()

    total_loss, epsilon_loss, x0_loss = (
        improved_diffusion_training_loss(
            model=model,
            low_light=low_light,
            normal_light=normal_light,
            alpha_bars=alpha_bars,
            x0_weight=0.1,
        )
    )

    print(
        f"\nTotal loss   : "
        f"{total_loss.item():.6f}"
    )

    print(
        f"Epsilon loss : "
        f"{epsilon_loss.item():.6f}"
    )

    print(
        f"x0 loss      : "
        f"{x0_loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------

    assert torch.isfinite(
        total_loss
    ), "Total loss contains NaN/Inf"

    assert torch.isfinite(
        epsilon_loss
    ), "Epsilon loss contains NaN/Inf"

    assert torch.isfinite(
        x0_loss
    ), "x0 loss contains NaN/Inf"

    print(
        "\n[PASS] Loss values are finite"
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    trainable_tensors = 0
    gradient_tensors = 0
    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            trainable_tensors += 1

            if parameter.grad is not None:

                gradient_tensors += 1

                if not torch.isfinite(
                    parameter.grad
                ).all():

                    invalid_gradients += 1

    print(
        f"Trainable tensors: "
        f"{trainable_tensors}"
    )

    print(
        f"Tensors with gradients: "
        f"{gradient_tensors}"
    )

    assert (
        gradient_tensors
        == trainable_tensors
    ), "Some trainable tensors have no gradients"

    assert (
        invalid_gradients == 0
    ), "Invalid gradients detected"

    print(
        "[PASS] Gradients propagated"
    )

    print(
        "[PASS] Gradients are finite"
    )

    # --------------------------------------------------------
    # Verify weighting
    # --------------------------------------------------------

    expected_total = (
        epsilon_loss
        + 0.1 * x0_loss
    )

    difference = abs(
        total_loss.item()
        - expected_total.item()
    )

    print(
        f"\nExpected total loss: "
        f"{expected_total.item():.6f}"
    )

    print(
        f"Difference: "
        f"{difference:.10f}"
    )

    assert difference < 1e-6

    print(
        "[PASS] Loss weighting verified"
    )

    print("\n" + "=" * 60)
    print(
        "TASK 15.5.3 VERIFICATION COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()