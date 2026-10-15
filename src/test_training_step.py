from pathlib import Path

import torch

from src.diffusion.loss import diffusion_training_loss
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def main():

    print("=" * 60)
    print("TASK 6 — DIFFUSION TRAINING OBJECTIVE VERIFICATION")
    print("=" * 60)

    device = torch.device("cpu")

    print("\nDevice:")
    print(f"  {device}")

    # ---------------------------------------------------------
    # Diffusion schedule
    # ---------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    alpha_bars = alpha_bars.to(device)

    print("\nDiffusion schedule:")
    print(f"  Timesteps : {len(alpha_bars)}")
    print("  [PASS] Schedule loaded")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print("\nLoading training batch...")

    train_loader, _, _ = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    low_light, normal_light = next(iter(train_loader))

    low_light = low_light.to(device)
    normal_light = normal_light.to(device)

    print(f"  Low-light shape   : {low_light.shape}")
    print(f"  Normal-light shape: {normal_light.shape}")

    assert low_light.shape == normal_light.shape
    assert low_light.shape == (2, 3, 256, 256)

    print("  [PASS] Paired batch shape")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nCreating Conditional U-Net...")

    model = ConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=32,
        time_dim=128,
    ).to(device)

    print("  [PASS] Model created")

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    print("\nOptimizer:")
    print("  AdamW")
    print("  Learning rate: 0.0001")
    print("  [PASS] Optimizer created")

    # ---------------------------------------------------------
    # Initial parameter snapshot
    # ---------------------------------------------------------

    initial_parameter = next(
        model.parameters()
    ).detach().clone()

    # ---------------------------------------------------------
    # Training step
    # ---------------------------------------------------------

    print("\nRunning training step...")

    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss = diffusion_training_loss(
        model=model,
        low_light=low_light,
        normal_light=normal_light,
        alpha_bars=alpha_bars,
    )

    print(f"  Loss : {loss.item():.6f}")

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    print("  [PASS] Loss is finite and positive")

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    print("\nRunning backward pass...")

    loss.backward()

    gradients_found = False
    maximum_gradient = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients_found = True

            gradient_value = parameter.grad.abs().max().item()

            maximum_gradient = max(
                maximum_gradient,
                gradient_value,
            )

    print(f"  Maximum gradient: {maximum_gradient:.8f}")

    assert gradients_found
    assert torch.isfinite(
        torch.tensor(maximum_gradient)
    )

    print("  [PASS] Gradients exist")
    print("  [PASS] Gradients are finite")

    # ---------------------------------------------------------
    # Optimizer update
    # ---------------------------------------------------------

    print("\nRunning optimizer update...")

    optimizer.step()

    updated_parameter = next(
        model.parameters()
    ).detach().clone()

    parameter_changed = not torch.equal(
        initial_parameter,
        updated_parameter,
    )

    assert parameter_changed

    print("  [PASS] Model parameters updated")

    # ---------------------------------------------------------
    # Second training step
    # ---------------------------------------------------------

    print("\nRunning second training step...")

    optimizer.zero_grad(set_to_none=True)

    loss_2 = diffusion_training_loss(
        model=model,
        low_light=low_light,
        normal_light=normal_light,
        alpha_bars=alpha_bars,
    )

    print(f"  Loss : {loss_2.item():.6f}")

    assert torch.isfinite(loss_2)

    loss_2.backward()

    optimizer.step()

    print("  [PASS] Second training step completed")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TASK 6 DIFFUSION TRAINING OBJECTIVE VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()