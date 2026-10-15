from pathlib import Path

import torch

from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def main():

    print("=" * 60)
    print("TASK 5 — CONDITIONAL U-NET VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device("cpu")

    print("\nDevice:")
    print(f"  {device}")

    # ---------------------------------------------------------
    # Load real dataset batch
    # ---------------------------------------------------------

    print("\nLoading real dataset batch...")

    _, _, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    low_light, normal_light = next(iter(test_loader))

    low_light = low_light.to(device)
    normal_light = normal_light.to(device)

    print(f"  Low-light shape  : {low_light.shape}")
    print(f"  Normal-light shape: {normal_light.shape}")

    # ---------------------------------------------------------
    # Create diffusion timestep
    # ---------------------------------------------------------

    timesteps = torch.tensor(
        [100, 500],
        dtype=torch.long,
        device=device,
    )

    print("\nDiffusion timesteps:")
    print(f"  {timesteps.tolist()}")

    # ---------------------------------------------------------
    # Create synthetic noisy target
    # ---------------------------------------------------------

    noise = torch.randn_like(normal_light)

    alpha_bar = torch.tensor(
        [0.8970, 0.0778],
        dtype=torch.float32,
        device=device,
    )

    alpha_bar = alpha_bar.reshape(
        2, 1, 1, 1
    )

    noisy_target = (
        torch.sqrt(alpha_bar) * normal_light
        + torch.sqrt(1.0 - alpha_bar) * noise
    )

    print("\nNoisy target:")
    print(f"  Shape : {noisy_target.shape}")
    print(f"  Min   : {noisy_target.min().item():.4f}")
    print(f"  Max   : {noisy_target.max().item():.4f}")

    # ---------------------------------------------------------
    # Create model
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
    # Parameter count
    # ---------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel parameters:")
    print(f"  Total     : {total_parameters:,}")
    print(f"  Trainable : {trainable_parameters:,}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    print("\nRunning forward pass...")

    model.eval()

    with torch.no_grad():

        predicted_noise = model(
            noisy_image=noisy_target,
            condition=low_light,
            timesteps=timesteps,
        )

    print("\nPredicted noise:")
    print(f"  Shape : {predicted_noise.shape}")
    print(f"  Dtype : {predicted_noise.dtype}")
    print(f"  Min   : {predicted_noise.min().item():.4f}")
    print(f"  Max   : {predicted_noise.max().item():.4f}")
    print(f"  Mean  : {predicted_noise.mean().item():.4f}")
    print(f"  Std   : {predicted_noise.std().item():.4f}")

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    print("\nVerification:")

    assert predicted_noise.shape == normal_light.shape
    print("  [PASS] Output shape matches target")

    assert predicted_noise.dtype == torch.float32
    print("  [PASS] Output dtype is float32")

    assert torch.isfinite(predicted_noise).all()
    print("  [PASS] No NaN/Inf")

    assert predicted_noise.shape[1] == 3
    print("  [PASS] Output has 3 channels")

    # ---------------------------------------------------------
    # Gradient verification
    # ---------------------------------------------------------

    print("\nGradient verification...")

    model.train()

    predicted_noise = model(
        noisy_image=noisy_target,
        condition=low_light,
        timesteps=timesteps,
    )

    loss = torch.mean(
        (predicted_noise - noise) ** 2
    )

    print(f"  Test MSE loss : {loss.item():.6f}")

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist

    print("  [PASS] Gradients successfully propagated")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TASK 5 CONDITIONAL U-NET VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()