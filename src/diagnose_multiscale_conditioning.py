from __future__ import annotations

from pathlib import Path

import torch

from src.configs import multiscale_training_config as config
from src.diffusion.forward import q_sample
from src.diffusion.schedule import get_diffusion_parameters
from src.models.multiscale_unet import MultiScaleConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


def mean_abs(tensor: torch.Tensor) -> float:
    return tensor.detach().abs().mean().item()


def main() -> None:
    print("=" * 70)
    print("MULTI-SCALE CONDITIONING ACTIVATION DIAGNOSTIC")
    print("=" * 70)

    device = torch.device(config.DEVICE)

    checkpoint_path = (
        config.CHECKPOINT_ROOT / "best_model.pt"
    )

    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = MultiScaleConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")

    # ---------------------------------------------------------
    # Load one validation batch
    # ---------------------------------------------------------
    _, val_loader, _ = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    low_light, target = next(iter(val_loader))

    low_light = low_light.to(device)
    target = target.to(device)

    print(f"Validation batch size: {low_light.shape[0]}")
    print(f"Low-light shape      : {tuple(low_light.shape)}")
    print(f"Target shape         : {tuple(target.shape)}")

    # ---------------------------------------------------------
    # Diffusion schedule
    # ---------------------------------------------------------
    _, _, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    alpha_bars = alpha_bars.to(device)

    # ---------------------------------------------------------
    # Select timestep
    # ---------------------------------------------------------
    timestep_value = 500

    timesteps = torch.full(
        (low_light.shape[0],),
        timestep_value,
        dtype=torch.long,
        device=device,
    )

    noise = torch.randn_like(target)

    # q_sample returns (noisy_image, noise)
    noisy_image, _ = q_sample(
        x_start=target,
        timesteps=timesteps,
        noise=noise,
        alpha_bars=alpha_bars,
    )

    print(f"Timestep              : {timestep_value}")
    print(f"Noisy image shape     : {tuple(noisy_image.shape)}")

    # ---------------------------------------------------------
    # Activation storage
    # ---------------------------------------------------------
    activations: dict[str, torch.Tensor] = {}

    hooks = []

    def save_activation(name: str):
        def hook(_module, _input, output):
            activations[name] = output.detach()

        return hook

    # U-Net feature activations
    hooks.append(
        model.input_conv.register_forward_hook(
            save_activation("unet_256")
        )
    )

    hooks.append(
        model.downsample1.register_forward_hook(
            save_activation("unet_128")
        )
    )

    hooks.append(
        model.downsample2.register_forward_hook(
            save_activation("unet_64")
        )
    )

    # Condition feature activations
    hooks.append(
        model.condition_conv_256.register_forward_hook(
            save_activation("condition_256")
        )
    )

    hooks.append(
        model.condition_projection_128.register_forward_hook(
            save_activation("condition_128")
        )
    )

    hooks.append(
        model.condition_projection_64.register_forward_hook(
            save_activation("condition_64")
        )
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    with torch.no_grad():
        prediction = model(
            noisy_image=noisy_image,
            condition=low_light,
            timesteps=timesteps,
        )

    # ---------------------------------------------------------
    # Remove hooks
    # ---------------------------------------------------------
    for hook in hooks:
        hook.remove()

    # ---------------------------------------------------------
    # Verify prediction
    # ---------------------------------------------------------
    print()
    print("-" * 70)
    print("MODEL OUTPUT")
    print("-" * 70)

    print(f"Prediction shape : {tuple(prediction.shape)}")
    print(f"Prediction finite: {torch.isfinite(prediction).all().item()}")
    print(f"Prediction MSE   : {torch.mean((prediction - noise) ** 2).item():.6f}")

    # ---------------------------------------------------------
    # Raw activation statistics
    # ---------------------------------------------------------
    print()
    print("-" * 70)
    print("RAW ACTIVATION MAGNITUDES")
    print("-" * 70)

    scale_names = [
        ("256", "unet_256", "condition_256"),
        ("128", "unet_128", "condition_128"),
        ("64", "unet_64", "condition_64"),
    ]

    results = {}

    for scale, unet_name, condition_name in scale_names:

        unet_feature = activations[unet_name]
        condition_feature = activations[condition_name]

        unet_mag = mean_abs(unet_feature)
        condition_mag = mean_abs(condition_feature)

        ratio = condition_mag / (unet_mag + 1e-12)

        results[scale] = {
            "unet_mag": unet_mag,
            "condition_mag": condition_mag,
            "ratio": ratio,
        }

        print()
        print(f"Scale {scale}x{scale}")
        print(f"  U-Net feature       : {tuple(unet_feature.shape)}")
        print(f"  Condition feature   : {tuple(condition_feature.shape)}")
        print(f"  U-Net mean |x|      : {unet_mag:.6f}")
        print(f"  Condition mean |x|  : {condition_mag:.6f}")
        print(f"  Condition/U-Net     : {ratio:.6f}")

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------
    print()
    print("-" * 70)
    print("CONDITIONING STRENGTH INTERPRETATION")
    print("-" * 70)

    for scale, values in results.items():

        ratio = values["ratio"]

        if ratio < 0.25:
            interpretation = "VERY WEAK"
        elif ratio < 0.50:
            interpretation = "WEAK"
        elif ratio < 1.00:
            interpretation = "MODERATE"
        elif ratio < 2.00:
            interpretation = "STRONG"
        else:
            interpretation = "VERY STRONG"

        print(
            f"Scale {scale}x{scale}: "
            f"ratio={ratio:.4f} -> {interpretation}"
        )

    # ---------------------------------------------------------
    # Combined feature statistics
    # ---------------------------------------------------------
    print()
    print("-" * 70)
    print("COMBINED FEATURE STATISTICS")
    print("-" * 70)

    # At 256 resolution
    combined_256 = (
        activations["unet_256"]
        + activations["condition_256"]
    )

    # At 128 resolution
    combined_128 = (
        activations["unet_128"]
        + activations["condition_128"]
    )

    # At 64 resolution
    combined_64 = (
        activations["unet_64"]
        + activations["condition_64"]
    )

    combined_features = [
        ("256", combined_256),
        ("128", combined_128),
        ("64", combined_64),
    ]

    for scale, feature in combined_features:

        print(
            f"Scale {scale}x{scale}: "
            f"shape={tuple(feature.shape)}, "
            f"mean |x|={mean_abs(feature):.6f}, "
            f"min={feature.min().item():.6f}, "
            f"max={feature.max().item():.6f}"
        )

    # ---------------------------------------------------------
    # Save diagnostic results
    # ---------------------------------------------------------
    output_dir = config.RESULTS_ROOT / "conditioning_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "conditioning_activation.txt"

    with output_file.open("w", encoding="utf-8") as file:

        file.write("MULTI-SCALE CONDITIONING ACTIVATION DIAGNOSTIC\n")
        file.write("=" * 70 + "\n\n")

        file.write(f"Device: {device}\n")
        file.write(f"Checkpoint: {checkpoint_path}\n")
        file.write(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n")
        file.write(f"Timestep: {timestep_value}\n\n")

        file.write("ACTIVATION MAGNITUDES\n")
        file.write("-" * 70 + "\n")

        for scale, values in results.items():

            file.write(
                f"Scale {scale}x{scale}\n"
                f"U-Net mean |x|     : {values['unet_mag']:.6f}\n"
                f"Condition mean |x| : {values['condition_mag']:.6f}\n"
                f"Condition/U-Net    : {values['ratio']:.6f}\n\n"
            )

    print()
    print("-" * 70)
    print(f"Diagnostic saved to:")
    print(output_file)
    print("-" * 70)

    print()
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
