from __future__ import annotations

import torch

from src.configs import training_config as config
from src.diffusion.forward import q_sample
from src.diffusion.reverse import predict_x_start_from_noise
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.normalization import tensor_to_image
from src.preprocessing.paired_dataset import LOLv2PairedDataset


def load_model(device: torch.device) -> ConditionalUNet:
    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint_path = config.CHECKPOINT_ROOT / "best_model.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main() -> None:
    device = torch.device(config.DEVICE)

    print("TASK 12.1 — DIFFUSION DIAGNOSTIC")
    print("=" * 55)
    print(f"Device: {device}")

    output_dir = config.RESULTS_ROOT / "diffusion_diagnostic"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load official test image
    # ---------------------------------------------------------

    test_split = config.SPLIT_ROOT / "test_pairs.txt"

    dataset = LOLv2PairedDataset(
        split_file=test_split,
        data_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        mode="test",
    )

    low_light, normal_light = dataset[0]

    low_light = low_light.unsqueeze(0).to(device)
    normal_light = normal_light.unsqueeze(0).to(device)

    print(f"Test pairs: {len(dataset)}")
    print(f"Low-light shape: {low_light.shape}")
    print(f"Normal-light shape: {normal_light.shape}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = load_model(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ---------------------------------------------------------
    # Diffusion schedule
    # ---------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    # ---------------------------------------------------------
    # Save original input/reference
    # ---------------------------------------------------------

    tensor_to_image(
        low_light.cpu().squeeze(0)
    ).save(output_dir / "low_light.png")

    tensor_to_image(
        normal_light.cpu().squeeze(0)
    ).save(output_dir / "normal_light.png")

    # ---------------------------------------------------------
    # Test several diffusion timesteps
    # ---------------------------------------------------------

    test_timesteps = [0, 100, 500, 999]

    print()
    print("Testing model at diffusion timesteps:")
    print(test_timesteps)
    print()

    for timestep_value in test_timesteps:

        t = torch.tensor(
            [timestep_value],
            device=device,
            dtype=torch.long,
        )

        # Use a fixed noise sample so comparisons are reproducible.
        torch.manual_seed(42)

        noise = torch.randn_like(normal_light)

        # Add forward diffusion noise to the known target.
        noisy_target, actual_noise = q_sample(
            x_start=normal_light,
            timesteps=t,
            alpha_bars=alpha_bars,
            noise=noise,
        )

        # Predict the noise using the trained conditional U-Net.
        with torch.no_grad():
            predicted_noise = model(
                noisy_image=noisy_target,
                condition=low_light,
                timesteps=t,
            )

        # -----------------------------------------------------
        # Noise prediction error
        # -----------------------------------------------------

        noise_mse = torch.mean(
            (predicted_noise - actual_noise) ** 2
        ).item()

        # -----------------------------------------------------
        # Reconstruct x0 from predicted noise
        # -----------------------------------------------------

        predicted_x0 = predict_x_start_from_noise(
            x_t=noisy_target,
            timesteps=t,
            predicted_noise=predicted_noise,
            alpha_bars=alpha_bars,
        )

        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

        x0_mse = torch.mean(
            (predicted_x0 - normal_light) ** 2
        ).item()

        print(
            f"t={timestep_value:3d} | "
            f"noise MSE={noise_mse:.6f} | "
            f"x0 MSE={x0_mse:.6f}"
        )

        # Save the noisy target.
        tensor_to_image(
            noisy_target.cpu().squeeze(0)
        ).save(
            output_dir / f"noisy_t{timestep_value:03d}.png"
        )

        # Save reconstructed x0.
        tensor_to_image(
            predicted_x0.cpu().squeeze(0)
        ).save(
            output_dir / f"predicted_x0_t{timestep_value:03d}.png"
        )

    print()
    print("TASK 12.1 DIAGNOSTIC COMPLETED")
    print()
    print(f"Results saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()