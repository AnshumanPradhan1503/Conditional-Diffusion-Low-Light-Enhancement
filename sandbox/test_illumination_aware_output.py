from __future__ import annotations

from pathlib import Path

import torch

from src.configs import training_config as config
from src.diffusion.reverse import reverse_diffusion
from src.diffusion.schedule import get_diffusion_parameters
from src.models.illumination_aware_unet import (
    IlluminationAwareConditionalUNet,
)
from src.preprocessing.dataloader import create_dataloaders
from src.preprocessing.illumination_proxy import (
    illumination_proxy_from_normalized_rgb,
)


SEED = 42
CHECKPOINT = (
    config.PROJECT_ROOT
    / "checkpoints"
    / "illumination_aware"
    / "best_model.pt"
)


def main() -> None:

    print("=" * 65)
    print("TASK 24.5A — ILLUMINATION-AWARE OUTPUT DIAGNOSTIC")
    print("=" * 65)

    device = torch.device(config.DEVICE)

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = IlluminationAwareConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    betas, alphas, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    test_dataset = test_loader.dataset

    # Use several representative images.
    indices = [0, 10, 25, 50, 75, 99]

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    for index in indices:

        print("\n" + "-" * 65)
        print(
            f"IMAGE {index}: "
            f"{Path(test_dataset.pairs[index][0]).name}"
        )
        print("-" * 65)

        low_light, normal_light = (
            test_dataset[index]
        )

        low_light = (
            low_light
            .unsqueeze(0)
            .to(device)
        )

        normal_light = (
            normal_light
            .unsqueeze(0)
            .to(device)
        )

        illumination = (
            illumination_proxy_from_normalized_rgb(
                low_light
            )
        )

        illumination = illumination.to(
            device=device,
            dtype=low_light.dtype,
        )

        print(
            "Low-light statistics:"
        )

        print(
            f"  mean = {low_light.mean().item():.6f}"
        )

        print(
            f"  std  = {low_light.std().item():.6f}"
        )

        print(
            f"  min  = {low_light.min().item():.6f}"
        )

        print(
            f"  max  = {low_light.max().item():.6f}"
        )

        print(
            "\nIllumination proxy statistics:"
        )

        print(
            f"  mean = {illumination.mean().item():.6f}"
        )

        print(
            f"  std  = {illumination.std().item():.6f}"
        )

        print(
            f"  min  = {illumination.min().item():.6f}"
        )

        print(
            f"  max  = {illumination.max().item():.6f}"
        )

        # ----------------------------------------------------
        # Fixed timestep predictions
        # ----------------------------------------------------

        torch.manual_seed(
            SEED + index
        )

        noise = torch.randn_like(
            normal_light
        )

        for timestep_value in [
            0,
            100,
            250,
            500,
            750,
            999,
        ]:

            timestep = torch.tensor(
                [timestep_value],
                dtype=torch.long,
                device=device,
            )

            alpha_bar = (
                alpha_bars[timestep]
                .view(-1, 1, 1, 1)
            )

            noisy_target = (
                torch.sqrt(alpha_bar)
                * normal_light
                +
                torch.sqrt(
                    1.0 - alpha_bar
                )
                * noise
            )

            with torch.no_grad():

                predicted_noise = model(
                    noisy_image=noisy_target,
                    condition=low_light,
                    timesteps=timestep,
                    illumination=illumination,
                )

            mse = torch.mean(
                (
                    predicted_noise
                    - noise
                ) ** 2
            ).item()

            print(
                f"\nt={timestep_value}:"
            )

            print(
                f"  predicted-noise MSE = {mse:.6f}"
            )

            print(
                f"  prediction mean     = "
                f"{predicted_noise.mean().item():.6f}"
            )

            print(
                f"  prediction std      = "
                f"{predicted_noise.std().item():.6f}"
            )

            print(
                f"  prediction min      = "
                f"{predicted_noise.min().item():.6f}"
            )

            print(
                f"  prediction max      = "
                f"{predicted_noise.max().item():.6f}"
            )

        # ----------------------------------------------------
        # Full reverse diffusion
        # ----------------------------------------------------

        generator = (
            torch.Generator(
                device=device
            ).manual_seed(
                SEED
            )
        )

        def predicted_noise_fn(
            x_t,
            condition,
            timesteps,
        ):

            return model(
                noisy_image=x_t,
                condition=condition,
                timesteps=timesteps,
                illumination=illumination,
            )

        with torch.no_grad():

            generated = reverse_diffusion(
                shape=low_light.shape,
                condition=low_light,
                predicted_noise_fn=predicted_noise_fn,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                device=device,
                generator=generator,
            )

        print(
            "\nFinal generated image statistics:"
        )

        print(
            f"  mean = {generated.mean().item():.6f}"
        )

        print(
            f"  std  = {generated.std().item():.6f}"
        )

        print(
            f"  min  = {generated.min().item():.6f}"
        )

        print(
            f"  max  = {generated.max().item():.6f}"
        )

        print(
            "\nReference statistics:"
        )

        print(
            f"  mean = {normal_light.mean().item():.6f}"
        )

        print(
            f"  std  = {normal_light.std().item():.6f}"
        )

        print(
            f"  min  = {normal_light.min().item():.6f}"
        )

        print(
            f"  max  = {normal_light.max().item():.6f}"
        )

    print("\n" + "=" * 65)
    print("TASK 24.5A COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()