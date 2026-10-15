from __future__ import annotations

import torch

from src.configs import training_config as config
from src.diffusion.forward import q_sample
from src.diffusion.reverse import p_sample
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main() -> None:
    device = torch.device(config.DEVICE)

    print("TASK 12.5 — CONTROLLED FULL REVERSE DIFFUSION")
    print("=" * 60)
    print(f"Device: {device}")

    output_dir = config.RESULTS_ROOT / "controlled_reverse"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = LOLv2PairedDataset(
        split_file=config.SPLIT_ROOT / "test_pairs.txt",
        data_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        mode="test",
    )

    low_light, normal_light = dataset[0]

    low_light = low_light.unsqueeze(0).to(device)
    normal_light = normal_light.unsqueeze(0).to(device)

    print(f"Test pairs: {len(dataset)}")
    print(f"Input shape: {low_light.shape}")

    # ---------------------------------------------------------
    # Model
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
    # Save reference
    # ---------------------------------------------------------

    tensor_to_image(
        normal_light.cpu().squeeze(0)
    ).save(
        output_dir / "normal_reference.png"
    )

    tensor_to_image(
        low_light.cpu().squeeze(0)
    ).save(
        output_dir / "low_light.png"
    )

    # ---------------------------------------------------------
    # Create a KNOWN x_999 from the normal-light image
    # ---------------------------------------------------------

    torch.manual_seed(42)

    timestep = torch.tensor(
        [config.TIMESTEPS - 1],
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(normal_light)

    x_t, actual_noise = q_sample(
        x_start=normal_light,
        timesteps=timestep,
        alpha_bars=alpha_bars,
        noise=noise,
    )

    print()
    print("Created known x_999 from ground-truth image.")
    print(
        f"x_999 mean: {x_t.mean().item():.6f}"
    )
    print(
        f"x_999 std : {x_t.std().item():.6f}"
    )

    # Save starting point.
    tensor_to_image(
        x_t.cpu().squeeze(0)
    ).save(
        output_dir / "known_x999.png"
    )

    # ---------------------------------------------------------
    # Run COMPLETE reverse process
    # ---------------------------------------------------------

    print()
    print("Starting controlled 1000-step reverse diffusion...")
    print("Starting from known x_999.")
    print()

    current = x_t.clone()

    with torch.no_grad():

        for timestep_value in reversed(
            range(config.TIMESTEPS)
        ):

            t = torch.full(
                (1,),
                timestep_value,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = model(
                noisy_image=current,
                condition=low_light,
                timesteps=t,
            )

            current = p_sample(
                x_t=current,
                timesteps=t,
                predicted_noise=predicted_noise,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
            )

            # Progress every 100 steps.
            if (
                timestep_value % 100 == 0
                or timestep_value == 999
            ):
                print(
                    f"t={timestep_value:3d} | "
                    f"mean={current.mean().item():.5f} | "
                    f"std={current.std().item():.5f}"
                )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    current = current.clamp(-1.0, 1.0)

    tensor_to_image(
        current.cpu().squeeze(0)
    ).save(
        output_dir / "controlled_generated.png"
    )

    # ---------------------------------------------------------
    # Compare with actual normal-light target
    # ---------------------------------------------------------

    mse = torch.mean(
        (current - normal_light) ** 2
    ).item()

    print()
    print(
        f"Final reconstruction MSE: {mse:.6f}"
    )

    print()
    print("TASK 12.5 CONTROLLED REVERSE COMPLETED")
    print()
    print(f"Results saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()