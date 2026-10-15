from __future__ import annotations

import torch

from src.configs import training_config as config
from src.diffusion.reverse import reverse_diffusion
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

    print("TASK 12 — ACTUAL IMAGE GENERATION")
    print("=" * 50)
    print(f"Device: {device}")

    output_dir = config.RESULTS_ROOT / "generated"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use the existing test split created during Phase 3.
    test_split = config.SPLIT_ROOT / "test_pairs.txt"

    dataset = LOLv2PairedDataset(
        split_file=test_split,
        data_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        mode="test",
    )

    print(f"Official test pairs: {len(dataset)}")

    # Generate only one image initially.
    index = 0

    low_tensor, high_tensor = dataset[index]

    low_tensor = low_tensor.unsqueeze(0).to(device)
    high_tensor = high_tensor.unsqueeze(0).to(device)

    print(f"Input shape: {low_tensor.shape}")

    model = load_model(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    def predicted_noise_fn(
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return model(
            noisy_image=x_t,
            condition=condition,
            timesteps=timesteps,
        )

    print()
    print("Starting 1000-step conditional reverse diffusion...")
    print("This may take several minutes on CPU.")
    print()

    with torch.no_grad():
        generated_tensor = reverse_diffusion(
            shape=low_tensor.shape,
            condition=low_tensor,
            predicted_noise_fn=predicted_noise_fn,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            device=device,
        )

    generated_tensor = generated_tensor.cpu().squeeze(0)
    low_tensor = low_tensor.cpu().squeeze(0)
    high_tensor = high_tensor.cpu().squeeze(0)

    generated_image = tensor_to_image(generated_tensor)
    low_image = tensor_to_image(low_tensor)
    high_image = tensor_to_image(high_tensor)

    low_path = output_dir / "low_000.png"
    generated_path = output_dir / "generated_000.png"
    high_path = output_dir / "high_000.png"

    low_image.save(low_path)
    generated_image.save(generated_path)
    high_image.save(high_path)

    print()
    print("Generation completed.")
    print(f"Low-light saved : {low_path}")
    print(f"Generated saved : {generated_path}")
    print(f"Reference saved : {high_path}")
    print()
    print("TASK 12 IMAGE GENERATION COMPLETED")


if __name__ == "__main__":
    main()