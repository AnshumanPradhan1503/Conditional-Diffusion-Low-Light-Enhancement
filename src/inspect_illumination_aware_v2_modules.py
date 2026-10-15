from __future__ import annotations

from pathlib import Path

import torch

from src.configs import illumination_aware_v2_training_config as config
from src.models.illumination_aware_unet_v2 import (
    IlluminationAwareConditionalUNet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "illumination_aware_v2"
    / "best_model.pt"
)


def main() -> None:
    print("=" * 80)
    print("Illumination-Aware V2 Module Inspection")
    print("=" * 80)

    device = torch.device(config.DEVICE)

    print(f"\nCheckpoint:")
    print(CHECKPOINT_FILE)

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_FILE}"
        )

    model = IlluminationAwareConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nCheckpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']}")
    print(
        f"Parameter count: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print("\n" + "=" * 80)
    print("ALL NAMED MODULES")
    print("=" * 80)

    for name, module in model.named_modules():
        if name == "":
            continue

        parameter_count = sum(
            parameter.numel()
            for parameter in module.parameters(recurse=False)
        )

        print(
            f"{name:<70} "
            f"{module.__class__.__name__:<35} "
            f"direct_params={parameter_count:,}"
        )

    print("\n" + "=" * 80)
    print("ILLUMINATION-RELATED MODULES")
    print("=" * 80)

    illumination_found = False

    for name, module in model.named_modules():

        name_lower = name.lower()

        if (
            "illum" in name_lower
            or "projection" in name_lower
            or "proj" in name_lower
        ):
            illumination_found = True

            parameter_count = sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
            )

            print(
                f"{name:<70} "
                f"{module.__class__.__name__:<35} "
                f"direct_params={parameter_count:,}"
            )

    if not illumination_found:
        print(
            "WARNING: No module names containing "
            "'illum', 'projection', or 'proj' were found."
        )

    print("\n" + "=" * 80)
    print("CONVOLUTION MODULES")
    print("=" * 80)

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            print(
                f"{name:<70} "
                f"Conv2d "
                f"in={module.in_channels} "
                f"out={module.out_channels} "
                f"kernel={module.kernel_size} "
                f"stride={module.stride}"
            )

    print("\n" + "=" * 80)
    print("Inspection completed.")
    print("=" * 80)

    print(
        "\nNo checkpoint was modified."
    )


if __name__ == "__main__":
    main()