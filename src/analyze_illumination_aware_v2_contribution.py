from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.configs import illumination_aware_v2_training_config as config
from src.models.illumination_aware_unet_v2 import (
    IlluminationAwareConditionalUNet,
)
from src.preprocessing.paired_dataset import LOLv2PairedDataset


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "illumination_aware_v2"
    / "best_model.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "illumination_aware_v2_diagnostics"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_ROOT
    / "illumination_contribution_analysis.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_ROOT
    / "illumination_contribution_summary.txt"
)


# ============================================================
# Diagnostic configuration
# ============================================================

# We do NOT evaluate every diffusion timestep.
# These represent early, middle, and late parts of the
# diffusion process.
DIAGNOSTIC_TIMESTEPS = [
    0,
    100,
    250,
    500,
    750,
    999,
]

# Analyze the complete official test set.
MAX_TEST_IMAGES = None


# ============================================================
# Illumination proxy
# ============================================================

def create_illumination_proxy(
    low_light: torch.Tensor,
) -> torch.Tensor:
    """
    Recreate the V2 low-frequency illumination proxy.

    Input:
        [B, 3, H, W] in [-1, 1]

    Output:
        [B, 1, H, W]

    Steps:
        1. Convert RGB to luminance.
        2. Convert [-1, 1] -> [0, 1].
        3. Apply 15x15 Gaussian smoothing with sigma=3.
        4. Use reflective padding.
    """

    # [-1, 1] -> [0, 1]
    rgb = (low_light + 1.0) / 2.0

    red = rgb[:, 0:1]
    green = rgb[:, 1:2]
    blue = rgb[:, 2:3]

    luminance = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    # --------------------------------------------------------
    # 15 x 15 Gaussian kernel, sigma = 3
    # --------------------------------------------------------

    kernel_size = 15
    sigma = 3.0

    radius = kernel_size // 2

    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=low_light.device,
        dtype=low_light.dtype,
    )

    gaussian_1d = torch.exp(
        -(coordinates ** 2)
        / (2.0 * sigma ** 2)
    )

    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = (
        gaussian_1d[:, None]
        * gaussian_1d[None, :]
    )

    kernel = gaussian_2d.view(
        1,
        1,
        kernel_size,
        kernel_size,
    )

    padded = F.pad(
        luminance,
        (
            radius,
            radius,
            radius,
            radius,
        ),
        mode="reflect",
    )

    proxy = F.conv2d(
        padded,
        kernel,
    )

    return proxy


# ============================================================
# RMS helper
# ============================================================

def rms(tensor: torch.Tensor) -> float:
    """
    Root-mean-square magnitude.
    """

    return float(
        torch.sqrt(
            torch.mean(
                tensor.detach() ** 2
            )
        ).item()
    )


def mean_absolute(tensor: torch.Tensor) -> float:
    """
    Mean absolute activation magnitude.
    """

    return float(
        torch.mean(
            torch.abs(
                tensor.detach()
            )
        ).item()
    )


# ============================================================
# Main diagnostic
# ============================================================

def main() -> None:

    print("=" * 80)
    print("Illumination-Aware V2 Contribution Analysis")
    print("=" * 80)

    device = torch.device(config.DEVICE)

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_FILE}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']}"
    )

    print(
        f"Parameter count: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = LOLv2PairedDataset(
        data_root=config.DATA_ROOT,
        split_file=config.SPLIT_ROOT / "test_pairs.txt",
        mode="test",
        image_size=config.IMAGE_SIZE,
    )

    total_images = len(test_dataset)

    if MAX_TEST_IMAGES is None:
        num_images = total_images
    else:
        num_images = min(
            MAX_TEST_IMAGES,
            total_images,
        )

    print(
        f"\nOfficial test images available: "
        f"{total_images}"
    )

    print(
        f"Images analyzed: "
        f"{num_images}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    records: list[dict[str, float | int | str]] = []

    # --------------------------------------------------------
    # Analyze images
    # --------------------------------------------------------

    with torch.no_grad():

        for image_index in range(num_images):

            low_light, normal_light = test_dataset[
                image_index
            ]

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

            # ------------------------------------------------
            # Illumination proxy
            # ------------------------------------------------

            illumination = (
                create_illumination_proxy(
                    low_light
                )
            )

            low_name = Path(
                test_dataset.pairs[image_index][0]
            ).name

            # ------------------------------------------------
            # Test every selected timestep
            # ------------------------------------------------

            for timestep_value in DIAGNOSTIC_TIMESTEPS:

                # ------------------------------------------------
                # Capture tensors
                # ------------------------------------------------

                captured: dict[str, torch.Tensor] = {}

                def make_projection_hook(
                    name: str,
                ):
                    def hook(
                        module,
                        inputs,
                        output,
                    ):
                        captured[
                            f"projection_{name}"
                        ] = output.detach().clone()

                    return hook

                def make_block_pre_hook(
                    name: str,
                ):
                    def hook(
                        module,
                        inputs,
                    ):
                        if inputs:
                            captured[
                                f"block_input_{name}"
                            ] = (
                                inputs[0]
                                .detach()
                                .clone()
                            )

                    return hook

                hooks = []

                # Illumination projections.
                hooks.append(
                    model.illumination_encoder.projection_256.register_forward_hook(
                        make_projection_hook("256")
                    )
                )

                hooks.append(
                    model.illumination_encoder.projection_128.register_forward_hook(
                        make_projection_hook("128")
                    )
                )

                hooks.append(
                    model.illumination_encoder.projection_64.register_forward_hook(
                        make_projection_hook("64")
                    )
                )

                # Inputs entering the actual backbone stages.
                hooks.append(
                    model.down_block1.register_forward_pre_hook(
                        make_block_pre_hook("256")
                    )
                )

                hooks.append(
                    model.down_block2.register_forward_pre_hook(
                        make_block_pre_hook("128")
                    )
                )

                hooks.append(
                    model.middle_block1.register_forward_pre_hook(
                        make_block_pre_hook("64")
                    )
                )

                # ------------------------------------------------
                # Timestep
                # ------------------------------------------------

                timestep = torch.tensor(
                    [timestep_value],
                    device=device,
                    dtype=torch.long,
                )

                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                noisy_target = torch.randn_like(
                    normal_light
                )

                model(
                    noisy_image=noisy_target,
                    condition=low_light,
                    timesteps=timestep,
                    illumination=illumination,
                )

                # ------------------------------------------------
                # Remove hooks immediately
                # ------------------------------------------------

                for hook in hooks:
                    hook.remove()

                # ------------------------------------------------
                # Verify captures
                # ------------------------------------------------

                required = [
                    "projection_256",
                    "projection_128",
                    "projection_64",
                    "block_input_256",
                    "block_input_128",
                    "block_input_64",
                ]

                missing = [
                    name
                    for name in required
                    if name not in captured
                ]

                if missing:
                    raise RuntimeError(
                        "Required activation was not captured: "
                        f"{missing}"
                    )

                # ------------------------------------------------
                # Record statistics
                # ------------------------------------------------

                row: dict[
                    str,
                    float | int | str
                ] = {
                    "image_index": image_index,
                    "image": low_name,
                    "timestep": timestep_value,
                }

                for resolution in [
                    "256",
                    "128",
                    "64",
                ]:

                    illumination_tensor = captured[
                        f"projection_{resolution}"
                    ]

                    block_tensor = captured[
                        f"block_input_{resolution}"
                    ]

                    illumination_rms = rms(
                        illumination_tensor
                    )

                    block_rms = rms(
                        block_tensor
                    )

                    illumination_abs = mean_absolute(
                        illumination_tensor
                    )

                    block_abs = mean_absolute(
                        block_tensor
                    )

                    ratio = (
                        illumination_rms
                        / (block_rms + 1e-12)
                    )

                    row[
                        f"illumination_rms_{resolution}"
                    ] = illumination_rms

                    row[
                        f"illumination_abs_{resolution}"
                    ] = illumination_abs

                    row[
                        f"block_input_rms_{resolution}"
                    ] = block_rms

                    row[
                        f"block_input_abs_{resolution}"
                    ] = block_abs

                    row[
                        f"illumination_to_block_rms_ratio_{resolution}"
                    ] = ratio

                records.append(row)

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (
                (image_index + 1) % 10 == 0
                or image_index == 0
                or image_index == num_images - 1
            ):
                print(
                    f"Processed "
                    f"{image_index + 1}/{num_images} "
                    f"test images"
                )

    # ========================================================
    # DataFrame
    # ========================================================

    results = pd.DataFrame(records)

    if results.empty:
        raise RuntimeError(
            "No diagnostic records were generated."
        )

    # ========================================================
    # Save CSV
    # ========================================================

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nSaved detailed contribution analysis:\n"
        f"{OUTPUT_CSV}"
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_lines = []

    summary_lines.append(
        "Illumination-Aware V2 Contribution Analysis"
    )

    summary_lines.append(
        "=" * 80
    )

    summary_lines.append("")

    summary_lines.append(
        "Checkpoint:"
    )

    summary_lines.append(
        str(CHECKPOINT_FILE)
    )

    summary_lines.append("")

    summary_lines.append(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    summary_lines.append(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']}"
    )

    summary_lines.append(
        f"Images analyzed: {num_images}"
    )

    summary_lines.append(
        f"Timesteps analyzed: "
        f"{DIAGNOSTIC_TIMESTEPS}"
    )

    summary_lines.append("")

    # --------------------------------------------------------
    # Aggregate by resolution
    # --------------------------------------------------------

    summary_lines.append(
        "Aggregate illumination contribution"
    )

    summary_lines.append(
        "-" * 80
    )

    for resolution in [
        "256",
        "128",
        "64",
    ]:

        rms_column = (
            f"illumination_rms_{resolution}"
        )

        block_column = (
            f"block_input_rms_{resolution}"
        )

        ratio_column = (
            "illumination_to_block_rms_ratio_"
            f"{resolution}"
        )

        summary_lines.append("")

        summary_lines.append(
            f"{resolution} resolution"
        )

        summary_lines.append(
            f"Mean illumination RMS: "
            f"{results[rms_column].mean():.8f}"
        )

        summary_lines.append(
            f"Mean backbone-stage input RMS: "
            f"{results[block_column].mean():.8f}"
        )

        summary_lines.append(
            f"Mean illumination/backbone RMS ratio: "
            f"{results[ratio_column].mean():.8f}"
        )

        summary_lines.append(
            f"Median illumination/backbone RMS ratio: "
            f"{results[ratio_column].median():.8f}"
        )

        summary_lines.append(
            f"Maximum illumination/backbone RMS ratio: "
            f"{results[ratio_column].max():.8f}"
        )

    # --------------------------------------------------------
    # Aggregate by timestep
    # --------------------------------------------------------

    summary_lines.append("")
    summary_lines.append(
        "Contribution by timestep"
    )
    summary_lines.append(
        "-" * 80
    )

    timestep_group = results.groupby(
        "timestep"
    )

    for timestep_value, group in timestep_group:

        summary_lines.append("")

        summary_lines.append(
            f"Timestep {int(timestep_value)}"
        )

        for resolution in [
            "256",
            "128",
            "64",
        ]:

            column = (
                "illumination_to_block_rms_ratio_"
                f"{resolution}"
            )

            summary_lines.append(
                f"  {resolution}: "
                f"{group[column].mean():.8f}"
            )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(
        f"\nSaved contribution summary:\n"
        f"{OUTPUT_SUMMARY}"
    )

    print("\n" + "=" * 80)
    print("Task 24.8.4 contribution analysis completed.")
    print("=" * 80)

    print(
        "\nNo checkpoint was modified."
    )

    print(
        "No model was retrained."
    )

    print(
        "No diffusion sampling/evaluation was performed."
    )


if __name__ == "__main__":
    main()