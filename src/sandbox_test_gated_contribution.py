from __future__ import annotations

from pathlib import Path

import torch

from src.models.multiscale_gated_unet import MultiScaleGatedConditionalUNet
from src.preprocessing.dataloader import create_dataloaders
from src.diffusion.schedule import get_diffusion_parameters
from src.diffusion.forward import q_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "multiscale_gated"
    / "best_model.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "multiscale_gated"
    / "gate_diagnostic"
)

OUTPUT_PATH = RESULTS_DIR / "gated_contribution.txt"

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

DEVICE = torch.device("cpu")
TIMESTEP = 500
BATCH_SIZE = 2


def mean_abs(x: torch.Tensor) -> float:
    return x.detach().abs().mean().item()


def main() -> None:
    print("=" * 70)
    print("TASK 21.8 - ACTUAL GATED CONDITIONING CONTRIBUTION")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Timestep   : {TIMESTEP}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    _, val_loader, _ = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    low_light, target = next(iter(val_loader))

    low_light = low_light.to(DEVICE)
    target = target.to(DEVICE)

    timesteps = torch.full(
        (low_light.shape[0],),
        TIMESTEP,
        dtype=torch.long,
        device=DEVICE,
    )

    print(f"Validation batch size: {low_light.shape[0]}")
    print(f"Low-light shape      : {tuple(low_light.shape)}")
    print(f"Target shape         : {tuple(target.shape)}")

    # ---------------------------------------------------------
    # Diffusion schedule
    # ---------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    # Make sure schedule tensors are on the same device.
    betas = betas.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alpha_bars = alpha_bars.to(DEVICE)

    noisy_image, _ = q_sample(
        target,
        timesteps,
        alpha_bars,
    )

    print(f"Noisy image shape     : {tuple(noisy_image.shape)}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MultiScaleGatedConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=32,
        time_dim=128,
    ).to(DEVICE)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            "Checkpoint does not contain model_state_dict or state_dict."
        )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")

    # ---------------------------------------------------------
    # Reproduce condition branch and U-Net stream
    # ---------------------------------------------------------

    with torch.no_grad():

        # -----------------------------------------------------
        # Multi-scale condition features
        # -----------------------------------------------------

        condition_256 = model.condition_conv_256(
            low_light
        )

        condition_128 = model.condition_down_128(
            condition_256
        )

        condition_64 = model.condition_down_64(
            condition_128
        )

        condition_128 = model.condition_projection_128(
            condition_128
        )

        condition_64 = model.condition_projection_64(
            condition_64
        )

        # -----------------------------------------------------
        # Timestep embedding
        # -----------------------------------------------------

        time_embedding = model.time_embedding(
            timesteps
        )

        # -----------------------------------------------------
        # Noisy target input
        # -----------------------------------------------------

        x = model.input_conv(
            noisy_image
        )

        # -----------------------------------------------------
        # 256x256 conditioning
        # -----------------------------------------------------

        unet_256 = x

        gated_256 = model.condition_gate_256(
            condition_256
        )

        combined_256 = (
            unet_256
            + gated_256
        )

        # -----------------------------------------------------
        # Encoder level 1
        # -----------------------------------------------------

        skip1 = model.down_block1(
            combined_256,
            time_embedding,
        )

        x = model.downsample1(
            skip1
        )

        # -----------------------------------------------------
        # 128x128 conditioning
        # -----------------------------------------------------

        unet_128 = x

        gated_128 = model.condition_gate_128(
            condition_128
        )

        combined_128 = (
            unet_128
            + gated_128
        )

        # -----------------------------------------------------
        # Encoder level 2
        # -----------------------------------------------------

        skip2 = model.down_block2(
            combined_128,
            time_embedding,
        )

        x = model.downsample2(
            skip2
        )

        # -----------------------------------------------------
        # 64x64 conditioning
        # -----------------------------------------------------

        unet_64 = x

        gated_64 = model.condition_gate_64(
            condition_64
        )

        combined_64 = (
            unet_64
            + gated_64
        )

        # -----------------------------------------------------
        # Complete forward pass
        # -----------------------------------------------------

        x = model.middle_block1(
            combined_64,
            time_embedding,
        )

        x = model.middle_block2(
            x,
            time_embedding,
        )

        x = model.upsample1(
            x
        )

        x = torch.cat(
            [
                x,
                skip2,
            ],
            dim=1,
        )

        x = model.up_block1(
            x,
            time_embedding,
        )

        x = model.upsample2(
            x
        )

        x = torch.cat(
            [
                x,
                skip1,
            ],
            dim=1,
        )

        x = model.up_block2(
            x,
            time_embedding,
        )

        x = model.output_norm(
            x
        )

        x = model.output_activation(
            x
        )

        prediction = model.output_conv(
            x
        )

    # ---------------------------------------------------------
    # Calculate statistics
    # ---------------------------------------------------------

    scales = [
        (
            "256x256",
            unet_256,
            condition_256,
            gated_256,
            model.condition_gate_256.gate.item(),
        ),
        (
            "128x128",
            unet_128,
            condition_128,
            gated_128,
            model.condition_gate_128.gate.item(),
        ),
        (
            "64x64",
            unet_64,
            condition_64,
            gated_64,
            model.condition_gate_64.gate.item(),
        ),
    ]

    print("\nGATED CONDITIONING CONTRIBUTION")
    print("-" * 70)

    report_lines = [
        "TASK 21.8 - ACTUAL GATED CONDITIONING CONTRIBUTION",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Timestep: {TIMESTEP}",
        "",
    ]

    for (
        scale,
        unet_feature,
        condition_feature,
        gated_feature,
        gate_value,
    ) in scales:

        unet_mag = mean_abs(
            unet_feature
        )

        condition_mag = mean_abs(
            condition_feature
        )

        gated_mag = mean_abs(
            gated_feature
        )

        ratio = (
            gated_mag / unet_mag
            if unet_mag != 0
            else float("inf")
        )

        print(f"\nScale: {scale}")
        print(
            f"  U-Net magnitude        : "
            f"{unet_mag:.8f}"
        )
        print(
            f"  Condition magnitude   : "
            f"{condition_mag:.8f}"
        )
        print(
            f"  Gate value             : "
            f"{gate_value:.8f}"
        )
        print(
            f"  Gated contribution     : "
            f"{gated_mag:.8f}"
        )
        print(
            f"  Gated/U-Net ratio      : "
            f"{ratio:.8f}"
        )

        report_lines.extend(
            [
                f"Scale: {scale}",
                f"  U-Net magnitude      : {unet_mag:.8f}",
                f"  Condition magnitude : {condition_mag:.8f}",
                f"  Gate value           : {gate_value:.8f}",
                f"  Gated contribution  : {gated_mag:.8f}",
                f"  Gated/U-Net ratio   : {ratio:.8f}",
                "",
            ]
        )

    # ---------------------------------------------------------
    # Final numerical check
    # ---------------------------------------------------------

    prediction_finite = torch.isfinite(
        prediction
    ).all().item()

    print("\nMODEL OUTPUT CHECK")
    print("-" * 70)
    print(
        f"Prediction shape : "
        f"{tuple(prediction.shape)}"
    )
    print(
        f"Prediction finite: "
        f"{prediction_finite}"
    )

    report_lines.extend(
        [
            "MODEL OUTPUT CHECK",
            f"Prediction shape : {tuple(prediction.shape)}",
            f"Prediction finite: {prediction_finite}",
        ]
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()