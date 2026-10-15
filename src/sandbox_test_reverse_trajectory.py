from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.configs import training_config as config
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders
from src.diffusion.schedule import get_diffusion_parameters
from src.diffusion.reverse import p_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "sampling_diagnostic"
    / "reverse_trajectory"
)

OUTPUT_PATH = RESULTS_DIR / "reverse_trajectory.txt"

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

DEVICE = torch.device("cpu")

SEED = 42

CHECKPOINT_TIMESTEPS = [
    999,
    750,
    500,
    250,
    100,
    0,
]

BATCH_SIZE = 2


def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    prediction = (
        prediction.clamp(-1.0, 1.0) + 1.0
    ) / 2.0

    target = (
        target.clamp(-1.0, 1.0) + 1.0
    ) / 2.0

    mse = F.mse_loss(
        prediction,
        target,
    ).item()

    if mse == 0:
        return float("inf")

    return 10.0 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


def main() -> None:

    print("=" * 70)
    print("TASK 22.2 - REVERSE-DIFFUSION TRAJECTORY DIAGNOSTIC")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Seed       : {SEED}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    torch.manual_seed(SEED)

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

    low_light, target = next(
        iter(val_loader)
    )

    low_light = low_light.to(DEVICE)
    target = target.to(DEVICE)

    print(
        f"Validation batch size: "
        f"{low_light.shape[0]}"
    )

    print(
        f"Low-light shape      : "
        f"{tuple(low_light.shape)}"
    )

    print(
        f"Target shape         : "
        f"{tuple(target.shape)}"
    )

    # ---------------------------------------------------------
    # Load diffusion schedule
    # ---------------------------------------------------------

    betas, alphas, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    betas = betas.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alpha_bars = alpha_bars.to(DEVICE)

    # ---------------------------------------------------------
    # Load original control model
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(DEVICE)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict' or 'state_dict'."
        )

    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"Checkpoint epoch     : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Model parameters     : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ---------------------------------------------------------
    # Start from Gaussian noise
    # ---------------------------------------------------------

    generator = torch.Generator(
        device=DEVICE
    )

    generator.manual_seed(SEED)

    x = torch.randn(
        target.shape,
        generator=generator,
        device=DEVICE,
    )

    print(
        f"Initial noise shape   : "
        f"{tuple(x.shape)}"
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    report_lines = [
        "TASK 22.2 - REVERSE-DIFFUSION TRAJECTORY DIAGNOSTIC",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Seed: {SEED}",
        "",
        "Reverse trajectory starts from Gaussian noise at t=999.",
        "",
        (
            "Metrics below are calculated against the validation "
            "normal-light target."
        ),
        "",
    ]

    # ---------------------------------------------------------
    # Reverse diffusion
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("REVERSE TRAJECTORY")
    print("=" * 70)

    header = (
        f"{'t':>6} | "
        f"{'MSE':>15} | "
        f"{'PSNR':>12} | "
        f"{'mean':>12} | "
        f"{'std':>12}"
    )

    print(header)
    print("-" * len(header))

    report_lines.extend(
        [
            header,
            "-" * len(header),
        ]
    )

    with torch.no_grad():

        for timestep_value in reversed(
            range(config.TIMESTEPS)
        ):

            timestep = torch.full(
                (x.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=DEVICE,
            )

            # -------------------------------------------------
            # Model predicts noise
            # -------------------------------------------------

            predicted_noise = model(
                x,
                low_light,
                timestep,
            )

            # -------------------------------------------------
            # Perform one reverse-diffusion step
            # -------------------------------------------------

            x = p_sample(
                x_t=x,
                timesteps=timestep,
                predicted_noise=predicted_noise,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                generator=generator,
            )

            # -------------------------------------------------
            # Record selected stages
            # -------------------------------------------------

            if timestep_value in CHECKPOINT_TIMESTEPS:

                prediction_01 = (
                    x.clamp(-1.0, 1.0) + 1.0
                ) / 2.0

                target_01 = (
                    target.clamp(-1.0, 1.0) + 1.0
                ) / 2.0

                mse = F.mse_loss(
                    prediction_01,
                    target_01,
                ).item()

                psnr = calculate_psnr(
                    x,
                    target,
                )

                mean_value = x.mean().item()
                std_value = x.std().item()
                min_value = x.min().item()
                max_value = x.max().item()

                print(
                    f"{timestep_value:6d} | "
                    f"{mse:15.8f} | "
                    f"{psnr:12.4f} | "
                    f"{mean_value:12.6f} | "
                    f"{std_value:12.6f}"
                )

                report_lines.append(
                    f"{timestep_value:6d} | "
                    f"{mse:15.8f} | "
                    f"{psnr:12.4f} | "
                    f"{mean_value:12.6f} | "
                    f"{std_value:12.6f} | "
                    f"min={min_value:.6f} | "
                    f"max={max_value:.6f}"
                )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    prediction_01 = (
        x.clamp(-1.0, 1.0) + 1.0
    ) / 2.0

    target_01 = (
        target.clamp(-1.0, 1.0) + 1.0
    ) / 2.0

    final_mse = F.mse_loss(
        prediction_01,
        target_01,
    ).item()

    final_psnr = calculate_psnr(
        x,
        target,
    )

    output_finite = torch.isfinite(
        x
    ).all().item()

    print("\n")
    print("=" * 70)
    print("FINAL OUTPUT CHECK")
    print("=" * 70)

    print(
        f"Final MSE     : {final_mse:.8f}"
    )

    print(
        f"Final PSNR    : {final_psnr:.4f} dB"
    )

    print(
        f"Output finite : {output_finite}"
    )

    report_lines.extend(
        [
            "",
            "FINAL OUTPUT CHECK",
            "",
            f"Final MSE: {final_mse:.8f}",
            f"Final PSNR: {final_psnr:.4f} dB",
            f"Output finite: {output_finite}",
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

    print("\n")
    print("=" * 70)
    print("TASK 22.2 COMPLETE")
    print("=" * 70)

    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()