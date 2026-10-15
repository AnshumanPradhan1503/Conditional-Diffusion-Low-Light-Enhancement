from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.configs import training_config as config
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders
from src.diffusion.schedule import get_diffusion_parameters
from src.diffusion.forward import q_sample


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
    / "timestep_weighting"
)

OUTPUT_PATH = RESULTS_DIR / "timestep_weighting.txt"

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

DEVICE = torch.device("cpu")

SEED = 42

BATCH_SIZE = 2

# More detailed timestep coverage than Task 22.1.
TIMESTEPS_TO_TEST = [
    0,
    25,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    750,
    800,
    900,
    950,
    999,
]


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


def reconstruct_x0(
    noisy_image: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:

    alpha_bar_t = (
        alpha_bars[timesteps]
        .reshape(-1, 1, 1, 1)
    )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar_t
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar_t
    )

    predicted_x0 = (
        noisy_image
        - sqrt_one_minus_alpha_bar * predicted_noise
    ) / sqrt_alpha_bar

    return predicted_x0


def main() -> None:

    print("=" * 70)
    print("TASK 22.3 - TIMESTEP-WEIGHTING DIAGNOSTIC")
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

    # ---------------------------------------------------------
    # Diffusion schedule
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

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("DETAILED TIMESTEP ANALYSIS")
    print("=" * 100)

    header = (
        f"{'t':>6} | "
        f"{'alpha_bar':>12} | "
        f"{'epsilon MSE':>15} | "
        f"{'x0 MSE':>15} | "
        f"{'x0 PSNR':>12}"
    )

    print(header)
    print("-" * 100)

    report_lines = [
        "TASK 22.3 - TIMESTEP-WEIGHTING DIAGNOSTIC",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Seed: {SEED}",
        "",
        "DETAILED TIMESTEP ANALYSIS",
        "",
        header,
        "-" * 100,
    ]

    results = []

    # ---------------------------------------------------------
    # Evaluate each timestep
    # ---------------------------------------------------------

    with torch.no_grad():

        for timestep_value in TIMESTEPS_TO_TEST:

            timesteps = torch.full(
                (low_light.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=DEVICE,
            )

            # Forward diffusion
            noisy_image, true_noise = q_sample(
                target,
                timesteps,
                alpha_bars,
            )

            # Predict noise
            predicted_noise = model(
                noisy_image,
                low_light,
                timesteps,
            )

            epsilon_mse = F.mse_loss(
                predicted_noise,
                true_noise,
            ).item()

            # Reconstruct x0
            predicted_x0 = reconstruct_x0(
                noisy_image,
                predicted_noise,
                timesteps,
                alpha_bars,
            )

            x0_mse = F.mse_loss(
                predicted_x0,
                target,
            ).item()

            x0_psnr = calculate_psnr(
                predicted_x0,
                target,
            )

            alpha_bar = alpha_bars[
                timestep_value
            ].item()

            print(
                f"{timestep_value:6d} | "
                f"{alpha_bar:12.8f} | "
                f"{epsilon_mse:15.8f} | "
                f"{x0_mse:15.8f} | "
                f"{x0_psnr:12.4f}"
            )

            report_lines.append(
                f"{timestep_value:6d} | "
                f"{alpha_bar:12.8f} | "
                f"{epsilon_mse:15.8f} | "
                f"{x0_mse:15.8f} | "
                f"{x0_psnr:12.4f}"
            )

            results.append(
                {
                    "t": timestep_value,
                    "alpha_bar": alpha_bar,
                    "epsilon_mse": epsilon_mse,
                    "x0_mse": x0_mse,
                    "x0_psnr": x0_psnr,
                }
            )

    # ---------------------------------------------------------
    # Aggregate timestep regions
    # ---------------------------------------------------------

    regions = {
        "Early (0-99)": [
            r for r in results
            if 0 <= r["t"] <= 99
        ],
        "Low-mid (100-249)": [
            r for r in results
            if 100 <= r["t"] <= 249
        ],
        "Mid (250-499)": [
            r for r in results
            if 250 <= r["t"] <= 499
        ],
        "High-mid (500-749)": [
            r for r in results
            if 500 <= r["t"] <= 749
        ],
        "High (750-999)": [
            r for r in results
            if 750 <= r["t"] <= 999
        ],
    }

    print("\n")
    print("=" * 100)
    print("TIMESTEP REGION SUMMARY")
    print("=" * 100)

    region_header = (
        f"{'Region':<22} | "
        f"{'Mean epsilon MSE':>18} | "
        f"{'Mean x0 MSE':>15} | "
        f"{'Mean x0 PSNR':>15}"
    )

    print(region_header)
    print("-" * 100)

    report_lines.extend(
        [
            "",
            "TIMESTEP REGION SUMMARY",
            "",
            region_header,
            "-" * 100,
        ]
    )

    for region_name, region_results in regions.items():

        if not region_results:
            continue

        mean_epsilon = sum(
            r["epsilon_mse"]
            for r in region_results
        ) / len(region_results)

        mean_x0_mse = sum(
            r["x0_mse"]
            for r in region_results
        ) / len(region_results)

        mean_x0_psnr = sum(
            r["x0_psnr"]
            for r in region_results
        ) / len(region_results)

        print(
            f"{region_name:<22} | "
            f"{mean_epsilon:18.8f} | "
            f"{mean_x0_mse:15.8f} | "
            f"{mean_x0_psnr:15.4f}"
        )

        report_lines.append(
            f"{region_name:<22} | "
            f"{mean_epsilon:18.8f} | "
            f"{mean_x0_mse:15.8f} | "
            f"{mean_x0_psnr:15.4f}"
        )

    # ---------------------------------------------------------
    # Find best/worst timesteps
    # ---------------------------------------------------------

    best_epsilon = min(
        results,
        key=lambda r: r["epsilon_mse"],
    )

    best_x0 = min(
        results,
        key=lambda r: r["x0_mse"],
    )

    best_psnr = max(
        results,
        key=lambda r: r["x0_psnr"],
    )

    worst_x0 = max(
        results,
        key=lambda r: r["x0_mse"],
    )

    print("\n")
    print("=" * 100)
    print("BEST / WORST TIMESTEPS")
    print("=" * 100)

    print(
        f"Best epsilon MSE : "
        f"t={best_epsilon['t']} "
        f"({best_epsilon['epsilon_mse']:.8f})"
    )

    print(
        f"Best x0 MSE      : "
        f"t={best_x0['t']} "
        f"({best_x0['x0_mse']:.8f})"
    )

    print(
        f"Best x0 PSNR     : "
        f"t={best_psnr['t']} "
        f"({best_psnr['x0_psnr']:.4f} dB)"
    )

    print(
        f"Worst x0 MSE     : "
        f"t={worst_x0['t']} "
        f"({worst_x0['x0_mse']:.8f})"
    )

    report_lines.extend(
        [
            "",
            "BEST / WORST TIMESTEPS",
            "",
            (
                f"Best epsilon MSE : "
                f"t={best_epsilon['t']} "
                f"({best_epsilon['epsilon_mse']:.8f})"
            ),
            (
                f"Best x0 MSE      : "
                f"t={best_x0['t']} "
                f"({best_x0['x0_mse']:.8f})"
            ),
            (
                f"Best x0 PSNR     : "
                f"t={best_psnr['t']} "
                f"({best_psnr['x0_psnr']:.4f} dB)"
            ),
            (
                f"Worst x0 MSE     : "
                f"t={worst_x0['t']} "
                f"({worst_x0['x0_mse']:.8f})"
            ),
        ]
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    all_finite = all(
        torch.isfinite(
            torch.tensor(
                [
                    r["epsilon_mse"],
                    r["x0_mse"],
                    r["x0_psnr"],
                ]
            )
        ).all().item()
        for r in results
    )

    print("\n")
    print("=" * 100)
    print("DIAGNOSTIC VALIDATION")
    print("=" * 100)

    print(
        f"[{'PASS' if all_finite else 'FAIL'}] "
        "All timestep metrics are finite"
    )

    report_lines.extend(
        [
            "",
            "DIAGNOSTIC VALIDATION",
            "",
            (
                f"[{'PASS' if all_finite else 'FAIL'}] "
                "All timestep metrics are finite"
            ),
        ]
    )

    # ---------------------------------------------------------
    # Save
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
    print("TASK 22.3 COMPLETE")
    print("=" * 70)

    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()