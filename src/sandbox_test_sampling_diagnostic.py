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
)

OUTPUT_PATH = RESULTS_DIR / "sampling_diagnostic.txt"

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

DEVICE = torch.device("cpu")

TIMESTEPS_TO_TEST = [0, 100, 250, 500, 750, 999]

BATCH_SIZE = 2

SEED = 42


def extract_alpha_bar(
    alpha_bars: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Extract alpha_bar[t] and reshape it for image broadcasting.
    """
    return alpha_bars[timesteps].reshape(
        -1, 1, 1, 1
    )


def reconstruct_x0_from_epsilon(
    noisy_image: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct x0 from x_t and predicted epsilon.
    """

    alpha_bar_t = extract_alpha_bar(
        alpha_bars,
        timesteps,
    )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar_t
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar_t
    )

    x0_pred = (
        noisy_image
        - sqrt_one_minus_alpha_bar * predicted_noise
    ) / sqrt_alpha_bar

    return x0_pred


def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Calculate PSNR after converting [-1, 1] to [0, 1].
    """

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
    print("TASK 22.1 - DIFFUSION OBJECTIVE & SAMPLING DIAGNOSTIC")
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

    print(
        f"Model parameters      : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ---------------------------------------------------------
    # Timestep diagnostics
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TIMESTEP DIAGNOSTICS")
    print("=" * 70)

    header = (
        f"{'t':>6} | "
        f"{'epsilon MSE':>15} | "
        f"{'x0 MSE':>15} | "
        f"{'x0 PSNR':>12}"
    )

    print(header)
    print("-" * len(header))

    report_lines = [
        "TASK 22.1 - DIFFUSION OBJECTIVE & SAMPLING DIAGNOSTIC",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Seed: {SEED}",
        "",
        "TIMESTEP DIAGNOSTICS",
        "",
        header,
        "-" * len(header),
    ]

    timestep_results = []

    with torch.no_grad():

        for timestep_value in TIMESTEPS_TO_TEST:

            timesteps = torch.full(
                (low_light.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=DEVICE,
            )

            # -------------------------------------------------
            # Generate forward-diffused target
            # -------------------------------------------------

            noisy_image, true_noise = q_sample(
                target,
                timesteps,
                alpha_bars,
            )

            # -------------------------------------------------
            # Model predicts epsilon
            # -------------------------------------------------

            predicted_noise = model(
                noisy_image,
                low_light,
                timesteps,
            )

            epsilon_mse = F.mse_loss(
                predicted_noise,
                true_noise,
            ).item()

            # -------------------------------------------------
            # Reconstruct x0
            # -------------------------------------------------

            predicted_x0 = (
                reconstruct_x0_from_epsilon(
                    noisy_image,
                    predicted_noise,
                    timesteps,
                    alpha_bars,
                )
            )

            x0_mse = F.mse_loss(
                predicted_x0,
                target,
            ).item()

            x0_psnr = calculate_psnr(
                predicted_x0,
                target,
            )

            print(
                f"{timestep_value:6d} | "
                f"{epsilon_mse:15.8f} | "
                f"{x0_mse:15.8f} | "
                f"{x0_psnr:12.4f}"
            )

            report_lines.append(
                f"{timestep_value:6d} | "
                f"{epsilon_mse:15.8f} | "
                f"{x0_mse:15.8f} | "
                f"{x0_psnr:12.4f}"
            )

            timestep_results.append(
                {
                    "t": timestep_value,
                    "epsilon_mse": epsilon_mse,
                    "x0_mse": x0_mse,
                    "x0_psnr": x0_psnr,
                }
            )

    # ---------------------------------------------------------
    # Alpha-bar diagnostics
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DIFFUSION SIGNAL LEVELS")
    print("=" * 70)

    print(
        f"{'t':>6} | "
        f"{'alpha_bar':>15} | "
        f"{'sqrt(alpha_bar)':>18} | "
        f"{'sqrt(1-alpha_bar)':>20}"
    )

    print("-" * 70)

    report_lines.extend(
        [
            "",
            "DIFFUSION SIGNAL LEVELS",
            "",
            (
                f"{'t':>6} | "
                f"{'alpha_bar':>15} | "
                f"{'sqrt(alpha_bar)':>18} | "
                f"{'sqrt(1-alpha_bar)':>20}"
            ),
            "-" * 70,
        ]
    )

    for timestep_value in TIMESTEPS_TO_TEST:

        alpha_bar = alpha_bars[
            timestep_value
        ].item()

        sqrt_alpha_bar = (
            alpha_bar ** 0.5
        )

        sqrt_one_minus_alpha_bar = (
            (1.0 - alpha_bar) ** 0.5
        )

        print(
            f"{timestep_value:6d} | "
            f"{alpha_bar:15.8f} | "
            f"{sqrt_alpha_bar:18.8f} | "
            f"{sqrt_one_minus_alpha_bar:20.8f}"
        )

        report_lines.append(
            f"{timestep_value:6d} | "
            f"{alpha_bar:15.8f} | "
            f"{sqrt_alpha_bar:18.8f} | "
            f"{sqrt_one_minus_alpha_bar:20.8f}"
        )

    # ---------------------------------------------------------
    # Numerical validity checks
    # ---------------------------------------------------------

    all_finite = True

    for result in timestep_results:

        values = [
            result["epsilon_mse"],
            result["x0_mse"],
            result["x0_psnr"],
        ]

        if not all(
            torch.isfinite(
                torch.tensor(value)
            ).item()
            for value in values
        ):
            all_finite = False

    print("\n")
    print("=" * 70)
    print("DIAGNOSTIC VALIDATION")
    print("=" * 70)

    print(
        f"[{'PASS' if all_finite else 'FAIL'}] "
        "All diagnostic metrics are finite"
    )

    report_lines.extend(
        [
            "",
            "DIAGNOSTIC VALIDATION",
            "",
            (
                f"[{'PASS' if all_finite else 'FAIL'}] "
                "All diagnostic metrics are finite"
            ),
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
    print("TASK 22.1 COMPLETE")
    print("=" * 70)

    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()