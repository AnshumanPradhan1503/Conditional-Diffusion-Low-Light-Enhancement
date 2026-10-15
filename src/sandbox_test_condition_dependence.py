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
    / "condition_dependence"
)

OUTPUT_PATH = RESULTS_DIR / "condition_dependence.txt"

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

DEVICE = torch.device("cpu")

SEED = 42

BATCH_SIZE = 4

# Use several timesteps rather than only one.
TIMESTEPS_TO_TEST = [
    100,
    250,
    500,
    750,
]


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

    return (
        10.0
        * torch.log10(
            torch.tensor(1.0 / mse)
        ).item()
    )


def main() -> None:

    print("=" * 75)
    print("TASK 22.4 - CONDITIONING DEPENDENCE DIAGNOSTIC")
    print("=" * 75)

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

    alpha_bars = alpha_bars.to(DEVICE)

    # ---------------------------------------------------------
    # Load model
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

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    print(
        f"Checkpoint epoch     : "
        f"{checkpoint_epoch}"
    )

    # ---------------------------------------------------------
    # Create alternative conditions
    # ---------------------------------------------------------

    # Correct condition.
    correct_condition = low_light.clone()

    # Completely remove conditioning.
    zero_condition = torch.zeros_like(
        low_light
    )

    # Replace each image's condition with
    # another image's condition.
    #
    # For a batch of 4:
    # image 0 <- condition from image 1
    # image 1 <- condition from image 2
    # image 2 <- condition from image 3
    # image 3 <- condition from image 0
    shuffled_condition = torch.roll(
        low_light,
        shifts=1,
        dims=0,
    )

    # ---------------------------------------------------------
    # Report header
    # ---------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("CONDITIONING DEPENDENCE")
    print("=" * 110)

    header = (
        f"{'t':>5} | "
        f"{'Condition':<12} | "
        f"{'Epsilon MSE':>15} | "
        f"{'Prediction Δ':>15} | "
        f"{'x0 MSE':>15} | "
        f"{'x0 PSNR':>12}"
    )

    print(header)
    print("-" * 110)

    report_lines = [
        "TASK 22.4 - CONDITIONING DEPENDENCE DIAGNOSTIC",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint_epoch}",
        f"Seed: {SEED}",
        "",
        "CONDITIONING DEPENDENCE",
        "",
        header,
        "-" * 110,
    ]

    # ---------------------------------------------------------
    # Run diagnostics
    # ---------------------------------------------------------

    all_finite = True

    with torch.no_grad():

        for timestep_value in TIMESTEPS_TO_TEST:

            timesteps = torch.full(
                (low_light.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=DEVICE,
            )

            # Use the same noisy target for all
            # three conditioning cases.
            noisy_image, true_noise = q_sample(
                target,
                timesteps,
                alpha_bars,
            )

            # -------------------------------------------------
            # 1. Correct condition
            # -------------------------------------------------

            prediction_correct = model(
                noisy_image,
                correct_condition,
                timesteps,
            )

            # -------------------------------------------------
            # 2. Zero condition
            # -------------------------------------------------

            prediction_zero = model(
                noisy_image,
                zero_condition,
                timesteps,
            )

            # -------------------------------------------------
            # 3. Shuffled condition
            # -------------------------------------------------

            prediction_shuffled = model(
                noisy_image,
                shuffled_condition,
                timesteps,
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            epsilon_mse_correct = F.mse_loss(
                prediction_correct,
                true_noise,
            ).item()

            epsilon_mse_zero = F.mse_loss(
                prediction_zero,
                true_noise,
            ).item()

            epsilon_mse_shuffled = F.mse_loss(
                prediction_shuffled,
                true_noise,
            ).item()

            # Prediction changes relative to
            # the correct-condition prediction.
            delta_zero = F.mse_loss(
                prediction_zero,
                prediction_correct,
            ).item()

            delta_shuffled = F.mse_loss(
                prediction_shuffled,
                prediction_correct,
            ).item()

            # Reconstruct x0 for each case.
            x0_correct = reconstruct_x0(
                noisy_image,
                prediction_correct,
                timesteps,
                alpha_bars,
            )

            x0_zero = reconstruct_x0(
                noisy_image,
                prediction_zero,
                timesteps,
                alpha_bars,
            )

            x0_shuffled = reconstruct_x0(
                noisy_image,
                prediction_shuffled,
                timesteps,
                alpha_bars,
            )

            x0_mse_correct = F.mse_loss(
                x0_correct,
                target,
            ).item()

            x0_mse_zero = F.mse_loss(
                x0_zero,
                target,
            ).item()

            x0_mse_shuffled = F.mse_loss(
                x0_shuffled,
                target,
            ).item()

            x0_psnr_correct = calculate_psnr(
                x0_correct,
                target,
            )

            x0_psnr_zero = calculate_psnr(
                x0_zero,
                target,
            )

            x0_psnr_shuffled = calculate_psnr(
                x0_shuffled,
                target,
            )

            # -------------------------------------------------
            # Print compact results
            # -------------------------------------------------

            rows = [
                (
                    "Correct",
                    epsilon_mse_correct,
                    0.0,
                    x0_mse_correct,
                    x0_psnr_correct,
                ),
                (
                    "Zero",
                    epsilon_mse_zero,
                    delta_zero,
                    x0_mse_zero,
                    x0_psnr_zero,
                ),
                (
                    "Shuffled",
                    epsilon_mse_shuffled,
                    delta_shuffled,
                    x0_mse_shuffled,
                    x0_psnr_shuffled,
                ),
            ]

            for (
                condition_name,
                epsilon_mse,
                prediction_delta,
                x0_mse,
                x0_psnr,
            ) in rows:

                print(
                    f"{timestep_value:5d} | "
                    f"{condition_name:<12} | "
                    f"{epsilon_mse:15.8f} | "
                    f"{prediction_delta:15.8f} | "
                    f"{x0_mse:15.8f} | "
                    f"{x0_psnr:12.4f}"
                )

                report_lines.append(
                    f"{timestep_value:5d} | "
                    f"{condition_name:<12} | "
                    f"{epsilon_mse:15.8f} | "
                    f"{prediction_delta:15.8f} | "
                    f"{x0_mse:15.8f} | "
                    f"{x0_psnr:12.4f}"
                )

            # -------------------------------------------------
            # Relative degradation
            # -------------------------------------------------

            zero_eps_change = (
                epsilon_mse_zero
                - epsilon_mse_correct
            )

            shuffled_eps_change = (
                epsilon_mse_shuffled
                - epsilon_mse_correct
            )

            zero_x0_change = (
                x0_mse_zero
                - x0_mse_correct
            )

            shuffled_x0_change = (
                x0_mse_shuffled
                - x0_mse_correct
            )

            print(
                f"\n  t={timestep_value} "
                f"relative changes:"
            )

            print(
                f"    Zero condition "
                f"epsilon MSE change : "
                f"{zero_eps_change:+.8f}"
            )

            print(
                f"    Shuffled condition "
                f"epsilon MSE change : "
                f"{shuffled_eps_change:+.8f}"
            )

            print(
                f"    Zero condition "
                f"x0 MSE change      : "
                f"{zero_x0_change:+.8f}"
            )

            print(
                f"    Shuffled condition "
                f"x0 MSE change      : "
                f"{shuffled_x0_change:+.8f}"
            )

            report_lines.extend(
                [
                    "",
                    (
                        f"t={timestep_value} "
                        f"relative changes:"
                    ),
                    (
                        f"  Zero condition "
                        f"epsilon MSE change : "
                        f"{zero_eps_change:+.8f}"
                    ),
                    (
                        f"  Shuffled condition "
                        f"epsilon MSE change : "
                        f"{shuffled_eps_change:+.8f}"
                    ),
                    (
                        f"  Zero condition "
                        f"x0 MSE change      : "
                        f"{zero_x0_change:+.8f}"
                    ),
                    (
                        f"  Shuffled condition "
                        f"x0 MSE change      : "
                        f"{shuffled_x0_change:+.8f}"
                    ),
                ]
            )

            # -------------------------------------------------
            # Finite-value check
            # -------------------------------------------------

            tensors_to_check = [
                prediction_correct,
                prediction_zero,
                prediction_shuffled,
                x0_correct,
                x0_zero,
                x0_shuffled,
            ]

            for tensor in tensors_to_check:

                if not torch.isfinite(
                    tensor
                ).all():

                    all_finite = False

    # ---------------------------------------------------------
    # Interpretation indicators
    # ---------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("INTERPRETATION INDICATORS")
    print("=" * 110)

    report_lines.extend(
        [
            "",
            "INTERPRETATION INDICATORS",
            "",
        ]
    )

    # Use t=500 as a representative moderate/high
    # noise level for the simple summary.
    t_reference = 500

    timesteps = torch.full(
        (low_light.shape[0],),
        t_reference,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        noisy_image, true_noise = q_sample(
            target,
            timesteps,
            alpha_bars,
        )

        prediction_correct = model(
            noisy_image,
            correct_condition,
            timesteps,
        )

        prediction_zero = model(
            noisy_image,
            zero_condition,
            timesteps,
        )

        prediction_shuffled = model(
            noisy_image,
            shuffled_condition,
            timesteps,
        )

        reference_eps = F.mse_loss(
            prediction_correct,
            true_noise,
        ).item()

        zero_eps = F.mse_loss(
            prediction_zero,
            true_noise,
        ).item()

        shuffled_eps = F.mse_loss(
            prediction_shuffled,
            true_noise,
        ).item()

        zero_delta = F.mse_loss(
            prediction_zero,
            prediction_correct,
        ).item()

        shuffled_delta = F.mse_loss(
            prediction_shuffled,
            prediction_correct,
        ).item()

    print(
        f"Reference timestep: {t_reference}"
    )

    print(
        f"Correct-condition epsilon MSE : "
        f"{reference_eps:.8f}"
    )

    print(
        f"Zero-condition epsilon MSE    : "
        f"{zero_eps:.8f}"
    )

    print(
        f"Shuffled-condition epsilon MSE: "
        f"{shuffled_eps:.8f}"
    )

    print(
        f"Correct -> Zero prediction MSE: "
        f"{zero_delta:.8f}"
    )

    print(
        f"Correct -> Shuffled prediction MSE: "
        f"{shuffled_delta:.8f}"
    )

    report_lines.extend(
        [
            f"Reference timestep: {t_reference}",
            (
                f"Correct-condition epsilon MSE : "
                f"{reference_eps:.8f}"
            ),
            (
                f"Zero-condition epsilon MSE    : "
                f"{zero_eps:.8f}"
            ),
            (
                f"Shuffled-condition epsilon MSE: "
                f"{shuffled_eps:.8f}"
            ),
            (
                f"Correct -> Zero prediction MSE: "
                f"{zero_delta:.8f}"
            ),
            (
                f"Correct -> Shuffled prediction MSE: "
                f"{shuffled_delta:.8f}"
            ),
        ]
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("DIAGNOSTIC VALIDATION")
    print("=" * 110)

    print(
        f"[{'PASS' if all_finite else 'FAIL'}] "
        "All model outputs are finite"
    )

    report_lines.extend(
        [
            "",
            "DIAGNOSTIC VALIDATION",
            "",
            (
                f"[{'PASS' if all_finite else 'FAIL'}] "
                "All model outputs are finite"
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
    print("=" * 75)
    print("TASK 22.4 COMPLETE")
    print("=" * 75)

    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()