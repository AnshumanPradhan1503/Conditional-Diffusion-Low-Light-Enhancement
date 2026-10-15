from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.configs import training_config as config
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


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
    / "spatial_condition"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "spatial_condition.txt"
)

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

DEVICE = torch.device("cpu")

SEED = 42

BATCH_SIZE = 4


def tensor_statistics(
    tensor: torch.Tensor,
) -> dict[str, float]:

    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "mean_abs": tensor.abs().mean().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


def spatial_difference(
    original: torch.Tensor,
    modified: torch.Tensor,
) -> float:

    return F.mse_loss(
        original,
        modified,
    ).item()


def main() -> None:

    print("=" * 75)
    print("TASK 22.5 - SPATIAL CONDITIONING REPRESENTATION DIAGNOSTIC")
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

    # ---------------------------------------------------------
    # Load original model
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
    # Inspect first convolution
    # ---------------------------------------------------------

    input_weights = (
        model.input_conv.weight.detach()
    )

    print("\n")
    print("=" * 75)
    print("INPUT CONDITIONING WEIGHT ANALYSIS")
    print("=" * 75)

    # Original input is:
    #
    # [noisy target: 3 channels]
    # +
    # [low-light condition: 3 channels]
    #
    # Therefore:
    # channels 0-2 = noisy image
    # channels 3-5 = low-light condition

    noisy_weights = input_weights[:, :3]
    condition_weights = input_weights[:, 3:]

    noisy_weight_magnitude = (
        noisy_weights.abs().mean().item()
    )

    condition_weight_magnitude = (
        condition_weights.abs().mean().item()
    )

    print(
        f"Noisy-image weight magnitude : "
        f"{noisy_weight_magnitude:.8f}"
    )

    print(
        f"Condition weight magnitude   : "
        f"{condition_weight_magnitude:.8f}"
    )

    if noisy_weight_magnitude > 0:
        print(
            f"Condition / noisy ratio      : "
            f"{condition_weight_magnitude / noisy_weight_magnitude:.8f}"
        )

    # ---------------------------------------------------------
    # Construct controlled spatial modifications
    # ---------------------------------------------------------

    # 1. Original condition.
    condition_original = low_light.clone()

    # 2. Horizontally flipped condition.
    condition_flipped = torch.flip(
        low_light,
        dims=[3],
    )

    # 3. Spatially shifted condition.
    condition_shifted = torch.roll(
        low_light,
        shifts=32,
        dims=3,
    )

    # 4. Spatially constant condition.
    #
    # Each image keeps its own channel means,
    # but all spatial variation is removed.
    condition_mean = (
        low_light.mean(
            dim=(2, 3),
            keepdim=True,
        )
    )

    condition_constant = (
        condition_mean.expand_as(
            low_light
        )
    )

    # ---------------------------------------------------------
    # Compare condition representations
    # ---------------------------------------------------------

    condition_variants = {
        "Original": condition_original,
        "HorizontalFlip": condition_flipped,
        "Shifted32px": condition_shifted,
        "SpatialMean": condition_constant,
    }

    print("\n")
    print("=" * 75)
    print("CONDITION INPUT STATISTICS")
    print("=" * 75)

    stats_header = (
        f"{'Condition':<18} | "
        f"{'Mean':>12} | "
        f"{'Std':>12} | "
        f"{'MeanAbs':>12}"
    )

    print(stats_header)
    print("-" * 75)

    report_lines = [
        "TASK 22.5 - SPATIAL CONDITIONING REPRESENTATION DIAGNOSTIC",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Seed: {SEED}",
        "",
        "INPUT CONDITIONING WEIGHT ANALYSIS",
        "",
        (
            f"Noisy-image weight magnitude : "
            f"{noisy_weight_magnitude:.8f}"
        ),
        (
            f"Condition weight magnitude   : "
            f"{condition_weight_magnitude:.8f}"
        ),
        "",
        "CONDITION INPUT STATISTICS",
        "",
        stats_header,
        "-" * 75,
    ]

    for name, condition in condition_variants.items():

        stats = tensor_statistics(
            condition
        )

        print(
            f"{name:<18} | "
            f"{stats['mean']:12.8f} | "
            f"{stats['std']:12.8f} | "
            f"{stats['mean_abs']:12.8f}"
        )

        report_lines.append(
            f"{name:<18} | "
            f"{stats['mean']:12.8f} | "
            f"{stats['std']:12.8f} | "
            f"{stats['mean_abs']:12.8f}"
        )

    # ---------------------------------------------------------
    # Run model with same noisy input
    # ---------------------------------------------------------

    # A moderate timestep is useful because the model must
    # rely on both noisy target information and conditioning.
    timestep_value = 500

    timesteps = torch.full(
        (low_light.shape[0],),
        timestep_value,
        dtype=torch.long,
        device=DEVICE,
    )

    # Generate a fixed noisy target.
    #
    # We deliberately generate this once and reuse it for
    # every condition variant so that only the condition changes.
    from src.diffusion.schedule import get_diffusion_parameters
    from src.diffusion.forward import q_sample

    _, _, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    alpha_bars = alpha_bars.to(DEVICE)

    noisy_image, true_noise = q_sample(
        target,
        timesteps,
        alpha_bars,
    )

    predictions = {}

    with torch.no_grad():

        for name, condition in condition_variants.items():

            predictions[name] = model(
                noisy_image,
                condition,
                timesteps,
            )

    # ---------------------------------------------------------
    # Prediction sensitivity
    # ---------------------------------------------------------

    print("\n")
    print("=" * 95)
    print(
        f"PREDICTION SENSITIVITY AT t={timestep_value}"
    )
    print("=" * 95)

    sensitivity_header = (
        f"{'Condition':<18} | "
        f"{'Epsilon MSE':>15} | "
        f"{'Δ from Original':>18} | "
        f"{'Prediction Std':>15}"
    )

    print(sensitivity_header)
    print("-" * 95)

    report_lines.extend(
        [
            "",
            (
                f"PREDICTION SENSITIVITY AT "
                f"t={timestep_value}"
            ),
            "",
            sensitivity_header,
            "-" * 95,
        ]
    )

    original_prediction = predictions[
        "Original"
    ]

    for name, prediction in predictions.items():

        epsilon_mse = F.mse_loss(
            prediction,
            true_noise,
        ).item()

        delta = spatial_difference(
            prediction,
            original_prediction,
        )

        prediction_std = (
            prediction.std().item()
        )

        print(
            f"{name:<18} | "
            f"{epsilon_mse:15.8f} | "
            f"{delta:18.8f} | "
            f"{prediction_std:15.8f}"
        )

        report_lines.append(
            f"{name:<18} | "
            f"{epsilon_mse:15.8f} | "
            f"{delta:18.8f} | "
            f"{prediction_std:15.8f}"
        )

    # ---------------------------------------------------------
    # Spatial correspondence test
    # ---------------------------------------------------------

    print("\n")
    print("=" * 95)
    print("SPATIAL CORRESPONDENCE TEST")
    print("=" * 95)

    flip_delta = spatial_difference(
        predictions["HorizontalFlip"],
        predictions["Original"],
    )

    shift_delta = spatial_difference(
        predictions["Shifted32px"],
        predictions["Original"],
    )

    mean_delta = spatial_difference(
        predictions["SpatialMean"],
        predictions["Original"],
    )

    print(
        f"Horizontal flip prediction Δ : "
        f"{flip_delta:.8f}"
    )

    print(
        f"32-pixel shift prediction Δ  : "
        f"{shift_delta:.8f}"
    )

    print(
        f"Spatial-mean prediction Δ    : "
        f"{mean_delta:.8f}"
    )

    report_lines.extend(
        [
            "",
            "SPATIAL CORRESPONDENCE TEST",
            "",
            (
                f"Horizontal flip prediction Δ : "
                f"{flip_delta:.8f}"
            ),
            (
                f"32-pixel shift prediction Δ  : "
                f"{shift_delta:.8f}"
            ),
            (
                f"Spatial-mean prediction Δ    : "
                f"{mean_delta:.8f}"
            ),
        ]
    )

    # ---------------------------------------------------------
    # Interpretation indicators
    # ---------------------------------------------------------

    print("\n")
    print("=" * 95)
    print("INTERPRETATION INDICATORS")
    print("=" * 95)

    print(
        "The Original condition is the reference."
    )

    print(
        "A large prediction Δ after spatial modification "
        "suggests the model is sensitive to spatial structure."
    )

    print(
        "A small prediction Δ suggests the model is relying "
        "more on condition statistics than exact spatial arrangement."
    )

    report_lines.extend(
        [
            "",
            "INTERPRETATION INDICATORS",
            "",
            "The Original condition is the reference.",
            (
                "A large prediction Δ after spatial modification "
                "suggests the model is sensitive to spatial structure."
            ),
            (
                "A small prediction Δ suggests the model is relying "
                "more on condition statistics than exact spatial arrangement."
            ),
        ]
    )

    # ---------------------------------------------------------
    # Finite validation
    # ---------------------------------------------------------

    all_finite = True

    tensors_to_check = list(
        predictions.values()
    )

    for tensor in tensors_to_check:

        if not torch.isfinite(
            tensor
        ).all():

            all_finite = False

    print("\n")
    print("=" * 95)
    print("DIAGNOSTIC VALIDATION")
    print("=" * 95)

    print(
        f"[{'PASS' if all_finite else 'FAIL'}] "
        "All predictions are finite"
    )

    report_lines.extend(
        [
            "",
            "DIAGNOSTIC VALIDATION",
            "",
            (
                f"[{'PASS' if all_finite else 'FAIL'}] "
                "All predictions are finite"
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
    print("TASK 22.5 COMPLETE")
    print("=" * 75)

    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()