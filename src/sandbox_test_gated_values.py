from __future__ import annotations

from pathlib import Path

import torch

from src.models.multiscale_gated_unet import MultiScaleGatedConditionalUNet


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

OUTPUT_PATH = RESULTS_DIR / "gate_values.txt"

INITIAL_GATE = 0.1


def main() -> None:
    print("=" * 60)
    print("TASK 21.7 - LEARNED GATE VALUE DIAGNOSTIC")
    print("=" * 60)

    print(f"Checkpoint : {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = MultiScaleGatedConditionalUNet(
        image_channels=3,
        condition_channels=3,
        base_channels=32,
        time_dim=128,
    )

    # The training checkpoint stores the model parameters.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            "Could not find model_state_dict or state_dict in checkpoint."
        )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")

    print("\nGATE PARAMETERS")
    print("-" * 60)

    gate_parameters = {}

    for name, parameter in model.named_parameters():
        name_lower = name.lower()

        if "gate" in name_lower:
            value = parameter.detach().cpu().item()
            gate_parameters[name] = value

            print(f"{name:<45} {value:.8f}")

    if not gate_parameters:
        raise RuntimeError(
            "No parameters containing 'gate' were found."
        )

    print("\nCHANGE FROM INITIAL VALUE")
    print("-" * 60)

    report_lines = [
        "TASK 21.7 - LEARNED GATE VALUE DIAGNOSTIC",
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Epoch: {checkpoint.get('epoch', 'unknown')}",
        "",
        f"Initial gate value: {INITIAL_GATE:.8f}",
        "",
        "Learned gate values:",
    ]

    for name, value in gate_parameters.items():
        change = value - INITIAL_GATE

        if value < INITIAL_GATE:
            interpretation = "DECREASED - conditioning suppressed"
        elif value > INITIAL_GATE:
            interpretation = "INCREASED - conditioning strengthened"
        else:
            interpretation = "UNCHANGED"

        print(f"{name}")
        print(f"  Initial : {INITIAL_GATE:.8f}")
        print(f"  Learned : {value:.8f}")
        print(f"  Change  : {change:+.8f}")
        print(f"  Meaning : {interpretation}")
        print()

        report_lines.extend(
            [
                f"{name}:",
                f"  Initial value : {INITIAL_GATE:.8f}",
                f"  Learned value : {value:.8f}",
                f"  Change        : {change:+.8f}",
                f"  Interpretation: {interpretation}",
                "",
            ]
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print(f"Report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()