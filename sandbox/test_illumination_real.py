from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.preprocessing.dataloader import create_dataloaders
from src.preprocessing.illumination import illumination_from_normalized_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "illumination_analysis"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor):
    """
    Convert [C,H,W] tensor in [-1,1] to [H,W,C] numpy image in [0,1].
    """
    image = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).cpu().numpy()
    return image


def illumination_to_image(tensor: torch.Tensor):
    """
    Convert [1,H,W] illumination tensor in [0,1]
    to [H,W] numpy image.
    """
    return tensor.squeeze(0).cpu().numpy()


def main() -> None:
    print("Loading LOLv2-real test dataset...")

    _, _, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
    )

    test_dataset = test_loader.dataset

    print(f"Test images available: {len(test_dataset)}")

    # ---------------------------------------------------------
    # Select representative test images.
    #
    # These indices are deliberately fixed so the diagnostic
    # remains reproducible.
    # ---------------------------------------------------------
    selected_indices = [0, 10, 25, 50, 75, 99]

    rows = []

    print("\nGenerating illumination diagnostics...")

    for index in selected_indices:
        low_light, normal_light = test_dataset[index]

        illumination = illumination_from_normalized_rgb(low_light)

        low_image = tensor_to_image(low_light)
        normal_image = tensor_to_image(normal_light)
        illumination_image = illumination_to_image(illumination)

        low_brightness = float(low_image.mean())
        illumination_mean = float(illumination.mean())
        illumination_std = float(illumination.std())
        illumination_min = float(illumination.min())
        illumination_max = float(illumination.max())

        image_name = Path(test_dataset.pairs[index][0]).name

        rows.append(
            {
                "index": index,
                "image": image_name,
                "low_brightness": low_brightness,
                "illumination_mean": illumination_mean,
                "illumination_std": illumination_std,
                "illumination_min": illumination_min,
                "illumination_max": illumination_max,
            }
        )

        print(
            f"{index:3d} | {image_name:15s} | "
            f"brightness={low_brightness:.6f} | "
            f"illumination mean={illumination_mean:.6f} | "
            f"std={illumination_std:.6f} | "
            f"range=[{illumination_min:.6f}, {illumination_max:.6f}]"
        )

        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(low_image)
        axes[0].set_title("Low-Light RGB")
        axes[0].axis("off")

        axes[1].imshow(illumination_image, cmap="gray")
        axes[1].set_title("Illumination Map")
        axes[1].axis("off")

        axes[2].imshow(normal_image)
        axes[2].set_title("Normal-Light Target")
        axes[2].axis("off")

        figure.suptitle(image_name)

        output_path = OUTPUT_ROOT / f"illumination_{index:03d}.png"

        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    # ---------------------------------------------------------
    # Save numerical summary
    # ---------------------------------------------------------
    summary_path = OUTPUT_ROOT / "summary.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("ILLUMINATION REPRESENTATION ANALYSIS\n")
        file.write("===================================\n\n")
        file.write(f"Test images available: {len(test_dataset)}\n")
        file.write(f"Selected images: {selected_indices}\n\n")

        file.write(
            "index,image,low_brightness,"
            "illumination_mean,illumination_std,"
            "illumination_min,illumination_max\n"
        )

        for row in rows:
            file.write(
                f"{row['index']},{row['image']},"
                f"{row['low_brightness']:.6f},"
                f"{row['illumination_mean']:.6f},"
                f"{row['illumination_std']:.6f},"
                f"{row['illumination_min']:.6f},"
                f"{row['illumination_max']:.6f}\n"
            )

    print("\nDiagnostic images saved to:")
    print(OUTPUT_ROOT)

    print("\nSummary saved to:")
    print(summary_path)

    print("\nTask 24.2B completed successfully.")


if __name__ == "__main__":
    main()