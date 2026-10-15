from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.configs import training_config as config
from src.preprocessing.dataloader import create_dataloaders
from src.preprocessing.illumination import illumination_from_normalized_rgb
from src.preprocessing.illumination_proxy import (
    illumination_proxy_from_normalized_rgb,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "illumination_proxy_analysis"
)


def main() -> None:
    print(
        "Testing low-frequency illumination proxy "
        "on LOLv2-real..."
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use the project's validated dataset configuration.
    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    test_dataset = test_loader.dataset

    selected_indices = [0, 10, 25, 50, 75, 99]

    rows = []

    for index in selected_indices:

        low_light, normal_light = test_dataset[index]

        low_batch = low_light.unsqueeze(0)

        # Convert normalized RGB [-1, 1] to [0, 1].
        low_01 = (
            (low_batch + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        # -------------------------------------------------
        # Ordinary luminance map
        # -------------------------------------------------
        luminance = illumination_from_normalized_rgb(
            low_batch
        )

        # -------------------------------------------------
        # Low-frequency illumination proxy
        # -------------------------------------------------
        proxy = illumination_proxy_from_normalized_rgb(
            low_batch,
            kernel_size=15,
            sigma=3.0,
        )

        # -------------------------------------------------
        # Difference between luminance and proxy
        # -------------------------------------------------
        difference = (
            luminance - proxy
        ).abs()

        image_name = Path(
            test_dataset.pairs[index][0]
        ).name

        brightness = low_01.mean().item()

        luminance_mean = luminance.mean().item()
        luminance_std = luminance.std().item()

        proxy_mean = proxy.mean().item()
        proxy_std = proxy.std().item()

        difference_mean = difference.mean().item()
        difference_std = difference.std().item()

        rows.append(
            (
                index,
                image_name,
                brightness,
                luminance_mean,
                luminance_std,
                proxy_mean,
                proxy_std,
                difference_mean,
                difference_std,
            )
        )

        print()
        print(
            f"Image {index}: {image_name}"
        )

        print(
            f"Brightness:       "
            f"{brightness:.6f}"
        )

        print(
            f"Luminance mean:   "
            f"{luminance_mean:.6f}"
        )

        print(
            f"Luminance std:    "
            f"{luminance_std:.6f}"
        )

        print(
            f"Proxy mean:       "
            f"{proxy_mean:.6f}"
        )

        print(
            f"Proxy std:        "
            f"{proxy_std:.6f}"
        )

        print(
            f"Difference mean:  "
            f"{difference_mean:.6f}"
        )

        print(
            f"Difference std:   "
            f"{difference_std:.6f}"
        )

        # -------------------------------------------------
        # Prepare images for visualization
        # -------------------------------------------------

        low_rgb = (
            low_01
            .squeeze(0)
            .permute(1, 2, 0)
        )

        luminance_img = (
            luminance
            .squeeze(0)
            .squeeze(0)
        )

        proxy_img = (
            proxy
            .squeeze(0)
            .squeeze(0)
        )

        difference_img = (
            difference
            .squeeze(0)
            .squeeze(0)
        )

        figure, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4),
        )

        # Low-light RGB
        axes[0].imshow(
            low_rgb.cpu().numpy()
        )

        axes[0].set_title(
            f"Low-light\n{image_name}"
        )

        axes[0].axis("off")

        # Luminance
        axes[1].imshow(
            luminance_img.cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[1].set_title(
            "Luminance"
        )

        axes[1].axis("off")

        # Low-frequency illumination proxy
        axes[2].imshow(
            proxy_img.cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[2].set_title(
            "Low-frequency\nillumination proxy"
        )

        axes[2].axis("off")

        # Difference
        axes[3].imshow(
            difference_img.cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[3].set_title(
            "|Luminance - Proxy|"
        )

        axes[3].axis("off")

        figure.tight_layout()

        output_path = (
            RESULTS_ROOT
            / f"comparison_{index:03d}.png"
        )

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

    # -----------------------------------------------------
    # Save numerical summary
    # -----------------------------------------------------

    summary_path = (
        RESULTS_ROOT / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Task 24.2D - LOLv2-real "
            "Illumination Proxy Analysis\n"
        )

        file.write(
            "Low-frequency illumination proxy "
            "uses Gaussian smoothing of luminance.\n"
        )

        file.write(
            "This is a proxy, not a ground-truth "
            "physical illumination map.\n\n"
        )

        file.write(
            "index,image,brightness,"
            "luminance_mean,luminance_std,"
            "proxy_mean,proxy_std,"
            "difference_mean,difference_std\n"
        )

        for row in rows:

            file.write(
                f"{row[0]},"
                f"{row[1]},"
                f"{row[2]:.6f},"
                f"{row[3]:.6f},"
                f"{row[4]:.6f},"
                f"{row[5]:.6f},"
                f"{row[6]:.6f},"
                f"{row[7]:.6f},"
                f"{row[8]:.6f}\n"
            )

    print()
    print(
        "Analysis completed."
    )

    print(
        f"Results saved to: "
        f"{RESULTS_ROOT}"
    )


if __name__ == "__main__":
    main()
