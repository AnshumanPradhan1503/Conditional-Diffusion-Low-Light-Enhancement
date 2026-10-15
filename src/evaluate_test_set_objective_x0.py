from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from src.configs import objective_x0_config as config
from src.diffusion.reverse import reverse_diffusion
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders
from src.preprocessing.normalization import tensor_to_image


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:

    psnr = peak_signal_noise_ratio(
        target,
        prediction,
        data_range=255,
    )

    ssim = structural_similarity(
        target,
        prediction,
        channel_axis=2,
        data_range=255,
    )

    return float(psnr), float(ssim)


def load_model(
    device: torch.device,
) -> ConditionalUNet:

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint_path = (
        config.CHECKPOINT_ROOT
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_previous_results(
    csv_path: Path,
) -> list[dict]:

    if not csv_path.exists():
        return []

    rows = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(row)

    return rows


def save_metrics(
    csv_path: Path,
    rows: list[dict],
) -> None:

    fieldnames = [
        "index",
        "low_file",
        "high_file",
        "low_psnr",
        "low_ssim",
        "generated_psnr",
        "generated_ssim",
        "psnr_improvement",
        "ssim_improvement",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def main():

    print("=" * 70)
    print(
        "TASK 15.5.5 — OFFICIAL TEST EVALUATION"
    )
    print(
        "OBJECTIVE: EPSILON + 0.1 x0 LOSS"
    )
    print("=" * 70)

    device = torch.device(
        config.DEVICE
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print(
        "\nCreating test DataLoader..."
    )

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_dataset = test_loader.dataset

    total_images = len(test_dataset)

    print(
        f"Official test pairs: "
        f"{total_images}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading objective-x0 best model..."
    )

    model = load_model(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Diffusion schedule
    # --------------------------------------------------------

    betas, alphas, alpha_bars = (
        get_diffusion_parameters(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END,
        )
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    print(
        f"Diffusion steps: "
        f"{config.TIMESTEPS}"
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir = (
        config.RESULTS_ROOT
        / "test_evaluation"
    )

    visual_dir = (
        output_dir
        / "visual_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "test_metrics_objective_x0.csv"
    )

    # --------------------------------------------------------
    # Resume previous evaluation
    # --------------------------------------------------------

    previous_results = load_previous_results(
        csv_path
    )

    completed_indices = {
        int(row["index"])
        for row in previous_results
    }

    metrics = previous_results.copy()

    print(
        f"\nPreviously completed: "
        f"{len(completed_indices)}/{total_images}"
    )

    if len(completed_indices) == total_images:

        print(
            "\nAll test images are already evaluated."
        )

        print(
            f"Results file:\n{csv_path}"
        )

        return

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nStarting official test evaluation..."
    )

    print(
        "Progress will be saved after every image."
    )

    for index in range(total_images):

        if index in completed_indices:

            print(
                f"[SKIP] Image "
                f"{index + 1}/{total_images} "
                f"already completed."
            )

            continue

        print(
            f"\n[START] Image "
            f"{index + 1}/{total_images}"
        )

        # ----------------------------------------------------
        # Load one deterministic test pair
        # ----------------------------------------------------

        low_light, normal_light = (
            test_dataset[index]
        )

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

        low_file = (
            test_dataset.low_files[index].name
            if hasattr(test_dataset, "low_files")
            else f"index_{index}"
        )

        high_file = (
            test_dataset.high_files[index].name
            if hasattr(test_dataset, "high_files")
            else f"index_{index}"
        )

        # ----------------------------------------------------
        # Convert target and low-light images
        # ----------------------------------------------------

        low_np = np.asarray(
            tensor_to_image(
                low_light[0]
            )
        ).astype(np.uint8)

        target_np = np.asarray(
            tensor_to_image(
                normal_light[0]
            )
        ).astype(np.uint8)

        low_psnr, low_ssim = (
            calculate_metrics(
                low_np,
                target_np,
            )
        )

        # ----------------------------------------------------
        # Conditional reverse diffusion
        # ----------------------------------------------------

        def predicted_noise_fn(
            x_t,
            condition,
            timesteps,
        ):

            return model(
                noisy_image=x_t,
                condition=condition,
                timesteps=timesteps,
            )

        print(
            "  Running 1000-step reverse diffusion..."
        )

        with torch.no_grad():

            generated = reverse_diffusion(
                shape=normal_light.shape,
                condition=low_light,
                predicted_noise_fn=predicted_noise_fn,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                device=device,
            )

        print(
            "  Reverse diffusion completed."
        )

        # ----------------------------------------------------
        # Generated image
        # ----------------------------------------------------

        generated_np = np.asarray(
            tensor_to_image(
                generated[0]
            )
        ).astype(np.uint8)

        generated_psnr, generated_ssim = (
            calculate_metrics(
                generated_np,
                target_np,
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        result = {
            "index": index,
            "low_file": low_file,
            "high_file": high_file,
            "low_psnr": f"{low_psnr:.6f}",
            "low_ssim": f"{low_ssim:.6f}",
            "generated_psnr":
                f"{generated_psnr:.6f}",
            "generated_ssim":
                f"{generated_ssim:.6f}",
            "psnr_improvement":
                f"{generated_psnr - low_psnr:.6f}",
            "ssim_improvement":
                f"{generated_ssim - low_ssim:.6f}",
        }

        metrics.append(result)

        completed_indices.add(index)

        # ----------------------------------------------------
        # Save CSV IMMEDIATELY
        # ----------------------------------------------------

        metrics.sort(
            key=lambda row: int(row["index"])
        )

        save_metrics(
            csv_path,
            metrics,
        )

        # ----------------------------------------------------
        # Save images IMMEDIATELY
        # ----------------------------------------------------

        low_image = tensor_to_image(
            low_light[0]
        )

        generated_image = tensor_to_image(
            generated[0]
        )

        high_image = tensor_to_image(
            normal_light[0]
        )

        low_image.save(
            visual_dir
            / f"{index:03d}_low.png"
        )

        generated_image.save(
            visual_dir
            / f"{index:03d}_generated.png"
        )

        high_image.save(
            visual_dir
            / f"{index:03d}_high.png"
        )

        print(
            f"  PSNR: "
            f"{generated_psnr:.4f} dB"
        )

        print(
            f"  SSIM: "
            f"{generated_ssim:.6f}"
        )

        print(
            f"  PSNR change: "
            f"{generated_psnr - low_psnr:+.4f} dB"
        )

        print(
            f"  SSIM change: "
            f"{generated_ssim - low_ssim:+.6f}"
        )

        print(
            f"[SAVED] "
            f"{len(completed_indices)}/{total_images}"
        )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del (
            low_light,
            normal_light,
            generated,
            low_np,
            target_np,
            generated_np,
        )

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Final aggregate results
    # --------------------------------------------------------

    generated_psnr_values = np.array(
        [
            float(row["generated_psnr"])
            for row in metrics
        ]
    )

    generated_ssim_values = np.array(
        [
            float(row["generated_ssim"])
            for row in metrics
        ]
    )

    low_psnr_values = np.array(
        [
            float(row["low_psnr"])
            for row in metrics
        ]
    )

    low_ssim_values = np.array(
        [
            float(row["low_ssim"])
            for row in metrics
        ]
    )

    psnr_improvements = (
        generated_psnr_values
        - low_psnr_values
    )

    ssim_improvements = (
        generated_ssim_values
        - low_ssim_values
    )

    low_psnr_mean = np.mean(
        low_psnr_values
    )

    low_ssim_mean = np.mean(
        low_ssim_values
    )

    generated_psnr_mean = np.mean(
        generated_psnr_values
    )

    generated_ssim_mean = np.mean(
        generated_ssim_values
    )

    psnr_improvement_mean = np.mean(
        psnr_improvements
    )

    ssim_improvement_mean = np.mean(
        ssim_improvements
    )

    psnr_improved = np.sum(
        psnr_improvements > 0
    )

    ssim_improved = np.sum(
        ssim_improvements > 0
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "LOW-LIGHT BASELINE"
    )

    print(
        f"Mean PSNR: "
        f"{low_psnr_mean:.4f} dB"
    )

    print(
        f"Mean SSIM: "
        f"{low_ssim_mean:.6f}"
    )

    print(
        "\nOBJECTIVE-X0 DIFFUSION MODEL"
    )

    print(
        f"Mean PSNR: "
        f"{generated_psnr_mean:.4f} dB"
    )

    print(
        f"Mean SSIM: "
        f"{generated_ssim_mean:.6f}"
    )

    print(
        "\nIMPROVEMENT"
    )

    print(
        f"Mean PSNR change: "
        f"{psnr_improvement_mean:+.4f} dB"
    )

    print(
        f"Mean SSIM change: "
        f"{ssim_improvement_mean:+.6f}"
    )

    print(
        f"\nPSNR improved on "
        f"{psnr_improved}/{total_images} images"
    )

    print(
        f"SSIM improved on "
        f"{ssim_improved}/{total_images} images"
    )

    # --------------------------------------------------------
    # Save final summary
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "summary_objective_x0.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "TASK 15.5.5 — OFFICIAL TEST EVALUATION\n"
        )

        file.write(
            "OBJECTIVE: EPSILON + 0.1 x0 LOSS\n\n"
        )

        file.write(
            f"Model parameters: "
            f"{parameter_count:,}\n"
        )

        file.write(
            f"Test images: "
            f"{total_images}\n"
        )

        file.write(
            f"Diffusion steps: "
            f"{config.TIMESTEPS}\n\n"
        )

        file.write(
            "LOW-LIGHT BASELINE\n"
        )

        file.write(
            f"Mean PSNR: "
            f"{low_psnr_mean:.6f} dB\n"
        )

        file.write(
            f"Mean SSIM: "
            f"{low_ssim_mean:.6f}\n\n"
        )

        file.write(
            "OBJECTIVE-X0 DIFFUSION MODEL\n"
        )

        file.write(
            f"Mean PSNR: "
            f"{generated_psnr_mean:.6f} dB\n"
        )

        file.write(
            f"Mean SSIM: "
            f"{generated_ssim_mean:.6f}\n\n"
        )

        file.write(
            "IMPROVEMENT\n"
        )

        file.write(
            f"Mean PSNR change: "
            f"{psnr_improvement_mean:+.6f} dB\n"
        )

        file.write(
            f"Mean SSIM change: "
            f"{ssim_improvement_mean:+.6f}\n\n"
        )

        file.write(
            f"PSNR improved: "
            f"{psnr_improved}/{total_images}\n"
        )

        file.write(
            f"SSIM improved: "
            f"{ssim_improved}/{total_images}\n"
        )

    print(
        f"\nCSV saved to:\n{csv_path}"
    )

    print(
        f"\nSummary saved to:\n{summary_path}"
    )

    print(
        f"\nVisual results saved to:\n{visual_dir}"
    )

    print("\n" + "=" * 70)
    print(
        "TASK 15.5.5 EVALUATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()