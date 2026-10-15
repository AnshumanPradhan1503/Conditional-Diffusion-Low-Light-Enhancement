from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


from src.configs import capacity_48_config as config
from src.diffusion.reverse import p_sample
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.normalization import tensor_to_image
from src.preprocessing.paired_dataset import LOLv2PairedDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(device: torch.device) -> ConditionalUNet:
    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

    checkpoint_path = (
        config.CHECKPOINT_ROOT / "best_model.pt"
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


def calculate_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:

    prediction_np = (
        prediction.detach()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .div(2.0)
        .permute(1, 2, 0)
        .numpy()
    )

    reference_np = (
        reference.detach()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .div(2.0)
        .permute(1, 2, 0)
        .numpy()
    )

    psnr = peak_signal_noise_ratio(
        reference_np,
        prediction_np,
        data_range=1.0,
    )

    ssim = structural_similarity(
        reference_np,
        prediction_np,
        channel_axis=2,
        data_range=1.0,
    )

    return float(psnr), float(ssim)


def generate_batch(
    model: ConditionalUNet,
    condition: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    batch_size = condition.shape[0]

    current = torch.randn(
        condition.shape,
        device=device,
    )

    with torch.no_grad():

        for timestep_value in reversed(
            range(config.TIMESTEPS)
        ):

            t = torch.full(
                (batch_size,),
                timestep_value,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = model(
                noisy_image=current,
                condition=condition,
                timesteps=t,
            )

            current = p_sample(
                x_t=current,
                timesteps=t,
                predicted_noise=predicted_noise,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
            )

    return current.clamp(-1.0, 1.0)


def save_visual_result(
    low_image: torch.Tensor,
    generated_image: torch.Tensor,
    high_image: torch.Tensor,
    index: int,
    output_dir: Path,
) -> None:

    low_pil = tensor_to_image(
        low_image.detach().cpu()
    )

    generated_pil = tensor_to_image(
        generated_image.detach().cpu()
    )

    high_pil = tensor_to_image(
        high_image.detach().cpu()
    )

    low_path = (
        output_dir / f"low_{index:03d}.png"
    )

    generated_path = (
        output_dir / f"generated_{index:03d}.png"
    )

    high_path = (
        output_dir / f"high_{index:03d}.png"
    )

    low_pil.save(low_path)
    generated_pil.save(generated_path)
    high_pil.save(high_path)


def main() -> None:

    print("=" * 60)
    print("PHASE 4 — TASK 13.2: OFFICIAL TEST-SET EVALUATION")
    print("=" * 60)

    device = torch.device(config.DEVICE)

    print(f"Device: {device}")
    print(f"Diffusion steps: {config.TIMESTEPS}")
    print("Evaluation set: official LOLv2-real test set")

    set_seed(config.RANDOM_SEED)

    output_dir = (
        config.RESULTS_ROOT / "test_evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Visual output directory
    # ---------------------------------------------------------

    visual_output_dir = (
        output_dir / "visual_results"
    )

    visual_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Visual results directory: "
        f"{visual_output_dir}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = LOLv2PairedDataset(
        split_file=config.SPLIT_ROOT / "test_pairs.txt",
        data_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        mode="test",
    )

    print(f"Test pairs: {len(dataset)}")

    if len(dataset) != 100:
        raise ValueError(
            f"Expected 100 official test pairs, "
            f"found {len(dataset)}."
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = load_model(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
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

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    batch_size = config.BATCH_SIZE

    results = []

    total_batches = (
        len(dataset) + batch_size - 1
    ) // batch_size

    print()
    print(
        f"Evaluation batch size: {batch_size}"
    )

    print(
        f"Total batches: {total_batches}"
    )

    print()

    print(
        "Starting 1000-step generation for "
        "all official test images..."
    )

    print()

    for batch_start in range(
        0,
        len(dataset),
        batch_size,
    ):

        batch_end = min(
            batch_start + batch_size,
            len(dataset),
        )

        low_images = []
        high_images = []
        names = []

        for index in range(
            batch_start,
            batch_end,
        ):

            low_image, high_image = dataset[index]

            low_images.append(low_image)
            high_images.append(high_image)

            low_name, high_name = dataset.pairs[index]

            names.append(
                (
                    Path(low_name).stem,
                    Path(high_name).stem,
                )
            )

        low_batch = torch.stack(
            low_images
        ).to(device)

        high_batch = torch.stack(
            high_images
        ).to(device)

        # -----------------------------------------------------
        # Generate
        # -----------------------------------------------------

        generated_batch = generate_batch(
            model=model,
            condition=low_batch,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            device=device,
        )

        # -----------------------------------------------------
        # Metrics + visual results
        # -----------------------------------------------------

        for local_index in range(
            batch_end - batch_start
        ):

            global_index = (
                batch_start + local_index
            )

            low_psnr, low_ssim = (
                calculate_metrics(
                    low_batch[local_index],
                    high_batch[local_index],
                )
            )

            generated_psnr, generated_ssim = (
                calculate_metrics(
                    generated_batch[local_index],
                    high_batch[local_index],
                )
            )

            low_name, high_name = names[
                local_index
            ]

            results.append(
                {
                    "index": global_index,
                    "low_file": low_name,
                    "high_file": high_name,
                    "low_psnr": low_psnr,
                    "low_ssim": low_ssim,
                    "generated_psnr": generated_psnr,
                    "generated_ssim": generated_ssim,
                    "psnr_improvement": (
                        generated_psnr - low_psnr
                    ),
                    "ssim_improvement": (
                        generated_ssim - low_ssim
                    ),
                }
            )

            # -------------------------------------------------
            # Save low / generated / reference images
            # -------------------------------------------------

            save_visual_result(
                low_image=low_batch[local_index],
                generated_image=generated_batch[local_index],
                high_image=high_batch[local_index],
                index=global_index,
                output_dir=visual_output_dir,
            )

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        completed = batch_end

        current_generated_psnr = np.mean(
            [
                row["generated_psnr"]
                for row in results
            ]
        )

        current_generated_ssim = np.mean(
            [
                row["generated_ssim"]
                for row in results
            ]
        )

        print(
            f"Completed "
            f"{completed}/{len(dataset)} "
            f"images | "
            f"Current mean PSNR: "
            f"{current_generated_psnr:.4f} dB | "
            f"Current mean SSIM: "
            f"{current_generated_ssim:.6f}"
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = (
        output_dir / "test_metrics.csv"
    )

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
        writer.writerows(results)

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    low_psnr_values = np.array(
        [
            row["low_psnr"]
            for row in results
        ]
    )

    low_ssim_values = np.array(
        [
            row["low_ssim"]
            for row in results
        ]
    )

    generated_psnr_values = np.array(
        [
            row["generated_psnr"]
            for row in results
        ]
    )

    generated_ssim_values = np.array(
        [
            row["generated_ssim"]
            for row in results
        ]
    )

    psnr_improvement_values = (
        generated_psnr_values
        - low_psnr_values
    )

    ssim_improvement_values = (
        generated_ssim_values
        - low_ssim_values
    )

    summary_path = (
        output_dir / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHASE 4 — TASK 13.2\n"
            "OFFICIAL LOLv2-REAL TEST-SET EVALUATION\n"
            "\n"
        )

        file.write(
            f"Number of test pairs: {len(results)}\n"
        )

        file.write(
            f"Diffusion timesteps: {config.TIMESTEPS}\n"
        )

        file.write(
            "Model: Conditional U-Net\n"
        )

        file.write(
            "Checkpoint: best_model.pt\n\n"
        )

        file.write(
            "LOW-LIGHT BASELINE\n"
        )

        file.write(
            f"Mean PSNR: "
            f"{low_psnr_values.mean():.4f} dB\n"
        )

        file.write(
            f"Median PSNR: "
            f"{np.median(low_psnr_values):.4f} dB\n"
        )

        file.write(
            f"Mean SSIM: "
            f"{low_ssim_values.mean():.6f}\n"
        )

        file.write(
            f"Median SSIM: "
            f"{np.median(low_ssim_values):.6f}\n\n"
        )

        file.write(
            "DIFFUSION MODEL\n"
        )

        file.write(
            f"Mean PSNR: "
            f"{generated_psnr_values.mean():.4f} dB\n"
        )

        file.write(
            f"Median PSNR: "
            f"{np.median(generated_psnr_values):.4f} dB\n"
        )

        file.write(
            f"Mean SSIM: "
            f"{generated_ssim_values.mean():.6f}\n"
        )

        file.write(
            f"Median SSIM: "
            f"{np.median(generated_ssim_values):.6f}\n\n"
        )

        file.write(
            "IMPROVEMENT\n"
        )

        file.write(
            f"Mean PSNR improvement: "
            f"{psnr_improvement_values.mean():+.4f} dB\n"
        )

        file.write(
            f"Median PSNR improvement: "
            f"{np.median(psnr_improvement_values):+.4f} dB\n"
        )

        file.write(
            f"Mean SSIM improvement: "
            f"{ssim_improvement_values.mean():+.6f}\n"
        )

        file.write(
            f"Median SSIM improvement: "
            f"{np.median(ssim_improvement_values):+.6f}\n\n"
        )

        file.write(
            "OUTPERFORMANCE COUNT\n"
        )

        file.write(
            f"Images where PSNR improved: "
            f"{np.sum(psnr_improvement_values > 0)}"
            f"/{len(results)}\n"
        )

        file.write(
            f"Images where SSIM improved: "
            f"{np.sum(ssim_improvement_values > 0)}"
            f"/{len(results)}\n"
        )

    # ---------------------------------------------------------
    # Final terminal output
    # ---------------------------------------------------------

    print()

    print("=" * 60)
    print(
        "TASK 13.2 OFFICIAL TEST EVALUATION "
        "COMPLETED"
    )
    print("=" * 60)

    print()

    print("LOW-LIGHT BASELINE")

    print(
        f"Mean PSNR: "
        f"{low_psnr_values.mean():.4f} dB"
    )

    print(
        f"Mean SSIM: "
        f"{low_ssim_values.mean():.6f}"
    )

    print()

    print("DIFFUSION MODEL")

    print(
        f"Mean PSNR: "
        f"{generated_psnr_values.mean():.4f} dB"
    )

    print(
        f"Mean SSIM: "
        f"{generated_ssim_values.mean():.6f}"
    )

    print()

    print("IMPROVEMENT")

    print(
        f"Mean PSNR change: "
        f"{psnr_improvement_values.mean():+.4f} dB"
    )

    print(
        f"Mean SSIM change: "
        f"{ssim_improvement_values.mean():+.6f}"
    )

    print()

    print(
        f"PSNR improved on "
        f"{np.sum(psnr_improvement_values > 0)}"
        f"/{len(results)} images"
    )

    print(
        f"SSIM improved on "
        f"{np.sum(ssim_improvement_values > 0)}"
        f"/{len(results)} images"
    )

    print()

    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(
        f"Visual results: "
        f"{visual_output_dir}"
    )


if __name__ == "__main__":
    main()