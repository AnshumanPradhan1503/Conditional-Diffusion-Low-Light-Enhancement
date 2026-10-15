from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from src.configs import training_config as config
from src.diffusion.reverse import reverse_diffusion
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.preprocessing.dataloader import create_dataloaders
from src.preprocessing.normalization import tensor_to_image


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# IMAGE CONVERSION
# ============================================================

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor in [-1, 1] to
    a NumPy RGB image in [0, 1].
    """

    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)

    image = tensor.squeeze(0)
    image = image.permute(1, 2, 0)

    return image.numpy().astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0,
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=2,
        data_range=1.0,
    )

    return float(psnr), float(ssim)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ConditionalUNet:

    model = ConditionalUNet(
        image_channels=config.IMAGE_CHANNELS,
        condition_channels=config.CONDITION_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
    ).to(device)

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


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_image(
    model: ConditionalUNet,
    low_light: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:

    shape = low_light.shape

    def predicted_noise_fn(
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        return model(
            noisy_image=x_t,
            condition=condition,
            timesteps=timesteps,
        )

    generated = reverse_diffusion(
        shape=shape,
        condition=low_light,
        predicted_noise_fn=predicted_noise_fn,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=generator,
    )

    return generated


# ============================================================
# SAVE CSV
# ============================================================

def save_results(
    csv_path: Path,
    results: list[dict],
) -> None:

    fieldnames = [
        "index",
        "low_file",
        "high_file",
        "low_psnr",
        "low_ssim",
        "original_psnr",
        "original_ssim",
        "x0_psnr",
        "x0_ssim",
        "original_psnr_improvement",
        "original_ssim_improvement",
        "x0_psnr_improvement",
        "x0_ssim_improvement",
    ]

    with csv_path.open(
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


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_existing_results(
    csv_path: Path,
) -> list[dict]:

    if not csv_path.exists():
        return []

    results = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            results.append(row)

    return results


# ============================================================
# SAVE VISUAL RESULT
# ============================================================

def save_visual_results(
    output_dir: Path,
    index: int,
    low_light: torch.Tensor,
    original_generated: torch.Tensor,
    objective_x0_generated: torch.Tensor,
    normal_light: torch.Tensor,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    low_image = tensor_to_image(
        low_light.squeeze(0).cpu()
    )

    original_image = tensor_to_image(
        original_generated.squeeze(0).cpu()
    )

    x0_image = tensor_to_image(
        objective_x0_generated.squeeze(0).cpu()
    )

    reference_image = tensor_to_image(
        normal_light.squeeze(0).cpu()
    )

    low_image.save(
        output_dir / f"{index:03d}_low.png"
    )

    original_image.save(
        output_dir / f"{index:03d}_original.png"
    )

    x0_image.save(
        output_dir / f"{index:03d}_x0.png"
    )

    reference_image.save(
        output_dir / f"{index:03d}_reference.png"
    )


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    summary_path: Path,
    results: list[dict],
) -> None:

    if not results:
        return

    low_psnr = np.array(
        [float(r["low_psnr"]) for r in results]
    )

    low_ssim = np.array(
        [float(r["low_ssim"]) for r in results]
    )

    original_psnr = np.array(
        [float(r["original_psnr"]) for r in results]
    )

    original_ssim = np.array(
        [float(r["original_ssim"]) for r in results]
    )

    x0_psnr = np.array(
        [float(r["x0_psnr"]) for r in results]
    )

    x0_ssim = np.array(
        [float(r["x0_ssim"]) for r in results]
    )

    original_psnr_improvement = (
        original_psnr - low_psnr
    )

    original_ssim_improvement = (
        original_ssim - low_ssim
    )

    x0_psnr_improvement = (
        x0_psnr - low_psnr
    )

    x0_ssim_improvement = (
        x0_ssim - low_ssim
    )

    original_psnr_wins = np.sum(
        original_psnr > x0_psnr
    )

    x0_psnr_wins = np.sum(
        x0_psnr > original_psnr
    )

    original_ssim_wins = np.sum(
        original_ssim > x0_ssim
    )

    x0_ssim_wins = np.sum(
        x0_ssim > original_ssim
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "TASK 15.5.6 — CONTROLLED FULL EVALUATION\n"
        )
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Images evaluated: {len(results)}\n"
        )

        file.write(
            f"Random seed: {SEED}\n"
        )

        file.write(
            f"Diffusion steps: {config.TIMESTEPS}\n"
        )

        file.write(
            f"Image size: {config.IMAGE_SIZE}\n\n"
        )

        file.write(
            "LOW-LIGHT BASELINE\n"
        )
        file.write(
            f"Mean PSNR: {low_psnr.mean():.6f} dB\n"
        )
        file.write(
            f"Mean SSIM: {low_ssim.mean():.6f}\n\n"
        )

        file.write(
            "ORIGINAL 32-CHANNEL MODEL\n"
        )
        file.write(
            f"Mean PSNR: {original_psnr.mean():.6f} dB\n"
        )
        file.write(
            f"Mean SSIM: {original_ssim.mean():.6f}\n"
        )
        file.write(
            f"PSNR improvement: "
            f"{original_psnr_improvement.mean():+.6f} dB\n"
        )
        file.write(
            f"SSIM improvement: "
            f"{original_ssim_improvement.mean():+.6f}\n"
        )
        file.write(
            f"PSNR improved on "
            f"{np.sum(original_psnr_improvement > 0)}"
            f"/{len(results)} images\n"
        )
        file.write(
            f"SSIM improved on "
            f"{np.sum(original_ssim_improvement > 0)}"
            f"/{len(results)} images\n\n"
        )

        file.write(
            "X0-OBJECTIVE MODEL\n"
        )
        file.write(
            f"Mean PSNR: {x0_psnr.mean():.6f} dB\n"
        )
        file.write(
            f"Mean SSIM: {x0_ssim.mean():.6f}\n"
        )
        file.write(
            f"PSNR improvement: "
            f"{x0_psnr_improvement.mean():+.6f} dB\n"
        )
        file.write(
            f"SSIM improvement: "
            f"{x0_ssim_improvement.mean():+.6f}\n"
        )
        file.write(
            f"PSNR improved on "
            f"{np.sum(x0_psnr_improvement > 0)}"
            f"/{len(results)} images\n"
        )
        file.write(
            f"SSIM improved on "
            f"{np.sum(x0_ssim_improvement > 0)}"
            f"/{len(results)} images\n\n"
        )

        file.write(
            "DIRECT MODEL COMPARISON\n"
        )
        file.write(
            f"Original PSNR wins: "
            f"{original_psnr_wins}/{len(results)}\n"
        )
        file.write(
            f"X0 PSNR wins: "
            f"{x0_psnr_wins}/{len(results)}\n"
        )
        file.write(
            f"Original SSIM wins: "
            f"{original_ssim_wins}/{len(results)}\n"
        )
        file.write(
            f"X0 SSIM wins: "
            f"{x0_ssim_wins}/{len(results)}\n"
        )

        file.write("\n")
        file.write("=" * 60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print("TASK 15.5.6 — CONTROLLED FULL 100-IMAGE EVALUATION")
    print("=" * 60)

    device = torch.device(
        config.DEVICE
    )

    print(f"Device: {device}")
    print(f"Random seed: {SEED}")

    set_seed(SEED)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_root = (
        config.PROJECT_ROOT
        / "results"
        / "objective_comparison"
    )

    visual_dir = (
        output_root
        / "visual_results"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_root
        / "test_metrics.csv"
    )

    summary_path = (
        output_root
        / "summary.txt"
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
        f"Diffusion steps: {config.TIMESTEPS}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    test_dataset = test_loader.dataset

    total_images = len(test_dataset)

    print(
        f"Official test images: {total_images}"
    )

    if total_images != 100:
        raise RuntimeError(
            f"Expected 100 official test images, "
            f"but found {total_images}."
        )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    original_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "best_model.pt"
    )

    objective_x0_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "objective_x0"
        / "best_model.pt"
    )

    if not original_checkpoint.exists():
        raise FileNotFoundError(
            f"Original checkpoint not found:\n"
            f"{original_checkpoint}"
        )

    if not objective_x0_checkpoint.exists():
        raise FileNotFoundError(
            f"X0 checkpoint not found:\n"
            f"{objective_x0_checkpoint}"
        )

    print()
    print("Loading original 32-channel model...")

    original_model = load_model(
        original_checkpoint,
        device,
    )

    print("Original model loaded.")

    print()
    print("Loading x0-objective model...")

    objective_x0_model = load_model(
        objective_x0_checkpoint,
        device,
    )

    print("X0-objective model loaded.")

    # --------------------------------------------------------
    # Resume existing results
    # --------------------------------------------------------

    existing_results = load_existing_results(
        csv_path
    )

    completed_indices = {
        int(row["index"])
        for row in existing_results
    }

    results = existing_results.copy()

    if completed_indices:

        print()
        print(
            f"Existing results found: "
            f"{len(completed_indices)}"
        )

        print(
            "Evaluation will resume automatically."
        )

    else:

        print()
        print(
            "No previous results found."
        )

        print(
            "Starting evaluation from image 0."
        )

    # --------------------------------------------------------
    # Evaluate all test images
    # --------------------------------------------------------

    for index in range(total_images):

        if index in completed_indices:

            print(
                f"[{index + 1}/{total_images}] "
                f"Skipping already completed image."
            )

            continue

        print()
        print("-" * 60)
        print(
            f"[{index + 1}/{total_images}] "
            f"Processing test image"
        )

        # ----------------------------------------------------
        # Load paired image
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

        # ----------------------------------------------------
        # Get filenames
        # ----------------------------------------------------

        try:

            low_file = test_dataset.low_files[index]
            high_file = test_dataset.high_files[index]

            low_file = Path(
                low_file
            ).name

            high_file = Path(
                high_file
            ).name

        except AttributeError:

            low_file = f"index_{index:03d}"
            high_file = f"index_{index:03d}"

        print(
            f"Low : {low_file}"
        )

        print(
            f"High: {high_file}"
        )

        # ----------------------------------------------------
        # Baseline metrics
        # ----------------------------------------------------

        low_np = tensor_to_numpy(
            low_light
        )

        reference_np = tensor_to_numpy(
            normal_light
        )

        low_psnr, low_ssim = (
            calculate_metrics(
                low_np,
                reference_np,
            )
        )

        print(
            f"Baseline PSNR: "
            f"{low_psnr:.4f} dB"
        )

        print(
            f"Baseline SSIM: "
            f"{low_ssim:.4f}"
        )

        # ----------------------------------------------------
        # Original model
        #
        # IMPORTANT:
        # New generator is created for every image.
        # ----------------------------------------------------

        print(
            "Generating original model..."
        )

        original_generator = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        original_generated = (
            generate_image(
                model=original_model,
                low_light=low_light,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                device=device,
                generator=original_generator,
            )
        )

        print(
            "Original generation completed."
        )

        # ----------------------------------------------------
        # X0 model
        #
        # IMPORTANT:
        # Separate generator with SAME seed.
        # Therefore both models receive identical
        # initial/random diffusion noise.
        # ----------------------------------------------------

        print(
            "Generating x0-objective model..."
        )

        objective_x0_generator = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        objective_x0_generated = (
            generate_image(
                model=objective_x0_model,
                low_light=low_light,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                device=device,
                generator=objective_x0_generator,
            )
        )

        print(
            "X0 generation completed."
        )

        # ----------------------------------------------------
        # Convert generated outputs
        # ----------------------------------------------------

        original_np = tensor_to_numpy(
            original_generated
        )

        objective_x0_np = tensor_to_numpy(
            objective_x0_generated
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        original_psnr, original_ssim = (
            calculate_metrics(
                original_np,
                reference_np,
            )
        )

        objective_x0_psnr, objective_x0_ssim = (
            calculate_metrics(
                objective_x0_np,
                reference_np,
            )
        )

        # ----------------------------------------------------
        # Improvements
        # ----------------------------------------------------

        original_psnr_improvement = (
            original_psnr - low_psnr
        )

        original_ssim_improvement = (
            original_ssim - low_ssim
        )

        x0_psnr_improvement = (
            objective_x0_psnr - low_psnr
        )

        x0_ssim_improvement = (
            objective_x0_ssim - low_ssim
        )

        # ----------------------------------------------------
        # Print image results
        # ----------------------------------------------------

        print()
        print(
            "ORIGINAL MODEL"
        )

        print(
            f"PSNR: "
            f"{original_psnr:.4f} dB"
        )

        print(
            f"SSIM: "
            f"{original_ssim:.4f}"
        )

        print(
            f"PSNR change: "
            f"{original_psnr_improvement:+.4f} dB"
        )

        print(
            f"SSIM change: "
            f"{original_ssim_improvement:+.4f}"
        )

        print()
        print(
            "X0-OBJECTIVE MODEL"
        )

        print(
            f"PSNR: "
            f"{objective_x0_psnr:.4f} dB"
        )

        print(
            f"SSIM: "
            f"{objective_x0_ssim:.4f}"
        )

        print(
            f"PSNR change: "
            f"{x0_psnr_improvement:+.4f} dB"
        )

        print(
            f"SSIM change: "
            f"{x0_ssim_improvement:+.4f}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "index": index,
            "low_file": low_file,
            "high_file": high_file,
            "low_psnr": f"{low_psnr:.6f}",
            "low_ssim": f"{low_ssim:.6f}",
            "original_psnr": f"{original_psnr:.6f}",
            "original_ssim": f"{original_ssim:.6f}",
            "x0_psnr": f"{objective_x0_psnr:.6f}",
            "x0_ssim": f"{objective_x0_ssim:.6f}",
            "original_psnr_improvement": (
                f"{original_psnr_improvement:.6f}"
            ),
            "original_ssim_improvement": (
                f"{original_ssim_improvement:.6f}"
            ),
            "x0_psnr_improvement": (
                f"{x0_psnr_improvement:.6f}"
            ),
            "x0_ssim_improvement": (
                f"{x0_ssim_improvement:.6f}"
            ),
        }

        results.append(result)

        # ----------------------------------------------------
        # Save CSV immediately
        # ----------------------------------------------------

        save_results(
            csv_path,
            results,
        )

        # ----------------------------------------------------
        # Save visual outputs
        # ----------------------------------------------------

        save_visual_results(
            output_dir=visual_dir,
            index=index,
            low_light=low_light,
            original_generated=original_generated,
            objective_x0_generated=objective_x0_generated,
            normal_light=normal_light,
        )

        # ----------------------------------------------------
        # Save current summary
        # ----------------------------------------------------

        write_summary(
            summary_path,
            results,
        )

        print()
        print(
            f"[SAVED] Image {index + 1}/"
            f"{total_images}"
        )

        print(
            f"[SAVED] CSV: {csv_path}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FULL EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"Images evaluated: "
        f"{len(results)}/{total_images}"
    )

    print()
    print(
        f"Metrics CSV:\n{csv_path}"
    )

    print()
    print(
        f"Summary:\n{summary_path}"
    )

    print()
    print(
        f"Visual results:\n{visual_dir}"
    )

    # --------------------------------------------------------
    # Final numbers
    # --------------------------------------------------------

    low_psnr = np.array(
        [float(r["low_psnr"]) for r in results]
    )

    low_ssim = np.array(
        [float(r["low_ssim"]) for r in results]
    )

    original_psnr = np.array(
        [float(r["original_psnr"]) for r in results]
    )

    original_ssim = np.array(
        [float(r["original_ssim"]) for r in results]
    )

    x0_psnr = np.array(
        [float(r["x0_psnr"]) for r in results]
    )

    x0_ssim = np.array(
        [float(r["x0_ssim"]) for r in results]
    )

    print()
    print("LOW-LIGHT BASELINE")
    print(
        f"Mean PSNR: "
        f"{low_psnr.mean():.6f} dB"
    )
    print(
        f"Mean SSIM: "
        f"{low_ssim.mean():.6f}"
    )

    print()
    print("ORIGINAL 32-CHANNEL MODEL")
    print(
        f"Mean PSNR: "
        f"{original_psnr.mean():.6f} dB"
    )
    print(
        f"Mean SSIM: "
        f"{original_ssim.mean():.6f}"
    )

    print()
    print("X0-OBJECTIVE MODEL")
    print(
        f"Mean PSNR: "
        f"{x0_psnr.mean():.6f} dB"
    )
    print(
        f"Mean SSIM: "
        f"{x0_ssim.mean():.6f}"
    )

    print()
    print("ORIGINAL VS X0")

    print(
        f"Original PSNR wins: "
        f"{np.sum(original_psnr > x0_psnr)}/"
        f"{len(results)}"
    )

    print(
        f"X0 PSNR wins: "
        f"{np.sum(x0_psnr > original_psnr)}/"
        f"{len(results)}"
    )

    print(
        f"Original SSIM wins: "
        f"{np.sum(original_ssim > x0_ssim)}/"
        f"{len(results)}"
    )

    print(
        f"X0 SSIM wins: "
        f"{np.sum(x0_ssim > original_ssim)}/"
        f"{len(results)}"
    )

    print()
    print("=" * 60)
    print("TASK 15.5.6 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()