from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.configs import training_config as config
from src.diffusion.reverse import reverse_diffusion
from src.diffusion.schedule import get_diffusion_parameters
from src.models.unet import ConditionalUNet
from src.models.spatial_unet import SpatialConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


# ============================================================
# FIXED EXPERIMENT SETTINGS
# ============================================================

SEED = 42

# Frozen from validation.
# Do NOT tune gamma on the official test set.
GAMMA = 0.4


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TENSOR -> NUMPY IMAGE
# ============================================================

def tensor_to_numpy(
    tensor: torch.Tensor,
) -> np.ndarray:

    tensor = tensor.detach().cpu()

    # [-1, 1] -> [0, 1]
    tensor = (tensor + 1.0) / 2.0

    tensor = tensor.clamp(
        0.0,
        1.0,
    )

    image = tensor.squeeze(0).permute(
        1,
        2,
        0,
    )

    return image.numpy().astype(
        np.float32
    )


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
# GAMMA BASELINE
# ============================================================

def gamma_correct(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:

    return np.clip(
        np.power(image, gamma),
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# LOAD ORIGINAL CONTROL MODEL
# ============================================================

def load_original(
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
# LOAD SPATIAL MODEL
# ============================================================

def load_spatial(
    checkpoint_path: Path,
    device: torch.device,
) -> SpatialConditionalUNet:

    model = SpatialConditionalUNet(
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
# DIFFUSION IMAGE GENERATION
# ============================================================

@torch.no_grad()
def generate_image(
    model,
    low_light: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:

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

    return reverse_diffusion(
        shape=low_light.shape,
        condition=low_light,
        predicted_noise_fn=predicted_noise_fn,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=device,
        generator=generator,
    )


# ============================================================
# SAVE CSV RESULTS
# ============================================================

def save_results(
    path: Path,
    results: list[dict],
) -> None:

    fields = [
        "index",
        "low_file",
        "high_file",

        "low_psnr",
        "low_ssim",

        "gamma_psnr",
        "gamma_ssim",

        "original_psnr",
        "original_ssim",

        "spatial_psnr",
        "spatial_ssim",

        "original_psnr_change",
        "original_ssim_change",

        "spatial_psnr_change",
        "spatial_ssim_change",

        "spatial_vs_original_psnr",
        "spatial_vs_original_ssim",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_results(
    path: Path,
) -> list[dict]:

    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# SAVE VISUAL RESULT
# ============================================================

def save_visual(
    path: Path,
    image: np.ndarray,
) -> None:

    from PIL import Image

    Image.fromarray(
        np.clip(
            image * 255.0,
            0,
            255,
        ).astype(np.uint8),
        mode="RGB",
    ).save(path)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "TASK 22.6.5 — OFFICIAL SPATIAL "
        "CONDITIONING CONTROLLED EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE / SETTINGS
    # --------------------------------------------------------

    device = torch.device(
        config.DEVICE
    )

    set_seed(SEED)

    print(
        f"Device: {device}"
    )

    print(
        f"Random seed: {SEED}"
    )

    print(
        f"Diffusion steps: {config.TIMESTEPS}"
    )

    print(
        f"Image size: {config.IMAGE_SIZE}"
    )

    print(
        f"Frozen gamma baseline: {GAMMA}"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    output_root = (
        config.PROJECT_ROOT
        / "results"
        / "spatial_evaluation"
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
    # CHECKPOINTS
    # --------------------------------------------------------

    original_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "best_model.pt"
    )

    spatial_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "spatial"
        / "best_model.pt"
    )

    for checkpoint in (
        original_checkpoint,
        spatial_checkpoint,
    ):

        if not checkpoint.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{checkpoint}"
            )

    # --------------------------------------------------------
    # DIFFUSION SCHEDULE
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

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    _, _, test_loader = (
        create_dataloaders(
            data_root=config.DATA_ROOT,
            split_root=config.SPLIT_ROOT,
        )
    )

    test_dataset = (
        test_loader.dataset
    )

    if not hasattr(
        test_dataset,
        "pairs",
    ):

        raise AttributeError(
            "LOLv2PairedDataset must expose "
            "paired filenames through "
            "test_dataset.pairs."
        )

    total_images = len(
        test_dataset
    )

    if total_images != 100:

        raise RuntimeError(
            f"Expected 100 official test "
            f"images, found {total_images}."
        )

    print(
        f"Official test images: "
        f"{total_images}"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "Loading original 32-channel control..."
    )

    original_model = load_original(
        original_checkpoint,
        device,
    )

    print(
        "Loading spatial-conditioning model..."
    )

    spatial_model = load_spatial(
        spatial_checkpoint,
        device,
    )

    print(
        "Both models loaded successfully."
    )

    print(
        "Original parameters: "
        f"{sum(p.numel() for p in original_model.parameters()):,}"
    )

    print(
        "Spatial parameters: "
        f"{sum(p.numel() for p in spatial_model.parameters()):,}"
    )

    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    results = load_results(
        csv_path
    )

    completed = {
        int(row["index"])
        for row in results
    }

    if completed:

        print(
            f"Resuming: "
            f"{len(completed)}/100 images already completed."
        )

    # --------------------------------------------------------
    # EVALUATE EACH TEST IMAGE
    # --------------------------------------------------------

    for index in range(
        total_images
    ):

        if index in completed:

            print(
                f"[{index + 1}/100] "
                f"Skipping completed image."
            )

            continue

        # ----------------------------------------------------
        # LOAD TEST PAIR
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

        low_name = Path(
            test_dataset.pairs[index][0]
        ).name

        high_name = Path(
            test_dataset.pairs[index][1]
        ).name

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        low_np = tensor_to_numpy(
            low_light
        )

        reference_np = tensor_to_numpy(
            normal_light
        )

        # ----------------------------------------------------
        # LOW-LIGHT BASELINE
        # ----------------------------------------------------

        low_psnr, low_ssim = (
            calculate_metrics(
                low_np,
                reference_np,
            )
        )

        # ----------------------------------------------------
        # GAMMA BASELINE
        # ----------------------------------------------------

        gamma_np = gamma_correct(
            low_np,
            GAMMA,
        )

        gamma_psnr, gamma_ssim = (
            calculate_metrics(
                gamma_np,
                reference_np,
            )
        )

        # ----------------------------------------------------
        # ORIGINAL MODEL
        # ----------------------------------------------------

        print(
            f"[{index + 1}/100] "
            f"{low_name} — generating original..."
        )

        original_gen = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        original_generated = (
            generate_image(
                original_model,
                low_light,
                betas,
                alphas,
                alpha_bars,
                device,
                original_gen,
            )
        )

        # ----------------------------------------------------
        # SPATIAL MODEL
        # ----------------------------------------------------

        print(
            f"[{index + 1}/100] "
            f"{low_name} — generating spatial..."
        )

        spatial_gen = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        spatial_generated = (
            generate_image(
                spatial_model,
                low_light,
                betas,
                alphas,
                alpha_bars,
                device,
                spatial_gen,
            )
        )

        # ----------------------------------------------------
        # CONVERT GENERATED IMAGES
        # ----------------------------------------------------

        original_np = (
            tensor_to_numpy(
                original_generated
            )
        )

        spatial_np = (
            tensor_to_numpy(
                spatial_generated
            )
        )

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        original_psnr, original_ssim = (
            calculate_metrics(
                original_np,
                reference_np,
            )
        )

        spatial_psnr, spatial_ssim = (
            calculate_metrics(
                spatial_np,
                reference_np,
            )
        )

        # ----------------------------------------------------
        # RESULT ROW
        # ----------------------------------------------------

        row = {

            "index": index,

            "low_file": low_name,

            "high_file": high_name,

            "low_psnr": f"{low_psnr:.6f}",

            "low_ssim": f"{low_ssim:.6f}",

            "gamma_psnr": f"{gamma_psnr:.6f}",

            "gamma_ssim": f"{gamma_ssim:.6f}",

            "original_psnr": f"{original_psnr:.6f}",

            "original_ssim": f"{original_ssim:.6f}",

            "spatial_psnr": f"{spatial_psnr:.6f}",

            "spatial_ssim": f"{spatial_ssim:.6f}",

            "original_psnr_change":
                f"{original_psnr - low_psnr:.6f}",

            "original_ssim_change":
                f"{original_ssim - low_ssim:.6f}",

            "spatial_psnr_change":
                f"{spatial_psnr - low_psnr:.6f}",

            "spatial_ssim_change":
                f"{spatial_ssim - low_ssim:.6f}",

            "spatial_vs_original_psnr":
                f"{spatial_psnr - original_psnr:.6f}",

            "spatial_vs_original_ssim":
                f"{spatial_ssim - original_ssim:.6f}",
        }

        results.append(row)

        results.sort(
            key=lambda r: int(
                r["index"]
            )
        )

        # Save after every image.
        save_results(
            csv_path,
            results,
        )

        # ----------------------------------------------------
        # SAVE VISUAL RESULTS
        # ----------------------------------------------------

        save_visual(
            visual_dir
            / f"{index:03d}_low.png",
            low_np,
        )

        save_visual(
            visual_dir
            / f"{index:03d}_gamma.png",
            gamma_np,
        )

        save_visual(
            visual_dir
            / f"{index:03d}_original.png",
            original_np,
        )

        save_visual(
            visual_dir
            / f"{index:03d}_spatial.png",
            spatial_np,
        )

        save_visual(
            visual_dir
            / f"{index:03d}_reference.png",
            reference_np,
        )

        # ----------------------------------------------------
        # PRINT IMAGE RESULTS
        # ----------------------------------------------------

        print(
            f"  Baseline : "
            f"PSNR {low_psnr:.4f} | "
            f"SSIM {low_ssim:.6f}\n"

            f"  Gamma    : "
            f"PSNR {gamma_psnr:.4f} | "
            f"SSIM {gamma_ssim:.6f}\n"

            f"  Original : "
            f"PSNR {original_psnr:.4f} | "
            f"SSIM {original_ssim:.6f}\n"

            f"  Spatial  : "
            f"PSNR {spatial_psnr:.4f} | "
            f"SSIM {spatial_ssim:.6f}\n"

            f"  Spatial-Original: "
            f"PSNR "
            f"{spatial_psnr - original_psnr:+.4f} | "
            f"SSIM "
            f"{spatial_ssim - original_ssim:+.6f}"
        )

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        del (
            low_light,
            normal_light,
            original_generated,
            spatial_generated,
        )

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(results) != 100:

        raise RuntimeError(
            f"Expected 100 completed "
            f"results, found {len(results)}."
        )

    # ========================================================
    # ARRAYS
    # ========================================================

    arrays = {

        key: np.array(
            [
                float(row[key])
                for row in results
            ]
        )

        for key in [

            "low_psnr",
            "low_ssim",

            "gamma_psnr",
            "gamma_ssim",

            "original_psnr",
            "original_ssim",

            "spatial_psnr",
            "spatial_ssim",
        ]
    }

    def mean(
        key: str,
    ) -> float:

        return float(
            np.mean(
                arrays[key]
            )
        )

    def median(
        key: str,
    ) -> float:

        return float(
            np.median(
                arrays[key]
            )
        )

    # ========================================================
    # DELTAS
    # ========================================================

    original_psnr_delta = (
        arrays["original_psnr"]
        - arrays["low_psnr"]
    )

    original_ssim_delta = (
        arrays["original_ssim"]
        - arrays["low_ssim"]
    )

    spatial_psnr_delta = (
        arrays["spatial_psnr"]
        - arrays["low_psnr"]
    )

    spatial_ssim_delta = (
        arrays["spatial_ssim"]
        - arrays["low_ssim"]
    )

    spatial_original_psnr = (
        arrays["spatial_psnr"]
        - arrays["original_psnr"]
    )

    spatial_original_ssim = (
        arrays["spatial_ssim"]
        - arrays["original_ssim"]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""
TASK 22.6.5 — OFFICIAL SPATIAL CONDITIONING
CONTROLLED EVALUATION

Dataset: LOLv2-real official test set
Test images: 100
Random seed: {SEED}
Diffusion steps: {config.TIMESTEPS}
Image size: {config.IMAGE_SIZE}
Frozen gamma: {GAMMA}


MODEL PARAMETERS

Original 32-channel:
{sum(p.numel() for p in original_model.parameters()):,}

Spatial conditioning:
{sum(p.numel() for p in spatial_model.parameters()):,}


LOW-LIGHT BASELINE

Mean PSNR:
{mean("low_psnr"):.6f} dB

Mean SSIM:
{mean("low_ssim"):.6f}


GAMMA BASELINE (gamma={GAMMA})

Mean PSNR:
{mean("gamma_psnr"):.6f} dB

Mean SSIM:
{mean("gamma_ssim"):.6f}


ORIGINAL 32-CHANNEL CONTROL

Mean PSNR:
{mean("original_psnr"):.6f} dB

Median PSNR:
{median("original_psnr"):.6f} dB

Mean SSIM:
{mean("original_ssim"):.6f}

Median SSIM:
{median("original_ssim"):.6f}

PSNR improvement vs low-light:
{np.mean(original_psnr_delta):+.6f} dB

SSIM improvement vs low-light:
{np.mean(original_ssim_delta):+.6f}

PSNR improved vs low-light:
{int(np.sum(original_psnr_delta > 0))}/100

SSIM improved vs low-light:
{int(np.sum(original_ssim_delta > 0))}/100


SPATIAL CONDITIONING MODEL

Mean PSNR:
{mean("spatial_psnr"):.6f} dB

Median PSNR:
{median("spatial_psnr"):.6f} dB

Mean SSIM:
{mean("spatial_ssim"):.6f}

Median SSIM:
{median("spatial_ssim"):.6f}

PSNR improvement vs low-light:
{np.mean(spatial_psnr_delta):+.6f} dB

SSIM improvement vs low-light:
{np.mean(spatial_ssim_delta):+.6f}

PSNR improved vs low-light:
{int(np.sum(spatial_psnr_delta > 0))}/100

SSIM improved vs low-light:
{int(np.sum(spatial_ssim_delta > 0))}/100


SPATIAL CONDITIONING VS ORIGINAL CONTROL

Mean PSNR difference:
{np.mean(spatial_original_psnr):+.6f} dB

Mean SSIM difference:
{np.mean(spatial_original_ssim):+.6f}

Spatial PSNR wins:
{int(np.sum(spatial_original_psnr > 0))}/100

Original PSNR wins:
{int(np.sum(spatial_original_psnr < 0))}/100

Tied PSNR:
{int(np.sum(spatial_original_psnr == 0))}/100

Spatial SSIM wins:
{int(np.sum(spatial_original_ssim > 0))}/100

Original SSIM wins:
{int(np.sum(spatial_original_ssim < 0))}/100

Tied SSIM:
{int(np.sum(spatial_original_ssim == 0))}/100


OUTPUTS

CSV:
{csv_path}

Visual results:
{visual_dir}
"""

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL TERMINAL OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK 22.6.5 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Original : "
        f"{mean('original_psnr'):.4f} dB | "
        f"{mean('original_ssim'):.6f}"
    )

    print(
        f"Spatial  : "
        f"{mean('spatial_psnr'):.4f} dB | "
        f"{mean('spatial_ssim'):.6f}"
    )

    print(
        f"Gamma    : "
        f"{mean('gamma_psnr'):.4f} dB | "
        f"{mean('gamma_ssim'):.6f}"
    )

    print(
        f"Spatial-Original: "
        f"{np.mean(spatial_original_psnr):+.4f} dB | "
        f"{np.mean(spatial_original_ssim):+.6f}"
    )

    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()