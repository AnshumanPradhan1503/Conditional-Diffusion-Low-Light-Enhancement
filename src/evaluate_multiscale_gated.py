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
from src.models.multiscale_unet import MultiScaleConditionalUNet
from src.models.multiscale_gated_unet import MultiScaleGatedConditionalUNet
from src.preprocessing.dataloader import create_dataloaders


SEED = 42
GAMMA = 0.4  # Frozen from validation; do not tune on test set.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)

    image = tensor.squeeze(0).permute(1, 2, 0)

    return image.numpy().astype(np.float32)


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


def gamma_correct(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:

    return np.clip(
        np.power(image, gamma),
        0.0,
        1.0,
    ).astype(np.float32)


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


def load_multiscale(
    checkpoint_path: Path,
    device: torch.device,
) -> MultiScaleConditionalUNet:

    model = MultiScaleConditionalUNet(
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


def load_multiscale_gated(
    checkpoint_path: Path,
    device: torch.device,
) -> MultiScaleGatedConditionalUNet:

    model = MultiScaleGatedConditionalUNet(
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


@torch.no_grad()
def generate_image(
    model,
    low_light,
    betas,
    alphas,
    alpha_bars,
    device,
    generator,
):

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

        "multiscale_psnr",
        "multiscale_ssim",

        "gated_psnr",
        "gated_ssim",

        "original_psnr_change",
        "original_ssim_change",

        "multiscale_psnr_change",
        "multiscale_ssim_change",

        "gated_psnr_change",
        "gated_ssim_change",

        "multiscale_vs_original_psnr",
        "multiscale_vs_original_ssim",

        "gated_vs_original_psnr",
        "gated_vs_original_ssim",

        "gated_vs_multiscale_psnr",
        "gated_vs_multiscale_ssim",
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


def load_results(path: Path) -> list[dict]:

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


def main() -> None:

    print("=" * 75)
    print("TASK 21.6 — OFFICIAL GATED MULTI-SCALE CONTROLLED EVALUATION")
    print("=" * 75)

    device = torch.device(config.DEVICE)

    set_seed(SEED)

    print(f"Device: {device}")
    print(f"Random seed: {SEED}")
    print(f"Diffusion steps: {config.TIMESTEPS}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Frozen gamma baseline: {GAMMA}")

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------

    output_root = (
        config.PROJECT_ROOT
        / "results"
        / "multiscale_gated_evaluation"
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

    # ------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------

    original_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "best_model.pt"
    )

    multiscale_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "multiscale"
        / "best_model.pt"
    )

    gated_checkpoint = (
        config.PROJECT_ROOT
        / "checkpoints"
        / "multiscale_gated"
        / "best_model.pt"
    )

    for checkpoint in (
        original_checkpoint,
        multiscale_checkpoint,
        gated_checkpoint,
    ):

        if not checkpoint.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n{checkpoint}"
            )

    # ------------------------------------------------------------
    # Diffusion parameters
    # ------------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    )

    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    # ------------------------------------------------------------
    # Test dataset
    # ------------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        split_root=config.SPLIT_ROOT,
    )

    test_dataset = test_loader.dataset

    if not hasattr(test_dataset, "pairs"):

        raise AttributeError(
            "LOLv2PairedDataset must expose "
            "paired filenames through test_dataset.pairs."
        )

    total_images = len(test_dataset)

    if total_images != 100:

        raise RuntimeError(
            f"Expected 100 official test images, "
            f"found {total_images}."
        )

    print(
        f"Official test images: {total_images}"
    )

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    print(
        "Loading original 32-channel control..."
    )

    original_model = load_original(
        original_checkpoint,
        device,
    )

    print(
        "Loading previous multi-scale model..."
    )

    multiscale_model = load_multiscale(
        multiscale_checkpoint,
        device,
    )

    print(
        "Loading gated multi-scale model..."
    )

    gated_model = load_multiscale_gated(
        gated_checkpoint,
        device,
    )

    print(
        "All three diffusion models loaded successfully."
    )

    # ------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Official evaluation loop
    # ------------------------------------------------------------

    for index in range(total_images):

        if index in completed:

            print(
                f"[{index + 1}/100] "
                f"Skipping completed image."
            )

            continue

        # --------------------------------------------------------
        # Load test pair
        # --------------------------------------------------------

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

        low_np = tensor_to_numpy(
            low_light
        )

        reference_np = tensor_to_numpy(
            normal_light
        )

        # --------------------------------------------------------
        # Low-light baseline
        # --------------------------------------------------------

        low_psnr, low_ssim = calculate_metrics(
            low_np,
            reference_np,
        )

        # --------------------------------------------------------
        # Gamma baseline
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Original diffusion model
        # --------------------------------------------------------

        print(
            f"[{index + 1}/100] "
            f"{low_name} — generating original..."
        )

        original_gen = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        original_generated = generate_image(
            original_model,
            low_light,
            betas,
            alphas,
            alpha_bars,
            device,
            original_gen,
        )

        # --------------------------------------------------------
        # Previous multi-scale diffusion model
        # --------------------------------------------------------

        print(
            f"[{index + 1}/100] "
            f"{low_name} — generating multi-scale..."
        )

        multiscale_gen = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        multiscale_generated = generate_image(
            multiscale_model,
            low_light,
            betas,
            alphas,
            alpha_bars,
            device,
            multiscale_gen,
        )

        # --------------------------------------------------------
        # Gated multi-scale diffusion model
        # --------------------------------------------------------

        print(
            f"[{index + 1}/100] "
            f"{low_name} — generating gated..."
        )

        gated_gen = (
            torch.Generator(
                device=device
            ).manual_seed(SEED)
        )

        gated_generated = generate_image(
            gated_model,
            low_light,
            betas,
            alphas,
            alpha_bars,
            device,
            gated_gen,
        )

        # --------------------------------------------------------
        # Convert generated tensors
        # --------------------------------------------------------

        original_np = tensor_to_numpy(
            original_generated
        )

        multiscale_np = tensor_to_numpy(
            multiscale_generated
        )

        gated_np = tensor_to_numpy(
            gated_generated
        )

        # --------------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------------

        original_psnr, original_ssim = (
            calculate_metrics(
                original_np,
                reference_np,
            )
        )

        multiscale_psnr, multiscale_ssim = (
            calculate_metrics(
                multiscale_np,
                reference_np,
            )
        )

        gated_psnr, gated_ssim = (
            calculate_metrics(
                gated_np,
                reference_np,
            )
        )

        # --------------------------------------------------------
        # Store result
        # --------------------------------------------------------

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

            "multiscale_psnr": (
                f"{multiscale_psnr:.6f}"
            ),

            "multiscale_ssim": (
                f"{multiscale_ssim:.6f}"
            ),

            "gated_psnr": (
                f"{gated_psnr:.6f}"
            ),

            "gated_ssim": (
                f"{gated_ssim:.6f}"
            ),

            "original_psnr_change": (
                f"{original_psnr - low_psnr:.6f}"
            ),

            "original_ssim_change": (
                f"{original_ssim - low_ssim:.6f}"
            ),

            "multiscale_psnr_change": (
                f"{multiscale_psnr - low_psnr:.6f}"
            ),

            "multiscale_ssim_change": (
                f"{multiscale_ssim - low_ssim:.6f}"
            ),

            "gated_psnr_change": (
                f"{gated_psnr - low_psnr:.6f}"
            ),

            "gated_ssim_change": (
                f"{gated_ssim - low_ssim:.6f}"
            ),

            "multiscale_vs_original_psnr": (
                f"{multiscale_psnr - original_psnr:.6f}"
            ),

            "multiscale_vs_original_ssim": (
                f"{multiscale_ssim - original_ssim:.6f}"
            ),

            "gated_vs_original_psnr": (
                f"{gated_psnr - original_psnr:.6f}"
            ),

            "gated_vs_original_ssim": (
                f"{gated_ssim - original_ssim:.6f}"
            ),

            "gated_vs_multiscale_psnr": (
                f"{gated_psnr - multiscale_psnr:.6f}"
            ),

            "gated_vs_multiscale_ssim": (
                f"{gated_ssim - multiscale_ssim:.6f}"
            ),
        }

        results.append(row)

        results.sort(
            key=lambda r: int(r["index"])
        )

        # Save after every image so evaluation
        # can resume if interrupted.
        save_results(
            csv_path,
            results,
        )

        # --------------------------------------------------------
        # Save visual results
        # --------------------------------------------------------

        save_visual(
            visual_dir / f"{index:03d}_low.png",
            low_np,
        )

        save_visual(
            visual_dir / f"{index:03d}_gamma.png",
            gamma_np,
        )

        save_visual(
            visual_dir / f"{index:03d}_original.png",
            original_np,
        )

        save_visual(
            visual_dir / f"{index:03d}_multiscale.png",
            multiscale_np,
        )

        save_visual(
            visual_dir / f"{index:03d}_gated.png",
            gated_np,
        )

        save_visual(
            visual_dir / f"{index:03d}_reference.png",
            reference_np,
        )

        # --------------------------------------------------------
        # Print image-level results
        # --------------------------------------------------------

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

            f"  Multi    : "
            f"PSNR {multiscale_psnr:.4f} | "
            f"SSIM {multiscale_ssim:.6f}\n"

            f"  Gated    : "
            f"PSNR {gated_psnr:.4f} | "
            f"SSIM {gated_ssim:.6f}\n"

            f"  Gated-Original: "
            f"PSNR "
            f"{gated_psnr - original_psnr:+.4f} | "
            f"SSIM "
            f"{gated_ssim - original_ssim:+.6f}"
        )

        # --------------------------------------------------------
        # Free memory
        # --------------------------------------------------------

        del (
            low_light,
            normal_light,
            original_generated,
            multiscale_generated,
            gated_generated,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    if len(results) != 100:

        raise RuntimeError(
            f"Expected 100 completed results, "
            f"found {len(results)}."
        )

    # ------------------------------------------------------------
    # Convert results to arrays
    # ------------------------------------------------------------

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

            "multiscale_psnr",
            "multiscale_ssim",

            "gated_psnr",
            "gated_ssim",
        ]
    }

    def mean(key):

        return float(
            np.mean(arrays[key])
        )

    def median(key):

        return float(
            np.median(arrays[key])
        )

    # ------------------------------------------------------------
    # Differences
    # ------------------------------------------------------------

    original_psnr_delta = (
        arrays["original_psnr"]
        - arrays["low_psnr"]
    )

    original_ssim_delta = (
        arrays["original_ssim"]
        - arrays["low_ssim"]
    )

    multiscale_psnr_delta = (
        arrays["multiscale_psnr"]
        - arrays["low_psnr"]
    )

    multiscale_ssim_delta = (
        arrays["multiscale_ssim"]
        - arrays["low_ssim"]
    )

    gated_psnr_delta = (
        arrays["gated_psnr"]
        - arrays["low_psnr"]
    )

    gated_ssim_delta = (
        arrays["gated_ssim"]
        - arrays["low_ssim"]
    )

    ms_orig_psnr = (
        arrays["multiscale_psnr"]
        - arrays["original_psnr"]
    )

    ms_orig_ssim = (
        arrays["multiscale_ssim"]
        - arrays["original_ssim"]
    )

    gated_orig_psnr = (
        arrays["gated_psnr"]
        - arrays["original_psnr"]
    )

    gated_orig_ssim = (
        arrays["gated_ssim"]
        - arrays["original_ssim"]
    )

    gated_ms_psnr = (
        arrays["gated_psnr"]
        - arrays["multiscale_psnr"]
    )

    gated_ms_ssim = (
        arrays["gated_ssim"]
        - arrays["multiscale_ssim"]
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = f"""
TASK 21.6 — OFFICIAL GATED MULTI-SCALE CONTROLLED EVALUATION

Dataset: LOLv2-real official test set
Test images: 100
Random seed: {SEED}
Diffusion steps: {config.TIMESTEPS}
Image size: {config.IMAGE_SIZE}
Frozen gamma: {GAMMA}


MODEL PARAMETERS

Original 32-channel:
{sum(p.numel() for p in original_model.parameters()):,}

Multi-scale:
{sum(p.numel() for p in multiscale_model.parameters()):,}

Gated multi-scale:
{sum(p.numel() for p in gated_model.parameters()):,}


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


PREVIOUS MULTI-SCALE MODEL

Mean PSNR:
{mean("multiscale_psnr"):.6f} dB

Median PSNR:
{median("multiscale_psnr"):.6f} dB

Mean SSIM:
{mean("multiscale_ssim"):.6f}

Median SSIM:
{median("multiscale_ssim"):.6f}

PSNR improvement vs low-light:
{np.mean(multiscale_psnr_delta):+.6f} dB

SSIM improvement vs low-light:
{np.mean(multiscale_ssim_delta):+.6f}

PSNR improved vs low-light:
{int(np.sum(multiscale_psnr_delta > 0))}/100

SSIM improved vs low-light:
{int(np.sum(multiscale_ssim_delta > 0))}/100


GATED MULTI-SCALE MODEL

Mean PSNR:
{mean("gated_psnr"):.6f} dB

Median PSNR:
{median("gated_psnr"):.6f} dB

Mean SSIM:
{mean("gated_ssim"):.6f}

Median SSIM:
{median("gated_ssim"):.6f}

PSNR improvement vs low-light:
{np.mean(gated_psnr_delta):+.6f} dB

SSIM improvement vs low-light:
{np.mean(gated_ssim_delta):+.6f}

PSNR improved vs low-light:
{int(np.sum(gated_psnr_delta > 0))}/100

SSIM improved vs low-light:
{int(np.sum(gated_ssim_delta > 0))}/100


MULTI-SCALE VS ORIGINAL CONTROL

Mean PSNR difference:
{np.mean(ms_orig_psnr):+.6f} dB

Mean SSIM difference:
{np.mean(ms_orig_ssim):+.6f}

Multi-scale PSNR wins:
{int(np.sum(ms_orig_psnr > 0))}/100

Original PSNR wins:
{int(np.sum(ms_orig_psnr < 0))}/100

Tied PSNR:
{int(np.sum(ms_orig_psnr == 0))}/100

Multi-scale SSIM wins:
{int(np.sum(ms_orig_ssim > 0))}/100

Original SSIM wins:
{int(np.sum(ms_orig_ssim < 0))}/100

Tied SSIM:
{int(np.sum(ms_orig_ssim == 0))}/100


GATED VS ORIGINAL CONTROL

Mean PSNR difference:
{np.mean(gated_orig_psnr):+.6f} dB

Mean SSIM difference:
{np.mean(gated_orig_ssim):+.6f}

Gated PSNR wins:
{int(np.sum(gated_orig_psnr > 0))}/100

Original PSNR wins:
{int(np.sum(gated_orig_psnr < 0))}/100

Tied PSNR:
{int(np.sum(gated_orig_psnr == 0))}/100

Gated SSIM wins:
{int(np.sum(gated_orig_ssim > 0))}/100

Original SSIM wins:
{int(np.sum(gated_orig_ssim < 0))}/100

Tied SSIM:
{int(np.sum(gated_orig_ssim == 0))}/100


GATED VS PREVIOUS MULTI-SCALE

Mean PSNR difference:
{np.mean(gated_ms_psnr):+.6f} dB

Mean SSIM difference:
{np.mean(gated_ms_ssim):+.6f}

Gated PSNR wins:
{int(np.sum(gated_ms_psnr > 0))}/100

Previous multi-scale PSNR wins:
{int(np.sum(gated_ms_psnr < 0))}/100

Tied PSNR:
{int(np.sum(gated_ms_psnr == 0))}/100

Gated SSIM wins:
{int(np.sum(gated_ms_ssim > 0))}/100

Previous multi-scale SSIM wins:
{int(np.sum(gated_ms_ssim < 0))}/100

Tied SSIM:
{int(np.sum(gated_ms_ssim == 0))}/100


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

    # ------------------------------------------------------------
    # Final terminal output
    # ------------------------------------------------------------

    print("\n" + "=" * 75)
    print("TASK 21.6 COMPLETED")
    print("=" * 75)

    print(
        f"Original   : "
        f"{mean('original_psnr'):.4f} dB | "
        f"{mean('original_ssim'):.6f}"
    )

    print(
        f"Multi-scale: "
        f"{mean('multiscale_psnr'):.4f} dB | "
        f"{mean('multiscale_ssim'):.6f}"
    )

    print(
        f"Gated      : "
        f"{mean('gated_psnr'):.4f} dB | "
        f"{mean('gated_ssim'):.6f}"
    )

    print(
        f"Gamma      : "
        f"{mean('gamma_psnr'):.4f} dB | "
        f"{mean('gamma_ssim'):.6f}"
    )

    print(
        f"Gated-Original: "
        f"{np.mean(gated_orig_psnr):+.4f} dB | "
        f"{np.mean(gated_orig_ssim):+.6f}"
    )

    print(
        f"Gated-Multi-scale: "
        f"{np.mean(gated_ms_psnr):+.4f} dB | "
        f"{np.mean(gated_ms_ssim):+.6f}"
    )

    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()