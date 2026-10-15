from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


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


def main() -> None:

    print("============================================================")
    print("PHASE 4 — TASK 13: QUANTITATIVE EVALUATION")
    print("============================================================")

    generated_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "generated"
    )

    low_path = generated_dir / "low_000.png"
    generated_path = generated_dir / "generated_000.png"
    high_path = generated_dir / "high_000.png"

    print()
    print("Evaluation image:")
    print(f"Low-light : {low_path}")
    print(f"Generated : {generated_path}")
    print(f"Reference : {high_path}")

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    for path in [low_path, generated_path, high_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required image not found: {path}"
            )

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------

    low_light = load_image(low_path)
    generated = load_image(generated_path)
    reference = load_image(high_path)

    print()
    print(f"Image shape: {reference.shape}")

    # ---------------------------------------------------------
    # Verify dimensions
    # ---------------------------------------------------------

    if (
        low_light.shape != reference.shape
        or generated.shape != reference.shape
    ):
        raise ValueError(
            "Image dimensions do not match."
        )

    # ---------------------------------------------------------
    # Baseline: low-light vs normal-light
    # ---------------------------------------------------------

    low_psnr, low_ssim = calculate_metrics(
        prediction=low_light,
        reference=reference,
    )

    # ---------------------------------------------------------
    # Model: generated vs normal-light
    # ---------------------------------------------------------

    generated_psnr, generated_ssim = calculate_metrics(
        prediction=generated,
        reference=reference,
    )

    # ---------------------------------------------------------
    # Improvement
    # ---------------------------------------------------------

    psnr_improvement = generated_psnr - low_psnr
    ssim_improvement = generated_ssim - low_ssim

    print()
    print("------------------------------------------------------------")
    print("RESULTS")
    print("------------------------------------------------------------")

    print()
    print("LOW-LIGHT BASELINE")
    print(f"PSNR: {low_psnr:.4f} dB")
    print(f"SSIM: {low_ssim:.6f}")

    print()
    print("DIFFUSION MODEL")
    print(f"PSNR: {generated_psnr:.4f} dB")
    print(f"SSIM: {generated_ssim:.6f}")

    print()
    print("IMPROVEMENT")
    print(f"PSNR change: {psnr_improvement:+.4f} dB")
    print(f"SSIM change: {ssim_improvement:+.6f}")

    print()

    if generated_psnr > low_psnr:
        print("[PSNR] Model improves over low-light baseline.")
    else:
        print("[PSNR] Model does NOT improve over low-light baseline.")

    if generated_ssim > low_ssim:
        print("[SSIM] Model improves over low-light baseline.")
    else:
        print("[SSIM] Model does NOT improve over low-light baseline.")

    print()
    print("TASK 13.1 COMPLETED")


if __name__ == "__main__":
    main()