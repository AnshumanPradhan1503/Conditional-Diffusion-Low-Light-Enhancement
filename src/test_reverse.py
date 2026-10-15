from pathlib import Path

import torch

from src.diffusion.reverse import (
    p_sample,
    predict_x_start_from_noise,
    reverse_diffusion,
)
from src.diffusion.schedule import get_diffusion_parameters


def main():
    print("=" * 60)
    print("TASK 4 — REVERSE DIFFUSION VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Diffusion schedule
    # ---------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    print("\nDiffusion schedule:")
    print(f"  Timesteps : {len(betas)}")

    assert len(betas) == 1000
    assert len(alphas) == 1000
    assert len(alpha_bars) == 1000

    print("  [PASS] Schedule dimensions")

    # ---------------------------------------------------------
    # 2. Create synthetic clean image
    # ---------------------------------------------------------

    batch_size = 2

    clean_image = torch.randn(
        batch_size,
        3,
        256,
        256,
    )

    print("\nSynthetic clean image:")
    print(f"  Shape : {clean_image.shape}")

    # ---------------------------------------------------------
    # 3. Test x_0 reconstruction
    # ---------------------------------------------------------

    print("\nTesting x_0 reconstruction:")

    for timestep in [0, 100, 500, 999]:

        t = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
        )

        noise = torch.randn_like(clean_image)

        alpha_bar = alpha_bars[timestep]

        x_t = (
            torch.sqrt(alpha_bar) * clean_image
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

        reconstructed = predict_x_start_from_noise(
            x_t=x_t,
            timesteps=t,
            predicted_noise=noise,
            alpha_bars=alpha_bars,
        )

        reconstruction_error = torch.mean(
            torch.abs(reconstructed - clean_image)
        ).item()

        print(
            f"  t={timestep:3d} "
            f"→ reconstruction error = "
            f"{reconstruction_error:.8f}"
        )

        assert reconstructed.shape == clean_image.shape
        assert torch.isfinite(reconstructed).all()

        # Numerical reconstruction should be extremely accurate.
        assert reconstruction_error < 1e-5

        print("    [PASS] x_0 reconstruction")

    # ---------------------------------------------------------
    # 4. Test one reverse step
    # ---------------------------------------------------------

    print("\nTesting single reverse-diffusion step:")

    timestep = 500

    t = torch.full(
        (batch_size,),
        timestep,
        dtype=torch.long,
    )

    noise = torch.randn_like(clean_image)

    x_t = (
        torch.sqrt(alpha_bars[timestep]) * clean_image
        + torch.sqrt(1.0 - alpha_bars[timestep]) * noise
    )

    x_previous = p_sample(
        x_t=x_t,
        timesteps=t,
        predicted_noise=noise,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )

    print(f"  Input shape  : {x_t.shape}")
    print(f"  Output shape : {x_previous.shape}")
    print(f"  Output min   : {x_previous.min().item():.4f}")
    print(f"  Output max   : {x_previous.max().item():.4f}")
    print(f"  Output mean  : {x_previous.mean().item():.4f}")
    print(f"  Output std   : {x_previous.std().item():.4f}")

    assert x_previous.shape == x_t.shape
    assert torch.isfinite(x_previous).all()

    print("  [PASS] Shape")
    print("  [PASS] No NaN/Inf")

    # ---------------------------------------------------------
    # 5. Test complete reverse loop
    # ---------------------------------------------------------

    print("\nTesting complete reverse-diffusion loop:")

    def controlled_noise_predictor(x_t, condition, t):
        """
        Controlled predictor used only for verification.

        A real trained U-Net will replace this function later.
        """

        # Condition is intentionally unused in this controlled test.
        return torch.zeros_like(x_t)

    condition = torch.randn(
        2,
        3,
        32,
        32,
    )

    generated = reverse_diffusion(
        shape=(2, 3, 32, 32),
        condition=condition,
        predicted_noise_fn=controlled_noise_predictor,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        device=torch.device("cpu"),
    )

    print(f"  Generated shape : {generated.shape}")
    print(f"  Generated min   : {generated.min().item():.4f}")
    print(f"  Generated max   : {generated.max().item():.4f}")
    print(f"  Generated mean  : {generated.mean().item():.4f}")
    print(f"  Generated std   : {generated.std().item():.4f}")

    assert generated.shape == (2, 3, 32, 32)
    assert torch.isfinite(generated).all()

    print("  [PASS] Complete reverse loop")
    print("  [PASS] No NaN/Inf")

    # ---------------------------------------------------------
    # 6. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TASK 4 REVERSE DIFFUSION VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()