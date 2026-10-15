from pathlib import Path

import torch

from src.diffusion.forward import q_sample
from src.diffusion.schedule import get_diffusion_parameters
from src.preprocessing.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def main():
    print("=" * 60)
    print("TASK 3 — FORWARD DIFFUSION VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load diffusion schedule
    # ---------------------------------------------------------

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    print("\nDiffusion schedule:")
    print(f"  Timesteps : {len(betas)}")

    # ---------------------------------------------------------
    # Load one real LOLv2 batch
    # ---------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    low, high = next(iter(test_loader))

    print("\nInput batch:")
    print(f"  Low-light shape  : {low.shape}")
    print(f"  Normal-light shape: {high.shape}")

    # ---------------------------------------------------------
    # Test different timesteps
    # ---------------------------------------------------------

    timesteps_to_test = [0, 100, 500, 999]

    print("\nForward diffusion results:")

    noisy_images = {}

    for timestep in timesteps_to_test:

        t = torch.full(
            (high.shape[0],),
            timestep,
            dtype=torch.long,
        )

        noisy, noise = q_sample(
            x_start=high,
            timesteps=t,
            alpha_bars=alpha_bars,
        )

        noisy_images[timestep] = noisy

        print(f"\n  Timestep t = {timestep}")
        print(f"    Shape     : {noisy.shape}")
        print(f"    Min       : {noisy.min().item():.4f}")
        print(f"    Max       : {noisy.max().item():.4f}")
        print(f"    Mean      : {noisy.mean().item():.4f}")
        print(f"    Std       : {noisy.std().item():.4f}")

        assert noisy.shape == high.shape
        assert noise.shape == high.shape
        assert torch.isfinite(noisy).all()
        assert torch.isfinite(noise).all()

        print("    [PASS] Shape")
        print("    [PASS] No NaN/Inf")

    # ---------------------------------------------------------
    # Verify noise increases with timestep
    # ---------------------------------------------------------

    print("\nNoise progression verification:")

    clean = high

    mse_values = {}

    for timestep, noisy in noisy_images.items():
        mse = torch.mean((noisy - clean) ** 2).item()
        mse_values[timestep] = mse

        print(
            f"  t={timestep:3d} → "
            f"MSE from clean image = {mse:.6f}"
        )

    assert mse_values[0] < mse_values[100]
    assert mse_values[100] < mse_values[500]
    assert mse_values[500] < mse_values[999]

    print("  [PASS] Noise increases with timestep")

    # ---------------------------------------------------------
    # Verify provided noise is reproduced exactly
    # ---------------------------------------------------------

    print("\nDeterministic noise verification:")

    t = torch.full(
        (high.shape[0],),
        500,
        dtype=torch.long,
    )

    fixed_noise = torch.randn_like(high)

    noisy_1, returned_noise = q_sample(
        x_start=high,
        timesteps=t,
        alpha_bars=alpha_bars,
        noise=fixed_noise,
    )

    noisy_2, _ = q_sample(
        x_start=high,
        timesteps=t,
        alpha_bars=alpha_bars,
        noise=fixed_noise,
    )

    assert torch.equal(returned_noise, fixed_noise)
    assert torch.equal(noisy_1, noisy_2)

    print("  [PASS] Supplied noise is preserved")
    print("  [PASS] Same input + same noise produces same output")

    print("\n" + "=" * 60)
    print("TASK 3 FORWARD DIFFUSION VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()