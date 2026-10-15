import torch

from src.diffusion.schedule import get_diffusion_parameters


def main():
    print("=" * 60)
    print("TASK 2 — DDPM NOISE SCHEDULE VERIFICATION")
    print("=" * 60)

    timesteps = 1000

    betas, alphas, alpha_bars = get_diffusion_parameters(
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=0.02,
    )

    print("\nSchedule configuration:")
    print(f"  Timesteps       : {timesteps}")
    print(f"  Beta start      : {betas[0].item():.6f}")
    print(f"  Beta end        : {betas[-1].item():.6f}")

    print("\nTensor shapes:")
    print(f"  Betas           : {betas.shape}")
    print(f"  Alphas          : {alphas.shape}")
    print(f"  Alpha bars      : {alpha_bars.shape}")

    print("\nSchedule values:")
    print(f"  beta[0]         : {betas[0].item():.6f}")
    print(f"  beta[499]       : {betas[499].item():.6f}")
    print(f"  beta[999]       : {betas[999].item():.6f}")

    print("\nAlpha values:")
    print(f"  alpha[0]        : {alphas[0].item():.6f}")
    print(f"  alpha[999]      : {alphas[-1].item():.6f}")

    print("\nCumulative alpha values:")
    print(f"  alpha_bar[0]    : {alpha_bars[0].item():.6f}")
    print(f"  alpha_bar[499]  : {alpha_bars[499].item():.6f}")
    print(f"  alpha_bar[999]  : {alpha_bars[-1].item():.6f}")

    checks = [
        (
            betas.shape == (1000,),
            "Beta schedule has 1000 values",
        ),
        (
            torch.all(betas > 0),
            "All beta values are positive",
        ),
        (
            torch.all(betas < 1),
            "All beta values are less than 1",
        ),
        (
            torch.all(alphas > 0),
            "All alpha values are positive",
        ),
        (
            torch.all(alpha_bars > 0),
            "All alpha_bar values are positive",
        ),
        (
            torch.all(alpha_bars <= 1),
            "All alpha_bar values are <= 1",
        ),
        (
            torch.all(betas[1:] > betas[:-1]),
            "Beta schedule increases monotonically",
        ),
        (
            torch.all(alpha_bars[1:] < alpha_bars[:-1]),
            "Alpha_bar decreases monotonically",
        ),
    ]

    print("\nSchedule checks:")

    for passed, description in checks:
        if passed:
            print(f"  [PASS] {description}")
        else:
            print(f"  [FAIL] {description}")

    assert all(passed for passed, _ in checks)

    print("\n" + "=" * 60)
    print("TASK 2 NOISE SCHEDULE VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()