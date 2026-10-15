from pathlib import Path

import torch

from src.preprocessing.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def verify_batch(name, loader):
    print(f"\n{name} batch verification:")

    low_batch, high_batch = next(iter(loader))

    print(f"  Low batch shape  : {low_batch.shape}")
    print(f"  High batch shape : {high_batch.shape}")

    print(f"  Low min/max      : {low_batch.min().item():.4f} / {low_batch.max().item():.4f}")
    print(f"  High min/max     : {high_batch.min().item():.4f} / {high_batch.max().item():.4f}")

    checks = [
        (
            low_batch.shape == (4, 3, 256, 256),
            "Low batch shape = [4, 3, 256, 256]",
        ),
        (
            high_batch.shape == (4, 3, 256, 256),
            "High batch shape = [4, 3, 256, 256]",
        ),
        (
            low_batch.dtype == torch.float32,
            "Low batch dtype = float32",
        ),
        (
            high_batch.dtype == torch.float32,
            "High batch dtype = float32",
        ),
        (
            torch.isfinite(low_batch).all(),
            "Low batch contains no NaN/Inf",
        ),
        (
            torch.isfinite(high_batch).all(),
            "High batch contains no NaN/Inf",
        ),
        (
            low_batch.min() >= -1.0 and low_batch.max() <= 1.0,
            "Low batch within [-1, 1]",
        ),
        (
            high_batch.min() >= -1.0 and high_batch.max() <= 1.0,
            "High batch within [-1, 1]",
        ),
        (
            low_batch.shape == high_batch.shape,
            "Low/high batch dimensions match",
        ),
    ]

    for passed, description in checks:
        if passed:
            print(f"  [PASS] {description}")
        else:
            print(f"  [FAIL] {description}")

    assert all(passed for passed, _ in checks)


def main():
    print("=" * 60)
    print("TASK 8 — LOLv2 BATCH VERIFICATION")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    verify_batch("Training", train_loader)
    verify_batch("Validation", val_loader)
    verify_batch("Test", test_loader)

    print("\n" + "=" * 60)
    print("TASK 8 BATCH VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()