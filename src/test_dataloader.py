from pathlib import Path

import torch

from src.preprocessing.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def verify_loader(name, loader):
    print(f"\n{name} DataLoader:")

    print(f"  Number of batches : {len(loader)}")
    print(f"  Batch size        : {loader.batch_size}")

    low_batch, high_batch = next(iter(loader))

    print(f"  Low batch shape   : {low_batch.shape}")
    print(f"  High batch shape  : {high_batch.shape}")
    print(f"  Low dtype         : {low_batch.dtype}")
    print(f"  High dtype        : {high_batch.dtype}")
    print(
        f"  Low range        : "
        f"{low_batch.min().item():.4f} to {low_batch.max().item():.4f}"
    )
    print(
        f"  High range       : "
        f"{high_batch.min().item():.4f} to {high_batch.max().item():.4f}"
    )

    expected_shape = (
        loader.batch_size,
        3,
        256,
        256,
    )

    checks = [
        (
            low_batch.shape == expected_shape,
            "Low batch shape = [4, 3, 256, 256]",
        ),
        (
            high_batch.shape == expected_shape,
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
            low_batch.min() >= -1.0 and low_batch.max() <= 1.0,
            "Low values within [-1, 1]",
        ),
        (
            high_batch.min() >= -1.0 and high_batch.max() <= 1.0,
            "High values within [-1, 1]",
        ),
    ]

    print(f"\n{name} checks:")

    for passed, description in checks:
        if passed:
            print(f"  [PASS] {description}")
        else:
            print(f"  [FAIL] {description}")

    assert all(check[0] for check in checks)


def main():
    print("=" * 60)
    print("TESTING LOLv2 DATALOADERS")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    verify_loader("Training", train_loader)
    verify_loader("Validation", val_loader)
    verify_loader("Test", test_loader)

    print("\n" + "=" * 60)
    print("TASK 7 DATALOADER VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()