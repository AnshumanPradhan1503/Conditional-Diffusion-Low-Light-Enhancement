from pathlib import Path

import torch

from src.preprocessing.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def main():
    print("=" * 60)
    print("TASK 9 — FINAL PREPROCESSING VALIDATION")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    # ---------------------------------------------------------
    # 1. Verify dataset sizes
    # ---------------------------------------------------------

    print("\nDataset sizes:")

    print(f"  Training pairs   : {len(train_loader.dataset)}")
    print(f"  Validation pairs : {len(val_loader.dataset)}")
    print(f"  Test pairs       : {len(test_loader.dataset)}")

    assert len(train_loader.dataset) == 551
    assert len(val_loader.dataset) == 138
    assert len(test_loader.dataset) == 100

    print("  [PASS] Dataset split sizes are correct")

    # ---------------------------------------------------------
    # 2. Verify batch structure
    # ---------------------------------------------------------

    print("\nBatch verification:")

    for name, loader in [
        ("Training", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ]:

        low, high = next(iter(loader))

        print(f"\n{name}:")
        print(f"  Low shape  : {low.shape}")
        print(f"  High shape : {high.shape}")

        assert low.shape == (4, 3, 256, 256)
        assert high.shape == (4, 3, 256, 256)

        assert low.dtype == torch.float32
        assert high.dtype == torch.float32

        assert torch.isfinite(low).all()
        assert torch.isfinite(high).all()

        assert low.min() >= -1.0
        assert low.max() <= 1.0

        assert high.min() >= -1.0
        assert high.max() <= 1.0

        print("  [PASS] Shape")
        print("  [PASS] dtype")
        print("  [PASS] finite values")
        print("  [PASS] normalization range")

    # ---------------------------------------------------------
    # 3. Verify training augmentation is stochastic
    # ---------------------------------------------------------

    print("\nTraining augmentation verification:")

    train_dataset = train_loader.dataset

    low_1, high_1 = train_dataset[0]
    low_2, high_2 = train_dataset[0]

    # Because training uses random crop and random horizontal flip,
    # two accesses to the same image should normally produce
    # different tensors.

    if not torch.equal(low_1, low_2):
        print("  [PASS] Training preprocessing is stochastic")
    else:
        print("  [WARNING] Two training samples happened to be identical")

    # ---------------------------------------------------------
    # 4. Verify validation preprocessing is deterministic
    # ---------------------------------------------------------

    print("\nValidation preprocessing verification:")

    val_dataset = val_loader.dataset

    low_1, high_1 = val_dataset[0]
    low_2, high_2 = val_dataset[0]

    assert torch.equal(low_1, low_2)
    assert torch.equal(high_1, high_2)

    print("  [PASS] Validation preprocessing is deterministic")

    # ---------------------------------------------------------
    # 5. Verify test preprocessing is deterministic
    # ---------------------------------------------------------

    print("\nTest preprocessing verification:")

    test_dataset = test_loader.dataset

    low_1, high_1 = test_dataset[0]
    low_2, high_2 = test_dataset[0]

    assert torch.equal(low_1, low_2)
    assert torch.equal(high_1, high_2)

    print("  [PASS] Test preprocessing is deterministic")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TASK 9 FINAL PREPROCESSING VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()