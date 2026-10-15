from pathlib import Path

from src.preprocessing.paired_dataset import LOLv2PairedDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"

TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train_pairs.txt"
VAL_SPLIT = PROJECT_ROOT / "data" / "splits" / "val_pairs.txt"
TEST_SPLIT = PROJECT_ROOT / "data" / "splits" / "test_pairs.txt"


def verify_dataset(dataset, expected_length, name):
    print(f"\n{name} dataset:")
    print(f"  Number of pairs: {len(dataset)}")

    low_tensor, high_tensor = dataset[0]

    print(f"  Low tensor shape : {low_tensor.shape}")
    print(f"  High tensor shape: {high_tensor.shape}")
    print(f"  Low dtype        : {low_tensor.dtype}")
    print(f"  High dtype       : {high_tensor.dtype}")
    print(
        f"  Low range        : "
        f"{low_tensor.min().item():.4f} "
        f"to {low_tensor.max().item():.4f}"
    )
    print(
        f"  High range       : "
        f"{high_tensor.min().item():.4f} "
        f"to {high_tensor.max().item():.4f}"
    )

    assert len(dataset) == expected_length

    assert low_tensor.shape == (3, 256, 256)
    assert high_tensor.shape == (3, 256, 256)

    assert low_tensor.dtype.is_floating_point
    assert high_tensor.dtype.is_floating_point

    assert low_tensor.min() >= -1.0
    assert low_tensor.max() <= 1.0

    assert high_tensor.min() >= -1.0
    assert high_tensor.max() <= 1.0

    print(f"\n{name} checks:")
    print(f"  [PASS] {expected_length} pairs")
    print("  [PASS] Low tensor = [3, 256, 256]")
    print("  [PASS] High tensor = [3, 256, 256]")
    print("  [PASS] Floating-point tensors")
    print("  [PASS] Values within [-1, 1]")


def main():

    print("=" * 60)
    print("TESTING LOLv2 TRAIN / VALIDATION / TEST DATASETS")
    print("=" * 60)

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    train_dataset = LOLv2PairedDataset(
        data_root=DATA_ROOT,
        split_file=TRAIN_SPLIT,
        mode="train",
        image_size=256,
    )

    verify_dataset(
        train_dataset,
        expected_length=551,
        name="Training",
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    val_dataset = LOLv2PairedDataset(
        data_root=DATA_ROOT,
        split_file=VAL_SPLIT,
        mode="val",
        image_size=256,
    )

    verify_dataset(
        val_dataset,
        expected_length=138,
        name="Validation",
    )

    # ----------------------------------------------------------
    # Official test
    # ----------------------------------------------------------

    test_dataset = LOLv2PairedDataset(
        data_root=DATA_ROOT,
        split_file=TEST_SPLIT,
        mode="test",
        image_size=256,
    )

    verify_dataset(
        test_dataset,
        expected_length=100,
        name="Test",
    )

    print("\n" + "=" * 60)
    print("TASK 6 DATASET VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()