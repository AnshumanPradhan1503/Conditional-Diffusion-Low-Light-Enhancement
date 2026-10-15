from pathlib import Path
import random


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_LOW_DIR = PROJECT_ROOT / "data" / "LOLv2-real" / "train" / "low"
TRAIN_HIGH_DIR = PROJECT_ROOT / "data" / "LOLv2-real" / "train" / "high"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ============================================================
# Create paired image list
# ============================================================

def get_image_pairs():
    low_images = sorted(TRAIN_LOW_DIR.glob("low*.png"))

    pairs = []

    for low_path in low_images:
        number = low_path.stem.replace("low", "")
        high_path = TRAIN_HIGH_DIR / f"normal{number}.png"

        if not high_path.exists():
            raise FileNotFoundError(
                f"Missing high-light image for: {low_path.name}"
            )

        pairs.append((low_path.name, high_path.name))

    return pairs


# ============================================================
# Create split
# ============================================================

def main():
    print("=" * 60)
    print("CREATING TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    pairs = get_image_pairs()

    print(f"\nTotal pairs found: {len(pairs)}")

    # Reproducible shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    train_size = int(len(pairs) * TRAIN_RATIO)

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    # Create split directory
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = SPLIT_DIR / "train_pairs.txt"
    val_file = SPLIT_DIR / "val_pairs.txt"

    # Save train split
    with open(train_file, "w", encoding="utf-8") as f:
        for low_name, high_name in train_pairs:
            f.write(f"{low_name},{high_name}\n")

    # Save validation split
    with open(val_file, "w", encoding="utf-8") as f:
        for low_name, high_name in val_pairs:
            f.write(f"{low_name},{high_name}\n")

    # ========================================================
    # Verification
    # ========================================================

    print("\nSplit created successfully.")
    print("-" * 60)
    print(f"Training pairs   : {len(train_pairs)}")
    print(f"Validation pairs : {len(val_pairs)}")
    print(f"Total            : {len(train_pairs) + len(val_pairs)}")
    print(f"Random seed      : {RANDOM_SEED}")
    print(f"Train ratio      : {TRAIN_RATIO:.0%}")

    print("\nFiles created:")
    print(f"  {train_file}")
    print(f"  {val_file}")

    print("\nFirst 5 training pairs:")
    for low_name, high_name in train_pairs[:5]:
        print(f"  {low_name} -> {high_name}")

    print("\nFirst 5 validation pairs:")
    for low_name, high_name in val_pairs[:5]:
        print(f"  {low_name} -> {high_name}")

    print("\n" + "=" * 60)
    print("TASK 2 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()