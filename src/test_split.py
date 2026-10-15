from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_LOW_DIR = (
    PROJECT_ROOT
    / "data"
    / "LOLv2-real"
    / "test"
    / "low"
)

TEST_HIGH_DIR = (
    PROJECT_ROOT
    / "data"
    / "LOLv2-real"
    / "test"
    / "high"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


def main():
    print("=" * 60)
    print("CREATING OFFICIAL TEST SPLIT")
    print("=" * 60)

    low_images = sorted(TEST_LOW_DIR.glob("low*.png"))

    pairs = []

    for low_path in low_images:

        number = low_path.stem.replace("low", "")
        high_path = TEST_HIGH_DIR / f"normal{number}.png"

        if not high_path.exists():
            raise FileNotFoundError(
                f"Missing normal-light image for {low_path.name}"
            )

        pairs.append(
            (low_path.name, high_path.name)
        )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = SPLIT_DIR / "test_pairs.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for low_name, high_name in pairs:
            file.write(f"{low_name},{high_name}\n")

    print(f"\nTest pairs found: {len(pairs)}")
    print(f"Saved to:")
    print(f"  {output_file}")

    print("\nFirst 5 test pairs:")

    for low_name, high_name in pairs[:5]:
        print(f"  {low_name} -> {high_name}")

    assert len(pairs) == 100

    print("\n[PASS] Exactly 100 official test pairs")

    print("\n" + "=" * 60)
    print("TEST SPLIT CREATED")
    print("=" * 60)


if __name__ == "__main__":
    main()