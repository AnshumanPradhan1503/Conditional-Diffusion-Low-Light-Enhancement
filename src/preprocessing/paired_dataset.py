from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.preprocessing.normalization import image_to_tensor
from src.preprocessing.paired_transforms import (
    paired_center_crop,
    paired_random_crop,
    paired_random_horizontal_flip,
)


class LOLv2PairedDataset(Dataset):
    """
    Paired LOLv2 dataset for low-light image enhancement.

    Returns:
        low_tensor  : [3, 256, 256]
        high_tensor : [3, 256, 256]

    Split behavior:

        train:
            images from data/LOLv2-real/train/
            random crop + random horizontal flip

        val:
            images from data/LOLv2-real/train/
            center crop only

        test:
            images from data/LOLv2-real/test/
            center crop only
    """

    def __init__(
        self,
        data_root: Path,
        split_file: Path,
        mode: str = "train",
        image_size: int = 256,
    ) -> None:

        if mode not in {"train", "val", "test"}:
            raise ValueError(
                "mode must be one of: 'train', 'val', 'test'"
            )

        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.mode = mode
        self.image_size = image_size

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_root}"
            )

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self.split_file}"
            )

        # ------------------------------------------------------
        # Determine the physical image directory
        # ------------------------------------------------------

        if mode in {"train", "val"}:
            image_root = self.data_root / "train"
        else:
            image_root = self.data_root / "test"

        self.low_dir = image_root / "low"
        self.high_dir = image_root / "high"

        if not self.low_dir.exists():
            raise FileNotFoundError(
                f"Low-light directory not found: {self.low_dir}"
            )

        if not self.high_dir.exists():
            raise FileNotFoundError(
                f"Normal-light directory not found: {self.high_dir}"
            )

        # Load filename pairs
        self.pairs = self._load_pairs()

        if len(self.pairs) == 0:
            raise ValueError(
                f"No image pairs found in {self.split_file}"
            )

    def _load_pairs(self) -> list[Tuple[str, str]]:
        """Load paired filenames from the split file."""

        pairs = []

        with open(self.split_file, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                parts = line.split(",")

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid pair at line {line_number}: {line}"
                    )

                low_name = parts[0].strip()
                high_name = parts[1].strip()

                pairs.append((low_name, high_name))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        low_name, high_name = self.pairs[index]

        low_path = self.low_dir / low_name
        high_path = self.high_dir / high_name

        if not low_path.exists():
            raise FileNotFoundError(
                f"Low-light image not found: {low_path}"
            )

        if not high_path.exists():
            raise FileNotFoundError(
                f"Normal-light image not found: {high_path}"
            )

        # ------------------------------------------------------
        # Load images
        # ------------------------------------------------------

        low_image = Image.open(low_path).convert("RGB")
        high_image = Image.open(high_path).convert("RGB")

        # ------------------------------------------------------
        # Spatial preprocessing
        # ------------------------------------------------------

        if self.mode == "train":

            low_image, high_image = paired_random_crop(
                low_image,
                high_image,
                size=self.image_size,
            )

            low_image, high_image = paired_random_horizontal_flip(
                low_image,
                high_image,
                probability=0.5,
            )

        else:

            low_image, high_image = paired_center_crop(
                low_image,
                high_image,
                size=self.image_size,
            )

        # ------------------------------------------------------
        # Normalization
        # ------------------------------------------------------

        low_tensor = image_to_tensor(low_image)
        high_tensor = image_to_tensor(high_image)

        return low_tensor, high_tensor