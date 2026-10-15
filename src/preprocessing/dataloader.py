from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.preprocessing.paired_dataset import LOLv2PairedDataset


def create_dataloaders(
    data_root: Path,
    split_root: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Create DataLoaders for the LOLv2 paired dataset.

    Returns:
        train_loader
        val_loader
        test_loader
    """

    train_dataset = LOLv2PairedDataset(
        data_root=data_root,
        split_file=split_root / "train_pairs.txt",
        mode="train",
    )

    val_dataset = LOLv2PairedDataset(
        data_root=data_root,
        split_file=split_root / "val_pairs.txt",
        mode="val",
    )

    test_dataset = LOLv2PairedDataset(
        data_root=data_root,
        split_file=split_root / "test_pairs.txt",
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader