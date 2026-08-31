"""
PyTorch DataLoader factory optimized for macOS Apple Silicon and general platforms.
Handles worker threads, memory pinning, and batch creation.
"""
import os
import platform
from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import SkinLesionDataset
from .augmentation import get_training_transforms, get_validation_transforms
from ..config import ConfigManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[ConfigManager] = None,
    use_weighted_sampler: bool = False,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test PyTorch DataLoaders.
    On macOS (Darwin), defaults to num_workers=0 to prevent posix shared memory IPC limitations.
    """
    cfg = config or ConfigManager()
    ds_cfg = cfg.dataset_config
    m_cfg = cfg.model_config

    eff_batch_size = batch_size or m_cfg.batch_size

    # On macOS, num_workers=0 avoids shared memory / multiprocessing IPC errors and runs extremely fast
    if num_workers is None:
        if platform.system() == "Darwin":
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 4)

    # Do not pin memory on MPS (unsupported / generates warnings in PyTorch)
    pin_memory = torch.cuda.is_available()

    # Transforms
    train_transform = get_training_transforms(
        image_size=ds_cfg.image_size,
        mean=ds_cfg.mean,
        std=ds_cfg.std,
    )
    val_transform = get_validation_transforms(
        image_size=ds_cfg.image_size,
        mean=ds_cfg.mean,
        std=ds_cfg.std,
    )

    # Datasets
    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=val_transform)
    test_dataset = SkinLesionDataset(test_df, transform=val_transform)

    # Optional Weighted Random Sampler for severe class imbalance
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        labels = train_dataset.get_labels()
        class_counts = pd.Series(labels).value_counts().to_dict()
        sample_weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=eff_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) > eff_batch_size else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eff_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eff_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"DataLoaders ready: Batch Size={eff_batch_size}, Workers={num_workers}, "
        f"PinMemory={pin_memory}, WeightedSampler={use_weighted_sampler}"
    )

    return train_loader, val_loader, test_loader
