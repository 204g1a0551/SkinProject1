"""Test SkinLesionDataset and DataLoader functionality."""
import pandas as pd
from PIL import Image
import torch
import pytest

from src.data.dataset import SkinLesionDataset
from src.data.augmentation import get_training_transforms, get_validation_transforms
from src.data.dataloader import create_dataloaders


def test_skin_lesion_dataset(tmp_path):
    # Create 4 dummy image files
    data_rows = []
    for i in range(4):
        p = tmp_path / f"img_{i}.jpg"
        img = Image.new("RGB", (224, 224), color=(100 + i * 20, 100, 100))
        img.save(p, format="JPEG")
        data_rows.append({"image_path": str(p), "class_idx": i % 2})

    df = pd.DataFrame(data_rows)
    transform = get_validation_transforms()
    dataset = SkinLesionDataset(df, transform=transform)

    assert len(dataset) == 4
    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert label in (0, 1)


def test_dataloaders_creation(tmp_path):
    # Create small dataset for train, val, test
    def make_df(count):
        rows = []
        for i in range(count):
            p = tmp_path / f"img_{count}_{i}.jpg"
            img = Image.new("RGB", (224, 224), color=(150, 120, 90))
            img.save(p, format="JPEG")
            rows.append({"image_path": str(p), "class_idx": i % 2})
        return pd.DataFrame(rows)

    train_df = make_df(8)
    val_df = make_df(4)
    test_df = make_df(4)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, batch_size=4, num_workers=0
    )

    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape == (4, 3, 224, 224)
    assert batch_labels.shape == (4,)
