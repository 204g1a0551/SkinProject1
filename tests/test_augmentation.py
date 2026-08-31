"""Test data augmentation ranges and medical bounds."""
from PIL import Image
import torch
import pytest

from src.data.augmentation import (
    get_training_transforms,
    get_validation_transforms,
    denormalize_tensor,
)


def test_augmentation_output_tensor_properties():
    pil_img = Image.new("RGB", (300, 300), color=(180, 120, 90))
    train_t = get_training_transforms(image_size=(224, 224))
    val_t = get_validation_transforms(image_size=(224, 224))

    tensor_train = train_t(pil_img)
    tensor_val = val_t(pil_img)

    assert isinstance(tensor_train, torch.Tensor)
    assert tensor_train.shape == (3, 224, 224)
    assert isinstance(tensor_val, torch.Tensor)
    assert tensor_val.shape == (3, 224, 224)

    # Test denormalization clamps between [0, 1]
    denorm = denormalize_tensor(tensor_val)
    assert denorm.shape == (3, 224, 224)
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0
