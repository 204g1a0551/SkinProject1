"""Test image preprocessing pipeline."""
import io
import numpy as np
from PIL import Image
import torch
import pytest

from src.data.preprocessor import ImagePreprocessor


def test_preprocessor_with_synthetic_image():
    preprocessor = ImagePreprocessor(image_size=(224, 224))

    # Create dummy RGB image in memory
    img = Image.new("RGB", (400, 300), color=(180, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    tensor_img, display_img = preprocessor.preprocess_image(image_bytes)

    # Check tensor properties
    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 3, 224, 224)
    assert tensor_img.dtype == torch.float32

    # Check display image properties
    assert isinstance(display_img, Image.Image)
    assert display_img.size == (224, 224)
