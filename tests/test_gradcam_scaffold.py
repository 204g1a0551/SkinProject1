"""Test Grad-CAM heatmap generation and overlay."""
import numpy as np
from PIL import Image
import torch
import pytest

from src.models.mobilenet_v2 import build_mobilenet_v2
from src.explainability.gradcam import GradCAM, apply_colormap_on_image


def test_gradcam_pipeline():
    model = build_mobilenet_v2(num_classes=7, pretrained=False)
    model.eval()  # Set to eval mode so BatchNorm behaves correctly for batch size 1
    gradcam = GradCAM(model=model)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, conf = gradcam.generate_heatmap(dummy_input)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert 0 <= pred_class < 7
    assert 0.0 <= conf <= 1.0

    # Test blending
    dummy_pil = Image.new("RGB", (224, 224), color=(200, 150, 120))
    blended = apply_colormap_on_image(dummy_pil, heatmap)
    assert isinstance(blended, Image.Image)
    assert blended.size == (224, 224)
