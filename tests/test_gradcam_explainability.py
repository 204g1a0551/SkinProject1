"""Comprehensive test suite for Grad-CAM Explainability."""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pytest

from src.explainability.gradcam import (
    GradCAM,
    find_last_conv_layer,
    apply_colormap_on_image,
    create_standalone_heatmap_image,
)
from src.explainability.explainer import GradCAMExplainer, CLINICAL_DECISION_SUPPORT_DISCLAIMER
from src.models.mobilenet_v2 import build_mobilenet_v2


def test_find_last_conv_layer():
    """Verify recursive module search finds the final Conv2d in MobileNetV2."""
    model = build_mobilenet_v2(num_classes=7, pretrained=False)
    name, layer = find_last_conv_layer(model)

    assert isinstance(layer, nn.Conv2d)
    # The last conv layer in MobileNetV2 feature extractor is in block 18
    assert "18" in name


def test_gradcam_heatmap_dimensions(tmp_path):
    """Verify Grad-CAM properly scales heatmap to match original image dimensions."""
    model = build_mobilenet_v2(num_classes=7, pretrained=False)
    model.eval()
    gradcam = GradCAM(model=model)

    # Input tensor at model resolution
    input_tensor = torch.randn(1, 3, 224, 224)
    original_size = (640, 480)  # (w, h)

    heatmap, target_idx, conf = gradcam.generate_heatmap(
        input_tensor=input_tensor,
        target_size=original_size,
    )

    # Height should be 480, Width should be 640
    assert heatmap.shape == (480, 640)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_gradcam_target_class_selection():
    """Verify specifying different target classes produces class-specific activations."""
    model = build_mobilenet_v2(num_classes=7, pretrained=False)
    model.eval()
    gradcam = GradCAM(model=model)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Explain class 0
    hm_0, idx_0, _ = gradcam.generate_heatmap(input_tensor, target_class_idx=0)
    # Explain class 1
    hm_1, idx_1, _ = gradcam.generate_heatmap(input_tensor, target_class_idx=1)

    assert idx_0 == 0
    assert idx_1 == 1
    assert hm_0.shape == hm_1.shape


def test_gradcam_explainer_end_to_end(tmp_path):
    """Test full GradCAMExplainer workflow including file exports and medical disclaimer."""
    # 1. Create a dummy test image
    img_path = tmp_path / "test_lesion.jpg"
    test_img = Image.new("RGB", (320, 240), color=(180, 90, 80))
    test_img.save(img_path)

    # 2. Run explainer
    explainer = GradCAMExplainer(device="cpu")
    explanation = explainer.explain_image(img_path, target_class="mel")

    assert explanation["success"] is True
    assert explanation["original_dimensions"] == [320, 240]
    assert explanation["explained_class"]["code"] == "mel"
    assert explanation["disclaimer"] == CLINICAL_DECISION_SUPPORT_DISCLAIMER

    # 3. Save artifacts
    out_dir = tmp_path / "artifacts"
    saved = explainer.save_explanation_artifacts(explanation, output_dir=out_dir, prefix="case_1")

    assert saved["original"].exists()
    assert saved["heatmap"].exists()
    assert saved["overlay"].exists()
    assert saved["metadata"].exists()


def test_gradcam_explainer_invalid_inputs():
    """Verify robust error handling on empty or invalid inputs."""
    explainer = GradCAMExplainer(device="cpu")

    with pytest.raises(ValueError, match="Empty image bytes"):
        explainer.explain_image(b"")

    with pytest.raises(ValueError, match="Unknown target class"):
        dummy_img = Image.new("RGB", (100, 100))
        explainer.explain_image(dummy_img, target_class="non_existent_disease")
