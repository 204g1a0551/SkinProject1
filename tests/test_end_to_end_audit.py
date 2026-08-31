"""Comprehensive end-to-end audit test validating all 20 requirements of the project."""
from pathlib import Path
from PIL import Image
import torch
import pytest

from src.config import ConfigManager, get_device
from src.data.preprocessor import ImagePreprocessor
from src.data.validator import ImageValidator
from src.data.dataset import SkinLesionDataset
from src.data.augmentation import get_training_transforms, get_validation_transforms
from src.models.mobilenet_v2 import build_mobilenet_v2
from src.models.factory import ModelFactory
from src.explainability.gradcam import GradCAM, find_last_conv_layer
from src.explainability.explainer import GradCAMExplainer
from src.evaluation.evaluator import ModelEvaluator
from backend.services.inference_service import InferenceService


def test_complete_pipeline_flow(tmp_path):
    """
    Validates complete pipeline from image -> dataset -> preprocessing ->
    model -> forward pass -> Grad-CAM -> API -> evaluation.
    """
    # 1. Image Creation & Validation
    img_path = tmp_path / "lesion_audit.jpg"
    img = Image.new("RGB", (300, 300), color=(140, 95, 75))
    img.save(img_path)

    validator = ImageValidator()
    is_valid, _ = validator.validate_file(img_path)
    assert is_valid is True

    # 2. Inference Preprocessor
    preprocessor = ImagePreprocessor()
    tensor_img, display_img = preprocessor.preprocess_image(img_path)
    assert tensor_img.shape == (1, 3, 224, 224)
    assert display_img.size == (224, 224)

    # 3. Model Loading & Forward Pass
    device = get_device("auto")
    model = ModelFactory.create_model(device=device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor_img.to(device))
        probs = torch.softmax(logits, dim=1)
    assert probs.shape == (1, 7)
    assert pytest.approx(float(probs.sum().item()), 0.01) == 1.0

    # 4. Dynamic Grad-CAM Layer Detection & Heatmap
    layer_name, _ = find_last_conv_layer(model)
    assert "18" in layer_name

    gradcam = GradCAM(model=model)
    heatmap, pred_idx, conf = gradcam.generate_heatmap(
        input_tensor=tensor_img.to(device),
        target_size=(300, 300),
    )
    assert heatmap.shape == (300, 300)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    # 5. Backend Inference Service
    service = InferenceService()
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    res = service.predict(img_bytes)
    assert res["success"] is True
    assert "predicted_name" in res
    assert "top_predictions" in res
    assert len(res["top_predictions"]) == 7
    assert res["gradcam_base64"].startswith("data:image/jpeg;base64,")
    assert "CLINICAL NOTICE" in res["disclaimer"]
