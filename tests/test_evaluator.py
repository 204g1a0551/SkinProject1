"""Test ModelEvaluator engine loading and inference on test set."""
import pandas as pd
from PIL import Image
import torch
import pytest

from src.evaluation.evaluator import ModelEvaluator
from src.models.mobilenet_v2 import build_mobilenet_v2


def test_model_evaluator_execution(tmp_path):
    # 1. Create a dummy model checkpoint with 3 classes
    model_p = tmp_path / "test_model.pth"
    model = build_mobilenet_v2(num_classes=3, pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, model_p)

    # 2. Create dummy test images & manifest
    test_manifest_p = tmp_path / "test_manifest.csv"
    rows = []
    for i in range(6):
        img_p = tmp_path / f"test_img_{i}.jpg"
        img = Image.new("RGB", (224, 224), color=(100 + i * 15, 80, 60))
        img.save(img_p, format="JPEG")
        rows.append({"image_path": str(img_p), "class_name": f"c_{i%3}", "class_idx": i % 3})

    pd.DataFrame(rows).to_csv(test_manifest_p, index=False)

    # 3. Instantiate evaluator and run
    evaluator = ModelEvaluator(
        model_path=model_p,
        device="cpu",
    )
    metrics = evaluator.evaluate_test_set(test_manifest_path=test_manifest_p, batch_size=2)

    assert metrics["total_samples"] == 6
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "confusion_matrix_norm" in metrics
    assert len(metrics["per_class"]) == 3  # Matches checkpoint's 3 classes
