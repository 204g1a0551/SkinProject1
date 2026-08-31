"""Test enhanced MetricsCalculator and diagnostic pair detection."""
import numpy as np
import pytest

from src.evaluation.metrics import MetricsCalculator


def test_metrics_calculator_perfect_score():
    class_names = ["Melanoma", "Melanocytic Nevi", "Basal Cell Carcinoma"]
    class_codes = ["mel", "nv", "bcc"]
    calc = MetricsCalculator(class_names=class_names, class_codes=class_codes)

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

    metrics = calc.compute(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert len(metrics["per_class"]) == 3
    assert len(metrics["top_confusion_pairs"]) == 0

    # Test normalized confusion matrix diagonal is all 1.0
    cmn = np.array(metrics["confusion_matrix_norm"])
    assert np.allclose(np.diag(cmn), 1.0)


def test_metrics_calculator_confusion_pairs():
    class_names = ["Melanoma", "Melanocytic Nevi"]
    class_codes = ["mel", "nv"]
    calc = MetricsCalculator(class_names=class_names, class_codes=class_codes)

    # 4 true melanomas: 2 predicted as melanoma, 2 misclassified as nevus
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 1]

    metrics = calc.compute(y_true, y_pred)

    assert pytest.approx(metrics["accuracy"], 0.01) == 4 / 6
    assert len(metrics["top_confusion_pairs"]) == 1
    pair = metrics["top_confusion_pairs"][0]
    assert pair["true_class"] == "Melanoma"
    assert pair["pred_class"] == "Melanocytic Nevi"
    assert pair["count"] == 2
    assert pair["percentage_of_true"] == 50.0
