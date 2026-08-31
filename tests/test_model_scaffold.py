"""Test model architecture instantiation, unfreezing, and output dimensions."""
import torch
import pytest

from src.models.mobilenet_v2 import SkinMobileNetV2, build_mobilenet_v2


def test_mobilenet_v2_shapes():
    num_classes = 7
    batch_size = 2
    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert outputs.shape == (batch_size, num_classes)


def test_mobilenet_v2_freeze_and_unfreeze():
    model = build_mobilenet_v2(num_classes=7, pretrained=False, freeze_features=True)

    # In frozen stage, backbone parameters should have requires_grad=False
    frozen_summary = model.get_parameter_summary()
    assert frozen_summary["frozen_parameters"] > 0

    # Unfreeze upper blocks (14..18)
    model.unfreeze_backbone(unfreeze_from_layer=14)
    unfrozen_summary = model.get_parameter_summary()
    assert unfrozen_summary["trainable_parameters"] > frozen_summary["trainable_parameters"]
