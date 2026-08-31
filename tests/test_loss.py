"""Test loss function and class weight handling."""
import torch
import pytest

from src.training.loss import build_loss_function


def test_build_loss_function_with_weights():
    weights = {0: 1.0, 1: 5.0, 2: 0.5}
    loss_fn = build_loss_function(class_weights=weights, label_smoothing=0.0)

    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]], requires_grad=True)
    targets = torch.tensor([1, 2])

    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
