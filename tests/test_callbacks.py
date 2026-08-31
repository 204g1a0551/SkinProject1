"""Test EarlyStopping and ModelCheckpoint callbacks."""
import torch
import torch.nn as nn
import pytest

from src.training.callbacks import EarlyStopping, ModelCheckpoint


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    assert es(1.0) is False   # best = 1.0
    assert es(0.95) is False  # best = 0.95 (improved)
    assert es(0.96) is False  # counter = 1
    assert es(0.97) is False  # counter = 2
    assert es(0.98) is True   # counter = 3 -> halt triggered
    assert es.early_stop is True


def test_model_checkpoint_save_and_load(tmp_path):
    ckpt_path = tmp_path / "test_model.pth"
    checkpoint = ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", mode="min")

    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Step with improvement
    saved = checkpoint.step(
        current_value=0.5,
        model=model,
        optimizer=optimizer,
        epoch=1,
        extra_metadata={"accuracy": 85.0},
    )
    assert saved is True
    assert ckpt_path.exists()

    # Load and verify state contents
    loaded = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert loaded["best_val_loss"] == 0.5
    assert loaded["accuracy"] == 85.0

    # Step without improvement
    saved_worse = checkpoint.step(
        current_value=0.8,
        model=model,
        optimizer=optimizer,
        epoch=2,
    )
    assert saved_worse is False
