"""Sanity test: complete pipeline pass on real forward and backward steps."""
import pandas as pd
from PIL import Image
import torch
import pytest

from src.models.mobilenet_v2 import build_mobilenet_v2
from src.data.dataset import SkinLesionDataset
from src.data.augmentation import get_training_transforms, get_validation_transforms
from src.data.dataloader import create_dataloaders
from src.training.trainer import ModelTrainer
from src.training.loss import build_loss_function


def test_end_to_end_sanity_pass(tmp_path):
    """
    Verifies:
    dataset -> transforms -> dataloader -> model -> forward pass -> loss -> backward pass -> output probs.
    """
    # Create 8 dummy image files
    rows = []
    for i in range(8):
        img_p = tmp_path / f"lesion_{i}.jpg"
        img = Image.new("RGB", (224, 224), color=(120 + i * 10, 100, 80))
        img.save(img_p, format="JPEG")
        rows.append({"image_path": str(img_p), "class_idx": i % 3})

    df = pd.DataFrame(rows)
    train_df = df.iloc[:6]
    val_df = df.iloc[6:]

    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, val_df, batch_size=2, num_workers=0
    )

    # 1. Instantiate MobileNetV2
    model = build_mobilenet_v2(num_classes=3, pretrained=False, freeze_features=True)
    device = torch.device("cpu")

    # 2. Setup loss and optimizer
    criterion = build_loss_function(class_weights=None, label_smoothing=0.0, device=device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    trainer = ModelTrainer(model=model, criterion=criterion, optimizer=optimizer, device=device)

    # 3. Train 1 epoch
    loss, acc = trainer.train_epoch(train_loader)
    assert loss > 0.0
    assert 0.0 <= acc <= 100.0

    # 4. Evaluate
    val_loss, val_acc, targets, preds = trainer.evaluate(val_loader)
    assert val_loss > 0.0
    assert len(preds) == len(val_df)

    # 5. Output probabilities test
    test_img = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(test_img)
        probs = torch.softmax(logits, dim=1)

    assert probs.shape == (1, 3)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
