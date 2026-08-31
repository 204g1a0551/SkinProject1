#!/usr/bin/env python3
"""
CLI Tool: Train Skin Disease Classifier using MobileNetV2 Transfer Learning
Supports Two-Stage Transfer Learning with MPS / CUDA / CPU automatic hardware acceleration.
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager, get_device
from src.models.mobilenet_v2 import build_mobilenet_v2
from src.data.dataloader import create_dataloaders
from src.data.imbalance import ImbalanceAnalyzer
from src.training.pipeline import TwoStageTrainingPipeline
from src.utils.logger import get_logger

logger = get_logger("train_cli")


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on skin disease dataset.")
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing train_manifest.csv and val_manifest.csv",
    )
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=5,
        help="Epochs for Stage 1 (Backbone frozen, training head)",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=10,
        help="Epochs for Stage 2 (Fine-tuning upper MobileNetV2 layers)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--stage1-lr",
        type=float,
        default=0.001,
        help="Learning rate for Stage 1",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=0.0001,
        help="Learning rate for Stage 2 fine-tuning",
    )
    parser.add_argument(
        "--unfreeze-layer",
        type=int,
        default=14,
        help="MobileNetV2 feature block index to unfreeze from (0..18)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('auto', 'mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/mobilenetv2_skin_disease_best.pth",
        help="Path to save best model checkpoint",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable automatic class-weighted loss",
    )
    args = parser.parse_args()

    cfg = ConfigManager()
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = PROJECT_ROOT / manifest_dir

    train_path = manifest_dir / "train_manifest.csv"
    val_path = manifest_dir / "val_manifest.csv"
    test_path = manifest_dir / "test_manifest.csv"

    if not (train_path.exists() and val_path.exists()):
        logger.error(
            f"Manifests not found in {manifest_dir}. "
            f"Please run `python scripts/prepare_splits.py` first."
        )
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else val_df

    logger.info(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Class mappings
    class_mapping = {c.code: idx for idx, c in enumerate(cfg.dataset_config.classes)}
    num_classes = len(class_mapping)

    # Save class_mapping.json alongside model
    model_out = Path(args.model_out)
    if not model_out.is_absolute():
        model_out = PROJECT_ROOT / model_out
    model_out.parent.mkdir(parents=True, exist_ok=True)

    mapping_file = model_out.parent / "class_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "classes": [
                    {"code": c.code, "name": c.name, "idx": idx, "severity": c.severity}
                    for idx, c in enumerate(cfg.dataset_config.classes)
                ],
                "code_to_idx": class_mapping,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved class label mapping to: {mapping_file}")

    # Compute class weights if enabled
    class_weights = None
    if not args.no_class_weights:
        analyzer = ImbalanceAnalyzer()
        analysis = analyzer.analyze_distribution(train_df["class_idx"].tolist())
        class_weights = analysis["balanced_weights"]
        logger.info(f"Computed balanced class weights: {class_weights}")

    # Detect hardware device
    target_device = get_device(args.device)
    logger.info(f"Selected training device: {target_device}")

    # Create DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=cfg,
        batch_size=args.batch_size,
    )

    # Build MobileNetV2
    model = build_mobilenet_v2(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=cfg.model_config.dropout_rate,
        freeze_features=True,
        device=target_device,
    )

    # Execute 2-stage transfer learning pipeline
    pipeline = TwoStageTrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=target_device,
        class_weights=class_weights,
        checkpoint_path=str(model_out),
        history_path=str(model_out.parent / "training_history.json"),
        class_mapping=class_mapping,
    )

    pipeline.run(
        stage1_epochs=args.epochs_stage1,
        stage1_lr=args.stage1_lr,
        stage2_epochs=args.epochs_stage2,
        stage2_lr=args.stage2_lr,
        unfreeze_from_layer=args.unfreeze_layer,
    )


if __name__ == "__main__":
    main()
