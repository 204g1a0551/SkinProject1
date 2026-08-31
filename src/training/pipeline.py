"""
Two-Stage Transfer Learning Pipeline Orchestrator.
Stage 1: Frozen feature extractor -> Trains custom classifier head.
Stage 2: Unfreezes upper inverted residual blocks -> Fine-tunes model at reduced LR.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping, ModelCheckpoint
from .loss import build_loss_function
from .trainer import ModelTrainer
from ..models.mobilenet_v2 import SkinMobileNetV2
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TwoStageTrainingPipeline:
    """Orchestrates 2-stage transfer learning for MobileNetV2."""

    def __init__(
        self,
        model: SkinMobileNetV2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        class_weights: Optional[Dict[int, float]] = None,
        checkpoint_path: str = "models/mobilenetv2_skin_disease_best.pth",
        history_path: str = "models/training_history.json",
        label_smoothing: float = 0.05,
        early_stopping_patience: int = 5,
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)
        self.class_mapping = class_mapping or {}

        # Build loss criterion
        self.criterion = build_loss_function(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            device=device,
        )

        self.early_stopping_patience = early_stopping_patience
        self.checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor="val_loss",
            mode="min",
        )

        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "stage": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def run(
        self,
        stage1_epochs: int = 5,
        stage1_lr: float = 0.001,
        stage2_epochs: int = 10,
        stage2_lr: float = 0.0001,
        unfreeze_from_layer: int = 14,
    ) -> Dict[str, List[Any]]:
        """Executes Stage 1 followed by Stage 2."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("   STARTING TWO-STAGE TRANSFER LEARNING TRAINING")
        logger.info(f"   Target Device: {self.device}")
        logger.info("=" * 60)

        # ----------------------------------------------------
        # STAGE 1: Feature Extraction (Backbone Frozen)
        # ----------------------------------------------------
        logger.info("\n>>> [STAGE 1] Training Classifier Head (Backbone Frozen)...")
        self.model.freeze_backbone()
        params_summary = self.model.get_parameter_summary()
        logger.info(
            f"Trainable Params: {params_summary['trainable_parameters']:,} | "
            f"Frozen: {params_summary['frozen_parameters']:,}"
        )

        # Only pass trainable parameters to optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_s1 = torch.optim.Adam(trainable_params, lr=stage1_lr, weight_decay=1e-4)
        early_stopping_s1 = EarlyStopping(patience=self.early_stopping_patience, mode="min")

        trainer_s1 = ModelTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=optimizer_s1,
            device=self.device,
        )

        current_epoch = 0
        for ep in range(1, stage1_epochs + 1):
            current_epoch += 1
            t_loss, t_acc = trainer_s1.train_epoch(self.train_loader)
            v_loss, v_acc, _, _ = trainer_s1.evaluate(self.val_loader)

            lr = optimizer_s1.param_groups[0]["lr"]
            self._record_history(current_epoch, "stage1", t_loss, t_acc, v_loss, v_acc, lr)

            logger.info(
                f"[Stage 1 | Epoch {ep:02d}/{stage1_epochs:02d}] "
                f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}% | "
                f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}% (lr={lr:.1e})"
            )

            self.checkpoint.step(
                current_value=v_loss,
                model=self.model,
                optimizer=optimizer_s1,
                epoch=current_epoch,
                extra_metadata={"class_mapping": self.class_mapping},
            )

            if early_stopping_s1(v_loss):
                logger.info(f"Stage 1 early stopping at epoch {ep}.")
                break

        # ----------------------------------------------------
        # STAGE 2: Controlled Fine-Tuning (Upper Blocks Unfrozen)
        # ----------------------------------------------------
        if stage2_epochs > 0:
            logger.info(f"\n>>> [STAGE 2] Unfreezing layers from {unfreeze_from_layer} for Fine-Tuning...")
            self.model.unfreeze_backbone(unfreeze_from_layer=unfreeze_from_layer)
            params_summary = self.model.get_parameter_summary()
            logger.info(
                f"Trainable Params: {params_summary['trainable_parameters']:,} | "
                f"Frozen: {params_summary['frozen_parameters']:,}"
            )

            trainable_params_s2 = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_s2 = torch.optim.Adam(trainable_params_s2, lr=stage2_lr, weight_decay=1e-4)
            scheduler_s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_s2, mode="min", factor=0.5, patience=2
            )
            early_stopping_s2 = EarlyStopping(patience=self.early_stopping_patience, mode="min")

            trainer_s2 = ModelTrainer(
                model=self.model,
                criterion=self.criterion,
                optimizer=optimizer_s2,
                device=self.device,
                scheduler=scheduler_s2,
            )

            for ep in range(1, stage2_epochs + 1):
                current_epoch += 1
                t_loss, t_acc = trainer_s2.train_epoch(self.train_loader)
                v_loss, v_acc, _, _ = trainer_s2.evaluate(self.val_loader)

                scheduler_s2.step(v_loss)
                lr = optimizer_s2.param_groups[0]["lr"]
                self._record_history(current_epoch, "stage2", t_loss, t_acc, v_loss, v_acc, lr)

                logger.info(
                    f"[Stage 2 | Epoch {ep:02d}/{stage2_epochs:02d}] "
                    f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}% | "
                    f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}% (lr={lr:.1e})"
                )

                self.checkpoint.step(
                    current_value=v_loss,
                    model=self.model,
                    optimizer=optimizer_s2,
                    epoch=current_epoch,
                    extra_metadata={"class_mapping": self.class_mapping},
                )

                if early_stopping_s2(v_loss):
                    logger.info(f"Stage 2 early stopping at epoch {ep}.")
                    break

        elapsed = time.time() - start_time
        logger.info(f"\n✅ Training completed in {elapsed:.1f}s. Best checkpoint: {self.checkpoint_path}")

        # Save training history JSON
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training metrics history saved to: {self.history_path}")

        return self.history

    def _record_history(
        self,
        epoch: int,
        stage: str,
        t_loss: float,
        t_acc: float,
        v_loss: float,
        v_acc: float,
        lr: float,
    ):
        self.history["epoch"].append(epoch)
        self.history["stage"].append(stage)
        self.history["train_loss"].append(round(t_loss, 4))
        self.history["train_acc"].append(round(t_acc, 2))
        self.history["val_loss"].append(round(v_loss, 4))
        self.history["val_acc"].append(round(v_acc, 2))
        self.history["learning_rate"].append(lr)
