"""Model factory for loading and configuring models."""
from pathlib import Path
from typing import Optional
import torch

from ..config import ConfigManager, get_device
from .mobilenet_v2 import SkinMobileNetV2, build_mobilenet_v2


class ModelFactory:
    """Central registry & loader for AI Skin models."""

    @staticmethod
    def create_model(
        config_manager: Optional[ConfigManager] = None,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> SkinMobileNetV2:
        if config_manager is None:
            config_manager = ConfigManager()

        cfg = config_manager.model_config
        target_device = device or get_device(cfg.device)

        # Check if custom weights path provided or exists in config
        resolved_weights = weights_path or (
            cfg.weights_path if Path(cfg.weights_path).exists() else None
        )

        model = build_mobilenet_v2(
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            dropout_rate=cfg.dropout_rate,
            freeze_features=cfg.freeze_features,
            weights_path=resolved_weights,
            device=target_device,
        )

        return model
