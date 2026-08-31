"""
MobileNetV2 architecture with Transfer Learning for skin disease classification.
Designed for efficient edge/mobile and local macOS Apple Silicon inference and training.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SkinMobileNetV2(nn.Module):
    """
    Skin Disease Classifier built on MobileNetV2 backbone.
    Features:
    - Pretrained ImageNet feature extractor
    - Modular classifier head with Dropout and Batch Normalization
    - Support for feature backbone freezing / unfreezing for multi-stage fine-tuning
    - Direct access to the last conv feature map for Grad-CAM explainability
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_features: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load MobileNetV2 with default weights (ImageNet1K_V2)
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)

        # Feature extractor backbone (19 sequential blocks: 0..18)
        self.features = base_model.features

        # Optionally freeze base feature weights for initial transfer learning
        if freeze_features:
            self.freeze_backbone()

        # In MobileNetV2, the feature extractor outputs 1280 channels
        in_features = base_model.last_channel

        # Custom high-capacity classification head with regularizations
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.features(x)
        # Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze all parameters in the feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("MobileNetV2 feature backbone frozen (Stage 1: Head Training).")

    def unfreeze_backbone(self, unfreeze_from_layer: Optional[int] = 14):
        """
        Fine-tuning utility: unfreeze feature extractor from a given inverted residual block.
        MobileNetV2 has 19 feature blocks (0 to 18).
        unfreeze_from_layer=14 unfreezes the top 5 blocks (14, 15, 16, 17, 18).
        """
        start_idx = unfreeze_from_layer if unfreeze_from_layer is not None else 0
        unfrozen_count = 0
        for i, block in enumerate(self.features):
            requires_grad = i >= start_idx
            for param in block.parameters():
                param.requires_grad = requires_grad
            if requires_grad:
                unfrozen_count += 1

        logger.info(
            f"MobileNetV2 backbone unfrozen from block {start_idx} onward "
            f"({unfrozen_count} blocks active for Stage 2: Fine-Tuning)."
        )

    def get_parameter_summary(self) -> Dict[str, int]:
        """Returns total, trainable, and frozen parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
        }


def build_mobilenet_v2(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_features: bool = True,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SkinMobileNetV2:
    """Factory helper to build and optionally load trained weights."""
    model = SkinMobileNetV2(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_features=freeze_features,
    )

    if weights_path:
        import os
        from pathlib import Path
        p = Path(weights_path)
        if p.exists():
            checkpoint = torch.load(str(p), map_location=device or "cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {weights_path}")

    if device:
        model = model.to(device)

    return model
