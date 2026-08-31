"""Typed configuration loader using PyYAML."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml


@dataclass
class DiseaseClass:
    code: str
    name: str
    severity: str
    description: str


@dataclass
class DatasetConfig:
    name: str
    description: str
    num_classes: int
    classes: List[DiseaseClass]
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    name: str = "mobilenet_v2"
    architecture: str = "MobileNetV2"
    pretrained: bool = True
    num_classes: int = 7
    input_size: Tuple[int, int, int] = (3, 224, 224)
    dropout_rate: float = 0.3
    freeze_features: bool = True
    weights_path: str = "models/mobilenetv2_skin_disease_best.pth"
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    optimizer: str = "Adam"
    lr_scheduler: str = "CosineAnnealingLR"
    early_stopping_patience: int = 5
    device: str = "auto"
    gradcam_target_layer: str = "features.18"
    gradcam_colormap: str = "JET"
    gradcam_alpha: float = 0.4


class ConfigManager:
    """Singleton-style configuration loader."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).resolve().parent.parent / "configs"
        self.config_dir = Path(config_dir)
        self.dataset_config = self._load_dataset_config()
        self.model_config = self._load_model_config()

    def _load_dataset_config(self) -> DatasetConfig:
        yaml_path = self.config_dir / "dataset_config.yaml"
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        classes = [
            DiseaseClass(
                code=c["code"],
                name=c["name"],
                severity=c.get("severity", "Unknown"),
                description=c.get("description", ""),
            )
            for c in raw["dataset"]["classes"]
        ]
        return DatasetConfig(
            name=raw["dataset"]["name"],
            description=raw["dataset"]["description"],
            num_classes=raw["dataset"]["num_classes"],
            classes=classes,
            train_ratio=raw["splits"]["train_ratio"],
            val_ratio=raw["splits"]["val_ratio"],
            test_ratio=raw["splits"]["test_ratio"],
            random_seed=raw["splits"]["random_seed"],
            image_size=tuple(raw["preprocessing"]["image_size"]),
            mean=tuple(raw["preprocessing"]["mean"]),
            std=tuple(raw["preprocessing"]["std"]),
        )

    def _load_model_config(self) -> ModelConfig:
        yaml_path = self.config_dir / "model_config.yaml"
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        m = raw.get("model", {})
        t = raw.get("training", {})
        g = raw.get("gradcam", {})

        return ModelConfig(
            name=m.get("name", "mobilenet_v2"),
            architecture=m.get("architecture", "MobileNetV2"),
            pretrained=m.get("pretrained", True),
            num_classes=m.get("num_classes", 7),
            input_size=tuple(m.get("input_size", [3, 224, 224])),
            dropout_rate=m.get("dropout_rate", 0.3),
            freeze_features=m.get("freeze_features", True),
            weights_path=m.get("weights_path", "models/mobilenetv2_skin_disease_best.pth"),
            batch_size=t.get("batch_size", 32),
            epochs=t.get("epochs", 25),
            learning_rate=t.get("learning_rate", 0.0003),
            weight_decay=t.get("weight_decay", 0.0001),
            optimizer=t.get("optimizer", "Adam"),
            lr_scheduler=t.get("lr_scheduler", "CosineAnnealingLR"),
            early_stopping_patience=t.get("early_stopping_patience", 5),
            device=t.get("device", "auto"),
            gradcam_target_layer=g.get("target_layer", "features.18"),
            gradcam_colormap=g.get("colormap", "JET"),
            gradcam_alpha=g.get("alpha", 0.4),
        )


def get_device(requested_device: str = "auto"):
    """Auto-detect best device for Apple Silicon, CUDA, or fallback to CPU."""
    import torch
    if requested_device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested_device)
