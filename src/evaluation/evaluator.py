"""
ModelEvaluator engine.
Loads trained weights, sets up deterministic inference pipeline,
evaluates on test dataset, and returns structured predictions and metrics.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import MetricsCalculator
from ..config import ConfigManager, get_device
from ..data.dataset import SkinLesionDataset
from ..data.augmentation import get_validation_transforms
from ..models.mobilenet_v2 import build_mobilenet_v2
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates MobileNetV2 on an untouched test manifest."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/mobilenetv2_skin_disease_best.pth",
        mapping_path: Union[str, Path] = "models/class_mapping.json",
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[ConfigManager] = None,
    ):
        self.config = config or ConfigManager()
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path)

        if isinstance(device, str):
            self.device = get_device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device("auto")

        self.class_mapping, self.class_names, self.class_codes = self._load_class_metadata()
        self.num_classes = len(self.class_names)
        self.model = self._load_model()
        self.calculator = MetricsCalculator(class_names=self.class_names, class_codes=self.class_codes)

    def _load_class_metadata(self) -> Tuple[Dict[str, int], List[str], List[str]]:
        """Load class mapping from JSON or inspect num_classes from checkpoint weights."""
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                state = checkpoint.get("model_state_dict", checkpoint)
                # Check output layer weight shape: [num_classes, 256]
                if "classifier.5.weight" in state:
                    ckpt_classes = state["classifier.5.weight"].shape[0]
                    # If mapping file exists and matches checkpoint classes, use it
                    if self.mapping_path.exists():
                        with open(self.mapping_path, "r") as f:
                            data = json.load(f)
                        classes_info = data.get("classes", [])
                        if len(classes_info) == ckpt_classes:
                            class_names = [c["name"] for c in classes_info]
                            class_codes = [c["code"] for c in classes_info]
                            code_to_idx = data.get("code_to_idx", {c["code"]: c["idx"] for c in classes_info})
                            return code_to_idx, class_names, class_codes

                    # If checkpoint itself stores class_mapping
                    if "class_mapping" in checkpoint and len(checkpoint["class_mapping"]) == ckpt_classes:
                        mapping = checkpoint["class_mapping"]
                        return mapping, list(mapping.keys()), list(mapping.keys())

                    # If dataset config matches
                    if len(self.config.dataset_config.classes) == ckpt_classes:
                        classes = self.config.dataset_config.classes
                        class_names = [c.name for c in classes]
                        class_codes = [c.code for c in classes]
                        code_to_idx = {c.code: i for i, c in enumerate(classes)}
                        return code_to_idx, class_names, class_codes

                    # Fallback generic classes matching checkpoint
                    names = [f"Class_{i}" for i in range(ckpt_classes)]
                    return {n: i for i, n in enumerate(names)}, names, names

        # Fallback to ConfigManager
        classes = self.config.dataset_config.classes
        class_names = [c.name for c in classes]
        class_codes = [c.code for c in classes]
        code_to_idx = {c.code: i for i, c in enumerate(classes)}
        return code_to_idx, class_names, class_codes

    def _load_model(self) -> torch.nn.Module:
        """Load MobileNetV2 with saved weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        model = build_mobilenet_v2(
            num_classes=self.num_classes,
            pretrained=False,
            freeze_features=False,
            weights_path=str(self.model_path),
            device=self.device,
        )
        model.eval()
        logger.info(f"Model successfully loaded on {self.device} from {self.model_path}")
        return model

    @torch.no_grad()
    def evaluate_test_set(
        self,
        test_manifest_path: Union[str, Path] = "data/processed/test_manifest.csv",
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Runs deterministic evaluation on test manifest.
        Uses exact inference preprocessing: Resize(224, 224) -> ToTensor() -> ImageNet Normalize.
        """
        manifest = Path(test_manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Test manifest not found: {manifest}")

        df = pd.read_csv(manifest)
        logger.info(f"Evaluating {len(df)} test samples across {self.num_classes} classes...")

        # Deterministic inference transform
        transform = get_validation_transforms(
            image_size=self.config.dataset_config.image_size,
            mean=self.config.dataset_config.mean,
            std=self.config.dataset_config.std,
        )

        dataset = SkinLesionDataset(df, transform=transform, return_path=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Optimal on macOS
        )

        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[List[float]] = []
        all_paths: List[str] = []

        for batch_imgs, batch_labels, batch_paths in loader:
            batch_imgs = batch_imgs.to(self.device)
            outputs = self.model(batch_imgs)
            probs = torch.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(batch_labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            all_paths.extend(batch_paths)

        # Compute full metrics
        metrics = self.calculator.compute(y_true=all_targets, y_pred=all_preds)

        # Add sample level metadata
        metrics["evaluation_device"] = str(self.device)
        metrics["model_checkpoint"] = str(self.model_path)
        metrics["test_manifest"] = str(manifest)

        return metrics
