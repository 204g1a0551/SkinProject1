"""
High-level Explainability Service.
Handles:
- Image loading (path, bytes, or PIL Image)
- Validation and graceful error handling
- Exact inference preprocessing alignment
- Dynamic Grad-CAM computation for predicted or user-selected target class
- Resizing heatmaps to original image dimensions
- Exporting: original image, standalone heatmap, overlay image, and decision-support disclaimer
"""
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .gradcam import GradCAM, apply_colormap_on_image, create_standalone_heatmap_image
from ..config import ConfigManager, get_device
from ..data.preprocessor import ImagePreprocessor
from ..data.validator import ImageValidator
from ..models.factory import ModelFactory
from ..utils.logger import get_logger

logger = get_logger(__name__)

CLINICAL_DECISION_SUPPORT_DISCLAIMER = (
    "DISCLAIMER: Grad-CAM visualizations indicate morphological feature regions that "
    "positively influenced the deep-learning model's classification score. "
    "These heatmaps are visual decision-support evidence and DO NOT constitute medical proof, "
    "definitive histological diagnosis, or a replacement for expert dermatological evaluation and biopsy."
)


class GradCAMExplainer:
    """End-to-end explainability service for skin lesion models."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        config: Optional[ConfigManager] = None,
        weights_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_layer: Optional[Union[torch.nn.Module, str]] = None,
    ):
        self.config = config or ConfigManager()

        if isinstance(device, str):
            self.device = get_device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(self.config.model_config.device)

        # Load or use passed model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = ModelFactory.create_model(
                config_manager=self.config,
                weights_path=weights_path,
                device=self.device,
            )
        self.model.eval()

        self.gradcam = GradCAM(model=self.model, target_layer=target_layer)
        self.validator = ImageValidator()

        self.preprocessor = ImagePreprocessor(
            image_size=self.config.dataset_config.image_size,
            mean=self.config.dataset_config.mean,
            std=self.config.dataset_config.std,
        )

        self.classes = self.config.dataset_config.classes
        self.code_to_idx = {c.code: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

    def explain_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        target_class: Optional[Union[int, str]] = None,
        colormap_name: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates full explainability output for an image.

        Args:
            image_input: File path, raw bytes, or PIL Image.
            target_class: Class code (e.g. 'mel') or class index (e.g. 0). If None, predicted class is used.
            colormap_name: Colormap for heatmap (default from config, e.g. 'jet').
            alpha: Transparency weight for overlay (0.0 to 1.0).
        """
        cmap = (colormap_name or self.config.model_config.gradcam_colormap).lower()
        overlay_alpha = alpha if alpha is not None else self.config.model_config.gradcam_alpha

        # 1. Load and validate original image
        pil_img, original_size, source_desc = self._load_and_validate_image(image_input)

        # 2. Apply deterministic inference preprocessing
        tensor_img = self.preprocessor.preprocess_pil(pil_img).unsqueeze(0).to(self.device)

        # 3. Determine top predicted class first
        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = F.softmax(logits, dim=1)
            top_pred_idx = int(torch.argmax(probs, dim=1).item())

        # 4. Resolve target class index
        target_idx = top_pred_idx
        if target_class is not None:
            if isinstance(target_class, str):
                if target_class.lower() in self.code_to_idx:
                    target_idx = self.code_to_idx[target_class.lower()]
                else:
                    # Check if matching class name
                    matched = False
                    for i, c in enumerate(self.classes):
                        if c.name.lower() == target_class.lower():
                            target_idx = i
                            matched = True
                            break
                    if not matched:
                        raise ValueError(f"Unknown target class: '{target_class}'. Valid codes: {list(self.code_to_idx.keys())}")
            elif isinstance(target_class, int):
                if 0 <= target_class < len(self.classes):
                    target_idx = target_class
                else:
                    raise ValueError(f"Target class index {target_class} out of bounds (0..{len(self.classes)-1}).")

        # 5. Generate Grad-CAM heatmap matching original dimensions
        heatmap_np, target_class_idx, confidence = self.gradcam.generate_heatmap(
            input_tensor=tensor_img,
            target_class_idx=target_idx,
            target_size=original_size,
        )

        # 6. Generate Visual Artifacts
        heatmap_img = create_standalone_heatmap_image(heatmap_np, colormap_name=cmap, target_size=original_size)
        overlay_img = apply_colormap_on_image(pil_img, heatmap_np, colormap_name=cmap, alpha=overlay_alpha)

        # 7. Retrieve diagnostic details
        target_info = self.idx_to_class.get(target_class_idx, None)
        pred_info = self.idx_to_class.get(top_pred_idx, None)

        return {
            "success": True,
            "source": source_desc,
            "original_dimensions": list(original_size),
            "target_layer": self.gradcam.target_layer_name,
            "predicted_class": {
                "idx": top_pred_idx,
                "code": pred_info.code if pred_info else str(top_pred_idx),
                "name": pred_info.name if pred_info else f"Class {top_pred_idx}",
                "severity": pred_info.severity if pred_info else "Unknown",
            },
            "explained_class": {
                "idx": target_class_idx,
                "code": target_info.code if target_info else str(target_class_idx),
                "name": target_info.name if target_info else f"Class {target_class_idx}",
                "confidence": round(confidence, 4),
                "percentage": round(confidence * 100, 2),
                "severity": target_info.severity if target_info else "Unknown",
            },
            "is_top_prediction": (target_class_idx == top_pred_idx),
            "heatmap_array": heatmap_np,
            "original_pil": pil_img,
            "heatmap_pil": heatmap_img,
            "overlay_pil": overlay_img,
            "disclaimer": CLINICAL_DECISION_SUPPORT_DISCLAIMER,
        }

    def save_explanation_artifacts(
        self,
        explanation: Dict[str, Any],
        output_dir: Union[str, Path] = "reports/explainability",
        prefix: str = "lesion",
    ) -> Dict[str, Path]:
        """Save original, standalone heatmap, and overlay images to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_path = out_dir / f"{prefix}_original.jpg"
        heat_path = out_dir / f"{prefix}_heatmap.jpg"
        over_path = out_dir / f"{prefix}_overlay.jpg"
        meta_path = out_dir / f"{prefix}_metadata.json"

        # Save images
        explanation["original_pil"].save(orig_path, "JPEG", quality=95)
        explanation["heatmap_pil"].save(heat_path, "JPEG", quality=95)
        explanation["overlay_pil"].save(over_path, "JPEG", quality=95)

        # Save metadata JSON (excluding raw PIL and numpy)
        serializable_meta = {
            "source": explanation.get("source"),
            "original_dimensions": explanation.get("original_dimensions"),
            "target_layer": explanation.get("target_layer"),
            "predicted_class": explanation.get("predicted_class"),
            "explained_class": explanation.get("explained_class"),
            "is_top_prediction": explanation.get("is_top_prediction"),
            "disclaimer": explanation.get("disclaimer"),
            "artifacts": {
                "original_image": str(orig_path.name),
                "heatmap_image": str(heat_path.name),
                "overlay_image": str(over_path.name),
            },
        }

        with open(meta_path, "w") as f:
            json.dump(serializable_meta, f, indent=2)

        return {
            "original": orig_path,
            "heatmap": heat_path,
            "overlay": over_path,
            "metadata": meta_path,
        }

    def _load_and_validate_image(
        self, image_input: Union[str, Path, bytes, Image.Image]
    ) -> Tuple[Image.Image, Tuple[int, int], str]:
        """Validates and loads image input into PIL Image."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            is_valid, reason = self.validator.validate_file(p)
            if not is_valid:
                raise ValueError(f"Invalid image file '{p}': {reason}")
            img = Image.open(p).convert("RGB")
            return img, img.size, str(p.name)

        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise ValueError("Empty image bytes provided.")
            try:
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
                return img, img.size, "raw_bytes"
            except Exception as e:
                raise ValueError(f"Failed to decode image bytes: {e}")

        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
            return img, img.size, "pil_image"

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
