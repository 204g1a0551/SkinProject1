"""
Production Inference Service for AI Skin.
Connects trained MobileNetV2, ImagePreprocessor, ImageValidator, and Grad-CAM Explainer.
Supports Apple Silicon MPS hardware acceleration and persistent singleton lifecycle.
"""
import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import ConfigManager, get_device
from src.data.preprocessor import ImagePreprocessor
from src.data.validator import ImageValidator
from src.explainability.gradcam import GradCAM, apply_colormap_on_image
from src.explainability.explainer import CLINICAL_DECISION_SUPPORT_DISCLAIMER
from src.models.factory import ModelFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRELIMINARY_SCREENING_DISCLAIMER = (
    "CLINICAL NOTICE: This AI diagnostic tool is developed solely for academic research and preliminary "
    "clinical decision support. Predictions and Grad-CAM saliency maps DO NOT constitute a definitive medical diagnosis, "
    "histopathological proof, or a replacement for an in-person evaluation by a board-certified dermatologist."
)


class InferenceService:
    """Persistent singleton inference service with caching and input validation."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.device = get_device(self.config.model_config.device)
        logger.info(f"Initializing InferenceService on device: {self.device}")

        self.preprocessor = ImagePreprocessor(
            image_size=self.config.dataset_config.image_size,
            mean=self.config.dataset_config.mean,
            std=self.config.dataset_config.std,
        )
        self.validator = ImageValidator()

        # Load model using ModelFactory (loads weights if available)
        self.model = ModelFactory.create_model(
            config_manager=self.config,
            device=self.device,
        )
        self.model.eval()

        self.gradcam = GradCAM(model=self.model)
        self.classes = self.config.dataset_config.classes
        self.code_to_idx = {c.code: i for i, c in enumerate(self.classes)}
        logger.info(
            f"InferenceService active with {len(self.classes)} classes. "
            f"Grad-CAM target layer: '{self.gradcam.target_layer_name}'"
        )

    def validate_image_bytes(self, image_bytes: bytes, max_size_bytes: int = 10 * 1024 * 1024) -> Image.Image:
        """
        Validates byte stream size, raster integrity, and format.
        Returns decoded PIL Image.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded image file is empty (0 bytes).")

        if len(image_bytes) > max_size_bytes:
            raise ValueError(f"Image payload exceeds maximum allowed size ({max_size_bytes // (1024*1024)}MB).")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()  # Verifies file integrity
            # Reopen after verify (verify closes the stream)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Image corruption or unreadable format: {str(e)}")

        return pil_img

    def predict(
        self,
        image_bytes: bytes,
        target_class: Optional[Union[str, int]] = None,
        include_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline:
        Validation -> Preprocessing -> Forward Pass -> Softmax Probabilities -> Grad-CAM Overlay -> Response Dict.
        """
        start_time = time.perf_counter()

        # 1. Validate and decode
        pil_img = self.validate_image_bytes(image_bytes)
        original_size = pil_img.size

        # 2. Preprocess
        tensor_img = self.preprocessor.preprocess_pil(pil_img).unsqueeze(0).to(self.device)

        # 3. Model forward pass
        with torch.no_grad():
            logits = self.model(tensor_img)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_pred_idx = int(probabilities.argmax())
        top_conf = float(probabilities[top_pred_idx])
        best_class = self.classes[top_pred_idx]

        # 4. Resolve Grad-CAM target class
        target_idx = top_pred_idx
        if target_class is not None:
            if isinstance(target_class, str) and target_class.lower() in self.code_to_idx:
                target_idx = self.code_to_idx[target_class.lower()]
            elif isinstance(target_class, int) and 0 <= target_class < len(self.classes):
                target_idx = target_class

        # 5. Generate Grad-CAM visualization if requested
        gradcam_b64 = None
        if include_gradcam:
            try:
                heatmap, _, _ = self.gradcam.generate_heatmap(
                    input_tensor=tensor_img,
                    target_class_idx=target_idx,
                    target_size=original_size,
                )
                blended_img = apply_colormap_on_image(
                    org_img=pil_img,
                    activation_map=heatmap,
                    colormap_name=self.config.model_config.gradcam_colormap.lower(),
                    alpha=self.config.model_config.gradcam_alpha,
                )
                buffered = io.BytesIO()
                blended_img.save(buffered, format="JPEG", quality=90)
                gradcam_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            except Exception as e:
                logger.error(f"Grad-CAM generation error: {e}", exc_info=True)

        # 6. Assemble ranked top predictions
        ranked_indices = probabilities.argsort()[::-1]
        top_predictions = []
        for idx in ranked_indices:
            cls_info = self.classes[idx]
            conf = float(probabilities[idx])
            top_predictions.append({
                "code": cls_info.code,
                "name": cls_info.name,
                "confidence": round(conf, 4),
                "percentage": round(conf * 100, 2),
                "severity": cls_info.severity,
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": True,
            "predicted_code": best_class.code,
            "predicted_name": best_class.name,
            "confidence": round(top_conf, 4),
            "percentage": round(top_conf * 100, 2),
            "severity": best_class.severity,
            "description": best_class.description,
            "top_predictions": top_predictions,
            "gradcam_base64": gradcam_b64,
            "target_layer": self.gradcam.target_layer_name,
            "explained_class_code": self.classes[target_idx].code,
            "inference_time_ms": round(elapsed_ms, 2),
            "device": str(self.device),
            "disclaimer": PRELIMINARY_SCREENING_DISCLAIMER,
        }
