"""FastAPI endpoint route definitions."""
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
import torch

from .schemas import DiseaseClassInfo, ErrorResponse, HealthCheckResponse, PredictionResponse
from ..services.inference_service import InferenceService
from src import __version__
from src.config import ConfigManager

router = APIRouter(prefix="/api")

# Singletons initialized in FastAPI startup lifecycle
_inference_service: Optional[InferenceService] = None
_config_manager: Optional[ConfigManager] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="System and Hardware Health Check",
)
async def health_check():
    """Returns application status, Apple Silicon MPS / CUDA availability, and loaded model metadata."""
    service = get_inference_service()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_avail = torch.cuda.is_available()

    return HealthCheckResponse(
        status="healthy",
        app_name="AI Skin: Intelligent Skin Diseases Detection",
        version=__version__,
        device=str(service.device),
        mps_available=mps_avail,
        cuda_available=cuda_avail,
        model_loaded=service.model is not None,
        weights_path=service.config.model_config.weights_path,
        num_classes=len(service.classes),
    )


@router.get(
    "/classes",
    response_model=List[DiseaseClassInfo],
    tags=["Metadata"],
    summary="Supported Skin Disease Categories",
)
async def get_disease_classes():
    """Returns the 7 supported skin disease classes, clinical names, risk levels, and descriptions."""
    cfg = get_config_manager()
    return [
        DiseaseClassInfo(
            code=c.code,
            name=c.name,
            severity=c.severity,
            description=c.description,
        )
        for c in cfg.dataset_config.classes
    ]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or corrupted image"},
        413: {"model": ErrorResponse, "description": "Image payload exceeds 10MB limit"},
        422: {"model": ErrorResponse, "description": "Unprocessable upload request"},
        500: {"model": ErrorResponse, "description": "Model inference internal server error"},
    },
    tags=["Inference"],
    summary="Classify Skin Lesion Image and Generate Grad-CAM",
)
async def predict_skin_disease(
    file: UploadFile = File(..., description="Skin lesion photograph (JPEG, PNG, WEBP)"),
    target_class: Optional[str] = Form(None, description="Optional class code to explain (e.g. 'mel')"),
    include_gradcam: bool = Form(True, description="Whether to include Grad-CAM overlay"),
):
    """
    Classify a skin lesion photograph using the fine-tuned MobileNetV2 classifier.
    Computes class probabilities, identifies clinical risk, and generates a Grad-CAM
    attention overlay aligned to original image dimensions.
    """
    # 1. MIME format validation
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: '{file.content_type}'. Please upload a JPEG, PNG, or WEBP image.",
        )

    # 2. Read bytes
    image_bytes = await file.read()

    # 3. Byte size validation
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty (0 bytes).",
        )

    max_size = 10 * 1024 * 1024  # 10 MB
    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size exceeds the 10MB upload limit.",
        )

    # 4. Inference & Grad-CAM execution
    try:
        service = get_inference_service()
        result = service.predict(
            image_bytes=image_bytes,
            target_class=target_class,
            include_gradcam=include_gradcam,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation error: {str(val_err)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )
