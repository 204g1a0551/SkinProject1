"""Pydantic schemas for the AI Skin API."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DiseaseClassInfo(BaseModel):
    code: str = Field(..., description="Canonical short diagnostic code (e.g. 'mel', 'nv', 'bcc')")
    name: str = Field(..., description="Full clinical disease name")
    severity: str = Field(..., description="Clinical risk category: High, Precancerous, Benign, or Moderate")
    description: str = Field(..., description="Brief medical summary of the disease")


class PredictionScore(BaseModel):
    code: str
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability between 0.0 and 1.0")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Probability formatted as percentage")
    severity: str


class PredictionResponse(BaseModel):
    success: bool
    predicted_code: str
    predicted_name: str
    confidence: float
    percentage: float
    severity: str
    description: str
    top_predictions: List[PredictionScore]
    gradcam_base64: Optional[str] = Field(None, description="Base64 data URI for Grad-CAM overlay")
    target_layer: Optional[str] = Field(None, description="Convolutional layer utilized for Grad-CAM")
    explained_class_code: Optional[str] = Field(None, description="Class code explained by Grad-CAM")
    inference_time_ms: float
    device: str
    disclaimer: str = Field(..., description="Official clinical decision-support and non-diagnostic disclaimer")


class HealthCheckResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    mps_available: bool
    cuda_available: bool
    model_loaded: bool
    weights_path: Optional[str] = None
    num_classes: int


class ErrorResponse(BaseModel):
    detail: str
