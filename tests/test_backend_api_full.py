"""Comprehensive test suite for the FastAPI backend inference API."""
import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _generate_valid_image_bytes(width=200, height=200, color=(160, 80, 70)) -> bytes:
    """Generates a small valid JPEG byte stream."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check_endpoint():
    """Verify GET /api/health returns healthy status and loaded model details."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 7
    assert "mps_available" in data


def test_classes_endpoint():
    """Verify GET /api/classes returns metadata for all 7 skin diseases."""
    response = client.get("/api/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 7
    codes = [c["code"] for c in classes]
    assert "mel" in codes
    assert "nv" in codes
    assert "bcc" in codes


def test_predict_valid_image():
    """Verify POST /api/predict with a valid JPEG returns 200, probabilities, and Grad-CAM."""
    img_bytes = _generate_valid_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("lesion.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_code" in data
    assert "predicted_name" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) == 7
    assert data["gradcam_base64"].startswith("data:image/jpeg;base64,")
    assert "disclaimer" in data
    assert "CLINICAL NOTICE" in data["disclaimer"]


def test_predict_with_target_class():
    """Verify POST /api/predict with custom target_class explains that class."""
    img_bytes = _generate_valid_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("lesion.jpg", img_bytes, "image/jpeg")},
        data={"target_class": "mel", "include_gradcam": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["explained_class_code"] == "mel"


def test_predict_invalid_mime_type():
    """Verify POST /api/predict rejects text files with HTTP 400."""
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"Invalid text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_empty_file():
    """Verify POST /api/predict rejects empty files with HTTP 400."""
    response = client.post(
        "/api/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_corrupted_image():
    """Verify POST /api/predict rejects corrupted image bytes with HTTP 400."""
    corrupted_bytes = b"\xff\xd8\xff\xe0" + b"garbage_corrupted_payload" * 50
    response = client.post(
        "/api/predict",
        files={"file": ("corrupt.jpg", corrupted_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Image validation error" in response.json()["detail"]


def test_predict_missing_file():
    """Verify POST /api/predict without file payload triggers HTTP 422."""
    response = client.post("/api/predict")
    assert response.status_code == 422


def test_predict_oversized_image():
    """Verify POST /api/predict rejects images > 10MB with HTTP 413."""
    # 10.5 MB fake payload with JPEG header
    oversized_bytes = b"\xff\xd8\xff\xe0" + b"0" * (10500000)
    response = client.post(
        "/api/predict",
        files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "10MB" in response.json()["detail"]
