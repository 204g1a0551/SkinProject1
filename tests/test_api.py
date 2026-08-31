"""Test FastAPI REST endpoints."""
import io
from PIL import Image
from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mps_available" in data
    assert data["num_classes"] == 7


def test_classes_endpoint():
    response = client.get("/api/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 7
    codes = [c["code"] for c in classes]
    assert "mel" in codes
    assert "nv" in codes


def test_predict_endpoint():
    # Generate test image
    img = Image.new("RGB", (224, 224), color=(190, 110, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_skin.jpg", buf, "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_name" in data
    assert "gradcam_base64" in data
    assert len(data["top_predictions"]) == 7
