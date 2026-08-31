# AI Skin: Intelligent Skin Diseases Detection — Backend API

Production-ready **FastAPI** REST API serving the **MobileNetV2** Transfer Learning model with **Grad-CAM explainability** and Apple Silicon MPS acceleration.

---

## 🚀 Quick Start on macOS

### 1. Prerequisites
Ensure you have created the project environment with Python 3.11:
```bash
cd /Users/maheshkumar/.gemini/antigravity/scratch/ai-skin-intelligent-skin-disease-detection
source venv/bin/activate
```

### 2. Launch the Development Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

When started, the API automatically warms up the model on Apple Silicon MPS (`mps:0`) or CPU:
```
INFO:     Initializing AI Skin deep-learning backend...
INFO:     InferenceService active with 7 classes. Grad-CAM target layer: 'features.18.0'
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 📚 API Endpoints & Documentation

Interactive API documentation is automatically generated:
* **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Reference**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### 1. `GET /api/health`
Checks backend service health, device acceleration, and loaded model metadata.

**Example Request:**
```bash
curl -s http://127.0.0.1:8000/api/health | jq
```

**Example Response:**
```json
{
  "status": "healthy",
  "app_name": "AI Skin: Intelligent Skin Diseases Detection",
  "version": "0.1.0",
  "device": "mps",
  "mps_available": true,
  "cuda_available": false,
  "model_loaded": true,
  "weights_path": "models/mobilenetv2_skin_disease_best.pth",
  "num_classes": 7
}
```

---

### 2. `GET /api/classes`
Retrieves metadata, diagnostic codes, risk severities, and descriptions for all 7 skin diseases.

**Example Request:**
```bash
curl -s http://127.0.0.1:8000/api/classes | jq
```

---

### 3. `POST /api/predict`
Uploads a skin lesion photograph for diagnostic classification and Grad-CAM saliency mapping.

* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  - `file` (*required*): Image binary (JPEG, PNG, WEBP $\le$ 10MB)
  - `target_class` (*optional*): Specific class code to explain (e.g. `mel`)
  - `include_gradcam` (*optional*, default `true`): Include Grad-CAM overlay data URI

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@data/sample_images/sample_lesion.jpg" | jq
```

**Example Response:**
```json
{
  "success": true,
  "predicted_code": "akiec",
  "predicted_name": "Actinic Keratoses / Intraepithelial Carcinoma",
  "confidence": 0.1839,
  "percentage": 18.39,
  "severity": "Precancerous",
  "description": "Precancerous scaly or crusty growths commonly caused by chronic UV sun exposure.",
  "top_predictions": [
    {
      "code": "akiec",
      "name": "Actinic Keratoses / Intraepithelial Carcinoma",
      "confidence": 0.1839,
      "percentage": 18.39,
      "severity": "Precancerous"
    }
  ],
  "gradcam_base64": "data:image/jpeg;base64,...",
  "target_layer": "features.18.0",
  "explained_class_code": "akiec",
  "inference_time_ms": 35.2,
  "device": "mps",
  "disclaimer": "CLINICAL NOTICE: This AI diagnostic tool is developed solely for academic research..."
}
```

---

## 🛡️ Input Validation & HTTP Status Codes

| Status Code | Reason | Example Trigger |
| :--- | :--- | :--- |
| **200 OK** | Successful inference | Valid skin lesion photograph |
| **400 Bad Request** | Invalid format or corrupted file | Uploading `.txt`, `.pdf`, empty file, or truncated byte stream |
| **413 Payload Too Large** | Size limit exceeded | Uploading file $> 10\text{ MB}$ |
| **422 Unprocessable** | Missing file field | Uploading without the required `file` multipart key |
| **500 Server Error** | Model pipeline fault | Internal execution error |

---

## ⚖️ Clinical Disclaimer
All API responses explicitly include the clinical decision-support notice:
> **CLINICAL NOTICE**: This AI diagnostic tool is developed solely for academic research and preliminary clinical decision support. Predictions and Grad-CAM saliency maps DO NOT constitute a definitive medical diagnosis, histopathological proof, or a replacement for an in-person evaluation by a board-certified dermatologist.
