"""
FastAPI application entrypoint for AI Skin: Intelligent Skin Diseases Detection.
Configures lifespan singleton warming, CORS middleware, OpenAPI documentation, and route routing.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router, get_inference_service
from src import __version__
from src.utils.logger import get_logger

logger = get_logger("backend")

API_TITLE = "AI Skin: Intelligent Skin Diseases Detection API"
API_DESCRIPTION = """
### Academic Final-Year Project: AI Skin — Intelligent Skin Diseases Detection

Deep-learning REST API leveraging **MobileNetV2 Transfer Learning** and **Grad-CAM (Gradient-Weighted Class Activation Mapping)** for 7 skin lesion categories:
- **Melanoma** (`mel`) [High Risk]
- **Melanocytic Nevi** (`nv`) [Benign]
- **Basal Cell Carcinoma** (`bcc`) [Moderate Risk]
- **Actinic Keratoses / Intraepithelial Carcinoma** (`akiec`) [Precancerous]
- **Benign Keratosis-like Lesions** (`bkl`) [Benign]
- **Dermatofibroma** (`df`) [Benign]
- **Vascular Lesions** (`vasc`) [Benign]

#### Hardware Acceleration:
- **Apple Silicon MPS (Metal Performance Shaders)** enabled on macOS Darwin
- **NVIDIA CUDA** enabled on compatible Linux/Windows systems
- **CPU** safe fallback

> **DISCLAIMER**: This software is intended strictly for preliminary clinical decision-support and academic research. It does NOT provide a certified medical diagnosis.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to warm up model weights as a persistent singleton."""
    logger.info("Initializing AI Skin deep-learning backend...")
    get_inference_service()
    logger.info("AI Skin backend ready to serve inference requests.")
    yield
    logger.info("Shutting down AI Skin backend...")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Vite development and external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include backend API routes under /api
app.include_router(api_router)

# If frontend/dist exists, mount static files to serve the complete full-stack app
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root path to interactive Swagger documentation."""
        return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
