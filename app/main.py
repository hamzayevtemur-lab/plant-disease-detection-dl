"""
FastAPI backend for Plant Disease Detection & Agricultural Diagnosis.

Endpoints:
  GET  /               — Serves the SPA frontend
  POST /predict        — Predicts crop & disease from uploaded leaf image
  POST /predict-url    — Predicts crop & disease from an image URL
  GET  /models         — Lists available trained models
  POST /models/switch  — Switches active model in real-time
  GET  /benchmark      — Returns benchmark metrics and leaderboard
  GET  /health         — Service health check
"""

import os
import sys
import csv
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.model_loader import get_model_manager

FRONTEND_DIR = Path(__file__).parent / "frontend"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

os.makedirs(FRONTEND_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

app = FastAPI(
    title="🌿 Plant Disease AI Classifier",
    description="Deep Learning API for classifying 38 crop leaf diseases with treatment remedies.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


@app.on_event("startup")
async def startup_event():
    """Eagerly load the best available model on server launch."""
    manager = get_model_manager()
    try:
        manager.load_model()
    except Exception as e:
        print(f"Warning: Could not load initial model on startup: {e}")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file and returns crop disease diagnosis and remedies."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image (JPEG/PNG/WebP).")

    try:
        image_bytes = await file.read()
        manager = get_model_manager()
        result = manager.predict(image_bytes)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


class PredictUrlRequest(BaseModel):
    url: str


@app.post("/predict-url")
async def predict_url(payload: PredictUrlRequest):
    """Downloads an image from a URL and returns crop disease diagnosis and remedies."""
    url = payload.url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            image_bytes = response.read()

        manager = get_model_manager()
        result = manager.predict(image_bytes)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load or process image from URL: {str(e)}")


@app.get("/models")
async def get_models():
    """Returns all available models in the experiments folder."""
    manager = get_model_manager()
    models = manager.get_available_models()
    return {
        "active_model": manager.active_model_name,
        "models": models
    }


class SwitchModelRequest(BaseModel):
    model_name: str


@app.post("/models/switch")
async def switch_model(payload: SwitchModelRequest):
    """Dynamically switches the active inference model."""
    manager = get_model_manager()
    try:
        active = manager.load_model(payload.model_name)
        return {"status": "success", "active_model": active}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/benchmark")
async def get_benchmark():
    """Returns the benchmark leaderboard from experiments/results.csv."""
    results_csv = EXPERIMENTS_DIR / "results.csv"
    if not results_csv.exists():
        return {"results": []}

    results = []
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return {"results": results}


@app.get("/health")
async def health():
    manager = get_model_manager()
    return {
        "status": "healthy",
        "active_model": manager.active_model_name,
        "device": str(manager.device)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

