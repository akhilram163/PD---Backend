"""
main.py
-------
FastAPI backend for BSEF — Early-Onset Parkinson's Disease Detection.

Endpoints:
    GET  /           → health check
    POST /predict    → upload a .wav file, get back a prediction
"""

import os
import uuid
import shutil
import joblib
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import predict_from_audio, load_model

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BSEF — Parkinson's Speech Screening API",
    description="Upload a voice recording and receive a screening result for Early-Onset Parkinson's Disease.",
    version="1.0.0",
)

# ── CORS — allows your React website to talk to this backend ─────────────────
# During development this allows all origins.
# Before going fully public, replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup (not on every request) ────────────────────────
MODEL_PATH = Path(__file__).parent / "model.joblib"

@app.on_event("startup")
def load_ml_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = load_model(str(MODEL_PATH))
    print(f"[startup] ✅ Model loaded from {MODEL_PATH}")

# ── Temp folder for uploaded audio files ─────────────────────────────────────
TEMP_DIR = Path("/tmp/bsef_audio")
TEMP_DIR.mkdir(exist_ok=True)

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str           # "Healthy", "Parkinson's Disease", "Early-Onset Parkinson's Disease"
    confidence: float         # e.g. 91.3  (percentage)
    probabilities: dict       # {"Healthy": 2.1, "Parkinson's Disease": 6.6, "Early-Onset...": 91.3}
    biomarkers: dict          # raw extracted feature values
    disclaimer: str

DISCLAIMER = (
    "This tool is a research-grade screening aid only. "
    "It is NOT a medical diagnosis. Please consult a licensed neurologist "
    "for any concerns about Parkinson's Disease."
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Simple health check — confirms the server is running."""
    return {"status": "ok", "message": "BSEF API is running."}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a .wav audio file and receive a Parkinson's screening result.

    - Accepts: .wav files only
    - Returns: prediction label, confidence, class probabilities, and raw biomarkers
    """

    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename.endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav audio files are accepted. Please convert your recording to .wav format."
        )

    # ── Save uploaded file to a temp path ─────────────────────────────────────
    temp_filename = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")

    # ── Run prediction ─────────────────────────────────────────────────────────
    try:
        result = predict_from_audio(str(temp_filename), model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction or prediction failed: {str(e)}"
        )
    finally:
        # Always clean up the temp file
        if temp_filename.exists():
            temp_filename.unlink()

    # ── Return response ────────────────────────────────────────────────────────
    return PredictionResponse(
        prediction=result.label,
        confidence=round(result.confidence * 100, 1),
        probabilities={k: round(v * 100, 1) for k, v in result.probabilities.items()},
        biomarkers={k: round(float(v), 6) for k, v in result.features.items()},
        disclaimer=DISCLAIMER,
    )
