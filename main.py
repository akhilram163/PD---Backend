"""
main.py
-------
FastAPI backend for BSEF — Early-Onset Parkinson's Disease Detection.

On startup, trains the model from the dataset files if model.joblib
doesn't exist yet. This avoids needing to store a large model file on GitHub.

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

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BSEF — Parkinson's Speech Screening API",
    description="Upload a voice recording and receive a screening result for Early-Onset Parkinson's Disease.",
    version="1.0.0",
)

# ── CORS — allows your React website to talk to this backend ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
DF1_PATH   = BASE_DIR / "parkinsons.data"
DF2_PATH   = BASE_DIR / "parkinsons_updrs.data"

# ── Load or train model on startup ───────────────────────────────────────────
model = None

@app.on_event("startup")
def load_or_train_model():
    global model

    if MODEL_PATH.exists():
        print("[startup] Loading existing model...")
        model = joblib.load(str(MODEL_PATH))
        print("[startup] ✅ Model loaded.")
    else:
        print("[startup] model.joblib not found — training from datasets...")
        if not DF1_PATH.exists() or not DF2_PATH.exists():
            raise RuntimeError(
                "Dataset files missing. Make sure parkinsons.data and "
                "parkinsons_updrs.data are in the repo."
            )
        from train import train
        model = train(str(DF1_PATH), str(DF2_PATH), str(MODEL_PATH))
        print("[startup] ✅ Model trained and saved.")

# ── Temp folder for uploaded audio files ─────────────────────────────────────
TEMP_DIR = Path("/tmp/bsef_audio")
TEMP_DIR.mkdir(exist_ok=True)

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    biomarkers: dict
    disclaimer: str

DISCLAIMER = (
    "This tool is a research-grade screening aid only. "
    "It is NOT a medical diagnosis. Please consult a licensed neurologist "
    "for any concerns about Parkinson's Disease."
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "BSEF API is running."}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a .wav audio file and receive a Parkinson's screening result.
    """
    if not file.filename.endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav audio files are accepted."
        )

    temp_filename = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")

    try:
        from predict import predict_from_audio
        result = predict_from_audio(str(temp_filename), model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        if temp_filename.exists():
            temp_filename.unlink()

    return PredictionResponse(
        prediction=result.label,
        confidence=round(result.confidence * 100, 1),
        probabilities={k: round(v * 100, 1) for k, v in result.probabilities.items()},
        biomarkers={k: round(float(v), 6) for k, v in result.features.items()},
        disclaimer=DISCLAIMER,
    )
