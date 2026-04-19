"""
predict.py
----------
Core inference engine. Takes a .wav file path, extracts features,
runs the model, and returns a structured prediction result.

This is the module your FastAPI backend will import.

Usage (standalone):
    python predict.py --audio path/to/voice.wav --model model.joblib
"""

import argparse
import joblib
import numpy as np
from typing import Optional

from data_prep import FEATURE_COLS, LABEL_MAP
from feature_extraction import extract_features, features_to_array


class PredictionResult:
    def __init__(
        self,
        label: str,
        label_id: int,
        confidence: float,
        probabilities: dict,
        features: dict,
    ):
        self.label = label
        self.label_id = label_id
        self.confidence = confidence          # probability of predicted class
        self.probabilities = probabilities    # {class_name: probability}
        self.features = features              # raw extracted biomarkers

    def to_dict(self) -> dict:
        return {
            "prediction": self.label,
            "confidence": round(self.confidence * 100, 1),
            "probabilities": {k: round(v * 100, 1) for k, v in self.probabilities.items()},
            "biomarkers": {k: round(float(v), 6) for k, v in self.features.items()},
        }

    def __repr__(self):
        return (
            f"Prediction: {self.label} "
            f"(confidence: {self.confidence*100:.1f}%)\n"
            f"Probabilities: {self.probabilities}"
        )


def load_model(model_path: str = "model.joblib"):
    """Load the trained pipeline from disk."""
    return joblib.load(model_path)


def predict_from_audio(audio_path: str, model) -> PredictionResult:
    """
    Full end-to-end prediction from a .wav file.

    Args:
        audio_path: Path to the .wav recording
        model: Loaded sklearn pipeline

    Returns:
        PredictionResult with label, confidence, probabilities, and raw features
    """
    # Extract biomarkers from audio
    features = extract_features(audio_path)

    # Handle any NaN values by using 0 as a safe fallback
    for k, v in features.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            features[k] = 0.0

    # Convert to model input format
    X = features_to_array(features)

    # Run model
    label_id = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]

    # Map probabilities to class names
    # model.classes_ gives the order of classes
    classes = model.classes_
    prob_dict = {LABEL_MAP[int(c)]: float(probas[i]) for i, c in enumerate(classes)}

    return PredictionResult(
        label=LABEL_MAP[label_id],
        label_id=label_id,
        confidence=float(probas[list(classes).index(label_id)]),
        probabilities=prob_dict,
        features=features,
    )


def predict_from_features(feature_dict: dict, model) -> PredictionResult:
    """
    Predict directly from a pre-extracted feature dictionary.
    Useful for testing without audio files.
    """
    X = features_to_array(feature_dict)
    label_id = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]
    classes = model.classes_
    prob_dict = {LABEL_MAP[int(c)]: float(probas[i]) for i, c in enumerate(classes)}

    return PredictionResult(
        label=LABEL_MAP[label_id],
        label_id=label_id,
        confidence=float(probas[list(classes).index(label_id)]),
        probabilities=prob_dict,
        features=feature_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSEF - EOPD Prediction from Voice")
    parser.add_argument("--audio", required=True, help="Path to .wav audio file")
    parser.add_argument("--model", default="model.joblib", help="Path to saved model")
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    print(f"Processing audio: {args.audio}")
    result = predict_from_audio(args.audio, model)

    print("\n" + "="*50)
    print(f"  RESULT: {result.label}")
    print(f"  Confidence: {result.confidence*100:.1f}%")
    print("  Class Probabilities:")
    for cls, prob in result.probabilities.items():
        print(f"    {cls}: {prob*100:.1f}%")
    print("="*50)
