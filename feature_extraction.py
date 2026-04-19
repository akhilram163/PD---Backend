"""
feature_extraction.py
---------------------
Extracts the 14 speech biomarker features from a .wav audio file
using praat-parselmouth, librosa, and nolds.

These features must exactly match the FEATURE_COLS used during training.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import parselmouth
    import nolds
    import librosa
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

from data_prep import FEATURE_COLS


def extract_features(file_path: str) -> dict:
    """
    Extract biomarker features from a .wav file.

    Returns a dict mapping feature name → float value,
    with keys matching FEATURE_COLS exactly.
    """
    if not AUDIO_LIBS_AVAILABLE:
        raise ImportError(
            "Audio processing libraries not installed. "
            "Run: pip install praat-parselmouth nolds librosa"
        )

    snd = parselmouth.Sound(file_path)

    # ── Pitch & Point Process ────────────────────────────────────────────────
    point_process = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )

    # ── Jitter features ───────────────────────────────────────────────────────
    jitter_local = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_abs = parselmouth.praat.call(
        point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_rap = parselmouth.praat.call(
        point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_ddp = jitter_rap * 3  # DDP = 3 × RAP by definition

    # ── Shimmer features ──────────────────────────────────────────────────────
    shimmer_local = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_db = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq3 = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq5 = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_dda = shimmer_apq3 * 3  # DDA = 3 × APQ3 by definition

    # ── HNR / NHR ─────────────────────────────────────────────────────────────
    harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    nhr = 1 / (10 ** (hnr / 10)) if hnr > 0 else np.nan  # Convert HNR (dB) → NHR ratio

    # ── RPDE (Recurrence Period Density Entropy) ──────────────────────────────
    # Approximated via sample entropy as a proxy
    y_audio, sr = librosa.load(file_path, sr=None, mono=True)
    amplitude_envelope = librosa.onset.onset_strength(y=y_audio, sr=sr)

    try:
        rpde = nolds.sampen(amplitude_envelope)
    except Exception:
        rpde = np.nan

    # ── DFA (Detrended Fluctuation Analysis) ──────────────────────────────────
    try:
        dfa = nolds.dfa(amplitude_envelope)
    except Exception:
        dfa = np.nan

    # ── PPE (Pitch Period Entropy) ────────────────────────────────────────────
    # Approximated via spectral entropy of the pitch contour
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
    if len(pitch_values) > 1:
        pitch_hist, _ = np.histogram(pitch_values, bins=30, density=True)
        pitch_hist = pitch_hist[pitch_hist > 0]
        ppe = -np.sum(pitch_hist * np.log2(pitch_hist)) / np.log2(len(pitch_hist))
    else:
        ppe = np.nan

    features = {
        "MDVP:Jitter(%)":    jitter_local * 100,
        "MDVP:Jitter(Abs)":  jitter_abs,
        "MDVP:RAP":          jitter_rap,
        "Jitter:DDP":        jitter_ddp,
        "MDVP:Shimmer":      shimmer_local,
        "MDVP:Shimmer(dB)":  shimmer_db,
        "Shimmer:APQ3":      shimmer_apq3,
        "Shimmer:APQ5":      shimmer_apq5,
        "Shimmer:DDA":       shimmer_dda,
        "NHR":               nhr,
        "HNR":               hnr,
        "RPDE":              rpde,
        "DFA":               dfa,
        "PPE":               ppe,
    }

    return features


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to a numpy array in the correct column order."""
    return np.array([[features[col] for col in FEATURE_COLS]])
