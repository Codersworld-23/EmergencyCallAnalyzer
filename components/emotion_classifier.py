"""
components/emotion_classifier.py
────────────────────────────────
Lightweight MFCC‑based emotion classifier with a **distress heuristic**
so that flat, traumatised voices in emergency calls are not blindly
labelled "Neutral".

• Loads your existing `ravdess_final_model.h5` (RAVDESS‑trained CNN).
• Optional `context_dict` lets you pass the full‑call transcript and
background events; the heuristic will upgrade Neutral→Fear when
other danger signals are present.
• Fully backward‑compatible with the old signature: you can still call
`classify_emotion(audio_bytes)` and it works as before.

Heuristic Rules (Option 1)
--------------------------
If the raw model predicts **Neutral** but either of these is true:
1. The transcript contains distress keywords (help, emergency, fire…)
2. YAMNet detected emergency sounds (Siren / Gunshot / Scream) above 0.2
then the emotion is forced to **Fear** with confidence ≥ 0.55.

Dependencies
------------
tensorflow, librosa, soundfile, numpy, tempfiles.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Tuple, List

import numpy as np
import librosa
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------
#  Model & labels
# ---------------------------------------------------------------------
_MODEL = load_model("ravdess_final_model.h5")
_LABELS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise", "Calm",
]

# ---------------------------------------------------------------------
#  Feature helpers
# ---------------------------------------------------------------------
def _extract_features(y: np.ndarray, sr: int, max_len: int = 130):
    """Return MFCC feature matrix (max_len × 120)."""
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=120)
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs.T

def _preprocess_audio(y: np.ndarray) -> np.ndarray:
    """Trim silence and peak‑normalise."""
    y, _ = librosa.effects.trim(y, top_db=25)
    if np.max(np.abs(y)) > 0:
        y = 0.98 * y / np.max(np.abs(y))
    return y

def _predict_chunks(y: np.ndarray, sr: int, win_sec: float = 2.0, hop_sec: float = 1.0):
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    preds = []
    for start in range(0, len(y) - win_len + 1, hop_len):
        feat = _extract_features(y[start:start + win_len], sr)
        feat = np.expand_dims(feat, 0)
        preds.append(_MODEL.predict(feat, verbose=0)[0])

    if not preds:  # very short audio fallback
        feat = _extract_features(y, sr)
        preds.append(_MODEL.predict(np.expand_dims(feat, 0), verbose=0)[0])

    return np.mean(preds, axis=0)

# ---------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------
def classify_emotion(
    audio_bytes: bytes,
    *,
    filename: str | None = None,
    context_dict: Dict | None = None,
) -> Tuple[str, float, Dict[str, float]]:
    """Classify emotion of *audio_bytes* and apply a distress heuristic.

    Parameters
    ----------
    audio_bytes : raw audio bytes (wav/mp3/etc.)
    filename     : optional filename (unused, kept for API compat)
    context_dict : optional dict with keys "full_transcript" and/or
                "background" (YAMNet events) for heuristic upgrade.

    Returns
    -------
    (emotion_label, confidence, probs_dict)
    """
    # 1. decode temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=None)
    os.unlink(tmp_path)

    # 2. preprocess & predict
    y = _preprocess_audio(y)
    avg_preds = _predict_chunks(y, sr)

    top_idx = int(np.argmax(avg_preds))
    emotion = _LABELS[top_idx]
    conf    = float(avg_preds[top_idx])

    # 3. distress heuristic --------------------------------------------------
    if emotion == "Neutral" and context_dict is not None:
        distress = False

        # 3a. keyword check in transcript
        kw = ("help", "hurry", "emergency", "fire", "accident", "police")
        transcript = context_dict.get("full_transcript", "").lower()
        if any(k in transcript for k in kw):
            distress = True

        # 3b. background event check
        for ev in context_dict.get("background", []):
            if ev["label"] in {"Siren", "Gunshot, gunfire", "Scream"} and ev["confidence"] > 0.20:
                distress = True
                break

        if distress:
            emotion = "Fear"
            conf = max(conf, 0.55)

    # 4. full probs dict
    probs = {lab: float(p) for lab, p in zip(_LABELS, avg_preds)}

    return emotion, conf, probs

# ---------------------------------------------------------------------
#  CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    wav = sys.argv[1]
    with open(wav, "rb") as f:
        emo, c, _ = classify_emotion(f.read())
    print(json.dumps({"emotion": emo, "conf": c}, indent=2))
