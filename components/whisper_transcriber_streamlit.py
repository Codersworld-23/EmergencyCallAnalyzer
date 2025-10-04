# whisper_transcriber.py
"""
Whisper-based transcriber (no translation)
Transcribes audio in the language it was spoken — including Hindi (Devanagari) or Punjabi (Gurmukhi)
"""

import os
import tempfile
import torch
import whisper
from utils.setup_ffmpeg import ensure_ffmpeg

# Ensure ffmpeg is set up for audio format conversion
ensure_ffmpeg()

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper large model (best accuracy)
MODEL = whisper.load_model("large-v3", device=DEVICE)


def transcribe_audio_file(audio_path: str, *, translate: bool = False) -> str:
    """
    Transcribe an audio file using Whisper.
    - Keeps original spoken language's script (e.g., Hindi → Devanagari) unless `translate=True`.

    Parameters
    ----------
    audio_path : str
        Path to audio file (.wav, .mp3, .webm, etc.)
    translate : bool
        If True, translates to English. If False, keeps original script. (Default: False)

    Returns
    -------
    str
        Transcribed text in the spoken language's script.
    """
    try:
        result = MODEL.transcribe(
            audio_path,
            task="translate" if translate else "transcribe",
            fp16=(DEVICE == "cuda"),
            verbose=False,
        )
        return result["text"].strip()
    except Exception as e:
        return f"[Whisper transcription error: {e}]"


def transcribe_audio_bytes(audio_bytes: bytes, *, suffix=".wav", translate: bool = False) -> str:
    """
    Transcribe audio from raw bytes (e.g. uploaded Streamlit file).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio bytes (from Streamlit uploader, etc.)
    suffix : str
        File extension for temporary file (default: .wav)
    translate : bool
        Translate to English (True) or preserve language script (False)

    Returns
    -------
    str
        Transcribed text.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return transcribe_audio_file(tmp_path, translate=translate)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
