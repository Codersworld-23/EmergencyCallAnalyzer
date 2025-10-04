"""
speaker_diarization.py  – v2.1
Embedding-based post-clustering to merge over-segmented speakers.
"""

from __future__ import annotations
from typing import List, Dict
import os, tempfile
import numpy as np

# Fix for NumPy ≥1.26 (pyannote expects np.NAN)
import numpy as np
if not hasattr(np, "NAN"):
    np.NAN = np.nan

import torch, librosa, soundfile as sf
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Pipeline, Model
from pyannote.audio.core.inference import Inference
from pyannote.core import Annotation, Segment
from dotenv import load_dotenv
import streamlit as st

# ─────────────────────────────────────────────────────────────
# ENV + CACHED DIARIZATION PIPELINE
# ─────────────────────────────────────────────────────────────
load_dotenv()
hf_token = os.getenv("hf_token")

@st.cache_resource(show_spinner="🔈 Loading speaker-diarization model…")
def _load_diar_pipe() -> Pipeline:
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            "Could not load pyannote speaker-diarization model. "
            "Is your hf_token valid and did you accept model access?\n"
            f"{e}"
        )
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return pipe

# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def get_speaker_segments(
    audio_path: str,
    min_dur: float = 0.25,
    similarity_thr: float = 0.78,
    collar: float = 0.25,
) -> List[Dict]:
    """
    Returns list[{speaker, start, end}] with clustering-based merge.

    similarity_thr : Cosine similarity threshold (0.78–0.85 good range)
    """

    diar_pipe = _load_diar_pipe()

    # 1) Resample to 16 kHz mono (pyannote requirement)
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        sf.write(wf.name, y, 16000)
        wav_rs = wf.name

    raw_ann: Annotation = diar_pipe(wav_rs)
    # do NOT unlink wav_rs yet (needed for embedding crop)

    # 2) Load embedding model and wrap with Inference
    try:
        # Load the raw model
        raw_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=hf_token
        )
        
        # Wrap with Inference to get crop functionality
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb_model = Inference(raw_model, device=device)
        
    except Exception as e:
        os.unlink(wav_rs)
        raise RuntimeError(
            "Could not load pyannote/embedding model "
            "(check HF token / access).\n" + str(e)
        )

    # 3) Extract embeddings for every raw segment >= min_dur
    embeds, labels = [], []
    for seg, _, spk in raw_ann.itertracks(yield_label=True):
        if seg.duration < min_dur:
            continue
        
        try:
            # Method 1: Try direct crop
            emb = emb_model.crop(wav_rs, seg)
            
            # Handle pooling manually - take mean across time dimension
            if isinstance(emb, torch.Tensor):
                if emb.dim() > 1:
                    emb = emb.mean(dim=1)  # Mean pool across time dimension
                emb = emb.detach().cpu().numpy()
            else:
                # If it's already numpy, pool across the appropriate dimension
                if emb.ndim > 1:
                    emb = np.mean(emb, axis=1)
            
        except Exception as e:
            print(f"Error with crop method: {e}")
            # Method 2: Fallback to sliding window approach
            try:
                # Use sliding window inference
                from pyannote.core import SlidingWindow
                
                # Load audio for this segment
                y_seg, _ = librosa.load(wav_rs, sr=16000, offset=seg.start, duration=seg.duration)
                
                # Apply model to the segment
                with torch.no_grad():
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(y_seg).unsqueeze(0).to(device)
                    emb = emb_model.model(audio_tensor)
                    
                    # Pool across time dimension
                    if emb.dim() > 2:
                        emb = emb.mean(dim=2)  # Mean pool across time
                    emb = emb.squeeze().detach().cpu().numpy()
                    
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")
                continue
        
        embeds.append(emb)
        labels.append(spk)

    os.unlink(wav_rs)  # cleanup resampled WAV

    if len(embeds) == 0:           # no valid segments
        return []

    embeds = np.vstack(embeds)
    uniq_raw = sorted(set(labels))

    # 4) Speaker-level centroids + agglomerative clustering
    centroid = {
        spk: embeds[[i for i, l in enumerate(labels) if l == spk]].mean(axis=0)
        for spk in uniq_raw
    }
    centroids = np.stack([centroid[l] for l in uniq_raw])

    if len(uniq_raw) == 1:
        # Only one speaker, no clustering needed
        label_map = {uniq_raw[0]: "SPEAKER_00"}
    else:
        dist_mat = squareform(pdist(centroids, metric="cosine"))
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - similarity_thr,
        ).fit(dist_mat)
        label_map = {
            raw: f"SPEAKER_{cid:02d}" for raw, cid in zip(uniq_raw, clusterer.labels_)
        }

    # 5) Build merged annotation
    merged = Annotation()
    for seg, _, raw_spk in raw_ann.itertracks(yield_label=True):
        if seg.duration < min_dur:
            continue
        merged[seg] = label_map[raw_spk]

    # 6) Consolidate segments per final speaker with collar merge
    final = Annotation()
    for spk in merged.labels():
        tl = merged.label_timeline(spk)
        cur = tl[0]
        for seg in tl[1:]:
            if seg.start - cur.end <= collar:
                cur = Segment(cur.start, seg.end)
            else:
                final[cur] = spk
                cur = seg
        final[cur] = spk

    # 7) Export to list[dict]
    return [
        {"speaker": spk, "start": round(seg.start, 2), "end": round(seg.end, 2)}
        for seg, _, spk in final.itertracks(yield_label=True)
        if seg.duration >= min_dur
    ]