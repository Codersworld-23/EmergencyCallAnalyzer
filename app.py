import tempfile
from pathlib import Path

import streamlit as st
from components.whisper_transcriber_streamlit import transcribe_audio_file
from components.emotion_classifier import classify_emotion
from components.urgency_classifier import classify_urgency
from components.log_manager import display_log, save_to_log
from components.audio_visualiser import visualise_audio
from components.speaker_diarization import get_speaker_segments           # (audio_path) -> list[dict]
from components.background_voices import detect_background_sounds  # (audio_bytes) -> list[dict]
from utils.setup_ffmpeg import ensure_ffmpeg
from utils.audio_utils import slice_audio_ffmpeg

ensure_ffmpeg()  


# ──────────────────────────────────────────────────────────────
#  Streamlit Dashboard
# ──────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Emergency Call Analyzer", layout="wide")
    st.title("🚨 Emergency Call Analyzer")

    # ── Sidebar: upload & logs ─────────────────────────────────
    with st.sidebar:
        st.header("🎧 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        )

        st.markdown("---")
        st.header("📃 Past Logs")
        display_log()

    # ── Main area ─────────────────────────────────────────────
    if uploaded_file:
        # ------------------------------------------------------
        # 1) Save upload to temp file
        # ------------------------------------------------------
        audio_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Show player + waveform
        st.audio(audio_bytes, format="audio/wav")
        visualise_audio(audio_bytes)

        # ------------------------------------------------------
        # 2) Run heavy pipelines inside spinners
        # ------------------------------------------------------
        with st.spinner("Running speaker diarization..."):
            speaker_segments = get_speaker_segments(tmp_path)  # list of dicts

        with st.spinner("Transcribing full audio (Whisper)..."):
            full_transcript = transcribe_audio_file(tmp_path)

        with st.spinner("Detecting background sounds..."):
            background_events = detect_background_sounds(audio_bytes)


        # Speaker‑wise transcription & emotion
        per_speaker_results = []
        for seg in speaker_segments:
            seg_start, seg_end = seg["start"], seg["end"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as sfile:
                slice_audio_ffmpeg(tmp_path, sfile.name, seg_start, seg_end)

            seg_text = transcribe_audio_file(sfile.name)
            with open(sfile.name, "rb") as f:
                seg_emotion, seg_conf, _ = classify_emotion(f.read(), filename=None)

            per_speaker_results.append(
                {
                    "speaker": seg["speaker"],
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                    "emotion": seg_emotion,
                    "confidence": seg_conf,
                }
            )
            Path(sfile.name).unlink(missing_ok=True)

        # Build combined_context after per_speaker_results is ready
        combined_context = {
            "full_transcript": full_transcript,
            "speaker_segments": per_speaker_results,
            "background": background_events,
        }

        with st.spinner("Classifying overall emotion..."):
            overall_emotion, overall_conf, overall_probs = classify_emotion(
                audio_bytes,
                context_dict=combined_context,   # new argument
                filename=uploaded_file.name
            )


        # ------------------------------------------------------
        # 3) Urgency classification (LLM)
        # ------------------------------------------------------
        combined_context = {
            "full_transcript": full_transcript,
            "speaker_segments": per_speaker_results,
            "background": background_events,
        }
        with st.spinner("Assessing urgency (LLM)…"):
            urgency_category, urgency_level, urgency_conf = classify_urgency(combined_context)

        # ------------------------------------------------------
        # 4) Display results
        # ------------------------------------------------------
        tab_full, tab_speakers, tab_bg, tab_urg = st.tabs(
            ["📄 Transcript", "🗣️ Speakers", "🔊 Background", "⚡ Urgency"]
        )

        with tab_full:
            st.subheader("Full Transcript")
            st.text(full_transcript)

        with tab_speakers:
            st.subheader("Speaker‑wise Analysis")
            for r in per_speaker_results:
                with st.expander(f"{r['speaker']}  [{r['start']:.1f}s–{r['end']:.1f}s]"):
                    st.write(r["text"])
                    st.info(f"Emotion: **{r['emotion']}** ({r['confidence']:.2f})")

        with tab_bg:
            st.subheader("Detected Background Sounds")
            if background_events:
                for ev in background_events:
                    st.write(f"• {ev['label']}  ({ev['confidence']:.2f})")
            else:
                st.write("No significant background events detected.")


        with tab_urg:
            st.subheader("Urgency Assessment")
            st.write(f"**Category:** {urgency_category}")
            st.write(f"🚨 **Level:** {urgency_level}")
            st.metric("Model Confidence (%)", f"{urgency_conf:.1f}")
            st.metric(
                "Overall Emotion",
                overall_emotion,
                f"{overall_conf:.2f} (p={overall_probs[overall_emotion]:.2f})"
            )

        # ------------------------------------------------------
        # 5) Save to log
        # ------------------------------------------------------
        save_to_log(
            uploaded_file.name,
            overall_emotion,
            overall_conf,
            urgency_level,
            urgency_category,
            urgency_conf,
            full_transcript,
        )

    else:
        st.info("Upload an audio file to start the analysis.")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
