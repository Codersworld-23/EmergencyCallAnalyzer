import streamlit as st
import streamlit.components.v1 as components
import os
import datetime
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(HERE, "st_audiorec", "frontend", "build")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)


def audio_input_section():
    st.sidebar.markdown("### 🎙 Upload or Record Audio")

    if 'audio_data_received' not in st.session_state:
        st.session_state.audio_data_received = None
    if 'processed_recordings' not in st.session_state:
        st.session_state.processed_recordings = set()

    uploaded_file = st.sidebar.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        return uploaded_file, audio_bytes

    st.sidebar.markdown("### Or Record Using Microphone")
    audio_data = st_audiorec()

    if audio_data and isinstance(audio_data, dict) and "data" in audio_data and audio_data["data"]:
        try:
            with st.spinner("Decoding and saving recorded audio..."):
                audio_bytes = base64.b64decode(audio_data["data"])

                # Limit audio length
                MAX_AUDIO_BYTES = 500_000  # ~500 KB
                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    st.sidebar.error("Recording too long. Keep it under 10 seconds.")
                    return None, None

                audio_id = f"{len(audio_bytes)}_{hash(audio_data['data'][:100])}"
                if audio_id not in st.session_state.processed_recordings:
                    save_dir = os.path.join("data", "recordings")
                    os.makedirs(save_dir, exist_ok=True)

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"mic_recorded_audio_{timestamp}.wav"
                    filepath = os.path.join(save_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)

                    st.sidebar.success("✅ Recording completed and saved!")
                    st.sidebar.write(f"📁 Saved as: {filename}")

                    st.session_state.processed_recordings.add(audio_id)
                    st.session_state.audio_data_received = {
                        "filename": filename,
                        "audio_bytes": audio_bytes,
                        "id": audio_id
                    }


        except Exception as e:
            st.sidebar.error(f"❌ Error processing recording: {str(e)}")
            return None, None

    # Return audio only when new recording is ready
    if st.session_state.audio_data_received:
        stored_data = st.session_state.audio_data_received
        return stored_data["filename"], stored_data["audio_bytes"]

    return None, None