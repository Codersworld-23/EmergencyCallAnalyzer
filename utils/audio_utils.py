import subprocess

def slice_audio_ffmpeg(input_path, output_path, start_time, end_time):
    """Slices input_path from start_time to end_time into output_path using FFmpeg."""
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-acodec", "copy",  # faster copy
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
