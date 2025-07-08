import subprocess
import os


def convert_mp4_to_wav(mp4_path: str, samplerate: int = 16000) -> str:
    base, _ = os.path.splitext(mp4_path)
    output_path = f"{base}_{samplerate}khz.wav"

    command = [
        "ffmpeg",
        "-i", mp4_path,
        "-ab", "160k",
        "-ac", "1",
        "-ar", str(samplerate),
        "-codec", "pcm_s16le",
        "-vn",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Converted: {output_path}")
    return output_path
