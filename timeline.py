"""Emotion over time, not just one label per clip - run the existing
classifier over a sliding window across a longer recording (a film scene,
a song, a recorded conversation) instead of only ever analyzing one short
utterance at a time.

The classical model (classifier.py) is reused as-is, including its 180-dim
feature extraction (Mel/MFCC/chroma, averaged per window) - this module's
real work is the windowing and the per-window inference loop, not a new
model."""

from dataclasses import dataclass

import librosa
import numpy as np
import pandas as pd

WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
SAMPLE_RATE = 16000


@dataclass
class TimelinePoint:
    start: float
    end: float
    emotion: str
    confidence: float


def sliding_windows(audio: np.ndarray, sr: int, window_seconds: float = WINDOW_SECONDS, hop_seconds: float = HOP_SECONDS):
    window_size = int(window_seconds * sr)
    hop_size = int(hop_seconds * sr)
    if len(audio) < window_size:
        yield 0.0, len(audio) / sr, audio
        return
    for start_sample in range(0, len(audio) - window_size + 1, hop_size):
        end_sample = start_sample + window_size
        yield start_sample / sr, end_sample / sr, audio[start_sample:end_sample]


def _extract_feature_row(audio: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    combined = np.concatenate([mel, mfcc, chroma], axis=0)
    return combined.mean(axis=1)


def analyze_timeline(audio: np.ndarray, sr: int, model) -> list[TimelinePoint]:
    """model: the classical sklearn Pipeline from classifier.py (has
    predict_proba and classes_). Returns one TimelinePoint per window -
    empty/near-silent windows are skipped rather than fed to the model,
    since a silent window has no real emotional content to classify."""
    feature_cols = [f"f{i}" for i in range(180)]
    points = []

    for start, end, segment in sliding_windows(audio, sr):
        if np.max(np.abs(segment)) < 1e-3:
            continue
        features = _extract_feature_row(segment, sr)
        row = pd.DataFrame([features], columns=feature_cols)
        proba = model.predict_proba(row)[0]
        top_index = int(np.argmax(proba))
        points.append(TimelinePoint(
            start=start, end=end, emotion=model.classes_[top_index], confidence=float(proba[top_index]),
        ))

    return points


def load_audio_file(file_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import io
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=sample_rate, mono=True)
    return audio
