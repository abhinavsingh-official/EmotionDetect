import numpy as np
import pandas as pd
import pytest

from AudioEmo import AudioEmotionRecognizer


@pytest.fixture
def tiny_db(tmp_path):
    """A minimal two-class feature table, enough to exercise
    AudioEmotionRecognizer's scaling/mean/predict logic without needing the
    real (large, not-committed) RAVDESS dataset."""
    rng = np.random.default_rng(0)
    n_features = 180
    rows = []
    for label, center in [("happy", 1.0), ("sad", -1.0)]:
        for _ in range(10):
            rows.append([*(rng.normal(center, 0.1, n_features)), label])
    columns = [f"f{i}" for i in range(n_features)] + ["target"]
    df = pd.DataFrame(rows, columns=columns)
    path = tmp_path / "tiny_db.pkl"
    df.to_pickle(path)
    return str(path)


def test_class_means_cover_every_label(tiny_db):
    recognizer = AudioEmotionRecognizer(tiny_db, sample_rate=16000)
    assert set(recognizer.class_means.keys()) == {"happy", "sad"}


def test_predict_returns_a_score_per_class(tiny_db):
    recognizer = AudioEmotionRecognizer(tiny_db, sample_rate=16000)
    # A short burst of real-valued "audio" is enough to exercise feature
    # extraction end to end - the actual content doesn't matter here.
    audio = np.random.default_rng(1).uniform(-0.5, 0.5, 16000).astype(np.float32)
    scores = recognizer.predict(audio, sr=16000)
    assert set(scores.keys()) == {"happy", "sad"}
    assert all(isinstance(v, float) for v in scores.values())


def test_predict_handles_near_silent_audio_without_crashing(tiny_db):
    # Regression test: _extract_features used to divide by np.max(np.abs(audio))
    # with no zero-guard, which crashed (or produced NaN/inf) on near-silent input.
    recognizer = AudioEmotionRecognizer(tiny_db, sample_rate=16000)
    silence = np.zeros(16000, dtype=np.int16)  # non-float dtype forces the normalization branch
    scores = recognizer.predict(silence, sr=16000)
    assert len(scores) == 2
    assert all(np.isfinite(v) for v in scores.values())
