import numpy as np

from timeline import HOP_SECONDS, SAMPLE_RATE, WINDOW_SECONDS, analyze_timeline, sliding_windows


def test_sliding_windows_covers_a_long_clip_with_expected_count():
    duration_seconds = 10.0
    audio = np.random.default_rng(0).uniform(-0.3, 0.3, int(duration_seconds * SAMPLE_RATE)).astype(np.float32)
    windows = list(sliding_windows(audio, SAMPLE_RATE))

    expected_starts = int((duration_seconds * SAMPLE_RATE - WINDOW_SECONDS * SAMPLE_RATE) // (HOP_SECONDS * SAMPLE_RATE)) + 1
    assert len(windows) == expected_starts
    first_start, first_end, first_segment = windows[0]
    assert first_start == 0.0
    assert abs(first_end - WINDOW_SECONDS) < 1e-6
    assert len(first_segment) == int(WINDOW_SECONDS * SAMPLE_RATE)


def test_sliding_windows_short_clip_returns_one_window():
    audio = np.random.default_rng(1).uniform(-0.3, 0.3, int(1.0 * SAMPLE_RATE)).astype(np.float32)
    windows = list(sliding_windows(audio, SAMPLE_RATE))
    assert len(windows) == 1
    start, end, segment = windows[0]
    assert start == 0.0
    assert len(segment) == len(audio)


class _FakeModel:
    classes_ = np.array(["happy", "sad", "neutral"])

    def predict_proba(self, df):
        # Always confidently "happy" - enough to exercise analyze_timeline's
        # plumbing without needing a real trained model in a unit test.
        return np.array([[0.8, 0.1, 0.1]])


def test_analyze_timeline_skips_silent_windows():
    sr = SAMPLE_RATE
    loud = np.random.default_rng(2).uniform(-0.3, 0.3, int(WINDOW_SECONDS * sr)).astype(np.float32)
    silent = np.zeros(int(WINDOW_SECONDS * sr), dtype=np.float32)
    audio = np.concatenate([loud, silent, loud])

    points = analyze_timeline(audio, sr, _FakeModel())

    assert all(p.emotion == "happy" for p in points)
    assert len(points) >= 1


def test_analyze_timeline_reports_confidence_and_time_range():
    sr = SAMPLE_RATE
    audio = np.random.default_rng(3).uniform(-0.3, 0.3, int(WINDOW_SECONDS * sr)).astype(np.float32)
    points = analyze_timeline(audio, sr, _FakeModel())
    assert len(points) == 1
    assert points[0].confidence == 0.8
    assert points[0].start == 0.0
