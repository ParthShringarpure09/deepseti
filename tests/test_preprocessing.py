import numpy as np

from deepseti.preprocessing.frames import extract_frames


def test_extract_frames_shape():
    data = np.ones((32, 8192), dtype=np.float32)

    frames = extract_frames(data)

    assert frames.shape == (4, 16, 4096)

from deepseti.preprocessing.frames import robust_normalize_frame


def test_robust_normalize_frame():
    rng = np.random.default_rng(42)

    frame = rng.uniform(
        low=1.0,
        high=100.0,
        size=(16, 4096),
    ).astype(np.float32)

    normalized = robust_normalize_frame(frame)

    assert normalized.shape == (16, 4096)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert np.isclose(np.median(normalized), 0.0, atol=1e-5)

from deepseti.preprocessing.frames import preprocess_spectrogram


def test_preprocess_spectrogram():
    rng = np.random.default_rng(42)

    data = rng.uniform(
        low=1.0,
        high=100.0,
        size=(32, 8192),
    ).astype(np.float32)

    processed = preprocess_spectrogram(data)

    assert processed.shape == (4, 16, 4096)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()