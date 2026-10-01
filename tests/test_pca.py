import numpy as np

from deepseti.models.pca import PCAAnomalyDetector


def test_pca_fit_reconstruct_and_score():
    rng = np.random.default_rng(42)

    frames = rng.normal(
        size=(20, 16, 64)
    ).astype(np.float32)

    model = PCAAnomalyDetector(
        n_components=5,
        random_state=42,
    )

    model.fit(frames)

    reconstructed = model.reconstruct(frames)
    scores = model.score(frames)

    assert reconstructed.shape == frames.shape
    assert scores.shape == (20,)
    assert np.isfinite(reconstructed).all()
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)