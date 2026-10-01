import numpy as np
from sklearn.decomposition import PCA


class PCAAnomalyDetector:
    """PCA reconstruction baseline for spectrogram anomaly detection."""

    def __init__(
        self,
        n_components: int,
        random_state: int = 42,
    ):
        self.n_components = n_components

        self.model = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
        )

        self.frame_shape: tuple[int, int] | None = None

    def fit(self, frames: np.ndarray) -> None:
        """Fit PCA using normal training frames only."""

        self.frame_shape = frames.shape[1:]

        flattened = frames.reshape(
            frames.shape[0],
            -1,
        )

        self.model.fit(flattened)

    def reconstruct(self, frames: np.ndarray) -> np.ndarray:
        """Reconstruct frames through the PCA representation."""

        if self.frame_shape is None:
            raise RuntimeError("Model must be fitted before reconstruction.")

        flattened = frames.reshape(
            frames.shape[0],
            -1,
        )

        encoded = self.model.transform(flattened)
        reconstructed = self.model.inverse_transform(encoded)

        return reconstructed.reshape(
            frames.shape
        )

    def score(self, frames: np.ndarray) -> np.ndarray:
        """Return mean squared reconstruction error per frame."""

        reconstructed = self.reconstruct(frames)

        squared_error = (
            frames - reconstructed
        ) ** 2

        return squared_error.mean(
            axis=(1, 2)
        )