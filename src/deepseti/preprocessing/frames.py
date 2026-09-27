import numpy as np


def robust_normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Log-transform and robustly normalize one spectrogram frame."""

    log_frame = np.log10(frame)

    median = np.median(log_frame)
    mad = np.median(np.abs(log_frame - median))

    if mad == 0:
        raise ValueError("MAD is zero; frame cannot be robustly normalized.")

    normalized = (log_frame - median) / mad

    return normalized.astype(np.float32)


def extract_frames(
    data: np.ndarray,
    time_window: int = 16,
    freq_window: int = 4096,
) -> np.ndarray:
    """Split a 2D spectrogram into non-overlapping frames."""

    frames = []

    for start_time in range(
        0,
        data.shape[0] - time_window + 1,
        time_window,
    ):
        for start_freq in range(
            0,
            data.shape[1] - freq_window + 1,
            freq_window,
        ):
            frame = data[
                start_time:start_time + time_window,
                start_freq:start_freq + freq_window,
            ]

            frames.append(frame)

    return np.stack(frames)

def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Robustly normalize a batch of spectrogram frames."""

    normalized_frames = [
        robust_normalize_frame(frame)
        for frame in frames
    ]

    return np.stack(normalized_frames)

def preprocess_spectrogram(
    data: np.ndarray,
    time_window: int = 16,
    freq_window: int = 4096,
) -> np.ndarray:
    """Convert a 2D spectrogram into normalized model-ready frames."""

    frames = extract_frames(
        data,
        time_window=time_window,
        freq_window=freq_window,
    )

    return normalize_frames(frames)