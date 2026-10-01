import numpy as np


def inject_stationary_narrowband(
    frame: np.ndarray,
    frequency_bin: int,
    snr: float,
    bandwidth_bins: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a stationary narrowband signal into a raw power spectrogram.

    Parameters
    ----------
    frame:
        2D spectrogram with shape (time, frequency).

    frequency_bin:
        Centre frequency-bin index where the signal is injected.

    snr:
        Injection strength in units of the frame's robust noise scale.

    bandwidth_bins:
        Number of neighbouring frequency bins occupied by the signal.

    Returns
    -------
    injected_frame:
        Copy of the original frame containing the synthetic signal.

    signal_mask:
        Boolean mask showing exactly where the signal was injected.
    """

    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if snr <= 0:
        raise ValueError("snr must be greater than zero.")

    if bandwidth_bins < 1:
        raise ValueError("bandwidth_bins must be at least 1.")

    num_freq_bins = frame.shape[1]

    if not 0 <= frequency_bin < num_freq_bins:
        raise ValueError("frequency_bin is outside the frame.")

    median = np.median(frame)
    mad = np.median(np.abs(frame - median))

    if mad == 0:
        raise ValueError("Frame has zero MAD; cannot estimate noise scale.")

    robust_sigma = 1.4826 * mad
    signal_power = snr * robust_sigma

    start = frequency_bin - bandwidth_bins // 2
    stop = start + bandwidth_bins

    if start < 0 or stop > num_freq_bins:
        raise ValueError("Injected signal would extend outside the frame.")

    injected_frame = frame.astype(np.float32, copy=True)

    signal_mask = np.zeros(
        frame.shape,
        dtype=bool,
    )

    injected_frame[:, start:stop] += signal_power
    signal_mask[:, start:stop] = True

    return injected_frame, signal_mask

def inject_linear_drift(
    frame: np.ndarray,
    start_frequency_bin: int,
    drift_bins_per_time: float,
    snr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a linearly drifting narrowband signal.

    The signal moves in frequency as time progresses.
    """

    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if snr <= 0:
        raise ValueError("snr must be greater than zero.")

    median = np.median(frame)
    mad = np.median(np.abs(frame - median))

    if mad == 0:
        raise ValueError("Frame has zero MAD; cannot estimate noise scale.")

    robust_sigma = 1.4826 * mad
    signal_power = snr * robust_sigma

    injected_frame = frame.astype(np.float32, copy=True)
    signal_mask = np.zeros(frame.shape, dtype=bool)

    num_time_bins, num_freq_bins = frame.shape

    for time_bin in range(num_time_bins):
        frequency_bin = round(
            start_frequency_bin
            + drift_bins_per_time * time_bin
        )

        if not 0 <= frequency_bin < num_freq_bins:
            raise ValueError(
                "Drifting signal extends outside the frame."
            )

        injected_frame[time_bin, frequency_bin] += signal_power
        signal_mask[time_bin, frequency_bin] = True

    return injected_frame, signal_mask

def inject_intermittent(
    frame: np.ndarray,
    frequency_bin: int,
    snr: float,
    active_time_bins: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a stationary narrowband signal that is present
    only during selected time bins.
    """

    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if snr <= 0:
        raise ValueError("snr must be greater than zero.")

    num_time_bins, num_freq_bins = frame.shape

    if not 0 <= frequency_bin < num_freq_bins:
        raise ValueError("frequency_bin is outside the frame.")

    if not active_time_bins:
        raise ValueError("active_time_bins cannot be empty.")

    for time_bin in active_time_bins:
        if not 0 <= time_bin < num_time_bins:
            raise ValueError(
                "active_time_bins contains an invalid time bin."
            )

    median = np.median(frame)
    mad = np.median(np.abs(frame - median))

    if mad == 0:
        raise ValueError("Frame has zero MAD; cannot estimate noise scale.")

    robust_sigma = 1.4826 * mad
    signal_power = snr * robust_sigma

    injected_frame = frame.astype(np.float32, copy=True)
    signal_mask = np.zeros(frame.shape, dtype=bool)

    for time_bin in active_time_bins:
        injected_frame[time_bin, frequency_bin] += signal_power
        signal_mask[time_bin, frequency_bin] = True

    return injected_frame, signal_mask

def inject_multiple_carriers(
    frame: np.ndarray,
    frequency_bins: list[int],
    snr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject multiple stationary narrowband carriers.
    """

    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if snr <= 0:
        raise ValueError("snr must be greater than zero.")

    if not frequency_bins:
        raise ValueError("frequency_bins cannot be empty.")

    num_freq_bins = frame.shape[1]

    for frequency_bin in frequency_bins:
        if not 0 <= frequency_bin < num_freq_bins:
            raise ValueError(
                "frequency_bins contains a bin outside the frame."
            )

    median = np.median(frame)
    mad = np.median(np.abs(frame - median))

    if mad == 0:
        raise ValueError(
            "Frame has zero MAD; cannot estimate noise scale."
        )

    robust_sigma = 1.4826 * mad
    signal_power = snr * robust_sigma

    injected_frame = frame.astype(np.float32, copy=True)
    signal_mask = np.zeros(frame.shape, dtype=bool)

    for frequency_bin in frequency_bins:
        injected_frame[:, frequency_bin] += signal_power
        signal_mask[:, frequency_bin] = True

    return injected_frame, signal_mask

def inject_nonlinear_drift(
    frame: np.ndarray,
    start_frequency_bin: int,
    linear_drift_bins_per_time: float,
    curvature_bins_per_time2: float,
    snr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a nonlinear drifting narrowband signal.

    Frequency follows a quadratic trajectory over time.
    """

    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if snr <= 0:
        raise ValueError("snr must be greater than zero.")

    median = np.median(frame)
    mad = np.median(np.abs(frame - median))

    if mad == 0:
        raise ValueError(
            "Frame has zero MAD; cannot estimate noise scale."
        )

    robust_sigma = 1.4826 * mad
    signal_power = snr * robust_sigma

    injected_frame = frame.astype(np.float32, copy=True)
    signal_mask = np.zeros(frame.shape, dtype=bool)

    num_time_bins, num_freq_bins = frame.shape

    for time_bin in range(num_time_bins):
        frequency_bin = round(
            start_frequency_bin
            + linear_drift_bins_per_time * time_bin
            + curvature_bins_per_time2 * time_bin**2
        )

        if not 0 <= frequency_bin < num_freq_bins:
            raise ValueError(
                "Nonlinear signal extends outside the frame."
            )

        injected_frame[time_bin, frequency_bin] += signal_power
        signal_mask[time_bin, frequency_bin] = True

    return injected_frame, signal_mask

def inject_burst(
    frame: np.ndarray,
    start_time_bin: int,
    duration_bins: int,
    center_frequency_bin: int,
    bandwidth_bins: int,
    snr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a short-duration, finite-bandwidth burst signal.
    """

    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if snr <= 0:
        raise ValueError("snr must be greater than zero.")

    if duration_bins < 1:
        raise ValueError("duration_bins must be at least 1.")

    if bandwidth_bins < 1:
        raise ValueError("bandwidth_bins must be at least 1.")

    num_time_bins, num_freq_bins = frame.shape

    end_time_bin = start_time_bin + duration_bins

    start_freq_bin = center_frequency_bin - bandwidth_bins // 2
    end_freq_bin = start_freq_bin + bandwidth_bins

    if start_time_bin < 0 or end_time_bin > num_time_bins:
        raise ValueError("Burst extends outside the time axis.")

    if start_freq_bin < 0 or end_freq_bin > num_freq_bins:
        raise ValueError("Burst extends outside the frequency axis.")

    median = np.median(frame)
    mad = np.median(np.abs(frame - median))

    if mad == 0:
        raise ValueError(
            "Frame has zero MAD; cannot estimate noise scale."
        )

    robust_sigma = 1.4826 * mad
    signal_power = snr * robust_sigma

    injected_frame = frame.astype(np.float32, copy=True)
    signal_mask = np.zeros(frame.shape, dtype=bool)

    injected_frame[
        start_time_bin:end_time_bin,
        start_freq_bin:end_freq_bin,
    ] += signal_power

    signal_mask[
        start_time_bin:end_time_bin,
        start_freq_bin:end_freq_bin,
    ] = True

    return injected_frame, signal_mask