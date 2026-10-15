"""Synthetic signal injection for the Fine V2 validation benchmark.

This module is specific to the Fine V2 geometry and protocol (frame shape
16 x 4096, channel width ~2.7939677 Hz, time spacing ~18.253611 s). It is
deliberately separate from ``deepseti.injection.signals`` /
``deepseti.injection.benchmark``, which implement an older, unitless
"channels/row" drift convention for a different (MID-era) dataset; that
convention must not be reused here.

Design, fixed by the Fine V2 validation benchmark development revision:

* Injection happens in the LINEAR-POWER domain, before the frozen
  log10 + per-frame median/MAD normalization (``deepseti.preprocessing.
  fine.normalize_fine_frame``). Only positive additive power is injected.
* Physical drift rate -> fractional channels advanced per time row:
      channels_per_row = drift_hz_per_s * dt_seconds / channel_width_hz
* Local SNR: for each injected track, the noise scale is estimated from a
  frequency guard/background region flanking the track's channel span,
  excluding the signal columns (plus a small margin). The estimator is
      sigma_local = 1.4826 * MAD(background pixels)
  and the injected amplitude (in linear power) is
      signal_amplitude = requested_snr * sigma_local
* Geometry-only placement: track/carrier locations are chosen from a
  seeded RNG constrained only by frame geometry (edges, track span,
  non-overlap). Content (signal/RFI appearance) is never used to select
  frequencies or frames.
* Held-out morphologies ("nonlinear_chirp", "burst") are implemented for
  later generalization evaluation but are never produced by the
  development sampling plan (see ``deepseti.evaluation.fine_benchmark``).
"""

from __future__ import annotations

import numpy as np

CHANNEL_WIDTH_HZ = 2.7939677238464355
TSAMP_S = 18.253611008
N_ROWS = 16
N_CHANNELS = 4096

# Fixed, documented morphology constants (not swept; see module docstring).
BANDWIDTH_CHANNELS = 2
GUARD_CHANNELS = 256
MARGIN_CHANNELS = 4
CARRIER_SPACING_CHANNELS = 400

DEVELOPMENT_MORPHOLOGIES = (
    "stationary", "linear_drift", "intermittent", "multiple_carriers",
)
# Reserved for later unseen/generalization evaluation. The development
# sampling plan in deepseti.evaluation.fine_benchmark must never emit these.
HELD_OUT_MORPHOLOGIES = ("nonlinear_chirp", "burst")


def drift_hz_per_s_to_channels_per_row(
    drift_hz_per_s: float,
    dt_seconds: float = TSAMP_S,
    channel_width_hz: float = CHANNEL_WIDTH_HZ,
) -> float:
    """Physical linear drift rate (Hz/s) -> fractional channels per time row."""

    return drift_hz_per_s * dt_seconds / channel_width_hz


# ---------------------------------------------------------------------------
# signal masks (raw-frame pixel coordinates)
# ---------------------------------------------------------------------------


def stationary_mask(start_channel: int, bandwidth_channels: int = BANDWIDTH_CHANNELS,
                    n_rows: int = N_ROWS, n_cols: int = N_CHANNELS) -> np.ndarray:
    if not 0 <= start_channel <= n_cols - bandwidth_channels:
        raise ValueError("stationary signal leaves the frame")
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[:, start_channel:start_channel + bandwidth_channels] = True
    return mask


def linear_drift_columns(start_channel: int, channels_per_row: float,
                         n_rows: int = N_ROWS) -> np.ndarray:
    return np.array([round(start_channel + channels_per_row * row)
                     for row in range(n_rows)])


def linear_drift_mask(start_channel: int, channels_per_row: float,
                      bandwidth_channels: int = BANDWIDTH_CHANNELS,
                      n_rows: int = N_ROWS, n_cols: int = N_CHANNELS) -> np.ndarray:
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for row, c in enumerate(linear_drift_columns(start_channel, channels_per_row, n_rows)):
        if not 0 <= c <= n_cols - bandwidth_channels:
            raise ValueError("linear drift track leaves the frame")
        mask[row, c:c + bandwidth_channels] = True
    return mask


def intermittent_rows(duty_cycle: float, n_rows: int = N_ROWS) -> np.ndarray:
    """Deterministic, evenly spaced "on" rows for a given duty cycle.

    Fixed phase (no randomization): rank ``k`` of ``n_on`` active rows sits at
    ``round(k * (n_rows - 1) / (n_on - 1))`` (or row 0 when ``n_on == 1``),
    which keeps temporal gaps between "on" rows and is identical across
    examples sharing the same duty cycle.
    """

    n_on = max(1, round(duty_cycle * n_rows))
    if n_on == 1:
        return np.array([0])
    rows = np.round(np.linspace(0, n_rows - 1, n_on)).astype(int)
    return np.unique(rows)


def intermittent_mask(start_channel: int, duty_cycle: float,
                      bandwidth_channels: int = BANDWIDTH_CHANNELS,
                      n_rows: int = N_ROWS, n_cols: int = N_CHANNELS) -> np.ndarray:
    if not 0 <= start_channel <= n_cols - bandwidth_channels:
        raise ValueError("intermittent signal leaves the frame")
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    rows = intermittent_rows(duty_cycle, n_rows)
    mask[np.ix_(rows, np.arange(start_channel, start_channel + bandwidth_channels))] = True
    return mask


def multiple_carriers_mask(start_channel: int, carrier_count: int,
                           bandwidth_channels: int = BANDWIDTH_CHANNELS,
                           spacing_channels: int = CARRIER_SPACING_CHANNELS,
                           n_rows: int = N_ROWS, n_cols: int = N_CHANNELS) -> np.ndarray:
    if carrier_count < 2:
        raise ValueError("multiple_carriers requires at least 2 carriers")
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for k in range(carrier_count):
        c = start_channel + k * spacing_channels
        if not 0 <= c <= n_cols - bandwidth_channels:
            raise ValueError("carrier leaves the frame")
        mask[:, c:c + bandwidth_channels] = True
    return mask


# ---------------------------------------------------------------------------
# geometric (content-independent) placement sampling
# ---------------------------------------------------------------------------


def sample_start_channel(rng: np.random.Generator, span_channels: int,
                         guard_channels: int = GUARD_CHANNELS,
                         n_cols: int = N_CHANNELS) -> int:
    """Uniformly sample a start channel so ``span_channels`` plus guard bands

    on both sides stays fully inside the frame. Geometry-only: no data
    values are consulted.
    """

    lo = guard_channels
    hi = n_cols - guard_channels - span_channels
    if hi < lo:
        raise ValueError(
            f"span {span_channels} + 2*guard {guard_channels} exceeds frame width {n_cols}")
    return int(rng.integers(lo, hi + 1))


def linear_drift_span_channels(channels_per_row: float, bandwidth_channels: int = BANDWIDTH_CHANNELS,
                               n_rows: int = N_ROWS) -> int:
    return int(np.ceil(abs(channels_per_row) * (n_rows - 1))) + bandwidth_channels


def sample_linear_drift_start_channel(rng: np.random.Generator, channels_per_row: float,
                                      bandwidth_channels: int = BANDWIDTH_CHANNELS,
                                      guard_channels: int = GUARD_CHANNELS,
                                      n_rows: int = N_ROWS, n_cols: int = N_CHANNELS) -> int:
    """Sample a row-0 start channel for a linear-drift track, honoring sign.

    The track's row-0 column is ``start_channel``; row ``n_rows - 1`` sits at
    ``start_channel + channels_per_row * (n_rows - 1)``. Bounds are derived
    so that whichever end is the extreme (lower for positive drift, upper for
    negative) still clears the guard band on both sides. Geometry-only.
    """

    shift = channels_per_row * (n_rows - 1)
    lo = guard_channels + max(0.0, -shift)
    hi = n_cols - guard_channels - bandwidth_channels - max(0.0, shift)
    if hi < lo:
        raise ValueError("linear drift span does not fit inside the frame with this guard")
    return int(rng.integers(int(np.ceil(lo)), int(np.floor(hi)) + 1))


def multiple_carriers_span_channels(carrier_count: int, bandwidth_channels: int = BANDWIDTH_CHANNELS,
                                    spacing_channels: int = CARRIER_SPACING_CHANNELS) -> int:
    return (carrier_count - 1) * spacing_channels + bandwidth_channels


# ---------------------------------------------------------------------------
# local noise estimation and injection
# ---------------------------------------------------------------------------


def local_background_mask(frame_shape: tuple[int, int], signal_mask: np.ndarray,
                          guard_channels: int = GUARD_CHANNELS,
                          margin_channels: int = MARGIN_CHANNELS) -> np.ndarray:
    """Boolean mask of background pixels used to estimate local noise.

    The columns touched by the signal (dilated by ``margin_channels``) are
    excluded; the remaining columns inside a ``guard_channels``-wide band
    flanking the signal's overall channel span (both sides, clipped to the
    frame) are the background. All time rows are included, independent of
    which rows a given morphology activates.
    """

    n_rows, n_cols = frame_shape
    cols_with_signal = np.where(signal_mask.any(axis=0))[0]
    if len(cols_with_signal) == 0:
        raise ValueError("signal_mask has no active columns")
    lo_sig, hi_sig = int(cols_with_signal.min()), int(cols_with_signal.max())
    win_lo = max(0, lo_sig - guard_channels)
    win_hi = min(n_cols - 1, hi_sig + guard_channels)

    excluded = np.zeros(n_cols, dtype=bool)
    for c in cols_with_signal:
        a, b = max(0, c - margin_channels), min(n_cols, c + margin_channels + 1)
        excluded[a:b] = True

    bg_cols = np.zeros(n_cols, dtype=bool)
    bg_cols[win_lo:win_hi + 1] = True
    bg_cols &= ~excluded
    if not bg_cols.any():
        raise ValueError("no background columns available; widen guard_channels")

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[:, bg_cols] = True
    return mask


def local_sigma(frame: np.ndarray, background_mask: np.ndarray) -> float:
    """``1.4826 * MAD`` of the background pixels (robust noise sigma)."""

    bg = frame[background_mask]
    med = np.median(bg)
    mad = np.median(np.abs(bg - med))
    if mad == 0:
        raise ValueError("background MAD is zero; cannot estimate local noise")
    return float(1.4826 * mad)


def inject_signal(raw_frame: np.ndarray, signal_mask: np.ndarray, requested_snr: float,
                  guard_channels: int = GUARD_CHANNELS,
                  margin_channels: int = MARGIN_CHANNELS) -> tuple[np.ndarray, dict]:
    """Add positive additive power at ``signal_mask`` pixels.

    Returns ``(injected_frame, diagnostics)``; ``diagnostics`` records the
    estimated local sigma, the injected amplitude, and the pixel counts used.
    """

    if raw_frame.shape != signal_mask.shape:
        raise ValueError("frame/mask shape mismatch")
    if requested_snr <= 0:
        raise ValueError("requested_snr must be positive")

    background_mask = local_background_mask(raw_frame.shape, signal_mask,
                                            guard_channels, margin_channels)
    sigma_local = local_sigma(raw_frame, background_mask)
    amplitude = requested_snr * sigma_local

    injected = raw_frame.astype(np.float32, copy=True)
    injected[signal_mask] += amplitude

    diagnostics = {
        "sigma_local": sigma_local,
        "amplitude": float(amplitude),
        "background_pixels": int(background_mask.sum()),
        "signal_pixels": int(signal_mask.sum()),
    }
    return injected, diagnostics


def achieved_local_snr(clean_normalized: np.ndarray, injected_normalized: np.ndarray,
                       signal_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """Empirical contrast/SNR diagnostic measured in the final (normalized)

    domain, independent of the nominal ``requested_snr`` used to set the
    raw-domain amplitude. Background noise sigma is measured on the CLEAN
    normalized frame (so the injected signal never contaminates the noise
    estimate); the signal level is the shift in median value at the signal
    pixels between the injected and clean normalized frames.
    """

    bg = clean_normalized[background_mask]
    med_bg = np.median(bg)
    mad_bg = np.median(np.abs(bg - med_bg))
    if mad_bg == 0:
        raise ValueError("normalized background MAD is zero")
    sigma_norm = 1.4826 * mad_bg
    delta = np.median(injected_normalized[signal_mask]) - np.median(clean_normalized[signal_mask])
    return float(delta / sigma_norm)
