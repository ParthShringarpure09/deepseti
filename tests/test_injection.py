import numpy as np

from deepseti.injection.signals import (
    inject_linear_drift,
    inject_stationary_narrowband,
)

from deepseti.injection.signals import (
    inject_intermittent,
    inject_linear_drift,
    inject_multiple_carriers,
    inject_nonlinear_drift,
    inject_stationary_narrowband,
)
from deepseti.injection.signals  import *


def test_stationary_narrowband_injection():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    injected, mask = inject_stationary_narrowband(
        frame,
        frequency_bin=2000,
        snr=5.0,
        bandwidth_bins=1,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape
    assert mask.dtype == bool

    assert mask[:, 2000].all()
    assert mask.sum() == 16

    assert np.all(injected[:, 2000] > frame[:, 2000])

    unchanged = ~mask

    assert np.allclose(
        injected[unchanged],
        frame[unchanged],
    )


def test_linear_drift_injection():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    injected, mask = inject_linear_drift(
        frame,
        start_frequency_bin=1900,
        drift_bins_per_time=10,
        snr=5.0,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape
    assert mask.dtype == bool

    assert mask.sum() == 16

    expected_bins = [
        1900 + 10 * t
        for t in range(16)
    ]

    for time_bin, frequency_bin in enumerate(expected_bins):
        assert mask[time_bin, frequency_bin]

        assert (
            injected[time_bin, frequency_bin]
            > frame[time_bin, frequency_bin]
        )

    unchanged = ~mask

    assert np.allclose(
        injected[unchanged],
        frame[unchanged],
    )

from deepseti.injection.signals import (
    inject_intermittent,
    inject_linear_drift,
    inject_stationary_narrowband,
)

def test_intermittent_injection():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    active_bins = [0, 1, 2, 7, 8, 12, 13]

    injected, mask = inject_intermittent(
        frame,
        frequency_bin=2000,
        snr=5.0,
        active_time_bins=active_bins,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape

    assert mask.sum() == len(active_bins)

    for time_bin in active_bins:
        assert mask[time_bin, 2000]
        assert injected[time_bin, 2000] > frame[time_bin, 2000]

    unchanged = ~mask

    assert np.allclose(
        injected[unchanged],
        frame[unchanged],
    )

from deepseti.injection.signals import (
    inject_intermittent,
    inject_linear_drift,
    inject_multiple_carriers,
    inject_stationary_narrowband,
)

def test_multiple_carriers_injection():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    carrier_bins = [1200, 2000, 2800]

    injected, mask = inject_multiple_carriers(
        frame,
        frequency_bins=carrier_bins,
        snr=5.0,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape

    assert mask.sum() == 16 * len(carrier_bins)

    for frequency_bin in carrier_bins:
        assert mask[:, frequency_bin].all()
        assert np.all(
            injected[:, frequency_bin]
            > frame[:, frequency_bin]
        )

    unchanged = ~mask

    assert np.allclose(
        injected[unchanged],
        frame[unchanged],
    )

def test_nonlinear_drift_injection():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    injected, mask = inject_nonlinear_drift(
        frame,
        start_frequency_bin=1800,
        linear_drift_bins_per_time=3.0,
        curvature_bins_per_time2=0.8,
        snr=5.0,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape

    assert mask.sum() == 16

    for time_bin in range(16):
        frequency_bin = round(
            1800
            + 3.0 * time_bin
            + 0.8 * time_bin**2
        )

        assert mask[time_bin, frequency_bin]

        assert (
            injected[time_bin, frequency_bin]
            > frame[time_bin, frequency_bin]
        )

    unchanged = ~mask

    assert np.allclose(
        injected[unchanged],
        frame[unchanged],
    )

def test_burst_injection():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    injected, mask = inject_burst(
        frame,
        start_time_bin=5,
        duration_bins=4,
        center_frequency_bin=2000,
        bandwidth_bins=20,
        snr=5.0,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape

    assert mask.sum() == 4 * 20

    assert mask[5:9, 1990:2010].all()

    unchanged = ~mask

    assert np.allclose(
        injected[unchanged],
        frame[unchanged],
    )