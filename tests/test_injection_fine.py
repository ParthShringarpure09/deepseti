from itertools import pairwise

import numpy as np
import pytest

from deepseti.injection import fine as inj
from deepseti.preprocessing.fine import normalize_fine_frame


def test_channels_per_row_conversion():
    # 1 Hz/s over the frozen dt/channel-width should match the raw formula.
    rate = inj.drift_hz_per_s_to_channels_per_row(1.0)
    assert rate == pytest.approx(inj.TSAMP_S / inj.CHANNEL_WIDTH_HZ)
    assert inj.drift_hz_per_s_to_channels_per_row(-2.0) == pytest.approx(-2 * rate)
    assert inj.drift_hz_per_s_to_channels_per_row(0.0) == 0.0


def test_stationary_mask_shape_and_bounds():
    mask = inj.stationary_mask(100, bandwidth_channels=2)
    assert mask.shape == (16, 4096)
    assert mask.sum() == 16 * 2
    assert mask[:, 100:102].all()
    with pytest.raises(ValueError):
        inj.stationary_mask(4095, bandwidth_channels=2)


def test_linear_drift_mask_tracks_physical_rate():
    cpr = inj.drift_hz_per_s_to_channels_per_row(5.0)
    mask = inj.linear_drift_mask(1000, cpr, bandwidth_channels=1)
    cols = [np.where(mask[row])[0][0] for row in range(16)]
    expected = [round(1000 + cpr * row) for row in range(16)]
    assert cols == expected
    # Negative drift mirrors positive.
    neg_mask = inj.linear_drift_mask(2000, -cpr, bandwidth_channels=1)
    neg_cols = [np.where(neg_mask[row])[0][0] for row in range(16)]
    assert neg_cols[0] > neg_cols[-1]


def test_linear_drift_mask_rejects_out_of_frame():
    cpr = inj.drift_hz_per_s_to_channels_per_row(10.0)
    with pytest.raises(ValueError):
        inj.linear_drift_mask(4000, cpr, bandwidth_channels=2)


@pytest.mark.parametrize("duty_cycle,expected_on", [(0.25, 4), (0.5, 8), (0.75, 12)])
def test_intermittent_rows_duty_cycle_counts(duty_cycle, expected_on):
    rows = inj.intermittent_rows(duty_cycle)
    assert len(rows) == expected_on
    assert len(set(rows)) == len(rows)
    assert rows.min() >= 0 and rows.max() <= 15


def test_intermittent_rows_deterministic():
    assert list(inj.intermittent_rows(0.5)) == list(inj.intermittent_rows(0.5))


def test_multiple_carriers_mask_non_overlapping():
    mask = inj.multiple_carriers_mask(500, carrier_count=3, bandwidth_channels=2,
                                      spacing_channels=400)
    assert mask.sum() == 16 * 3 * 2
    cols = np.where(mask[0])[0]
    assert list(cols) == [500, 501, 900, 901, 1300, 1301]


def test_multiple_carriers_requires_at_least_two():
    with pytest.raises(ValueError):
        inj.multiple_carriers_mask(100, carrier_count=1)


def test_local_background_mask_excludes_signal_and_margin():
    mask = inj.stationary_mask(1000, bandwidth_channels=2)
    bg = inj.local_background_mask((16, 4096), mask, guard_channels=256, margin_channels=4)
    # Background must not touch the signal columns or its margin.
    assert not bg[:, 996:1006].any()
    # But should include columns just outside the margin, within the guard band.
    assert bg[:, 1010].all()
    # The guard window itself is bounded: nothing outside [lo_sig-guard, hi_sig+guard].
    assert not bg[:, 1000 - 256 - 1].any()
    assert bg[:, 1000 - 256].all()


def test_local_sigma_matches_manual_mad():
    rng = np.random.default_rng(0)
    frame = rng.normal(loc=10.0, scale=2.0, size=(16, 4096)).astype(np.float32)
    mask = inj.stationary_mask(2000, bandwidth_channels=2)
    bg_mask = inj.local_background_mask((16, 4096), mask)
    sigma = inj.local_sigma(frame, bg_mask)
    bg = frame[bg_mask]
    med = np.median(bg)
    mad = np.median(np.abs(bg - med))
    assert sigma == pytest.approx(1.4826 * mad)


def test_inject_signal_adds_requested_amplitude_only_at_signal_pixels():
    rng = np.random.default_rng(1)
    frame = rng.normal(loc=100.0, scale=5.0, size=(16, 4096)).astype(np.float32)
    frame = np.abs(frame) + 1.0  # keep strictly positive (linear power)
    mask = inj.stationary_mask(1500, bandwidth_channels=2)
    injected, diag = inj.inject_signal(frame, mask, requested_snr=5.0)
    assert np.allclose(injected[~mask], frame[~mask])
    delta = injected[mask] - frame[mask]
    assert np.allclose(delta, diag["amplitude"])
    assert diag["amplitude"] == pytest.approx(5.0 * diag["sigma_local"])
    assert diag["signal_pixels"] == mask.sum()


def test_inject_signal_rejects_nonpositive_snr():
    frame = np.ones((16, 4096), dtype=np.float32) * 10
    mask = inj.stationary_mask(100)
    with pytest.raises(ValueError):
        inj.inject_signal(frame, mask, requested_snr=0.0)


def test_achieved_local_snr_tracks_injection_strength_monotonically():
    rng = np.random.default_rng(2)
    raw = np.abs(rng.normal(loc=1000.0, scale=20.0, size=(16, 4096))).astype(np.float32) + 1.0
    mask = inj.stationary_mask(2048, bandwidth_channels=2)
    bg_mask = inj.local_background_mask((16, 4096), mask)
    clean_norm = normalize_fine_frame(raw)

    achieved = []
    for snr in (0.5, 1.0, 2.0, 5.0, 10.0):
        injected_raw, _ = inj.inject_signal(raw, mask, requested_snr=snr)
        injected_norm = normalize_fine_frame(injected_raw)
        achieved.append(inj.achieved_local_snr(clean_norm, injected_norm, mask, bg_mask))

    assert all(a2 > a1 for a1, a2 in pairwise(achieved))
    assert all(a > 0 for a in achieved)


def test_sample_start_channel_is_in_bounds_and_seeded():
    rng_a = np.random.default_rng(20261013)
    rng_b = np.random.default_rng(20261013)
    span = inj.linear_drift_span_channels(inj.drift_hz_per_s_to_channels_per_row(10.0))
    a = inj.sample_start_channel(rng_a, span)
    b = inj.sample_start_channel(rng_b, span)
    assert a == b
    assert inj.GUARD_CHANNELS <= a <= inj.N_CHANNELS - inj.GUARD_CHANNELS - span


@pytest.mark.parametrize("drift_hz_per_s", [0.05, 0.5, 2, 5, 10])
def test_sample_linear_drift_start_channel_stays_in_bounds_both_signs(drift_hz_per_s):
    rng = np.random.default_rng(20261013)
    for sign in (1, -1):
        cpr = sign * inj.drift_hz_per_s_to_channels_per_row(drift_hz_per_s)
        start = inj.sample_linear_drift_start_channel(rng, cpr)
        mask = inj.linear_drift_mask(start, cpr)  # must not raise
        assert mask.sum() == 16 * inj.BANDWIDTH_CHANNELS


def test_sample_start_channel_rejects_impossible_span():
    with pytest.raises(ValueError):
        inj.sample_start_channel(np.random.default_rng(0), span_channels=10_000)


def test_held_out_morphologies_are_named_but_not_in_development_set():
    assert set(inj.HELD_OUT_MORPHOLOGIES).isdisjoint(inj.DEVELOPMENT_MORPHOLOGIES)
    assert inj.HELD_OUT_MORPHOLOGIES == ("nonlinear_chirp", "burst")
