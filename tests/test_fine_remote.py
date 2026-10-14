import io

import numpy as np
import pytest

from deepseti.data.fine_remote import (
    FRAME_CHANNELS,
    SLAB_CHANNELS,
    FineGeometry,
    FullBodyResponse,
    HTTPRangeFile,
    RangeError,
    assign_strata,
    check_fine_geometry,
    check_range_response,
    eligible_slab_starts,
    eligible_window_starts,
    extract_windows,
    freq_to_channel,
    select_windows,
    slab_window_offsets,
    slice_frames,
    slot_fraction,
    stratum_slab_choice,
    stratum_slots,
)
from deepseti.preprocessing.fine import normalize_fine_frame, normalize_fine_frames

FOFF = -2.7939677238464355e-06


def geom(fch1=2802.83203125, foff=FOFF, nchans=359661568, n_ints=16):
    return FineGeometry(fch1, foff, nchans, n_ints, 18.253611008, (1, 1, 1048576), "32008")


def test_freq_to_channel_negative_foff():
    g = geom()
    assert freq_to_channel(g, g.fch1) == 0
    assert freq_to_channel(g, g.fch1 + 1000 * g.foff) == 1000
    # Lower frequency -> higher channel index when foff < 0.
    assert freq_to_channel(g, 2300.0) > freq_to_channel(g, 2700.0)
    with pytest.raises(ValueError):
        freq_to_channel(g, 3000.0)


def test_freq_to_channel_positive_foff():
    g = geom(fch1=1700.0, foff=-FOFF, nchans=1 << 20)
    assert freq_to_channel(g, 1700.0 + 4096 * g.foff) == 4096


def test_window_bounds_reversed_axis():
    g = geom()
    lo, hi = g.window_freq_bounds(0)
    assert hi == pytest.approx(g.fch1 + abs(FOFF) / 2)
    assert hi - lo == pytest.approx(FRAME_CHANNELS * abs(FOFF))


def test_eligible_windows_respect_band_and_mask():
    g = geom()
    starts = eligible_window_starts(g)
    assert (starts % FRAME_CHANNELS == 0).all()
    assert (starts + FRAME_CHANNELS <= g.nchans).all()
    lo = np.array([g.window_freq_bounds(int(s))[0] for s in starts])
    hi = np.array([g.window_freq_bounds(int(s))[1] for s in starts])
    assert (lo >= 1700.0).all() and (hi <= 2900.0).all()
    assert ((hi <= 2300.0) | (lo >= 2360.0)).all()
    # Grid windows touching the mask exist in the file but are excluded.
    all_starts = np.arange(0, g.nchans - FRAME_CHANNELS + 1, FRAME_CHANNELS)
    assert len(starts) < len(all_starts)


def test_select_windows_deterministic_and_stratified():
    eligible = np.arange(0, 1000 * 4096, 4096)
    a = select_windows(eligible, 16, 20261013, "cad|setup")
    b = select_windows(eligible, 16, 20261013, "cad|setup")
    c = select_windows(eligible, 16, 20261013, "other")
    assert (a == b).all() and not (a == c).all()
    assert len(set(a.tolist())) == 16 and (np.diff(a) > 0).all()
    # One pick per contiguous stratum of the eligible grid.
    strata = np.searchsorted(eligible, a) // (1000 // 16)
    assert len(set(np.minimum(strata, 15).tolist())) == 16
    with pytest.raises(ValueError):
        select_windows(eligible[:4], 16, 1, "k")


def test_geometry_check():
    assert check_fine_geometry(geom()) == []
    assert check_fine_geometry(geom(n_ints=15))
    assert check_fine_geometry(geom(foff=-2.835e-06))


class FakeDataset:
    def __init__(self, arr):
        self.arr, self.shape = arr, arr.shape

    def __getitem__(self, key):
        return self.arr[key]


def test_extract_windows_shape_16_by_4096():
    arr = np.random.default_rng(0).uniform(1, 2, (16, 1, 3 * 4096)).astype(np.float32)
    frames = extract_windows(FakeDataset(arr), np.array([0, 8192]))
    assert frames.shape == (2, 16, 4096)
    assert np.array_equal(frames[1], arr[:, 0, 8192:12288])
    with pytest.raises(ValueError):
        extract_windows(FakeDataset(arr), np.array([10000]))


def test_normalization_finite():
    rng = np.random.default_rng(1)
    frame = rng.lognormal(20, 1, (16, 4096)).astype(np.float32)
    out = normalize_fine_frames(frame[None])[0]
    assert out.shape == (16, 4096) and np.isfinite(out).all()
    assert np.median(out) == pytest.approx(0, abs=1e-5)
    assert np.median(np.abs(out)) == pytest.approx(1, abs=1e-4)
    with pytest.raises(ValueError):
        normalize_fine_frame(np.zeros((16, 4096), np.float32))
    with pytest.raises(ValueError):
        normalize_fine_frame(frame[:, :4095])


# ---- HTTP 206 enforcement -------------------------------------------------


class FakeResp(io.BytesIO):
    def __init__(self, status, body, content_range):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}
        self.read_calls = 0

    def read(self, n=-1):
        self.read_calls += 1
        return super().read(n)


def make_opener(status, total=1 << 20, record=None):
    def opener(req, timeout):
        start, stop = map(int, req.headers["Range"].split("=")[1].split("-"))
        body = bytes(stop - start + 1) if status == 206 else bytes(total)
        cr = f"bytes {start}-{stop}/{total}" if status == 206 else None
        resp = FakeResp(status, body, cr)
        if record is not None:
            record.append(resp)
        return resp

    return opener


def test_check_range_response():
    assert check_range_response(206, "bytes 0-9/100", 0, 9) == 100
    with pytest.raises(FullBodyResponse):
        check_range_response(200, None, 0, 9)
    with pytest.raises(RangeError):
        check_range_response(206, "bytes 0-19/100", 0, 9)
    with pytest.raises(RangeError):
        check_range_response(206, None, 0, 9)


def test_range_file_accepts_206():
    rf = HTTPRangeFile("http://x", budget=1 << 20, opener=make_opener(206))
    assert rf.size == 1 << 20 and rf.stats.statuses == {206: 1}
    rf.seek(100_000)
    assert len(rf.read(200_000)) == 200_000  # large read: one exact range request
    assert rf.stats.requests == 2


def test_range_file_aborts_on_200_without_reading_body():
    record = []
    with pytest.raises(FullBodyResponse):
        HTTPRangeFile("http://x", budget=1 << 30, opener=make_opener(200, record=record))
    assert record[0].read_calls == 0


def test_range_file_budget():
    rf = HTTPRangeFile("http://x", budget=100_000, opener=make_opener(206))
    with pytest.raises(RangeError):
        rf.read(200_000)


# ---- chunk-aware slab rule ------------------------------------------------

def test_slab_window_offsets_42_even_nonoverlapping():
    off = slab_window_offsets()
    assert len(off) == 42 and len(set(off.tolist())) == 42
    assert (off % 4096 == 0).all() and off.min() >= 0
    assert off.max() + 4096 <= SLAB_CHANNELS
    assert (np.diff(off) >= 4096).all()
    assert set(np.diff(off // 4096).tolist()) <= {6, 7}
    assert (slab_window_offsets() == off).all()


def test_eligible_slabs_exclude_mask():
    g = geom()
    s = eligible_slab_starts(g)
    assert (s % SLAB_CHANNELS == 0).all()
    bounds = [g.window_freq_bounds(int(x), SLAB_CHANNELS) for x in s]
    assert all(hi <= 2300.0 or lo >= 2360.0 for lo, hi in bounds)
    assert 300 < len(s) < 343


def test_stratum_choice_and_assignment():
    g = geom()
    s = eligible_slab_starts(g)
    centres = g.channel_freq_mhz(s + SLAB_CHANNELS // 2)
    picks = [stratum_slab_choice(centres, k) for k in range(15)]
    assert picks == sorted(picks) and len(set(picks)) == 15
    groups = [f"G{i:03d}" for i in range(60)]
    a = assign_strata(groups, 20261013, "train")
    assert a == assign_strata(list(reversed(groups)), 20261013, "train")
    assert sorted(np.bincount(list(a.values())).tolist()) == [4] * 15
    v = assign_strata(groups[:15], 20261013, "validation")
    assert sorted(v.values()) == list(range(15))


def test_slice_frames_from_slab():
    slab = np.arange(16 * SLAB_CHANNELS, dtype=np.float32).reshape(16, SLAB_CHANNELS)
    off = slab_window_offsets()
    frames = slice_frames(slab, off)
    assert frames.shape == (42, 16, 4096)
    assert np.array_equal(frames[5], slab[:, off[5] : off[5] + 4096])


def test_prefetch_serves_reads_locally():
    rf = HTTPRangeFile("http://x", budget=1 << 20, opener=make_opener(206))
    rf.prefetch([(200_000, 100_000), (300_000, 50_000), (600_000, 10)])
    assert rf.stats.requests == 3  # first block + two merged spans
    before = rf.stats.bytes
    rf.seek(250_000)
    assert len(rf.read(90_000)) == 90_000
    assert rf.stats.bytes == before


def test_stratum_slots_and_fractional_choice():
    groups = [f"G{i:03d}" for i in range(60)]
    slots = stratum_slots(groups, 20261013, "train")
    assert {k for k, _, _ in slots.values()} == set(range(15))
    assert all(m == 4 for _, _, m in slots.values())
    for k in range(15):
        assert sorted(j for kk, j, _ in slots.values() if kk == k) == [0, 1, 2, 3]
    assert [slot_fraction(j, 4) for j in range(4)] == [0.125, 0.375, 0.625, 0.875]
    assert slot_fraction(0, 1) == 0.5

    g = geom()
    s = eligible_slab_starts(g)
    centres = g.channel_freq_mhz(s + SLAB_CHANNELS // 2)
    edges = np.linspace(0, len(centres), 16).round().astype(int)
    for k in range(15):
        picks = [stratum_slab_choice(centres, k, slot_fraction(j, 4)) for j in range(4)]
        assert all(edges[k] <= i < edges[k + 1] for i in picks)
        assert picks == sorted(picks) and len(set(picks)) == 4
    # q = 0.5 keeps the pilot-2 midpoint choice.
    assert stratum_slab_choice(centres, 3, 0.5) == stratum_slab_choice(centres, 3)
