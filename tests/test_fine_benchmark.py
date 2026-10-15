import numpy as np
import pandas as pd
import pytest

from deepseti.evaluation import fine_benchmark as fb
from deepseti.injection import fine as inj


def _fake_validation_metadata(n_groups=15, frames_per_group=4):
    rows = []
    for g in range(n_groups):
        group_id = f"G{g:03d}"
        scan_id = f"{group_id}_scan0"
        for i in range(frames_per_group):
            rows.append({
                "frame_id": f"{scan_id}_f{i:02d}",
                "npy": f"frames/{scan_id}.npy",
                "npy_index": i,
                "split": "validation",
                "group_id": group_id,
                "cadence_id": f"cad_{group_id}",
            })
    return pd.DataFrame(rows)


def _write_fake_raw_cache(tmp_path, metadata, seed=0):
    rng = np.random.default_rng(seed)
    for (scan_id,), rows in metadata.assign(
            scan_id=metadata["npy"].str.removeprefix("frames/").str.removesuffix(".npy")
    ).groupby(["scan_id"]):
        n = int(rows["npy_index"].max()) + 1
        arr = np.abs(rng.normal(loc=100.0, scale=5.0, size=(n, 16, 4096))).astype(np.float32) + 1.0
        np.save(tmp_path / f"{scan_id}.npy", arr)


MINI_COMBOS = [
    {"morphology": "stationary", "requested_snr": 5.0},
    {"morphology": "linear_drift", "requested_snr": 5.0, "drift_hz_per_s": 2.0},
    {"morphology": "linear_drift", "requested_snr": 5.0, "drift_hz_per_s": -2.0},
    {"morphology": "intermittent", "requested_snr": 5.0, "duty_cycle": 0.5},
]


def test_build_combo_list_is_only_development_morphologies():
    combos = fb.build_combo_list()
    assert len(combos) == 80
    assert {c["morphology"] for c in combos} == set(inj.DEVELOPMENT_MORPHOLOGIES)


def test_assign_base_frames_are_unique_per_group_and_deterministic():
    meta = _fake_validation_metadata(n_groups=15, frames_per_group=5)
    a = fb.assign_base_frames(meta, n_per_group=3, seed=42)
    b = fb.assign_base_frames(meta, n_per_group=3, seed=42)
    for g in a:
        assert len(set(a[g])) == 3
        assert list(a[g]) == list(b[g])


def test_assign_base_frames_rejects_insufficient_pool():
    meta = _fake_validation_metadata(n_groups=15, frames_per_group=2)
    with pytest.raises(ValueError):
        fb.assign_base_frames(meta, n_per_group=3, seed=42)


def test_build_benchmark_mini_end_to_end(tmp_path):
    meta_in = _fake_validation_metadata(n_groups=15, frames_per_group=4)
    _write_fake_raw_cache(tmp_path, meta_in)

    clean, injected, meta = fb.build_benchmark(meta_in, tmp_path, seed=123, combos=MINI_COMBOS)

    n_expected = len(MINI_COMBOS) * 15
    assert clean.shape == (n_expected, 16, 4096)
    assert injected.shape == (n_expected, 16, 4096)
    assert np.isfinite(clean).all() and np.isfinite(injected).all()
    assert meta["base_frame_id"].nunique() == n_expected  # every example a distinct base frame
    assert set(meta["group_id"]) == set(meta_in["group_id"])
    assert (meta["requested_snr"] == 5.0).all()
    assert meta["achieved_local_snr"].gt(0).all()
    assert (meta["outside_mask_max_abs_diff"] < fb.OUTSIDE_MASK_DIFF_TOLERANCE).all()


def test_build_benchmark_is_deterministic(tmp_path):
    meta_in = _fake_validation_metadata(n_groups=15, frames_per_group=4)
    _write_fake_raw_cache(tmp_path, meta_in)

    clean1, injected1, meta1 = fb.build_benchmark(meta_in, tmp_path, seed=7, combos=MINI_COMBOS)
    clean2, injected2, meta2 = fb.build_benchmark(meta_in, tmp_path, seed=7, combos=MINI_COMBOS)

    assert np.array_equal(clean1, clean2)
    assert np.array_equal(injected1, injected2)
    pd.testing.assert_frame_equal(meta1, meta2)


def test_audit_benchmark_mini_passes_structural_checks(tmp_path):
    meta_in = _fake_validation_metadata(n_groups=15, frames_per_group=4)
    _write_fake_raw_cache(tmp_path, meta_in)
    clean, injected, meta = fb.build_benchmark(meta_in, tmp_path, seed=99, combos=MINI_COMBOS)

    audit = fb.audit_benchmark(clean, injected, meta, combos=MINI_COMBOS)
    assert audit["all_source_rows_validation"] is True
    assert audit["all_base_frames_unique"] is True
    assert audit["clean_shape_ok"] and audit["injected_shape_ok"]
    assert audit["clean_finite"] and audit["injected_finite"]
    assert audit["morphology_counts_match_expected"] is True
    assert audit["held_out_morphology_count_zero"] is True
    assert audit["no_track_crosses_boundary"] is True
    assert audit["outside_mask_diff_negligible"] is True
    assert audit["all_pass"] is True


def test_expected_combo_counts_sum_to_1200_for_real_design():
    counts = fb.expected_combo_counts()
    assert sum(counts.values()) == 1200
    assert counts == {
        "stationary": 75,
        "linear_drift": 750,
        "intermittent": 225,
        "multiple_carriers": 150,
    }
