"""End-to-end dry runs of the two orchestration scripts against tiny fake

fixtures (not real Fine V2 data), to catch script-level glue bugs (paths,
preflight checks, plotting calls) that the pure-function unit tests for
``deepseti.evaluation.fine_benchmark`` / ``pca_scoring`` don't exercise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_fine_v2_validation_benchmark as bb
import run_fine_v2_pca_baseline as pb

from deepseti.evaluation import fine_benchmark as fb

MINI_COMBOS = [
    {"morphology": "stationary", "requested_snr": 5.0},
    {"morphology": "linear_drift", "requested_snr": 5.0, "drift_hz_per_s": 0.05},
    {"morphology": "linear_drift", "requested_snr": 5.0, "drift_hz_per_s": -0.05},
    {"morphology": "linear_drift", "requested_snr": 5.0, "drift_hz_per_s": 10.0},
    {"morphology": "linear_drift", "requested_snr": 5.0, "drift_hz_per_s": -10.0},
    {"morphology": "intermittent", "requested_snr": 5.0, "duty_cycle": 0.5},
    {"morphology": "multiple_carriers", "requested_snr": 5.0, "carrier_count": 2},
]


def _fake_validation_metadata(n_groups=15, frames_per_group=8):
    rows = []
    for g in range(n_groups):
        group_id = f"G{g:03d}"
        scan_id = f"{group_id}_scan0"
        for i in range(frames_per_group):
            rows.append({
                "frame_id": f"{scan_id}_f{i:02d}", "npy": f"frames/{scan_id}.npy",
                "npy_index": i, "split": "validation", "group_id": group_id,
                "cadence_id": f"cad_{group_id}",
            })
    return pd.DataFrame(rows)


def _write_fake_raw_cache(raw_dir, metadata, seed=0):
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scans = metadata.assign(scan_id=metadata["npy"].str.removeprefix("frames/")
                            .str.removesuffix(".npy")).groupby("scan_id")
    for scan_id, rows in scans:
        n = int(rows["npy_index"].max()) + 1
        arr = np.abs(rng.normal(loc=100.0, scale=5.0, size=(n, 16, 4096))).astype(np.float32) + 1.0
        np.save(raw_dir / f"{scan_id}.npy", arr)


@pytest.fixture
def fake_validation_setup(tmp_path, monkeypatch):
    val_meta = _fake_validation_metadata()
    raw_dir = tmp_path / "validation_raw"
    _write_fake_raw_cache(raw_dir, val_meta)
    val_meta_path = tmp_path / "validation_metadata.csv"
    val_meta.to_csv(val_meta_path, index=False)

    monkeypatch.setattr(bb, "VALIDATION_METADATA", str(val_meta_path))
    monkeypatch.setattr(bb, "RAW_CACHE_DIR", str(raw_dir))
    monkeypatch.setattr(bb, "FINAL_TEST_DIR", tmp_path / "never_created_test_dir")
    monkeypatch.setattr(bb, "OUT_FRAMES_DIR", tmp_path / "out_frames")
    monkeypatch.setattr(bb, "OUT_AUDIT", tmp_path / "out_audit.json")
    monkeypatch.setattr(bb, "OUT_FIGURES", tmp_path / "out_figures")
    monkeypatch.setattr(fb, "build_combo_list", lambda: MINI_COMBOS)

    return tmp_path


def test_build_benchmark_script_preflight_refuses_if_test_dir_exists(fake_validation_setup, monkeypatch):
    monkeypatch.setattr(bb, "FINAL_TEST_DIR", fake_validation_setup)  # exists
    with pytest.raises(SystemExit, match="final test must remain untouched"):
        bb.preflight_checks()


def test_build_benchmark_script_preflight_refuses_missing_raw_cache(fake_validation_setup, monkeypatch):
    monkeypatch.setattr(bb, "RAW_CACHE_DIR", str(fake_validation_setup / "nonexistent"))
    with pytest.raises(SystemExit, match="raw-cache scans missing"):
        bb.preflight_checks()


def test_build_benchmark_script_main_end_to_end(fake_validation_setup):
    assert bb.main() == 0

    out_frames = bb.OUT_FRAMES_DIR
    assert (out_frames / "clean_frames.npy").exists()
    assert (out_frames / "injected_frames.npy").exists()
    assert (out_frames / "metadata.csv").exists()
    assert bb.OUT_AUDIT.exists()

    clean = np.load(out_frames / "clean_frames.npy")
    injected = np.load(out_frames / "injected_frames.npy")
    metadata = pd.read_csv(out_frames / "metadata.csv")
    n_expected = len(MINI_COMBOS) * 15
    assert clean.shape == (n_expected, 16, 4096)
    assert injected.shape == (n_expected, 16, 4096)
    assert len(metadata) == n_expected

    figures = sorted(bb.OUT_FIGURES.glob("*.png"))
    assert len(figures) == 6  # clean_baseline + 5 morphology picks

    import json
    audit = json.loads(bb.OUT_AUDIT.read_text())
    assert audit["all_pass"] is True
    assert audit["deterministic_regeneration"]["metadata_identical"] is True


@pytest.fixture
def fake_pca_setup(fake_validation_setup, monkeypatch):
    # Run the real benchmark builder first so run_fine_v2_pca_baseline has
    # genuine (fake-data) benchmark frames to score.
    assert bb.main() == 0

    train_root = fake_validation_setup / "train"
    frames_dir = train_root / "frames"
    frames_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    rows = []
    for s in range(10):
        scan_id = f"T{s:02d}"
        arr = np.abs(rng.normal(loc=100.0, scale=5.0, size=(42, 16, 4096))).astype(np.float32) + 1.0
        np.save(frames_dir / f"{scan_id}.npy", arr)
        for i in range(42):
            rows.append({"npy": f"frames/{scan_id}.npy", "npy_index": i})
    pd.DataFrame(rows).to_csv(train_root / "metadata.csv", index=False)

    monkeypatch.setattr(pb, "TRAIN_ROOT", train_root)
    monkeypatch.setattr(pb, "BENCHMARK_DIR", bb.OUT_FRAMES_DIR)
    monkeypatch.setattr(pb, "N_COMPONENTS", 8)
    monkeypatch.setattr(pb, "BATCH_SCANS", 5)
    monkeypatch.setattr(pb, "OUT_RESULTS", fake_validation_setup / "pca_results.json")
    monkeypatch.setattr(pb, "OUT_FIGURES", fake_validation_setup / "pca_figures")
    return fake_validation_setup


def test_pca_baseline_script_main_end_to_end(fake_pca_setup):
    assert pb.main() == 0

    import json
    results = json.loads(pb.OUT_RESULTS.read_text())
    assert results["pca"]["n_components"] == 8
    assert results["n_benchmark_examples"] == len(MINI_COMBOS) * 15
    assert "mean_score" in results["summary"] and "top_fraction_score" in results["summary"]
    for key in ("mean_score", "top_fraction_score"):
        assert 0.0 <= results["summary"][key]["auroc"] <= 1.0
        ci = results["summary"][key]["auroc_grouped_bootstrap_ci95"]
        assert ci[0] <= ci[1]

    figures = sorted(pb.OUT_FIGURES.glob("*.png"))
    # 5 summary figures + 6 residual-example figures.
    assert len(figures) == 11
