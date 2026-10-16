import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_fine_v2_convae_experiment as runner

from deepseti.evaluation import pca_scoring
from deepseti.evaluation.convae_preflight import PreflightError

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiments/fine_v2_convae.yaml"


# ---------------------------------------------------------------------------
# config parsing / frozen constants
# ---------------------------------------------------------------------------


def test_real_config_parses():
    cfg = runner.load_config(REPO_CONFIG)
    assert cfg["experiment"]["seed"] == 20261013
    assert cfg["data"]["expected_train_frames"] == 15120
    assert cfg["data"]["expected_validation_frames"] == 3780
    assert cfg["data"]["expected_benchmark_examples"] == 1200
    assert cfg["evaluation"]["top_fraction"] == pytest.approx(0.002)


def test_top_fraction_matches_frozen_scoring_module():
    cfg = runner.load_config(REPO_CONFIG)
    assert float(cfg["evaluation"]["top_fraction"]) == pca_scoring.TOP_FRACTION


def test_load_config_rejects_non_dict(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(TypeError, match="dict"):
        runner.load_config(path)


# ---------------------------------------------------------------------------
# windows-safe paths
# ---------------------------------------------------------------------------


def test_resolve_returns_absolute_pathlib_path():
    resolved = runner.resolve("data/processed/fine_v2/train")
    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
    assert resolved == runner.ROOT / "data" / "processed" / "fine_v2" / "train"


def test_resolve_passes_through_absolute_paths(tmp_path):
    resolved = runner.resolve(str(tmp_path))
    assert resolved == tmp_path


# ---------------------------------------------------------------------------
# result serialization
# ---------------------------------------------------------------------------


def test_write_json_round_trip(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    path = tmp_path / "out" / "result.json"
    runner.write_json(path, payload)
    assert path.exists()
    import json
    assert json.loads(path.read_text()) == payload


def test_save_checkpoint_round_trip(tmp_path):
    path = tmp_path / "ckpt" / "model.pt"
    state = {"epoch": 3, "value": 1.5}
    runner.save_checkpoint(path, state)
    loaded = torch.load(path, map_location="cpu")
    assert loaded == state


# ---------------------------------------------------------------------------
# final-test guard
# ---------------------------------------------------------------------------


def test_guard_final_test_untouched_raises_if_test_dir_exists(tmp_path, monkeypatch):
    (tmp_path / "test").mkdir()
    access_cfg_path = tmp_path / "access.yaml"
    access_cfg_path.write_text(yaml.safe_dump({
        "access": {"kind": "fine_v2_validation_access"},
        "split": {"extractable_splits": ["validation"]},
        "protocol": {"revision": 2},
    }))
    data_cfg = {
        "final_test_dir": str(tmp_path / "test"),
        "validation_access_config": str(access_cfg_path),
    }
    with pytest.raises(PreflightError, match="final test must remain untouched"):
        runner.guard_final_test_untouched(data_cfg)


def test_guard_final_test_untouched_passes_with_real_access_config(tmp_path):
    data_cfg = {
        "final_test_dir": str(tmp_path / "does_not_exist"),
        "validation_access_config": "configs/data/fine_v2_validation_access.yaml",
    }
    report = runner.guard_final_test_untouched(data_cfg)
    assert report["final_test_dir_absent"] is True
    assert report["test_split_refused"] is True


# ---------------------------------------------------------------------------
# frame loading (no synthetic data reachable through this path)
# ---------------------------------------------------------------------------


def _write_fake_split(tmp_path, n_scans=3, frames_per_scan=4, shape=(16, 4096)):
    import pandas as pd

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_scans):
        scan_id = f"S{s:02d}"
        arr = rng.normal(size=(frames_per_scan, *shape)).astype(np.float32)
        np.save(frames_dir / f"{scan_id}.npy", arr)
        for i in range(frames_per_scan):
            rows.append({"npy": f"frames/{scan_id}.npy", "npy_index": i, "split": "train"})
    metadata = pd.DataFrame(rows)
    return metadata, tmp_path


def test_load_frames_matches_source_arrays(tmp_path):
    metadata, root = _write_fake_split(tmp_path, n_scans=3, frames_per_scan=4)
    frames = runner.load_frames(metadata, root)
    assert frames.shape == (12, 16, 4096)
    for npy_rel, group in metadata.groupby("npy"):
        arr = np.load(root / npy_rel)
        for row_pos, npy_index in zip(group.index, group["npy_index"], strict=True):
            assert np.array_equal(frames[row_pos], arr[npy_index])


def test_load_frames_never_touches_synthetic_directory(tmp_path, monkeypatch):
    metadata, root = _write_fake_split(tmp_path, n_scans=2, frames_per_scan=2)

    real_load = np.load

    def guarded_load(path, *args, **kwargs):
        if "synthetic" in str(path):
            raise AssertionError(f"load_frames must never open synthetic data: {path}")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    runner.load_frames(metadata, root)


# ---------------------------------------------------------------------------
# GPU-memory batch-size sanity check is a no-op off CUDA
# ---------------------------------------------------------------------------


def test_sanity_check_batch_size_is_noop_on_cpu():
    model, _ = runner.build_model()
    batch_size = runner.sanity_check_batch_size(model, torch.device("cpu"), 64, 8)
    assert batch_size == 64


# ---------------------------------------------------------------------------
# checkpoint selection uses CLEAN VALIDATION loss only
# ---------------------------------------------------------------------------


def test_checkpoint_selection_uses_clean_validation_loss_only(tmp_path, monkeypatch):
    """Best epoch must track argmin(validation_loss), independent of the
    (monotonically-improving, intentionally misleading) training loss, and
    with no synthetic score ever in the picture."""

    model, _ = runner.build_model()
    train_losses = [0.9, 0.8, 0.7, 0.6, 0.5]  # always improving
    validation_losses = [0.9, 0.5, 0.6, 0.4, 0.7]  # best at epoch 4 (index 3)

    def fake_train_epoch_wrapper(model, loader, optimizer, device, **kwargs):
        value = train_losses[fake_train_epoch_wrapper.calls]
        fake_train_epoch_wrapper.calls += 1
        return {"loss": value}

    fake_train_epoch_wrapper.calls = 0

    def fake_evaluate_epoch(model, loader, device):
        value = validation_losses[fake_evaluate_epoch.calls]
        fake_evaluate_epoch.calls += 1
        return {"loss": value}

    fake_evaluate_epoch.calls = 0

    monkeypatch.setattr(runner, "train_convae_epoch", fake_train_epoch_wrapper)
    monkeypatch.setattr(runner, "evaluate_convae_epoch", fake_evaluate_epoch)

    frames = np.random.default_rng(0).normal(size=(8, 16, 4096)).astype(np.float32)
    training_cfg = {
        "num_workers": 0, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "amp": False, "grad_clip_norm": 1.0, "early_stopping_patience": 12,
    }

    result = runner.run_training(
        model, frames, frames, torch.device("cpu"), training_cfg,
        batch_size=4, epochs=5, seed=20261013,
        checkpoint_path=tmp_path / "best.pt",
    )

    assert result["best_epoch"] == 4
    assert result["best_validation_loss"] == pytest.approx(0.4)
    assert (tmp_path / "best.pt").exists()


# ---------------------------------------------------------------------------
# dry-run integration (real repo data; must stay fast and not full-train)
# ---------------------------------------------------------------------------


def test_dry_run_end_to_end(tmp_path):
    dry_run_checkpoint = runner.ROOT / "checkpoints" / "dry_run_fine_v2_convae_best.pt"
    try:
        code = runner.main([
            "--dry-run", "--allow-non-cuda", "--output-dir", str(tmp_path),
        ])
        assert code == 0
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "experiment_summary.json").exists()
        assert (tmp_path / "training_history.json").exists()
        assert (tmp_path / "comparison_with_pca.csv").exists()
        assert (tmp_path / "figures").is_dir()
        assert dry_run_checkpoint.exists()

        import json
        summary = json.loads((tmp_path / "experiment_summary.json").read_text())
        assert summary["dry_run"] is True
        assert summary["seed"] == 20261013
        assert summary["data_preflight"]["final_test_access"] == "disabled"
    finally:
        dry_run_checkpoint.unlink(missing_ok=True)


def test_check_only_reports_readiness():
    code = runner.main(["--check-only"])
    # On this machine (no CUDA) the honest answer is NOT READY; the point of
    # this test is that --check-only runs preflight and returns without
    # raising, not that this particular machine is training-ready.
    assert code in (0, 1)
