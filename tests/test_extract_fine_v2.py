import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_fine_v2 as ex

FOFF = -2.7939677238464355e-06


def test_refuses_validation_and_test(monkeypatch, capsys):
    for split in ("validation", "test"):
        monkeypatch.setattr(sys, "argv", ["extract_fine_v2.py", "--split", split])
        assert ex.main() == 2
        assert "REFUSED" in capsys.readouterr().out


def test_validate_split_access():
    train_cfg = yaml.safe_load(Path(ex.DEFAULT_CONFIG).read_text())
    val_cfg = yaml.safe_load(
        Path("configs/data/fine_v2_validation_access.yaml").read_text())

    assert ex.validate_split_access("train", train_cfg) is None
    # Validation is refused under the frozen training config, even though
    # that config is never supposed to change.
    assert ex.validate_split_access("validation", train_cfg) is not None
    # Validation is permitted only via the dedicated access config.
    assert ex.validate_split_access("validation", val_cfg) is None
    # A config without the marker, even one listing validation as
    # extractable, must not be enough on its own.
    unmarked = {**val_cfg, "access": {}}
    assert ex.validate_split_access("validation", unmarked) is not None
    # Test is refused unconditionally, regardless of config contents.
    assert ex.validate_split_access("test", train_cfg) is not None
    assert ex.validate_split_access("test", val_cfg) is not None
    permissive = {**val_cfg, "split": {"extractable_splits": ["train", "validation", "test"]}}
    assert ex.validate_split_access("test", permissive) is not None


def test_validation_access_config_mirrors_frozen_extraction_params():
    train_cfg = yaml.safe_load(Path(ex.DEFAULT_CONFIG).read_text())
    val_cfg = yaml.safe_load(
        Path("configs/data/fine_v2_validation_access.yaml").read_text())

    assert val_cfg["access"]["kind"] == "fine_v2_validation_access"
    assert val_cfg["access"]["split"] == "validation"
    assert val_cfg["split"]["extractable_splits"] == ["validation"]
    assert val_cfg["protocol"] == train_cfg["protocol"]
    assert val_cfg["inputs"] == train_cfg["inputs"]
    assert val_cfg["geometry"] == train_cfg["geometry"]
    assert val_cfg["sampling"] == train_cfg["sampling"]
    assert val_cfg["preprocessing"] == train_cfg["preprocessing"]
    assert val_cfg["transfer"] == train_cfg["transfer"]
    assert val_cfg["split"]["cadence_groups"] == train_cfg["split"]["cadence_groups"]

    n_cad = val_cfg["split"]["cadence_groups"]["validation"]
    assert n_cad == val_cfg["access"]["expected_cadence_groups"] == 15
    assert n_cad * val_cfg["sampling"]["scans_per_cadence"] == val_cfg["access"]["expected_scans"] == 90
    assert n_cad * val_cfg["sampling"]["frames_per_cadence"] == val_cfg["access"]["expected_frames"] == 3780
    assert val_cfg["access"]["expected_workers"] == val_cfg["transfer"]["workers"] == 4


def test_frozen_config_values():
    cfg = yaml.safe_load(Path(ex.DEFAULT_CONFIG).read_text())
    assert cfg["split"]["extractable_splits"] == ["train"]
    assert cfg["transfer"]["workers"] == 4
    s = cfg["sampling"]
    assert s["frames_per_scan"] * s["scans_per_cadence"] == s["frames_per_cadence"] == 252
    assert cfg["geometry"]["frame_shape"] == [16, 4096]
    assert cfg["protocol"]["seed"] == 20261013


def test_train_tasks_are_train_only():
    cfg = yaml.safe_load(Path(ex.DEFAULT_CONFIG).read_text())
    tasks = ex.build_tasks(cfg, "train")
    assert len(tasks) == 360 and {t["split"] for t in tasks} == {"train"}
    assert len({t["scan_id"] for t in tasks}) == 360
    # Same slab for all six positions of each cadence.
    by_group = {}
    for t in tasks:
        by_group.setdefault(t["group_id"], set()).add(t["slab_start_channel"])
    assert len(by_group) == 60 and all(len(v) == 1 for v in by_group.values())


def test_atomic_save_and_resume_check(tmp_path):
    arr = np.ones((42, 16, 4096), np.float32)
    npy = tmp_path / "S.npy"
    ex.atomic_save_npy(npy, arr)
    assert not list(tmp_path.glob(".*tmp*"))
    assert ex.is_complete(tmp_path, "S") is None  # no sidecar yet
    ex.atomic_write_json(tmp_path / "S.json", {"status": "complete",
                                              "npy_sha256": ex.sha256(npy)})
    assert ex.is_complete(tmp_path, "S") is not None
    arr[0, 0, 0] = 2
    np.save(npy, arr)  # corrupted / changed output must not count as complete
    assert ex.is_complete(tmp_path, "S") is None
    (tmp_path / "S.json").write_text("{trunc")
    assert ex.is_complete(tmp_path, "S") is None


def test_frame_metadata_channels_and_frequencies():
    rec = {
        "scan_id": "G001_pos0_1", "npy": "G001_pos0_1.npy", "split": "train",
        "group_id": "G001", "cadence_id": "c", "session": "s", "target": "T", "on_target": "T",
        "cadence_position": 0, "cadence_role": "ON_A", "archive_id": 1, "filename": "f",
        "url": "u", "archive_md5sum": "m", "reduction_pipeline": "gpuspec", "stratum": 0,
        "stratum_slot": 1, "slab_start_channel": 7340032, "slab_freq_mhz": [1.0, 2.0],
        "frame_offsets_in_slab": [12288, 36864],
        "geometry": {"fch1": 2802.83203125, "foff": FOFF},
    }
    meta = ex.frame_metadata([rec], [16, 4096])
    assert list(meta["frame_id"]) == ["G001_pos0_1_f00", "G001_pos0_1_f01"]
    assert list(meta["abs_channel_start"]) == [7340032 + 12288, 7340032 + 36864]
    width = meta["frame_hi_mhz"] - meta["frame_lo_mhz"]
    assert np.allclose(width, 4096 * abs(FOFF))
    hi0 = 2802.83203125 + (7340032 + 12288) * FOFF + abs(FOFF) / 2
    assert meta["frame_hi_mhz"].iloc[0] == pytest.approx(hi0)


def _write_complete_scan(frames_dir: Path, task: dict) -> None:
    arr = np.ones((task["frames_per_scan"], *task["frame_shape"]), np.float32)
    npy = frames_dir / f"{task['scan_id']}.npy"
    ex.atomic_save_npy(npy, arr)
    rec = {
        "scan_id": task["scan_id"], "split": task["split"], "group_id": task["group_id"],
        "cadence_id": "c", "session": "s", "target": "T", "on_target": "T",
        "cadence_position": 0, "cadence_role": "ON_A", "archive_id": task["archive_id"],
        "filename": "f", "url": "u", "archive_md5sum": "m", "reduction_pipeline": "gpuspec",
        "stratum": 0, "stratum_slot": 0, "slab_start_channel": 0, "slab_freq_mhz": [1.0, 2.0],
        "frame_offsets_in_slab": [0] * task["frames_per_scan"],
        "geometry": {"fch1": 1.0, "foff": -1e-6},
        "npy": npy.name, "status": "complete", "npy_sha256": ex.sha256(npy),
        "http_statuses": {"206": 1}, "full_body_responses": 0, "bytes": 0,
    }
    ex.atomic_write_json(frames_dir / f"{task['scan_id']}.json", rec)


def test_verify_split_contamination_invariant(tmp_path):
    manifest_csv = tmp_path / "manifest.csv"
    chunk_csv = tmp_path / "chunk_plan.csv"
    manifest_csv.write_text("archive_id,split\n1,validation\n2,train\n")
    chunk_csv.write_text("x\n1\n")
    cfg = {
        "sampling": {"frames_per_scan": 1, "scans_per_cadence": 1, "frames_per_cadence": 1},
        "geometry": {"frame_shape": [1, 2]},
        "split": {"cadence_groups": {"train": 1, "validation": 1, "test": 1}},
        "inputs": {
            "manifest": str(manifest_csv), "manifest_md5": ex.md5(manifest_csv),
            "chunk_plan": str(chunk_csv), "chunk_plan_md5": ex.md5(chunk_csv),
        },
    }
    out = tmp_path / "out"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True)
    task = {"scan_id": "G1_pos0_1", "split": "validation", "group_id": "G1",
            "archive_id": 1, "frames_per_scan": 1, "frame_shape": [1, 2]}
    _write_complete_scan(frames_dir, task)
    state = {"sessions": []}

    # Clean case: the only source row pulled is the validation row itself.
    result = ex.verify(cfg, "validation", out, [task], state)
    inv = result["invariants"]
    assert inv["validation_rows"] == 1
    assert inv["train_rows"] == 0 and inv["test_rows"] == 0
    assert inv["other_split_rows_requested_zero"] is True
    assert inv["all_pass"] is True

    # Contaminated case: a train-split row's archive_id leaked into the
    # extracted metadata (e.g. a manifest edit broke split isolation).
    manifest_csv.write_text("archive_id,split\n1,train\n")
    cfg["inputs"]["manifest_md5"] = ex.md5(manifest_csv)
    result = ex.verify(cfg, "validation", out, [task], state)
    inv = result["invariants"]
    assert inv["train_rows"] == 1
    assert inv["other_split_rows_requested_zero"] is False
    assert inv["all_pass"] is False


def test_failed_scan_records_attempts(tmp_path, monkeypatch):
    def boom(task, frames_dir, audit_dir):
        raise ex.PermanentFailure("geometry: test")

    monkeypatch.setattr(ex, "_attempt", boom)
    rec = ex.extract_scan({"scan_id": "X", "url": "u"}, str(tmp_path), str(tmp_path), 3)
    assert rec["status"] == "failed" and len(rec["attempts"]) == 1
    assert rec["attempts"][0]["permanent"]
    assert not (tmp_path / "X.json").exists()
    json.dumps(rec)
