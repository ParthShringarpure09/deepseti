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


def test_failed_scan_records_attempts(tmp_path, monkeypatch):
    def boom(task, frames_dir, audit_dir):
        raise ex.PermanentFailure("geometry: test")

    monkeypatch.setattr(ex, "_attempt", boom)
    rec = ex.extract_scan({"scan_id": "X", "url": "u"}, str(tmp_path), str(tmp_path), 3)
    assert rec["status"] == "failed" and len(rec["attempts"]) == 1
    assert rec["attempts"][0]["permanent"]
    assert not (tmp_path / "X.json").exists()
    json.dumps(rec)
