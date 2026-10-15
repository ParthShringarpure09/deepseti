import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_fine_v2_pca_baseline as pb


def _write_fake_train(tmp_path, n_scans=6, frames_per_scan=4, shape=(3, 5)):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for s in range(n_scans):
        scan_id = f"S{s:02d}"
        arr = rng.normal(size=(frames_per_scan, *shape)).astype(np.float32)
        np.save(frames_dir / f"{scan_id}.npy", arr)
        for i in range(frames_per_scan):
            rows.append({"npy": f"frames/{scan_id}.npy", "npy_index": i})
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)


def test_iter_train_batches_covers_every_frame_exactly_once(tmp_path):
    _write_fake_train(tmp_path, n_scans=6, frames_per_scan=4, shape=(3, 5))
    total = 0
    for batch in pb.iter_train_batches(tmp_path, batch_scans=2):
        assert batch.shape[1] == 3 * 5
        total += batch.shape[0]
    assert total == 6 * 4


def test_iter_train_batches_is_order_independent_of_batch_size(tmp_path):
    _write_fake_train(tmp_path, n_scans=6, frames_per_scan=4, shape=(3, 5))
    a = np.concatenate(list(pb.iter_train_batches(tmp_path, batch_scans=2)), axis=0)
    b = np.concatenate(list(pb.iter_train_batches(tmp_path, batch_scans=3)), axis=0)
    assert np.array_equal(a, b)


def test_breakdown_by_groups_correctly():
    metadata = pd.DataFrame({
        "requested_snr": [0.5, 0.5, 5.0, 5.0],
    })
    clean = np.array([0.0, 0.0, 0.0, 0.0])
    injected = np.array([0.1, 0.1, 10.0, 10.0])  # weak separation at 0.5, strong at 5.0
    out = pb.breakdown_by(metadata, clean, injected, "requested_snr")
    assert set(out.keys()) == {"0.5", "5.0"}
    assert out["5.0"]["auroc"] >= out["0.5"]["auroc"]
    assert out["0.5"]["n"] == 2 and out["5.0"]["n"] == 2


def test_breakdown_by_with_transform_pools_sign():
    metadata = pd.DataFrame({"drift_hz_s": [2.0, -2.0, 5.0, -5.0]})
    clean = np.zeros(4)
    injected = np.array([1.0, 1.0, 2.0, 2.0])
    out = pb.breakdown_by(metadata, clean, injected, "drift_hz_s", transform=abs)
    assert set(out.keys()) == {"2.0", "5.0"}
    assert out["2.0"]["n"] == 2 and out["5.0"]["n"] == 2
