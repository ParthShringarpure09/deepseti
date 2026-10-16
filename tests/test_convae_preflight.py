import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from deepseti.evaluation.convae_preflight import (
    PreflightError,
    verify_benchmark,
    verify_split_frames,
)

FRAME_SHAPE = (16, 4096)


def _write_split(tmp_path, split="train", n_scans=3, frames_per_scan=4,
                 shape=FRAME_SHAPE, finite=True):
    root = tmp_path / split
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_scans):
        scan_id = f"S{s:02d}"
        arr = rng.normal(size=(frames_per_scan, *shape)).astype(np.float32)
        if not finite:
            arr[0, 0, 0] = np.nan
        np.save(frames_dir / f"{scan_id}.npy", arr)
        for i in range(frames_per_scan):
            rows.append({"npy": f"frames/{scan_id}.npy", "npy_index": i, "split": split})
    metadata = pd.DataFrame(rows)
    metadata_path = root / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    return metadata_path, root, n_scans * frames_per_scan


def test_verify_split_frames_success(tmp_path):
    metadata_path, root, n = _write_split(tmp_path)
    report = verify_split_frames(metadata_path, root, "train", n, FRAME_SHAPE)
    assert report["n_frames"] == n
    assert report["split"] == "train"
    assert report["all_finite"] is True


def test_verify_split_frames_rejects_wrong_count(tmp_path):
    metadata_path, root, n = _write_split(tmp_path)
    with pytest.raises(PreflightError, match="expected"):
        verify_split_frames(metadata_path, root, "train", n + 1, FRAME_SHAPE)


def test_verify_split_frames_rejects_other_split(tmp_path):
    metadata_path, root, n = _write_split(tmp_path, split="train")
    with pytest.raises(PreflightError, match="splits other than"):
        verify_split_frames(metadata_path, root, "validation", n, FRAME_SHAPE)


def test_verify_split_frames_rejects_non_finite(tmp_path):
    metadata_path, root, n = _write_split(tmp_path, finite=False)
    with pytest.raises(PreflightError, match="non-finite"):
        verify_split_frames(metadata_path, root, "train", n, FRAME_SHAPE)


def test_verify_split_frames_rejects_missing_metadata(tmp_path):
    with pytest.raises(PreflightError, match="missing"):
        verify_split_frames(tmp_path / "nope.csv", tmp_path, "train", 10, FRAME_SHAPE)


def _write_benchmark(tmp_path, n=4, morphologies=None, tamper=None):
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    rng = np.random.default_rng(1)
    clean = rng.normal(size=(n, *FRAME_SHAPE)).astype(np.float32)
    injected = clean + rng.normal(scale=0.1, size=(n, *FRAME_SHAPE)).astype(np.float32)
    morphs = morphologies or ["stationary"] * n
    metadata = pd.DataFrame({
        "example_id": range(n),
        "group_id": [f"G{i:03d}" for i in range(n)],
        "morphology": morphs,
        "requested_snr": [1.0] * n,
    })

    np.save(bench_dir / "clean_frames.npy", clean)
    np.save(bench_dir / "injected_frames.npy", injected)
    metadata.to_csv(bench_dir / "metadata.csv", index=False)

    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def array_checksum(arr: np.ndarray) -> str:
        return sha256_bytes(np.ascontiguousarray(arr).tobytes())

    audit = {
        "all_pass": True,
        "clean_frames_checksum_sha256": array_checksum(clean),
        "injected_frames_checksum_sha256": array_checksum(injected),
        "metadata_checksum_sha256": sha256_bytes((bench_dir / "metadata.csv").read_bytes()),
    }
    if tamper:
        audit[tamper] = "deadbeef"

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    return bench_dir, audit_path


def test_verify_benchmark_success(tmp_path):
    bench_dir, audit_path = _write_benchmark(tmp_path, n=4)
    report = verify_benchmark(bench_dir, audit_path, expected_examples=4)
    assert report["n_examples"] == 4
    assert report["checksums_match_frozen"] is True
    assert report["held_out_morphologies_absent"] is True


def test_verify_benchmark_rejects_checksum_mismatch(tmp_path):
    bench_dir, audit_path = _write_benchmark(tmp_path, n=4,
                                             tamper="clean_frames_checksum_sha256")
    with pytest.raises(PreflightError, match="checksum mismatch"):
        verify_benchmark(bench_dir, audit_path, expected_examples=4)


def test_verify_benchmark_rejects_held_out_morphology(tmp_path):
    bench_dir, audit_path = _write_benchmark(
        tmp_path, n=4, morphologies=["stationary", "stationary", "stationary", "burst"])
    with pytest.raises(PreflightError, match="held-out"):
        verify_benchmark(bench_dir, audit_path, expected_examples=4)


def test_verify_benchmark_rejects_wrong_example_count(tmp_path):
    bench_dir, audit_path = _write_benchmark(tmp_path, n=4)
    with pytest.raises(PreflightError, match="expected 5"):
        verify_benchmark(bench_dir, audit_path, expected_examples=5)


def test_verify_benchmark_rejects_missing_audit(tmp_path):
    bench_dir, _ = _write_benchmark(tmp_path, n=4)
    with pytest.raises(PreflightError, match="missing"):
        verify_benchmark(bench_dir, tmp_path / "no_audit.json", expected_examples=4)
