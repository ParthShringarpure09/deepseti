"""Data preflight checks for the Fine V2 ConvAE experiment runner.

Verifies, before any training starts, that the local TRAIN split, the real
(clean) VALIDATION split used for checkpoint selection, and the frozen
synthetic VALIDATION benchmark are present, intact and exactly the frozen
scientific state (see ``deepseti.evaluation.fine_benchmark`` and
``configs/experiments/fine_v2_synthetic_benchmark.yaml``). Every check
raises ``PreflightError`` with a message that names the specific directory
or file the caller needs to fix, so a friend running the one-command script
on a fresh machine gets an actionable error instead of a stack trace.

This module never opens ``data/processed/fine_v2/test``; the final-test
refusal guard lives in the runner script itself (next to the frozen
``validate_split_access`` helper it reuses from ``scripts/extract_fine_v2.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_SHAPE = (16, 4096)


class PreflightError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_checksum(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def verify_split_frames(
    metadata_path: str | Path,
    root: str | Path,
    expected_split: str,
    expected_frames: int,
    frame_shape: tuple[int, int] = FRAME_SHAPE,
) -> dict:
    """Verify a processed Fine V2 split (TRAIN or clean VALIDATION) on disk.

    Checks frame count, frame shape, finiteness, and that the metadata
    contains only ``expected_split`` rows. Does not load every scan twice:
    frames are read once, batched by scan file.
    """

    metadata_path = Path(metadata_path)
    root = Path(root)
    _require(metadata_path.exists(),
             f"missing {metadata_path}; copy this directory from the source machine")
    metadata = pd.read_csv(metadata_path)

    _require(len(metadata) == expected_frames,
             f"{metadata_path} has {len(metadata)} rows, expected {expected_frames}")
    _require(set(metadata["split"].unique()) == {expected_split},
             f"{metadata_path} contains splits other than {expected_split!r}: "
             f"{sorted(metadata['split'].unique())}")

    n_checked = 0
    all_finite = True
    for npy_rel, group in metadata.groupby("npy"):
        npy_path = root / npy_rel
        _require(npy_path.exists(),
                 f"missing frame array {npy_path}; copy {root} from the source machine")
        arr = np.load(npy_path)
        indices = group["npy_index"].to_numpy()
        frames = arr[indices]
        if frames.shape[1:] != frame_shape:
            raise PreflightError(
                f"{npy_path} frames have shape {frames.shape[1:]}, expected {frame_shape}")
        if not np.isfinite(frames).all():
            all_finite = False
        n_checked += len(indices)

    _require(all_finite, f"non-finite values found among frames under {root}")
    _require(n_checked == expected_frames,
             f"checked {n_checked} frames under {root}, expected {expected_frames}")

    return {
        "metadata_path": str(metadata_path),
        "root": str(root),
        "n_frames": n_checked,
        "split": expected_split,
        "frame_shape": list(frame_shape),
        "all_finite": True,
    }


def verify_benchmark(
    benchmark_dir: str | Path,
    audit_path: str | Path,
    expected_examples: int,
    held_out_morphologies: tuple[str, ...] = ("nonlinear_chirp", "burst"),
) -> dict:
    """Verify the frozen synthetic VALIDATION benchmark against its audit record.

    Recomputes sha256 checksums of ``clean_frames.npy``, ``injected_frames.npy``
    and ``metadata.csv`` and compares them against the frozen values recorded
    in ``audit_path`` (written once by
    ``scripts/build_fine_v2_validation_benchmark.py`` and never regenerated).
    Never regenerates the benchmark itself.
    """

    benchmark_dir = Path(benchmark_dir)
    audit_path = Path(audit_path)

    _require(audit_path.exists(),
             f"missing frozen benchmark audit {audit_path}; the benchmark must not be regenerated")
    audit = json.loads(audit_path.read_text())
    _require(bool(audit.get("all_pass")),
             f"{audit_path} records all_pass=False; benchmark is not trustworthy")

    for name in ("clean_frames.npy", "injected_frames.npy", "metadata.csv"):
        path = benchmark_dir / name
        _require(path.exists(),
                 f"missing {path}; copy {benchmark_dir} from the source machine")

    clean = np.load(benchmark_dir / "clean_frames.npy")
    injected = np.load(benchmark_dir / "injected_frames.npy")
    # Checksum the file bytes exactly as written (never re-serialized through
    # pandas, which would change float/NaN formatting on a read/to_csv
    # round-trip and make the comparison spuriously fail).
    metadata = pd.read_csv(benchmark_dir / "metadata.csv")

    checks = {
        "clean_frames_checksum_sha256": _array_checksum(clean),
        "injected_frames_checksum_sha256": _array_checksum(injected),
        "metadata_checksum_sha256": _sha256_bytes((benchmark_dir / "metadata.csv").read_bytes()),
    }
    for key, computed in checks.items():
        frozen = audit.get(key)
        _require(frozen == computed,
                 f"{benchmark_dir} checksum mismatch for {key}: "
                 f"frozen={frozen} computed={computed}; benchmark data does not match the "
                 f"frozen development benchmark and must not be used")

    _require(len(metadata) == expected_examples,
             f"{benchmark_dir / 'metadata.csv'} has {len(metadata)} examples, "
             f"expected {expected_examples}")
    _require(clean.shape == (expected_examples, *FRAME_SHAPE),
             f"clean_frames.npy shape {clean.shape} != ({expected_examples}, {FRAME_SHAPE})")
    _require(injected.shape == (expected_examples, *FRAME_SHAPE),
             f"injected_frames.npy shape {injected.shape} != ({expected_examples}, {FRAME_SHAPE})")

    held_out_present = int(metadata["morphology"].isin(held_out_morphologies).sum())
    _require(held_out_present == 0,
             f"{benchmark_dir / 'metadata.csv'} contains {held_out_present} held-out-morphology "
             f"rows ({held_out_morphologies}); the development benchmark must never include them")

    return {
        "benchmark_dir": str(benchmark_dir),
        "n_examples": len(metadata),
        "checksums_match_frozen": True,
        "held_out_morphologies_absent": True,
    }
