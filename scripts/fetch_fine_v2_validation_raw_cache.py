"""Fetch pre-normalization (raw, linear-power) Fine V2 VALIDATION frames.

The frozen production extractor (``scripts/extract_fine_v2.py``) only stores
the final log10 + per-frame median/MAD normalized frames; it does not retain
enough information to invert that normalization (the per-frame median and MAD
used are not recorded, only aggregate sanity bounds). Synthetic signal
injection must happen in the linear-power domain *before* that normalization
(see ``src/deepseti/injection/fine.py``), so this script performs the
smallest additional VALIDATION-ONLY re-read needed to obtain those
pre-normalization frames.

It reuses the frozen validation task plan verbatim (``extract_fine_v2.
build_tasks``), so every scan, archive file and slab location is identical to
the already-frozen validation extraction -- no new selection is made here.
For every scan it re-fetches the same slab (HTTP 206 range reads only, same
chunk-aware fetch path), slices the same 42 frames, and as a safety check
re-normalizes them and asserts the result is bit-identical to the already
saved frozen normalized frames. Resume-safe, atomic writes, max 4 workers,
identical to the production extractor's transfer discipline.

Output: ``<out-root>/<scan_id>.npy`` -- (42, 16, 4096) float32 raw frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the bitshuffle filter
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_fine_v2 import (
    atomic_save_npy,
    atomic_write_json,
    build_tasks,
    md5,
    sha256,
    validate_split_access,
)

from deepseti.data.fine_remote import (
    FRAME_CHANNELS,
    FileUnavailable,
    HTTPRangeFile,
    RangeError,
    check_fine_geometry,
    fetch_slab,
    read_geometry,
    slab_chunk_spans,
    slab_window_offsets,
    slice_frames,
)
from deepseti.preprocessing.fine import normalize_fine_frames

DEFAULT_CONFIG = "configs/data/fine_v2_validation_access.yaml"
DEFAULT_OUT = "data/processed/fine_v2/validation_raw"
DEFAULT_NORMALIZED = "data/processed/fine_v2/validation/frames"
PER_FILE_BUDGET = 512 * 1024**2
MAX_WORKERS = 4


def is_cached(out_dir: Path, scan_id: str) -> bool:
    side, npy = out_dir / f"{scan_id}.json", out_dir / f"{scan_id}.npy"
    if not (side.exists() and npy.exists()):
        return False
    try:
        rec = json.loads(side.read_text())
    except json.JSONDecodeError:
        return False
    return rec.get("status") == "complete" and rec.get("sha256") == sha256(npy)


def fetch_raw_scan(task: dict, out_dir: str, normalized_dir: str, file_attempts: int) -> dict:
    """Re-fetch one scan's raw frames; never raises, returns a status record."""

    out_dir, normalized_dir = Path(out_dir), Path(normalized_dir)
    sid = task["scan_id"]
    rec = {"scan_id": sid, "status": "failed", "requests": 0, "bytes": 0,
           "http_statuses": {}, "full_body_responses": 0, "error": None}
    for attempt in range(1, file_attempts + 1):
        rf = HTTPRangeFile(task["url"], budget=PER_FILE_BUDGET)
        err = None
        try:
            with h5py.File(rf, "r") as h5:
                ds = h5["data"]
                g = read_geometry(ds)
                problems = check_fine_geometry(g)
                if problems:
                    raise RuntimeError(f"geometry: {problems}")
                start = task["slab_start_channel"]
                spans = slab_chunk_spans(ds, start, task["slab_channels"])
                slab = fetch_slab(ds, rf, start, task["slab_channels"], spans=spans)
            offsets = slab_window_offsets(task["frames_per_scan"], task["slab_channels"],
                                          FRAME_CHANNELS)
            raw = slice_frames(slab, offsets)
            del slab
            if raw.shape != (task["frames_per_scan"], 16, FRAME_CHANNELS):
                raise RuntimeError(f"raw shape {raw.shape}")
            frozen = np.load(normalized_dir / f"{sid}.npy")
            recomputed = normalize_fine_frames(raw)
            if not np.array_equal(recomputed, frozen):
                raise RuntimeError("re-fetched raw frames do not reproduce the frozen "
                                   "normalized frames bit-for-bit")
        except FileUnavailable as exc:
            err = repr(exc)
        except (RangeError, OSError, KeyError, ValueError, RuntimeError) as exc:
            err = repr(exc)
        rec["requests"] += rf.stats.requests
        rec["bytes"] += rf.stats.bytes
        rec["full_body_responses"] += rf.stats.full_body_responses
        for k, v in rf.stats.statuses.items():
            rec["http_statuses"][str(k)] = rec["http_statuses"].get(str(k), 0) + v
        if err is None:
            npy = out_dir / f"{sid}.npy"
            atomic_save_npy(npy, raw)
            rec.update({"status": "complete", "sha256": sha256(npy)})
            atomic_write_json(out_dir / f"{sid}.json", rec)
            return rec
        rec["error"] = err
        if attempt < file_attempts:
            time.sleep(10 * 2 ** (attempt - 1))
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out-root", default=DEFAULT_OUT)
    p.add_argument("--normalized-root", default=DEFAULT_NORMALIZED)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    refusal = validate_split_access("validation", cfg)
    if refusal:
        print(f"REFUSED: {refusal}")
        return 2
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"workers must be 1..{MAX_WORKERS}")
    for key in ("manifest", "chunk_plan"):
        if md5(cfg["inputs"][key]) != cfg["inputs"][f"{key}_md5"]:
            raise SystemExit(f"{key} checksum differs from frozen config; refusing to run")

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = Path(args.normalized_root)
    if not normalized_dir.exists():
        raise SystemExit(f"frozen normalized frames not found at {normalized_dir}")

    tasks = build_tasks(cfg, "validation")
    assert all(t["split"] == "validation" for t in tasks)
    done = {t["scan_id"] for t in tasks if is_cached(out_dir, t["scan_id"])}
    todo = [t for t in tasks if t["scan_id"] not in done]
    print(f"{len(done)} scans already cached, {len(todo)} to fetch, {args.workers} workers",
          flush=True)

    failures: dict[str, str] = {}
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch_raw_scan, t, str(out_dir), str(normalized_dir), 3)
                  for t in todo}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                rec = fut.result()
                if rec["status"] == "complete":
                    done.add(rec["scan_id"])
                    failures.pop(rec["scan_id"], None)
                    n_done += 1
                else:
                    failures[rec["scan_id"]] = rec["error"]
                    print(f"FAILED {rec['scan_id']}: {rec['error']}", flush=True)
                if n_done and n_done % 10 == 0:
                    el = time.time() - t0
                    eta = el / n_done * (len(todo) - n_done)
                    print(f"[progress] {len(done)}/{len(tasks)} scans cached | "
                          f"wall {el / 3600:.2f} h | ETA {eta / 3600:.2f} h", flush=True)

    ok = len(done) == len(tasks) and not failures
    print(json.dumps({
        "scans_cached": len(done), "scans_expected": len(tasks),
        "failures": failures, "all_pass": ok,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
