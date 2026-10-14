"""Production Fine Dataset V2 extraction (protocol revision 2).

Only ``--split train`` is allowed under revision 2; validation and test are
refused. For every scan (cadence position) of every planned cadence the
chunk-aware slab is fetched with HTTP Range requests (206 only), 42 frames are
sliced locally, normalized with the frozen log10 + median/MAD rule and stored.

Layout under ``<out-root>/<split>/``:
  frames/<scan_id>.npy        (42, 16, 4096) float32 normalized frames
  frames/<scan_id>.json       sidecar: provenance, sha256, checks, transfer stats
  audit_raw/<scan_id>_f00.npy raw frame 0 of cadence position 0 (one per cadence)
  extraction_state.json       progress, failures, retries, run sessions
  metadata.csv                one row per frame (written once all scans are done)

Resume: a scan is complete when its .npy and sidecar exist and the sidecar's
sha256 matches the .npy; complete scans are never re-downloaded. All files are
written to a temporary name and renamed atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the bitshuffle filter
import numpy as np
import pandas as pd
import yaml

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

DEFAULT_CONFIG = "configs/data/fine_v2_extraction.yaml"
SUMMARY_PATH = "results/data_audit/fine_v2_{split}_extraction_summary.json"
PER_FILE_BUDGET = 512 * 1024**2
MAX_WORKERS = 4


class PermanentFailure(RuntimeError):
    """A scan failure that retrying cannot fix (geometry, full body, 404, data)."""


def md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, default=to_json) + "\n").encode())


def to_json(obj):
    return obj.item() if isinstance(obj, np.generic) else str(obj)


def scan_id(row) -> str:
    return f"{row['group_id']}_pos{int(row['cadence_position'])}_{int(row['archive_id'])}"


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def _attempt(task: dict, frames_dir: Path, audit_dir: Path) -> dict:
    rf = HTTPRangeFile(task["url"], budget=PER_FILE_BUDGET)
    task["_rf"] = rf
    t0 = time.time()
    with h5py.File(rf, "r") as h5:
        ds = h5["data"]
        g = read_geometry(ds)
        problems = check_fine_geometry(g)
        if problems:
            raise PermanentFailure(f"geometry: {problems}")
        if g.nchans != task["hdr_nchans"] or abs(g.fch1 - task["hdr_fch1"]) > abs(g.foff) / 2:
            raise PermanentFailure("header differs from frozen manifest")
        start = task["slab_start_channel"]
        lo, hi = g.window_freq_bounds(start, task["slab_channels"])
        if abs(lo - task["slab_lo_mhz"]) > abs(g.foff) or abs(hi - task["slab_hi_mhz"]) > abs(g.foff):
            raise PermanentFailure(f"slab bounds {lo}-{hi} differ from plan")
        spans = slab_chunk_spans(ds, start, task["slab_channels"])
        slab = fetch_slab(ds, rf, start, task["slab_channels"], spans=spans)
    bytes_before = rf.stats.bytes
    offsets = slab_window_offsets(task["frames_per_scan"], task["slab_channels"], FRAME_CHANNELS)
    raw = slice_frames(slab, offsets)
    del slab
    if rf.stats.bytes != bytes_before:
        raise PermanentFailure("local slicing caused HTTP transfer")
    if len({hashlib.sha1(f.tobytes()).hexdigest() for f in raw}) != len(raw):
        raise PermanentFailure("duplicate raw frames in slab")
    try:
        norm = normalize_fine_frames(raw)
    except ValueError as exc:
        raise PermanentFailure(f"normalization: {exc}") from exc
    if norm.shape != (task["frames_per_scan"], 16, FRAME_CHANNELS) or not np.isfinite(norm).all():
        raise PermanentFailure("normalized frames failed shape/finite check")

    npy = frames_dir / f"{task['scan_id']}.npy"
    atomic_save_npy(npy, norm)
    audit_file = None
    if task["cadence_position"] == 0:
        audit_file = audit_dir / f"{task['scan_id']}_f00.npy"
        atomic_save_npy(audit_file, raw[0])
    med = np.median(norm, axis=(1, 2))
    mad = np.median(np.abs(norm - med[:, None, None]), axis=(1, 2))
    return {
        "geometry": {"fch1": g.fch1, "foff": g.foff, "nchans": g.nchans, "n_ints": g.n_ints,
                     "tsamp": g.tsamp, "chunks": list(g.chunks)},
        "slab_freq_mhz": [lo, hi],
        "frame_offsets_in_slab": offsets.tolist(),
        "npy": npy.name,
        "npy_sha256": sha256(npy),
        "audit_raw_file": audit_file.name if audit_file else None,
        "n_frames": len(norm),
        "frame_shape": list(norm.shape[1:]),
        "norm_median_absmax": float(np.abs(med).max()),
        "norm_mad_min": float(mad.min()),
        "norm_mad_max": float(mad.max()),
        "norm_min": float(norm.min()),
        "norm_max": float(norm.max()),
        "raw_min": float(raw.min()),
        "raw_median": float(np.median(raw)),
        "raw_max": float(raw.max()),
        "seconds": round(time.time() - t0, 2),
    }


def extract_scan(task: dict, frames_dir: str, audit_dir: str, file_attempts: int) -> dict:
    """Extract one scan with bounded retries. Never raises; returns a record."""

    frames_dir, audit_dir = Path(frames_dir), Path(audit_dir)
    rec = {k: v for k, v in task.items() if not k.startswith("_")}
    rec.update({"status": "failed", "attempts": [], "requests": 0, "bytes": 0,
                "http_statuses": {}, "full_body_responses": 0, "request_retries": 0})
    for attempt in range(1, file_attempts + 1):
        task.pop("_rf", None)
        err, permanent, result = None, False, None
        try:
            result = _attempt(task, frames_dir, audit_dir)
        except PermanentFailure as exc:
            err, permanent = repr(exc), True
        except FileUnavailable as exc:
            err, permanent = repr(exc), True
        except (RangeError, OSError, KeyError, ValueError) as exc:
            err = repr(exc)
        rf = task.get("_rf")
        if rf is not None:
            rec["requests"] += rf.stats.requests
            rec["bytes"] += rf.stats.bytes
            rec["full_body_responses"] += rf.stats.full_body_responses
            rec["request_retries"] += rf.stats.retries
            for k, v in rf.stats.statuses.items():
                rec["http_statuses"][str(k)] = rec["http_statuses"].get(str(k), 0) + v
            # h5py may wrap a FullBodyResponse in OSError: detect it from the stats.
            if rf.stats.full_body_responses:
                err, permanent = f"full-body HTTP response: {err}", True
        if err is None:
            rec.update(result)
            rec["status"] = "complete"
            rec["attempts"].append({"attempt": attempt, "ok": True})
            atomic_write_json(frames_dir / f"{task['scan_id']}.json", rec)
            return rec
        rec["attempts"].append({"attempt": attempt, "ok": False, "error": err,
                                "permanent": permanent})
        if permanent:
            break
        if attempt < file_attempts:
            time.sleep(30 * 2 ** (attempt - 1))
    return rec


# ---------------------------------------------------------------------------
# planning, state, verification
# ---------------------------------------------------------------------------


def build_tasks(cfg: dict, split: str) -> list[dict]:
    manifest = pd.read_csv(cfg["inputs"]["manifest"])
    plan = pd.read_csv(cfg["inputs"]["chunk_plan"])
    plan = plan[plan["split"] == split].sort_values("group_id")
    s = cfg["sampling"]
    tasks = []
    for _, p in plan.iterrows():
        rows = manifest[(manifest["split"] == split)
                        & (manifest["group_id"] == p["group_id"])
                        & (manifest["inferred_cadence_id"] == p["inferred_cadence_id"])
                        & (manifest["setup_key"].round(3) == round(p["slab_setup_key"], 3))]
        rows = rows.sort_values("cadence_position")
        if list(rows["cadence_position"]) != list(range(s["scans_per_cadence"])):
            raise SystemExit(f"{p['group_id']}: planned setup does not give 6 positions")
        for _, r in rows.iterrows():
            tasks.append({
                "scan_id": scan_id(r),
                "split": r["split"], "group_id": r["group_id"],
                "cadence_id": r["inferred_cadence_id"], "session": r["inferred_session_id"],
                "target": r["target"], "on_target": rows.iloc[0]["target"],
                "cadence_position": int(r["cadence_position"]), "cadence_role": r["cadence_role"],
                "archive_id": int(r["archive_id"]), "filename": r["filename"], "url": r["url"],
                "archive_md5sum": r["md5sum"], "reduction_pipeline": r["reduction_pipeline"],
                "setup_key": float(r["setup_key"]),
                "hdr_fch1": float(r["hdr_fch1"]), "hdr_nchans": int(r["hdr_nchans"]),
                "stratum": int(p["stratum"]), "stratum_slot": int(p["stratum_slot"]),
                "slot_q": float(p["slot_q"]),
                "slab_start_channel": int(p["slab_start_channel"]),
                "slab_lo_mhz": float(p["slab_lo_mhz"]), "slab_hi_mhz": float(p["slab_hi_mhz"]),
                "slab_channels": int(s["slab_channels"]),
                "frames_per_scan": int(s["frames_per_scan"]),
            })
    return tasks


def is_complete(frames_dir: Path, sid: str) -> dict | None:
    side, npy = frames_dir / f"{sid}.json", frames_dir / f"{sid}.npy"
    if not (side.exists() and npy.exists()):
        return None
    try:
        rec = json.loads(side.read_text())
    except json.JSONDecodeError:
        return None
    if rec.get("status") != "complete" or rec.get("npy_sha256") != sha256(npy):
        return None
    return rec


def frame_metadata(recs: list[dict], frame_shape) -> pd.DataFrame:
    rows = []
    for r in recs:
        g = r["geometry"]
        for i, off in enumerate(r["frame_offsets_in_slab"]):
            ch = r["slab_start_channel"] + off
            a = g["fch1"] + ch * g["foff"]
            b = g["fch1"] + (ch + frame_shape[1] - 1) * g["foff"]
            half = abs(g["foff"]) / 2
            rows.append({
                "frame_id": f"{r['scan_id']}_f{i:02d}",
                "npy": f"frames/{r['npy']}", "npy_index": i,
                "split": r["split"], "group_id": r["group_id"], "cadence_id": r["cadence_id"],
                "session": r["session"], "target": r["target"], "on_target": r["on_target"],
                "cadence_position": r["cadence_position"], "cadence_role": r["cadence_role"],
                "archive_id": r["archive_id"], "filename": r["filename"], "url": r["url"],
                "archive_md5sum": r["archive_md5sum"],
                "reduction_pipeline": r["reduction_pipeline"],
                "stratum": r["stratum"], "stratum_slot": r["stratum_slot"],
                "slab_start_channel": r["slab_start_channel"],
                "slab_lo_mhz": r["slab_freq_mhz"][0], "slab_hi_mhz": r["slab_freq_mhz"][1],
                "frame_index_in_scan": i, "frame_offset_in_slab": off,
                "abs_channel_start": ch,
                "frame_lo_mhz": min(a, b) - half, "frame_hi_mhz": max(a, b) + half,
                "fch1": g["fch1"], "foff": g["foff"],
            })
    return pd.DataFrame(rows)


def verify(cfg: dict, split: str, out: Path, tasks: list[dict], state: dict) -> dict:
    s = cfg["sampling"]
    frames_dir = out / "frames"
    recs = [is_complete(frames_dir, t["scan_id"]) for t in tasks]
    done = [r for r in recs if r is not None]
    meta = frame_metadata(done, cfg["geometry"]["frame_shape"]) if done else pd.DataFrame()
    shapes_ok, finite_ok = True, True
    med_absmax, mad_min, mad_max, vmin, vmax = 0.0, np.inf, -np.inf, np.inf, -np.inf
    for r in done:
        arr = np.load(frames_dir / r["npy"])
        shapes_ok &= arr.shape == (s["frames_per_scan"], *cfg["geometry"]["frame_shape"])
        finite_ok &= bool(np.isfinite(arr).all())
        med = np.median(arr, axis=(1, 2))
        mad = np.median(np.abs(arr - med[:, None, None]), axis=(1, 2))
        med_absmax = max(med_absmax, float(np.abs(med).max()))
        mad_min, mad_max = min(mad_min, float(mad.min())), max(mad_max, float(mad.max()))
        vmin, vmax = min(vmin, float(arr.min())), max(vmax, float(arr.max()))
    n_groups = cfg["split"]["cadence_groups"][split]
    per_scan = meta.groupby(["group_id", "cadence_position"]).size() if done else pd.Series()
    per_cad = meta.groupby("group_id").size() if done else pd.Series()
    statuses: dict[str, int] = {}
    for r in done:
        for k, v in r["http_statuses"].items():
            statuses[k] = statuses.get(k, 0) + v
    manifest = pd.read_csv(cfg["inputs"]["manifest"])
    src = manifest[manifest["archive_id"].isin(meta["archive_id"])] if done else manifest.iloc[:0]
    inv = {
        "cadence_groups_eq_expected": int(meta["group_id"].nunique()) == n_groups if done else False,
        "scans_eq_expected": len(done) == n_groups * s["scans_per_cadence"],
        "frames_eq_expected": len(meta) == n_groups * s["frames_per_cadence"],
        "every_frame_shape_16x4096": bool(shapes_ok),
        "all_normalized_values_finite": bool(finite_ok),
        "frames_per_scan_all_42": bool((per_scan == s["frames_per_scan"]).all()) if done else False,
        "frames_per_cadence_all_252": bool((per_cad == s["frames_per_cadence"]).all()) if done else False,
        "no_duplicate_frame_ids": bool(meta["frame_id"].is_unique) if done else False,
        "all_source_rows_in_split": bool((src["split"] == split).all() and len(src) == len(done)),
        "validation_rows": int((src["split"] == "validation").sum()),
        "test_rows": int((src["split"] == "test").sum()),
        "manifest_md5_unchanged": md5(cfg["inputs"]["manifest"]) == cfg["inputs"]["manifest_md5"],
        "chunk_plan_md5_unchanged": md5(cfg["inputs"]["chunk_plan"]) == cfg["inputs"]["chunk_plan_md5"],
        "only_http_206": bool(statuses) and all(
            set(x.get("http_statuses", {})) <= {"206"} for x in [{"http_statuses": statuses},
                                                              *state["sessions"]]),
        "full_body_responses_zero": sum(r["full_body_responses"] for r in done) == 0
        and sum(x.get("full_body_responses", 0) for x in state["sessions"]) == 0,
    }
    inv["all_pass"] = all(v is True or (k in ("validation_rows", "test_rows") and v == 0)
                          for k, v in inv.items())
    return {
        "records": done, "metadata": meta, "invariants": inv, "http_statuses": statuses,
        "normalization": {"per_frame_median_absmax": med_absmax, "per_frame_mad_min": mad_min,
                          "per_frame_mad_max": mad_max, "value_min": vmin, "value_max": vmax},
    }


def disk_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out-root", default="data/processed/fine_v2")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.split not in cfg["split"]["extractable_splits"]:
        print(f"REFUSED: split '{args.split}' is not extractable under protocol revision "
              f"{cfg['protocol']['revision']} (allowed: {cfg['split']['extractable_splits']}).")
        return 2
    workers = args.workers or cfg["transfer"]["workers"]
    if not 1 <= workers <= MAX_WORKERS:
        raise SystemExit(f"workers must be 1..{MAX_WORKERS}")
    for key in ("manifest", "chunk_plan"):
        if md5(cfg["inputs"][key]) != cfg["inputs"][f"{key}_md5"]:
            raise SystemExit(f"{key} checksum differs from frozen config; refusing to run")

    out = Path(args.out_root) / args.split
    frames_dir, audit_dir = out / "frames", out / "audit_raw"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    state_path = out / "extraction_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "split": args.split, "protocol_revision": cfg["protocol"]["revision"],
        "sessions": [], "failures": {}, "retried_scans": {}}

    tasks = build_tasks(cfg, args.split)
    assert all(t["split"] == args.split for t in tasks)
    n_groups = cfg["split"]["cadence_groups"][args.split]
    by_group: dict[str, list[str]] = {}
    for t in tasks:
        by_group.setdefault(t["group_id"], []).append(t["scan_id"])
    if len(by_group) != n_groups:
        raise SystemExit(f"plan gives {len(by_group)} groups, expected {n_groups}")

    if not args.verify_only:
        done = {t["scan_id"] for t in tasks if is_complete(frames_dir, t["scan_id"])}
        todo = [t for t in tasks if t["scan_id"] not in done]
        session = {"started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                   "workers": workers, "scans_already_complete": len(done),
                   "scans_attempted": len(todo), "bytes": 0, "requests": 0,
                   "request_retries": 0, "scan_retries": 0, "failed": 0,
                   "http_statuses": {}, "full_body_responses": 0}
        state["sessions"].append(session)
        atomic_write_json(state_path, state)
        print(f"{len(done)} scans already complete, {len(todo)} to extract, {workers} workers",
              flush=True)
        t0 = time.time()
        scans_done_run = 0
        frames_total = n_groups * cfg["sampling"]["frames_per_cadence"]
        file_attempts = cfg["transfer"]["file_attempts"]

        def progress(prefix: str) -> None:
            n_scans = len(done)
            cads = sum(all(s in done for s in v) for v in by_group.values())
            el = time.time() - t0
            remaining = len(tasks) - n_scans - state_failed()
            eta = el / scans_done_run * remaining if scans_done_run else float("nan")
            print(f"{prefix} cadences {cads}/{n_groups} | frames "
                  f"{n_scans * cfg['sampling']['frames_per_scan']}/{frames_total} | "
                  f"bytes {session['bytes'] / 1e9:.2f} GB | wall {el / 3600:.2f} h | "
                  f"retries {session['request_retries']}+{session['scan_retries']} | "
                  f"failures {session['failed']} | ETA {eta / 3600:.2f} h", flush=True)

        def state_failed() -> int:
            return sum(1 for t in tasks if t["scan_id"] in state["failures"]
                       and t["scan_id"] not in done)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(extract_scan, t, str(frames_dir), str(audit_dir), file_attempts)
                       for t in todo}
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    rec = fut.result()
                    sid = rec["scan_id"]
                    session["bytes"] += rec["bytes"]
                    session["requests"] += rec["requests"]
                    session["request_retries"] += rec["request_retries"]
                    session["full_body_responses"] += rec["full_body_responses"]
                    for k, v in rec["http_statuses"].items():
                        session["http_statuses"][k] = session["http_statuses"].get(k, 0) + v
                    if len(rec["attempts"]) > 1:
                        session["scan_retries"] += len(rec["attempts"]) - 1
                        state["retried_scans"][sid] = rec["attempts"]
                    if rec["status"] == "complete":
                        done.add(sid)
                        state["failures"].pop(sid, None)
                        scans_done_run += 1
                    else:
                        session["failed"] += 1
                        state["failures"][sid] = rec["attempts"]
                        print(f"FAILED {sid}: {rec['attempts'][-1]}", flush=True)
                    session["elapsed_s"] = round(time.time() - t0, 1)
                    session["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
                    atomic_write_json(state_path, state)
                    if rec["status"] == "complete" and all(
                            s in done for s in by_group[rec["group_id"]]):
                        progress(f"[cadence {rec['group_id']} complete]")
                    elif scans_done_run and scans_done_run % 12 == 0:
                        progress("[progress]")
        session["ended_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        atomic_write_json(state_path, state)
        progress("[run end]")

    result = verify(cfg, args.split, out, tasks, state)
    meta = result["metadata"]
    if result["invariants"]["scans_eq_expected"]:
        tmp = out / ".metadata.csv.tmp"
        meta.to_csv(tmp, index=False)
        os.replace(tmp, out / "metadata.csv")
    recs = result["records"]
    total_bytes = sum(r["bytes"] for r in recs)
    wall = sum(s.get("elapsed_s", 0) for s in state["sessions"])
    summary = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "config": args.config,
        "protocol_revision": cfg["protocol"]["revision"],
        "manifest_md5": md5(cfg["inputs"]["manifest"]),
        "chunk_plan_md5": md5(cfg["inputs"]["chunk_plan"]),
        "output_dir": str(out),
        "completed_cadences": int(meta["group_id"].nunique()) if len(meta) else 0,
        "completed_scans": len(recs),
        "frames": len(meta),
        "disk_bytes": disk_bytes(out),
        "frames_dir_bytes": disk_bytes(frames_dir),
        "audit_raw_bytes": disk_bytes(audit_dir),
        "transferred_bytes_completed_scans": total_bytes,
        "transferred_bytes_all_sessions": sum(s["bytes"] for s in state["sessions"]),
        "requests_all_sessions": sum(s["requests"] for s in state["sessions"]),
        "extraction_wall_seconds_all_sessions": wall,
        "effective_throughput_mb_per_s": round(
            sum(s["bytes"] for s in state["sessions"]) / wall / 1e6, 3) if wall else None,
        "request_retries": sum(s["request_retries"] for s in state["sessions"]),
        "scan_retries": sum(s["scan_retries"] for s in state["sessions"]),
        "failures_outstanding": state["failures"],
        "retried_scans": state["retried_scans"],
        "http_statuses": result["http_statuses"],
        "full_body_responses": sum(r["full_body_responses"] for r in recs),
        "normalization": result["normalization"],
        "invariants": result["invariants"],
        "sessions": state["sessions"],
    }
    Path(SUMMARY_PATH.format(split=args.split)).write_text(
        json.dumps(summary, indent=2, default=to_json) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sessions", "retried_scans")},
                     indent=1, default=to_json))
    return 0 if result["invariants"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
