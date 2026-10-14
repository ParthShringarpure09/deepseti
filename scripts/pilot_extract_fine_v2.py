"""TRAINING-ONLY Fine extraction pilot for Dataset V2.

Selection uses manifest order/metadata only:
  * gpuspec cadence: the first TRAIN row (manifest order) whose
    reduction_pipeline is gpuspec -> its whole inferred cadence.
  * rawspec smoke test: the first TRAIN row (manifest order) whose
    reduction_pipeline is rawspec -> that single node file.

Windows: fixed 4096-channel grid (start = k*4096), eligible when fully inside the
predeclared S-band window and clear of the predeclared 2300-2360 MHz mask;
16 picked by stratified seeded draw (``select_windows``). The same channel
windows are used for all six cadence positions.

Before extracting the full pilot, one deterministic frame (the first selected
window of cadence position 0) is read to measure transfer cost. If the measured
cost projected to 256 windows x 6 positions x 90 cadences exceeds the
thresholds, the gpuspec pilot stops there and reports the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the bitshuffle filter
import numpy as np
import pandas as pd

from deepseti.data.fine_remote import (
    FRAME_CHANNELS,
    MASK_MHZ,
    SBAND_MHZ,
    HTTPRangeFile,
    TransferStats,
    check_fine_geometry,
    eligible_window_starts,
    extract_windows,
    read_geometry,
    select_windows,
)
from deepseti.preprocessing.fine import normalize_fine_frames

SEED = 20261013
N_PILOT_WINDOWS = 16
N_RAWSPEC_SMOKE_WINDOWS = 4
PLANNED_WINDOWS_PER_CADENCE = 256
N_CADENCES = 90
PER_FILE_BUDGET = 2 * 1024**3
REPORT_PATH = "results/data_audit/fine_v2_extraction_pilot.json"


def frame_stats(raw: np.ndarray, norm: np.ndarray) -> dict:
    mad = np.median(np.abs(norm - np.median(norm, axis=(1, 2), keepdims=True)), axis=(1, 2))
    return {
        "raw_min": float(raw.min()),
        "raw_median": float(np.median(raw)),
        "raw_max": float(raw.max()),
        "norm_per_frame_median_absmax": float(np.abs(np.median(norm, axis=(1, 2))).max()),
        "norm_per_frame_mad_min": float(mad.min()),
        "norm_per_frame_mad_max": float(mad.max()),
        "norm_all_finite": bool(np.isfinite(norm).all()),
    }


def chunk_columns(starts: np.ndarray, chunk_channels: int) -> int:
    return len(set((starts // chunk_channels).tolist()))


def open_file(url: str):
    rf = HTTPRangeFile(url, budget=PER_FILE_BUDGET)
    h5 = h5py.File(rf, "r")
    return rf, h5


def run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    train = manifest[manifest["split"] == "train"]
    report: dict = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": args.manifest,
        "seed": SEED,
        "scope": "TRAIN split only; validation/test rows never read or requested",
        "window_algorithm": {
            "grid": f"start channel = k*{FRAME_CHANNELS} in file channel order",
            "eligible": f"window outer edges within {SBAND_MHZ} MHz and not overlapping {MASK_MHZ} MHz",
            "selection": "eligible grid cut into n contiguous near-equal-count strata "
                         "(np.linspace(0, len, n+1).round()); one draw per stratum with "
                         "random.Random(f'{seed}|{key}').randrange",
            "gpuspec_key": "{inferred_cadence_id}|{setup_key}|n={n}",
            "rawspec_key": "{inferred_cadence_id}|{setup_key}|{filename}|rawspec_smoke|n={n}",
            "frequency_order": "file channel order (descending frequency when foff < 0)",
            "content_used": False,
        },
        "normalization": "deepseti.preprocessing.frames.robust_normalize_frame: log10, "
                         "(x - median) / MAD per frame, no clipping, float32",
        "thresholds": {
            "max_projected_bytes": args.max_projected_bytes,
            "max_projected_days": args.max_projected_days,
        },
    }
    total = TransferStats()
    t_start = time.time()

    # ---------------- gpuspec cadence ----------------
    first = train[train["reduction_pipeline"] == "gpuspec"].iloc[0]
    cad = train[train["inferred_cadence_id"] == first["inferred_cadence_id"]].sort_values(
        "cadence_position"
    )
    assert (cad["split"] == "train").all() and len(cad) == 6
    gp: dict = {
        "group_id": first["group_id"],
        "inferred_cadence_id": first["inferred_cadence_id"],
        "session": first["inferred_session_id"],
        "setup_key": first["setup_key"],
        "files": [],
    }
    print(f"gpuspec cadence {gp['group_id']} {gp['inferred_cadence_id']}", flush=True)
    geoms = []
    for _, row in cad.iterrows():
        rf, h5 = open_file(row["url"])
        with h5:
            g = read_geometry(h5["data"])
        total.add(rf.stats)
        problems = check_fine_geometry(g)
        geoms.append(g)
        gp["files"].append({
            "cadence_position": int(row["cadence_position"]),
            "cadence_role": row["cadence_role"],
            "filename": row["filename"],
            "size_bytes": int(row["size_bytes"]),
            "fch1": g.fch1, "foff": g.foff, "nchans": g.nchans, "n_ints": g.n_ints,
            "tsamp": g.tsamp, "chunks": list(g.chunks), "filters": g.compression,
            "geometry_problems": problems,
            "metadata_requests": rf.stats.requests, "metadata_bytes": rf.stats.bytes,
        })
        print(f"  pos {row['cadence_position']} {g.n_ints}x{g.nchans} chunks={g.chunks} "
              f"problems={problems}", flush=True)
    g0 = geoms[0]
    same = all((g.fch1, g.foff, g.nchans) == (g0.fch1, g0.foff, g0.nchans) for g in geoms)
    gp["all_positions_compatible"] = same and not any(f["geometry_problems"] for f in gp["files"])
    lo, hi = sorted((g0.fch1, g0.channel_freq_mhz(g0.nchans - 1)))
    eligible = eligible_window_starts(g0)
    starts = select_windows(eligible, N_PILOT_WINDOWS, SEED,
                            f"{gp['inferred_cadence_id']}|{gp['setup_key']}|n={N_PILOT_WINDOWS}")
    gp["frequency_coverage_mhz"] = [lo, hi]
    gp["eligible_windows"] = len(eligible)
    gp["usable_span_mhz"] = [g0.window_freq_bounds(int(eligible.max()))[0],
                             g0.window_freq_bounds(int(eligible.min()))[1]]
    gp["window_starts"] = starts.tolist()
    gp["window_bounds_mhz"] = [g0.window_freq_bounds(int(s)) for s in starts]
    if not gp["all_positions_compatible"]:
        gp["status"] = "FAILED_GEOMETRY"
        return finish(report, gp, None, total, t_start, out_dir, 2)

    # Cost probe: first selected window, cadence position 0, all 16 integrations.
    rf, h5 = open_file(cad.iloc[0]["url"])
    t0 = time.time()
    with h5:
        raw = extract_windows(h5["data"], starts[:1])
    probe_s = time.time() - t0
    probe_bytes = rf.stats.bytes - gp["files"][0]["metadata_bytes"]
    total.add(rf.stats)
    norm = normalize_fine_frames(raw)
    np.save(out_dir / "gpuspec_probe_raw.npy", raw)
    np.save(out_dir / "gpuspec_probe_norm.npy", norm)
    chunk_ch = g0.chunks[-1]
    plan = select_windows(eligible, PLANNED_WINDOWS_PER_CADENCE, SEED,
                          f"{gp['inferred_cadence_id']}|{gp['setup_key']}|n={PLANNED_WINDOWS_PER_CADENCE}")
    per_chunk_bytes = probe_bytes / g0.n_ints
    cols_256 = chunk_columns(plan, chunk_ch)
    proj_cad_bytes = cols_256 * g0.n_ints * per_chunk_bytes * 6
    proj_bytes = proj_cad_bytes * N_CADENCES
    rate = probe_bytes / probe_s
    proj_days = proj_bytes / rate / 86400
    pilot_cols = chunk_columns(starts, chunk_ch)
    gp["cost_probe"] = {
        "window_start": int(starts[0]),
        "frame_shape": list(raw.shape[1:]),
        "requests": rf.stats.requests - gp["files"][0]["metadata_requests"],
        "bytes": int(probe_bytes),
        "seconds": round(probe_s, 2),
        "bytes_per_second": round(rate),
        "bytes_per_frame": int(probe_bytes),
        "raw_bytes_per_frame": int(raw[0].nbytes),
        "read_amplification": round(probe_bytes / raw[0].nbytes, 1),
        "bytes_per_compressed_chunk_mean": round(per_chunk_bytes),
        "stats": frame_stats(raw, norm),
    }
    gp["projection"] = {
        "assumption": "256 windows per cadence, same 256 windows in each of 6 positions, "
                      "90 cadences, measured gpuspec bytes/chunk and single-stream rate; "
                      "chunk sharing between windows counted exactly from the 256-window selection",
        "chunk_columns_touched_per_file_256": cols_256,
        "frames_total": PLANNED_WINDOWS_PER_CADENCE * 6 * N_CADENCES,
        "bytes_per_cadence": round(proj_cad_bytes),
        "bytes_total": round(proj_bytes),
        "days_total_single_stream": round(proj_days, 1),
        "processed_storage_bytes_float32_raw": PLANNED_WINDOWS_PER_CADENCE * 6 * N_CADENCES * 16 * 4096 * 4,
        "pilot_projection": {
            "frames": N_PILOT_WINDOWS * 6,
            "chunk_columns_per_file": pilot_cols,
            "bytes": round(pilot_cols * g0.n_ints * per_chunk_bytes * 6),
            "hours_single_stream": round(pilot_cols * g0.n_ints * per_chunk_bytes * 6 / rate / 3600, 2),
        },
    }
    print(json.dumps(gp["cost_probe"], indent=1), json.dumps(gp["projection"], indent=1), flush=True)

    if proj_bytes > args.max_projected_bytes or proj_days > args.max_projected_days:
        gp["status"] = "STOPPED_COST_GUARD"
        gp["stop_reason"] = (
            f"projected {proj_bytes / 1e12:.2f} TB / {proj_days:.0f} days for 256 windows x 90 "
            f"cadences exceeds thresholds; remaining {N_PILOT_WINDOWS * 6 - 1} pilot frames not read"
        )
    else:
        frames_raw = np.empty((6, N_PILOT_WINDOWS, 16, FRAME_CHANNELS), np.float32)
        for i, (_, row) in enumerate(cad.iterrows()):
            rf, h5 = open_file(row["url"])
            with h5:
                frames_raw[i] = extract_windows(h5["data"], starts)
            total.add(rf.stats)
        frames_norm = normalize_fine_frames(frames_raw.reshape(-1, 16, FRAME_CHANNELS)).reshape(
            frames_raw.shape)
        np.save(out_dir / "gpuspec_frames_raw.npy", frames_raw)
        np.save(out_dir / "gpuspec_frames_norm.npy", frames_norm)
        gp["stats"] = frame_stats(frames_raw.reshape(-1, 16, FRAME_CHANNELS),
                                  frames_norm.reshape(-1, 16, FRAME_CHANNELS))
        gp["status"] = "PASSED"

    # ---------------- rawspec smoke test ----------------
    rrow = train[train["reduction_pipeline"] == "rawspec"].iloc[0]
    rs: dict = {
        "group_id": rrow["group_id"],
        "inferred_cadence_id": rrow["inferred_cadence_id"],
        "cadence_position": int(rrow["cadence_position"]),
        "filename": rrow["filename"],
        "size_bytes": int(rrow["size_bytes"]),
    }
    print(f"rawspec smoke {rs['filename']}", flush=True)
    rf, h5 = open_file(rrow["url"])
    with h5:
        ds = h5["data"]
        g = read_geometry(ds)
        meta_req, meta_bytes = rf.stats.requests, rf.stats.bytes
        el = eligible_window_starts(g)
        key = (f"{rs['inferred_cadence_id']}|{rrow['setup_key']}|{rs['filename']}"
               f"|rawspec_smoke|n={N_RAWSPEC_SMOKE_WINDOWS}")
        rstarts = select_windows(el, N_RAWSPEC_SMOKE_WINDOWS, SEED, key)
        t0 = time.time()
        rraw = extract_windows(ds, rstarts)
        rsec = time.time() - t0
    total.add(rf.stats)
    rnorm = normalize_fine_frames(rraw)
    np.save(out_dir / "rawspec_smoke_raw.npy", rraw)
    np.save(out_dir / "rawspec_smoke_norm.npy", rnorm)
    rbytes = rf.stats.bytes - meta_bytes
    rs.update({
        "fch1": g.fch1, "foff": g.foff, "nchans": g.nchans, "n_ints": g.n_ints, "tsamp": g.tsamp,
        "chunks": list(g.chunks), "filters": g.compression,
        "geometry_problems": check_fine_geometry(g),
        "frequency_coverage_mhz": sorted((g.fch1, float(g.channel_freq_mhz(g.nchans - 1)))),
        "eligible_windows": len(el),
        "window_starts": rstarts.tolist(),
        "window_bounds_mhz": [g.window_freq_bounds(int(s)) for s in rstarts],
        "frame_shapes": [list(f.shape) for f in rraw],
        "metadata_requests": meta_req, "metadata_bytes": meta_bytes,
        "extract_requests": rf.stats.requests - meta_req,
        "extract_bytes": int(rbytes),
        "extract_seconds": round(rsec, 2),
        "bytes_per_frame": round(rbytes / len(rstarts)),
        "stats": frame_stats(rraw, rnorm),
    })
    rs["status"] = "PASSED" if (
        not rs["geometry_problems"] and rraw.shape == (N_RAWSPEC_SMOKE_WINDOWS, 16, FRAME_CHANNELS)
        and rs["stats"]["norm_all_finite"]
    ) else "FAILED"
    return finish(report, gp, rs, total, t_start, out_dir, 0)


def finish(report, gp, rs, total, t_start, out_dir, code) -> int:
    report["gpuspec"] = gp
    report["rawspec_smoke"] = rs
    report["transfer"] = {
        "requests": total.requests,
        "bytes": total.bytes,
        "http_statuses": {str(k): v for k, v in total.statuses.items()},
        "all_206": set(total.statuses) == {206},
        "full_body_responses": total.full_body_responses,
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    report["processed_disk_bytes"] = sum(p.stat().st_size for p in out_dir.glob("*.npy"))
    for path in (out_dir / "pilot_report.json", Path(REPORT_PATH)):
        path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report["transfer"], indent=1))
    return code



def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/data/fine_v2_manifest.csv")
    p.add_argument("--out-dir", default="data/processed/fine_v2_pilot")
    p.add_argument("--max-projected-bytes", type=float, default=1e12)
    p.add_argument("--max-projected-days", type=float, default=7.0)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
