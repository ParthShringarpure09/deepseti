"""Parallel chunk-aware extraction pilot (pilot 3), TRAINING ONLY.

Exactly 4 worker processes, one HTTP stream each, read the planned
1,048,576-channel slab of G001 cadence positions 0-3 (manifest order) using
the revision-2 chunk plan, slice the 42 frames locally and normalize them.
Every check of ``run_scan`` (206 only, no full-body response, 0 slicing bytes,
42 unique finite (16, 4096) frames) applies per worker.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from deepseti.data.fine_remote import RangeError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_chunked_fine_v2 as chunked

N_WORKERS = 4
GROUP = "G001"
REPORT_PATH = "results/data_audit/fine_v2_parallel_pilot.json"
PILOT2_REPORT = "results/data_audit/fine_v2_chunked_pilot.json"


def maxrss_bytes(who) -> int:
    rss = resource.getrusage(who).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def worker(row: pd.Series, plan_row: pd.Series, out_dir: str) -> dict:
    t0 = time.time()
    start = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        rec = chunked.run_scan(row, plan_row, Path(out_dir),
                               f"G001_pos{int(row['cadence_position'])}")
        rec["error"] = None
    except (RangeError, OSError, ValueError) as exc:  # recorded; the worker counts as failed
        rec = {"filename": row["filename"], "passed": False, "error": repr(exc)}
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rec.update({
        "worker_started_utc": start,
        "worker_seconds": round(time.time() - t0, 2),
        "worker_cpu_seconds": round(ru.ru_utime + ru.ru_stime, 2),
        "worker_peak_rss_bytes": maxrss_bytes(resource.RUSAGE_SELF),
    })
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/data/fine_v2_manifest.csv")
    p.add_argument("--plan", default=chunked.PLAN_PATH)
    p.add_argument("--out-dir", default="data/processed/fine_v2_parallel_pilot")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    plan = pd.read_csv(args.plan)
    plan_row = plan[plan["group_id"] == GROUP].iloc[0]
    assert plan_row["split"] == "train" and plan_row["reduction_pipeline"] == "gpuspec"
    rows = manifest[(manifest["split"] == "train") & (manifest["group_id"] == GROUP)]
    rows = rows.sort_values("cadence_position").iloc[:N_WORKERS]
    assert (rows["split"] == "train").all() and len(rows) == N_WORKERS

    pilot2 = json.loads(Path(PILOT2_REPORT).read_text())["gpuspec_scan"]
    serial_file_s = pilot2["wall_seconds"]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(worker, r, plan_row, args.out_dir) for _, r in rows.iterrows()]
        recs = [f.result() for f in futures]
    wall = time.time() - t0

    ok = [r for r in recs if r.get("error") is None]
    total_bytes = sum(r["total_bytes"] for r in ok)
    statuses: dict[str, int] = {}
    for r in ok:
        for k, v in r["http_statuses"].items():
            statuses[k] = statuses.get(k, 0) + v
    frames = sum(r["n_frames"] for r in ok)
    report = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "TRAIN only: G001 positions 0-3; no validation/test access",
        "plan": args.plan,
        "plan_slab": {k: plan_row[k] for k in ("stratum", "stratum_slot", "slot_q",
                                                "slab_start_channel", "slab_lo_mhz", "slab_hi_mhz")},
        "workers": N_WORKERS,
        "serial_reference": {
            "source": PILOT2_REPORT,
            "seconds_per_file": serial_file_s,
            "seconds_for_4_files": round(4 * serial_file_s, 1),
            "bytes_per_second_slab": pilot2["slab_bytes"] / pilot2["slab_fetch_seconds"],
        },
        "parallel_wall_seconds": round(wall, 1),
        "speedup_vs_serial": round(4 * serial_file_s / wall, 2),
        "total_bytes": total_bytes,
        "aggregate_mb_per_s": round(total_bytes / wall / 1e6, 3),
        "frames": frames,
        "bytes_per_retained_frame": round(total_bytes / frames) if frames else None,
        "http_statuses": statuses,
        "full_body_responses": sum(r.get("full_body_responses", 0) for r in ok),
        "failures": len(recs) - len(ok) + sum(not r["passed"] for r in ok),
        "retries": sum(r.get("retries", 0) for r in ok),
        "extra_http_bytes_from_slicing": sum(r["extra_http_bytes_from_slicing"] for r in ok),
        "peak_rss_bytes_max_worker": max(r["worker_peak_rss_bytes"] for r in recs),
        "parent_peak_rss_bytes": maxrss_bytes(resource.RUSAGE_SELF),
        "worker_records": [{k: v for k, v in r.items() if k != "window_offsets_in_slab"}
                           for r in recs],
        "processed_disk_bytes": sum(f.stat().st_size for f in Path(args.out_dir).glob("*.npy")),
    }
    for path in (Path(args.out_dir) / "pilot_report.json", Path(REPORT_PATH)):
        path.write_text(json.dumps(report, indent=2, default=chunked.to_json) + "\n")
    summary = {k: v for k, v in report.items() if k != "worker_records"}
    print(json.dumps(summary, indent=1, default=chunked.to_json))
    return 0 if report["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
