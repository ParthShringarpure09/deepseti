"""Chunk-aware Fine extraction pilot (pilot 2), TRAINING ONLY.

Rule (approved after pilot 1, before any Fine model results):
  * slab = 1,048,576 channels on the k*1,048,576 grid of a file
    (one gpuspec chunk; 16 rawspec chunks); eligible when fully inside the
    predeclared S-band window and not intersecting the 2300-2360 MHz mask.
  * per cadence: eligible slabs (all node setups of the cadence, metadata only)
    sorted by centre frequency, cut into 15 near-equal-count strata; the
    cadence's stratum k and slot j (of m groups in that stratum) come from a
    seeded balanced assignment within its split; the slab nearest the
    fractional position q = (j + 0.5) / m through the stratum (by frequency)
    is used for all six positions. (Revision 2: pilot 2 used q = 0.5 for all.)
  * per scan: 42 evenly spaced 4096-channel frames sliced locally from the slab.

The plan for all 90 cadences is computed from frozen manifest metadata only
(no HTTP to validation/test files). HTTP is used only for TRAIN files:
G001 position 0 (gpuspec) and one G116 rawspec node file (smoke test).
"""

from __future__ import annotations

import argparse
import hashlib
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
    FRAMES_PER_SCAN,
    MASK_MHZ,
    N_STRATA,
    SBAND_MHZ,
    SLAB_CHANNELS,
    FineGeometry,
    HTTPRangeFile,
    check_fine_geometry,
    eligible_slab_starts,
    fetch_slab,
    read_geometry,
    slab_chunk_spans,
    slab_window_offsets,
    slice_frames,
    slot_fraction,
    stratum_slab_choice,
    stratum_slots,
)
from deepseti.preprocessing.fine import normalize_fine_frames

SEED = 20261013
SPLITS = ("train", "validation", "test")
PER_FILE_BUDGET = 512 * 1024**2
FRAME_BYTES = 16 * FRAME_CHANNELS * 4
PLAN_PATH = "results/data_audit/fine_v2_chunk_plan.csv"
REPORT_PATH = "results/data_audit/fine_v2_chunked_pilot.json"
PLAN_SUMMARY_PATH = "results/data_audit/fine_v2_chunk_plan_summary.json"


def manifest_geometry(row) -> FineGeometry:
    return FineGeometry(
        fch1=float(row["hdr_fch1"]), foff=float(row["hdr_foff"]), nchans=int(row["hdr_nchans"]),
        n_ints=int(row["hdr_n_ints"]), tsamp=float(row["hdr_tsamp"]),
        chunks=tuple(int(x) for x in str(row["hdr_data_chunks"]).split("x")), compression=None,
    )


def build_plan(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        sm = manifest[manifest["split"] == split]
        slots = stratum_slots(sorted(sm["group_id"].unique()), SEED, split)
        for (gid, cid), cad in sm.groupby(["group_id", "inferred_cadence_id"], sort=True):
            cands = []
            for setup, srows in cad.groupby("setup_key", sort=True):
                srows = srows.sort_values("cadence_position")
                if sorted(srows["cadence_position"]) != list(range(6)):
                    raise ValueError(f"{cid} {setup}: incomplete positions")
                g = manifest_geometry(srows.iloc[0])
                for _, r in srows.iterrows():
                    gi = manifest_geometry(r)
                    if (gi.nchans, gi.foff, gi.chunks) != (g.nchans, g.foff, g.chunks) or \
                            abs(gi.fch1 - g.fch1) > abs(g.foff) / 2:
                        raise ValueError(f"{cid} {setup}: geometry differs across positions")
                if SLAB_CHANNELS % g.chunks[-1]:
                    raise ValueError(f"{cid}: chunk {g.chunks} does not tile the slab")
                for s in eligible_slab_starts(g):
                    lo, hi = g.window_freq_bounds(int(s), SLAB_CHANNELS)
                    per_pos = [manifest_geometry(r).window_freq_bounds(int(s), SLAB_CHANNELS)
                               for _, r in srows.iterrows()]
                    spread = max(abs(b[0] - lo) for b in per_pos)
                    cands.append(((lo + hi) / 2, setup, int(s), lo, hi, spread))
            cands.sort(key=lambda c: -c[0])
            centres = np.array([c[0] for c in cands])
            k, j, m = slots[gid]
            q = slot_fraction(j, m)
            _, setup, start, lo, hi, spread = cands[stratum_slab_choice(centres, k, q)]
            rows.append({
                "split": split, "group_id": gid, "inferred_cadence_id": cid,
                "reduction_pipeline": cad["reduction_pipeline"].iloc[0],
                "stratum": k, "stratum_slot": j, "stratum_groups": m, "slot_q": q,
                "n_eligible_slabs": len(cands),
                "slab_setup_key": setup, "slab_start_channel": start,
                "slab_lo_mhz": round(lo, 6), "slab_hi_mhz": round(hi, 6),
                "slab_lo_max_spread_across_positions_mhz": spread,
                "frames_per_scan": FRAMES_PER_SCAN, "frames_per_cadence": 6 * FRAMES_PER_SCAN,
            })
    return pd.DataFrame(rows)


def to_json(obj):
    return obj.item() if isinstance(obj, np.generic) else str(obj)


def summarize_plan(plan: pd.DataFrame, manifest_path: str) -> dict:
    """Metadata-only coverage checks of a chunk plan."""

    out: dict = {"plan_revision": 2, "manifest_md5": hashlib.md5(
        Path(manifest_path).read_bytes()).hexdigest()}
    for split in (*SPLITS, "all"):
        sp = plan if split == "all" else plan[plan["split"] == split]
        key = sp["slab_lo_mhz"].round(4)
        counts = key.value_counts()
        merged: list[list[float]] = []
        for lo, hi in sorted(zip(sp["slab_lo_mhz"], sp["slab_hi_mhz"], strict=True)):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out[split] = {
            "cadences": len(sp),
            "cadences_per_stratum": sp["stratum"].value_counts().reindex(range(N_STRATA),
                                                                       fill_value=0).tolist(),
            "slot_q_values": sorted(sp["slot_q"].round(4).unique().tolist()),
            "distinct_slab_frequencies": int(counts.size),
            "slabs_used_by_more_than_one_cadence": int((counts > 1).sum()),
            "cadences_on_a_repeated_slab": int(counts[counts > 1].sum()),
            "max_cadences_on_one_slab": int(counts.max()),
            "sampled_bandwidth_union_mhz": round(sum(h - lo for lo, h in merged), 4),
            "min_freq_mhz": float(sp["slab_lo_mhz"].min()),
            "max_freq_mhz": float(sp["slab_hi_mhz"].max()),
            "all_slabs_avoid_2300_2360": bool(
                ((sp["slab_hi_mhz"] <= MASK_MHZ[0][0]) | (sp["slab_lo_mhz"] >= MASK_MHZ[0][1])).all()),
            "max_slab_offset_across_positions_mhz": float(
                sp["slab_lo_max_spread_across_positions_mhz"].max()),
        }
    return out


def frame_stats(raw, norm) -> dict:
    med = np.median(norm, axis=(1, 2))
    mad = np.median(np.abs(norm - med[:, None, None]), axis=(1, 2))
    return {
        "raw_min": float(raw.min()), "raw_median": float(np.median(raw)),
        "raw_max": float(raw.max()),
        "norm_per_frame_median_absmax": float(np.abs(med).max()),
        "norm_per_frame_mad_min": float(mad.min()), "norm_per_frame_mad_max": float(mad.max()),
        "norm_all_finite": bool(np.isfinite(norm).all()),
    }


def run_scan(row, plan_row, out_dir: Path, tag: str) -> dict:
    """Fetch the planned slab of one TRAIN file and slice its 42 frames."""

    assert row["split"] == "train"
    start = int(plan_row["slab_start_channel"])
    rec: dict = {"filename": row["filename"], "cadence_position": int(row["cadence_position"]),
                 "slab_start_channel": start}
    t_open = time.time()
    rf = HTTPRangeFile(row["url"], budget=PER_FILE_BUDGET)
    with h5py.File(rf, "r") as h5:
        ds = h5["data"]
        g = read_geometry(ds)
        rec["geometry"] = {"fch1": g.fch1, "foff": g.foff, "nchans": g.nchans,
                           "n_ints": g.n_ints, "tsamp": g.tsamp, "chunks": list(g.chunks),
                           "filters": {str(k): list(v) for k, v in ds._filters.items()}}
        rec["geometry_problems"] = check_fine_geometry(g)
        mg = manifest_geometry(row)
        rec["matches_manifest"] = bool(
            (g.nchans, g.foff, g.chunks) == (mg.nchans, mg.foff, mg.chunks)
            and abs(g.fch1 - mg.fch1) <= abs(g.foff) / 2
        )
        lo, hi = g.window_freq_bounds(start, SLAB_CHANNELS)
        rec["slab_freq_mhz"] = [lo, hi]
        rec["slab_matches_plan"] = bool(abs(lo - plan_row["slab_lo_mhz"]) < 1e-5)
        rec["header_requests"], rec["header_bytes"] = rf.stats.requests, rf.stats.bytes
        spans = slab_chunk_spans(ds, start)
        rec["chunks_in_slab"] = len(spans)
        rec["index_requests"] = rf.stats.requests - rec["header_requests"]
        rec["index_bytes"] = rf.stats.bytes - rec["header_bytes"]
        b0, r0, t0 = rf.stats.bytes, rf.stats.requests, time.time()
        slab = fetch_slab(ds, rf, start, spans=spans)
        rec["slab_fetch_seconds"] = round(time.time() - t0, 2)
        rec["slab_requests"] = rf.stats.requests - r0
        rec["slab_bytes"] = rf.stats.bytes - b0
        rec["slab_compressed_chunk_bytes"] = sum(s for _, s in spans)
        rec["slab_shape"] = list(slab.shape)
    bytes_before_slicing = rf.stats.bytes
    offsets = slab_window_offsets()
    raw = slice_frames(slab, offsets)
    rec["extra_http_bytes_from_slicing"] = rf.stats.bytes - bytes_before_slicing
    del slab
    norm = normalize_fine_frames(raw)
    digests = {hashlib.sha1(f.tobytes()).hexdigest() for f in raw}
    rec.update({
        "window_offsets_in_slab": offsets.tolist(),
        "n_frames": len(raw),
        "unique_frames": len(digests),
        "frame_shapes_ok": all(f.shape == (16, FRAME_CHANNELS) for f in norm),
        "stats": frame_stats(raw, norm),
        "total_requests": rf.stats.requests,
        "total_bytes": rf.stats.bytes,
        "http_statuses": {str(k): v for k, v in rf.stats.statuses.items()},
        "full_body_responses": rf.stats.full_body_responses,
        "retries": rf.stats.retries,
        "wall_seconds": round(time.time() - t_open, 2),
        "bytes_per_retained_frame": round(rf.stats.bytes / len(raw)),
    })
    np.save(out_dir / f"{tag}_frames_raw.npy", raw)
    np.save(out_dir / f"{tag}_frames_norm.npy", norm)
    rec["passed"] = bool(
        not rec["geometry_problems"] and rec["matches_manifest"] and rec["slab_matches_plan"]
        and rec["extra_http_bytes_from_slicing"] == 0 and rec["n_frames"] == FRAMES_PER_SCAN
        and rec["unique_frames"] == FRAMES_PER_SCAN and rec["frame_shapes_ok"]
        and rec["stats"]["norm_all_finite"] and rec["full_body_responses"] == 0
        and set(rec["http_statuses"]) == {"206"}
    )
    print(json.dumps({k: v for k, v in rec.items() if k != "window_offsets_in_slab"}, default=to_json),
          flush=True)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/data/fine_v2_manifest.csv")
    p.add_argument("--out-dir", default="data/processed/fine_v2_chunked_pilot")
    p.add_argument("--plan-only", action="store_true",
                   help="write the metadata-only chunk plan and its summary; no HTTP")
    args = p.parse_args()
    if args.plan_only:
        plan = build_plan(pd.read_csv(args.manifest))
        plan.to_csv(PLAN_PATH, index=False)
        summary = summarize_plan(plan, args.manifest)
        Path(PLAN_SUMMARY_PATH).write_text(json.dumps(summary, indent=2, default=to_json) + "\n")
        print(json.dumps(summary, indent=1, default=to_json))
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)

    plan = build_plan(manifest)
    plan.to_csv(PLAN_PATH, index=False)
    t_start = time.time()

    train = manifest[manifest["split"] == "train"]
    gp_plan = plan[plan["group_id"] == "G001"].iloc[0]
    gp_row = train[(train["group_id"] == "G001") & (train["cadence_position"] == 0)].iloc[0]
    print(f"gpuspec G001 pos0 stratum {gp_plan['stratum']} slab {gp_plan['slab_start_channel']}",
          flush=True)
    gp = run_scan(gp_row, gp_plan, out_dir, "gpuspec_G001_pos0")

    rs_plan = plan[plan["reduction_pipeline"] == "rawspec"].iloc[0]
    assert rs_plan["split"] == "train"
    rs_rows = train[(train["inferred_cadence_id"] == rs_plan["inferred_cadence_id"])
                    & (train["setup_key"] == rs_plan["slab_setup_key"])
                    & (train["cadence_position"] == 0)]
    rs_row = rs_rows.iloc[0]
    print(f"rawspec {rs_plan['group_id']} {rs_row['filename']} slab "
          f"{rs_plan['slab_start_channel']}", flush=True)
    rs = run_scan(rs_row, rs_plan, out_dir, "rawspec_smoke")

    # ---- projection: 252 frames per cadence (42 x 6), 60/15/15 ----
    n_cad = plan.groupby("split").size().to_dict()
    gp_slab_file_bytes = gp["total_bytes"]
    rs_slab_file_bytes = rs["total_bytes"]
    n_rawspec = int((plan["reduction_pipeline"] == "rawspec").sum())
    n_gpuspec = len(plan) - n_rawspec
    rate = gp["slab_bytes"] / gp["slab_fetch_seconds"]
    per_file_s = gp["wall_seconds"]
    by_split = {}
    for split in SPLITS:
        sp = plan[plan["split"] == split]
        nr = int((sp["reduction_pipeline"] == "rawspec").sum())
        ng = len(sp) - nr
        b = 6 * (ng * gp_slab_file_bytes + nr * rs_slab_file_bytes)
        by_split[split] = {"cadences": len(sp), "frames": len(sp) * 6 * FRAMES_PER_SCAN,
                           "transfer_bytes": b}
    total_bytes = sum(v["transfer_bytes"] for v in by_split.values())
    frames = sum(v["frames"] for v in by_split.values())
    projection = {
        "frames_per_cadence": 6 * FRAMES_PER_SCAN,
        "cadences": n_cad,
        "by_split": by_split,
        "frames_total": frames,
        "transfer_bytes_total": total_bytes,
        "basis": "measured whole-file bytes (header + chunk index + slab) of G001 pos0 for each "
                 f"of {n_gpuspec * 6} gpuspec files, G116 smoke file for each of {n_rawspec * 6} "
                 "rawspec files",
        "storage_bytes_raw_float32": frames * FRAME_BYTES,
        "storage_bytes_normalized_float32": frames * FRAME_BYTES,
        "storage_bytes_both": 2 * frames * FRAME_BYTES,
        "single_stream_rate_bytes_per_s": round(rate),
        "wall_clock_hours_by_rate": round(total_bytes / rate / 3600, 1),
        "wall_clock_hours_by_measured_file_time": round(
            6 * (n_gpuspec * per_file_s + n_rawspec * rs["wall_seconds"]) / 3600, 1),
    }
    report = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "TRAIN only for HTTP; validation/test plan rows from frozen manifest metadata only",
        "seed": SEED,
        "rule": {
            "slab_channels": SLAB_CHANNELS,
            "eligible": f"slab outer edges within {SBAND_MHZ} MHz and not intersecting {MASK_MHZ} MHz",
            "strata": f"{N_STRATA} contiguous near-equal-count strata over the cadence's eligible "
                      "slabs sorted by descending centre frequency (all node setups pooled)",
            "stratum_assignment": "sorted group ids shuffled with random.Random(f'{seed}|chunk_strata|"
                                  "{split}'); stratum = rank mod 15",
            "slab_choice": "slot j of m in stratum k: target f_first + q*(f_last - f_first) with "
                           "q=(j+0.5)/m over the stratum's slabs; nearest eligible slab in the "
                           "stratum (ties -> higher frequency)",
            "frames": "42 windows at grid cell floor((i+0.5)*256/42), i=0..41, same for every scan",
            "same_slab_all_positions": True,
            "content_used": False,
        },
        "plan_file": PLAN_PATH,
        "plan_summary": {
            split: {
                "cadences": int((plan["split"] == split).sum()),
                "per_stratum": plan[plan["split"] == split]["stratum"].value_counts()
                .sort_index().tolist(),
                "distinct_slabs": int(plan[plan["split"] == split][
                    ["slab_lo_mhz"]].drop_duplicates().shape[0]),
                "slab_freq_range_mhz": [float(plan[plan["split"] == split]["slab_lo_mhz"].min()),
                                        float(plan[plan["split"] == split]["slab_hi_mhz"].max())],
            } for split in SPLITS
        },
        "gpuspec_scan": gp,
        "rawspec_smoke": rs,
        "projection": projection,
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "processed_disk_bytes": sum(f.stat().st_size for f in out_dir.glob("*.npy")),
    }
    for path in (out_dir / "pilot_report.json", Path(REPORT_PATH)):
        path.write_text(json.dumps(report, indent=2, default=to_json) + "\n")
    print(json.dumps(projection, indent=1, default=to_json))
    return 0 if gp["passed"] and rs["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
