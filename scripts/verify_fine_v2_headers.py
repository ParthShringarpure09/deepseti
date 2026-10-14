"""Header-verify every file of the frozen Dataset V2 manifest and apply the
pre-registered cadence replacement rule.

Uses only small HTTP range reads of HDF5 metadata (no observation data, no
signal content). The cadence is the atomic unit: if ANY of its files is
unavailable, unreadable, not a FILTERBANK HDF5, or off-geometry, the whole
cadence is rejected and the next pre-registered alternate of the SAME group
(same split) is tried. No new backups are invented and no group changes split.

The frozen selection (primaries + ordered alternates) is read from
results/data_audit/fine_v2_split_summary.json, so re-running reproduces the
same manifest and the same replacement log.

Usage:
    uv run python scripts/verify_fine_v2_headers.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_bl_fine_archive as audit
import select_fine_v2_split as select

CATALOG = Path("results/data_audit/bl_fine_sband_catalog.csv")
SPLIT_SUMMARY = Path("results/data_audit/fine_v2_split_summary.json")
MANIFEST = Path("configs/data/fine_v2_manifest.csv")
REPORT = Path("results/data_audit/fine_v2_header_verification.json")

CHANNEL_WIDTH_HZ = 2.7939677238464355
TSAMP_S = 18.253611008
N_INTS = 16
REL_TOL = 1e-3  # +/-0.1%: rejects e.g. the 2.8355 Hz / 17.986 s setup seen in the audit

HDR_COLUMNS = [
    "header_http_status", "hdr_channel_width_hz", "hdr_tsamp", "hdr_n_ints",
    "hdr_data_shape", "hdr_data_chunks", "hdr_fch1", "hdr_foff", "hdr_nchans",
    "hdr_f_lo_mhz", "hdr_f_hi_mhz", "hdr_bytes_fetched",
]


def check_file(url: str, timeout: float) -> dict:
    """Read one header and judge it. Runs in a worker process."""
    rec: dict = {"url": url}
    try:
        hdr = audit.read_header(url, Namespace(timeout=timeout))
    except audit.ArchiveFileUnavailable as exc:
        return {**rec, "header_status": "unavailable", "header_http_status": exc.status,
                "reason": f"unavailable: {exc}", "hdr_bytes_fetched": 0}
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        return {**rec, "header_status": "unreadable", "header_http_status": None,
                "reason": f"unreadable: {type(exc).__name__}: {exc}"[:300],
                "hdr_bytes_fetched": None}

    rec.update({k: hdr.get(k) for k in HDR_COLUMNS if k in hdr})
    rec["header_http_status"] = 206
    problems = []
    root = json.loads(hdr["hdr_root_attrs"])
    if root.get("CLASS") != "FILTERBANK":
        problems.append(f"root CLASS={root.get('CLASS')!r} (expected FILTERBANK)")
    if "data" not in hdr["hdr_datasets"].split(","):
        problems.append("no 'data' dataset")
    shape = [int(x) for x in hdr["hdr_data_shape"].split("x")]
    if len(shape) != 3 or shape[1] != 1:
        problems.append(f"unexpected data shape {hdr['hdr_data_shape']}")
    if not hdr.get("hdr_nchans_consistent"):
        problems.append("nchans attribute != data shape")
    width, tsamp, n_ints = hdr.get("hdr_channel_width_hz"), hdr.get("hdr_tsamp"), hdr.get("hdr_n_ints")
    if width is None or abs(width - CHANNEL_WIDTH_HZ) > REL_TOL * CHANNEL_WIDTH_HZ:
        problems.append(f"channel width {width} Hz")
    if tsamp is None or abs(tsamp - TSAMP_S) > REL_TOL * TSAMP_S:
        problems.append(f"tsamp {tsamp} s")
    if n_ints != N_INTS:
        problems.append(f"n_ints {n_ints} != {N_INTS}")
    if problems:
        rec.update(header_status="verified_ok", geometry_status="incompatible",
                   reason="geometry/structure: " + "; ".join(problems))
    else:
        rec.update(header_status="verified_ok", geometry_status="compatible", reason="")
    return rec


def verify_urls(urls: list[str], workers: int, timeout: float, cache: dict) -> None:
    todo = sorted(set(urls) - set(cache))
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(check_file, todo, [timeout] * len(todo)), 1):
            cache[rec["url"]] = rec
            if i % 50 == 0 or i == len(todo):
                bad = sum(cache[u].get("geometry_status") != "compatible" for u in todo[:i])
                print(f"  {i}/{len(todo)} checked, {bad} not compatible")


def cadence_verdict(cid: str, files_by_cadence: dict, cache: dict) -> tuple[bool, str]:
    reasons = []
    for _, f in files_by_cadence[cid].iterrows():
        rec = cache[f["url"]]
        if rec.get("geometry_status") != "compatible":
            reasons.append(f"archive_id {f['id']} ({f['derived_filename']}): {rec['reason']}")
    return (not reasons), "; ".join(reasons)


def leakage(cads: dict[str, dict]) -> tuple[set, set]:
    """Targets / observing days appearing in more than one split."""
    t_splits, d_splits = defaultdict(set), defaultdict(set)
    for c in cads.values():
        for t in c["targets"]:
            t_splits[t].add(c["split"])
        d_splits[c["day"]].add(c["split"])
    return ({t for t, s in t_splits.items() if len(s) > 1},
            {d for d, s in d_splits.items() if len(s) > 1})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    started = datetime.now(UTC)

    frozen = json.loads(SPLIT_SUMMARY.read_text(encoding="utf-8"))
    groups = frozen["selected_groups"]
    reserve = frozen["reserve_groups_ordered"]  # frozen order; never re-sorted
    frozen_sha = hashlib.sha256(json.dumps(
        {k: frozen[k] for k in ("seed", "selected_groups", "reserve_groups_ordered")},
        sort_keys=True).encode()).hexdigest()
    options = {gid: [g["primary_cadence"], *g["alternates"]] for gid, g in groups.items()}
    options.update({r["group_id"]: [r["primary_cadence"], *r["alternates"]] for r in reserve})
    catalog = pd.read_csv(CATALOG, low_memory=False)
    cad = catalog[catalog["cadence_inference_status"] == "inferred_abacad_like"].copy()
    cad["inferred_cadence_position"] = cad["inferred_cadence_position"].astype(int)
    files_by_cadence = {cid: g for cid, g in cad.groupby("inferred_cadence_id")}

    # The current manifest must be the frozen selection or a previous output of this
    # script: every group selected or reserve, every cadence a pre-registered option.
    current = pd.read_csv(MANIFEST)
    for (gid, split), m in current.groupby(["group_id", "split"]):
        if gid not in options:
            raise SystemExit(f"Manifest group {gid} is neither selected nor reserve")
        if gid in groups and groups[gid]["split"] != split:
            raise SystemExit(f"Manifest group {gid} moved split ({split})")
        if not set(m["inferred_cadence_id"]) <= set(options[gid]):
            raise SystemExit(f"Manifest cadence for {gid} is not a pre-registered option")
    if current["group_id"].nunique() != len(groups):
        raise SystemExit("Manifest group count differs from the frozen selection")

    def cadence_info(cid: str, split: str) -> dict:
        f = files_by_cadence[cid]
        a = f[f["inferred_cadence_position"] == 0].iloc[0]
        return {"split": split, "day": a["derived_observing_day_utc"],
                "session": a["inferred_session_id"], "on_target": a["derived_target_norm"],
                "targets": set(f["derived_target_norm"])}

    def try_cadence(cid: str, slot: str, split: str, accepted: dict) -> tuple[bool, str]:
        verify_urls(list(files_by_cadence[cid]["url"]), args.workers, args.timeout, cache)
        ok, why = cadence_verdict(cid, files_by_cadence, cache)
        if not ok:
            return False, why
        info = cadence_info(cid, split)
        others = [c for g, c in accepted.items() if g != slot]
        if any(c["session"] == info["session"] for c in others):
            return False, f"session {info['session']} already selected"
        if any(info["on_target"] in c["targets"] or c["on_target"] in info["targets"]
               for c in others if c["split"] != split):
            return False, "would add cross-split ON-target overlap"
        cross_t, cross_d = leakage({**accepted, slot: info})
        if cross_t or cross_d:
            return False, (f"would add cross-split overlap: targets {sorted(cross_t)} "
                           f"days {sorted(cross_d)}")
        return True, ""

    cache: dict[str, dict] = {}
    primaries = {gid: g["primary_cadence"] for gid, g in groups.items()}
    print(f"Verifying {sum(len(files_by_cadence[c]) for c in primaries.values())} files "
          f"of {len(primaries)} primary cadences")
    verify_urls([u for c in primaries.values() for u in files_by_cadence[c]["url"]],
                args.workers, args.timeout, cache)

    # slot -> (group actually used, cadence, selection_source)
    accepted: dict[str, dict] = {}
    final: dict[str, tuple[str, str, str]] = {}
    rejected_primary: dict[str, str] = {}
    for gid in sorted(groups):
        ok, why = cadence_verdict(primaries[gid], files_by_cadence, cache)
        if ok:
            final[gid] = (gid, primaries[gid], "primary")
            accepted[gid] = cadence_info(primaries[gid], groups[gid]["split"])
        else:
            rejected_primary[gid] = why

    replacement_log = []
    for gid in sorted(rejected_primary):
        split = groups[gid]["split"]
        log = {"group_id": gid, "split": split, "original_cadence": primaries[gid],
               "reason": rejected_primary[gid], "skipped_alternates": [],
               "backup_cadence_used": None, "reserve_candidates_tried": [],
               "reserve_group_used": None, "reserve_cadence_used": None}
        for alt in groups[gid]["alternates"]:
            ok, why = try_cadence(alt, gid, split, accepted)
            if ok:
                final[gid] = (gid, alt, "same_group_alternate")
                accepted[gid] = cadence_info(alt, split)
                log["backup_cadence_used"] = alt
                break
            log["skipped_alternates"].append({"cadence": alt, "reason": why})
        if log["backup_cadence_used"] is None:
            # Pre-registered fallback: first passing reserve group in frozen order.
            used = {g for g, _, _ in final.values()}
            for r in reserve:
                rid = r["group_id"]
                if rid in used:
                    continue
                tried = {"group_id": rid, "cadences": []}
                for cid in options[rid]:
                    ok, why = try_cadence(cid, gid, split, accepted)
                    tried["cadences"].append({"cadence": cid, "passed": ok, "reason": why})
                    if ok:
                        final[gid] = (rid, cid, f"reserve_for_{gid}")
                        accepted[gid] = cadence_info(cid, split)
                        log["reserve_group_used"], log["reserve_cadence_used"] = rid, cid
                        break
                log["reserve_candidates_tried"].append(tried)
                if log["reserve_group_used"]:
                    break
        replacement_log.append(log)

    exhausted = [r for r in replacement_log
                 if r["backup_cadence_used"] is None and r["reserve_group_used"] is None]
    if exhausted:
        for r in exhausted:
            print(f"! {r['split']} {r['group_id']}: no same-group alternate or reserve passed")
        print("Stopping WITHOUT writing a manifest.")
        REPORT.write_text(json.dumps({
            "status": "FAILED_NO_PASSING_REPLACEMENT",
            "generated_utc": started.isoformat(timespec="seconds"),
            "exhausted_slots": [r["group_id"] for r in exhausted],
            "files_attempted": len(cache),
            "total_header_bytes": int(sum(r.get("hdr_bytes_fetched") or 0
                                          for r in cache.values())),
            "replacement_log": replacement_log,
            "files": sorted(cache.values(), key=lambda r: r["url"]),
        }, indent=2, default=str) + "\n", encoding="utf-8")
        return 2

    # ---- build final manifest --------------------------------------------
    files = pd.concat([
        files_by_cadence[cid].assign(group_id=g, split=groups[slot]["split"],
                                     selection_source=source)
        for slot, (g, cid, source) in sorted(final.items())
    ])
    hdr = pd.DataFrame([cache[u] for u in files["url"]]).set_index("url")
    files["header_status"] = files["url"].map(hdr["header_status"])
    files["geometry_status"] = files["url"].map(hdr["geometry_status"])
    manifest = select.manifest_rows(files)
    manifest["selection_source"] = files["selection_source"].values
    for col in HDR_COLUMNS:
        manifest[col] = files["url"].map(hdr[col]).values
    manifest["header_http_status"] = manifest["header_http_status"].astype("Int64")
    manifest["hdr_n_ints"] = manifest["hdr_n_ints"].astype("Int64")
    manifest = select.sort_manifest(manifest)

    # ---- invariants ---------------------------------------------------------
    on = manifest[manifest["cadence_position"].isin([0, 2, 4])]
    per_cad = manifest.drop_duplicates("inferred_cadence_id")
    cross_t, cross_d = leakage(accepted)
    expected_files = sum(len(files_by_cadence[c]) for _, c, _ in final.values())
    split_groups = per_cad.groupby("split")["group_id"].nunique().to_dict()
    invariants = {
        "train_groups": split_groups.get("train", 0) == 60,
        "validation_groups": split_groups.get("validation", 0) == 15,
        "test_groups": split_groups.get("test", 0) == 15,
        "total_cadences": manifest["inferred_cadence_id"].nunique() == 90,
        "on_target_overlap_zero": int((on.groupby("target_norm")["split"].nunique() > 1).sum()) == 0,
        "session_overlap_zero": int((manifest.groupby("inferred_session_id")["split"]
                                     .nunique() > 1).sum()) == 0,
        "off_target_overlap_zero": len(cross_t) == 0,
        "observing_day_overlap_zero": len(cross_d) == 0,
        "unavailable_selected_files_zero": int((manifest["header_status"] == "unavailable").sum()) == 0,
        "geometry_mismatch_files_zero": int((manifest["geometry_status"] != "compatible").sum()) == 0,
        "all_rows_header_verified": bool((manifest["header_status"] == "verified_ok").all()),
        "all_rows_geometry_compatible": bool((manifest["geometry_status"] == "compatible").all()),
        "all_cadence_files_verified": len(manifest) == expected_files
        and bool((manifest.groupby("inferred_cadence_id").size() ==
                  manifest.groupby("inferred_cadence_id")["cadence_position"].count()).all())
        and bool((manifest.groupby(["inferred_cadence_id", "setup_key"])["cadence_position"]
                  .nunique() == 6).all()),
    }
    if not all(invariants.values()):
        raise SystemExit(f"Invariant failure: {invariants}")

    manifest.to_csv(MANIFEST, index=False)
    md5 = hashlib.md5(MANIFEST.read_bytes()).hexdigest()

    attempted = list(cache.values())
    status = Counter(r.get("geometry_status") or r["header_status"] for r in attempted)
    years = {s: dict(sorted(Counter(per_cad.loc[per_cad["split"] == s, "year"]).items()))
             for s in select.SPLITS}
    report = {
        "generated_utc": started.isoformat(timespec="seconds"),
        "inputs": {"split_summary": str(SPLIT_SUMMARY),
                   "frozen_selection_sha256": frozen_sha,
                   "catalog": str(CATALOG), "catalog_sha256": sha256(CATALOG)},
        "geometry_rule": {"channel_width_hz": CHANNEL_WIDTH_HZ, "tsamp_s": TSAMP_S,
                          "n_ints": N_INTS, "relative_tolerance": REL_TOL,
                          "structure": "root CLASS=FILTERBANK, 'data' dataset of shape "
                                       "(n_ints, 1, nchans), nchans attr == shape"},
        "method": "HTTP range reads of 64 KiB blocks, <= 1 MiB per file; whole cadence "
                  "rejected if any file fails; next pre-registered alternate of the same "
                  "group/split used; alternates must also add no cross-split target or "
                  "observing-day overlap.",
        "files_attempted": len(attempted),
        "files_passed": status.get("compatible", 0),
        "files_failed": {k: v for k, v in status.items() if k != "compatible"},
        "failure_reasons": dict(Counter(r["reason"].split(":")[0] for r in attempted
                                        if r.get("geometry_status") != "compatible")),
        "total_header_bytes": int(sum(r.get("hdr_bytes_fetched") or 0 for r in attempted)),
        "cadence_replacements": len(replacement_log),
        "replacement_log": replacement_log,
        "final": {
            "manifest": str(MANIFEST), "manifest_md5": md5, "rows": len(manifest),
            "cadences_per_split": per_cad["split"].value_counts().reindex(select.SPLITS).to_dict(),
            "pipeline_cadences": per_cad["reduction_pipeline"].value_counts().to_dict(),
            "pipeline_files": manifest["reduction_pipeline"].value_counts().to_dict(),
            "year_distribution_cadences": years,
            "invariants": invariants,
            "off_target_overlap_targets": sorted(cross_t),
            "observing_day_overlap_days": sorted(cross_d),
        },
        "files": sorted(attempted, key=lambda r: r["url"]),
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    # Provenance only: the frozen selection/reserve lists are left untouched.
    frozen["header_verification_replacements"] = {
        "source": str(REPORT),
        "frozen_selection_sha256": frozen_sha,
        "final_manifest_md5": md5,
        "replacements": [
            {"slot_group": r["group_id"], "split": r["split"],
             "original_cadence": r["original_cadence"],
             "reason": "rejected: incompatible geometry/structure/availability; "
                       + r["reason"].split(": ", 1)[-1].split(";")[0],
             "same_group_alternates_rejected": [a["cadence"] for a in r["skipped_alternates"]],
             "same_group_alternate_used": r["backup_cadence_used"],
             "reserve_groups_rejected": [t["group_id"] for t in r["reserve_candidates_tried"]
                                         if t["group_id"] != r["reserve_group_used"]],
             "reserve_group_used": r["reserve_group_used"],
             "reserve_cadence_used": r["reserve_cadence_used"]}
            for r in replacement_log
        ],
    }
    SPLIT_SUMMARY.write_text(json.dumps(frozen, indent=2, default=str) + "\n",
                             encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("files_attempted", "files_passed", "files_failed",
                                              "failure_reasons", "total_header_bytes",
                                              "cadence_replacements")}, indent=1))
    for r in replacement_log:
        used = r["backup_cadence_used"] or f"reserve {r['reserve_group_used']}: " \
                                          f"{r['reserve_cadence_used']}"
        tried = [t["group_id"] for t in r["reserve_candidates_tried"]]
        print(f"  {r['split']:10s} {r['group_id']}: {r['original_cadence']} -> {used}\n"
              f"      reason: {r['reason'][:120]}\n      reserve groups tried: {tried}")
    print(json.dumps({k: v for k, v in report["final"].items()
                      if k not in ("off_target_overlap_targets", "observing_day_overlap_days")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
