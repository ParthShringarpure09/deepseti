"""Freeze the Dataset V2 (GBT Fine S-band) 60/15/15 cadence-group split.

Input is the frozen archive-audit catalog written by
scripts/audit_bl_fine_archive.py. No network access, no data download, and no
signal content, RFI cleanliness, anomaly score or model result is used.

Independence rule (pre-registered):
  * split unit = group = connected component linking inferred session folders to
    the ON (A) targets of their inferred ABACAD-like cadences, so an ON target or
    a session can never appear in two splits;
  * off-target identity is NOT a hard exclusion; overlap is minimised and reported.

Selection, in priority order:
  1-2. whole groups are assigned to splits (no ON-target / session leakage);
  3-4. one primary cadence per group, avoiding reuse of a UTC observing day;
  5.   split assignment stratified by (year, reduction pipeline);
  6.   seeded same-stratum label swaps that only ever reduce cross-split
       off-target overlap, then shared observing days;
  7.   cadences with a known header geometry mismatch or unavailable file are
       ineligible; every other file is marked pending individual verification.

Pre-registered fallbacks: each selected group lists ordered alternate cadences
(same group, same split), and unselected groups form an ordered reserve list.

Usage:
    uv run python scripts/select_fine_v2_split.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

CATALOG = Path("results/data_audit/bl_fine_sband_catalog.csv")
AUDIT_SUMMARY = Path("results/data_audit/bl_fine_sband_summary.json")
MANIFEST = Path("configs/data/fine_v2_manifest.csv")
SUMMARY = Path("results/data_audit/fine_v2_split_summary.json")

SEED = 20261013
SPLIT_SIZES = {"train": 60, "validation": 15, "test": 15}
SPLITS = list(SPLIT_SIZES)
SWAP_ITERATIONS = 50_000

REQUIRED_GEOMETRY = {"channel_width_hz": 2.794, "tsamp_s": 18.2536, "n_ints": 16}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cadences(catalog: Path) -> pd.DataFrame:
    df = pd.read_csv(catalog, low_memory=False)
    cad = df[df["cadence_inference_status"] == "inferred_abacad_like"].copy()
    cad["inferred_cadence_position"] = cad["inferred_cadence_position"].astype(int)
    cad["header_status"] = cad["header_status"].fillna("").astype(str)
    return cad


def build_groups(cad: pd.DataFrame) -> dict[str, str]:
    """Map session -> group id (connected components of session ~ ON target)."""
    on = cad[cad["inferred_cadence_position"] == 0]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for session, target in on[["inferred_session_id", "derived_target_norm"]].itertuples(
        index=False
    ):
        parent[find(f"S:{session}")] = find(f"T:{target}")

    members: dict[str, list[str]] = defaultdict(list)
    for node in list(parent):
        if node.startswith("S:"):
            members[find(node)].append(node[2:])
    # Stable numbering: order components by their alphabetically first session.
    ordered = sorted(members.values(), key=lambda s: min(s))
    return {s: f"G{i:03d}" for i, sessions in enumerate(ordered, 1) for s in sessions}


def cadence_table(cad: pd.DataFrame) -> pd.DataFrame:
    def eligibility(g: pd.DataFrame) -> str:
        status = g["header_status"]
        if status.str.startswith(("unavailable", "failed")).any():
            return "ineligible: known unavailable/failed header read"
        verified = g[status == "ok"]
        if (verified["hdr_fine_geometry_match"].astype(str) != "True").any():
            return "ineligible: header-verified geometry mismatch"
        return "eligible"

    rows = []
    for cid, g in cad.groupby("inferred_cadence_id"):
        a = g[g["inferred_cadence_position"] == 0]
        rows.append({
            "cadence_id": cid,
            "group_id": g["group_id"].iloc[0],
            "session": g["inferred_session_id"].iloc[0],
            "on_target": a["derived_target_norm"].iloc[0],
            "off_targets": sorted(
                set(g.loc[g["inferred_cadence_position"] != 0, "derived_target_norm"])
            ),
            "day": a["derived_observing_day_utc"].iloc[0],
            "year": int(a["derived_year"].iloc[0]),
            "pipeline": a["derived_reduction_pipeline"].iloc[0],
            "era": a["derived_archive_era"].iloc[0],
            "n_files": len(g),
            "n_setups": g["inferred_setup_key"].nunique(),
            "size_bytes": int(g["size"].sum()),
            "n_header_verified": int((g["header_status"] == "ok").sum()),
            "eligibility": eligibility(g),
        })
    return pd.DataFrame(rows)


def choose_primary_cadences(cads: pd.DataFrame, seed: int) -> dict[str, dict]:
    """One primary cadence per group + ordered alternates; avoid reusing UTC days."""
    eligible = cads[cads["eligibility"] == "eligible"]
    ranked: dict[str, list[dict]] = {}
    for gid, g in eligible.groupby("group_id"):
        rng = random.Random(f"{seed}:cadence-rank:{gid}")
        recs = g.sort_values("cadence_id").to_dict("records")
        rng.shuffle(recs)
        ranked[gid] = recs

    order = sorted(ranked)
    random.Random(f"{seed}:group-order").shuffle(order)
    used_days: set[str] = set()
    chosen: dict[str, dict] = {}
    for gid in order:
        recs = ranked[gid]
        pick = next((i for i, r in enumerate(recs) if r["day"] not in used_days), 0)
        primary = recs[pick]
        used_days.add(primary["day"])
        chosen[gid] = {
            "primary": primary,
            "alternates": [r["cadence_id"] for i, r in enumerate(recs) if i != pick],
        }
    return chosen


def stratified_labels(chosen: dict[str, dict], seed: int) -> dict[str, str]:
    """Systematic allocation over groups sorted by (year, pipeline, seeded key)."""
    rng = random.Random(f"{seed}:stratify")
    keys = {gid: rng.random() for gid in sorted(chosen)}
    order = sorted(
        chosen,
        key=lambda g: (chosen[g]["primary"]["year"], chosen[g]["primary"]["pipeline"], keys[g]),
    )
    n = len(order)
    quotas = {**SPLIT_SIZES, "reserve": n - sum(SPLIT_SIZES.values())}
    if quotas["reserve"] < 0:
        raise SystemExit(f"Only {n} eligible groups; need {sum(SPLIT_SIZES.values())}.")
    assigned = Counter()
    labels = {}
    for k, gid in enumerate(order):
        # Largest-deficit rule keeps every stratum spread proportionally.
        label = max(quotas, key=lambda s: (quotas[s] * (k + 1) / n - assigned[s], quotas[s]))
        labels[gid] = label
        assigned[label] += 1
    return labels


def manifest_rows(files: pd.DataFrame) -> pd.DataFrame:
    """Catalog rows (with split, group_id, geometry_status, header_status) -> manifest."""
    return pd.DataFrame({
        "split": files["split"],
        "group_id": files["group_id"],
        "inferred_session_id": files["inferred_session_id"],
        "inferred_cadence_id": files["inferred_cadence_id"],
        "target": files["target"],
        "target_norm": files["derived_target_norm"],
        "cadence_position": files["inferred_cadence_position"],
        "cadence_role": files["inferred_cadence_position"].map(
            {0: "ON_A", 1: "OFF_B", 2: "ON_A", 3: "OFF_C", 4: "ON_A", 5: "OFF_D"}
        ),
        "archive_id": files["id"],
        "url": files["url"],
        "filename": files["derived_filename"],
        "reduction_pipeline": files["derived_reduction_pipeline"],
        "archive_era": files["derived_archive_era"],
        "mjd": files["mjd"],
        "utc": files["utc"],
        "observing_day_utc": files["derived_observing_day_utc"],
        "center_freq": files["center_freq"],
        "setup_key": files["inferred_setup_key"],
        "year": files["derived_year"],
        "session_folder": files["derived_session_folder"],
        "scan_number": files["derived_scan_number"],
        "quality": files["quality"],
        "size_bytes": files["size"],
        "md5sum": files["md5sum"],
        "geometry_status": files["geometry_status"],
        "header_status": files["header_status"],
    })


def sort_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    split_order = {s: i for i, s in enumerate(SPLITS)}
    return manifest.sort_values(
        ["split", "group_id", "setup_key", "cadence_position"],
        key=lambda c: c.map(split_order) if c.name == "split" else c,
    ).reset_index(drop=True)


def overlap_cost(labels: dict[str, str], chosen: dict[str, dict]) -> tuple[int, int]:
    target_splits: dict[str, set] = defaultdict(set)
    day_splits: dict[str, set] = defaultdict(set)
    for gid, label in labels.items():
        if label == "reserve":
            continue
        p = chosen[gid]["primary"]
        for t in [p["on_target"], *p["off_targets"]]:
            target_splits[t].add(label)
        day_splits[p["day"]].add(label)
    cross_targets = sum(len(s) > 1 for s in target_splits.values())
    cross_days = sum(len(s) > 1 for s in day_splits.values())
    return cross_targets, cross_days


def minimise_overlap(labels, chosen, seed):
    rng = random.Random(f"{seed}:swap-search")
    stratum = {g: (chosen[g]["primary"]["year"], chosen[g]["primary"]["pipeline"])
               for g in labels}
    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for g in sorted(labels):
        by_stratum[stratum[g]].append(g)
    swappable = [s for s, gs in sorted(by_stratum.items())
                 if len({labels[g] for g in gs}) > 1]
    start = cost = overlap_cost(labels, chosen)
    accepted = 0
    for _ in range(SWAP_ITERATIONS):
        if not swappable:
            break
        gs = by_stratum[rng.choice(swappable)]
        a, b = rng.sample(gs, 2)
        if labels[a] == labels[b]:
            continue
        labels[a], labels[b] = labels[b], labels[a]
        new = overlap_cost(labels, chosen)
        if new < cost:
            cost, accepted = new, accepted + 1
        else:
            labels[a], labels[b] = labels[b], labels[a]
    return labels, {"start": start, "end": cost, "accepted_swaps": accepted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cad = load_cadences(CATALOG)
    session_group = build_groups(cad)
    cad["group_id"] = cad["inferred_session_id"].map(session_group)
    cads = cadence_table(cad)
    chosen = choose_primary_cadences(cads, args.seed)
    labels = stratified_labels(chosen, args.seed)
    labels, search = minimise_overlap(labels, chosen, args.seed)

    selected = {gid: lab for gid, lab in labels.items() if lab != "reserve"}
    primary_ids = {chosen[g]["primary"]["cadence_id"]: g for g in selected}
    files = cad[cad["inferred_cadence_id"].isin(primary_ids)].copy()
    files["split"] = files["group_id"].map(selected)

    ok = files["header_status"] == "ok"
    files["geometry_status"] = "pending_header_verification"
    files.loc[ok, "geometry_status"] = "header_verified_match"
    files["header_status"] = files["header_status"].replace("", "not_sampled")

    manifest = sort_manifest(manifest_rows(files))

    # ---- hard checks -----------------------------------------------------
    on = manifest[manifest["cadence_position"].isin([0, 2, 4])]
    on_splits = on.groupby("target_norm")["split"].nunique()
    session_splits = manifest.groupby("inferred_session_id")["split"].nunique()
    group_splits = manifest.groupby("group_id")["split"].nunique()
    # ON targets of ANY inferred cadence (selected or not) must stay within one split.
    all_on = cad[cad["inferred_cadence_position"] == 0]
    all_on_group = all_on.groupby("derived_target_norm")["group_id"].nunique()
    unavailable = manifest["header_status"].str.startswith("unavailable").sum()
    mismatch = (manifest["geometry_status"] == "header_verified_mismatch").sum()
    checks = {
        "on_target_overlap_across_splits": int((on_splits > 1).sum()),
        "session_overlap_across_splits": int((session_splits > 1).sum()),
        "group_overlap_across_splits": int((group_splits > 1).sum()),
        "on_targets_spanning_multiple_groups_catalog_wide": int((all_on_group > 1).sum()),
        "selected_unavailable_files": int(unavailable),
        "selected_known_geometry_mismatch_files": int(mismatch),
    }
    if any(checks.values()):
        raise SystemExit(f"Independence/availability check failed: {checks}")

    target_splits = manifest.groupby("target_norm")["split"].agg(set)
    off_only = manifest[~manifest["target_norm"].isin(on["target_norm"])]
    off_cross = {t for t, s in target_splits.items() if len(s) > 1}
    on_as_off_cross = sorted(off_cross & set(on["target_norm"]))
    day_splits = manifest.groupby("observing_day_utc")["split"].nunique()

    per_split = {}
    for s in SPLITS:
        m = manifest[manifest["split"] == s]
        g = m.drop_duplicates("inferred_cadence_id")
        per_split[s] = {
            "groups": int(m["group_id"].nunique()),
            "cadences": int(m["inferred_cadence_id"].nunique()),
            "files": len(m),
            "unique_on_targets": int(m.loc[m["cadence_position"].isin([0, 2, 4]),
                                           "target_norm"].nunique()),
            "unique_sessions": int(m["inferred_session_id"].nunique()),
            "unique_observing_days": int(m["observing_day_utc"].nunique()),
            "year_distribution_cadences": {str(k): int(v) for k, v in
                                           sorted(Counter(g["year"]).items())},
            "pipeline_distribution_cadences": dict(Counter(g["reduction_pipeline"])),
            "pipeline_distribution_files": dict(Counter(m["reduction_pipeline"])),
            "era_distribution_cadences": dict(Counter(g["archive_era"])),
            "files_pending_header_verification": int(
                (m["geometry_status"] == "pending_header_verification").sum()),
            "storage_tib": round(m["size_bytes"].sum() / 1024**4, 3),
        }

    reserve = sorted((g for g, lab in labels.items() if lab == "reserve"),
                     key=lambda g: random.Random(f"{args.seed}:reserve:{g}").random())
    group_counts = cads.groupby("group_id")["eligibility"].agg(
        lambda e: int((e == "eligible").sum()))
    summary = {
        "dataset": "DeepSeti Dataset V2 - GBT Fine HDF5 S-band (provisional), frozen split",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "inputs": {
            "catalog": str(CATALOG), "catalog_sha256": sha256(CATALOG),
            "audit_summary": str(AUDIT_SUMMARY), "audit_summary_sha256": sha256(AUDIT_SUMMARY),
        },
        "outputs": {"manifest": str(MANIFEST), "summary": str(SUMMARY)},
        "independence_rule": {
            "split_unit": "group = connected component of inferred session folders and "
                          "the ON (A) targets of their inferred ABACAD-like cadences",
            "hard": ["no ON target in more than one split",
                     "no inferred session/cadence in more than one split"],
            "soft": "off-target identity overlap minimised (not excluded) and reported",
        },
        "selection_procedure": [
            ("Eligible cadence: every file passed audit filters; no file with an "
             "unavailable/failed header read; no header-verified geometry mismatch."),
            ("Per group, eligible cadences are ranked by a seeded shuffle "
             "(Random('<seed>:cadence-rank:<group>'))."),
            ("Groups are visited in seeded order; the primary cadence is the first "
             "ranked cadence whose UTC observing day is not yet used (else the first)."),
            ("Groups are sorted by (year, pipeline, seeded key) and allocated "
             f"{SPLIT_SIZES} + reserve with a largest-deficit systematic rule."),
            (f"{SWAP_ITERATIONS} seeded same-stratum label swaps; a swap is kept only "
             "if it strictly lowers (cross-split target count, cross-split day count)."),
            "No signal content, RFI cleanliness, anomaly score or model result used.",
        ],
        "groups": {
            "total": len(set(session_group.values())),
            "with_eligible_cadence": len(chosen),
            "without_eligible_cadence": sorted(set(session_group.values()) - set(chosen)),
            "eligible_cadences_per_group": describe_counts(group_counts),
            "ineligible_cadences": dict(Counter(cads.loc[cads["eligibility"] != "eligible",
                                                         "eligibility"])),
        },
        "split_counts": per_split,
        "checks": checks,
        "off_target_overlap": {
            "targets_in_more_than_one_split": len(off_cross),
            "on_targets_also_off_target_in_another_split": len(on_as_off_cross),
            "off_only_targets_in_more_than_one_split": len(
                off_cross & set(off_only["target_norm"])),
            "targets": sorted(off_cross),
            "swap_search": search,
        },
        "observing_days_shared_across_splits": int((day_splits > 1).sum()),
        "geometry": {
            "required": REQUIRED_GEOMETRY,
            "rule": "Every selected file must be individually header-verified before "
                    "extraction; files without 16 usable integrations are excluded "
                    "from the final manifest.",
            "files_header_verified_match": int(
                (manifest["geometry_status"] == "header_verified_match").sum()),
            "files_pending_header_verification": int(
                (manifest["geometry_status"] == "pending_header_verification").sum()),
        },
        "fallback_rule": (
            "If any file of a primary cadence fails individual header verification "
            "(unavailable, or not ~2.794 Hz / ~18.2536 s / 16 integrations), replace that "
            "cadence with the next alternate cadence of the SAME group (same split). If a "
            "group has no passing alternate, replace the group with the first reserve "
            "group (in listed order) that passes and introduces no ON-target or session "
            "leakage; record every replacement."
        ),
        "selected_groups": {
            gid: {"split": selected[gid],
                  "primary_cadence": chosen[gid]["primary"]["cadence_id"],
                  "alternates": chosen[gid]["alternates"]}
            for gid in sorted(selected)
        },
        "reserve_groups_ordered": [
            {"group_id": g, "primary_cadence": chosen[g]["primary"]["cadence_id"],
             "alternates": chosen[g]["alternates"]}
            for g in reserve
        ],
        "manifest_rows": len(manifest),
        "limitations": [
            "Session, cadence and group IDs are inferred, not archive-native.",
            "S-band membership is provisional (receiver not exposed by the archive).",
            "Quality is 'Ungraded' for all files; Grade A was not available.",
            "Most files are not yet header-verified (geometry_status records this).",
            ("The 2300-2360 MHz region is not removed here; it is a predefined "
             "frame-level mask for extraction."),
        ],
    }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST, index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST} ({len(manifest)} rows)\nWrote {SUMMARY}")
    print(json.dumps({"split_counts": per_split, "checks": checks,
                      "off_target_overlap": {k: v for k, v in
                                             summary["off_target_overlap"].items()
                                             if k != "targets"},
                      "observing_days_shared_across_splits":
                          summary["observing_days_shared_across_splits"],
                      "geometry": summary["geometry"],
                      "groups": {k: v for k, v in summary["groups"].items()
                                 if k != "eligible_cadences_per_group"}}, indent=1))
    return 0


def describe_counts(s: pd.Series) -> dict:
    return {"min": int(s.min()), "median": float(s.median()), "max": int(s.max())}


if __name__ == "__main__":
    raise SystemExit(main())
