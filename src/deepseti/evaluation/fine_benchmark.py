"""Fine V2 validation synthetic benchmark: deterministic sampling plan + builder.

Benchmark development revision 1 (frozen once generated; see
``configs/experiments/fine_v2_synthetic_benchmark.yaml``, which mirrors every
constant in this module -- ``test_fine_benchmark_config_matches_module``
keeps the two from silently drifting apart).

Scope and held-out policy
--------------------------
Only VALIDATION real observations are used as base frames; training remains
normal-only (synthetic injections never enter PCA/ConvAE/ViT-MAE training).
Four morphologies are used for this development benchmark:
``stationary``, ``linear_drift``, ``intermittent``, ``multiple_carriers``.
Two morphologies are intentionally reserved and NEVER produced by
``build_combo_list`` / the generator script: ``nonlinear_chirp`` and
``burst`` (see ``deepseti.injection.fine.HELD_OUT_MORPHOLOGIES``). They exist
as library functions for later unseen/generalization evaluation only.

Sampling design
---------------
80 distinct (morphology, SNR, ...) settings x 15 repeats (one per validation
cadence group) = 1200 examples. Every example uses a DISTINCT base frame:
for each of the 15 validation cadence groups, its pool of real frames
(252 frames = 6 scans x 42 frames/scan) is deterministically shuffled with a
group-specific seed, and the first 80 shuffled frames are assigned one each
to the 80 settings (in the fixed order ``build_combo_list`` returns). No
frame is ever reused across settings or examples. Frame/track placement
(start channel) is drawn from a single seeded RNG consumed in that same
fixed (setting, group) order, constrained only by frame geometry -- never by
signal/RFI content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from deepseti.injection import fine as inj
from deepseti.preprocessing.fine import normalize_fine_frame

SEED = 20261013
SNR_LEVELS = (0.5, 1.0, 2.0, 5.0, 10.0)
DRIFT_HZ_PER_S_LEVELS = (0.05, 0.5, 2.0, 5.0, 10.0)
DUTY_CYCLE_LEVELS = (0.25, 0.5, 0.75)
CARRIER_COUNTS = (2, 3)
N_VALIDATION_GROUPS = 15
REPEATS_PER_SETTING = N_VALIDATION_GROUPS
FRAME_SHAPE = (16, 4096)
# Normalization is a frame-wide affine transform (median/MAD over all 65536
# pixels), so injecting a handful of signal pixels can shift every other
# pixel's normalized value by a hair. This bounds how large that shift may
# be (in MAD units) before it is flagged as non-negligible.
OUTSIDE_MASK_DIFF_TOLERANCE = 0.1


def build_combo_list() -> list[dict]:
    """Fixed, enumerable order of the 80 development (morphology, params) settings."""

    combos: list[dict] = []
    for snr in SNR_LEVELS:
        combos.append({"morphology": "stationary", "requested_snr": snr})
    for snr in SNR_LEVELS:
        for drift in DRIFT_HZ_PER_S_LEVELS:
            for sign in (1, -1):
                combos.append({"morphology": "linear_drift", "requested_snr": snr,
                               "drift_hz_per_s": sign * drift})
    for snr in SNR_LEVELS:
        for duty in DUTY_CYCLE_LEVELS:
            combos.append({"morphology": "intermittent", "requested_snr": snr,
                           "duty_cycle": duty})
    for snr in SNR_LEVELS:
        for carriers in CARRIER_COUNTS:
            combos.append({"morphology": "multiple_carriers", "requested_snr": snr,
                           "carrier_count": carriers})
    assert all(c["morphology"] in inj.DEVELOPMENT_MORPHOLOGIES for c in combos)
    return combos


def _group_rng(seed: int, group_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|validation_group|{group_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def assign_base_frames(validation_metadata: pd.DataFrame, n_per_group: int,
                       seed: int = SEED) -> dict[str, np.ndarray]:
    """For every cadence group, a deterministic, non-repeating assignment of

    ``n_per_group`` distinct row-indices (into ``validation_metadata``) of
    that group's own frames, one for each of the fixed settings in
    ``build_combo_list()`` order.
    """

    assignment = {}
    for group_id, rows in validation_metadata.groupby("group_id"):
        idx = rows.index.to_numpy()
        if len(idx) < n_per_group:
            raise ValueError(f"group {group_id} has {len(idx)} frames, need {n_per_group}")
        perm = _group_rng(seed, str(group_id)).permutation(len(idx))
        assignment[group_id] = idx[perm[:n_per_group]]
    return assignment


def _build_mask(combo: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    morphology = combo["morphology"]
    if morphology == "stationary":
        start = inj.sample_start_channel(rng, inj.BANDWIDTH_CHANNELS)
        mask = inj.stationary_mask(start)
        return mask, {"injection_start_channel": start, "bandwidth_channels": inj.BANDWIDTH_CHANNELS}
    if morphology == "linear_drift":
        cpr = inj.drift_hz_per_s_to_channels_per_row(combo["drift_hz_per_s"])
        start = inj.sample_linear_drift_start_channel(rng, cpr)
        mask = inj.linear_drift_mask(start, cpr)
        return mask, {"injection_start_channel": start, "bandwidth_channels": inj.BANDWIDTH_CHANNELS,
                     "drift_channels_per_row": cpr}
    if morphology == "intermittent":
        start = inj.sample_start_channel(rng, inj.BANDWIDTH_CHANNELS)
        mask = inj.intermittent_mask(start, combo["duty_cycle"])
        return mask, {"injection_start_channel": start, "bandwidth_channels": inj.BANDWIDTH_CHANNELS}
    if morphology == "multiple_carriers":
        span = inj.multiple_carriers_span_channels(combo["carrier_count"])
        start = inj.sample_start_channel(rng, span)
        mask = inj.multiple_carriers_mask(start, combo["carrier_count"])
        return mask, {"injection_start_channel": start, "bandwidth_channels": inj.BANDWIDTH_CHANNELS}
    raise ValueError(f"unknown development morphology: {morphology}")


def scan_id_of(row: pd.Series) -> str:
    return Path(row["npy"]).stem


class RawFrameCache:
    """Lazy per-scan raw-frame loader (each scan's (42, 16, 4096) array is

    loaded once and reused for every example that draws a base frame from
    that scan).
    """

    def __init__(self, raw_cache_dir: str | Path):
        self.raw_cache_dir = Path(raw_cache_dir)
        self._cache: dict[str, np.ndarray] = {}

    def frame(self, scan_id: str, npy_index: int) -> np.ndarray:
        if scan_id not in self._cache:
            self._cache[scan_id] = np.load(self.raw_cache_dir / f"{scan_id}.npy")
        return self._cache[scan_id][npy_index]


def build_benchmark(validation_metadata: pd.DataFrame, raw_cache_dir: str | Path,
                    seed: int = SEED,
                    combos: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the full deterministic benchmark.

    Returns ``(clean_frames, injected_frames, metadata)`` where the two frame
    arrays are ``(N, 16, 4096)`` float32 normalized frames and ``metadata``
    has one row per example with every field required for downstream scoring,
    grouping and auditing. ``combos`` defaults to ``build_combo_list()``; a
    smaller explicit list is accepted so tests can exercise this function at
    a fraction of the real size.
    """

    is_default_combos = combos is None
    combos = build_combo_list() if is_default_combos else combos
    group_ids = sorted(validation_metadata["group_id"].unique())
    if is_default_combos and len(group_ids) != N_VALIDATION_GROUPS:
        raise ValueError(f"expected {N_VALIDATION_GROUPS} validation cadence groups, "
                         f"got {len(group_ids)}")
    assignment = assign_base_frames(validation_metadata, len(combos), seed=seed)
    placement_rng = np.random.default_rng(seed)
    raw_cache = RawFrameCache(raw_cache_dir)

    clean_frames, injected_frames, rows = [], [], []
    example_id = 0
    for combo_index, combo in enumerate(combos):
        for group_id in group_ids:
            row_idx = assignment[group_id][combo_index]
            meta_row = validation_metadata.loc[row_idx]
            scan_id = scan_id_of(meta_row)
            npy_index = int(meta_row["npy_index"])
            raw_base = raw_cache.frame(scan_id, npy_index)
            if raw_base.shape != FRAME_SHAPE or not np.isfinite(raw_base).all() \
                    or (raw_base <= 0).any():
                raise ValueError(f"invalid raw base frame for {meta_row['frame_id']}")

            clean_norm = normalize_fine_frame(raw_base)
            mask, placement = _build_mask(combo, placement_rng)
            injected_raw, diag = inj.inject_signal(raw_base, mask, combo["requested_snr"])
            injected_norm = normalize_fine_frame(injected_raw)
            background_mask = inj.local_background_mask(raw_base.shape, mask)
            achieved = inj.achieved_local_snr(clean_norm, injected_norm, mask, background_mask)
            # Normalization is a frame-wide affine transform, so even pixels
            # the injection never touches can shift by a tiny amount (the
            # median/MAD move by a hair once a handful of pixel values
            # change); this records how negligible that shift actually is.
            outside_mask_max_abs_diff = float(
                np.abs(clean_norm[~mask] - injected_norm[~mask]).max())

            clean_frames.append(clean_norm)
            injected_frames.append(injected_norm)
            rows.append({
                "example_id": example_id,
                "base_frame_id": meta_row["frame_id"],
                "split": meta_row["split"],
                "group_id": group_id,
                "cadence_id": meta_row["cadence_id"],
                "scan_id": scan_id,
                "frame_index": npy_index,
                "morphology": combo["morphology"],
                "requested_snr": combo["requested_snr"],
                "achieved_local_snr": achieved,
                "drift_hz_s": combo.get("drift_hz_per_s"),
                "drift_channels_per_row": placement.get("drift_channels_per_row"),
                "bandwidth_channels": placement["bandwidth_channels"],
                "duty_cycle": combo.get("duty_cycle"),
                "carrier_count": combo.get("carrier_count"),
                "injection_start_channel": placement["injection_start_channel"],
                "seed": seed,
                "sigma_local": diag["sigma_local"],
                "amplitude": diag["amplitude"],
                "signal_pixels": diag["signal_pixels"],
                "background_pixels": diag["background_pixels"],
                "outside_mask_max_abs_diff": outside_mask_max_abs_diff,
            })
            example_id += 1

    metadata = pd.DataFrame(rows)
    return np.stack(clean_frames), np.stack(injected_frames), metadata


# ---------------------------------------------------------------------------
# Part G sanity audit
# ---------------------------------------------------------------------------


def expected_combo_counts(combos: list[dict] | None = None,
                          repeats_per_setting: int = REPEATS_PER_SETTING) -> dict[str, int]:
    counts: dict[str, int] = {}
    for combo in (build_combo_list() if combos is None else combos):
        counts[combo["morphology"]] = counts.get(combo["morphology"], 0) + 1
    return {m: n * repeats_per_setting for m, n in counts.items()}


def audit_benchmark(clean_frames: np.ndarray, injected_frames: np.ndarray,
                    metadata: pd.DataFrame, combos: list[dict] | None = None,
                    repeats_per_setting: int = REPEATS_PER_SETTING) -> dict:
    """Run every Part G sanity check and return a JSON-serializable report."""

    n = len(metadata)
    checks: dict[str, object] = {}

    checks["all_source_rows_validation"] = bool((metadata["split"] == "validation").all())
    checks["n_examples"] = n
    checks["n_unique_base_frames"] = int(metadata["base_frame_id"].nunique())
    checks["all_base_frames_unique"] = checks["n_unique_base_frames"] == n

    checks["clean_shape_ok"] = clean_frames.shape == (n, *FRAME_SHAPE)
    checks["injected_shape_ok"] = injected_frames.shape == (n, *FRAME_SHAPE)
    checks["clean_finite"] = bool(np.isfinite(clean_frames).all())
    checks["injected_finite"] = bool(np.isfinite(injected_frames).all())

    morphology_counts = metadata["morphology"].value_counts().to_dict()
    checks["morphology_counts"] = {str(k): int(v) for k, v in morphology_counts.items()}
    checks["morphology_counts_match_expected"] = (
        checks["morphology_counts"] == expected_combo_counts(combos, repeats_per_setting))

    snr_counts = metadata["requested_snr"].value_counts().to_dict()
    checks["snr_counts"] = {float(k): int(v) for k, v in snr_counts.items()}
    checks["snr_counts_balanced"] = len(set(snr_counts.values())) == 1

    drift_rows = metadata[metadata["morphology"] == "linear_drift"]
    drift_abs_counts = drift_rows["drift_hz_s"].abs().value_counts().to_dict()
    checks["drift_rate_counts"] = {float(k): int(v) for k, v in drift_abs_counts.items()}
    checks["drift_rate_counts_balanced"] = len(set(drift_abs_counts.values())) == 1
    n_pos = int((drift_rows["drift_hz_s"] > 0).sum())
    n_neg = int((drift_rows["drift_hz_s"] < 0).sum())
    checks["drift_sign_counts"] = {"positive": n_pos, "negative": n_neg}
    checks["drift_sign_balanced"] = n_pos == n_neg

    checks["max_outside_mask_diff"] = float(metadata["outside_mask_max_abs_diff"].max())
    checks["outside_mask_diff_negligible"] = (
        checks["max_outside_mask_diff"] < OUTSIDE_MASK_DIFF_TOLERANCE)

    by_snr = metadata.groupby("requested_snr")["achieved_local_snr"].mean().sort_index()
    checks["achieved_snr_by_requested"] = {float(k): float(v) for k, v in by_snr.items()}
    checks["requested_vs_achieved_monotonic"] = bool(by_snr.is_monotonic_increasing)

    held_out_present = int(metadata["morphology"].isin(inj.HELD_OUT_MORPHOLOGIES).sum())
    checks["held_out_morphology_count"] = held_out_present
    checks["held_out_morphology_count_zero"] = held_out_present == 0

    boundary_ok = True
    for _, row in metadata.iterrows():
        combo = {"morphology": row["morphology"], "requested_snr": row["requested_snr"]}
        if row["morphology"] == "linear_drift":
            combo["drift_hz_per_s"] = row["drift_hz_s"]
        elif row["morphology"] == "intermittent":
            combo["duty_cycle"] = row["duty_cycle"]
        elif row["morphology"] == "multiple_carriers":
            combo["carrier_count"] = int(row["carrier_count"])
        try:
            if row["morphology"] == "stationary":
                mask = inj.stationary_mask(int(row["injection_start_channel"]))
            elif row["morphology"] == "linear_drift":
                mask = inj.linear_drift_mask(int(row["injection_start_channel"]),
                                             row["drift_channels_per_row"])
            elif row["morphology"] == "intermittent":
                mask = inj.intermittent_mask(int(row["injection_start_channel"]), row["duty_cycle"])
            else:
                mask = inj.multiple_carriers_mask(int(row["injection_start_channel"]),
                                                  int(row["carrier_count"]))
            if mask.sum() != row["signal_pixels"]:
                boundary_ok = False
        except ValueError:
            boundary_ok = False
    checks["no_track_crosses_boundary"] = boundary_ok

    bool_keys = [k for k, v in checks.items() if isinstance(v, bool)]
    checks["all_pass"] = all(checks[k] for k in bool_keys)
    return checks
