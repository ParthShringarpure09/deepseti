"""Build the Fine V2 VALIDATION synthetic injection benchmark (development revision 1).

Reads only the frozen VALIDATION metadata/raw-frame cache (never train, never
final test), builds the deterministic 1200-example benchmark (see
``deepseti.evaluation.fine_benchmark``), runs every Part G sanity check
(including rebuilding the whole benchmark a second time to verify
deterministic regeneration), writes a small set of deterministic diagnostic
plots, and saves:

  data/synthetic/fine_v2_validation/clean_frames.npy    (gitignored)
  data/synthetic/fine_v2_validation/injected_frames.npy (gitignored)
  data/synthetic/fine_v2_validation/metadata.csv         (gitignored)
  results/benchmarks/fine_v2_validation_benchmark_audit.json
  results/benchmarks/figures/*.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_fine_v2 import validate_split_access

from deepseti.evaluation import fine_benchmark as fb

CONFIG_PATH = "configs/experiments/fine_v2_synthetic_benchmark.yaml"
VALIDATION_METADATA = "data/processed/fine_v2/validation/metadata.csv"
RAW_CACHE_DIR = "data/processed/fine_v2/validation_raw"
VALIDATION_ACCESS_CONFIG = "configs/data/fine_v2_validation_access.yaml"
OUT_FRAMES_DIR = Path("data/synthetic/fine_v2_validation")
OUT_AUDIT = Path("results/benchmarks/fine_v2_validation_benchmark_audit.json")
OUT_FIGURES = Path("results/benchmarks/figures")
FINAL_TEST_DIR = Path("data/processed/fine_v2/test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_checksum(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def preflight_checks() -> dict:
    """Confirm validation-only access and that final test is untouched."""

    val_cfg = yaml.safe_load(Path(VALIDATION_ACCESS_CONFIG).read_text())
    refusal_test = validate_split_access("test", val_cfg)
    if refusal_test is None:
        raise SystemExit("REFUSED: split 'test' must remain refused; aborting benchmark build")
    if FINAL_TEST_DIR.exists():
        raise SystemExit(f"REFUSED: {FINAL_TEST_DIR} exists; final test must remain untouched")

    metadata = pd.read_csv(VALIDATION_METADATA)
    if not (metadata["split"] == "validation").all():
        raise SystemExit("REFUSED: validation metadata contains non-validation rows")

    raw_dir = Path(RAW_CACHE_DIR)
    scan_ids = {Path(p).stem for p in metadata["npy"]}
    missing = [s for s in scan_ids if not (raw_dir / f"{s}.npy").exists()]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} validation raw-cache scans missing "
                         f"(run scripts/fetch_fine_v2_validation_raw_cache.py first), "
                         f"e.g. {missing[:3]}")

    return {
        "final_test_untouched": not FINAL_TEST_DIR.exists(),
        "validation_only_confirmed": True,
        "test_split_refused": True,
        "validation_metadata_rows": len(metadata),
        "validation_metadata_all_validation_split": True,
    }


def make_diagnostic_plots(metadata: pd.DataFrame, clean: np.ndarray, injected: np.ndarray,
                          out_dir: Path) -> list[str]:
    """A small, deterministic set of clean/injected spectrogram pairs -- one

    per development morphology plus the plain clean baseline, selected by a
    fixed rule (first example of each category in metadata order), never by
    visual inspection.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    picks = {
        "clean_baseline": metadata.index[0],
        "stationary": metadata.index[metadata["morphology"] == "stationary"][0],
        "linear_drift_slow": metadata.index[
            (metadata["morphology"] == "linear_drift") & (metadata["drift_hz_s"].abs() == 0.05)][0],
        "linear_drift_fast": metadata.index[
            (metadata["morphology"] == "linear_drift") & (metadata["drift_hz_s"].abs() == 10.0)][0],
        "intermittent": metadata.index[metadata["morphology"] == "intermittent"][0],
        "multiple_carriers": metadata.index[metadata["morphology"] == "multiple_carriers"][0],
    }
    paths = []
    for name, idx in picks.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        axes[0].imshow(clean[idx], aspect="auto", cmap="viridis")
        axes[0].set_title("clean")
        axes[1].imshow(injected[idx], aspect="auto", cmap="viridis")
        axes[1].set_title("injected")
        fig.suptitle(f"Fine V2 validation development benchmark: {name} "
                     f"(example_id={metadata.loc[idx, 'example_id']})")
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> int:
    preflight = preflight_checks()
    config_checksum = sha256_bytes(Path(CONFIG_PATH).read_bytes())
    print(f"preflight: {json.dumps(preflight)}")
    print(f"config checksum: {config_checksum}")

    metadata_in = pd.read_csv(VALIDATION_METADATA)

    clean_a, injected_a, meta_a = fb.build_benchmark(metadata_in, RAW_CACHE_DIR, seed=fb.SEED)
    clean_b, injected_b, meta_b = fb.build_benchmark(metadata_in, RAW_CACHE_DIR, seed=fb.SEED)

    deterministic = {
        "clean_frames_identical": bool(np.array_equal(clean_a, clean_b)),
        "injected_frames_identical": bool(np.array_equal(injected_a, injected_b)),
        "metadata_identical": meta_a.equals(meta_b),
    }
    if not all(deterministic.values()):
        raise SystemExit(f"REFUSED: benchmark regeneration is not deterministic: {deterministic}")

    clean, injected, metadata = clean_a, injected_a, meta_a
    audit = fb.audit_benchmark(clean, injected, metadata)
    audit["preflight"] = preflight
    audit["deterministic_regeneration"] = deterministic
    audit["config_checksum_sha256"] = config_checksum
    audit["clean_frames_checksum_sha256"] = array_checksum(clean)
    audit["injected_frames_checksum_sha256"] = array_checksum(injected)
    audit["metadata_checksum_sha256"] = sha256_bytes(
        metadata.to_csv(index=False).encode())

    figure_paths = make_diagnostic_plots(metadata, clean, injected, OUT_FIGURES)
    audit["figure_paths"] = figure_paths

    if not audit["all_pass"]:
        print(json.dumps(audit, indent=2, default=str))
        raise SystemExit("REFUSED: benchmark failed one or more Part G sanity checks")

    OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FRAMES_DIR / "clean_frames.npy", clean)
    np.save(OUT_FRAMES_DIR / "injected_frames.npy", injected)
    metadata.to_csv(OUT_FRAMES_DIR / "metadata.csv", index=False)

    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, default=str) + "\n")

    print(json.dumps({k: v for k, v in audit.items()
                      if k not in ("morphology_counts", "snr_counts", "drift_rate_counts",
                                   "achieved_snr_by_requested")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
