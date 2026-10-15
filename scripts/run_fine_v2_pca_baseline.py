"""Fine V2 PCA baseline: fit on TRAIN (normal-only), score the VALIDATION

synthetic development benchmark.

Fits a scalable IncrementalPCA (64 components, predeclared) on the 15,120
frozen Fine V2 TRAIN frames, streamed in batches so the full
15,120 x 65,536 matrix is never held in memory at once. Synthetic examples
are never used to fit PCA. Scores the benchmark's paired clean/injected
VALIDATION frames with two predeclared aggregations (whole-frame mean
squared error, and the mean of the top 0.2% largest-residual pixels),
reports AUROC/AUPRC, paired success rate, grouped-bootstrap confidence
intervals (grouped by cadence ``group_id``), and breakdowns by SNR,
morphology and drift rate. Writes the required figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepseti.evaluation import pca_scoring as ps

TRAIN_ROOT = Path("data/processed/fine_v2/train")
BENCHMARK_DIR = Path("data/synthetic/fine_v2_validation")
N_COMPONENTS = 64
BATCH_SCANS = 20  # 20 scans * 42 frames/scan = 840 frames/batch, well above n_components
OUT_RESULTS = Path("results/benchmarks/fine_v2_pca_baseline_results.json")
OUT_FIGURES = Path("results/benchmarks/figures")
SEED = 20261013


def iter_train_batches(train_root: Path, batch_scans: int = BATCH_SCANS):
    metadata = pd.read_csv(train_root / "metadata.csv")
    scan_npy_paths = sorted(metadata["npy"].unique())
    for i in range(0, len(scan_npy_paths), batch_scans):
        batch_paths = scan_npy_paths[i:i + batch_scans]
        arrays = [np.load(train_root / p) for p in batch_paths]
        batch = np.concatenate(arrays, axis=0)
        yield batch.reshape(batch.shape[0], -1).astype(np.float32)


def fit_train_pca(train_root: Path, n_components: int = N_COMPONENTS) -> tuple[IncrementalPCA, float, int]:
    model = IncrementalPCA(n_components=n_components)
    t0 = time.time()
    n_frames = 0
    for batch in iter_train_batches(train_root):
        model.partial_fit(batch)
        n_frames += batch.shape[0]
    return model, time.time() - t0, n_frames


def score_frames(model: IncrementalPCA, frames: np.ndarray) -> dict[str, np.ndarray]:
    flat = frames.reshape(frames.shape[0], -1).astype(np.float32)
    reconstructed = model.inverse_transform(model.transform(flat)).reshape(frames.shape)
    residual_sq = ps.residual_squared(frames, reconstructed)
    return {
        "mean": ps.mean_score(residual_sq),
        "top_fraction": ps.top_fraction_score(residual_sq),
        "reconstructed": reconstructed,
    }


def breakdown_by(metadata: pd.DataFrame, clean_scores: np.ndarray, injected_scores: np.ndarray,
                 key: str, transform=None) -> dict:
    values = metadata[key] if transform is None else metadata[key].map(transform)
    out = {}
    for v in sorted(values.dropna().unique(), key=str):
        idx = (values == v).to_numpy()
        if idx.sum() == 0:
            continue
        result = ps.auroc_auprc(clean_scores[idx], injected_scores[idx])
        result["n"] = int(idx.sum())
        result["paired_success_fraction"] = ps.paired_success_fraction(
            clean_scores[idx], injected_scores[idx])
        out[str(v)] = result
    return out


def make_figures(metadata: pd.DataFrame, clean: dict, injected: dict,
                 clean_frames: np.ndarray, injected_frames: np.ndarray,
                 by_snr: dict, by_morphology: dict, by_drift: dict,
                 summary: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("mean", "top_fraction"),
                              ("whole-frame mean score", "top-0.2% local score"), strict=True):
        ax.hist(clean[key], bins=40, alpha=0.6, label="clean", density=True)
        ax.hist(injected[key], bins=40, alpha=0.6, label="injected", density=True)
        ax.set_title(f"Fine V2 validation development benchmark: {title}")
        ax.legend()
    fig.tight_layout()
    p = out_dir / "score_distributions.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["mean score", "top-0.2% score"]
    aurocs = [summary["mean_score"]["auroc"], summary["top_fraction_score"]["auroc"]]
    auprcs = [summary["mean_score"]["auprc"], summary["top_fraction_score"]["auprc"]]
    x = np.arange(2)
    ax.bar(x - 0.15, aurocs, width=0.3, label="AUROC")
    ax.bar(x + 0.15, auprcs, width=0.3, label="AUPRC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fine V2 validation development benchmark: AUROC/AUPRC")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "auroc_auprc_summary.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    snr_levels = sorted(float(k) for k in by_snr["mean"])
    ax.plot(snr_levels, [by_snr["mean"][str(k)]["auroc"] for k in snr_levels], "o-", label="mean score")
    ax.plot(snr_levels, [by_snr["top_fraction"][str(k)]["auroc"] for k in snr_levels], "s-",
           label="top-0.2% score")
    ax.set_xscale("log")
    ax.set_xlabel("requested SNR")
    ax.set_ylabel("AUROC")
    ax.set_title("Fine V2 validation development benchmark: AUROC vs SNR")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "auroc_vs_snr.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    drift_levels = sorted(float(k) for k in by_drift["mean"])
    ax.plot(drift_levels, [by_drift["mean"][str(k)]["auroc"] for k in drift_levels], "o-",
           label="mean score")
    ax.plot(drift_levels, [by_drift["top_fraction"][str(k)]["auroc"] for k in drift_levels], "s-",
           label="top-0.2% score")
    ax.set_xscale("log")
    ax.set_xlabel("|drift rate| (Hz/s)")
    ax.set_ylabel("AUROC")
    ax.set_title("Fine V2 validation development benchmark: linear-drift AUROC vs drift rate")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "auroc_vs_drift_rate.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4))
    morphs = list(by_morphology["mean"])
    xm = np.arange(len(morphs))
    ax.bar(xm - 0.15, [by_morphology["mean"][m]["auroc"] for m in morphs], width=0.3, label="mean score")
    ax.bar(xm + 0.15, [by_morphology["top_fraction"][m]["auroc"] for m in morphs], width=0.3,
          label="top-0.2% score")
    ax.set_xticks(xm, morphs, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Fine V2 validation development benchmark: AUROC by morphology")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "auroc_by_morphology.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

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
    for name, idx in picks.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
        axes[0].imshow(clean_frames[idx], aspect="auto", cmap="viridis")
        axes[0].set_title("clean")
        axes[1].imshow(injected_frames[idx], aspect="auto", cmap="viridis")
        axes[1].set_title("injected")
        residual = (injected_frames[idx] - injected["reconstructed"][idx]) ** 2
        axes[2].imshow(residual, aspect="auto", cmap="inferno")
        axes[2].set_title("PCA residual (injected)")
        fig.suptitle(f"Fine V2 validation development benchmark: {name}")
        fig.tight_layout()
        p = out_dir / f"residual_example_{name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))

    return paths


def main() -> int:
    model, fit_seconds, n_train_frames = fit_train_pca(TRAIN_ROOT)
    explained_variance_ratio_sum = float(model.explained_variance_ratio_.sum())

    clean_frames = np.load(BENCHMARK_DIR / "clean_frames.npy")
    injected_frames = np.load(BENCHMARK_DIR / "injected_frames.npy")
    metadata = pd.read_csv(BENCHMARK_DIR / "metadata.csv")

    clean = score_frames(model, clean_frames)
    injected = score_frames(model, injected_frames)
    groups = metadata["group_id"].to_numpy()

    summary = {}
    for key, label in (("mean", "mean_score"), ("top_fraction", "top_fraction_score")):
        auc = ps.auroc_auprc(clean[key], injected[key])
        ci = ps.grouped_bootstrap_ci(clean[key], injected[key], groups,
                                     metric_fn=lambda c, i: ps.auroc_auprc(c, i)["auroc"],
                                     seed=SEED)
        summary[label] = {
            **auc,
            "auroc_grouped_bootstrap_ci95": ci,
            "paired_success_fraction": ps.paired_success_fraction(clean[key], injected[key]),
            "score_delta_mean": float(np.mean(injected[key] - clean[key])),
            "score_delta_median": float(np.median(injected[key] - clean[key])),
        }

    by_snr = {key: breakdown_by(metadata, clean[key], injected[key], "requested_snr")
             for key in ("mean", "top_fraction")}
    by_morphology = {key: breakdown_by(metadata, clean[key], injected[key], "morphology")
                     for key in ("mean", "top_fraction")}
    drift_meta = metadata[metadata["morphology"] == "linear_drift"]
    drift_idx = drift_meta.index.to_numpy()
    by_drift = {key: breakdown_by(drift_meta, clean[key][drift_idx], injected[key][drift_idx],
                                  "drift_hz_s", transform=abs)
               for key in ("mean", "top_fraction")}

    figures = make_figures(metadata, clean, injected, clean_frames, injected_frames,
                           by_snr, by_morphology, by_drift, summary, OUT_FIGURES)

    results = {
        "pca": {
            "n_components": N_COMPONENTS,
            "fit_seconds": fit_seconds,
            "n_train_frames": n_train_frames,
            "explained_variance_ratio_sum": explained_variance_ratio_sum,
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        },
        "top_fraction": ps.TOP_FRACTION,
        "n_benchmark_examples": len(metadata),
        "summary": summary,
        "by_requested_snr": by_snr,
        "by_morphology": by_morphology,
        "by_abs_drift_hz_s": by_drift,
        "figure_paths": figures,
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("by_requested_snr", "by_morphology", "by_abs_drift_hz_s")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
