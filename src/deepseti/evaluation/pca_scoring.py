"""Scoring, metrics and grouped-bootstrap utilities for the Fine V2 PCA baseline.

Two predeclared anomaly-score aggregations (not tuned against benchmark
labels):

1. ``mean_score`` -- whole-frame mean squared reconstruction error.
2. ``top_fraction_score`` -- mean squared error of the top ``TOP_FRACTION``
   (0.2%) of pixels by residual magnitude, a sparse/local anomaly score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_FRACTION = 0.002


def residual_squared(frames: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    return (frames - reconstructed) ** 2


def mean_score(residual_sq: np.ndarray) -> np.ndarray:
    return residual_sq.reshape(residual_sq.shape[0], -1).mean(axis=1)


def top_fraction_score(residual_sq: np.ndarray, top_fraction: float = TOP_FRACTION) -> np.ndarray:
    flat = residual_sq.reshape(residual_sq.shape[0], -1)
    k = max(1, round(top_fraction * flat.shape[1]))
    top = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1)


def auroc_auprc(clean_scores: np.ndarray, injected_scores: np.ndarray) -> dict:
    """AUROC/AUPRC for the clean (label 0) vs injected (label 1) score pools."""

    scores = np.concatenate([clean_scores, injected_scores])
    labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(injected_scores))])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def paired_success_fraction(clean_scores: np.ndarray, injected_scores: np.ndarray) -> float:
    """Fraction of paired examples where the injected score exceeds the clean score."""

    return float(np.mean(injected_scores > clean_scores))


def grouped_bootstrap_ci(clean_scores: np.ndarray, injected_scores: np.ndarray,
                         groups: np.ndarray, metric_fn, n_boot: int = 1000,
                         seed: int = 20261013, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for ``metric_fn(clean, injected)``, resampling by

    group (e.g. cadence ``group_id``) rather than by individual example, so
    paired examples sharing a base observation are not treated as
    independent draws.
    """

    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[g] for g in chosen])
        stats.append(metric_fn(clean_scores[idx], injected_scores[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
