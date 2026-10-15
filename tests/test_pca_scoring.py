import numpy as np
import pytest

from deepseti.evaluation import pca_scoring as ps


def test_mean_score_is_whole_frame_mse():
    residual_sq = np.full((3, 4, 5), 2.0)
    scores = ps.mean_score(residual_sq)
    assert np.allclose(scores, 2.0)


def test_top_fraction_score_uses_only_largest_pixels():
    residual_sq = np.zeros((1, 100))
    residual_sq[0, :5] = 9.0  # top 5% are large, rest are zero
    scores = ps.top_fraction_score(residual_sq, top_fraction=0.05)
    assert scores[0] == pytest.approx(9.0)
    # A larger fraction must dilute the score with the zero background.
    scores_wide = ps.top_fraction_score(residual_sq, top_fraction=0.5)
    assert scores_wide[0] < scores[0]


def test_top_fraction_score_matches_manual_topk():
    rng = np.random.default_rng(0)
    residual_sq = rng.random((4, 1000))
    k = round(0.01 * 1000)
    expected = np.sort(residual_sq, axis=1)[:, -k:].mean(axis=1)
    got = ps.top_fraction_score(residual_sq, top_fraction=0.01)
    assert np.allclose(got, expected)


def test_auroc_auprc_perfect_separation():
    clean = np.array([0.1, 0.2, 0.15])
    injected = np.array([0.9, 0.95, 0.99])
    result = ps.auroc_auprc(clean, injected)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)


def test_auroc_auprc_random_separation_near_half():
    rng = np.random.default_rng(1)
    clean = rng.normal(size=2000)
    injected = rng.normal(size=2000)
    result = ps.auroc_auprc(clean, injected)
    assert 0.4 < result["auroc"] < 0.6


def test_paired_success_fraction():
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    injected = np.array([2.0, 1.0, 4.0, 5.0])  # wins at idx 0,2,3 -> 3/4
    assert ps.paired_success_fraction(clean, injected) == pytest.approx(0.75)


def test_grouped_bootstrap_ci_contains_point_estimate_for_perfect_separation():
    clean = np.zeros(60)
    injected = np.ones(60)
    groups = np.repeat(np.arange(15), 4)
    lo, hi = ps.grouped_bootstrap_ci(
        clean, injected, groups,
        metric_fn=lambda c, i: ps.auroc_auprc(c, i)["auroc"],
        n_boot=200, seed=0,
    )
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_grouped_bootstrap_ci_widens_with_group_variability():
    rng = np.random.default_rng(2)
    n_groups = 15
    groups = np.repeat(np.arange(n_groups), 10)
    # Make a few groups have much weaker separation than the rest.
    clean = rng.normal(size=len(groups))
    injected = rng.normal(size=len(groups)) + 2.0
    weak = np.isin(groups, [0, 1])
    injected[weak] = clean[weak] + rng.normal(scale=0.1, size=weak.sum())

    lo, hi = ps.grouped_bootstrap_ci(
        clean, injected, groups,
        metric_fn=lambda c, i: ps.auroc_auprc(c, i)["auroc"],
        n_boot=300, seed=1,
    )
    assert lo < hi
    assert 0.0 <= lo <= hi <= 1.0
