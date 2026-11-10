"""Rank direction, small-n permutation inference, ties, and analysis gates."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rank_hou_sampling import correlations, exact_statistics, interval, load_inputs, sampling_draws


def test_small_series_perfect_order_does_not_imply_small_exact_p():
    result = exact_statistics([-30, -22, -21, -19], [-12, -10, -9, -8])
    assert result["spearman_rho"] == pytest.approx(1)
    assert result["kendall_tau_b"] == pytest.approx(1)
    assert result["spearman_p_exact_two_sided"] == pytest.approx(2 / 24)
    assert result["kendall_p_exact_two_sided"] == pytest.approx(2 / 24)


def test_rank_direction_and_seven_candidate_pair_count():
    rho, tau = correlations([2, 3, 4, 5, 7, 6, 1], [1, 4, 5, 6, 7, 3, 2])
    assert rho == pytest.approx(.75)
    assert tau == pytest.approx((17 - 4) / 21)


@pytest.mark.parametrize("x,y", [([1, 1, 3, 4], [2, 2, 4, 3]),
                                 ([1, 2, 2, 4], [1, 3, 4, 4])])
def test_candidate_bootstrap_ties_match_scipy(x, y):
    rho, tau = correlations(x, y)
    assert rho == pytest.approx(stats.spearmanr(x, y).statistic)
    assert tau == pytest.approx(stats.kendalltau(x, y).statistic)


def test_undefined_candidate_bootstrap_draws_are_counted():
    rho, _ = correlations([[1, 1, 1], [1, 2, 3]], [[2, 2, 2], [1, 2, 3]])
    result = interval(rho)
    assert result["defined_draws"] == 1 and result["undefined_draws"] == 1


def test_block_resampling_preserves_constant_scores_at_all_block_lengths():
    blocks = np.broadcast_to(np.arange(7)[:, None, None], (7, 6, 10))
    for block_ns in (1, 2, 5):
        result = sampling_draws(blocks, np.random.default_rng(12), 50, block_ns)
        assert result.shape == (50, 7)
        np.testing.assert_array_equal(result, np.broadcast_to(np.arange(7), (50, 7)))


def test_partial_sampling_cannot_be_ranked_as_a_seven_sample_result(tmp_path):
    (tmp_path / "audit.json").write_text(json.dumps(dict(n_frames=42000, full_window_usable_replicas=39)))
    with pytest.raises(RuntimeError, match="all seven"):
        load_inputs(tmp_path, tmp_path)
