import pytest
import numpy as np
from multielo.score_functions import linear_score_function, create_exponential_score_function
from multielo import MultiElo

from typing import List


def test_2_player_score_function():
    # should always return [1, 0]
    linear_scores = linear_score_function(2)
    assert np.allclose(linear_scores, np.array([1, 0])), \
        f"linear score function does not return [1, 0] for 2-player matchup: {linear_scores}"

    for base in np.random.uniform(1, 3, 10).tolist():
        func = create_exponential_score_function(base)
        exp_scores = func(2)
        assert np.allclose(exp_scores, np.array([1, 0])), \
            f"exponential score function does not return [1, 0] for 2-player matchup for base={base}: {exp_scores}"


def test_linear_score_function():
    for n in range(3, 11):
        scores = linear_score_function(n)
        score_diffs = np.diff(scores)
        assert np.allclose(scores.sum(), 1), f"linear score function does not sum to 1 for n={n}: {scores}"
        assert np.allclose(scores.min(), 0), \
            f"linear score function does not have minimum score of 0 for n={n}: {scores}"
        assert np.all(score_diffs < 0), \
            f"linear score function is not monotonically decreasing for n={n}: {scores}"


def test_exponential_score_function():
    for base in np.random.uniform(1, 3, 10).tolist():
        for n in range(3, 11):
            func = create_exponential_score_function(base)
            scores = func(n)
            score_diffs = np.diff(scores)
            assert np.allclose(scores.sum(), 1), \
                f"exponential score function does not sum to 1 for base={base}, n={n}"
            assert np.allclose(scores.min(), 0), \
                f"exponential score function does not have minimum score of 0 for n={n}, base={base}: {scores}"
            assert np.all(score_diffs < 0), \
                f"exponential score function is not monotonically decreasing for base={base}, n={n}: {scores}"
            # differences should also be monotonically increasing (less negative) for exponential score function
            assert np.all(np.diff(score_diffs) > 0), \
                f"exponential score function diffs are not monotonically increasing for base={base}, n={n}: {scores}"


@pytest.mark.parametrize(
    "result_order, actual_scores, base",
    [
        ([1, 1], [0.5, 0.5], 1),
        ([1, 1], [0.5, 0.5], 2),
        ([1, 1, 1], [1/3, 1/3, 1/3], 1),
        ([1, 2, 2], [2/3, 1/6, 1/6], 1),
        ([1, 1, 2], [0.5, 0.5, 0], 1),
        ([1, 1, 1], [1/3, 1/3, 1/3], 2),
        ([1, 2, 2], [0.75, 0.125, 0.125], 2),
        ([1, 1, 2], [0.5, 0.5, 0], 2),
    ]
)
def test_tie_actual_scores(
        result_order: List[int],
        actual_scores: List[float],
        base: float,
):
    elo = MultiElo(score_function_base=base)
    n = len(result_order)
    scores = elo.get_actual_scores(n=n, result_order=result_order)
    assert np.allclose(scores, actual_scores)
