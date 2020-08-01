import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from multielo.score_functions import linear_score_function, create_exponential_score_function


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
