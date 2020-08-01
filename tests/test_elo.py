import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multielo import MultiElo
import numpy as np


def test_elo_defaults():
    # test if defaults are specified in config.py file
    try:
        from multielo.config import defaults
    except ImportError:
        assert False, "could not import defaults"

    # test that the expected keys are in defaults
    expected_keys = [
        "INITIAL_RATING",
        "K_VALUE",
        "D_VALUE",
        "SCORING_FUNCTION_BASE",
    ]
    for key in expected_keys:
        assert key in defaults, f"{key} is not included in defaults"
    for key in defaults:
        assert key in expected_keys, f"{key} is in defaults but is not one of the expected keys"


def test_elo_changes():
    # test a couple known changes in elo values
    elo = MultiElo(k_value=32, d_value=400)
    ratings = np.array([1000, 1000])
    assert np.allclose(elo.get_expected_scores(ratings), [0.5, 0.5])
    assert np.allclose(elo.get_new_ratings(ratings), [1016, 984])
    ratings_2 = [1200, 1000]  # should also work with a list
    assert np.allclose(elo.get_expected_scores(ratings_2), [0.75974693, 0.24025307])
    assert np.allclose(elo.get_new_ratings(ratings_2), [1207.68809835, 992.31190165])
    assert np.allclose(elo.get_expected_scores(ratings_2[::-1]), [0.24025307, 0.75974693])
    assert np.allclose(elo.get_new_ratings(ratings_2[::-1]), [1024.31190165, 1175.68809835])
    ratings_3 = np.array([1200, 800])
    assert np.allclose(elo.get_expected_scores(ratings_3), [0.90909091, 0.09090909])
    assert np.allclose(elo.get_new_ratings(ratings_3), [1202.90909091, 797.09090909])


def test_zero_sum():
    # make sure expected scores sum to 1 and rating changes are zero sum
    for n_players in [2, 3, 4, 10]:
        for _ in range(10):
            k = np.random.uniform(16, 64)
            d = np.random.uniform(200, 800)
            ratings = np.random.uniform(600, 1400, size=n_players)
            elo = MultiElo(k_value=k, d_value=d)
            assert np.allclose(elo.get_expected_scores(ratings).sum(), 1), \
                f"expected ratings do not sum to 1 for k={k}, d={d}, ratings={ratings}"
            assert np.allclose(elo.get_new_ratings(ratings).sum(), ratings.sum()), \
                f"rating changes are not zero sum for k={k}, d={d}, ratings={ratings}"
