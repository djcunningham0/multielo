import pytest
from multielo import MultiElo
import numpy as np

from typing import List


@pytest.mark.parametrize(
    "k, d, s, ratings, true_expected, true_new",
    [
        (32, 400, 1, np.array([1000, 1000]), [0.5, 0.5], [1016, 984]),
        (32, 400, 1, [1200, 1000], [0.75974693, 0.24025307], [1207.68809835, 992.31190165]),
        (32, 400, 1, [1000, 1200], [0.24025307, 0.75974693], [1024.31190165, 1175.68809835]),
        (32, 400, 1, np.array([1200, 800]), [0.90909091, 0.09090909], [1202.90909091, 797.09090909]),
        (64, 400, 1, [1200, 1000], [0.75974693, 0.24025307], [1215.37619669,  984.62380331]),
        (64, 800, 1, [1200, 1000], [0.640065, 0.359935], [1223.03584001,  976.96415999]),
        (32, 800, 1, [1200, 1000], [0.640065, 0.359935], [1211.51792001,  988.48207999]),
        (32, 200, 1, [1200, 1000], [0.90909091, 0.09090909], [1202.90909091,  997.09090909]),
        (32, 400, 1.5, [1200, 1000], [0.75974693, 0.24025307], [1207.68809835, 992.31190165]),
        (32, 400, 1, [1200, 1000, 900], [0.53625579, 0.29343936, 0.17030485],
         [1208.34629612, 1002.55321444, 889.10048944]),
        (32, 400, 1, [1000, 1200, 900], [0.29343936, 0.53625579, 0.17030485],
         [1023.88654777, 1187.01296279, 889.10048944]),
        (32, 400, 1.25, [1200, 1000, 900], [0.53625579, 0.29343936, 0.17030485],
         [1209.98732176, 1000.9121888, 889.10048944]),
        (32, 400, 1.5, [1200, 1000, 900], [0.53625579, 0.29343936, 0.17030485],
         [1211.39391517, 999.50559539, 889.10048944]),
        (32, 400, 2, [1200, 1000, 900], [0.53625579, 0.29343936, 0.17030485],
         [1213.67962945, 997.21988111, 889.10048944]),
        (32, 400, 1.25, [1200, 1000, 900, 1050], [0.38535873, 0.21814249, 0.13458826, 0.26191052],
         [1214.82857088, 1009.6423915, 900.67244749, 1024.85659012]),
    ]
)
def test_elo_changes(k, d, s, ratings, true_expected, true_new):
    """
    Test some known values to make sure Elo is calculating the correct updates.
    """
    elo = MultiElo(k_value=k, d_value=d, score_function_base=s)
    assert np.allclose(elo.get_expected_scores(ratings), true_expected)
    assert np.allclose(elo.get_new_ratings(ratings), true_new)


def test_zero_sum():
    """
    make sure expected scores sum to 1 and rating changes are zero sum
    """
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


@pytest.mark.parametrize(
    "results, result_order, new_ratings",
    [
        ([1000, 1000], [1, 1], [1000, 1000]),
        ([1200, 1000], [0, 0], [1191.6880983472654, 1008.3119016527346]),
        ([1200, 1000, 800], [1, 2, 2], [1207.06479284,  989.33333333,  803.60187383]),
        ([1200, 1000, 800], [1, 1, 2], [1196.39812617, 1010.66666667,  792.93520716]),
        ([1200, 1000, 800], [1, 1, 1], [1185.7314595, 1000,  814.2685405]),
    ]
)
def test_ties(results: List[float], result_order: List[int], new_ratings: List[float]):
    elo = MultiElo(k_value=32, d_value=400, score_function_base=1)
    assert np.allclose(elo.get_new_ratings(results, result_order=result_order), new_ratings)


def test_out_of_order_ratings():
    """If we reverse the change the order of ratings and account for it in result_order,
    the new ratings should be the same"""
    elo = MultiElo()
    result_1 = elo.get_new_ratings([1200, 1000])
    result_2 = elo.get_new_ratings([1000, 1200], result_order=[2, 1])
    print(f"result_1: {result_1}")
    print(f"result_2: {result_2}")
    assert result_1[0] == result_2[1]
    assert result_1[1] == result_2[0]

    result_1 = elo.get_new_ratings([1200, 1000, 800], result_order=[1, 2, 2])
    result_2 = elo.get_new_ratings([1000, 800, 1200], result_order=[2, 2, 1])
    print(f"result_1: {result_1}")
    print(f"result_2: {result_2}")
    assert result_1[0] == result_2[2]
    assert result_1[1] == result_2[0]
    assert result_1[2] == result_2[1]
