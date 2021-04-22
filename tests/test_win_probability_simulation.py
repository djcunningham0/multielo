import pytest
import multielo
import numpy as np
import warnings


@pytest.mark.parametrize(
    "scores, true_counts",
    [
        (
            np.array([
                [100, 100, 100, 100, 100, 100],
                [110, 90,  110, 90,  110, 90 ],
                [85,  95,  105, 85,  95,  105],
            ]),
            np.array([
                [2, 3, 1],
                [3, 1, 2],
                [1, 2, 3]
            ])
        ),
        (
            np.array([
                [100, 100, 100, 100, 100, 100],
                [110, 90,  110, 90,  110, 90 ],
                [85,  95,  105, 85,  95,  105],
                [0,   0,   0,   0,   0,   0  ],
            ]),
            np.array([
                [2, 3, 1, 0],
                [3, 1, 2, 0],
                [1, 2, 3, 0],
                [0, 0, 0, 6],
            ])
        ),
        (
            np.array([
                [100, 100, 100, 100, 100, 100, 100],
                [110, 101, 110, 101, 110, 101, 110],
                [85,  95,  105, 85,  95,  105, 85 ],
            ]),
            np.array([
                [0, 5, 2],
                [6, 1, 0],
                [1, 1, 5]
            ])
        ),
    ]
)
def test_scores_to_result_proportions(scores: np.ndarray, true_counts: np.ndarray):
    """
    Provide some score arrays and make sure MultiElo calculates the correct
    result proportions.
    """
    scores_to_proportions = multielo.MultiElo._convert_scores_to_result_proportions
    n_scores_per_player = scores.shape[1]
    true_proportions = true_counts / n_scores_per_player
    assert np.all(scores_to_proportions(scores) == true_proportions)


def test_invalid_scores_to_result_proportions():
    # inconsistent lengths
    with pytest.raises(Exception):
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        scores = np.array([
            [100, 90, 100],
            [95, 105],
            [80, 120, 80],
        ])
        multielo.MultiElo._convert_scores_to_result_proportions(scores)
