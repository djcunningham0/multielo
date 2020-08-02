import numpy as np
from numpy import ndarray


def linear_score_function(n: int) -> ndarray:
    """
    With the linear score function the "points" awarded scale linearly from first place
    through last place. For example, improving from 2nd to 1st place has the same sized
    benefit as improving from 5th to 4th place.

    :param n: number of players
    :return: array of the points to assign to each place (summing to 1)
    """
    return np.array([(n - p) / (n * (n - 1) / 2) for p in range(1, n + 1)])


def create_exponential_score_function(base: float):
    """
    With an exponential score function with base > 1, more points are awarded to the top
    finishers and the point distribution is flatter at the bottom. For example, improving
    from 2nd to 1st place is more valuable than improving from 5th place to 4th place. A
    larger base value means the scores will be more weighted towards the top finishers.

    :param base: base for teh exponential score function (> 1)
    :return: a function that takes parameter n for number of players and returns an array
    of the points to assign to each place (summing to 1)
    """
    return lambda n: _exponential_score_template(n, base)


def _exponential_score_template(n: int, base: float) -> ndarray:
    if base < 1:
        raise ValueError("base must be >= 1")
    if base == 1:
        return linear_score_function(n)  # it converges to this as base -> 1

    out = np.array([base ** (n - p) - 1 for p in range(1, n + 1)])
    return out / sum(out)
