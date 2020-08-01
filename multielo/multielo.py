import numpy as np
from numpy import ndarray
from .config import defaults
from .score_functions import create_exponential_score_function


class MultiElo:
    """
    Generalized Elo for multiplayer matchups (also simplifies to standard Elo for 1-vs-1 matchups).
    Does not allow ties.
    """

    def __init__(
        self,
        k_value=defaults["K_VALUE"],
        d_value=defaults["D_VALUE"],
        score_function_base=defaults["SCORING_FUNCTION_BASE"],
        custom_score_function=None,
    ):
        """
        :param k_value: K parameter in Elo algorithm that determines how much ratings increase or decrease
        after each match
        :type k_value: float
        :param d_value: D parameter in Elo algorithm that determines how much Elo difference affects win
        probability
        :type d_value: float
        :param score_function_base: base value to use for scoring function; scores are approximately
        multiplied by this value as you improve from one place to the next (minimum allowed value is 1,
        which results in a linear scoring function)
        :type score_function_base: float
        :param custom_score_function: a function that takes an integer input and returns a numpy array
        of monotonically decreasing values summing to 1
        """
        self.k = k_value
        self.d = d_value
        self._score_func = custom_score_function or create_exponential_score_function(base=score_function_base)

    def get_new_ratings(self, initial_ratings):
        """
        Update ratings based on results. Takes an array of ratings before the matchup and returns an array with
        the updated ratings. Provided array should be ordered by the actual results (first place finisher's
        initial rating first, second place next, and so on).

        Example usage:
        >>> elo = MultiElo()
        >>> elo.get_new_ratings([1200, 1000])
        array([1207.68809835,  992.31190165])
        >>> elo.get_new_ratings([1200, 1000, 1100, 900])
        array([1212.01868209, 1012.15595083, 1087.84404917,  887.98131791])

        :param initial_ratings: array of ratings (float values) in order of actual results
        :type initial_ratings: ndarray or list

        :return: array of updated ratings (float values) in same order as input
        :rtype: ndarray
        """
        if not isinstance(initial_ratings, ndarray):
            initial_ratings = np.array(initial_ratings)
        n = len(initial_ratings)  # number of players
        actual_scores = self.get_actual_scores(n)
        expected_scores = self.get_expected_scores(initial_ratings)
        scale_factor = self.k * (n - 1)
        return initial_ratings + scale_factor * (actual_scores - expected_scores)

    def get_actual_scores(self, n):
        """
        Return the scores to be awarded to the players based on the results.

        :param n: number of players in the matchup
        :type n: int

        :return: array of length n of scores to be assigned to first place, second place, and so on
        :rtype: ndarray
        """
        scores = self._score_func(n)
        self._validate_actual_scores(scores)
        return scores

    @staticmethod
    def _validate_actual_scores(scores):
        if not np.allclose(1, sum(scores)):
            raise ValueError("scoring function does not return scores summing to 1")
        if min(scores) != 0:
            raise ValueError("scoring function does not return minimum value of 0")
        if not np.all(np.diff(scores) < 0):
            raise ValueError("scoring function does not return monotonically decreasing values")

    def get_expected_scores(self, ratings):
        """
        Get the expected scores for all players given their ratings before the matchup.

        :param ratings: array of ratings for each player in a matchup
        :type ratings: ndarray or list

        :return: array of expected scores for all players
        :rtype: ndarray
        """
        if not isinstance(ratings, ndarray):
            ratings = np.array(ratings)
        if ratings.ndim > 1:
            raise ValueError(f"ratings should be 1-dimensional array (received {ratings.ndim})")

        # get all pairwise differences
        diff_mx = ratings - ratings[:, np.newaxis]

        # get individual contributions to expected score using logistic function
        logistic_mx = 1 / (1 + 10 ** (diff_mx / self.d))
        np.fill_diagonal(logistic_mx, 0)

        # get each expected score (sum individual contributions, then scale)
        expected_scores = logistic_mx.sum(axis=1)
        n = len(ratings)
        denom = n * (n - 1) / 2  # number of individual head-to-head matchups between n players
        expected_scores = expected_scores / denom

        # this should be guaranteed, but check to make sure
        if not np.allclose(1, sum(expected_scores)):
            raise ValueError("expected scores do not sum to 1")
        return expected_scores
