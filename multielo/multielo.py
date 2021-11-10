import numpy as np
from typing import Union, List, Callable
import logging

from multielo.score_functions import create_exponential_score_function


DEFAULT_K_VALUE = 32
DEFAULT_D_VALUE = 400
DEFAULT_SCORING_FUNCTION_BASE = 1

_default_logger = logging.getLogger("multielo.multielo")


class MultiElo:
    """
    Generalized Elo for multiplayer matchups (also simplifies to standard Elo for 1-vs-1 matchups).
    Does not allow ties.
    """

    def __init__(
        self,
        k_value: float = DEFAULT_K_VALUE,
        d_value: float = DEFAULT_D_VALUE,
        score_function_base: float = DEFAULT_SCORING_FUNCTION_BASE,
        custom_score_function: Callable = None,
        log_base: int = 10,
        logger: logging.Logger = None,
    ):
        """
        :param k_value: K parameter in Elo algorithm that determines how much ratings increase or decrease
        after each match
        :param d_value: D parameter in Elo algorithm that determines how much Elo difference affects win
        probability
        :param score_function_base: base value to use for scoring function; scores are approximately
        multiplied by this value as you improve from one place to the next (minimum allowed value is 1,
        which results in a linear scoring function)
        :param custom_score_function: a function that takes an integer input and returns a numpy array
        of monotonically decreasing values summing to 1
        :param log_base: base to use for logarithms throughout the Elo algorithm. Traditionally Elo
        uses base-10 logs
        :param logger: logger to use (optional)
        """
        self.k = k_value
        self.d = d_value
        self._score_func = custom_score_function or create_exponential_score_function(base=score_function_base)
        self._log_base = log_base
        self.logger = logger or _default_logger

    def get_new_ratings(
            self,
            initial_ratings: Union[List[float], np.ndarray],
            result_order: List[int] = None,
    ) -> np.ndarray:
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
        :param result_order: list where each value indicates the place the player in the same index of
        initial_ratings finished in. Lower is better. Identify ties by entering the same value for players
        that tied. For example, [1, 2, 3] indicates that the first listed player won, the second listed player
        finished 2nd, and the third listed player finished 3rd. [1, 2, 2] would indicate that the second
        and third players tied for 2nd place. (default = range(len(initial_ratings))
        :return: array of updated ratings (float values) in same order as input
        """
        if not isinstance(initial_ratings, np.ndarray):
            initial_ratings = np.array(initial_ratings)
        n = len(initial_ratings)  # number of players
        actual_scores = self.get_actual_scores(n, result_order)
        expected_scores = self.get_expected_scores(initial_ratings)
        scale_factor = self.k * (n - 1)
        self.logger.debug(f"scale factor: {scale_factor}")
        return initial_ratings + scale_factor * (actual_scores - expected_scores)

    def get_actual_scores(self, n: int, result_order: List[int] = None) -> np.ndarray:
        """
        Return the scores to be awarded to the players based on the results.

        :param n: number of players in the matchup
        :param result_order: list indicating order of finish (see docstring for MultiElo.get_new_ratings
        for more details
        :return: array of length n of scores to be assigned to first place, second place, and so on
        """
        # calculate actual scores according to score function, then sort in order of finish
        result_order = result_order or list(range(n))
        scores = self._score_func(n)
        scores = scores[np.argsort(np.argsort(result_order))]

        # if there are ties, average the scores of all tied players
        distinct_results = set(result_order)
        if len(distinct_results) != n:
            for place in distinct_results:
                idx = [i for i, x in enumerate(result_order) if x == place]
                scores[idx] = scores[idx].mean()

        self._validate_actual_scores(scores, result_order)
        self.logger.debug(f"calculated actual scores: {scores}")
        return scores

    @staticmethod
    def _validate_actual_scores(scores: np.ndarray, result_order: List[int]):
        if not np.allclose(1, sum(scores)):
            raise ValueError("scoring function does not return scores summing to 1")
        if min(scores) != 0:
            # tie for last place means minimum score doesn't have to be zero,
            # so only raise error if there isn't a tie for last place
            last_place = max(result_order)
            if result_order.count(last_place) == 1:
                raise ValueError("scoring function does not return minimum value of 0")
        if not np.all(np.diff(scores[np.argsort(result_order)]) <= 0):
            raise ValueError("scoring function does not return monotonically decreasing values")

    def get_expected_scores(self, ratings: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Get the expected scores for all players given their ratings before the matchup.

        :param ratings: array of ratings for each player in a matchup
        :return: array of expected scores for all players
        """
        self.logger.debug(f"computing expected scores for {ratings}")
        if not isinstance(ratings, np.ndarray):
            ratings = np.array(ratings)
        if ratings.ndim > 1:
            raise ValueError(f"ratings should be 1-dimensional array (received {ratings.ndim})")

        # get all pairwise differences
        diff_mx = ratings - ratings[:, np.newaxis]
        self.logger.debug(f"diff_mx = \n{diff_mx}")

        # get individual contributions to expected score using logistic function
        logistic_mx = 1 / (1 + self._log_base ** (diff_mx / self.d))
        np.fill_diagonal(logistic_mx, 0)
        self.logger.debug(f"logistic_mx = \n{logistic_mx}")

        # get each expected score (sum individual contributions, then scale)
        expected_scores = logistic_mx.sum(axis=1)
        n = len(ratings)
        denom = n * (n - 1) / 2  # number of individual head-to-head matchups between n players
        expected_scores = expected_scores / denom

        # this should be guaranteed, but check to make sure
        if not np.allclose(1, sum(expected_scores)):
            raise ValueError("expected scores do not sum to 1")
        self.logger.debug(f"calculated expected scores: {expected_scores}")
        return expected_scores

    def simulate_win_probabilities(
            self,
            ratings: Union[List[float], np.ndarray],
            n_sim: int = int(1e5),
            seed: int = None,
    ) -> np.ndarray:
        """
        Estimate the probability of each player finishing in each possible
        place using a simulation. Returns a matrix where (i, j) values are the
        probability that player i finishes in place j.

        To simulate a game including players in the
        ratings array, we generate a score for each player using a Gumbel
        distribution. If a player has rating R, then that player's score is
        sampled from a Gumbel(R, D) distribution, where D is the Elo D
        parameter. Then we rank the players in descending order of their
        scores to determine first place, second place, ..., last place. We
        count the number of times each player finishes in each place and then
        divide by the number of simulations to calculate the proportions.

        We generate scores using Gumbel distributions because of the property:
        ~~    Gumbel(a_1, b) - Gumbel(a_2, b) ~ Logistic(a_1 - a_2, b)    ~~

        The Logistic(a_1 - a_2, b) distribution is the same distribution that
        describes the pairwise win probability if two payers have Elo ratings
        a_1 and a_2. In other words, a score sampled from Gumbel(a_1, b) will
        be greater than a score sampled from Gumbel(a_2, b) with the same
        probability that a player with Elo rating a_1 will beat a player with
        Elo rating a_2 in a 1-on-1 matchup.

        :param ratings: array of ratings of the players involved
        :param n_sim: number of simulations to run
        :param seed: (optional) seed for random number generation

        :return: matrix (a numpy array) where (i, j) values are the probability
        that player i finishes in place j
        """
        if seed is not None:
            np.random.seed(seed)

        # sort so we always get the same result for same distinct ratings, but
        # keep track of original order
        idx = np.argsort(ratings)
        ratings = sorted(ratings)

        # simulate n_sim scores for each player from Gumbel distributions
        n_players = len(ratings)
        n_sim = int(n_sim)
        scores = np.zeros((n_players, n_sim))
        self.logger.debug(f"simulating {n_sim:,} scores for each player")
        for i, rating in enumerate(ratings):
            scores[idx[i], :] = _gumbel_sample(
                loc=rating,
                scale=self.d,
                size=int(n_sim),
                base=self._log_base
            )
            self.logger.debug(f"finished sampling {n_sim:,} scores for player {i+1} of {n_players}")

        # use the scores to decide the order of finish (highest score wins) and
        # create matrix with proportion of times each player finishes in each place
        result_mx = self._convert_scores_to_result_proportions(scores)
        self.logger.debug(f"finished simulation")
        return result_mx

    @staticmethod
    def _convert_scores_to_result_proportions(scores: np.ndarray) -> np.ndarray:
        """
        Take an array of scores with one row per player and one column per
        simulation, and return a matrix with one row per player and one column
        per place. Each (row, col) value in the returned matrix is the count of
        times player "row" finished in place "col".
        """
        # sort scores from high to low for each simulation
        results = np.argsort(-scores, axis=0)

        # put it into a matrix where row = player, column = place, value = count
        # of times player finished in place
        n = scores.shape[0]
        count_mx = np.zeros((n, n))
        for i, x in enumerate(results):
            counts = np.bincount(x, minlength=n)
            count_mx[:, i] = counts

        proportion_mx = count_mx / scores.shape[1]
        return proportion_mx


def _gumbel_sample(
        loc: float,
        scale: float,
        size: int = 1,
        base: float = np.exp(1),
) -> np.ndarray:
    """
    Sample from a Gumbel distribution (optionally with a different log base).

    :param loc: location parameter for distribution
    :param scale: scale parameter for distribution (> 0)
    :param size: number of samples to draw
    :param base: base for logarithm (defaults to natural log)

    :return: sample(s) from Gumbel distribution
    """
    if scale <= 0:
        raise ValueError("scale parameter for Gumbel distribution must be > 0")
    p = np.random.rand(int(size))
    return loc - scale * _log(-_log(p, base=base), base=base)


def _log(x, base=np.exp(1)):
    return np.log(x) / np.log(base)
