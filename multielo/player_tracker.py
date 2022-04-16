import numpy as np
import pandas as pd
from typing import Union, List, Tuple
import logging
import warnings
import pickle

from multielo.multielo import MultiElo


DEFAULT_INITIAL_RATING = 1000

logger = logging.getLogger("multielo.player_tracker")
warnings.simplefilter("once", DeprecationWarning)


class Player:
    """
    The Player object stores the current and historical ratings of an individual player (or team, etc.). Attributes of
    interest are Player.rating (the current rating) and Player.rating_history (a list of historical ratings).
    """

    def __init__(
            self,
            player_id: str,
            rating: float = DEFAULT_INITIAL_RATING,
            rating_history: List[Tuple[Union[str, float], float]] = None,
            date: Union[str, float] = None,
    ):
        """
        Instantiate a player.

        :param player_id: player ID (e.g., the player's name)
        :param rating: player's current rating
        :param rating_history: history of player's ratings (each entry is a (date, rating) tuple); if None, create
        the first entry in the player's history
        :param date: date of this rating (e.g., player's first matchup date)
        """
        self.id = player_id
        self.rating = rating
        if rating_history is None:
            self.rating_history = []
            self._update_rating_history(rating, date)
        else:
            self.rating_history = rating_history
        logger.info(f"created player with ID {player_id} and rating {rating}")

    def update_rating(self, new_rating: float, date: Union[str, float], keep_history: bool = True):
        """
        Update a player's rating and rating history. (updates the self.rating and self.rating_history attributes)

        :param new_rating: player's new rating
        :param date: date the new rating was achieved
        :param keep_history: if True, add an entry to the player's rating history
        """
        logger.info(f"Updating rating for {self.id}: {self.rating:.3f} --> {new_rating:.3f}")
        self.rating = new_rating
        self._update_rating_history(rating=new_rating, date=date, keep_history=keep_history)

    def get_rating_as_of_date(
        self,
        date: Union[str, float],
        default_rating: float = DEFAULT_INITIAL_RATING
    ) -> float:
        """
        Retrieve a player's rating as of a specified date. Finds an entry in self.rating history for the latest
        date less than or equal to the specified date. If there are multiple entries on that date, take the
        one corresponding to a game result.

        :param date: as-of-date to get a player's rating
        :param default_rating: the default rating to return for dates before the earliest date in the player's
        rating history (i.e., the default rating for new players)

        :return: player's rating as of the specified date
        """
        history_df = pd.DataFrame(self.rating_history, columns=["date", "rating"])

        # only select one entry per distinct date
        history_df["r"] = history_df.groupby(["date"]).rank(method="first", ascending=False)
        history_df = history_df[history_df["r"] == 1]

        # get the rating for the latest date
        history_df = history_df[history_df["date"] <= date].sort_values("date", ascending=False)
        if history_df.shape[0] == 0:
            return default_rating
        else:
            return history_df.reset_index().loc[0, "rating"]

    def count_games(self) -> int:
        """Counts games played by this Player."""
        return len(self.rating_history) - 1

    def _update_rating_history(self, rating: float, date: Union[str, float], keep_history: bool = True):
        """
        Update a player's rating history (self.rating_history)

        :param rating: player's new rating effective on this date
        :param date: effective date for new rating
        :param keep_history: if True, append to the rating history; otherwise replace
        the last date in the history
        """
        if keep_history:
            self.rating_history.append((date, rating))
        else:
            if self.rating_history:
                self.rating_history[-1] = (date, rating)
            else:
                self.rating_history = [(date, rating)]

    def __str__(self):
        n_games = self.count_games()
        return f"{self.id}: {self.rating:.2f} ({n_games} game{'s' * (n_games != 1)})"

    def __repr__(self):
        return f"Player(id = {self.id}, rating = {self.rating:.2f}, n_games = {self.count_games()})"

    def __eq__(self, other):
        return self.rating == other

    def __lt__(self, other):
        return self.rating < other

    def __le__(self, other):
        return self.rating <= other

    def __gt__(self, other):
        return self.rating > other

    def __ge__(self, other):
        return self.rating >= other


class Tracker:
    """
    The Tracker object can be used to track rating changes over time for a group of
    players (or teams, etc.) with multiple matchups against each other. The tracker
    stores and updates a list of Player objects (in Tracker.players) and those Player
    objects store the current ratings and (optionally) the full rating histories for the
    individual players.
    """

    def __init__(
        self,
        elo_rater: MultiElo = MultiElo(),
        initial_rating: float = DEFAULT_INITIAL_RATING,
        players: Union[List[Player], str] = None,
        keep_history: bool = True,
        date_col: str = None,
    ):
        """
        Instantiate a tracker that will track player's ratings over time as matchups occur.

        :param elo_rater: a MultiElo object
        :param initial_rating: initial rating value for new players
        :param players: list of existing players, or a filepath of a saved player
        list (from the Tracker.save_player_data method). If None, begin with no
        players in the tracker. New players will be added to the tracker when they
        appear in a matchup for the first time.
        :param keep_history: if True, keep the rating history for all players; otherwise
        only maintain the current rating (setting to False will use less memory for
        large datasets)
        :param date_col: name of the date column to use when processing data and
        creating the history dataframe. If not provided, it will take the value used the
        first time the process_data method is called.
        """
        self.elo = elo_rater
        self.initial_player_rating = initial_rating

        if players is None:
            players = []
        elif isinstance(players, str):
            with open(players, "rb") as f:
                players = pickle.load(f)
        self.players = players
        self._validate_player_list()
        self.keep_history = keep_history
        self.date_col = date_col
        logger.info(f"Created Tracker with Elo parameters K={self.elo.k}, D={self.elo.d}")

    def process_data(self, matchup_history_df: pd.DataFrame, date_col: str = None):
        """
        Process the full matchup history of a group of players. Update the ratings and rating history for all
        players in found in the matchup history.

        Annotate ties using a tuple of player IDs. For example, this data would indicate that Lisa came in
        first place, Bart and Marge tied for second, and Homer came in 3rd:

        |   date | 1st                        | 2nd               | 3rd   | 4th |
        |--------|----------------------------|-------------------|-------|-----|
        |      1 | Lisa                       | ('Bart', 'Marge') | Homer |     |

        Note: Only the order of the players matters, not the specific columns they appear in. For example,
        Homer could appear in either the 3rd or 4th column in the example data above and the results would
        be the same.

        Example usage:
        >>> elo = MultiElo()
        >>> tracker = Tracker(elo)
        >>> data = pd.DataFrame([[1, "Lisa", "Homer"], [2, "Marge", "Bart"], [3, "Lisa", "Bart"]])
        >>> data.columns = ["date", "1st", "2nd"]
        >>> tracker.process_data(data)
        >>> tracker.get_current_ratings()
           rank player_id  n_games       rating
        0     1      Lisa        2  1030.530498
        1     2     Marge        1  1016.000000
        2     3     Homer        1   984.000000
        3     4      Bart        2   969.469502


        :param matchup_history_df: dataframe of matchup history with a column for date and one column for each
        possible finishing place (e.g., "date", "1st", "2nd", "3rd", ...). Finishing place columns should be in
        order of first to last. Column names do not matter.
        :param date_col: name of the date column. If self.date_col is None, then self.date_col will be set to
        this value. Otherwise, this parameter should not be specified (or it should be equal to self.date_col).
        (default = self.date_col or "date")
        """
        if date_col is not None:
            warnings.warn("The date_col parameter will be removed from process_data in a "
                          "future version. Set the date_col parameter when instantiating the "
                          "Tracker object instead, i.e., `tracker = Tracker(date_col=...)`",
                          DeprecationWarning)

        if self.date_col is None:
            self.date_col = date_col if date_col is not None else "date"
        elif date_col is None:
            pass
        elif date_col != self.date_col:
            raise ValueError(f"The provided date column ({date_col}) does not match the established date column "
                             f"in the Tracker ({self.date_col}")

        if self.date_col not in matchup_history_df.columns:
            raise ValueError(f"The tracker's date column ({self.date_col}) does not appear in the columns of "
                             f"the dataframe being processed (available columns: {matchup_history_df.columns})")

        matchup_history_df = matchup_history_df.sort_values(self.date_col).reset_index(drop=True)
        place_cols = [x for x in matchup_history_df.columns if x != self.date_col]
        matchup_history_df = matchup_history_df.dropna(how="all", axis=0, subset=place_cols)  # drop rows if all NaN

        # loop through each row of history, then loop through players for each date
        for _, row in matchup_history_df.iterrows():
            date = row[self.date_col]
            players = []
            result_order = []
            for i, col in enumerate(place_cols):
                current_player = row[col]
                if current_player is None:
                    pass
                elif isinstance(current_player, (tuple, list)):
                    # multiple players (a tie)
                    players += [self._get_or_create_player(x) for x in current_player]
                    result_order += [i] * len(current_player)
                else:
                    # one player
                    players.append(self._get_or_create_player(current_player))
                    result_order.append(i)
            initial_ratings = np.array([player.rating for player in players])
            logger.debug(f"found players: {players}")
            logger.debug(f"found ratings: {list(initial_ratings)}")
            logger.debug(f"found result_order: {result_order}")

            # process data for one date
            new_ratings = self.elo.get_new_ratings(initial_ratings, result_order=result_order)
            logger.info(f"processing rating changes for date {date}...")
            for i, player in enumerate(players):
                player.update_rating(new_ratings[i], date=date, keep_history=self.keep_history)

    def get_current_ratings(self) -> pd.DataFrame:
        """
        Retrieve the current ratings of all players in this Tracker.

        :return: dataframe with all players' ratings and number of games played
        """
        df = pd.DataFrame({"player": self.players})
        df["player_id"] = df["player"].apply(lambda x: x.id)
        df["rating"] = df["player"].apply(lambda x: x.rating)
        df["n_games"] = df["player"].apply(lambda x: x.count_games())
        df = df.sort_values("player", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, df.shape[0] + 1)
        df = df[["rank", "player_id", "n_games", "rating"]]
        return df

    def get_history_df(self) -> pd.DataFrame:
        """
        Retrieve the rating history for all players in this Tracker.

        :return: dataframe with all players' ratings on each date that they changed
        """
        date_col = self.date_col or "date"
        history_df = pd.DataFrame(columns=["player_id", date_col, "rating"])
        history_df["rating"] = history_df["rating"].astype(float)

        for player in self.players:
            # check if there are any missing dates after the first entry (the initial rating)
            if any([x[0] is None for x in player.rating_history[1:]]):
                warnings.warn(f"WARNING: possible missing dates in history for Player {player.id}")

            player_history_df = pd.DataFrame(player.rating_history, columns=[date_col, "rating"])
            player_history_df = player_history_df[~player_history_df[date_col].isna()]
            player_history_df["player_id"] = player.id
            history_df = pd.concat([history_df, player_history_df], sort=False)

        return history_df.reset_index(drop=True)

    def retrieve_existing_player(self, player_id: str) -> Player:
        """Retrieve a player in the Tracker with a given ID."""
        try:
            return [x for x in self.players if x.id == player_id][0]
        except IndexError:
            raise ValueError(f"no player found with ID {player_id}")

    def save_player_data(self, file: str, save_full_history: bool = True):
        """
        Save the player dataframe (self.players) so the ratings can be loaded into a
        new tracker later. The self.players list is saved to a pickle file.

        :param file: path to write the tracker data to
        :param save_full_history: if True, save the full rating history for each player
        in the tracker; otherwise only save the current ratings. Recommended to set to
        False when you do not care about the full rating history, especially if you have
        a very large dataset.
        """
        if save_full_history:
            player_list = self.players
        else:
            # only save the ID and rating for each player (no rating history)
            player_list = [Player(player_id=x.id, rating=x.rating) for x in self.players]

        with open(file, "wb") as f:
            pickle.dump(player_list, f)

    def _get_or_create_player(self, player_id: str) -> Player:
        try:
            return self.retrieve_existing_player(player_id)
        except ValueError:
            return self._create_new_player(player_id)

    def _create_new_player(self, player_id: str) -> Player:
        try:
            # first check if the player already exists
            self.retrieve_existing_player(player_id)
        except ValueError:
            # exception means we didn't find the player
            # create and add the player to the database
            player = Player(player_id, rating=self.initial_player_rating)
            self.players.append(player)
            self._validate_player_list()
            return player
        else:
            # no exception means the player already exists
            raise ValueError(f"a player with ID {player_id} already exists in the tracker")

    def _validate_player_list(self):
        if not isinstance(self.players, list):
            raise TypeError(f"players should be a list (found a type of {type(self.players)})")

        if not all([isinstance(x, Player) for x in self.players]):
            raise TypeError("The players list should contain Player objects")

        if len(self.players) > len(set([x.id for x in self.players])):
            raise ValueError("Player IDs must be unique")

    def __repr__(self):
        return f"Tracker({len(self.players)} total players)"

    def __eq__(self, other):
        return sorted(self.players) == sorted(other.players)
