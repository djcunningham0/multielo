import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Union, List, Tuple
import logging
import warnings
from .multielo import MultiElo
from .config import defaults


class Player:
    """
    The Player object stores the current and historical ratings of an individual player (or team, etc.). Attributes of
    interest are Player.rating (the current rating) and Player.rating_history (a list of historical ratings).
    """

    def __init__(
            self,
            player_id: str,
            rating: float = defaults["INITIAL_RATING"],
            rating_history: List[Tuple[Union[str, float], float]] = None,
            date: Union[str, float] = None,
            logger: logging.Logger = None,
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

        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"created player with ID {player_id} and rating {rating}")

    def update_rating(self, new_rating: float, date: Union[str, float]):
        """
        Update a player's rating and rating history. (updates the self.rating and self.rating_history attributes)

        :param new_rating: player's new rating
        :param date: date the new rating was achieved
        """
        self.logger.info(f"Updating rating for {self.id}: {self.rating:.3f} --> {new_rating:.3f}")
        self.rating = new_rating
        self._update_rating_history(rating=new_rating, date=date)

    def get_rating_as_of_date(
        self,
        date: Union[str, float],
        default_rating: float = defaults["INITIAL_RATING"]
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
        history_df = DataFrame(self.rating_history, columns=["date", "rating"])

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

    def _update_rating_history(self, rating: float, date: Union[str, float]):
        """
        Update a player's rating history (self.rating_history)

        :param rating: player's new rating effective on this date
        :param date: effective date for new rating
        """
        self.rating_history.append((date, rating))

    def __str__(self):
        n_games = self.count_games()
        return f"{self.id}: {self.rating:.2f} ({n_games} game{'s' * (n_games != 1)})"

    def __repr__(self):
        return f"Player(id = {self.id}, rating = {self.rating:.2f)}, n_games = {self.count_games()})"

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
    The Tracker object can be used to track rating changes over time for a group of players (or teams, etc.) with
    multiple matchups against each other. The tracker stores and updates a dataframe of Player objects (in
    Tracker.player_df) and those Player objects store the rating histories for the individual players.
    """

    def __init__(
        self,
        elo_rater: MultiElo = MultiElo(),
        initial_rating: float = defaults["INITIAL_RATING"],
        player_df: DataFrame = None,
        logger: logging.Logger = None,
    ):
        """
        Instantiate a tracker that will track player's ratings over time as matchups occur.

        :param elo_rater:
        :param initial_rating: initial rating value for new players
        :param player_df: dataframe of existing players. New players will be added to the dataframe when they
        appear in a matchup for the first time. If None, begin with no players in the dataframe.
        :param logger: use this logger if specified, otherwise create a logger with logging.getLogger()
        """
        self.elo = elo_rater
        self.initial_player_rating = initial_rating

        if player_df is None:
            player_df = DataFrame(columns=["player_id", "player"], dtype=object)

        self.player_df = player_df
        self._validate_player_df()

        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Created Tracker with Elo paramers K={self.elo.k}, D={self.elo.d}")

    def process_data(self, matchup_history_df: DataFrame, date_col: str = "date"):
        """
        Process the full matchup history of a group of players. Update the ratings and rating history for all
        players in found in the matchup history.

        :param matchup_history_df: dataframe of matchup history with a column for date and one column for each
        possible finishing place (e.g., "date", "1st", "2nd", "3rd", ...). Finishing place columns should be in
        order of first to last.
        :param date_col: name of the date column
        """
        matchup_history_df = matchup_history_df.sort_values(date_col).reset_index(drop=True)
        place_cols = [x for x in matchup_history_df.columns if x != date_col]
        matchup_history_df = matchup_history_df.dropna(how="all", axis=0, subset=place_cols)  # drop rows if all NaN
        for _, row in matchup_history_df.iterrows():
            date = row[date_col]
            players = [self._get_or_create_player(row[x]) for x in place_cols if not pd.isna(row[x])]
            initial_ratings = np.array([player.rating for player in players])
            new_ratings = self.elo.get_new_ratings(initial_ratings)
            self.logger.info(f"processing rating changes for date {date}...")
            for i, player in enumerate(players):
                player.update_rating(new_ratings[i], date=date)

    def get_current_ratings(self) -> DataFrame:
        """
        Retrieve the current ratings of all players in this Tracker.

        :return: dataframe with all players' ratings and number of games played
        """
        df = self.player_df.copy()
        df["rating"] = df["player"].apply(lambda x: x.rating)
        df["n_games"] = df["player"].apply(lambda x: x.count_games())
        df = df.sort_values("player", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, df.shape[0] + 1)
        df = df[["rank", "player_id", "n_games", "rating"]]
        return df

    def get_history_df(self) -> DataFrame:
        """
        Retrieve the rating history for all players in this Tracker.

        :return: dataframe with all players' ratings on each date that they changed
        """
        history_df = DataFrame(columns=["player_id", "date", "rating"])
        history_df["rating"] = history_df["rating"].astype(float)

        players = [player for player in self.player_df["player"]]
        for player in players:
            # check if there are any missing dates after the first entry (the initial rating)
            if any([x[0] is None for x in player.rating_history[1:]]):
                warnings.warn(f"WARNING: possible missing dates in history for Player {player.id}")

            player_history_df = DataFrame(player.rating_history, columns=["date", "rating"])
            player_history_df = player_history_df[~player_history_df["date"].isna()]
            player_history_df["player_id"] = player.id
            history_df = pd.concat([history_df, player_history_df], sort=False)

        return history_df.reset_index(drop=True)

    def retrieve_existing_player(self, player_id: str) -> Player:
        """Retrieve a player in the Tracker with a given ID."""
        if player_id in self.player_df["player_id"].tolist():
            player = self.player_df.loc[self.player_df["player_id"] == player_id, "player"].tolist()[0]
            return player
        else:
            raise ValueError(f"no player found with ID {player_id}")

    def _get_or_create_player(self, player_id: str) -> Player:
        if player_id in self.player_df["player_id"].tolist():
            return self.retrieve_existing_player(player_id)
        else:
            return self._create_new_player(player_id)

    def _create_new_player(self, player_id: str) -> Player:
        # first check if the player already exists
        if player_id in self.player_df["player_id"].tolist():
            raise ValueError(f"a player with ID {player_id} already exists in the tracker")

        # create and add the player to the database
        player = Player(player_id, rating=self.initial_player_rating, logger=self.logger)
        add_df = DataFrame({"player_id": [player_id], "player": [player]})
        self.player_df = pd.concat([self.player_df, add_df])
        self._validate_player_df()
        return player

    def _validate_player_df(self):
        if not self.player_df["player_id"].is_unique:
            raise ValueError("Player IDs must be unique")

        if not all([isinstance(x, Player) for x in self.player_df["player"]]):
            raise ValueError("The player column should contain Player objects")

        self.player_df = self.player_df.sort_values("player_id").reset_index(drop=True)

    def __repr__(self):
        return f"Tracker({self.player_df.shape[0]} total players)"
