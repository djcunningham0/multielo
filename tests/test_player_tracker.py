from multielo import Player, Tracker, MultiElo
from multielo.player_tracker import DEFAULT_INITIAL_RATING
import pandas as pd
import numpy as np


def test_default_player():
    player = Player("test")
    assert player.rating == DEFAULT_INITIAL_RATING


def test_player_with_history():
    player = Player("test", rating_history=[(1, 1000), (2, 1100)], date=3)
    assert player.rating == DEFAULT_INITIAL_RATING
    assert player.get_rating_as_of_date(1) == 1000
    assert player.get_rating_as_of_date(2) == 1100


def test_update_rating():
    player = Player("test", rating=1000)
    player.update_rating(new_rating=1200, date=1)
    assert player.rating == 1200


def test_get_rating_as_of_date():
    player = Player("test", rating=900, date="2020-01-01")
    player.update_rating(new_rating=1100, date="2020-02-01")
    player.update_rating(new_rating=1200, date="2020-03-01")
    player.update_rating(new_rating=1300, date="2020-04-01")
    assert player.get_rating_as_of_date("2019-12-31") == DEFAULT_INITIAL_RATING
    assert player.get_rating_as_of_date("2020-01-15") == 900
    assert player.get_rating_as_of_date("2020-02-15") == 1100
    assert player.get_rating_as_of_date("2020-03-15") == 1200
    assert player.rating == 1300


def test_player_equality():
    player_1 = Player("test1", rating=1200)
    player_2 = Player("test2", rating=1000)
    player_3 = Player("test3", rating=800)
    player_4 = Player("test4", rating=800)
    assert player_1 > player_2
    assert player_3 < player_2
    assert player_3 == player_4


def test_tracker():
    # test with some known results
    elo = MultiElo(k_value=32, d_value=400, score_function_base=1)
    tracker = Tracker(elo_rater=elo, initial_rating=1000)
    data = pd.DataFrame({
        "date": ["2020-03-29", "2020-04-05", "2020-04-12"],
        "1st": ["Homer", "Lisa", "Lisa"],
        "2nd": ["Marge", "Bart", "Marge"],
        "3rd": ["Bart", "Homer", "Homer"]
    })
    tracker.process_data(data)
    homer = tracker.retrieve_existing_player("Homer")
    marge = tracker.retrieve_existing_player("Marge")
    bart = tracker.retrieve_existing_player("Bart")
    lisa = tracker.retrieve_existing_player("Lisa")
    assert homer.count_games() == 3
    assert marge.count_games() == 2
    assert bart.count_games() == 2
    assert lisa.count_games() == 2
    assert np.mean([homer.rating, marge.rating, bart.rating, lisa.rating]) == 1000
    assert homer == 977.4832443375076
    assert marge == 1000.5940386640012
    assert bart == 980.6241719618897
    assert lisa == 1041.2985450366014
