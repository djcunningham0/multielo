from multielo import Player, Tracker, MultiElo
from multielo.player_tracker import DEFAULT_INITIAL_RATING

import pandas as pd
import numpy as np
import pickle
import tempfile
import os
import pytest


DATA = pd.DataFrame({
    "date": ["2020-03-29", "2020-04-05", "2020-04-12"],
    "1st": ["Homer", "Lisa", "Lisa"],
    "2nd": ["Marge", "Bart", "Marge"],
    "3rd": ["Bart", "Homer", "Homer"]
})


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
    assert player_1 >= player_2
    assert player_3 < player_2
    assert player_3 <= player_2
    assert player_3 == player_4


def test_tracker():
    """test with some known results"""
    elo = MultiElo(k_value=32, d_value=400, score_function_base=1)
    tracker = Tracker(elo_rater=elo, initial_rating=1000)
    tracker.process_data(DATA)
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


def test_tracker_custom_date_col_name():
    data = DATA.rename(columns={"date": "week"}).copy()
    tracker = Tracker()
    with pytest.deprecated_call():
        tracker.process_data(data, date_col="week")
    assert "week" in tracker.get_history_df().columns


def test_tracker_error_date_col_does_not_match():
    data = DATA.rename(columns={"date": "week"}).copy()
    tracker = Tracker(date_col="date")
    with pytest.raises(ValueError):
        with pytest.deprecated_call():
            tracker.process_data(data, date_col="week")


def test_tracker_error_different_date_cols():
    data = DATA.rename(columns={"date": "week"}).copy()
    tracker = Tracker()
    tracker.process_data(DATA)  # date_col is "date"
    with pytest.raises(ValueError):
        tracker.process_data(data)  # "date" is not in the columns
    with pytest.raises(ValueError):
        with pytest.deprecated_call():
            tracker.process_data(data, date_col="week")  # already implicitly set date_col to "date"


def test_tracker_equal():
    tracker_1 = Tracker()
    tracker_2 = Tracker()
    tracker_1.process_data(DATA)
    tracker_2.process_data(DATA)
    assert tracker_1 == tracker_2


def test_tie_data():
    elo = MultiElo(k_value=32, d_value=400, score_function_base=1)
    tracker = Tracker(elo_rater=elo, initial_rating=1000)
    data = pd.DataFrame({
        "date": [1, 2, 3, 4, 5],
        "1st": ["Homer", ("Lisa", "Bart"), ("Lisa", "Marge"), ("Marge", "Homer", "Bart"), "Lisa"],
        "2nd": ["Marge", None, "Homer", None, ("Bart", "Marge")],
        "3rd": ["Bart", "Homer", None, None, "Homer"]
    })
    tracker.process_data(data)
    homer = tracker.retrieve_existing_player("Homer")
    marge = tracker.retrieve_existing_player("Marge")
    bart = tracker.retrieve_existing_player("Bart")
    lisa = tracker.retrieve_existing_player("Lisa")
    assert homer == 956.5862536380732
    assert marge == 1008.4435164345906
    assert bart == 992.2443714740361
    assert lisa == 1042.7258584533


def test_pickle_player():
    player = Player("test")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "player.pickle")
        # test that we can pickle the Player object
        with open(file, "wb") as f:
            pickle.dump(player, f)
        # test that we can load from the pickle file and it's equivalent to the original object
        with open(file, "rb") as f:
            new_player = pickle.load(f)
            assert new_player == player


def test_pickle_tracker_player_df():
    """Test that we can pickle the Tracker.player_df object. This is the object we need
    to export and load for batch processing."""
    tracker = Tracker()
    tracker.process_data(DATA)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "players.pickle")
        # test that we can pickle the Tracker object
        with open(file, "wb") as f:
            pickle.dump(tracker.players, f)
        # test that we can load from the pickle file and it's equivalent to the original object
        with open(file, "rb") as f:
            new_players = pickle.load(f)
            new_tracker = Tracker(players=new_players)
            assert new_tracker == tracker


def test_save_and_load_player_data():
    tracker = Tracker()
    tracker.process_data(DATA)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "player_df.pickle")
        tracker.save_player_data(file)
        new_tracker = Tracker(players=file)
        assert tracker == new_tracker


def test_save_and_load_player_data_no_history():
    tracker = Tracker()
    tracker.process_data(DATA)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "player_df.pickle")
        tracker.save_player_data(file, save_full_history=False)
        new_tracker = Tracker(players=file)
        assert tracker == new_tracker  # this checks that all player IDs and ratings are equal
        assert all(x.count_games() == 0 for x in new_tracker.players)


def test_good_player_list():
    players = [Player("test"), Player("test_2")]
    _ = Tracker(players=players)
    # should not raise an exception


def test_bad_player_list():
    # wrong data type (this was the old syntax)
    players = pd.DataFrame({
        "player_id": ["test", "test_2"],
        "player": [Player("test"), Player("test_2")],
    })
    with pytest.raises(TypeError):
        _ = Tracker(players=players)

    # duplicate players
    players = [Player("test"), Player("test")]
    with pytest.raises(ValueError):
        _ = Tracker(players=players)

    # bad type
    players = [{"id": "test", "rating": 1000}, {"id": "test_2", "rating": 1000}]
    with pytest.raises(TypeError):
        _ = Tracker(players=players)

    # bad type
    with pytest.raises(TypeError):
        _ = Tracker(players=1)

    # string that isn't a valid filepath
    with pytest.raises(FileNotFoundError):
        _ = Tracker(players="/file/that/doesnt/exist")

    # pickled object with wrong format
    l = [1, 2, 3]
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "players.pickle")
        with open(file, "wb") as f:
            pickle.dump(l, f)
        with pytest.raises(TypeError):
            _ = Tracker(players=file)
