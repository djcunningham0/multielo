# Change Log

## [0.4.1] -- 2022-04-16

### Fixed

* The correct date column name is now used in the tracker history df.
Also added some input validation around the date_col parameter in Tracker.process_data (now all subsequent calls of that method must use the same date_col).

### Deprecated

* The date_col parameter will be removed from the Tracker.process_data method in a future release (0.5.x).
Instead, the date_col parameter should be provided when instantiating the Tracker (`tracker = Tracker(date_col=...)).

## [0.4.0] -- 2022-04-09

***~~ Contains breaking changes ~~***

### Added

* Ability to save and load the player ratings from a Tracker object.
* Added the option to disable tracking the full history of each player in the Tracker object.
Intended for use with large datasets, where disabling full history could reduce memory footprint substantially.

### Changed

* **\[Breaking change\]** Refactored the Tracker object to store players in a list rather than a dataframe.
The 'player_df' attribute was renamed to 'players' and its type was changed to a list.
Your code is affected if you previously provided the player_df parameter when instantiating a Tracker object (i.e., `Tracker(player_df=...)` -- the new syntax is `Tracker(players=...)`) or if you directly accessed the `Tracker.player_df` attribute (new attribute is `Tracker.players`).
* **\[Breaking change\]** Removed the `logger` attribute from the MultiElo, Player, and Tracker objects.
Technically a breaking change, but it is unlikely anyone was supplying a logger when instantiating these objects.


## [0.3.0] -- 2021-11-09

### Added

* Can now handle ties in the MultiElo and Tracker objects.

### Changed

* Refactored how default parameters are set and removed multielo/config.py. Could possibly
  break old code if you are directly accessing the default values, but impact should be small.

## [0.2.0] -- 2021-04-21

### Added

* Estimate result probabilities with simulate_win_probabilities method in MultiElo class
* Improved logging, tests, documentation, and other small coding style improvements

### Changed

### Fixed

## [0.1.0] -- 2020-08-02

### Added

* Initial commit

### Changed

### Fixed

