"""models/hub.py (every league's games from portal_pivot_db) and the rolling builder's
snapshot schedule, on small synthetic databases."""
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd

from models import hub

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rolling_ratings", REPO / "scripts" / "rolling_ratings.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

COLS = "league_id TEXT, season INT, game_id INT, date TEXT, team_id INT, pts INT, poss REAL, location TEXT, game_type TEXT"


def make(tmp_path):
    c = sqlite3.connect(tmp_path / "ncaa.db")
    c.execute(f"CREATE TABLE team_boxscore ({COLS})")
    c.executemany("INSERT INTO team_boxscore VALUES (?,?,?,?,?,?,?,?,?)", [
        ("NCAA-D1", 2027, 1, "2026-11-04", 10, 80, 70.0, "H", "regular_season"),
        ("NCAA-D1", 2027, 1, "2026-11-04", 20, 70, 72.0, "A", "regular_season"),
        ("NCAA-D1", 2027, 2, "2026-11-05", 10, 60, 20.0, "N", "regular_season"),      # under 25 possessions: dropped
        ("NCAA-D1", 2027, 2, "2026-11-05", 30, 50, 66.0, "N", "regular_season"),
        ("NCAA-D1", 2027, 3, "2026-11-06", 20, 75, 69.0, None, "regular_season"),     # no venue: neutral
        ("NCAA-D1", 2027, 3, "2026-11-06", 30, 71, 69.0, None, "regular_season")])
    c.execute(f"CREATE TABLE team_boxscore_espn ({COLS}, espn_team_id INT, realgm_game_id INT)")
    c.executemany("INSERT INTO team_boxscore_espn VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
        ("NCAA-D1", 2027, 900, "2026-11-10", 10, 90, 68.0, "H", None, 101, None),     # RealGM lacks it: taken
        ("NCAA-D1", 2027, 900, "2026-11-10", 30, 60, 68.0, "A", None, 103, None),
        ("NCAA-D1", 2027, 901, "2026-11-04", 10, 80, 70.0, "H", None, 101, 1),        # RealGM has it: skipped
        ("NCAA-D1", 2027, 901, "2026-11-04", 20, 70, 72.0, "A", None, 102, 1),
        ("NCAA-D1", 2027, 902, "2026-11-11", 20, 77, 70.0, "H", None, 102, None),     # opponent not D-I: skipped
        ("NCAA-D1", 2027, 902, "2026-11-11", None, 50, 70.0, "A", None, 999, None)])
    c.commit()
    n = sqlite3.connect(tmp_path / "nba.db")
    n.execute(f"CREATE TABLE team_boxscore ({COLS})")
    n.executemany("INSERT INTO team_boxscore VALUES (?,?,?,?,?,?,?,?,?)", [
        ("NBA", 2027, 7, "2026-10-10", 1, 100, 98.0, "H", "Pre Season"), ("NBA", 2027, 7, "2026-10-10", 2, 99, 98.0, "A", "Pre Season"),
        ("NBA", 2027, 8, "2026-10-25", 1, 110, 99.0, "H", "Regular Season"), ("NBA", 2027, 8, "2026-10-25", 2, 101, 99.0, "A", "Regular Season")])
    n.commit()


def test_load_pairs_filters_venue_and_espn_fill(tmp_path):
    make(tmp_path)
    plain = hub.load("ncaa", db_dir=tmp_path)
    assert sorted(plain.game_id.unique()) == [1, 3]
    g1 = plain[plain.game_id == 1].set_index("team_id")
    assert g1.loc[10, "h"] == 1 and g1.loc[20, "h"] == -1 and g1.loc[10, "poss"] == 71.0 and g1.loc[10, "opp_id"] == 20
    assert set(plain[plain.game_id == 3].h) == {0}
    filled = hub.load("ncaa", espn_seasons=[2027], db_dir=tmp_path)
    assert sorted(filled.game_id.unique()) == [1, 3, 900]
    assert set(filled[filled.game_id == 900].source) == {"espn"} and set(filled[filled.game_id == 1].source) == {"realgm"}
    rows = hub.rows(filled)
    assert len(rows) == 6 and rows[0].game_id == 1


def test_espn_fill_seasons_in_progress_and_where_realgm_ends_early(tmp_path):
    make(tmp_path)
    c = sqlite3.connect(tmp_path / "ncaa.db")
    c.executemany("INSERT INTO team_boxscore VALUES (?,?,?,?,?,?,?,?,?)", [
        ("NCAA-D1", 2026, 50, "2026-04-02", 10, 70, 70.0, "N", "postseason"),
        ("NCAA-D1", 2025, 60, "2025-04-07", 10, 70, 70.0, "N", "postseason")])
    c.executemany("INSERT INTO team_boxscore_espn VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
        ("NCAA-D1", 2026, 950, "2026-04-06", 10, 69, 66.0, "N", None, 101, None),    # RealGM stops 04-02
        ("NCAA-D1", 2025, 960, "2025-04-07", 10, 70, 70.0, "N", None, 101, 60)])     # RealGM complete
    c.commit()
    from datetime import date
    assert hub.espn_fill_seasons(tmp_path, today=date(2026, 11, 19)) == [2026, 2027]
    # a year on: 2026 and the fixture's 2027 (ESPN to 11-11, RealGM to 11-06) still end early
    assert hub.espn_fill_seasons(tmp_path, today=date(2027, 11, 19)) == [2026, 2027, 2028]


def test_exhibitions_are_left_out(tmp_path):
    make(tmp_path)
    assert list(hub.load("nba", db_dir=tmp_path).game_id.unique()) == [8]


def test_snapshots_are_weekly_after_enough_games_and_end_on_the_last_day():
    days = pd.Series(pd.to_datetime(["2026-11-01"] * 40 + ["2026-11-09"] * 40 + ["2026-11-20"] * 20))
    snaps = R.snapshot_dates(days, n_teams=60)          # 30 games needed; 20 played by Nov 7
    assert [d.date().isoformat() for d in snaps] == ["2026-11-14", "2026-11-20"]
    assert hub.database("INTL-4") == "intl" and hub.database("NT-10") == "national" and hub.database("GLEAGUE") == "nba"
