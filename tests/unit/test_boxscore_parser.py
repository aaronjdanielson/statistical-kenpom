"""Unit tests for parsers/boxscore.py using a real fixture page."""
from pathlib import Path

import pytest

from ncaa_scraper.parsers.boxscore import parse_boxscore_page

FIXTURE = Path(__file__).parent.parent / "fixtures" / "boxscore.html"

# Duke at Tennessee, Oct 26 2025, GameID 512240, Season 2026
GAME_ID = 512240
SEASON = 2026


@pytest.fixture(scope="module")
def rows():
    return parse_boxscore_page(FIXTURE.read_text(encoding="utf-8"), GAME_ID, SEASON)


def test_returns_two_rows(rows):
    assert len(rows) == 2


def test_required_fields_present(rows):
    expected = {
        "GameID", "TeamID", "TeamCode", "Season", "Home",
        "Minutes", "FGM", "FGA", "FG3M", "FG3A",
        "FTM", "FTA", "OREB", "DREB", "REB",
        "AST", "PF", "STL", "TO", "BLK", "PTS", "POSS",
    }
    assert set(rows[0].keys()) == expected


def test_game_id_and_season_injected(rows):
    for r in rows:
        assert r["GameID"] == GAME_ID
        assert r["Season"] == SEASON


def test_home_flags_are_complementary(rows):
    homes = {r["Home"] for r in rows}
    assert homes == {0, 1}


def test_team_ids_are_different(rows):
    assert rows[0]["TeamID"] != rows[1]["TeamID"]


def test_known_scores(rows):
    by_team = {r["TeamID"]: r for r in rows}
    duke = by_team[31]
    tenn = by_team[263]
    assert duke["PTS"] == 83
    assert tenn["PTS"] == 76
    assert duke["Home"] == 0
    assert tenn["Home"] == 1


def test_poss_is_positive_float(rows):
    for r in rows:
        assert isinstance(r["POSS"], float)
        assert r["POSS"] > 0


def test_poss_formula(rows):
    # POSS = FGA - OREB + TO + 0.44*FTA
    for r in rows:
        expected = round(r["FGA"] - r["OREB"] + r["TO"] + 0.44 * r["FTA"], 3)
        assert r["POSS"] == expected


def test_known_poss(rows):
    by_team = {r["TeamID"]: r for r in rows}
    assert by_team[31]["POSS"] == 71.88
    assert by_team[263]["POSS"] == 73.16


def test_stats_are_non_negative(rows):
    int_fields = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                  "OREB", "DREB", "REB", "AST", "STL", "TO", "BLK", "PTS"]
    for r in rows:
        for f in int_fields:
            assert r[f] >= 0, f"{f}={r[f]} is negative"


def test_fgm_leq_fga(rows):
    for r in rows:
        assert r["FGM"] <= r["FGA"]
        assert r["FG3M"] <= r["FG3A"]
        assert r["FTM"] <= r["FTA"]


def test_empty_html_returns_empty():
    assert parse_boxscore_page("", 512240, 2026) == []
