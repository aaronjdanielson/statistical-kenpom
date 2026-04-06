"""Unit tests for parsers/schedule.py using a real fixture page."""
from pathlib import Path

import pytest

from ncaa_scraper.parsers.schedule import parse_schedule_page

FIXTURE = Path(__file__).parent.parent / "fixtures" / "schedule.html"

# Duke 2026 schedule: team_id=31, year=2026
TEAM_ID = 31
YEAR = 2026


@pytest.fixture(scope="module")
def schedule():
    return parse_schedule_page(FIXTURE.read_text(encoding="utf-8"), TEAM_ID, YEAR)


def test_returns_games(schedule):
    assert len(schedule) > 0


def test_required_fields_present(schedule):
    expected = {"GameID", "Date", "DateSlug", "Versus", "Location",
                "WL", "TeamScore", "OppScore", "TeamID", "Year"}
    assert set(schedule[0].keys()) == expected


def test_game_ids_are_integers(schedule):
    for g in schedule:
        assert isinstance(g["GameID"], int)
        assert g["GameID"] > 0


def test_no_duplicate_game_ids(schedule):
    ids = [g["GameID"] for g in schedule]
    assert len(ids) == len(set(ids))


def test_team_id_and_year_injected(schedule):
    for g in schedule:
        assert g["TeamID"] == TEAM_ID
        assert g["Year"] == YEAR


def test_date_slug_format(schedule):
    # DateSlug is the raw URL component, e.g. "2025-10-26"
    import re
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for g in schedule:
        assert pattern.match(g["DateSlug"]), f"Bad DateSlug: {g['DateSlug']!r}"


def test_location_values(schedule):
    valid = {"H", "A", "N"}
    for g in schedule:
        assert g["Location"] in valid


def test_known_game_present(schedule):
    # Duke at Tennessee, Oct 26 2025, GameID 512240
    game = next((g for g in schedule if g["GameID"] == 512240), None)
    assert game is not None
    assert game["DateSlug"] == "2025-10-26"
    assert game["Versus"] == "Duke-at-Tennessee"
    assert game["Location"] == "A"
    assert game["WL"] == "W"


def test_empty_html_returns_empty():
    assert parse_schedule_page("", 31, 2026) == []
