"""Unit tests for parsers/teams.py using a real fixture page."""
from pathlib import Path

import pytest

from ncaa_scraper.parsers.teams import parse_teams_page

FIXTURE = Path(__file__).parent.parent / "fixtures" / "teams.html"


@pytest.fixture(scope="module")
def teams():
    return parse_teams_page(FIXTURE.read_text(encoding="utf-8"), 2026)


def test_returns_expected_count(teams):
    # 2026 season has 365 D-I teams
    assert len(teams) == 365


def test_required_fields_present(teams):
    expected = {"year", "ConferenceCode", "ConferenceID", "TeamCode", "TeamID", "School"}
    assert set(teams[0].keys()) == expected


def test_ids_are_integers(teams):
    for t in teams:
        assert isinstance(t["TeamID"], int)
        assert isinstance(t["ConferenceID"], int)


def test_known_team_present(teams):
    duke = next((t for t in teams if t["TeamID"] == 31), None)
    assert duke is not None
    assert duke["School"] == "Duke"
    assert duke["TeamCode"] == "Duke"
    assert duke["ConferenceCode"] == "Atlantic-Coast-Conference"
    assert duke["year"] == 2026


def test_no_duplicate_team_ids(teams):
    ids = [t["TeamID"] for t in teams]
    assert len(ids) == len(set(ids))


def test_no_numeric_school_names(teams):
    for t in teams:
        assert not t["School"].isdigit()
        assert t["School"] != ""


def test_empty_html_returns_empty():
    assert parse_teams_page("", 2026) == []
