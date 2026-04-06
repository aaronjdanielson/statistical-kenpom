"""
Integration tests — make real HTTP requests to RealGM.

Run with:
    pytest -m integration
"""
import pytest

from ncaa_scraper.config import REALGM_BASE, HEADERS, RATE_LIMIT_RPS, RATE_LIMIT_JITTER
from ncaa_scraper.scrapers.client import RealGMClient
from ncaa_scraper.parsers.teams import parse_teams_page
from ncaa_scraper.parsers.schedule import parse_schedule_page
from ncaa_scraper.parsers.boxscore import parse_boxscore_page


@pytest.fixture(scope="module")
def client():
    c = RealGMClient(HEADERS, RATE_LIMIT_RPS, RATE_LIMIT_JITTER)
    yield c
    c.close()


@pytest.mark.integration
def test_teams_live(client):
    html = client.get(f"{REALGM_BASE}/ncaa/teams/2026")
    teams = parse_teams_page(html, 2026)
    assert len(teams) >= 300
    ids = [t["TeamID"] for t in teams]
    assert len(ids) == len(set(ids)), "duplicate TeamIDs"
    assert any(t["TeamID"] == 31 for t in teams), "Duke missing"


@pytest.mark.integration
def test_schedule_live(client):
    html = client.get(
        f"{REALGM_BASE}/ncaa/conferences/Atlantic-Coast-Conference/1/Duke/31/schedule/2026"
    )
    games = parse_schedule_page(html, 31, 2026)
    assert len(games) >= 30
    for g in games:
        assert g["TeamID"] == 31
        assert g["Year"] == 2026
        assert g["GameID"] > 0
        assert g["DateSlug"]


@pytest.mark.integration
def test_boxscore_live(client):
    html = client.get(
        f"{REALGM_BASE}/ncaa/boxscore/2025-10-26/Duke-at-Tennessee/512240"
    )
    rows = parse_boxscore_page(html, 512240, season=2026)
    assert len(rows) == 2
    by_team = {r["TeamID"]: r for r in rows}
    assert 31 in by_team and 263 in by_team
    assert by_team[31]["PTS"] == 83
    assert by_team[263]["PTS"] == 76
    for r in rows:
        assert r["POSS"] > 0
        expected = round(r["FGA"] - r["OREB"] + r["TO"] + 0.44 * r["FTA"], 3)
        assert r["POSS"] == expected


@pytest.mark.integration
def test_client_retries_on_rate_limit(client):
    # Rapid-fire 3 requests — should all succeed without raising despite rate limiting
    for year in [2024, 2025, 2026]:
        html = client.get(f"{REALGM_BASE}/ncaa/teams/{year}")
        teams = parse_teams_page(html, year)
        assert len(teams) >= 300
