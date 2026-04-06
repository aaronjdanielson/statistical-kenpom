"""Data loading for the rating models.

Loads game-level box score data from ncaa.db into the GameRow dataclass used
by all three models.  Each row represents ONE team's offensive output in one
game (so each game produces two rows — one per side).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Default DB path — kept here so the models package has no dependency on
# ncaa_scraper.  Pass a different path to open_db() to override.
_DEFAULT_DB = Path("~/Dropbox/kenpom/ncaa.db").expanduser()


@dataclass(frozen=True)
class GameRow:
    """One team's offensive half of a game."""
    game_id: int
    season: int
    team_id: int       # the team scoring
    opp_id: int        # the team defending
    pts: int           # points scored by team_id
    poss: float        # avg possessions for the game (both teams)
    h: int             # venue for team_id: +1=home, 0=neutral, -1=away


_LOCATION_TO_H = {"Home": 1, "Neutral": 0, "Away": -1}

_QUERY = """
SELECT
    b1.GameID,
    b1.Season,
    b1.TeamID                          AS team_id,
    b2.TeamID                          AS opp_id,
    b1.PTS                             AS pts,
    (b1.POSS + b2.POSS) / 2.0         AS poss,
    COALESCE(s.Location, 'Neutral')    AS location
FROM boxscores b1
JOIN  boxscores  b2 ON b2.GameID = b1.GameID AND b2.TeamID != b1.TeamID
LEFT JOIN schedules s ON s.GameID = b1.GameID AND s.TeamID = b1.TeamID
WHERE b1.Season = ?
  AND b1.POSS  >= ?
  AND b2.POSS  >= ?
  AND b1.PTS   >= 0
  AND b2.PTS   >= 0
ORDER BY b1.GameID, b1.TeamID
"""


def load_season_games(
    conn: sqlite3.Connection,
    season: int,
    min_poss: float = 25.0,
) -> list[GameRow]:
    """
    Load all game rows for one season.

    Returns two GameRow objects per game — one for each team's offensive output.
    Games where either team's possession estimate is below min_poss are dropped
    (these are typically data-quality issues or very incomplete games).
    """
    rows = []
    for gid, ssn, tid, oid, pts, poss, loc in conn.execute(_QUERY, (season, min_poss, min_poss)):
        h = _LOCATION_TO_H.get(loc, 0)
        rows.append(GameRow(
            game_id=gid,
            season=ssn,
            team_id=tid,
            opp_id=oid,
            pts=int(pts),
            poss=float(poss),
            h=h,
        ))
    return rows


def open_db(path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    """Open ncaa.db read-only with WAL mode for concurrent access."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
