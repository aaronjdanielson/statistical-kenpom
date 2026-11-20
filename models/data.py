"""Data loading for the rating models.

Loads game-level box score data from ncaa.db into the GameRow dataclass used
by all three models.  Each row represents ONE team's offensive output in one
game (so each game produces two rows — one per side).
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Default DB path resolution order:
#   1. NCAA_DB_PATH environment variable   (colleagues set this in their env)
#   2. portal_pivot_db's ncaa.db           (the official database: PORTAL_DB_DIR, default
#                                           ~/Dropbox/portal_pivot_db/db). Ratings built from it
#                                           match the ones built from this repo's own ncaa.db
#                                           exactly for 2002-2026 (checked 2026-11-18), and it
#                                           carries the season in progress.
#   3. ~/Dropbox/kenpom/ncaa.db            (this repo's own copy, frozen since 2026-04; to be
#                                           deleted once nothing reads it)
#   4. ./ncaa.db                            (fallback — CWD)
_PORTAL_DB = Path(os.environ.get("PORTAL_DB_DIR", "~/Dropbox/portal_pivot_db/db")).expanduser() / "ncaa.db"
_DROPBOX_DB = Path("~/Dropbox/kenpom/ncaa.db").expanduser()
_DEFAULT_DB: Path = (
    Path(os.environ["NCAA_DB_PATH"])
    if "NCAA_DB_PATH" in os.environ
    else next((p for p in (_PORTAL_DB, _DROPBOX_DB) if p.exists()), Path("ncaa.db"))
)


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


# Venue code -> h. schedules.Location is written by ncaa_scraper/parsers/schedule.py as the
# single letters 'H' / 'A' / 'N'. This map used to list only the long names, so every game
# silently fell through to neutral (h=0) and no model ever saw home court. The long names
# stay for callers that pass them; an unrecognised code now raises rather than quietly
# becoming neutral. A game with no schedules row is still treated as neutral (COALESCE).
_LOCATION_TO_H = {
    "H": 1, "A": -1, "N": 0,
    "Home": 1, "Away": -1, "Neutral": 0,
}

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
        try:
            h = _LOCATION_TO_H[loc]
        except KeyError:
            raise ValueError(
                f"unrecognised schedules.Location {loc!r} for GameID={gid} TeamID={tid}; "
                f"expected one of {sorted(_LOCATION_TO_H)}"
            ) from None
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


def parse_date(s: str) -> "datetime":
    """Parse RealGM date strings like 'Nov 14, 2025' or 'Nov 14,2025'."""
    from datetime import datetime
    for fmt in ("%b %d, %Y", "%b %d,%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {s!r}")


def open_db(path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    """Open ncaa.db read-only with WAL mode for concurrent access."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
