"""Every league's games from portal_pivot_db, the official database.

One loader for every league the hub holds, from its standard `team_boxscore` view
(one row per team-game: pts, poss, location H/A/N) in

    ncaa.db      NCAA-D1                       (RealGM box scores)
    intl.db      INTL-<RealGM league id>       154 leagues
    national.db  NT-<RealGM tournament id>     78 competitions
    nba.db       NBA, GLEAGUE                  (poss by RealGM's formula, filled in the view)

returned as two rows per game, one per team's offence: exactly the rows
models.data.load_season_games and intl_scraper.models_data.load_season_league_games
build from this repo's own databases (same filters: both teams at least 25
possessions, points not negative, the two teams' possessions averaged; for NCAA-D1
the rows and the venue are identical for 2002-2026).

The NCAA season in progress can add games RealGM has not posted yet from ESPN
(the hub's team_boxscore_espn view, built nightly by its ncaa.espn_team_box step);
a game RealGM has is always taken from RealGM, so these rows disappear as RealGM
catches up. Each row says where it came from (`source`).

Venue: a team's own location, H / A / N; no location → neutral (as load_season_games).
Exhibitions (NBA / G League preseason, All-Star games) are left out.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from models.data import GameRow

DB_DIR = Path(os.environ.get("PORTAL_DB_DIR", "~/Dropbox/portal_pivot_db/db")).expanduser()
MIN_POSS = 25.0
#: exhibitions left out of every league's ratings (NBA / G League season types)
EXHIBITION = ("Pre Season", "All Star", "All-Star")
H = {"H": 1, "A": -1, "N": 0}
DATABASES = ("ncaa", "intl", "national", "nba")
COLUMNS = ["league_id", "season", "game_id", "date", "team_id", "opp_id", "pts", "poss", "h", "source"]


def database(league_id: str) -> str:
    if league_id.startswith("NCAA-"):
        return "ncaa"
    if league_id.startswith("INTL-"):
        return "intl"
    if league_id.startswith("NT-"):
        return "national"
    if league_id in ("NBA", "GLEAGUE"):
        return "nba"
    raise ValueError(f"no database for league {league_id!r}")


def current_season(today: Optional[date] = None) -> int:
    """Season end year by the NCAA calendar: October on belongs to the next year's season."""
    today = today or date.today()
    return today.year + 1 if today.month >= 10 else today.year


def espn_fill_seasons(db_dir: Path = DB_DIR, today: Optional[date] = None) -> list[int]:
    """NCAA-D1 seasons that take ESPN's games RealGM has not posted: the season in
    progress, and any season whose RealGM box scores stop before ESPN's last game day
    (2025-26: RealGM ends 2 April 2026, so the Final Four and the title game come from
    ESPN until RealGM's rows are collected — then they are linked and ESPN's drop out)."""
    out = {current_season(today)}
    conn = connect("ncaa", db_dir)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'team_boxscore_espn'").fetchone():
            out |= {int(s) for (s,) in conn.execute(
                "WITH r AS (SELECT season, MAX(date) d FROM team_boxscore WHERE league_id = 'NCAA-D1' GROUP BY season), "
                "e AS (SELECT season, MAX(date) d FROM team_boxscore_espn GROUP BY season) "
                "SELECT e.season FROM e LEFT JOIN r USING (season) WHERE r.d IS NULL OR r.d < e.d")}
    finally:
        conn.close()
    return sorted(out)


def connect(db: str, db_dir: Path = DB_DIR) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{Path(db_dir) / (db + '.db')}?mode=ro", uri=True, timeout=120)
    c.execute("PRAGMA busy_timeout = 120000")
    return c


def _where(leagues, seasons) -> tuple[str, list]:
    clauses, args = [], []
    if leagues:
        leagues = sorted(set(leagues))
        clauses.append(f"league_id IN ({','.join('?' * len(leagues))})")
        args += leagues
    if seasons:
        seasons = sorted({int(s) for s in seasons})
        clauses.append(f"season IN ({','.join('?' * len(seasons))})")
        args += seasons
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", args


def _pairs(team_games: pd.DataFrame, source: str, min_poss: float) -> pd.DataFrame:
    """Team-game rows → one row per team's offence against its opponent."""
    a = team_games.rename(columns={"team_id": "team_id", "pts": "pts", "poss": "poss_a"})
    b = team_games[["league_id", "game_id", "team_id", "pts", "poss"]].rename(
        columns={"team_id": "opp_id", "pts": "opp_pts", "poss": "poss_b"})
    j = a.merge(b, on=["league_id", "game_id"])
    j = j[(j.team_id != j.opp_id) & (j.poss_a >= min_poss) & (j.poss_b >= min_poss)
          & (j.pts >= 0) & (j.opp_pts >= 0)].copy()
    j["poss"] = (j.poss_a + j.poss_b) / 2.0
    j["h"] = j["location"].map(H).fillna(0).astype(int)
    j["date"] = pd.to_datetime(j["date"], errors="coerce")
    j["source"] = source
    return j


def load(db: str, leagues: Optional[Iterable[str]] = None, seasons: Optional[Iterable[int]] = None,
         espn_seasons: Optional[Iterable[int]] = None, db_dir: Path = DB_DIR,
         min_poss: float = MIN_POSS) -> pd.DataFrame:
    """Offence rows for every (league, season) asked for in one database, sorted by
    league, season, game, team. *espn_seasons*: NCAA-D1 seasons that also take ESPN's
    games RealGM has not posted (default none)."""
    where, args = _where(leagues, seasons)
    conn = connect(db, db_dir)
    try:
        tg = pd.read_sql(f"SELECT league_id, season, game_id, date, team_id, pts, poss, location, game_type "
                         f"FROM team_boxscore{where}", conn, params=args)
        tg = tg[~tg.game_type.isin(EXHIBITION)].drop(columns="game_type").dropna(subset=["team_id", "pts", "poss"])
        out = [_pairs(tg, "realgm" if db != "nba" else "nba_stats", min_poss)]
        espn_seasons = sorted({int(s) for s in espn_seasons or []})
        if db == "ncaa" and espn_seasons and (not leagues or "NCAA-D1" in set(leagues)):
            has = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'team_boxscore_espn'").fetchone()
            if has:
                e = pd.read_sql(
                    f"SELECT league_id, season, game_id, date, team_id, pts, poss, location FROM team_boxscore_espn "
                    f"WHERE realgm_game_id IS NULL AND team_id IS NOT NULL "
                    f"AND season IN ({','.join('?' * len(espn_seasons))})", conn, params=espn_seasons)
                both = e.groupby("game_id").team_id.transform("size") == 2       # both teams D-I
                out.append(_pairs(e[both], "espn", min_poss))
    finally:
        conn.close()
    df = pd.concat(out, ignore_index=True)
    df["season"] = df["season"].astype(int)
    df["team_id"] = df["team_id"].astype("int64")
    df["opp_id"] = df["opp_id"].astype("int64")
    return df.sort_values(["league_id", "season", "game_id", "team_id"])[COLUMNS].reset_index(drop=True)


def league_seasons(db: str, db_dir: Path = DB_DIR, min_games: int = 50) -> list[tuple[str, int, int]]:
    """(league_id, season, games) with at least *min_games* box-scored games."""
    conn = connect(db, db_dir)
    try:
        return [(l, int(s), int(n)) for l, s, n in conn.execute(
            "SELECT league_id, season, COUNT(DISTINCT game_id) FROM team_boxscore "
            "WHERE league_id IS NOT NULL AND season IS NOT NULL GROUP BY 1, 2 HAVING COUNT(DISTINCT game_id) >= ? "
            "ORDER BY 1, 2", (min_games,))]
    finally:
        conn.close()


def rows(frame: pd.DataFrame) -> list[GameRow]:
    """DataFrame rows → GameRow, the models' input."""
    return [GameRow(game_id=int(g), season=int(s), team_id=int(t), opp_id=int(o), pts=int(p), poss=float(q), h=int(h))
            for g, s, t, o, p, q, h in zip(frame.game_id, frame.season, frame.team_id, frame.opp_id,
                                           frame.pts, frame.poss, frame.h)]
