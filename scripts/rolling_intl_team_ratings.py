#!/usr/bin/env python
"""Rolling intl team ratings — one snapshot per K games elapsed.

Fits Model 2 on the growing prefix of a (season × league) universe: after
the first `min_games`, then re-fit every `step_games` games. Writes one
row per (team, season, league, snapshot_idx) with cumulative adj_o/adj_d/
adj_pace plus the number of games elapsed at that snapshot.

CAVEAT — for now we order games by `game_id` (RealGM assigns sequentially,
so this is a good chronology proxy). Once we extract Date from the
boxscore page HTML and populate schedules.Date, upgrade to weekly-date
milestones (mirroring scripts/rolling_net_rtg_2026.py for NCAA).

Writes output/intl_team_ratings_rolling.db. intl.db is opened READ-ONLY.

Usage:
    PY=~/Dropbox/nba_draft_prediction/.venv/bin/python
    $PY scripts/rolling_intl_team_ratings.py --seasons 2024-2026 --step 50
    $PY scripts/rolling_intl_team_ratings.py --seasons 2025 --leagues 4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from intl_scraper.models_data import (          # noqa: E402
    load_season_league_games, enumerate_league_seasons, open_intl_db,
)
from models.model2 import Model2                # noqa: E402

SOURCE_DB   = REPO / "intl.db"
DEFAULT_OUT = REPO / "output" / "intl_team_ratings_rolling.db"
FIRST_SEASON, LAST_SEASON = 2001, 2026
LAMBDA_TEAM = 500.0
DEFAULT_MIN_GAMES = 100     # cumulative games before first fit
DEFAULT_STEP_GAMES = 50     # games between snapshots

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_ratings_rolling (
    team_id       INTEGER NOT NULL,
    season        INTEGER NOT NULL,
    league_id     INTEGER NOT NULL,
    snapshot_idx  INTEGER NOT NULL,
    games_elapsed INTEGER NOT NULL,
    adj_o         REAL,
    adj_d         REAL,
    adj_pace      REAL,
    PRIMARY KEY (team_id, season, league_id, snapshot_idx)
);
CREATE INDEX IF NOT EXISTS idx_rolling_league_season
    ON team_ratings_rolling(league_id, season);
CREATE TABLE IF NOT EXISTS build_info (
    season        INTEGER NOT NULL,
    league_id     INTEGER NOT NULL,
    snapshot_idx  INTEGER NOT NULL,
    built_at      TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    lambda_team   REAL    NOT NULL,
    n_rows        INTEGER,
    n_teams       INTEGER,
    games_elapsed INTEGER,
    home_effect   REAL,
    PRIMARY KEY (season, league_id, snapshot_idx)
);
"""


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(s) for s in spec.split(",") if s.strip()})


def parse_leagues(spec: str | None) -> set[int] | None:
    return None if not spec else {int(x) for x in spec.split(",") if x.strip()}


def build_rolling(seasons: set[int], leagues: set[int] | None, out: Path,
                  lambda_team: float, min_games: int, step_games: int) -> int:
    if not SOURCE_DB.exists():
        raise SystemExit(f"source database not found: {SOURCE_DB}")

    src = open_intl_db(SOURCE_DB)
    dst = sqlite3.connect(out)
    dst.executescript(SCHEMA)
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_snapshots = 0

    try:
        candidates = enumerate_league_seasons(src, min_games=min_games)
        candidates = [(s, lid, n) for s, lid, n in candidates
                      if s in seasons and (leagues is None or lid in leagues)]
        print(f"rolling ratings for {len(candidates)} (season × league) universes  "
              f"(min={min_games}, step={step_games}):")

        for season, league_id, total_games in candidates:
            all_rows = load_season_league_games(src, season, league_id)
            if len(all_rows) < min_games * 2:  # rows = 2 × games
                continue

            # Sort by game_id (chronology proxy — see caveat in docstring)
            all_rows.sort(key=lambda r: r.game_id)

            # Snapshot milestones: after min_games, then every step_games
            unique_gids = []
            seen: set[int] = set()
            for r in all_rows:
                if r.game_id not in seen:
                    seen.add(r.game_id); unique_gids.append(r.game_id)
            milestones: list[int] = []
            g = min_games
            while g < len(unique_gids):
                milestones.append(g)
                g += step_games
            milestones.append(len(unique_gids))  # final full-season fit

            dst.execute("DELETE FROM team_ratings_rolling "
                        "WHERE season = ? AND league_id = ?",
                        (season, league_id))
            dst.execute("DELETE FROM build_info "
                        "WHERE season = ? AND league_id = ?",
                        (season, league_id))

            for snap_idx, k in enumerate(milestones):
                keep = set(unique_gids[:k])
                prefix = [r for r in all_rows if r.game_id in keep]
                if len(prefix) < 20:
                    continue
                model = Model2(lambda_team=lambda_team)
                model.fit_rows(prefix, season=season)
                summaries = model.point_summary()
                T = len(model.teams_)
                home_effect = float(model.theta_hat_[2*T+1])
                batch = [(int(tid), season, league_id, snap_idx, k,
                          float(summaries[tid].adj_o),
                          float(summaries[tid].adj_d),
                          float(summaries[tid].adj_pace))
                         for tid in model.teams_ if tid in summaries]
                dst.executemany(
                    "INSERT INTO team_ratings_rolling VALUES (?,?,?,?,?,?,?,?)",
                    batch)
                dst.execute(
                    "INSERT INTO build_info VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (season, league_id, snap_idx, built_at, "Model2",
                     lambda_team, len(prefix), len(batch), k, home_effect))
                n_snapshots += 1
            dst.commit()
            print(f"  s={season} L={league_id:>3}  {len(milestones)} snapshots  "
                  f"({len(unique_gids)} games, {len(all_rows)} rows)")
    finally:
        src.close(); dst.close()
    return n_snapshots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons",    default=f"{FIRST_SEASON}-{LAST_SEASON}")
    ap.add_argument("--leagues",    default=None)
    ap.add_argument("--out",        type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-games",  type=int, default=DEFAULT_MIN_GAMES)
    ap.add_argument("--step",       type=int, default=DEFAULT_STEP_GAMES,
                    dest="step_games")
    args = ap.parse_args()

    seasons = set(parse_seasons(args.seasons))
    leagues = parse_leagues(args.leagues)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model2 lambda_team={LAMBDA_TEAM}  "
          f"min_games={args.min_games}  step={args.step_games}")
    n = build_rolling(seasons, leagues, args.out, LAMBDA_TEAM,
                      args.min_games, args.step_games)
    print(f"\nemitted {n} rating snapshots → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
