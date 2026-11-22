#!/usr/bin/env python
"""Season-based intl team ratings — one row per (team, season, league).

Mirrors scripts/build_team_ratings.py exactly (Model 2, ridge, per-season fit)
but iterates over (season × league) instead of just season, because intl teams
from different domestic leagues rarely play each other. Each league's rating
system is fit independently.

Reads portal_pivot_db's intl.db, the official database, through its standard
team_boxscore view (models/hub.py; PORTAL_DB_DIR, read-only): the same rows this
repo's own intl.db gave (both teams in the league, at least 25 possessions each),
with each game in RealGM's own competition season. This repo's intl.db is a frozen
copy that keeps the old June-1 season rule and is no longer read.

A run over every league and the default seasons rebuilds the file from empty, so a
(league, season) that no longer exists in the hub leaves no rows behind; a run
limited by --seasons / --leagues replaces only those league-seasons.

Writes intl_team_ratings.db in the hub's handoff directory (models.hub.out_dir()).

Usage:
    PY=~/Dropbox/nba_draft_prediction/.venv/bin/python
    $PY scripts/build_intl_team_ratings.py
    $PY scripts/build_intl_team_ratings.py --seasons 2024-2026
    $PY scripts/build_intl_team_ratings.py --leagues 4,6,15   # Spanish-ACB, Italian-LBA, German-BBL
"""
from __future__ import annotations

import argparse
import collections
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models import hub                          # noqa: E402
from models.model2 import Model2                # noqa: E402

SOURCE_DB   = hub.DB_DIR / "intl.db"            # portal_pivot_db's, read-only
DEFAULT_OUT = hub.out_dir() / "intl_team_ratings.db"   # the hub's handoff directory
FIRST_SEASON, LAST_SEASON = 2001, 2026
LAMBDA_TEAM = 500.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_ratings (
    team_id    INTEGER NOT NULL,
    season     INTEGER NOT NULL,
    league_id  INTEGER NOT NULL,
    adj_o      REAL,
    adj_d      REAL,
    adj_pace   REAL,
    adj_o_sd   REAL,
    adj_d_sd   REAL,
    PRIMARY KEY (team_id, season, league_id)
);
CREATE INDEX IF NOT EXISTS idx_ratings_league_season
    ON team_ratings(league_id, season);

CREATE TABLE IF NOT EXISTS build_info (
    season         INTEGER NOT NULL,
    league_id      INTEGER NOT NULL,
    built_at       TEXT    NOT NULL,
    git_commit     TEXT,
    git_dirty      INTEGER,
    model          TEXT    NOT NULL,
    lambda_team    REAL    NOT NULL,
    n_rows         INTEGER,
    n_teams        INTEGER,
    games_home     INTEGER,
    games_away     INTEGER,
    games_neutral  INTEGER,
    home_effect    REAL,
    PRIMARY KEY (season, league_id)
);
"""


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(s) for s in spec.split(",") if s.strip()})


def parse_leagues(spec: str | None) -> set[int] | None:
    if not spec: return None
    return {int(x) for x in spec.split(",") if x.strip()}


def git_state() -> tuple[str | None, int | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=REPO, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty  = subprocess.run(["git", "status", "--porcelain", "--", "models"],
                                cwd=REPO, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        return commit or None, int(bool(dirty))
    except OSError:
        return None, None


def build(seasons: set[int], leagues: set[int] | None,
          out: Path, lambda_team: float, fresh: bool = False) -> int:
    if not isinstance(lambda_team, float):
        raise TypeError("lambda_team must be a float")
    if not SOURCE_DB.exists():
        raise SystemExit(f"source database not found: {SOURCE_DB}")

    # (league, season) with at least 50 box-scored games, as before; league ids are
    # the hub's 'INTL-<RealGM league id>', stored here as the integer RealGM id
    candidates = [(s, int(l.split("-", 1)[1]), l, n) for l, s, n in hub.league_seasons("intl", min_games=50)]
    candidates = sorted((s, lid, l, n) for s, lid, l, n in candidates
                        if s in seasons and (leagues is None or lid in leagues))
    frame = hub.load("intl", leagues={l for _, _, l, _ in candidates}, seasons={s for s, _, _, _ in candidates})
    groups = dict(tuple(frame.groupby(["league_id", "season"])))

    target = out.with_suffix(out.suffix + ".tmp") if fresh else out
    if fresh and target.exists():
        target.unlink()
    dst = sqlite3.connect(target)
    dst.executescript(SCHEMA)
    commit, dirty = git_state()
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_fits = 0

    try:
        print(f"fitting {len(candidates)} (season × league) universes:")

        for season, league_id, hub_league, n_games in candidates:
            games = groups.get((hub_league, season))
            rows = hub.rows(games) if games is not None else []
            if len(rows) < 20:
                print(f"  s={season} L={league_id:>3}  skipped (too few rows)")
                continue
            model = Model2(lambda_team=lambda_team)
            model.fit_rows(rows, season=season)
            summaries = model.point_summary()

            T = len(model.teams_)
            sigma = getattr(model, "_Sigma_eff", None)
            if sigma is not None:
                diag = np.diag(sigma)
                sd_o, sd_d = np.sqrt(diag[1:1+T]), np.sqrt(diag[1+T:1+2*T])
            else:
                sd_o = sd_d = [None] * T
            home_effect = float(model.theta_hat_[2*T+1])

            batch = [(int(tid), season, league_id,
                      float(summaries[tid].adj_o), float(summaries[tid].adj_d),
                      float(summaries[tid].adj_pace),
                      float(sd_o[i]) if sd_o[i] is not None else None,
                      float(sd_d[i]) if sd_d[i] is not None else None)
                     for i, tid in enumerate(model.teams_) if tid in summaries]
            venue = collections.Counter(r.h for r in rows)
            dst.execute("DELETE FROM team_ratings WHERE season = ? AND league_id = ?",
                        (season, league_id))
            dst.executemany(
                "INSERT INTO team_ratings VALUES (?,?,?,?,?,?,?,?)", batch)
            dst.execute(
                "INSERT OR REPLACE INTO build_info VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (season, league_id, built_at, commit, dirty, "Model2",
                 lambda_team, len(rows), len(batch),
                 venue[1], venue[-1], venue[0], home_effect))
            dst.commit()
            n_fits += 1

            print(f"  s={season} L={league_id:>3}  {len(batch):>3} teams  "
                  f"{len(rows):>5,} rows  H/A/N {venue[1]:>4}/{venue[-1]:>4}/{venue[0]:>3}  "
                  f"home {home_effect:+.2f}")
    finally:
        dst.close()
    if fresh:
        target.replace(out)

    return n_fits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", default=None,
                    help=f"range like 2024-2026 or list; default {FIRST_SEASON}-{LAST_SEASON}")
    ap.add_argument("--leagues", default=None,
                    help="comma list of RealGM LeagueIDs; default all in the hub's intl.db")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    fresh = args.seasons is None and args.leagues is None      # everything: rebuild from empty
    seasons = set(parse_seasons(args.seasons or f"{FIRST_SEASON}-{LAST_SEASON}"))
    leagues = parse_leagues(args.leagues)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model2 lambda_team={LAMBDA_TEAM}  seasons={sorted(seasons)}  "
          f"leagues={sorted(leagues) if leagues else 'all'}  source={SOURCE_DB} (read-only)"
          f"{'  [rebuild from empty]' if fresh else ''}")
    n = build(seasons, leagues, args.out, LAMBDA_TEAM, fresh=fresh)
    print(f"\nfit {n} (season × league) universes → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
