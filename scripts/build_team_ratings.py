#!/usr/bin/env python
"""Build per-season team ratings with Model 2 and store them in this repo.

Writes output/team_ratings.db (gitignored by *.db). The source is models.data's default
database -- portal_pivot_db's ncaa.db, the official one, unless NCAA_DB_PATH says otherwise
(--source overrides both) -- opened READ-ONLY, and nothing downstream is touched -- roster.db
and draft.db keep the ratings they have.

Mirrors nba_draft_prediction/db/ratings.py exactly -- Model2(lambda_team=500.0), seasons
2002-2026, adj_o/adj_d/adj_pace from point_summary(), adj_o_sd/adj_d_sd from the diagonal of
the posterior covariance -- so the only difference from the ratings stored downstream is the
venue fix in models/data.py. Before that fix the loader expected 'Home'/'Away'/'Neutral'
while schedules.Location holds 'H'/'A'/'N', so every game loaded as neutral and the ratings
carried no home-court adjustment.

Pass lambda as a FLOAT. An int makes Model 2's penalty array integer, which truncates the
1e-4 stabilising term on the intercept and home effect to 0 and makes the solve singular.

Tables written:
    team_ratings  (team_id, season) -> adj_o, adj_d, adj_pace, adj_o_sd, adj_d_sd
                  same columns as downstream ncaa_team_ratings
    build_info    one row per season: when, git commit, lambda, row/team counts, venue
                  counts, fitted home effect (points per 100 possessions)

Usage (needs numpy + scipy; the nba_draft_prediction venv has both):
    PY=~/Dropbox/nba_draft_prediction/.venv/bin/python
    $PY scripts/build_team_ratings.py
    $PY scripts/build_team_ratings.py --seasons 2024-2026
    $PY scripts/build_team_ratings.py --compare ~/Dropbox/ncaa_roster_prediction/roster.db
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

from models import hub  # noqa: E402
from models.data import _DEFAULT_DB, load_season_games  # noqa: E402
from models.model2 import Model2  # noqa: E402

SOURCE_DB = _DEFAULT_DB
DEFAULT_OUT = hub.out_dir() / "team_ratings.db"        # the hub's handoff directory
FIRST_SEASON, LAST_SEASON = 2002, 2026
LAMBDA_TEAM = 500.0          # production value; must stay a float

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_ratings (
    team_id   INTEGER NOT NULL,
    season    INTEGER NOT NULL,
    adj_o     REAL,
    adj_d     REAL,
    adj_pace  REAL,
    adj_o_sd  REAL,
    adj_d_sd  REAL,
    PRIMARY KEY (team_id, season)
);
CREATE TABLE IF NOT EXISTS build_info (
    season         INTEGER PRIMARY KEY,
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
    home_effect    REAL
);
"""


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(s) for s in spec.split(",") if s.strip()})


def git_state() -> tuple[str | None, int | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "models"], cwd=REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit or None, int(bool(dirty))
    except OSError:
        return None, None


def ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def build(seasons: list[int], out: Path, lambda_team: float) -> dict[int, dict]:
    if not isinstance(lambda_team, float):
        raise TypeError("lambda_team must be a float (an int zeroes Model 2's stabilising term)")
    if not SOURCE_DB.exists():
        raise SystemExit(f"source database not found: {SOURCE_DB}")

    src = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(out)
    dst.executescript(SCHEMA)
    commit, dirty = git_state()
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    balance: dict[int, dict] = {}

    try:
        for season in seasons:
            rows = load_season_games(src, season)
            if not rows:
                print(f"  {season}: no games -- skipped")
                continue
            model = Model2(lambda_team=lambda_team)
            model.fit_rows(rows, season=season)
            summaries = model.point_summary()

            T = len(model.teams_)
            sigma = getattr(model, "_Sigma_eff", None)
            if sigma is not None:
                diag = np.diag(sigma)
                sd_o, sd_d = np.sqrt(diag[1:1 + T]), np.sqrt(diag[1 + T:1 + 2 * T])
            else:
                sd_o = sd_d = [None] * T
            home_effect = float(model.theta_hat_[2 * T + 1])   # layout [mu, o x T, d x T, eta]

            batch = [
                (int(tid), season,
                 float(summaries[tid].adj_o), float(summaries[tid].adj_d),
                 float(summaries[tid].adj_pace),
                 float(sd_o[i]) if sd_o[i] is not None else None,
                 float(sd_d[i]) if sd_d[i] is not None else None)
                for i, tid in enumerate(model.teams_) if tid in summaries
            ]
            venue = collections.Counter(r.h for r in rows)
            dst.execute("DELETE FROM team_ratings WHERE season = ?", (season,))
            dst.executemany("INSERT INTO team_ratings VALUES (?,?,?,?,?,?,?)", batch)
            dst.execute(
                "INSERT OR REPLACE INTO build_info VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (season, built_at, commit, dirty, "Model2", lambda_team, len(rows), len(batch),
                 venue[1], venue[-1], venue[0], home_effect))
            dst.commit()

            bal = collections.Counter()
            for r in rows:
                bal[r.team_id] += r.h
            balance[season] = bal
            print(f"  {season}: {len(batch):>3} teams  {len(rows):>6,} rows  "
                  f"home/away/neutral {venue[1]:>5,}/{venue[-1]:>5,}/{venue[0]:>4,}  "
                  f"home effect {home_effect:+.2f}")
    finally:
        src.close()
        dst.close()
    return balance


def audit(out: Path) -> None:
    con = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    n, s0, s1 = con.execute("SELECT COUNT(*), MIN(season), MAX(season) FROM team_ratings").fetchone()
    gap = con.execute("SELECT MAX(ABS(o - d)) FROM (SELECT AVG(adj_o) o, AVG(adj_d) d "
                      "FROM team_ratings GROUP BY season)").fetchone()[0]
    no_venue = con.execute("SELECT COUNT(*) FROM build_info WHERE games_home = 0 AND n_rows > 100"
                           ).fetchone()[0]
    con.close()
    print(f"\nstored {n:,} team-seasons, {s0}-{s1} -> {out}")
    print(f"  league-mean identity mean(adj_o) == mean(adj_d): max gap {gap:.2e}")
    print(f"  seasons with games but no home games (venue not reaching the model): {no_venue}")


def compare(out: Path, other: Path, balance: dict[int, dict]) -> None:
    """Read-only comparison against another database's ncaa_team_ratings."""
    con = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    con.execute("ATTACH ? AS o", (f"file:{other}?mode=ro",))
    print(f"\nvs {other} ncaa_team_ratings (read-only):")
    print(f"  {'season':<7}{'teams':>6}{'net sd change':>15}{'max |change|':>14}"
          f"{'rank corr net':>15}{'corr(home-away, change)':>25}")
    for season in sorted(balance):
        data = con.execute(
            "SELECT n.team_id, n.adj_o - n.adj_d, x.adj_o - x.adj_d "
            "FROM team_ratings n JOIN o.ncaa_team_ratings x USING (team_id, season) "
            "WHERE season = ?", (season,)).fetchall()
        if len(data) < 10:
            continue
        tid = np.array([d[0] for d in data])
        new, old = np.array([d[1] for d in data]), np.array([d[2] for d in data])
        change = new - old
        imbalance = np.array([balance[season].get(int(t), 0) for t in tid], dtype=float)
        rc = np.corrcoef(ranks(new), ranks(old))[0, 1]
        ci = np.corrcoef(imbalance, change)[0, 1] if imbalance.std() > 0 else float("nan")
        print(f"  {season:<7}{len(data):>6}{change.std():>15.2f}{np.abs(change).max():>14.2f}"
              f"{rc:>15.3f}{ci:>+25.3f}")
    con.close()


def main() -> int:
    global SOURCE_DB
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", default=f"{FIRST_SEASON}-{LAST_SEASON}",
                    help="range like 2002-2026 or list like 2024,2025")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--source", type=Path, default=None,
                    help=f"ncaa.db to read (default {SOURCE_DB})")
    ap.add_argument("--compare", type=Path, default=None,
                    help="another database with ncaa_team_ratings to compare against, read-only")
    args = ap.parse_args()
    if args.source:
        SOURCE_DB = args.source

    seasons = parse_seasons(args.seasons)
    print(f"Model2 lambda_team={LAMBDA_TEAM} -> {args.out}  (source {SOURCE_DB}, read-only)")
    balance = build(seasons, args.out, LAMBDA_TEAM)
    audit(args.out)
    if args.compare:
        if not args.compare.exists():
            print(f"\ncompare target not found: {args.compare}")
        else:
            compare(args.out, args.compare, balance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
