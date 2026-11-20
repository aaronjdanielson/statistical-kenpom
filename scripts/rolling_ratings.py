#!/usr/bin/env python
"""Rolling team ratings for every league in portal_pivot_db (Model 2).

For each league-season, Model 2 (ridge, exact Gaussian posterior; lambda_team 500, as
the published NCAA and international ratings) is fitted on every game played on or
before each snapshot date: weekly from the season's first game, plus the latest game
day, so the last snapshot of a finished season is that season's full fit. The first
snapshot waits until the league has played at least max(20, teams / 2) games.

Inputs come from portal_pivot_db's standard team_boxscore view (models/hub.py): NCAA-D1,
154 international leagues, 78 national-team competitions, NBA and G League. The NCAA
season in progress also takes ESPN's games RealGM has not posted yet; a game RealGM has
is always RealGM's.

Output, one pair of files per league-season (portal_pivot_db imports them nightly):
    output/rolling/<league_id>/<season>.parquet
        league_id, season, snapshot_date, snapshot_idx, team_id (the league's own team
        id: RealGM's, or NBA stats' for NBA / G League), adj_o, adj_d, adj_net (o - d),
        adj_pace, adj_o_sd, adj_d_sd, adj_net_sd (posterior sd), n_games, n_games_espn,
        games_elapsed (league games so far), home_effect (points per 100 possessions)
    output/rolling/<league_id>/<season>.json
        key (what the ratings depend on: model and settings, a hash of the input games),
        created, code_commit, n_snapshots, n_games, n_games_espn, last_game_date
A league-season whose key is unchanged is skipped, so a nightly run refits only the
league-seasons that gained games.

    PY=~/Dropbox/nba_draft_prediction/.venv/bin/python
    $PY scripts/rolling_ratings.py compute                    # league-seasons in progress (default)
    $PY scripts/rolling_ratings.py compute --all              # every league-season with >= 50 games
    $PY scripts/rolling_ratings.py compute --leagues NCAA-D1 --seasons 2020-2027
    $PY scripts/rolling_ratings.py status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models import hub  # noqa: E402
from models.model2 import Model2  # noqa: E402

import logging  # noqa: E402
logging.getLogger("models").setLevel(logging.WARNING)       # Model 2 logs every fit at INFO

OUT = REPO / "output" / "rolling"
LAMBDA_TEAM = 500.0          # production value (float: an int zeroes Model 2's stabilising term)
LAMBDA_PACE = 50.0           # Model 2's default
STEP_DAYS = 7
MIN_GAMES_FLOOR = 20
IN_PROGRESS_DAYS = 30        # a league-season with a game this recent is in progress
LAYOUT = "rolling-ratings-v1"


def code_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "models", "scripts"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def paths(league_id: str, season: int, root: Path = OUT) -> tuple[Path, Path]:
    base = root / league_id
    return base / f"{season}.parquet", base / f"{season}.json"


def snapshot_dates(dates: pd.Series, n_teams: int) -> list[pd.Timestamp]:
    days = pd.to_datetime(dates).dt.normalize()
    per_day = days.value_counts().sort_index()
    cum = per_day.cumsum() / 2                       # two rows per game
    first, last = per_day.index[0], per_day.index[-1]
    need = max(MIN_GAMES_FLOOR, n_teams / 2)
    out, d = [], first + timedelta(days=STEP_DAYS - 1)
    while d < last:
        if cum[cum.index <= d].iloc[-1] >= need:
            out.append(d)
        d += timedelta(days=STEP_DAYS)
    out.append(last)
    return out


def key_of(league_id: str, season: int, games: pd.DataFrame, espn: bool) -> dict:
    h = hashlib.sha256(pd.util.hash_pandas_object(
        games[["game_id", "date", "team_id", "opp_id", "pts", "poss", "h", "source"]], index=False).values.tobytes())
    return {"layout": LAYOUT, "league_id": league_id, "season": int(season), "model": "Model2",
            "lambda_team": LAMBDA_TEAM, "lambda_pace": LAMBDA_PACE, "min_poss": hub.MIN_POSS,
            "step_days": STEP_DAYS, "first_snapshot": f"games >= max({MIN_GAMES_FLOOR}, teams/2)",
            "espn_fill": bool(espn), "inputs_sha256": h.hexdigest()[:16], "n_rows": int(len(games))}


def fit_snapshots(league_id: str, season: int, games: pd.DataFrame) -> pd.DataFrame:
    games = games.dropna(subset=["date"])
    n_teams = len(set(games.team_id) | set(games.opp_id))
    out = []
    for idx, d in enumerate(snapshot_dates(games.date, n_teams), start=1):
        g = games[games.date.dt.normalize() <= d]
        m = Model2(lambda_team=LAMBDA_TEAM, lambda_pace=LAMBDA_PACE).fit_rows(hub.rows(g), season=season)
        s = m.point_summary()
        T = len(m.teams_)
        S = m._Sigma_eff
        var_o, var_d = np.diag(S)[1:1 + T], np.diag(S)[1 + T:1 + 2 * T]
        cov_od = np.array([S[1 + i, 1 + T + i] for i in range(T)])
        n = pd.concat([g.team_id, g.opp_id]).value_counts()     # two rows per game: a team appears twice
        n_espn = pd.concat([g[g.source == "espn"].team_id, g[g.source == "espn"].opp_id]).value_counts()
        home = float(m.theta_hat_[2 * T + 1])
        for i, tid in enumerate(m.teams_):
            if int(tid) not in s:
                continue
            k = s[int(tid)]
            out.append((league_id, season, d.date().isoformat(), idx, int(tid), float(k.adj_o), float(k.adj_d),
                        float(k.adj_o - k.adj_d), float(k.adj_pace), float(np.sqrt(var_o[i])), float(np.sqrt(var_d[i])),
                        float(np.sqrt(max(var_o[i] + var_d[i] - 2 * cov_od[i], 0.0))),
                        int(n.get(tid, 0) // 2), int(n_espn.get(tid, 0) // 2), int(len(g) // 2), home))
    return pd.DataFrame(out, columns=["league_id", "season", "snapshot_date", "snapshot_idx", "team_id", "adj_o",
                                      "adj_d", "adj_net", "adj_pace", "adj_o_sd", "adj_d_sd", "adj_net_sd",
                                      "n_games", "n_games_espn", "games_elapsed", "home_effect"])


def write(league_id: str, season: int, df: pd.DataFrame, key: dict, games: pd.DataFrame, root: Path = OUT) -> None:
    pq_path, js_path = paths(league_id, season, root)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(pq_path)
    meta = {"key": key, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_commit": code_commit(), "n_snapshots": int(df.snapshot_idx.max()) if len(df) else 0,
            "n_games": int(len(games) // 2), "n_games_espn": int((games.source == "espn").sum() // 2),
            "last_game_date": games.date.max().date().isoformat() if len(games) else None,
            "source": "portal_pivot_db standard team_boxscore" + (" + team_boxscore_espn" if key["espn_fill"] else "")}
    tmp = js_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(js_path)


def wanted(args) -> dict[str, list[tuple[str, int]]]:
    """database → [(league, season)] to (re)compute."""
    today = date.today()
    out: dict[str, list[tuple[str, int]]] = {}
    for db in hub.DATABASES:
        try:
            ls = hub.league_seasons(db, min_games=args.min_games)
        except Exception as e:                       # a database being rebuilt: skip it this run
            print(f"  {db}: cannot list league-seasons ({e})")
            continue
        if args.leagues:
            ls = [x for x in ls if x[0] in args.leagues]
        if args.seasons:
            ls = [x for x in ls if x[1] in args.seasons]
        if not (args.all or args.leagues or args.seasons):           # in progress: a recent game
            conn = hub.connect(db)
            try:
                recent = {(l, int(s)) for l, s in conn.execute(
                    "SELECT DISTINCT league_id, season FROM team_boxscore WHERE date >= ?",
                    ((today - timedelta(days=IN_PROGRESS_DAYS)).isoformat(),))}
            finally:
                conn.close()
            if db == "ncaa":                         # the NCAA season in progress, even before 50 games
                recent.add(("NCAA-D1", hub.current_season(today)))
            have = {(l, s) for l, s, _ in ls}
            ls = [x for x in ls if (x[0], x[1]) in recent] + [
                (l, s, 0) for l, s in sorted(recent) if (l, s) not in have and l == "NCAA-D1"]
        if ls:
            out[db] = [(l, s) for l, s, _ in ls]
    return out


def compute(args) -> int:
    todo = wanted(args)
    fill = set(hub.espn_fill_seasons()) if "ncaa" in todo and not args.no_espn else set()
    done = skipped = failed = 0
    t0 = time.time()
    for db, items in todo.items():
        leagues = sorted({l for l, _ in items})
        seasons = sorted({s for _, s in items})
        espn = sorted(fill & set(seasons)) if db == "ncaa" else []
        frame = hub.load(db, leagues=leagues, seasons=seasons, espn_seasons=espn)
        groups = dict(tuple(frame.groupby(["league_id", "season"])))
        for league_id, season in items:
            games = groups.get((league_id, season))
            if games is None or games.empty:
                continue
            key = key_of(league_id, season, games, season in espn)
            _, js = paths(league_id, season)
            if not args.force and js.exists():
                try:
                    if json.loads(js.read_text()).get("key") == key:
                        skipped += 1
                        continue
                except ValueError:
                    pass
            try:
                df = fit_snapshots(league_id, season, games)
            except Exception as e:                   # a degenerate league-season: report, carry on
                failed += 1
                print(f"  {league_id} {season}: failed ({type(e).__name__}: {e})")
                continue
            write(league_id, season, df, key, games)
            done += 1
            if args.verbose or done % 100 == 0:
                print(f"  {league_id} {season}: {df.snapshot_idx.max()} snapshots, {len(games) // 2} games "
                      f"({(games.source == 'espn').sum() // 2} from ESPN)  [{done} built, {time.time() - t0:.0f}s]")
    print(f"rolling ratings: {done} built, {skipped} unchanged, {failed} failed  ({time.time() - t0:.0f}s)")
    return 1 if failed and not done else 0


def status(args) -> int:
    rows = []
    for js in sorted(OUT.glob("*/*.json")):
        m = json.loads(js.read_text())
        rows.append((js.parent.name, int(js.stem), m.get("n_snapshots"), m.get("n_games"), m.get("n_games_espn"),
                     m.get("last_game_date"), m.get("created")))
    df = pd.DataFrame(rows, columns=["league_id", "season", "snapshots", "games", "espn", "last_game", "created"])
    print(f"{len(df)} league-seasons, {df.league_id.nunique() if len(df) else 0} leagues")
    if len(df):
        print(df.sort_values("created").tail(15).to_string(index=False))
    return 0


def parse_seasons(s: str | None):
    if not s:
        return None
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--all", action="store_true", help="every league-season with >= --min-games games")
    c.add_argument("--leagues", type=lambda s: set(s.split(",")), default=None)
    c.add_argument("--seasons", type=parse_seasons, default=None)
    c.add_argument("--min-games", type=int, default=50)
    c.add_argument("--force", action="store_true", help="refit even when the key is unchanged")
    c.add_argument("--no-espn", action="store_true", help="RealGM only (no ESPN fill for the NCAA season in progress, or seasons RealGM ends early)")
    c.add_argument("--verbose", "-v", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()
    return compute(args) if args.cmd == "compute" else status(args)


if __name__ == "__main__":
    sys.exit(main())
