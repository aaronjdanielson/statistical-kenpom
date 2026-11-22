#!/usr/bin/env python
"""Rolling team ratings for every league in portal_pivot_db (Model2Chain).

For each league-season the model is fitted on every game played on or before each
snapshot date: weekly from the season's first game, plus the latest game day, so the last
snapshot of a finished season is that season's full fit. The first snapshot waits until
the league has played at least max(20, teams / 2) games.

The model (models/model2_chain.py) is Model 2's ridge with an exact Gaussian posterior,
made one link of a chain over a league's seasons:

  * a team starts the season at carry_over x its final rating of the league's previous
    season (the most recent one within MAX_GAP seasons that ended before this one began;
    carry_over ** gap across a gap), and moves away from it as games arrive; a team with
    no previous season starts at league average with the wider, stationary prior;
  * a team's offence and defence effects are correlated a priori (od_corr);
  * the home effect starts at the league's previous estimate and may drift HOME_DRIFT_SD
    a season, so strong teams hosting weak ones early no longer inflate it;
  * the noise variance starts at the previous season's.

Shrinkage, carry-over and correlation are set per league type (SETTINGS), chosen by
predicting each next week's games from the games before it over every season in the hub
(2026-11-20; margin error per 100 possessions against Model 2 with lambda 500, which every
league used before: NCAA 17.4 -> 16.3, international 17.2 -> 16.8, national teams
20.9 -> 19.7, NBA 12.90 -> 12.83, G League 13.39 -> 13.25; settings chosen on seasons to
2015 score the same on 2016 on).

Inputs come from portal_pivot_db's standard team_boxscore view (models/hub.py): NCAA-D1,
international leagues, national-team competitions, NBA and G League. The NCAA season in
progress also takes ESPN's games RealGM has not posted yet; a game RealGM has is always
RealGM's.

Output, one pair of files per league-season (portal_pivot_db imports them nightly), under the
hub's handoff directory $PORTAL_HANDOFF_DIR/kenpom/ (default ~/Dropbox/portal_pivot_db/data/handoff):
    rolling/<league_id>/<season>.parquet
        league_id, season, snapshot_date, snapshot_idx, team_id (the league's own team
        id: RealGM's, or NBA stats' for NBA / G League), adj_o, adj_d, adj_net (o - d),
        adj_pace, adj_o_sd, adj_d_sd, adj_net_sd (posterior sd), n_games, n_games_espn,
        games_elapsed (league games so far), home_effect (points per 100 possessions),
        n_components (groups of teams linked by games so far; 1 = all linked), component
        (this team's group, 0 = the largest), pairs_met_share (share of team pairs that
        have met), prior_adj_o, prior_adj_d (the team's starting point in adj_o / adj_d
        units; null = no previous season, started at league average)
    rolling/<league_id>/<season>.json
        key (what the ratings depend on: model and settings, a hash of the input games,
        the previous season's build they start from), created, code_commit, n_snapshots,
        n_games, n_games_espn, last_game_date, low_overlap (the league-season is in
        several unlinked groups, or its two most weakly linked halves are joined by under
        30 games: ratings compare teams across those groups by assumption, not by games),
        n_components, pairs_met_share, bridge_games, final (league mean, home effect and
        noise variance of the last snapshot: the next season starts from them)
A league-season whose key is unchanged is skipped, so a nightly run refits only the
league-seasons that gained games. A season that is refitted changes the starting point,
hence the key, of the league's later seasons: compute --all refits those too.

    PY=~/Dropbox/nba_draft_prediction/.venv/bin/python
    $PY scripts/rolling_ratings.py compute                    # league-seasons in progress (default)
    $PY scripts/rolling_ratings.py compute --all              # every league-season with >= 50 games
    $PY scripts/rolling_ratings.py compute --leagues NCAA-D1 --seasons 2020-2027
    $PY scripts/rolling_ratings.py status
"""
from __future__ import annotations

import argparse
import functools
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

from models import hub, overlap, provenance  # noqa: E402
from models.model2_chain import Model2Chain  # noqa: E402

import logging  # noqa: E402
logging.getLogger("models").setLevel(logging.WARNING)       # Model 2 logs every fit at INFO

OUT = hub.out_dir() / "rolling"       # the hub's handoff directory
LAYOUT = "rolling-ratings-v2"
MODEL = "Model2Chain"
#: per league type: ridge penalty on a team's effects around its starting point, the share of
#: last season's rating a team starts from, and the prior offence-defence correlation
#: (chosen by one-step-ahead prediction over every season in the hub, 2026-11-20)
SETTINGS = {
    "NCAA-D1": dict(lambda_team=500.0, carry_over=0.9, od_corr=0.5),
    "INTL":    dict(lambda_team=500.0, carry_over=0.8, od_corr=0.5),
    "NT":      dict(lambda_team=250.0, carry_over=0.7, od_corr=0.5),
    "NBA":     dict(lambda_team=1500.0, carry_over=0.6, od_corr=0.4),
    "GLEAGUE": dict(lambda_team=1500.0, carry_over=0.5, od_corr=0.0),
}
MAX_GAP = 4                  # a starting point may come from up to this many seasons back
LAMBDA_PACE = 50.0           # Model 2's default
#: home effect, points per 100 possessions: where a league with no history starts, how sure
#: that is, and how far it may drift from one season to the next
HOME_START = {"NCAA-D1": 3.0, "INTL": 2.1, "NT": 0.5, "NBA": 1.5, "GLEAGUE": 0.9}
HOME_START_SD = 1.0
HOME_DRIFT_SD = 0.25
SIGMA2_START = 11000.0       # per-possession-scale noise variance, to put the home prior in possession units
SIGMA2_ROWS = 200.0          # weight of last season's noise variance, in team-game rows
STEP_DAYS = 7
MIN_GAMES_FLOOR = 20
IN_PROGRESS_DAYS = 30        # a league-season with a game this recent is in progress
COLUMNS = ["league_id", "season", "snapshot_date", "snapshot_idx", "team_id", "adj_o", "adj_d", "adj_net", "adj_pace",
           "adj_o_sd", "adj_d_sd", "adj_net_sd", "n_games", "n_games_espn", "games_elapsed", "home_effect",
           "n_components", "component", "pairs_met_share", "prior_adj_o", "prior_adj_d"]


def league_type(league_id: str) -> str:
    return league_id.split("-")[0] if league_id.startswith(("INTL-", "NT-")) else league_id


@functools.lru_cache(maxsize=1)
def code() -> dict:
    """The code and environment of this run (one git call per run, not per build)."""
    return {"code": provenance.code_state(), "environment": provenance.environment()}


def status_path() -> Path:
    """The file every compute run writes (the spoke standard's R5): next to rolling/."""
    return OUT.parent / "last_run.json"


def write_status(started: datetime, args, built: int, unchanged: int, failures: list[dict]) -> None:
    path = status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "started_at": started.isoformat(timespec="seconds"), "command": "compute",
           "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items() if k != "cmd"},
           "built": built, "unchanged": unchanged, "failed": len(failures), "failures": failures, **code()}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(path)


def code_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "models", "scripts"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def paths(league_id: str, season: int, root: Path | None = None) -> tuple[Path, Path]:
    base = (root or OUT) / league_id
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


def key_of(league_id: str, season: int, games: pd.DataFrame, espn: bool, prior: dict | None) -> dict:
    h = hashlib.sha256(pd.util.hash_pandas_object(
        games[["game_id", "date", "team_id", "opp_id", "pts", "poss", "h", "source"]], index=False).values.tobytes())
    st = SETTINGS[league_type(league_id)]
    return {"layout": LAYOUT, "league_id": league_id, "season": int(season), "model": MODEL,
            "lambda_team": st["lambda_team"], "carry_over": st["carry_over"], "od_corr": st["od_corr"], "max_gap": MAX_GAP,
            "home_start": HOME_START[league_type(league_id)], "home_start_sd": HOME_START_SD, "home_drift_sd": HOME_DRIFT_SD,
            "sigma2_rows": SIGMA2_ROWS, "lambda_pace": LAMBDA_PACE, "min_poss": hub.MIN_POSS,
            "step_days": STEP_DAYS, "first_snapshot": f"games >= max({MIN_GAMES_FLOOR}, teams/2)",
            "low_overlap_min_bridge_games": overlap.MIN_BRIDGE_GAMES, "low_overlap_max_bridge_share": overlap.MAX_BRIDGE_SHARE,
            "espn_fill": bool(espn), "inputs_sha256": h.hexdigest()[:16], "n_rows": int(len(games)),
            "prior_from": None if prior is None else {"season": prior["season"], "build": prior["id"]}}


def load_prior(league_id: str, season: int, first_game: pd.Timestamp, root: Path = None) -> dict | None:
    """The starting point for (league, season): the final fit of the league's most recent
    earlier season within MAX_GAP that ended before *first_game* (no look-ahead)."""
    st = SETTINGS[league_type(league_id)]
    rho = st["carry_over"]
    for back in range(1, MAX_GAP + 1):
        pq_path, js_path = paths(league_id, season - back, root or OUT)
        if not (pq_path.exists() and js_path.exists()):
            continue
        meta = json.loads(js_path.read_text())
        if (meta.get("key") or {}).get("layout") != LAYOUT or "final" not in meta:
            raise RuntimeError(f"{league_id} {season - back} was built with an older layout; rebuild it first (compute --all)")
        if not meta.get("last_game_date") or pd.Timestamp(meta["last_game_date"]) >= first_game.normalize():
            continue                                   # that season was still being played: not a starting point
        df = pd.read_parquet(pq_path, columns=["snapshot_idx", "team_id", "adj_o", "adj_d"])
        last = df[df.snapshot_idx == df.snapshot_idx.max()]
        fin = meta["final"]
        widen = sum(rho ** (2 * j) for j in range(back))          # innovation variance accumulated over the gap
        return {"season": season - back, "back": back,
                "mean": {int(t): (rho ** back * (o - fin["mu"]), rho ** back * (fin["mu"] - d))
                         for t, o, d in zip(last.team_id, last.adj_o, last.adj_d)},
                "penalty": st["lambda_team"] / widen,
                "home": fin["home_effect"], "home_sd": fin["home_effect_sd"], "sigma2": fin["sigma2"],
                "id": hashlib.sha256(json.dumps(meta["key"], sort_keys=True).encode()).hexdigest()[:16]}
    return None


def model_for(league_id: str, prior: dict | None) -> Model2Chain:
    """One snapshot's model given the league's starting point."""
    kind = league_type(league_id)
    st = SETTINGS[kind]
    lam, rho = st["lambda_team"], st["carry_over"]
    stationary = lam * (1.0 - rho ** 2)          # a team with no previous season: the chain's long-run spread
    if prior is None:
        return Model2Chain(lambda_team=stationary, lambda_pace=LAMBDA_PACE, od_corr=st["od_corr"],
                           home_prior=HOME_START[kind], lambda_home=SIGMA2_START / HOME_START_SD ** 2)
    m = Model2Chain(lambda_team=stationary, lambda_pace=LAMBDA_PACE, od_corr=st["od_corr"],
                    prior_mean=prior["mean"], prior_penalty={t: (prior["penalty"], prior["penalty"]) for t in prior["mean"]},
                    home_prior=prior["home"], lambda_home=prior["sigma2"] / (prior["home_sd"] ** 2 + HOME_DRIFT_SD ** 2),
                    sigma2_prior=prior["sigma2"], sigma2_rows=SIGMA2_ROWS)
    return m


def fit_snapshots(league_id: str, season: int, games: pd.DataFrame, prior: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """(snapshot rows, summary of the last snapshot)."""
    games = games.dropna(subset=["date"])
    n_teams = len(set(games.team_id) | set(games.opp_id))
    out, info = [], {}
    for idx, d in enumerate(snapshot_dates(games.date, n_teams), start=1):
        g = games[games.date.dt.normalize() <= d]
        m = model_for(league_id, prior).fit_rows(hub.rows(g), season=season)
        s = m.point_summary()
        T = len(m.teams_)
        S = m._Sigma_eff
        var_o, var_d = np.diag(S)[1:1 + T], np.diag(S)[1 + T:1 + 2 * T]
        cov_od = np.array([S[1 + i, 1 + T + i] for i in range(T)])
        n = pd.concat([g.team_id, g.opp_id]).value_counts()     # two rows per game: a team appears twice
        n_espn = pd.concat([g[g.source == "espn"].team_id, g[g.source == "espn"].opp_id]).value_counts()
        mu, home = float(m.theta_hat_[0]), float(m.theta_hat_[2 * T + 1])
        one = g.drop_duplicates("game_id")
        teams, group, n_groups = overlap.groups(one.team_id.values, one.opp_id.values)
        group = dict(zip(teams.tolist(), group.tolist()))
        met = overlap.pairs_met_share(one.team_id.values, one.opp_id.values)
        for i, tid in enumerate(m.teams_):
            tid = int(tid)
            k = s[tid]
            start = prior["mean"].get(tid) if prior else None
            # adj_net = adj_o - adj_d = o + d, so its variance adds the covariance of o and d
            out.append((league_id, season, d.date().isoformat(), idx, tid, float(k.adj_o), float(k.adj_d),
                        float(k.adj_o - k.adj_d), float(k.adj_pace), float(np.sqrt(var_o[i])), float(np.sqrt(var_d[i])),
                        float(np.sqrt(max(var_o[i] + var_d[i] + 2 * cov_od[i], 0.0))),
                        int(n.get(tid, 0) // 2), int(n_espn.get(tid, 0) // 2), int(len(g) // 2), home,
                        n_groups, int(group[tid]), met,
                        None if start is None else mu + start[0], None if start is None else mu - start[1]))
        info = {"mu": mu, "home_effect": home, "home_effect_sd": float(np.sqrt(S[2 * T + 1, 2 * T + 1])),
                "sigma2": float(m._sigma2_poss), "n_components": n_groups, "pairs_met_share": met, "games": one}
    bridge, share = overlap.weakest_split(info["games"].team_id.values, info["games"].opp_id.values) if info["n_components"] == 1 else (0, 0.0)
    summary = {"low_overlap": overlap.low_overlap(info["n_components"], bridge, share), "n_components": info["n_components"],
               "pairs_met_share": info["pairs_met_share"], "bridge_games": int(bridge), "bridge_share": float(share),
               "final": {k: info[k] for k in ("mu", "home_effect", "home_effect_sd", "sigma2")}}
    df = pd.DataFrame(out, columns=COLUMNS)
    for c in ("prior_adj_o", "prior_adj_d"):
        df[c] = df[c].astype("float64")
    return df, summary


def write(league_id: str, season: int, df: pd.DataFrame, key: dict, games: pd.DataFrame, summary: dict,
          root: Path = None) -> None:
    pq_path, js_path = paths(league_id, season, root or OUT)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(pq_path)
    meta = {"key": key, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_commit": code_commit(), "n_snapshots": int(df.snapshot_idx.max()) if len(df) else 0,
            "n_games": int(len(games) // 2), "n_games_espn": int((games.source == "espn").sum() // 2),
            "last_game_date": games.date.max().date().isoformat() if len(games) else None,
            "source": "portal_pivot_db standard team_boxscore" + (" + team_boxscore_espn" if key["espn_fill"] else ""),
            **summary, **code()}
    tmp = js_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(js_path)


def wanted(args, failures: list | None = None) -> dict[str, list[tuple[str, int]]]:
    """database → [(league, season)] to (re)compute. A database that cannot be read is
    recorded in *failures*: the run did not cover it."""
    today = date.today()
    out: dict[str, list[tuple[str, int]]] = {}
    for db in hub.DATABASES:
        try:
            ls = hub.league_seasons(db, min_games=args.min_games)
        except Exception as e:                       # a database being rebuilt: this run cannot cover it
            print(f"  {db}: cannot list league-seasons ({e})")
            if failures is not None:
                failures.append({"database": db, "league_id": None, "season": None,
                                 "error": f"cannot list league-seasons: {type(e).__name__}: {e}"})
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
    started, failures = datetime.now(timezone.utc), []
    done = skipped = 0
    t0 = time.time()
    try:
        todo = wanted(args, failures)
        fill = set(hub.espn_fill_seasons()) if "ncaa" in todo and not args.no_espn else set()
        for db, items in todo.items():
            leagues = sorted({l for l, _ in items})
            seasons = sorted({s for _, s in items})
            espn = sorted(fill & set(seasons)) if db == "ncaa" else []
            try:
                frame = hub.load(db, leagues=leagues, seasons=seasons, espn_seasons=espn)
            except Exception as e:                   # a database being rebuilt: report, carry on
                failures.append({"database": db, "league_id": None, "season": None, "error": f"{type(e).__name__}: {e}"})
                print(f"  {db}: cannot load games ({type(e).__name__}: {e})")
                continue
            groups = dict(tuple(frame.groupby(["league_id", "season"])))
            for league_id, season in sorted(items):      # a league's seasons in order: each starts from the one before
                games = groups.get((league_id, season))
                if games is None or games.empty:
                    continue
                try:
                    prior = load_prior(league_id, season, games.date.min())
                    key = key_of(league_id, season, games, season in espn, prior)
                    _, js = paths(league_id, season)
                    if not args.force and js.exists():
                        try:
                            if json.loads(js.read_text()).get("key") == key:
                                skipped += 1
                                continue
                        except ValueError:
                            pass
                    df, summary = fit_snapshots(league_id, season, games, prior)
                except Exception as e:               # a degenerate league-season: report, carry on
                    failures.append({"database": db, "league_id": league_id, "season": int(season),
                                     "error": f"{type(e).__name__}: {e}"})
                    print(f"  {league_id} {season}: failed ({type(e).__name__}: {e})")
                    continue
                write(league_id, season, df, key, games, summary)
                done += 1
                if args.verbose or done % 100 == 0:
                    print(f"  {league_id} {season}: {df.snapshot_idx.max()} snapshots, {len(games) // 2} games "
                          f"({(games.source == 'espn').sum() // 2} from ESPN), starts from "
                          f"{prior['season'] if prior else 'league average'}  [{done} built, {time.time() - t0:.0f}s]")
    except BaseException as e:                       # anything else: still say so in the status file
        failures.append({"database": None, "league_id": None, "season": None, "error": f"run aborted: {type(e).__name__}: {e}"})
        write_status(started, args, done, skipped, failures)
        raise
    write_status(started, args, done, skipped, failures)
    print(f"rolling ratings: {done} built, {skipped} unchanged, {len(failures)} failed  ({time.time() - t0:.0f}s)"
          f"  -> {status_path()}")
    return 1 if failures else 0                      # any failure fails the hub's nightly step


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
