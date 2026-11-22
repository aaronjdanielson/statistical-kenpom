"""Model2Chain (one link of the season-to-season chain), the overlap statistics, and the
rolling builder's chain: starting points, no look-ahead, the cascade of keys."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models import overlap
from models.data import GameRow
from models.model2 import Model2
from models.model2_chain import Model2Chain

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rolling_ratings", REPO / "scripts" / "rolling_ratings.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def league(seed, teams=range(8), games=120, strength=None, season=2025, start_gid=0, sigma2=11000.0):
    rng = np.random.default_rng(seed)
    teams = list(teams)
    o = strength[0] if strength else {t: rng.normal(0, 5) for t in teams}
    d = strength[1] if strength else {t: rng.normal(0, 5) for t in teams}
    rows = []
    for g in range(games):
        i, j = rng.choice(teams, 2, replace=False)
        poss = float(rng.normal(70, 4))
        for a, b, h in ((i, j, 1), (j, i, -1)):
            eff = 100 + o[a] - d[b] + 2.5 * h + rng.normal(0, np.sqrt(sigma2 / poss))
            rows.append(GameRow(start_gid + g, season, int(a), int(b), eff * poss / 100, poss, h))
    return rows, (o, d)


def frame(rows, league_id="INTL-900", first="2024-10-01", per_day=4):
    """GameRows -> the frame models.hub.load returns, a few games a day."""
    days = pd.Timestamp(first) + pd.to_timedelta([r.game_id % 100000 // per_day for r in rows], unit="D")
    return pd.DataFrame({"league_id": league_id, "season": [r.season for r in rows], "game_id": [r.game_id for r in rows],
                         "date": days, "team_id": [r.team_id for r in rows], "opp_id": [r.opp_id for r in rows],
                         "pts": [r.pts for r in rows], "poss": [r.poss for r in rows], "h": [r.h for r in rows], "source": "realgm"})


# ── Model2Chain ───────────────────────────────────────────────────────────────

def test_with_no_prior_it_is_model2():
    rows, _ = league(1)
    a, b = Model2(lambda_team=300.0).fit_rows(rows, 2025), Model2Chain(lambda_team=300.0).fit_rows(rows, 2025)
    assert np.allclose(a.theta_hat_, b.theta_hat_, atol=1e-10) and np.allclose(a._Sigma_eff, b._Sigma_eff, atol=1e-10)


def test_prior_mean_is_where_a_team_starts_and_games_move_it():
    rows, (o, d) = league(2, games=200)
    prior = {t: (2.0 * (t - 3.5), -1.5 * (t - 3.5)) for t in range(8)}                    # nothing to do with the truth
    few = Model2Chain(lambda_team=300.0, prior_mean=prior).fit_rows(rows[:8], 2025)        # 4 games
    many = Model2Chain(lambda_team=300.0, prior_mean=prior).fit_rows(rows, 2025)

    def centred(model, values):
        T = len(model.teams_)
        est = model.theta_hat_[1:1 + 2 * T].copy()
        ref = np.array([values[0][t] for t in model.teams_] + [values[1][t] for t in model.teams_], dtype=float)
        for x in (est, ref):
            x[:T] -= x[:T].mean(); x[T:] -= x[T:].mean()
        return est, ref

    start = ({t: v[0] for t, v in prior.items()}, {t: v[1] for t, v in prior.items()})
    near_start_few = np.corrcoef(*centred(few, start))[0, 1]
    near_start_many = np.corrcoef(*centred(many, start))[0, 1]
    assert near_start_few > near_start_many + 0.2                                          # four games: still pulled to the start
    assert np.corrcoef(*centred(many, (o, d)))[0, 1] > 0.8                                 # 200 games: the truth


def test_home_effect_is_anchored_only_when_asked():
    rows, _ = league(3, games=12)
    free = Model2Chain(lambda_team=300.0).fit_rows(rows, 2025)
    held = Model2Chain(lambda_team=300.0, home_prior=2.5, lambda_home=1e7).fit_rows(rows, 2025)
    T = len(held.teams_)
    assert abs(held.theta_hat_[2 * T + 1] - 2.5) < 0.05 and abs(free.theta_hat_[2 * T + 1] - 2.5) > 0.05


def test_offence_defence_correlation_moves_a_teams_two_effects_together():
    rows, _ = league(4, games=40)
    ind = Model2Chain(lambda_team=500.0).fit_rows(rows, 2025)
    cor = Model2Chain(lambda_team=500.0, od_corr=0.6).fit_rows(rows, 2025)
    T = len(ind.teams_)
    c_ind = np.mean([ind._Sigma_eff[1 + i, 1 + T + i] for i in range(T)])
    c_cor = np.mean([cor._Sigma_eff[1 + i, 1 + T + i] for i in range(T)])
    assert c_cor > c_ind + 0.5
    with pytest.raises(ValueError):
        Model2Chain(od_corr=1.0)


def test_noise_variance_prior_steadies_the_first_weeks_and_fades():
    rows, _ = league(5, games=400)
    early = Model2Chain(lambda_team=300.0, sigma2_prior=11000.0, sigma2_rows=200.0).fit_rows(rows[:20], 2025)
    late = Model2Chain(lambda_team=300.0, sigma2_prior=30000.0, sigma2_rows=200.0).fit_rows(rows, 2025)
    assert 8000 < early._sigma2_poss < 14000            # 20 rows alone could say almost anything
    assert 9500 < late._sigma2_poss < 16500             # 800 rows outweigh a prior that is far too high


# ── overlap ───────────────────────────────────────────────────────────────────

def test_groups_pairs_and_the_low_overlap_flag():
    t = np.array([1, 2, 3, 10, 11, 12, 1]); o = np.array([2, 3, 1, 11, 12, 10, 2])      # two triangles that never meet
    teams, group, n = overlap.groups(t, o)
    assert n == 2 and dict(zip(teams, group)) == {1: 0, 2: 0, 3: 0, 10: 1, 11: 1, 12: 1}   # same size: smallest id is group 0
    assert overlap.pairs_met_share(t, o) == pytest.approx(6 / 15)
    assert overlap.low_overlap(n, 0, 0.0)
    # two round-robin blocks of 6 joined by 3 games: connected, but barely
    a = [(i, j) for i in range(6) for j in range(i + 1, 6)] * 2; b = [(i + 10, j + 10) for i, j in a]
    g = np.array(a + b + [(0, 10), (1, 11), (2, 12)])
    bridge, share = overlap.weakest_split(g[:, 0], g[:, 1])
    assert bridge == 3 and share < 0.2 and overlap.low_overlap(1, bridge, share)
    # one full double round robin of 10: nothing weak about it
    rr = np.array([(i, j) for i in range(10) for j in range(i + 1, 10)] * 2)
    bridge, share = overlap.weakest_split(rr[:, 0], rr[:, 1])
    assert not overlap.low_overlap(1, bridge, share) and overlap.pairs_met_share(rr[:, 0], rr[:, 1]) == 1.0


# ── the chain in the rolling builder ──────────────────────────────────────────

def build(tmp_path, season, games, prior):
    key = R.key_of("INTL-900", season, games, False, prior)
    df, summary = R.fit_snapshots("INTL-900", season, games, prior)
    R.write("INTL-900", season, df, key, games, summary, root=tmp_path)
    return df, key


def test_a_season_starts_from_the_one_before_and_never_from_one_still_being_played(tmp_path):
    rows1, strength = league(6, games=160, season=2025)
    rows2, _ = league(7, games=160, season=2026, strength=strength, start_gid=100000)
    g1, g2 = frame(rows1, first="2024-10-01"), frame(rows2, first="2025-10-01")
    assert R.load_prior("INTL-900", 2025, g1.date.min(), root=tmp_path) is None
    df1, key1 = build(tmp_path, 2025, g1, None)
    assert df1.prior_adj_o.isna().all() and list(df1.columns) == R.COLUMNS

    prior = R.load_prior("INTL-900", 2026, g2.date.min(), root=tmp_path)
    assert prior["season"] == 2025 and prior["back"] == 1
    last = df1[df1.snapshot_idx == df1.snapshot_idx.max()].set_index("team_id")
    mu = json.loads((tmp_path / "INTL-900" / "2025.json").read_text())["final"]["mu"]
    rho = R.SETTINGS["INTL"]["carry_over"]
    assert prior["mean"][3] == pytest.approx((rho * (last.adj_o[3] - mu), rho * (mu - last.adj_d[3])))

    df2, key2 = build(tmp_path, 2026, g2, prior)
    first = df2[df2.snapshot_idx == 1].set_index("team_id")
    assert first.prior_adj_o.notna().all()
    # after a few games a team sits between its starting point and what a prior-free fit says
    free, _ = R.fit_snapshots("INTL-900", 2026, g2, None)
    free = free[free.snapshot_idx == 1].set_index("team_id")
    start_net = first.prior_adj_o - first.prior_adj_d
    assert np.abs(first.adj_net - start_net).mean() < 0.8 * np.abs(free.adj_net - start_net).mean()
    assert key2["prior_from"]["season"] == 2025 and key1["prior_from"] is None

    # the cascade: last season refitted on different games -> this season's key changes
    build(tmp_path, 2025, g1.iloc[:-2], None)
    again = R.load_prior("INTL-900", 2026, g2.date.min(), root=tmp_path)
    assert R.key_of("INTL-900", 2026, g2, False, again) != key2

    # no look-ahead: a "previous" season whose games run past this season's first game is not a starting point
    assert R.load_prior("INTL-900", 2026, g1.date.min(), root=tmp_path) is None


def test_a_gap_of_seasons_discounts_and_widens_the_starting_point(tmp_path):
    rows1, _ = league(8, games=160, season=2022)
    build(tmp_path, 2022, frame(rows1, first="2021-10-01"), None)
    rho, lam = R.SETTINGS["INTL"]["carry_over"], R.SETTINGS["INTL"]["lambda_team"]
    p1 = R.load_prior("INTL-900", 2023, pd.Timestamp("2022-10-01"), root=tmp_path)
    p3 = R.load_prior("INTL-900", 2025, pd.Timestamp("2024-10-01"), root=tmp_path)
    assert p3["back"] == 3 and p3["mean"][2][0] == pytest.approx(p1["mean"][2][0] * rho ** 2)
    assert p1["penalty"] == lam and p3["penalty"] == pytest.approx(lam / (1 + rho ** 2 + rho ** 4))
    assert R.load_prior("INTL-900", 2027, pd.Timestamp("2026-10-01"), root=tmp_path) is None      # beyond MAX_GAP


def test_unlinked_groups_are_flagged_and_numbered(tmp_path):
    a, _ = league(9, teams=range(6), games=70, season=2025)
    b, _ = league(10, teams=range(20, 26), games=70, season=2025, start_gid=70)
    df, summary = R.fit_snapshots("INTL-900", 2025, frame(a + b, per_day=8), None)
    last = df[df.snapshot_idx == df.snapshot_idx.max()]
    assert summary["low_overlap"] and summary["n_components"] == 2 and summary["bridge_games"] == 0
    assert set(last.n_components) == {2} and set(last[last.team_id < 10].component) == {0} and set(last[last.team_id >= 20].component) == {1}
    one, s1 = R.fit_snapshots("INTL-900", 2025, frame(a), None)
    assert not s1["low_overlap"] and set(one.component) == {0}


def test_a_run_with_any_failure_exits_non_zero(tmp_path, monkeypatch):
    """The hub's nightly step fails only on a non-zero exit: one failed league-season must
    give one, even when others were built (2026-11-21: 45 failed, 1 built, exit 0)."""
    rows_a, _ = league(11, games=120, season=2025)
    rows_b, _ = league(12, games=120, season=2025, start_gid=500)
    games = pd.concat([frame(rows_a, league_id="INTL-900"), frame(rows_b, league_id="INTL-901")], ignore_index=True)
    out = tmp_path / "rolling"
    monkeypatch.setattr(R, "OUT", out)
    monkeypatch.setattr(R, "wanted", lambda args, failures=None: {"intl": [("INTL-900", 2025), ("INTL-901", 2025)]})
    monkeypatch.setattr(R.hub, "load", lambda db, **kw: games)
    old = out / "INTL-901"                          # INTL-901's previous season is in the old layout
    old.mkdir(parents=True)
    (old / "2024.json").write_text(json.dumps({"key": {"layout": "rolling-ratings-v1"}, "last_game_date": "2024-05-01"}))
    (old / "2024.parquet").write_bytes(b"")
    assert R.compute(argparse.Namespace(no_espn=True, force=False, verbose=False)) == 1
    assert (out / "INTL-900" / "2025.json").exists() and not (out / "INTL-901" / "2025.json").exists()
    # the status file (the spoke standard's R5) says what happened
    st = json.loads((tmp_path / "last_run.json").read_text())
    assert (st["built"], st["unchanged"], st["failed"]) == (1, 0, 1)
    assert st["failures"][0]["league_id"] == "INTL-901" and "older layout" in st["failures"][0]["error"]
    assert len(st["code"]["tree_sha256"]) == 64 and st["finished_at"]
    meta = json.loads((out / "INTL-900" / "2025.json").read_text())
    assert meta["code"]["tree_sha256"] == st["code"]["tree_sha256"] and "numpy" in meta["environment"]
    # a second run changes nothing and says so
    (old / "2024.json").unlink(); (old / "2024.parquet").unlink()
    assert R.compute(argparse.Namespace(no_espn=True, force=False, verbose=False)) == 0
    st = json.loads((tmp_path / "last_run.json").read_text())
    assert (st["built"], st["unchanged"], st["failed"]) == (1, 1, 0)


def test_code_tree_hash_follows_the_code_not_the_commit(tmp_path):
    from models import provenance
    (tmp_path / "models").mkdir(); (tmp_path / "scripts").mkdir()
    (tmp_path / "models" / "a.py").write_text("x = 1\n")
    (tmp_path / "scripts" / "rolling_ratings.py").write_text("y = 2\n")
    first = provenance.tree_sha256(tmp_path)
    (tmp_path / "models" / "notes.md").write_text("not code")
    assert provenance.tree_sha256(tmp_path) == first
    (tmp_path / "models" / "a.py").write_text("x = 2\n")
    assert provenance.tree_sha256(tmp_path) != first
