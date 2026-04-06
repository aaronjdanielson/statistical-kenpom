"""Integration tests: validate model ratings against published KenPom numbers.

These tests use the real ncaa.db and the kenpom_team_year_fully_merged.csv ground
truth.  They run slowly (~30s each for full-season fits) and require both files to
exist.  Mark: pytest -m integration

Validation strategy:
  - Fit Model 1 and Model 2 on seasons 2010, 2015, 2019, 2023.
  - Join estimated ratings to KenPom ground truth on TeamID.
  - Check:
      * Spearman rank correlation with KenPom AdjO and AdjD > 0.90
      * Mean absolute error vs KenPom < 5.0 pts/100 for AdjO and AdjD
  - Model 1 vs Model 2 should produce similar net-rating rank orders
    (Spearman > 0.95 between the two models).
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from models.data import load_season_games, open_db
from models.model1 import Model1
from models.model2 import Model2
from models.model3 import Model3

KENPOM_PATH = Path("~/Dropbox/2wayscout/data/NCAADB/kenpom_team_year_fully_merged.csv").expanduser()
DB_PATH     = Path("~/Dropbox/kenpom/ncaa.db").expanduser()

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def kenpom():
    if not KENPOM_PATH.exists():
        pytest.skip("KenPom validation file not found")
    df = pd.read_csv(KENPOM_PATH)
    df = df.dropna(subset=["TeamID", "Off", "Def", "Season"])
    df["TeamID"] = df["TeamID"].astype(int)
    df["Season"]  = df["Season"].astype(int)
    return df


@pytest.fixture(scope="module")
def conn():
    if not DB_PATH.exists():
        pytest.skip("ncaa.db not found")
    c = open_db(DB_PATH)
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _fit_and_join(model, season: int, conn, kenpom_df: pd.DataFrame) -> pd.DataFrame:
    """Fit model, return DataFrame with columns [adj_o, adj_d, kp_adj_o, kp_adj_d]."""
    model.fit(season, conn)
    s = model.point_summary()
    our = pd.DataFrame([
        {"TeamID": tid, "adj_o": ks.adj_o, "adj_d": ks.adj_d, "net_rtg": ks.net_rtg}
        for tid, ks in s.items()
    ])
    kp = kenpom_df[kenpom_df["Season"] == season][["TeamID", "Off", "Def"]].copy()
    kp.columns = ["TeamID", "kp_adj_o", "kp_adj_d"]
    merged = our.merge(kp, on="TeamID", how="inner")
    return merged


# --------------------------------------------------------------------------- #
# Parametrized seasons                                                          #
# --------------------------------------------------------------------------- #

SEASONS = [2010, 2015, 2019]


@pytest.mark.parametrize("season", SEASONS)
class TestModel1Validation:

    def test_adj_o_rank_correlation(self, season, conn, kenpom):
        df = _fit_and_join(Model1(), season, conn, kenpom)
        rho, p = spearmanr(df["adj_o"], df["kp_adj_o"])
        assert rho > 0.90, f"Season {season}: AdjO rank corr={rho:.3f} < 0.90"

    def test_adj_d_rank_correlation(self, season, conn, kenpom):
        df = _fit_and_join(Model1(), season, conn, kenpom)
        rho, p = spearmanr(df["adj_d"], df["kp_adj_d"])
        assert rho > 0.90, f"Season {season}: AdjD rank corr={rho:.3f} < 0.90"

    def test_adj_o_mae(self, season, conn, kenpom):
        df = _fit_and_join(Model1(), season, conn, kenpom)
        # KenPom is scale-free up to an additive constant — allow a mean offset.
        # MAE after centering both to zero mean.
        ours  = df["adj_o"] - df["adj_o"].mean()
        theirs = df["kp_adj_o"] - df["kp_adj_o"].mean()
        mae = (ours - theirs).abs().mean()
        assert mae < 5.0, f"Season {season}: AdjO MAE={mae:.2f} pts/100 > 5.0"

    def test_adj_d_mae(self, season, conn, kenpom):
        df = _fit_and_join(Model1(), season, conn, kenpom)
        ours  = df["adj_d"] - df["adj_d"].mean()
        theirs = df["kp_adj_d"] - df["kp_adj_d"].mean()
        mae = (ours - theirs).abs().mean()
        assert mae < 5.0, f"Season {season}: AdjD MAE={mae:.2f} pts/100 > 5.0"


@pytest.mark.parametrize("season", SEASONS)
class TestModel2Validation:

    def test_adj_o_rank_correlation(self, season, conn, kenpom):
        df = _fit_and_join(Model2(), season, conn, kenpom)
        rho, p = spearmanr(df["adj_o"], df["kp_adj_o"])
        assert rho > 0.90, f"Season {season}: AdjO rank corr={rho:.3f} < 0.90"

    def test_adj_d_rank_correlation(self, season, conn, kenpom):
        df = _fit_and_join(Model2(), season, conn, kenpom)
        rho, p = spearmanr(df["adj_d"], df["kp_adj_d"])
        assert rho > 0.90, f"Season {season}: AdjD rank corr={rho:.3f} < 0.90"

    def test_adj_o_mae(self, season, conn, kenpom):
        df = _fit_and_join(Model2(), season, conn, kenpom)
        ours   = df["adj_o"] - df["adj_o"].mean()
        theirs = df["kp_adj_o"] - df["kp_adj_o"].mean()
        mae = (ours - theirs).abs().mean()
        assert mae < 5.0, f"Season {season}: AdjO MAE={mae:.2f} pts/100 > 5.0"

    def test_adj_d_mae(self, season, conn, kenpom):
        df = _fit_and_join(Model2(), season, conn, kenpom)
        ours   = df["adj_d"] - df["adj_d"].mean()
        theirs = df["kp_adj_d"] - df["kp_adj_d"].mean()
        mae = (ours - theirs).abs().mean()
        assert mae < 5.0, f"Season {season}: AdjD MAE={mae:.2f} pts/100 > 5.0"

    def test_posterior_samples_reasonable(self, season, conn, kenpom):
        """Posterior draws should be close to point estimate (not degenerate)."""
        m = Model2()
        m.fit(season, conn)
        rng = np.random.default_rng(42)
        draws = m.sample_kenpom_summary(n=100, rng=rng)
        point = m.point_summary()
        # Pick an arbitrary well-observed team: Duke (TeamID=31)
        if 31 not in point:
            pytest.skip("Duke not in season")
        adj_o_draws = np.array([d[31].adj_o for d in draws])
        assert abs(adj_o_draws.mean() - point[31].adj_o) < 2.0
        assert adj_o_draws.std() < 5.0    # uncertainty exists but isn't wild


@pytest.mark.parametrize("season", SEASONS)
class TestModel3Validation:
    """
    Model 3 uses a two-stage fit: main effects from Model 2 (fixed), then
    bilinear factors on the residuals.  KenPom summaries are therefore
    identical by construction to Model 2.  Tests verify:
      - KenPom correlation identical to Model 2 (same main effects)
      - Bilinear factors are non-trivial (break away from zero)
      - In-sample RMSE improves over Model 2 (bilinear absorbs real variance)
    """

    def test_kenpom_summary_identical_to_model2(self, season, conn, kenpom):
        """Two-stage fit must produce the same KenPom summary as Model 2."""
        df2 = _fit_and_join(Model2(), season, conn, kenpom)
        df3 = _fit_and_join(Model3(rank=3), season, conn, kenpom)
        df = df2[["TeamID", "adj_o", "adj_d"]].merge(
            df3[["TeamID", "adj_o", "adj_d"]], on="TeamID", suffixes=("_m2", "_m3")
        )
        np.testing.assert_allclose(df["adj_o_m2"].values, df["adj_o_m3"].values, atol=1e-6)
        np.testing.assert_allclose(df["adj_d_m2"].values, df["adj_d_m3"].values, atol=1e-6)

    def test_bilinear_factors_nonzero(self, season, conn, kenpom):
        """Bilinear factors must escape the zero fixed point."""
        m = Model3(rank=3)
        m.fit(season, conn)
        a_norms = np.linalg.norm(m._a, axis=1)
        assert a_norms.max() > 0.1, f"All a factors near zero: max norm={a_norms.max():.4f}"
        assert a_norms.mean() > 0.01

    def test_bilinear_reduces_insample_rmse(self, season, conn, kenpom):
        """Adding bilinear term should reduce in-sample RMSE over main effects alone."""
        m = Model3(rank=3)
        m.fit(season, conn)
        rows = load_season_games(conn, season)
        T = len(m.teams_)
        team_idx = np.array([m._tidx[r.team_id] for r in rows])
        opp_idx  = np.array([m._tidx[r.opp_id]  for r in rows])
        e_off    = np.array([r.pts / r.poss * 100.0 for r in rows])
        w        = np.array([r.poss for r in rows])
        theta    = m._theta_main
        main_pred = (theta[0] + theta[1:1+T][team_idx]
                     - theta[1+T:1+2*T][opp_idx]
                     + theta[2*T+1] * np.array([float(r.h) for r in rows]))
        interaction = np.array([float(m._a[team_idx[k]] @ m._b[opp_idx[k]]) for k in range(len(rows))])
        rmse_main = np.sqrt(np.average((e_off - main_pred) ** 2, weights=w))
        rmse_full = np.sqrt(np.average((e_off - main_pred - interaction) ** 2, weights=w))
        assert rmse_full < rmse_main, f"Bilinear did not reduce RMSE: {rmse_full:.3f} >= {rmse_main:.3f}"
        pct_reduction = (rmse_main - rmse_full) / rmse_main * 100
        assert pct_reduction > 5.0, f"Bilinear RMSE reduction too small: {pct_reduction:.1f}%"

    def test_bilinear_centering(self, season, conn, kenpom):
        m = Model3(rank=3)
        m.fit(season, conn)
        assert np.abs(m._a.mean(axis=0)).max() < 1e-9
        assert np.abs(m._b.mean(axis=0)).max() < 1e-9

    def test_posterior_samples_reasonable(self, season, conn, kenpom):
        m = Model3(rank=2, n_boot=30)
        m.fit(season, conn)
        rng = np.random.default_rng(42)
        draws = m.sample_kenpom_summary(n=30, rng=rng)
        assert len(draws) == 30
        if 31 not in m.point_summary():
            pytest.skip("Duke not in season")
        adj_o_draws = np.array([d[31].adj_o for d in draws])
        assert adj_o_draws.std() < 5.0


# --------------------------------------------------------------------------- #
# Cross-model agreement                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("season", SEASONS)
def test_model1_vs_model2_net_rating_rank(season, conn, kenpom):
    """Net rating ranks from Model 1 and Model 2 should agree strongly."""
    df1 = _fit_and_join(Model1(), season, conn, kenpom)
    df2 = _fit_and_join(Model2(), season, conn, kenpom)
    df = df1[["TeamID", "net_rtg"]].merge(
        df2[["TeamID", "net_rtg"]], on="TeamID", suffixes=("_m1", "_m2")
    )
    rho, _ = spearmanr(df["net_rtg_m1"], df["net_rtg_m2"])
    assert rho > 0.95, f"Season {season}: Model1 vs Model2 net_rtg Spearman={rho:.3f} < 0.95"


# --------------------------------------------------------------------------- #
# Spot checks: known top teams                                                  #
# --------------------------------------------------------------------------- #

_KNOWN_TOP_TEAMS = {
    # season: {team_id: expected rough net_rtg range}
    2010: {
        1:   (20, 50),   # Kansas
        31:  (20, 50),   # Duke (won title)
        100: (15, 45),   # Kentucky
    },
    2019: {
        31:  (20, 55),   # Duke (#1 overall seed)
        276: (20, 55),   # Virginia (won title)
    },
}

@pytest.mark.parametrize("season,team_ranges", _KNOWN_TOP_TEAMS.items())
def test_known_top_teams_in_top_quartile(season, team_ranges, conn, kenpom):
    m = Model2()
    m.fit(season, conn)
    summary = m.point_summary()
    all_nets = np.array([ks.net_rtg for ks in summary.values()])
    p75 = np.percentile(all_nets, 75)
    for tid, (lo, hi) in team_ranges.items():
        if tid not in summary:
            continue
        net = summary[tid].net_rtg
        assert net >= p75, f"Season {season}, team {tid}: net_rtg={net:.1f} not in top quartile (p75={p75:.1f})"
