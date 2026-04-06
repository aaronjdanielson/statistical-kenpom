"""Unit tests for the three rating models.

Uses a small synthetic game set so tests run in milliseconds with no DB.
All three models must:
  1. Fit without error.
  2. Return KenPomSummary for every team that played.
  3. Honour the sample_kenpom_summary contract.
  4. Produce reasonable AdjO/AdjD relative to known team strengths.
"""
import numpy as np
import pytest

from models.base import KenPomSummary
from models.data import GameRow
from models.model1 import Model1
from models.model2 import Model2
from models.model3 import Model3


# --------------------------------------------------------------------------- #
# Synthetic data fixture                                                        #
# --------------------------------------------------------------------------- #

def _make_rows() -> list[GameRow]:
    """
    3 teams, 18 games (each pair plays 6 times, alternating home/away).

    Latent effects centered at 0 (as in the additive model e = μ + o_i - d_j):
      o = offense lift: Team 1: +15,  Team 2:  0, Team 3: -15
      d = defense reduction (positive=better): Team 1: 0, Team 2: +10, Team 3: -10

    Implied KenPom summaries (μ=100):
      AdjO: Team 1=115, Team 2=100, Team 3=85   (1 > 2 > 3)
      AdjD: Team 1=100, Team 2= 90, Team 3=110  (2 < 1 < 3,  lower=better)

    Possessions fixed at 70.
    """
    rng = np.random.default_rng(42)
    o = {1: +15.0, 2:   0.0, 3: -15.0}   # offense effect
    d = {1:   0.0, 2: +10.0, 3: -10.0}   # defense effect (positive → suppresses opponent)
    mu = 100.0
    poss = 70.0
    home_effect = 3.0
    rows: list[GameRow] = []
    gid = 0
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i == j:
                continue
            for rep in range(3):
                gid += 1
                h_i = 1 if rep % 2 == 0 else -1
                e_i = mu + o[i] - d[j] + home_effect * h_i
                e_j = mu + o[j] - d[i] + home_effect * (-h_i)
                noise = rng.normal(0, 8, size=2)
                pts_i = max(1, int(round((e_i + noise[0]) * poss / 100.0)))
                pts_j = max(1, int(round((e_j + noise[1]) * poss / 100.0)))
                rows.append(GameRow(gid, 2024, i, j, pts_i, poss, h_i))
                rows.append(GameRow(gid, 2024, j, i, pts_j, poss, -h_i))
    return rows


@pytest.fixture
def game_rows() -> list[GameRow]:
    return _make_rows()


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _fake_conn(rows: list[GameRow]):
    """Minimal object that load_season_games delegates to when we bypass it."""
    return None  # not used — we patch _rows directly


def _fit_model1(rows: list[GameRow]) -> Model1:
    m = Model1(max_iter=300, tol=1e-6, n_boot=50)
    # Bypass DB — inject rows directly
    teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
    m.teams_ = np.array(teams, dtype=np.int64)
    m._tidx   = {tid: i for i, tid in enumerate(teams)}
    m._rows   = rows
    m.season_ = 2024
    m.theta_hat_ = m._fixed_point(rows)
    return m


def _fit_model2(rows: list[GameRow]) -> Model2:
    m = Model2(lambda_team=20.0, lambda_pace=10.0)
    teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
    m.teams_ = np.array(teams, dtype=np.int64)
    m._tidx   = {tid: i for i, tid in enumerate(teams)}
    m.season_ = 2024
    import numpy as np_
    team_idx = np_.array([m._tidx[r.team_id] for r in rows], dtype=np_.int32)
    opp_idx  = np_.array([m._tidx[r.opp_id]  for r in rows], dtype=np_.int32)
    h_arr    = np_.array([float(r.h)          for r in rows])
    w        = np_.array([r.poss              for r in rows])
    theta_eff, Sigma_eff = m._fit_efficiency(rows)
    theta_pace, Sigma_pace = m._fit_pace(rows)
    m.theta_hat_ = np_.concatenate([theta_eff, theta_pace])
    m._Sigma_eff  = Sigma_eff
    m._Sigma_pace = Sigma_pace
    m._rows = rows
    return m


def _fit_model3(rows: list[GameRow]) -> Model3:
    m = Model3(rank=2, lambda_team=20.0, lambda_pace=10.0, lambda_ab=10.0,
               max_iter=50, tol=1e-4, n_boot=20)
    teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
    m.teams_ = np.array(teams, dtype=np.int64)
    m._tidx   = {tid: i for i, tid in enumerate(teams)}
    m.season_ = 2024
    m._rows   = rows

    from models.model2 import Model2 as M2

    # Stage 1: main effects + pace from Model 2
    m2 = M2(lambda_team=m.lambda_team, lambda_pace=m.lambda_pace)
    m2.teams_ = m.teams_
    m2._tidx  = m._tidx
    theta_main, Sigma_main = m2._fit_efficiency(rows)
    theta_pace, Sigma_pace = m2._fit_pace(rows)
    m._theta_main  = theta_main
    m._Sigma_main  = Sigma_main
    m._sigma2_eff  = m2._sigma2_eff
    m._theta_pace  = theta_pace
    m._Sigma_pace  = Sigma_pace
    m._sigma2_pace = m2._sigma2_pace

    # Stage 2: bilinear on main-effect residuals
    a, b, sigma2_full = m._als_bilinear(rows, theta_main)
    m._a           = a
    m._b           = b
    m._sigma2_full = sigma2_full
    m.theta_hat_   = np.concatenate([theta_main, theta_pace, a.ravel(), b.ravel()])
    return m


# --------------------------------------------------------------------------- #
# KenPomSummary                                                                 #
# --------------------------------------------------------------------------- #

def test_kenpom_summary_net_rtg():
    s = KenPomSummary(adj_o=110.0, adj_d=95.0, adj_pace=68.0)
    assert s.net_rtg == pytest.approx(15.0)


def test_kenpom_summary_frozen():
    s = KenPomSummary(adj_o=110.0, adj_d=95.0, adj_pace=68.0)
    with pytest.raises((AttributeError, TypeError)):
        s.adj_o = 99.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Model 1                                                                       #
# --------------------------------------------------------------------------- #

class TestModel1:
    def test_fit_returns_summaries_for_all_teams(self, game_rows):
        m = _fit_model1(game_rows)
        summary = m.point_summary()
        assert set(summary.keys()) == {1, 2, 3}

    def test_kenpom_values_plausible(self, game_rows):
        m = _fit_model1(game_rows)
        s = m.point_summary()
        # Team 1 should have highest AdjO
        assert s[1].adj_o > s[2].adj_o > s[3].adj_o
        # Team 2 should have lowest AdjD (best defence)
        assert s[2].adj_d < s[1].adj_d < s[3].adj_d

    def test_sample_kenpom_summary_contract(self, game_rows):
        m = _fit_model1(game_rows)
        rng = np.random.default_rng(0)
        draws = m.sample_kenpom_summary(n=30, rng=rng)
        assert len(draws) == 30
        assert all(isinstance(d[1], KenPomSummary) for d in draws)

    def test_bootstrap_draws_vary(self, game_rows):
        m = _fit_model1(game_rows)
        rng = np.random.default_rng(1)
        draws = m.sample_kenpom_summary(n=50, rng=rng)
        adj_o_1 = [d[1].adj_o for d in draws]
        assert np.std(adj_o_1) > 0.01  # draws are not all the same

    def test_league_mean_anchored(self, game_rows):
        m = _fit_model1(game_rows)
        T = len(m.teams_)
        O = m.theta_hat_[:T]
        D = m.theta_hat_[T:2*T]
        # Mean of AdjO and AdjD should be approximately equal (league average)
        assert abs(O.mean() - D.mean()) < 2.0


# --------------------------------------------------------------------------- #
# Model 2                                                                       #
# --------------------------------------------------------------------------- #

class TestModel2:
    def test_fit_returns_summaries_for_all_teams(self, game_rows):
        m = _fit_model2(game_rows)
        summary = m.point_summary()
        assert set(summary.keys()) == {1, 2, 3}

    def test_kenpom_values_plausible(self, game_rows):
        m = _fit_model2(game_rows)
        s = m.point_summary()
        assert s[1].adj_o > s[2].adj_o > s[3].adj_o
        assert s[2].adj_d < s[1].adj_d < s[3].adj_d

    def test_sample_kenpom_summary_contract(self, game_rows):
        m = _fit_model2(game_rows)
        rng = np.random.default_rng(0)
        draws = m.sample_kenpom_summary(n=50, rng=rng)
        assert len(draws) == 50
        for d in draws:
            assert set(d.keys()) == {1, 2, 3}
            for ks in d.values():
                assert isinstance(ks, KenPomSummary)

    def test_posterior_mean_near_point_estimate(self, game_rows):
        m = _fit_model2(game_rows)
        rng = np.random.default_rng(7)
        draws = m.sample_kenpom_summary(n=500, rng=rng)
        mean_adj_o_1 = np.mean([d[1].adj_o for d in draws])
        point_adj_o_1 = m.point_summary()[1].adj_o
        assert abs(mean_adj_o_1 - point_adj_o_1) < 2.0  # Monte Carlo close to point

    def test_adj_pace_positive(self, game_rows):
        m = _fit_model2(game_rows)
        s = m.point_summary()
        for ks in s.values():
            assert ks.adj_pace > 0

    def test_sigma_matrices_positive_definite(self, game_rows):
        m = _fit_model2(game_rows)
        # Cholesky will raise if not PD
        np.linalg.cholesky(m._Sigma_eff)
        np.linalg.cholesky(m._Sigma_pace)


# --------------------------------------------------------------------------- #
# Model 3                                                                       #
# --------------------------------------------------------------------------- #

class TestModel3:
    def test_fit_returns_summaries_for_all_teams(self, game_rows):
        m = _fit_model3(game_rows)
        summary = m.point_summary()
        assert set(summary.keys()) == {1, 2, 3}

    def test_kenpom_values_plausible(self, game_rows):
        m = _fit_model3(game_rows)
        s = m.point_summary()
        assert s[1].adj_o > s[3].adj_o
        assert s[2].adj_d < s[3].adj_d

    def test_summary_excludes_bilinear(self, game_rows):
        """_summary_from_theta must ignore a,b — centering constraint."""
        m = _fit_model3(game_rows)
        # perturb a,b in theta_hat and verify summaries are unchanged
        theta_perturbed = m.theta_hat_.copy()
        T = len(m.teams_)
        k = m.rank
        theta_perturbed[3 * T + 4:] += 999.0  # trash a,b portion
        s_orig  = m._summary_from_theta(m.theta_hat_)
        s_perturb = m._summary_from_theta(theta_perturbed)
        for tid in (1, 2, 3):
            assert s_orig[tid].adj_o    == pytest.approx(s_perturb[tid].adj_o)
            assert s_orig[tid].adj_d    == pytest.approx(s_perturb[tid].adj_d)
            assert s_orig[tid].adj_pace == pytest.approx(s_perturb[tid].adj_pace)

    def test_sample_kenpom_summary_contract(self, game_rows):
        m = _fit_model3(game_rows)
        rng = np.random.default_rng(0)
        draws = m.sample_kenpom_summary(n=10, rng=rng)
        assert len(draws) == 10
        for d in draws:
            assert isinstance(d[1], KenPomSummary)

    def test_bilinear_centering(self, game_rows):
        """a and b should be centered (mean≈0 across teams)."""
        m = _fit_model3(game_rows)
        assert np.abs(m._a.mean(axis=0)).max() < 1e-10
        assert np.abs(m._b.mean(axis=0)).max() < 1e-10

    def test_theta_layout_consistent(self, game_rows):
        """theta_hat_ length should be 3T+4 + 2kT."""
        m = _fit_model3(game_rows)
        T = len(m.teams_)
        k = m.rank
        assert len(m.theta_hat_) == 3 * T + 4 + 2 * k * T


# --------------------------------------------------------------------------- #
# Cross-model: KenPom interface contract                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fit_fn", [_fit_model1, _fit_model2, _fit_model3])
def test_sample_kenpom_summary_derives_from_posterior(fit_fn, game_rows):
    """
    Hard contract: sample_kenpom_summary must equal
    [_summary_from_theta(θ) for θ in sample_posterior(n, rng)].
    """
    m = fit_fn(game_rows)
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    direct = m.sample_kenpom_summary(n=5, rng=rng1)
    from_parts = [m._summary_from_theta(t) for t in m.sample_posterior(n=5, rng=rng2)]
    for d_direct, d_parts in zip(direct, from_parts):
        for tid in (1, 2, 3):
            assert d_direct[tid].adj_o    == pytest.approx(d_parts[tid].adj_o)
            assert d_direct[tid].adj_d    == pytest.approx(d_parts[tid].adj_d)
            assert d_direct[tid].adj_pace == pytest.approx(d_parts[tid].adj_pace)


@pytest.mark.parametrize("fit_fn", [_fit_model1, _fit_model2, _fit_model3])
def test_adj_o_adj_d_in_reasonable_range(fit_fn, game_rows):
    m = fit_fn(game_rows)
    s = m.point_summary()
    for ks in s.values():
        assert 60 < ks.adj_o < 160, f"AdjO out of range: {ks.adj_o}"
        assert 60 < ks.adj_d < 160, f"AdjD out of range: {ks.adj_d}"
        assert 40 < ks.adj_pace < 120, f"AdjPace out of range: {ks.adj_pace}"
