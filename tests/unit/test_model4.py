"""Unit tests for Model 4 (weekly Kalman state-space model).

Uses the same 3-team, 18-game synthetic fixture as test_models.py so results
can be sanity-checked against Model 2.  All tests run without a DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from models.base import KenPomSummary
from models.data import GameRow
from models.model2 import Model2
from models.model4 import Model4, _forward_filter, _preprocess_weeks, _rts_smooth


# --------------------------------------------------------------------------- #
# Synthetic data fixture  (identical to test_models.py)                        #
# --------------------------------------------------------------------------- #

def _make_rows(n_reps: int = 3) -> list[GameRow]:
    """
    3 teams, games played in n_reps rounds.

    Known latent effects (μ=100):
      o: Team 1=+15, Team 2=0, Team 3=-15
      d: Team 1=0, Team 2=+10, Team 3=-10
    """
    rng = np.random.default_rng(42)
    o = {1: +15.0, 2:   0.0, 3: -15.0}
    d = {1:   0.0, 2: +10.0, 3: -10.0}
    mu, poss, home_effect = 100.0, 70.0, 3.0
    rows: list[GameRow] = []
    gid = 0
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i == j:
                continue
            for rep in range(n_reps):
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


def _make_game_dates(rows: list[GameRow]) -> dict[int, datetime]:
    """Spread games evenly across a 5-month season."""
    gids = sorted({r.game_id for r in rows})
    start = datetime(2023, 11, 1)
    return {gid: start + timedelta(days=7 * k) for k, gid in enumerate(gids)}


@pytest.fixture
def game_rows() -> list[GameRow]:
    return _make_rows()


@pytest.fixture
def game_dates(game_rows) -> dict[int, datetime]:
    return _make_game_dates(game_rows)


@pytest.fixture
def fitted_m4(game_rows, game_dates) -> Model4:
    m = Model4(
        lambda_team=20.0,
        lambda_pace=10.0,
        tau_o=1.0,
        tau_d=1.0,
        optimize_hyperparams=False,
        week_step=7,
    )
    m.fit_rows(game_rows, 2024, game_dates=game_dates)
    return m


# --------------------------------------------------------------------------- #
# Helper: forward_filter on trivial data                                        #
# --------------------------------------------------------------------------- #

class TestForwardFilter:
    def test_empty_week_does_not_update_state(self):
        """A week with no games should leave the state unchanged except for Q."""
        N = 2
        week_arrays = [None]   # one empty week
        m_list, C_list, R_list, ll = _forward_filter(
            week_arrays, N,
            mu=100.0, eta=3.0,
            tau_o2=1.0, tau_d2=1.0,
            sigma2=16.0,
            lambda_team=100.0,
        )
        assert len(m_list) == 1
        # With no observations ll should be exactly 0
        assert ll == 0.0
        # Mean unchanged (zeros)
        np.testing.assert_array_equal(m_list[0], np.zeros(2 * N))

    def test_log_likelihood_finite_with_games(self):
        """One week with one game should produce a finite log-likelihood."""
        N = 2
        tidx = {1: 0, 2: 1}
        rows = [GameRow(1, 2024, 1, 2, 75, 70.0, 0)]
        week_arrays = _preprocess_weeks([rows], tidx)
        _, _, _, ll = _forward_filter(
            week_arrays, N,
            mu=100.0, eta=3.0,
            tau_o2=1.0, tau_d2=1.0,
            sigma2=25.0,
            lambda_team=100.0,
        )
        assert np.isfinite(ll)
        assert ll < 0.0   # log-likelihood of a Gaussian is negative (or zero)

    def test_state_covariance_decreases_after_observation(self):
        """After observing a game the filtered covariance should be <= predictive."""
        N = 2
        tidx = {1: 0, 2: 1}
        rows = [GameRow(1, 2024, 1, 2, 75, 70.0, 0)]
        week_arrays = _preprocess_weeks([rows], tidx)
        _, C_list, R_list, _ = _forward_filter(
            week_arrays, N,
            mu=100.0, eta=3.0,
            tau_o2=0.0, tau_d2=0.0,   # no process noise so R==C_prev
            sigma2=25.0,
            lambda_team=100.0,
        )
        # filtered variance should be <= predictive variance (information gained)
        assert np.diag(C_list[0]).max() <= np.diag(R_list[0]).max() + 1e-12


# --------------------------------------------------------------------------- #
# Helper: RTS smoother                                                           #
# --------------------------------------------------------------------------- #

class TestRTSSmoother:
    def test_smoother_returns_same_length(self, game_rows, game_dates):
        m = Model4(
            lambda_team=20.0, lambda_pace=10.0,
            optimize_hyperparams=False, week_step=7,
        )
        m.fit_rows(game_rows, 2024, game_dates=game_dates)
        smoothed = _rts_smooth(m._m_filtered, m._C_filtered, m._R_predicted)
        assert len(smoothed) == len(m._m_filtered)

    def test_smoother_last_equals_filter(self, game_rows, game_dates):
        """By definition m_T^s = m_T^f (no future data to propagate back)."""
        m = Model4(
            lambda_team=20.0, lambda_pace=10.0,
            optimize_hyperparams=False, week_step=7,
        )
        m.fit_rows(game_rows, 2024, game_dates=game_dates)
        smoothed = _rts_smooth(m._m_filtered, m._C_filtered, m._R_predicted)
        np.testing.assert_allclose(smoothed[-1], m._m_filtered[-1], atol=1e-12)


# --------------------------------------------------------------------------- #
# Model 4 core contract                                                          #
# --------------------------------------------------------------------------- #

class TestModel4:
    def test_fit_returns_self(self, game_rows, game_dates):
        m = Model4(optimize_hyperparams=False)
        result = m.fit_rows(game_rows, 2024, game_dates=game_dates)
        assert result is m

    def test_fit_sets_season(self, fitted_m4):
        assert fitted_m4.season_ == 2024

    def test_all_teams_in_summary(self, fitted_m4):
        summary = fitted_m4.point_summary()
        assert set(summary.keys()) == {1, 2, 3}

    def test_kenpom_values_plausible(self, fitted_m4):
        s = fitted_m4.point_summary()
        assert s[1].adj_o > s[2].adj_o > s[3].adj_o
        assert s[2].adj_d < s[1].adj_d < s[3].adj_d

    def test_adj_pace_positive(self, fitted_m4):
        for ks in fitted_m4.point_summary().values():
            assert ks.adj_pace > 0

    # ── theta_hat_ layout ────────────────────────────────────────────────

    def test_theta_layout_length(self, fitted_m4):
        """theta_hat_ must follow Model 2 layout: 3T + 4."""
        T = len(fitted_m4.teams_)
        assert len(fitted_m4.theta_hat_) == 3 * T + 4

    def test_theta_mu_at_index_zero(self, fitted_m4):
        """Index 0 must equal the fitted μ."""
        assert fitted_m4.theta_hat_[0] == pytest.approx(fitted_m4._mu)

    def test_theta_eta_at_correct_index(self, fitted_m4):
        """Home advantage η must be at index 2T+1."""
        T = len(fitted_m4.teams_)
        assert fitted_m4.theta_hat_[2 * T + 1] == pytest.approx(fitted_m4._eta)

    # ── Posterior sampling ────────────────────────────────────────────────

    def test_sample_posterior_count(self, fitted_m4):
        rng = np.random.default_rng(0)
        draws = fitted_m4.sample_posterior(20, rng)
        assert len(draws) == 20

    def test_sample_posterior_correct_length(self, fitted_m4):
        T = len(fitted_m4.teams_)
        rng = np.random.default_rng(0)
        draws = fitted_m4.sample_posterior(5, rng)
        for theta in draws:
            assert len(theta) == 3 * T + 4

    def test_sample_posterior_finite(self, fitted_m4):
        rng = np.random.default_rng(1)
        draws = fitted_m4.sample_posterior(50, rng)
        for theta in draws:
            assert np.all(np.isfinite(theta))

    def test_sample_kenpom_derives_from_posterior(self, fitted_m4):
        """sample_kenpom_summary must equal [_summary_from_theta(θ) for θ ∈ posterior]."""
        rng1 = np.random.default_rng(77)
        rng2 = np.random.default_rng(77)
        direct = fitted_m4.sample_kenpom_summary(n=5, rng=rng1)
        from_parts = [
            fitted_m4._summary_from_theta(t)
            for t in fitted_m4.sample_posterior(n=5, rng=rng2)
        ]
        for d_dir, d_pts in zip(direct, from_parts):
            for tid in (1, 2, 3):
                assert d_dir[tid].adj_o    == pytest.approx(d_pts[tid].adj_o)
                assert d_dir[tid].adj_d    == pytest.approx(d_pts[tid].adj_d)
                assert d_dir[tid].adj_pace == pytest.approx(d_pts[tid].adj_pace)

    # ── Trajectory outputs ────────────────────────────────────────────────

    def test_trajectory_length_matches_weeks(self, fitted_m4):
        traj = fitted_m4.point_summary_trajectory()
        assert len(traj) == len(fitted_m4._week_cutoffs)

    def test_trajectory_entries_are_tuples(self, fitted_m4):
        for cutoff, summary in fitted_m4.point_summary_trajectory():
            assert isinstance(cutoff, datetime)
            assert isinstance(summary, dict)
            for ks in summary.values():
                assert isinstance(ks, KenPomSummary)

    def test_rts_smoother_length(self, fitted_m4):
        smoothed = fitted_m4.rts_smoother()
        assert len(smoothed) == len(fitted_m4._week_cutoffs)

    def test_rts_smoother_last_equals_filter(self, fitted_m4):
        """Terminal smoothed state equals the terminal filtered state."""
        traj   = fitted_m4.point_summary_trajectory()
        smooth = fitted_m4.rts_smoother()
        _, s_last  = traj[-1]
        _, sm_last = smooth[-1]
        for tid in (1, 2, 3):
            assert s_last[tid].adj_o  == pytest.approx(sm_last[tid].adj_o,  abs=1e-6)
            assert s_last[tid].adj_d  == pytest.approx(sm_last[tid].adj_d,  abs=1e-6)

    # ── game_dates=None fallback ──────────────────────────────────────────

    def test_fit_rows_without_game_dates(self, game_rows):
        """fit_rows must succeed when game_dates is not supplied."""
        m = Model4(optimize_hyperparams=False, week_step=1)
        m.fit_rows(game_rows, 2024, game_dates=None)
        summary = m.point_summary()
        assert set(summary.keys()) == {1, 2, 3}

    # ── Near-static approximation ─────────────────────────────────────────

    def test_near_static_preserves_ordering(self, game_rows, game_dates):
        """With tau→0 and no optimisation, Model4 should still rank teams correctly.

        Note: Model4 and Model2 are NOT equivalent even with tau=0 because Model4
        pre-estimates μ/η from Model2 then runs the Kalman filter only on (o, d),
        while Model2 jointly solves all parameters.  The correct invariant is that
        team ordering is preserved, not that ratings are numerically close.
        """
        m4 = Model4(
            lambda_team=20.0,
            lambda_pace=10.0,
            tau_o=1e-4,
            tau_d=1e-4,
            optimize_hyperparams=False,
            week_step=7,
        )
        m4.fit_rows(game_rows, 2024, game_dates=game_dates)
        s4 = m4.point_summary()

        # Team 1 best offense, Team 3 worst offense
        assert s4[1].adj_o > s4[2].adj_o > s4[3].adj_o
        # Team 2 best defense (lowest AdjD), Team 3 worst
        assert s4[2].adj_d < s4[1].adj_d < s4[3].adj_d
        # All values in plausible range
        for ks in s4.values():
            assert 60.0 < ks.adj_o < 160.0
            assert 60.0 < ks.adj_d < 160.0

    # ── Hyperparameter optimisation ───────────────────────────────────────

    def test_optimize_hyperparams_runs(self, game_rows, game_dates):
        """Optimisation must finish and set tau_o_, tau_d_."""
        m = Model4(
            lambda_team=20.0,
            lambda_pace=10.0,
            optimize_hyperparams=True,
            week_step=7,
            max_opt_iter=20,
        )
        m.fit_rows(game_rows, 2024, game_dates=game_dates)
        assert hasattr(m, "tau_o_")
        assert hasattr(m, "tau_d_")
        assert m.tau_o_ > 0
        assert m.tau_d_ > 0

    def test_optimize_improves_or_ties_log_lik(self, game_rows, game_dates):
        """Optimised log-lik must be >= fixed-hyperparameter log-lik."""
        m_fixed = Model4(
            lambda_team=20.0, lambda_pace=10.0,
            tau_o=1.0, tau_d=1.0,
            optimize_hyperparams=False, week_step=7,
        )
        m_fixed.fit_rows(game_rows, 2024, game_dates=game_dates)

        m_opt = Model4(
            lambda_team=20.0, lambda_pace=10.0,
            tau_o=1.0, tau_d=1.0,
            optimize_hyperparams=True, week_step=7, max_opt_iter=50,
        )
        m_opt.fit_rows(game_rows, 2024, game_dates=game_dates)

        assert m_opt._log_lik >= m_fixed._log_lik - 1e-4

    # ── Predict efficiency interface ──────────────────────────────────────

    def test_predict_efficiency_finite(self, fitted_m4, game_rows):
        pred = fitted_m4.predict_efficiency(game_rows)
        assert np.all(np.isfinite(pred))
        assert len(pred) == len(game_rows)

    def test_unseen_team_fallback(self, fitted_m4):
        """Unseen teams default to zero effect — prediction stays finite."""
        unseen = GameRow(9999, 2024, 999, 1, 70, 70.0, 0)
        pred = fitted_m4.predict_efficiency([unseen])
        assert np.isfinite(pred[0])
        assert 60.0 < pred[0] < 160.0
