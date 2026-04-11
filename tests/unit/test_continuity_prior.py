"""
Unit tests for Model2ContinuityPrior and build_continuity_prior.

Three core invariants (per the build plan):
  1. r=0 for all teams → prior mean=0, regardless of prev_effects
  2. r=1 with tiny tau_lo → estimates strongly anchored to prev-season values
  3. prev_effects=0 for all teams → solution has same structure as a shifted ridge
     with zero mean (indistinguishable from team-specific ridge without prior shift)
"""
from __future__ import annotations

import numpy as np
import pytest

from models.data import GameRow
from models.model2 import Model2
from models.model2_continuity import Model2ContinuityPrior
from models.priors import build_continuity_prior, extract_prev_effects


# ── Synthetic fixture ──────────────────────────────────────────────────────────

def _make_rows(n_reps: int = 3) -> list[GameRow]:
    """3 teams, n_reps rounds each — identical fixture to test_models.py."""
    rng = np.random.default_rng(42)
    o = {1: +10.0, 2: 0.0, 3: -10.0}
    d = {1:   0.0, 2: +5.0, 3:  -5.0}
    mu, poss = 100.0, 70.0
    rows: list[GameRow] = []
    gid = 0
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i == j:
                continue
            for _ in range(n_reps):
                gid += 1
                noise = rng.normal(0, 8, 2)
                pts_i = max(1, int(round((mu + o[i] - d[j] + noise[0]) * poss / 100)))
                pts_j = max(1, int(round((mu + o[j] - d[i] + noise[1]) * poss / 100)))
                rows.append(GameRow(gid, 2024, i, j, pts_i, poss, 0))
                rows.append(GameRow(gid, 2024, j, i, pts_j, poss, 0))
    return rows


ROWS   = _make_rows()
TEAMS  = np.array([1, 2, 3], dtype=np.int64)
SIGMA2 = 160.0


# ── build_continuity_prior ─────────────────────────────────────────────────────

class TestBuildContinuityPrior:

    def test_r_zero_gives_zero_mean(self):
        """r=0 for all teams → prior mean is zero regardless of prev_effects."""
        prev = {1: (10.0, -5.0), 2: (3.0, 2.0), 3: (-8.0, 4.0)}
        m, _ = build_continuity_prior(
            TEAMS, prev, r_minutes={},   # all missing → r=0
            tau_o_lo=1.0, tau_o_hi=5.0,
            tau_d_lo=1.0, tau_d_hi=5.0,
            sigma2_ref=SIGMA2,
        )
        np.testing.assert_array_equal(m[1:4], [0.0, 0.0, 0.0])   # o block
        np.testing.assert_array_equal(m[4:7], [0.0, 0.0, 0.0])   # d block

    def test_r_one_mean_equals_prev_effects(self):
        """r=1 → prior mean is exactly the previous-season effects."""
        prev = {1: (10.0, -3.0), 2: (2.0, 1.0), 3: (-5.0, 4.0)}
        r    = {1: 1.0, 2: 1.0, 3: 1.0}
        m, _ = build_continuity_prior(
            TEAMS, prev, r,
            tau_o_lo=1.0, tau_o_hi=5.0,
            tau_d_lo=1.0, tau_d_hi=5.0,
            sigma2_ref=SIGMA2,
        )
        np.testing.assert_allclose(m[1:4], [10.0, 2.0, -5.0])   # o
        np.testing.assert_allclose(m[4:7], [-3.0, 1.0,  4.0])   # d

    def test_r_zero_precision_equals_sigma2_over_tau_hi_sq(self):
        """At r=0, precision = sigma2_ref / tau_hi²."""
        tau_hi = 5.0
        _, p = build_continuity_prior(
            TEAMS, {}, {},
            tau_o_lo=1.0, tau_o_hi=tau_hi,
            tau_d_lo=1.0, tau_d_hi=tau_hi,
            sigma2_ref=SIGMA2,
        )
        expected = SIGMA2 / tau_hi ** 2
        np.testing.assert_allclose(p[1:4], expected)
        np.testing.assert_allclose(p[4:7], expected)

    def test_r_one_precision_equals_sigma2_over_tau_lo_sq(self):
        """At r=1, precision = sigma2_ref / tau_lo²."""
        tau_lo = 1.5
        r = {1: 1.0, 2: 1.0, 3: 1.0}
        _, p = build_continuity_prior(
            TEAMS, {}, r,
            tau_o_lo=tau_lo, tau_o_hi=7.0,
            tau_d_lo=tau_lo, tau_d_hi=7.0,
            sigma2_ref=SIGMA2,
        )
        expected = SIGMA2 / tau_lo ** 2
        np.testing.assert_allclose(p[1:4], expected)
        np.testing.assert_allclose(p[4:7], expected)

    def test_mu_eta_precision_zero(self):
        """mu (index 0) and eta (index 2T+1) must have zero precision."""
        _, p = build_continuity_prior(
            TEAMS, {}, {},
            tau_o_lo=1.0, tau_o_hi=5.0,
            tau_d_lo=1.0, tau_d_hi=5.0,
            sigma2_ref=SIGMA2,
        )
        assert p[0] == 0.0
        assert p[2 * len(TEAMS) + 1] == 0.0

    def test_precision_monotone_in_r(self):
        """Higher r → higher precision (tighter prior)."""
        prev = {1: (5.0, -2.0), 2: (0.0, 0.0), 3: (-5.0, 2.0)}
        r_lo = {1: 0.2, 2: 0.2, 3: 0.2}
        r_hi = {1: 0.8, 2: 0.8, 3: 0.8}
        _, p_lo = build_continuity_prior(
            TEAMS, prev, r_lo,
            tau_o_lo=1.5, tau_o_hi=6.0,
            tau_d_lo=1.5, tau_d_hi=6.0,
            sigma2_ref=SIGMA2,
        )
        _, p_hi = build_continuity_prior(
            TEAMS, prev, r_hi,
            tau_o_lo=1.5, tau_o_hi=6.0,
            tau_d_lo=1.5, tau_d_hi=6.0,
            sigma2_ref=SIGMA2,
        )
        assert np.all(p_hi[1:4] > p_lo[1:4])   # higher r → higher precision
        assert np.all(p_hi[4:7] > p_lo[4:7])

    def test_prev_var_inflates_prior_variance(self):
        """Passing prev_var should increase prior variance → decrease precision."""
        prev      = {1: (5.0, -2.0), 2: (0.0, 0.0), 3: (-5.0, 2.0)}
        r         = {1: 0.8, 2: 0.8, 3: 0.8}
        prev_var  = {1: (4.0, 4.0), 2: (4.0, 4.0), 3: (4.0, 4.0)}
        _, p_base = build_continuity_prior(
            TEAMS, prev, r,
            tau_o_lo=2.0, tau_o_hi=6.0,
            tau_d_lo=2.0, tau_d_hi=6.0,
            sigma2_ref=SIGMA2,
        )
        _, p_infl = build_continuity_prior(
            TEAMS, prev, r,
            tau_o_lo=2.0, tau_o_hi=6.0,
            tau_d_lo=2.0, tau_d_hi=6.0,
            sigma2_ref=SIGMA2,
            prev_var=prev_var,
        )
        # Inflated variance → lower precision
        assert np.all(p_infl[1:4] < p_base[1:4])
        assert np.all(p_infl[4:7] < p_base[4:7])


# ── extract_prev_effects ───────────────────────────────────────────────────────

class TestExtractPrevEffects:

    def test_returns_all_teams(self):
        m = Model2().fit_rows(ROWS, 2024)
        effects, variances = extract_prev_effects(m)
        assert set(effects.keys()) == {1, 2, 3}
        assert set(variances.keys()) == {1, 2, 3}

    def test_effects_match_theta(self):
        m  = Model2().fit_rows(ROWS, 2024)
        T  = len(m.teams_)
        effects, _ = extract_prev_effects(m)
        for idx, tid in enumerate(m.teams_):
            o_expected = float(m.theta_hat_[1 + idx])
            d_expected = float(m.theta_hat_[1 + T + idx])
            assert effects[int(tid)][0] == pytest.approx(o_expected)
            assert effects[int(tid)][1] == pytest.approx(d_expected)

    def test_variances_positive(self):
        m = Model2().fit_rows(ROWS, 2024)
        _, variances = extract_prev_effects(m)
        for tid, (vo, vd) in variances.items():
            assert vo > 0.0
            assert vd > 0.0


# ── Model2ContinuityPrior ──────────────────────────────────────────────────────

class TestModel2ContinuityPrior:

    # ── Invariant 1: r=0 → same structure as zero-centered ridge ────────────

    def test_r_zero_preserves_team_ordering(self):
        """r=0, prev_effects=0 → team ordering matches Model2."""
        m2 = Model2().fit_rows(ROWS, 2024)
        mc = Model2ContinuityPrior(
            prev_effects={1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)},
            r_minutes={},   # all r=0
            sigma2_prev=float(m2._sigma2_eff),
            tau_o_lo=1.0, tau_o_hi=5.0,
            tau_d_lo=1.0, tau_d_hi=5.0,
        ).fit_rows(ROWS, 2024)
        s2 = m2.point_summary()
        sc = mc.point_summary()
        assert (s2[1].adj_o > s2[2].adj_o) == (sc[1].adj_o > sc[2].adj_o)
        assert (s2[2].adj_o > s2[3].adj_o) == (sc[2].adj_o > sc[3].adj_o)

    # ── Invariant 2: r=1 with tiny tau → anchored to prev season ────────────

    def test_full_return_tight_tau_anchors_to_prior(self):
        """Very small tau_lo with r=1 → estimates close to prior-season values."""
        prev = {1: (+12.0, 0.0), 2: (0.0, 0.0), 3: (-12.0, 0.0)}
        r    = {1: 1.0, 2: 1.0, 3: 1.0}
        mc   = Model2ContinuityPrior(
            prev_effects=prev,
            r_minutes=r,
            tau_o_lo=0.01,   # near-zero: prior dominates
            tau_o_hi=0.01,
            tau_d_lo=0.01,
            tau_d_hi=0.01,
            sigma2_prev=160.0,
        ).fit_rows(ROWS, 2024)
        T  = len(mc.teams_)
        mu = mc.theta_hat_[0]
        # AdjO_i = mu + o_i; with strong prior, o_i ≈ prev o_i
        assert abs(mc.point_summary()[1].adj_o - (mu + 12.0)) < 2.0
        assert abs(mc.point_summary()[3].adj_o - (mu - 12.0)) < 2.0

    # ── Invariant 3: prev_effects=0 → no shift, result is team-specific ridge

    def test_zero_prev_effects_no_shift(self):
        """All prev_effects=0 → Pm=0; result is just team-specific ridge."""
        prev = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}
        mc   = Model2ContinuityPrior(
            prev_effects=prev,
            r_minutes={1: 0.7, 2: 0.5, 3: 0.3},
            tau_o_lo=1.5, tau_o_hi=5.0,
            tau_d_lo=1.5, tau_d_hi=5.0,
        ).fit_rows(ROWS, 2024)
        # Ordering still preserved (same as ridge with any positive penalty)
        s = mc.point_summary()
        assert s[1].adj_o > s[2].adj_o > s[3].adj_o
        assert s[2].adj_d < s[1].adj_d < s[3].adj_d

    # ── Interface contracts ──────────────────────────────────────────────────

    def test_fit_returns_self(self):
        m = Model2ContinuityPrior(prev_effects={}, r_minutes={})
        assert m.fit_rows(ROWS, 2024) is m

    def test_season_stored(self):
        m = Model2ContinuityPrior(prev_effects={}, r_minutes={})
        m.fit_rows(ROWS, 2024)
        assert m.season_ == 2024

    def test_all_teams_in_summary(self):
        m = Model2ContinuityPrior(prev_effects={}, r_minutes={})
        m.fit_rows(ROWS, 2024)
        assert set(m.point_summary().keys()) == {1, 2, 3}

    def test_theta_length(self):
        m = Model2ContinuityPrior(prev_effects={}, r_minutes={})
        m.fit_rows(ROWS, 2024)
        assert len(m.theta_hat_) == 3 * len(m.teams_) + 4

    def test_posterior_sampling_finite(self):
        prev = {1: (5.0, -2.0), 2: (0.0, 0.0), 3: (-5.0, 2.0)}
        r    = {1: 0.7, 2: 0.5, 3: 0.3}
        m    = Model2ContinuityPrior(prev_effects=prev, r_minutes=r)
        m.fit_rows(ROWS, 2024)
        draws = m.sample_posterior(20, np.random.default_rng(0))
        assert len(draws) == 20
        for d in draws:
            assert np.all(np.isfinite(d))

    def test_missing_team_in_prev_defaults_to_zero_mean(self):
        """Teams not in prev_effects get prior mean=0 (behaves like Model2)."""
        mc = Model2ContinuityPrior(
            prev_effects={},   # no prior for any team
            r_minutes={1: 0.8, 2: 0.5, 3: 0.3},
        ).fit_rows(ROWS, 2024)
        s = mc.point_summary()
        # Still produces valid, ordered ratings
        assert s[1].adj_o > s[3].adj_o

    def test_predict_efficiency_finite(self):
        prev = {1: (5.0, 0.0), 2: (0.0, 0.0), 3: (-5.0, 0.0)}
        m    = Model2ContinuityPrior(prev_effects=prev,
                                     r_minutes={1: 0.8, 2: 0.6, 3: 0.4})
        m.fit_rows(ROWS, 2024)
        pred = m.predict_efficiency(ROWS[:10])
        assert np.all(np.isfinite(pred))
        assert len(pred) == 10

    # ── Round-trip: extract_prev_effects → continuity prior ─────────────────

    def test_roundtrip_from_model2(self):
        """Full round-trip: fit Model2, extract effects, build continuity prior."""
        m2 = Model2().fit_rows(ROWS, 2024)
        effects, variances = extract_prev_effects(m2)
        r = {1: 0.8, 2: 0.5, 3: 0.3}

        mc = Model2ContinuityPrior(
            prev_effects=effects,
            r_minutes=r,
            sigma2_prev=float(m2._sigma2_eff),
            prev_var=variances,
        ).fit_rows(ROWS, 2024)

        s = mc.point_summary()
        # Basic sanity: all teams have plausible ratings
        for ks in s.values():
            assert 60.0 < ks.adj_o < 160.0
            assert 60.0 < ks.adj_d < 160.0
            assert ks.adj_pace > 0
