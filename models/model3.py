"""Model 3: Additive + bilinear (Hoff-style) interaction model.

Efficiency model:
    e_off_{ig} = μ + o_i - d_j + a_i·b_j + η·h_g + ε,  ε ~ N(0, σ²/p_g)

Pace model: identical to Model 2 (bilinear only applied to offense/defense).

The bilinear term a_i·b_j (rank-k inner product) captures style matchup effects
beyond what scalar main effects explain.

Estimation: alternating ridge regression (ALS):
  - Fix a, b → solve for (μ, o, d, η) via ridge
  - Fix (μ, o, d, η) → solve for a, b via separate ridge problems per team
  - Repeat until convergence

Posterior: hybrid Laplace/bootstrap approach
  - Main effects (μ, o, d, η): exact Gaussian from Laplace approx at MAP
  - Bilinear factors (a, b): parametric bootstrap — resimulate y from MAP,
    refit ALS, collect draws

KenPom summary: same mapping as Model 2 (bilinear excluded by centering):
    AdjO_i    = μ + o_i
    AdjD_i    = μ - d_i
    AdjPace_i = exp(γ₀ + r_i)

Centering constraint: sum(a_i) = 0 and sum(b_j) = 0, enforced at each ALS
step so that the bilinear term has zero mean and does not distort main effects.

Theta layout (length 3T+4 + 2kT):
    [0]                  = μ
    [1..T]               = o
    [T+1..2T]            = d
    [2T+1]               = η
    [2T+2]               = γ₀
    [2T+3..3T+2]         = r
    [3T+3]               = γ_h
    [3T+4..3T+4+kT]      = a flattened (T×k, row-major)
    [3T+4+kT..3T+4+2kT]  = b flattened (T×k, row-major)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy import linalg

from models.base import BaseModel, KenPomSummary
from models.data import GameRow
from models.model2 import Model2, _safe_cholesky

logger = logging.getLogger(__name__)


class Model3(BaseModel):
    """
    Additive + bilinear latent-effects model.

    Parameters
    ----------
    rank : int
        Dimensionality k of the interaction vectors a_i, b_j.
    lambda_team : float
        Ridge penalty on main effects (o_i, d_i).
    lambda_pace : float
        Ridge penalty on pace effects (r_i).
    lambda_ab : float
        Ridge penalty on bilinear factors (a_i, b_j).
    max_iter : int
        Max ALS iterations.
    tol : float
        Convergence tolerance (max |Δθ|).
    n_boot : int
        Parametric bootstrap draws for bilinear uncertainty.
    """

    def __init__(
        self,
        rank: int = 3,
        lambda_team: float = 100.0,
        lambda_pace: float = 50.0,
        lambda_ab: float = 5000.0,
        max_iter: int = 100,
        tol: float = 1e-5,
        n_boot: int = 200,
    ) -> None:
        self.rank = rank
        self.lambda_team = lambda_team
        self.lambda_pace = lambda_pace
        self.lambda_ab = lambda_ab
        self.max_iter = max_iter
        self.tol = tol
        self.n_boot = n_boot

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit_rows(self, rows, season: int) -> "Model3":
        self.season_ = season
        self._rows = list(rows)

        teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
        self.teams_ = np.array(teams, dtype=np.int64)
        self._tidx = {tid: i for i, tid in enumerate(teams)}

        # Stage 1: fit Model 2 main effects and pace (shared with Model 3).
        # Main effects are fixed from this stage — not re-estimated in ALS.
        # This two-stage approach prevents the bilinear term from confounding
        # with the main effects through schedule structure imbalance.
        m2 = Model2(lambda_team=self.lambda_team, lambda_pace=self.lambda_pace)
        m2.teams_ = self.teams_
        m2._tidx  = self._tidx
        theta_main, Sigma_main = m2._fit_efficiency(rows)
        theta_pace, Sigma_pace = m2._fit_pace(rows)
        self._theta_main  = theta_main
        self._Sigma_main  = Sigma_main
        self._sigma2_eff  = m2._sigma2_eff   # residual from main-effects-only fit
        self._theta_pace  = theta_pace
        self._Sigma_pace  = Sigma_pace
        self._sigma2_pace = m2._sigma2_pace

        # Stage 2: fit bilinear factors on Model 2 residuals.
        # Main effects are held fixed; only a, b are updated in this ALS.
        a, b, sigma2_full = self._als_bilinear(rows, theta_main)
        self._a           = a         # (T, k)
        self._b           = b         # (T, k)
        self._sigma2_full = sigma2_full   # residual after main + bilinear

        self.theta_hat_ = np.concatenate([
            theta_main, theta_pace,
            a.ravel(), b.ravel(),
        ])

        logger.info(
            "Model3 season=%d  teams=%d  rank=%d  σ_main=%.3f  σ_full=%.3f",
            season, len(teams), self.rank,
            float(np.sqrt(self._sigma2_eff)),
            float(np.sqrt(sigma2_full)),
        )
        return self

    # ------------------------------------------------------------------ #
    # Posterior (hybrid Laplace + bootstrap)                               #
    # ------------------------------------------------------------------ #

    def sample_posterior(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """
        Hybrid posterior draws.

        Main effects + pace: exact Gaussian from Laplace approx at MAP.
        Bilinear factors: parametric bootstrap — generate synthetic efficiency
            observations from the MAP fit, refit ALS, collect (a, b) draws.

        Both are combined into the full theta vector.
        """
        T = len(self.teams_)
        k = self.rank
        n_main  = 2 * T + 2
        n_pace  = T + 2

        # ---- Gaussian draws for main effects ----
        L_main  = _safe_cholesky(self._Sigma_main)
        L_pace  = _safe_cholesky(self._Sigma_pace)
        z_main  = rng.standard_normal((n, n_main))
        z_pace  = rng.standard_normal((n, n_pace))
        draws_main = self._theta_main + z_main @ L_main.T
        draws_pace = self._theta_pace + z_pace @ L_pace.T

        # ---- Parametric bootstrap for bilinear factors ----
        draws_ab = self._bootstrap_ab(n, rng)

        return [
            np.concatenate([draws_main[i], draws_pace[i], draws_ab[i]])
            for i in range(n)
        ]

    def _predict_from_theta(self, theta: np.ndarray, rows) -> np.ndarray:
        """Additive + bilinear prediction: μ + o_i − d_j + aᵢ·bⱼ + η·h."""
        T    = len(self.teams_)
        k    = self.rank
        mu   = theta[0]
        o    = theta[1:1 + T]
        d    = theta[1 + T:1 + 2 * T]
        eta  = theta[2 * T + 1]
        # bilinear factors from the theta vector (may differ from self._a/_b in draws)
        a_flat = theta[3 * T + 4:3 * T + 4 + k * T]
        b_flat = theta[3 * T + 4 + k * T:3 * T + 4 + 2 * k * T]
        a    = a_flat.reshape(T, k)
        b    = b_flat.reshape(T, k)
        tidx = self._tidx
        out  = np.empty(len(rows))
        for idx, r in enumerate(rows):
            i = tidx.get(r.team_id)
            j = tidx.get(r.opp_id)
            o_i = float(o[i])   if i is not None else 0.0
            d_j = float(d[j])   if j is not None else 0.0
            ab  = float(a[i] @ b[j]) if (i is not None and j is not None) else 0.0
            out[idx] = mu + o_i - d_j + ab + eta * r.h
        return out

    def _summary_from_theta(self, theta: np.ndarray) -> dict[int, KenPomSummary]:
        T = len(self.teams_)
        # Same mapping as Model 2 — bilinear excluded (centering constraint)
        mu     = theta[0]
        o      = theta[1:1 + T]
        d      = theta[1 + T:1 + 2 * T]
        gamma0 = theta[2 + 2 * T]
        r      = theta[3 + 2 * T:3 + 3 * T]

        return {
            int(tid): KenPomSummary(
                adj_o    = float(mu + o[i]),
                adj_d    = float(mu - d[i]),
                adj_pace = float(np.exp(gamma0 + r[i])),
            )
            for i, tid in enumerate(self.teams_)
        }

    # ------------------------------------------------------------------ #
    # Bilinear ALS (Stage 2 — main effects fixed from Model 2)           #
    # ------------------------------------------------------------------ #

    def _als_bilinear(
        self,
        rows: Sequence[GameRow],
        theta_main: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Fit bilinear factors (a, b) on Model 2 residuals with main effects
        held fixed throughout.  This two-stage approach prevents the bilinear
        term from confounding with the main effects through schedule imbalance.

        Returns (a, b, sigma2_full).
        """
        T = len(self.teams_)
        k = self.rank
        tidx = self._tidx

        team_idx = np.array([tidx[r.team_id] for r in rows], dtype=np.int32)
        opp_idx  = np.array([tidx[r.opp_id]  for r in rows], dtype=np.int32)
        e_off    = np.array([r.pts / r.poss * 100.0 for r in rows])
        h_arr    = np.array([float(r.h) for r in rows])
        w        = np.array([r.poss for r in rows])

        # Fixed main-effect predictions
        mu  = theta_main[0]
        o   = theta_main[1:1 + T]
        d   = theta_main[1 + T:1 + 2 * T]
        eta = theta_main[2 * T + 1]
        main_pred = mu + o[team_idx] - d[opp_idx] + eta * h_arr
        resid_main = e_off - main_pred   # fixed throughout

        # Initialize with small random noise to avoid zero fixed point
        rng_init = np.random.default_rng(0)
        scale = 0.1
        a = rng_init.standard_normal((T, k)) * scale
        b = rng_init.standard_normal((T, k)) * scale
        a -= a.mean(axis=0, keepdims=True)
        b -= b.mean(axis=0, keepdims=True)

        # rank=0: no bilinear term — skip ALS entirely
        if k == 0:
            sigma2 = float(np.average(resid_main ** 2, weights=w))
            return a, b, sigma2

        lam_eye = self.lambda_ab * np.eye(k)

        for iteration in range(self.max_iter):
            # Vectorized update of a given b.
            # For each team i: a_new[i] = (B_i'W_iB_i + λI)^{-1} B_i'W_ir_i
            # where B_i = b[opp of i's games], W_i = diag(w), r_i = resid.
            b_opp_all = b[opp_idx]                        # (N, k)
            wb_opp    = w[:, None] * b_opp_all            # (N, k)
            BwB = np.einsum("ni,nj->nij", wb_opp, b_opp_all)  # (N, k, k)
            BwB_sum = np.zeros((T, k, k))
            np.add.at(BwB_sum, team_idx, BwB)
            BwB_sum += lam_eye[None, :, :]
            Bwr = np.zeros((T, k))
            np.add.at(Bwr, team_idx, wb_opp * resid_main[:, None])
            # Batched solve: (T, k, k) @ (T, k, 1) → (T, k)
            a_new = np.linalg.solve(BwB_sum, Bwr[..., None]).squeeze(-1)
            a_new -= a_new.mean(axis=0, keepdims=True)

            # Vectorized update of b given a_new.
            a_team_all = a_new[team_idx]                  # (N, k)
            wa_team    = w[:, None] * a_team_all          # (N, k)
            AwA = np.einsum("ni,nj->nij", wa_team, a_team_all)  # (N, k, k)
            AwA_sum = np.zeros((T, k, k))
            np.add.at(AwA_sum, opp_idx, AwA)
            AwA_sum += lam_eye[None, :, :]
            Awr = np.zeros((T, k))
            np.add.at(Awr, opp_idx, wa_team * resid_main[:, None])
            b_new = np.linalg.solve(AwA_sum, Awr[..., None]).squeeze(-1)
            b_new -= b_new.mean(axis=0, keepdims=True)

            delta = max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b)))
            a, b = a_new, b_new
            if delta < self.tol:
                logger.debug("ALS_bilinear converged at iteration %d (Δ=%.2e)", iteration + 1, delta)
                break

        interaction = (a[team_idx] * b[opp_idx]).sum(axis=1)
        full_resid = resid_main - interaction
        sigma2_full = float(np.average(full_resid ** 2, weights=w))
        return a, b, sigma2_full

    # ------------------------------------------------------------------ #
    # Parametric bootstrap for bilinear factors                           #
    # ------------------------------------------------------------------ #

    def _bootstrap_ab(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n bootstrap replicates of (a, b) by:
          1. Generate synthetic efficiency observations from the MAP fit
          2. Refit ALS on each synthetic dataset
          3. Return flattened [a.ravel(), b.ravel()] for each draw
        """
        T = len(self.teams_)
        k = self.rank
        tidx = self._tidx

        team_idx = np.array([tidx[r.team_id] for r in self._rows], dtype=np.int32)
        opp_idx  = np.array([tidx[r.opp_id]  for r in self._rows], dtype=np.int32)
        h_arr    = np.array([float(r.h)       for r in self._rows])
        poss_arr = np.array([r.poss           for r in self._rows])

        # MAP fitted values
        mu  = self._theta_main[0]
        o   = self._theta_main[1:1 + T]
        d   = self._theta_main[1 + T:1 + 2 * T]
        eta = self._theta_main[2 * T + 1]
        fitted = (
            mu
            + o[team_idx]
            - d[opp_idx]
            + np.array([float(self._a[team_idx[m]] @ self._b[opp_idx[m]]) for m in range(len(self._rows))])
            + eta * h_arr
        )

        # Use sigma2_full (residual after main + bilinear) for noise simulation
        sigma_eff = float(np.sqrt(self._sigma2_full))
        draws_ab = np.empty((n, 2 * T * k))

        for s in range(n):
            # Simulate new efficiency observations around the full MAP fit
            noise = rng.normal(0, sigma_eff / np.sqrt(poss_arr / poss_arr.mean()))
            y_sim = fitted + noise

            # Build synthetic GameRow list with simulated pts
            sim_rows = [
                GameRow(
                    game_id  = r.game_id,
                    season   = r.season,
                    team_id  = r.team_id,
                    opp_id   = r.opp_id,
                    pts      = max(0, int(round(y_sim[m] * r.poss / 100.0))),
                    poss     = r.poss,
                    h        = r.h,
                )
                for m, r in enumerate(self._rows)
            ]

            # Refit only bilinear on simulated residuals; main effects fixed
            a_s, b_s, _ = self._als_bilinear(sim_rows, self._theta_main)
            draws_ab[s] = np.concatenate([a_s.ravel(), b_s.ravel()])

        return draws_ab
