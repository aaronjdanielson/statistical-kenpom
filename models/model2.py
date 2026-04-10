"""Model 2: Ridge (RAPM-style) additive latent-effects model.

Efficiency model (one row per team-game):
    e_off_{ig} = μ + o_i - d_j + η·h_g + ε,    ε ~ N(0, σ²/p_g)

Pace model (one row per game):
    log(p_g) = γ₀ + r_i + r_j + γ_h·h_g + ξ,    ξ ~ N(0, σ_p²)

Both are ridge regression problems with Gaussian priors on the team effects.
The posterior is exact Gaussian, enabling analytic sampling.

KenPom summary mapping:
    AdjO_i    = μ + o_i          (score vs avg defense, d=0)
    AdjD_i    = μ - d_i          (pts allowed vs avg offense, o=0)
    AdjPace_i = exp(γ₀ + r_i)   (poss vs avg opponent, r_opp=0)

Theta layout (length 3T+4):
    [0]          = μ
    [1 .. T]     = o_0,...,o_{T-1}
    [T+1 .. 2T]  = d_0,...,d_{T-1}
    [2T+1]       = η
    [2T+2]       = γ₀
    [2T+3 .. 3T+2] = r_0,...,r_{T-1}
    [3T+3]       = γ_h
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy import linalg

from models.base import BaseModel, KenPomSummary
from models.data import GameRow

logger = logging.getLogger(__name__)


def _safe_cholesky(M: np.ndarray) -> np.ndarray:
    """Cholesky with progressive jitter fallback for near-singular matrices."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        pass
    for exp in range(-9, 0):
        jitter = 10.0 ** exp
        try:
            L = np.linalg.cholesky(M + jitter * np.eye(len(M)))
            logger.warning("Cholesky required jitter=%.0e", jitter)
            return L
        except np.linalg.LinAlgError:
            continue
    # Last resort: diagonal approximation
    logger.error("Cholesky failed even with jitter; falling back to diagonal")
    return np.diag(np.sqrt(np.maximum(np.diag(M), 1e-10)))


class Model2(BaseModel):
    """
    Ridge latent-effects model with exact Gaussian posterior sampling.

    Parameters
    ----------
    lambda_team : float
        Ridge penalty on all team offense/defense effects (o_i, d_i).
    lambda_pace : float
        Ridge penalty on team pace effects (r_i).

    Both intercepts (μ, γ₀) and home effects (η, γ_h) are left unpenalized.
    """

    def __init__(self, lambda_team: float = 100.0, lambda_pace: float = 50.0) -> None:
        self.lambda_team = lambda_team
        self.lambda_pace = lambda_pace

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit_rows(self, rows, season: int) -> "Model2":
        self.season_ = season
        self._rows = list(rows)

        teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
        self.teams_ = np.array(teams, dtype=np.int64)
        self._tidx = {tid: i for i, tid in enumerate(teams)}

        theta_eff,  self._Sigma_eff  = self._fit_efficiency(rows)
        theta_pace, self._Sigma_pace = self._fit_pace(rows)
        self.theta_hat_ = np.concatenate([theta_eff, theta_pace])

        logger.info(
            "Model2 season=%d  teams=%d  eff_obs=%d  pace_obs=%d  "
            "σ_eff=%.3f  σ_pace=%.3f",
            season, len(teams), len(rows), len(rows) // 2,
            float(np.sqrt(self._sigma2_eff)), float(np.sqrt(self._sigma2_pace)),
        )
        return self

    # ------------------------------------------------------------------ #
    # Posterior sampling                                                   #
    # ------------------------------------------------------------------ #

    def sample_posterior(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """
        Draw n samples from the joint Gaussian posterior.

        Efficiency and pace posteriors are independent (separate design matrices
        and noise terms), so samples are drawn independently and concatenated.
        """
        T = len(self.teams_)
        n_eff  = 2 * T + 2
        n_pace = T + 2
        theta_eff  = self.theta_hat_[:n_eff]
        theta_pace = self.theta_hat_[n_eff:]

        L_eff  = _safe_cholesky(self._Sigma_eff)
        L_pace = _safe_cholesky(self._Sigma_pace)

        z_eff  = rng.standard_normal((n, n_eff))
        z_pace = rng.standard_normal((n, n_pace))

        samples_eff  = theta_eff  + z_eff  @ L_eff.T
        samples_pace = theta_pace + z_pace @ L_pace.T

        return [
            np.concatenate([samples_eff[i], samples_pace[i]])
            for i in range(n)
        ]

    def _predict_from_theta(self, theta: np.ndarray, rows) -> np.ndarray:
        """Additive prediction: μ + o_i − d_j + η·h."""
        T    = len(self.teams_)
        mu   = theta[0]
        o    = theta[1:1 + T]
        d    = theta[1 + T:1 + 2 * T]
        eta  = theta[2 * T + 1]
        tidx = self._tidx
        out  = np.empty(len(rows))
        for k, r in enumerate(rows):
            i = tidx.get(r.team_id)
            j = tidx.get(r.opp_id)
            o_i = float(o[i]) if i is not None else 0.0
            d_j = float(d[j]) if j is not None else 0.0
            out[k] = mu + o_i - d_j + eta * r.h
        return out

    def _summary_from_theta(self, theta: np.ndarray) -> dict[int, KenPomSummary]:
        T  = len(self.teams_)
        mu = theta[0]
        o  = theta[1:1 + T]
        d  = theta[1 + T:1 + 2 * T]
        # eta = theta[1 + 2*T]
        gamma0 = theta[2 + 2 * T]
        r      = theta[3 + 2 * T:3 + 3 * T]
        # gamma_h = theta[3 + 3*T]

        return {
            int(tid): KenPomSummary(
                adj_o    = float(mu + o[i]),
                adj_d    = float(mu - d[i]),
                adj_pace = float(np.exp(gamma0 + r[i])),
            )
            for i, tid in enumerate(self.teams_)
        }

    # ------------------------------------------------------------------ #
    # Ridge solvers                                                        #
    # ------------------------------------------------------------------ #

    def _fit_efficiency(
        self, rows: Sequence[GameRow]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the efficiency ridge model.

        Design matrix columns: [μ, o_0,...,o_{T-1}, d_0,...,d_{T-1}, η]
        Penalty: λ_team on o and d; 0 on μ and η.
        Weight: poss (efficiency variance ∝ 1/poss).

        Returns (theta_hat, Sigma_post).
        """
        T = len(self.teams_)
        tidx = self._tidx
        N = len(rows)
        n_cols = 2 * T + 2  # [μ, o×T, d×T, η]

        X = np.zeros((N, n_cols))
        y = np.empty(N)
        w = np.empty(N)

        for k, r in enumerate(rows):
            i = tidx[r.team_id]
            j = tidx[r.opp_id]
            X[k, 0]         =  1.0          # μ
            X[k, 1 + i]     =  1.0          # +o_i
            X[k, 1 + T + j] = -1.0          # -d_j
            X[k, 2 * T + 1] =  float(r.h)  # η · h
            y[k] = r.pts / r.poss * 100.0
            w[k] = r.poss

        # Penalty matrix (diagonal): ridge on o_i and d_j.
        # μ and η nominally unpenalized, but a tiny jitter (1e-4) is added to
        # the full diagonal to break the structural linear dependence
        # (intercept col = sum of all o cols), ensuring a non-singular solve.
        lam = np.full(n_cols, self.lambda_team)
        lam[0]         = 1e-4   # μ: near-zero ridge for stability
        lam[2 * T + 1] = 1e-4   # η: near-zero ridge for stability

        Xw   = X * w[:, None]
        A    = Xw.T @ X + np.diag(lam)      # (X'WX + Λ)
        b    = Xw.T @ y                      # X'Wy
        theta_hat = linalg.solve(A, b, assume_a="sym")

        # Noise variance from weighted residuals
        resid = y - X @ theta_hat
        self._sigma2_eff = float(np.average(resid ** 2, weights=w))

        Sigma = self._sigma2_eff * linalg.inv(A)
        return theta_hat, Sigma

    def _fit_pace(
        self, rows: Sequence[GameRow]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the pace ridge model (one row per game).

        Design matrix columns: [γ₀, r_0,...,r_{T-1}, γ_h]
        Both teams contribute +1 in their own r column for the same game row.
        Uses games de-duplicated by (min(team_id), max(team_id)).

        Returns (theta_hat, Sigma_post).
        """
        T = len(self.teams_)
        tidx = self._tidx
        n_cols = T + 2  # [γ₀, r×T, γ_h]

        # De-duplicate: one row per game
        seen: set[int] = set()
        game_rows: list[GameRow] = []
        for r in rows:
            if r.game_id not in seen:
                seen.add(r.game_id)
                game_rows.append(r)

        # For each de-duplicated row we still need both team ids
        # Build a game_id → (team_i, team_j, poss, h) mapping
        game_data: dict[int, tuple[int, int, float, int]] = {}
        for r in rows:
            gid = r.game_id
            if gid not in game_data:
                game_data[gid] = (r.team_id, r.opp_id, r.poss, r.h)

        N = len(game_data)
        X = np.zeros((N, n_cols))
        y = np.empty(N)

        for k, (gid, (ti, tj, poss, h)) in enumerate(game_data.items()):
            i = tidx[ti]
            j = tidx[tj]
            X[k, 0]     =  1.0       # γ₀
            X[k, 1 + i] =  1.0       # +r_i
            X[k, 1 + j] =  1.0       # +r_j
            X[k, T + 1] =  float(h)  # γ_h · h
            y[k] = np.log(max(poss, 1.0))

        # Penalty: λ_pace on r; tiny ridge on γ₀ and γ_h for stability
        lam = np.full(n_cols, self.lambda_pace)
        lam[0]     = 1e-4
        lam[T + 1] = 1e-4

        A = X.T @ X + np.diag(lam)
        b = X.T @ y
        theta_hat = linalg.solve(A, b, assume_a="sym")

        resid = y - X @ theta_hat
        self._sigma2_pace = float(np.mean(resid ** 2))

        Sigma = self._sigma2_pace * linalg.inv(A)
        return theta_hat, Sigma
