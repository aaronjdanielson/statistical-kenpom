"""Model 4: Weekly linear-Gaussian state-space model (Kalman filter).

Extends Model 2 by promoting team offensive and defensive effects to
time-varying latent states that evolve as independent random walks:

    o_i(t) = o_i(t-1) + ξ_i^o,   ξ_i^o ~ N(0, τ_o²)
    d_i(t) = d_i(t-1) + ξ_i^d,   ξ_i^d ~ N(0, τ_d²)

The observation equation for game g in week t is identical to Model 2:

    y_g = μ + o_{team}(t) - d_{opp}(t) + η·h_g + ε_g,
    ε_g ~ N(0, σ²/p_g)

Because the model is linear-Gaussian, the filtering posterior at each week
is exactly Gaussian — no MCMC, no approximation.

For one-step-ahead prediction the forward filter is used (causal).
For retrospective rating trajectories the Rauch-Tung-Striebel smoother
is available via ``rts_smoother()``.

KenPom summaries follow the same mapping as Model 2:
    AdjO_i(t) = μ + o_i(t)
    AdjD_i(t) = μ - d_i(t)
    AdjPace_i  = exp(γ₀ + r_i)   (static, identical to Model 2)

The external ``theta_hat_`` is in Model 2's layout using the *final* filtered
state, so all BaseModel prediction and evaluation methods work unchanged.

Hyperparameter estimation
--------------------------
τ_o, τ_d, σ, μ, η are estimated by maximising the marginal log-likelihood
computed via the prediction-error decomposition of the Kalman filter.
``optimize_hyperparams=False`` uses the constructor values directly (useful
for testing or when initialising from a prior fit).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
from scipy import linalg, optimize

from models.base import BaseModel, KenPomSummary
from models.data import GameRow

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level helpers                                                          #
# --------------------------------------------------------------------------- #

def _preprocess_weeks(
    weeks_rows: list[list[GameRow]],
    tidx: dict[int, int],
) -> list[tuple | None]:
    """
    Convert each week's GameRow list into aligned numpy arrays.

    Returns a list of length W.  Entry w is None if the week has no games,
    otherwise (team_idx, opp_idx, y, w_poss, h) with dtype-correct arrays.
    """
    result: list[tuple | None] = []
    for rows_t in weeks_rows:
        if not rows_t:
            result.append(None)
        else:
            result.append((
                np.array([tidx[r.team_id] for r in rows_t], dtype=np.int32),
                np.array([tidx[r.opp_id]  for r in rows_t], dtype=np.int32),
                np.array([r.pts / r.poss * 100.0 for r in rows_t]),
                np.array([r.poss for r in rows_t]),
                np.array([float(r.h) for r in rows_t]),
            ))
    return result


def _forward_filter(
    week_arrays: list[tuple | None],
    N: int,
    mu: float,
    eta: float,
    tau_o2: float,
    tau_d2: float,
    sigma2: float,
    lambda_team: float,
) -> tuple[list, list, list, float]:
    """
    Kalman forward filter for the 2N-dimensional team efficiency state.

    State layout: α = [o_0,...,o_{N-1},  d_0,...,d_{N-1}]
    Observation:  y_g = μ + α[team_idx] - α[N + opp_idx] + η·h + ε
                  ε ~ N(0, sigma2)   [homoscedastic; sigma2 ≈ RMSE² ≈ 169]

    Noise model note
    ----------------
    sigma2 here is Var(y_k) in (pts/100)² — the same quantity as Model 2's
    sigma2_eff (the possession-weighted mean squared residual).  NCAA
    possessions are approximately constant (≈70 ± 5), so the heteroscedastic
    weighting by 1/poss_k is negligible and we use a homoscedastic model for
    numerical stability.

    Exploits the two-nonzero-per-row sparsity of the observation matrix H:
      • R H'  computed by column-slicing R
      • H R H' computed by row-slicing (R H')
    This avoids materialising H explicitly and reduces cost to O(K_t · N²)
    per week rather than O(K_t² · N) from dense matrix products.

    Returns
    -------
    m_filtered : list of (2N,) filtered means, one per week
    C_filtered : list of (2N, 2N) filtered covariances, one per week
    R_predicted: list of (2N, 2N) predictive covariances (= C_{t-1} + Q),
                 needed by the RTS smoother
    log_lik    : scalar total marginal log-likelihood
    """
    two_N = 2 * N
    q_diag = np.concatenate([np.full(N, tau_o2), np.full(N, tau_d2)])

    # Initial state: mean = 0, covariance = (σ²/λ) I
    #
    # Model 2's conjugate prior is θ ~ N(0, σ²_eff/λ · I), because its
    # posterior covariance is σ²_eff · (X'WX + λI)^{-1}.  Using (1/λ) I
    # here would be off by a factor of σ²_eff ≈ 160, making the prior
    # ~160× too tight and causing the filter to ignore observations.
    # Setting C₀ = (σ²/λ) I ensures that the Kalman MAP posterior equals
    # Model 2's WLS posterior exactly when τ→0 (verified by the RTS test).
    m = np.zeros(two_N)
    C = np.eye(two_N) * (sigma2 / lambda_team)

    m_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    ll = 0.0

    for wa in week_arrays:
        # ── Prediction step: R = C + Q ────────────────────────────────────
        R = C + np.diag(q_diag)
        R_list.append(R)

        if wa is None:
            # No games this week — state propagates without an update
            m_list.append(m.copy())
            C_list.append(R.copy())
            C = R
            continue

        team_idx, opp_idx, y_t, w_t, h_t = wa
        K_t = len(y_t)

        # ── R H' ─ column-index exploit (sparse H) ────────────────────────
        # H[k, team_idx[k]] = +1,  H[k, N + opp_idx[k]] = -1
        # ⟹ (R H')[:, k] = R[:, team_idx[k]] - R[:, N + opp_idx[k]]
        RHt = R[:, team_idx] - R[:, N + opp_idx]   # (2N, K_t)

        # ── H R H' ─ row-index exploit ────────────────────────────────────
        # (H R H')[k, l] = RHt[team_idx[k], l] - RHt[N+opp_idx[k], l]
        HRHT = RHt[team_idx, :] - RHt[N + opp_idx, :]   # (K_t, K_t)

        # ── Innovation variance S = H R H' + diag(σ²/poss) ─────────────────
        # Matches Model 2's WLS noise model: Var(y_k) = σ²_eff/poss_k.
        # With sigma2 = σ²_eff ≈ 160 (pts/100)² and poss ≈ 70, this gives
        # per-game noise ≈ 2.3 (pts/100)², equivalent to WLS weight = poss_k.
        # Combined with C₀ = σ²/λ · I (above), the Kalman MAP posterior
        # equals Model 2's batch ridge posterior when τ=0.
        S = HRHT + np.diag(sigma2 / w_t)

        # ── Innovation v = y - (H m + μ + η h) ───────────────────────────
        v = y_t - (m[team_idx] - m[N + opp_idx] + mu + eta * h_t)

        # ── Cholesky of S for stable log-det and solve ────────────────────
        try:
            L_S, _ = linalg.cho_factor(S, lower=True)
        except linalg.LinAlgError:
            S += 1e-6 * np.eye(K_t)
            L_S, _ = linalg.cho_factor(S, lower=True)
        cf = (L_S, True)

        log_det_S = 2.0 * np.sum(np.log(np.diag(L_S)))
        Sinv_v    = linalg.cho_solve(cf, v)
        ll += -0.5 * (log_det_S + float(v @ Sinv_v) + K_t * np.log(2.0 * np.pi))

        # ── Kalman gain: K = R H' S^{-1}  →  (2N, K_t) ──────────────────
        # solve(S, RHt.T) = S^{-1} RHt.T;  transpose → RHt S^{-1} = K
        Kgain = linalg.cho_solve(cf, RHt.T).T

        # ── State update ──────────────────────────────────────────────────
        m = m + Kgain @ v
        C = R - Kgain @ RHt.T
        C = 0.5 * (C + C.T)          # symmetrize against floating-point drift

        m_list.append(m.copy())
        C_list.append(C.copy())

    return m_list, C_list, R_list, ll


def _rts_smooth(
    m_filtered: list[np.ndarray],
    C_filtered: list[np.ndarray],
    R_predicted: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Rauch-Tung-Striebel backward smoother.

    For a random walk transition α_t = α_{t-1} + ω, the smoother gain is:
        G_t = C_t  R_{t+1}^{-1}

    because a_{t+1} = m_t (predictive mean equals previous filter mean).

    Returns list of smoothed state means (same length as m_filtered).
    Use for retrospective analysis only — not for one-step-ahead prediction.
    """
    W = len(m_filtered)
    ms: list[np.ndarray] = [None] * W   # type: ignore[list-item]
    ms[W - 1] = m_filtered[W - 1].copy()

    for t in range(W - 2, -1, -1):
        R_tp1 = R_predicted[t + 1]
        # Smoother gain G_t = C_t R_{t+1}^{-1}
        # Equivalently: G_t^T = R_{t+1}^{-T} C_t^T = R_{t+1}^{-1} C_t (symmetric)
        G_t = linalg.solve(R_tp1, C_filtered[t], assume_a="pos")
        # ms[t] = m[t] + G_t (ms[t+1] - m[t])    [a_{t+1} = m_t for rand walk]
        ms[t] = m_filtered[t] + G_t @ (ms[t + 1] - m_filtered[t])

    return ms


# --------------------------------------------------------------------------- #
# Model 4                                                                       #
# --------------------------------------------------------------------------- #

class Model4(BaseModel):
    """
    Weekly linear-Gaussian state-space model for dynamic team efficiency.

    Parameters
    ----------
    lambda_team          : Ridge / initial-state precision on team effects
                           (same interpretation as Model 2).
    lambda_pace          : Ridge precision on team pace effects.
    tau_o, tau_d         : Initial / fixed offensive and defensive drift std
                           (pts/100 per week).  Overridden by optimisation
                           when optimize_hyperparams=True.
    optimize_hyperparams : Estimate τ_o, τ_d, σ, μ, η by maximising the
                           marginal log-likelihood.
    week_step            : Width of each time bin in days.
    max_opt_iter         : L-BFGS-B iteration budget for hyperparameter
                           optimisation.
    """

    def __init__(
        self,
        lambda_team: float = 100.0,
        lambda_pace: float = 50.0,
        tau_o: float = 1.0,
        tau_d: float = 1.0,
        optimize_hyperparams: bool = True,
        week_step: int = 7,
        max_opt_iter: int = 100,
    ) -> None:
        self.lambda_team = lambda_team
        self.lambda_pace = lambda_pace
        self.tau_o = tau_o
        self.tau_d = tau_d
        self.optimize_hyperparams = optimize_hyperparams
        self.week_step = week_step
        self.max_opt_iter = max_opt_iter

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, season: int, conn) -> "Model4":
        """Load rows and game dates from DB, then call fit_rows."""
        from models.data import load_season_games, parse_date
        rows = load_season_games(conn, season)
        if not rows:
            raise ValueError(f"No games found for season {season}")
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT s.GameID, s.Date
            FROM schedules s JOIN boxscores b ON s.GameID = b.GameID
            WHERE s.Year = ? ORDER BY s.GameID
            """,
            (season,),
        )
        game_dates = {gid: parse_date(ds) for gid, ds in cur.fetchall()}
        return self.fit_rows(rows, season, game_dates=game_dates)

    def fit_rows(
        self,
        rows,
        season: int,
        *,
        game_dates: dict[int, datetime] | None = None,
    ) -> "Model4":
        """
        Fit the Kalman state-space model on pre-loaded GameRows.

        Parameters
        ----------
        rows       : sequence of GameRow
        season     : season end-year
        game_dates : dict mapping game_id → datetime used for weekly binning.
                     If None, games are binned by game_id rank order (a
                     reasonable proxy since IDs are approximately date-ordered).
        """
        from models.model2 import Model2, _safe_cholesky

        self._safe_cholesky = _safe_cholesky
        self.season_ = season
        rows = list(rows)

        teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
        self.teams_ = np.array(teams, dtype=np.int64)
        self._tidx   = {tid: i for i, tid in enumerate(teams)}
        N = len(teams)

        # ── Pace: static solve identical to Model 2 ───────────────────────
        m2_tmp = Model2(lambda_team=self.lambda_team, lambda_pace=self.lambda_pace)
        m2_tmp.fit_rows(rows, season)
        n_eff_m2 = 2 * N + 2
        self._theta_pace  = m2_tmp.theta_hat_[n_eff_m2:].copy()
        self._Sigma_pace  = m2_tmp._Sigma_pace
        self._sigma2_pace = m2_tmp._sigma2_pace

        # Bootstrap hyperparameter starting values from Model 2
        sigma0 = float(np.sqrt(m2_tmp._sigma2_eff))
        mu0    = float(m2_tmp.theta_hat_[0])
        eta0   = float(m2_tmp.theta_hat_[2 * N + 1])

        # ── Build weekly bins ─────────────────────────────────────────────
        if game_dates is None:
            # Fallback: rank game_ids as a proxy for date ordering
            sorted_gids = sorted({r.game_id for r in rows})
            game_dates = {
                gid: datetime(2000, 1, 1) + timedelta(days=rank)
                for rank, gid in enumerate(sorted_gids)
            }

        self._game_dates = game_dates
        season_start = min(game_dates[r.game_id] for r in rows)
        season_end   = max(game_dates[r.game_id] for r in rows)
        self._season_start = season_start
        self._season_end   = season_end

        # Week cutoff boundaries: each bin covers [cutoff_{t-1}, cutoff_t)
        cutoffs: list[datetime] = []
        d = season_start + timedelta(days=self.week_step)
        while d <= season_end + timedelta(days=1):
            cutoffs.append(d)
            d += timedelta(days=self.week_step)
        if not cutoffs or cutoffs[-1] <= season_end:
            cutoffs.append(season_end + timedelta(days=1))
        self._week_cutoffs = cutoffs
        W = len(cutoffs)

        # Assign each row to a week (binary search would be faster but
        # Python list is fine for W ~ 25)
        row_week: list[int] = []
        for r in rows:
            dt = game_dates[r.game_id]
            assigned = W - 1
            for w, cut in enumerate(cutoffs):
                if dt < cut:
                    assigned = w
                    break
            row_week.append(assigned)

        weeks_rows: list[list[GameRow]] = [[] for _ in range(W)]
        for r, w in zip(rows, row_week):
            weeks_rows[w].append(r)
        self._weeks_rows = weeks_rows

        # Pre-build aligned numpy arrays per week (avoids rebuilding in optim)
        self._week_arrays = _preprocess_weeks(weeks_rows, self._tidx)

        # ── Hyperparameter optimisation ───────────────────────────────────
        if self.optimize_hyperparams:
            x0 = np.array([
                np.log(max(self.tau_o, 1e-3)),
                np.log(max(self.tau_d, 1e-3)),
                np.log(max(sigma0, 1.0)),
                mu0,
                eta0,
            ])
            # σ = sqrt(sigma2_eff) ≈ RMSE ≈ 13-14 pts/100 for NCAA.
            # Under the heteroscedastic noise model Var(y_k) = σ²/poss_k:
            #   σ² = sigma2_eff  (the possession-weighted mean sq. residual)
            #   Per-game noise std ≈ √(σ²/70) ≈ √(160/70) ≈ 1.5 pts/100
            # The optimisation is over log(σ), so σ ∈ [2.7, 55] covers NCAA.
            bounds = [
                (-5.0, 3.0),    # log τ_o  → τ_o ∈ [0.007, 20] pts/100/wk
                (-5.0, 3.0),    # log τ_d
                (1.0,  4.0),    # log σ    → σ ∈ [2.7, 55] pts/100 (NCAA ≈ 13)
                (80.0, 130.0),  # μ        (league mean ≈ 105)
                (-3.0, 10.0),   # η        (home advantage ≈ 3)
            ]

            def _nll(x: np.ndarray) -> float:
                tao = float(np.exp(x[0]))
                tad = float(np.exp(x[1]))
                sig = float(np.exp(x[2]))
                _, _, _, ll = _forward_filter(
                    self._week_arrays, N,
                    mu=float(x[3]), eta=float(x[4]),
                    tau_o2=tao ** 2, tau_d2=tad ** 2,
                    sigma2=sig ** 2,
                    lambda_team=self.lambda_team,
                )
                return -ll

            res = optimize.minimize(
                _nll, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_opt_iter, "ftol": 1e-8},
            )
            x_opt = res.x
            tau_o_fit = float(np.exp(x_opt[0]))
            tau_d_fit = float(np.exp(x_opt[1]))
            sigma_fit  = float(np.exp(x_opt[2]))
            mu_fit     = float(x_opt[3])
            eta_fit    = float(x_opt[4])
            logger.info(
                "Model4 hyperopt  τ_o=%.3f  τ_d=%.3f  σ=%.3f  "
                "μ=%.2f  η=%.3f  nll=%.1f  (%d iters)",
                tau_o_fit, tau_d_fit, sigma_fit,
                mu_fit, eta_fit, res.fun, res.nit,
            )
        else:
            tau_o_fit = self.tau_o
            tau_d_fit = self.tau_d
            sigma_fit  = sigma0
            mu_fit     = mu0
            eta_fit    = eta0

        self.tau_o_      = tau_o_fit
        self.tau_d_      = tau_d_fit
        self._sigma2_eff = sigma_fit ** 2
        self._mu         = mu_fit
        self._eta        = eta_fit

        # ── Final forward filter ──────────────────────────────────────────
        m_list, C_list, R_list, ll = _forward_filter(
            self._week_arrays, N,
            mu=mu_fit, eta=eta_fit,
            tau_o2=tau_o_fit ** 2, tau_d2=tau_d_fit ** 2,
            sigma2=sigma_fit ** 2,
            lambda_team=self.lambda_team,
        )
        self._m_filtered  = m_list
        self._C_filtered  = C_list
        self._R_predicted = R_list
        self._log_lik     = ll

        # ── Build theta_hat_ in Model 2 layout ───────────────────────────
        m_T = m_list[-1]
        C_T = C_list[-1]

        theta_eff = np.concatenate([[mu_fit], m_T[:N], m_T[N:], [eta_fit]])
        self.theta_hat_ = np.concatenate([theta_eff, self._theta_pace])

        # Efficiency posterior covariance — state blocks only (μ, η fixed)
        n_eff = 2 * N + 2
        Sigma_eff = np.zeros((n_eff, n_eff))
        Sigma_eff[1:N + 1,     1:N + 1]     = C_T[:N, :N]    # o-o
        Sigma_eff[N + 1:2*N+1, N + 1:2*N+1] = C_T[N:, N:]   # d-d
        Sigma_eff[1:N + 1,     N + 1:2*N+1] = C_T[:N, N:]   # o-d cross
        Sigma_eff[N + 1:2*N+1, 1:N + 1]     = C_T[N:, :N]   # d-o cross
        self._Sigma_eff = Sigma_eff

        logger.info(
            "Model4  season=%d  teams=%d  weeks=%d  "
            "τ_o=%.3f  τ_d=%.3f  σ_eff=%.3f  ll=%.1f",
            season, N, W, tau_o_fit, tau_d_fit, sigma_fit, ll,
        )
        return self

    # ------------------------------------------------------------------ #
    # Posterior sampling                                                   #
    # ------------------------------------------------------------------ #

    def sample_posterior(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """
        Draw n samples from the final filter posterior.

        Samples (o, d) jointly from N(m_T, C_T).  Static parameters μ and η
        are treated as known (zero posterior variance here).  Pace is sampled
        from its own posterior, identical to Model 2.
        """
        from models.model2 import _safe_cholesky
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

    # ------------------------------------------------------------------ #
    # Prediction and summaries (Model 2 layout — inherited interface)     #
    # ------------------------------------------------------------------ #

    def _predict_from_theta(self, theta: np.ndarray, rows) -> np.ndarray:
        """Additive prediction: μ + o_i − d_j + η·h  (identical to Model 2)."""
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
        T      = len(self.teams_)
        mu     = theta[0]
        o      = theta[1:1 + T]
        d      = theta[1 + T:1 + 2 * T]
        gamma0 = theta[2 + 2 * T]
        r_pace = theta[3 + 2 * T:3 + 3 * T]
        return {
            int(tid): KenPomSummary(
                adj_o    = float(mu + o[i]),
                adj_d    = float(mu - d[i]),
                adj_pace = float(np.exp(gamma0 + r_pace[i])),
            )
            for i, tid in enumerate(self.teams_)
        }

    # ------------------------------------------------------------------ #
    # Time-indexed outputs (Model 4 only)                                  #
    # ------------------------------------------------------------------ #

    def point_summary_trajectory(
        self,
    ) -> list[tuple[datetime, dict[int, KenPomSummary]]]:
        """
        Filtered (causal) KenPom summary for each team at each weekly bin.

        Returns a list of (cutoff_date, {team_id: KenPomSummary}) tuples,
        one per week, using the forward filter state only.  Suitable for
        one-step-ahead-style retrospective inspection.
        """
        T = len(self.teams_)
        result = []
        for cutoff, m_w in zip(self._week_cutoffs, self._m_filtered):
            theta = np.concatenate([
                [self._mu], m_w[:T], m_w[T:], [self._eta],
                self._theta_pace,
            ])
            result.append((cutoff, self._summary_from_theta(theta)))
        return result

    def rts_smoother(self) -> list[tuple[datetime, dict[int, KenPomSummary]]]:
        """
        Retrospective KenPom summaries using the Rauch-Tung-Striebel smoother.

        Uses data from the entire season in both directions — **not suitable
        for one-step-ahead evaluation**.  Use for end-of-season retrospective
        analysis: "how strong was this team in week t, given everything we know?"
        """
        T = len(self.teams_)
        ms_list = _rts_smooth(
            self._m_filtered, self._C_filtered, self._R_predicted
        )
        result = []
        for cutoff, ms_w in zip(self._week_cutoffs, ms_list):
            theta = np.concatenate([
                [self._mu], ms_w[:T], ms_w[T:], [self._eta],
                self._theta_pace,
            ])
            result.append((cutoff, self._summary_from_theta(theta)))
        return result
