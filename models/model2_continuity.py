"""
Model2ContinuityPrior — Model 2 with a cross-season continuity prior.

The efficiency solver replaces Model2's uniform zero-centered ridge penalty with
a shifted, team-specific Gaussian prior built from last season's ratings and each
team's returning-minutes fraction.

Closed-form MAP (same structure as Model2, just different P and b):

    θ̂ = (X'WX + P)⁻¹ (X'Wy + Pm)

Posterior covariance:

    Σ = σ²_eff · (X'WX + P)⁻¹

Everything else — pace model, predict_efficiency, sample_posterior, KenPom
summary — is inherited from Model2 unchanged.

Typical usage
-------------
Fit Model2 on season s-1 to obtain previous-season effects, then:

    from models.model2 import Model2
    from models.model2_continuity import Model2ContinuityPrior
    from models.priors import extract_prev_effects, load_returning_minutes

    m_prev = Model2().fit_rows(prev_rows, season - 1)
    prev_effects, prev_var = extract_prev_effects(m_prev)
    r = load_returning_minutes(season)

    m = Model2ContinuityPrior(
        prev_effects=prev_effects,
        r_minutes=r,
        sigma2_prev=m_prev._sigma2_eff,
        prev_var=prev_var,           # optional: propagate posterior uncertainty
    )
    m.fit_rows(current_rows, season)
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from models.model2 import Model2
from models.priors import build_continuity_prior

logger = logging.getLogger(__name__)


class Model2ContinuityPrior(Model2):
    """
    Model 2 with a cross-season continuity prior on team offense/defense effects.

    Parameters
    ----------
    prev_effects : {team_id: (o_hat, d_hat)}
        Raw (o_i, d_i) effects from a Model2 fit on the previous season.
        Obtain via ``priors.extract_prev_effects(prev_model)``.
        Teams absent from this dict receive prior mean 0 (equivalent to Model2).
    r_minutes    : {team_id: r_i in [0, 1]}
        Fraction of minutes returned.  Obtain via ``priors.load_returning_minutes``.
        Teams absent default to r=0 (no prior information, widest variance).
    tau_o_lo     : prior std for offense when r=1 (full return).  pts/100.
    tau_o_hi     : prior std for offense when r=0 (no return).   pts/100.
    tau_d_lo     : prior std for defense when r=1.  pts/100.
    tau_d_hi     : prior std for defense when r=0.  pts/100.
    sigma2_prev  : σ²_eff from the previous season's Model2 fit, used to scale
                   the prior precision into solver units.  Pass
                   ``prev_model._sigma2_eff``.  Defaults to 196 (14² pts/100)
                   if not provided.
    prev_var     : optional {team_id: (var_o, var_d)}
                   Previous-season posterior variances from ``extract_prev_effects``.
                   When supplied, the prior variance is inflated by r_i² × Var(θ_{s-1})
                   so that teams with uncertain previous ratings enter the new season
                   with appropriately wider priors.
    lambda_team  : passed to Model2 (only affects the tiny mu/eta stability jitter).
    lambda_pace  : passed to Model2 for the pace ridge.
    """

    def __init__(
        self,
        prev_effects: dict[int, tuple[float, float]],
        r_minutes: dict[int, float],
        *,
        tau_o_lo: float = 2.0,
        tau_o_hi: float = 7.0,
        tau_d_lo: float = 2.0,
        tau_d_hi: float = 7.0,
        sigma2_prev: float = 196.0,
        prev_var: dict[int, tuple[float, float]] | None = None,
        shift_only: bool = False,
        lambda_team: float = 100.0,
        lambda_pace: float = 50.0,
    ) -> None:
        super().__init__(lambda_team=lambda_team, lambda_pace=lambda_pace)
        self._prev_effects = dict(prev_effects)
        self._r_minutes    = dict(r_minutes)
        self.tau_o_lo      = tau_o_lo
        self.tau_o_hi      = tau_o_hi
        self.tau_d_lo      = tau_d_lo
        self.tau_d_hi      = tau_d_hi
        self.sigma2_prev   = float(sigma2_prev)
        self._prev_var     = prev_var
        self.shift_only    = shift_only

    # ------------------------------------------------------------------ #
    # Override: efficiency solver only                                     #
    # ------------------------------------------------------------------ #

    def _fit_efficiency(self, rows, sample_weight=None):
        """
        Shifted ridge solve with the continuity prior.

        Replaces the diagonal lambda*I penalty with team-specific P and shifts
        the linear term by Pm so the optimum is pulled toward last season's
        ratings in proportion to returning minutes.
        """
        T      = len(self.teams_)
        tidx   = self._tidx
        N      = len(rows)
        n_cols = 2 * T + 2   # [mu, o×T, d×T, eta]

        X = np.zeros((N, n_cols))
        y = np.empty(N)
        w = np.empty(N)

        for k, r in enumerate(rows):
            i = tidx[r.team_id]
            j = tidx[r.opp_id]
            X[k, 0]         =  1.0
            X[k, 1 + i]     =  1.0
            X[k, 1 + T + j] = -1.0
            X[k, 2 * T + 1] =  float(r.h)
            y[k] = r.pts / r.poss * 100.0
            w[k] = r.poss * (float(sample_weight[k]) if sample_weight is not None else 1.0)

        prior_m, prior_p_diag = build_continuity_prior(
            self.teams_,
            self._prev_effects,
            self._r_minutes,
            self.tau_o_lo,
            self.tau_o_hi,
            self.tau_d_lo,
            self.tau_d_hi,
            self.sigma2_prev,
            prev_var=self._prev_var,
        )

        if self.shift_only:
            # Fix B: keep P identical to Model2's uniform ridge — only the mean
            # shifts.  Isolates the question "does centering at last year's
            # rating help?" without changing regularization strength.
            lam            = np.full(n_cols, self.lambda_team)
            lam[0]         = 1e-4
            lam[2 * T + 1] = 1e-4
            prior_p_diag   = lam

        # mu and eta: same tiny jitter as Model2 for numerical stability
        prior_p_diag[0]         = 1e-4
        prior_p_diag[2 * T + 1] = 1e-4

        P  = np.diag(prior_p_diag)
        Xw = X * w[:, None]
        A  = Xw.T @ X + P             # (X'WX + P)
        b  = Xw.T @ y + P @ prior_m   # X'Wy + Pm

        theta_hat = linalg.solve(A, b, assume_a="sym")

        resid = y - X @ theta_hat
        self._sigma2_eff = float(np.average(resid ** 2, weights=w))

        Sigma = self._sigma2_eff * linalg.inv(A)

        n_with_prior = sum(
            1 for tid in self.teams_
            if int(tid) in self._prev_effects and int(tid) in self._r_minutes
        )
        logger.debug(
            "Model2ContinuityPrior: %d/%d teams have informative prior",
            n_with_prior, T,
        )
        return theta_hat, Sigma
