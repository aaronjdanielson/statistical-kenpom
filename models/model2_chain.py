"""Model2Chain — Model 2 as one link in a season-to-season chain.

Model 2 starts every season from nothing: each team's offence and defence effect has
prior mean 0 (league average), and the home effect is free. Both cost accuracy early in
a season, when there are few games: last season's rating is a better starting point
than "average", and a free home effect soaks up the fact that strong teams host early.

Here a league's seasons are linked the way the weeks of a season are linked in a
state-space model, with an exact conjugate (Normal-Inverse-Gamma) update per season:

    o_{i,s} = rho * o_{i,s-1} + u,     u ~ N(0, s2 / lambda_team)        (same for d;
                                       corr(u_o, u_d) = od_corr: good teams are good at both ends)
    eta_s   = eta_{s-1} + v,           v ~ N(0, s2 / lambda_home)
    e_k     = mu + o_i - d_j + eta*h + eps,   eps ~ N(0, s2 / poss_k)
    s2      ~ scaled-inverse-chi2(nu0, s2_0)                 (last season's value)

so the prior mean of a team's effect is rho times last season's posterior mean, a team
with no last season starts at 0 as in Model 2, and the posterior is still exact:

    theta_hat = (X'WX + P)^-1 (X'Wy + P m),     Cov = E[s2 | data] * (X'WX + P)^-1
    E[s2 | data] = (nu0 * s2_0 + sum(w r^2) + (theta_hat - m)' P (theta_hat - m)) / (nu0 + n - 3)

With no prior mean, no home prior and no s2 prior this is Model 2 with the same point
estimates. Pace, prediction and sampling are inherited from Model 2 unchanged.

The chain itself (which season feeds which, rho per league type) lives in
scripts/rolling_ratings.py; this class is one fit given its prior.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from scipy import linalg

from models.model2 import Model2


class Model2Chain(Model2):
    """
    Parameters
    ----------
    lambda_team  : ridge penalty on team effects around their prior mean (s2 / innovation variance)
    prior_mean   : {team_id: (o, d)} prior means in points per 100 possessions (already
                   multiplied by rho); teams absent start at 0
    prior_penalty: {team_id: (pen_o, pen_d)} per-team penalties replacing lambda_team
                   (a team whose last-season rating was uncertain gets a weaker one)
    od_corr      : prior correlation between a team's offence and defence effects (d is
                   "points prevented", so positive = good at both); 0 = independent, as Model 2
    home_prior   : prior mean of the home effect; None leaves it free as in Model 2
    lambda_home  : penalty on (eta - home_prior), in possessions
    sigma2_prior : prior value of the per-possession-scale noise variance s2 (last season's)
    sigma2_rows  : its weight, in rows (nu0); 0 = no prior
    """

    def __init__(self, lambda_team: float = 100.0, lambda_pace: float = 50.0, *,
                 prior_mean: Optional[Mapping[int, tuple[float, float]]] = None,
                 prior_penalty: Optional[Mapping[int, tuple[float, float]]] = None,
                 od_corr: float = 0.0,
                 home_prior: Optional[float] = None, lambda_home: float = 0.0,
                 sigma2_prior: Optional[float] = None, sigma2_rows: float = 0.0) -> None:
        super().__init__(lambda_team=lambda_team, lambda_pace=lambda_pace)
        self.prior_mean = dict(prior_mean or {})
        self.prior_penalty = dict(prior_penalty or {})
        if not -1.0 < od_corr < 1.0:
            raise ValueError("od_corr must be strictly between -1 and 1")
        self.od_corr = float(od_corr)
        self.home_prior = home_prior
        self.lambda_home = float(lambda_home)
        self.sigma2_prior = sigma2_prior
        self.sigma2_rows = float(sigma2_rows) if sigma2_prior is not None else 0.0

    def _fit_efficiency(self, rows, sample_weight=None):
        T = len(self.teams_)
        tidx = self._tidx
        N = len(rows)
        n_cols = 2 * T + 2                      # [mu, o x T, d x T, eta]

        ti = np.fromiter((tidx[r.team_id] for r in rows), dtype=np.int64, count=N)
        oi = np.fromiter((tidx[r.opp_id] for r in rows), dtype=np.int64, count=N)
        h = np.fromiter((r.h for r in rows), dtype=np.float64, count=N)
        poss = np.fromiter((r.poss for r in rows), dtype=np.float64, count=N)
        y = np.fromiter((r.pts for r in rows), dtype=np.float64, count=N) / poss * 100.0
        w = poss * (np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else 1.0)

        X = np.zeros((N, n_cols))
        k = np.arange(N)
        X[k, 0] = 1.0
        X[k, 1 + ti] = 1.0
        X[k, 1 + T + oi] = -1.0
        X[k, 2 * T + 1] = h

        pen = np.full(n_cols, float(self.lambda_team))
        m = np.zeros(n_cols)
        for i, tid in enumerate(self.teams_):
            tid = int(tid)
            if tid in self.prior_mean:
                m[1 + i], m[1 + T + i] = self.prior_mean[tid]
            if tid in self.prior_penalty:
                pen[1 + i], pen[1 + T + i] = self.prior_penalty[tid]

        # Prior precision P. Per team the (o_i, d_i) block is the inverse of
        # [[1/pen_o, r/sqrt(pen_o pen_d)], [r/sqrt(pen_o pen_d), 1/pen_d]]; r = 0 gives Model 2's diagonal.
        io, idd, r = 1 + np.arange(T), 1 + T + np.arange(T), self.od_corr
        P = np.zeros((n_cols, n_cols))
        P[io, io] = pen[io] / (1.0 - r * r)
        P[idd, idd] = pen[idd] / (1.0 - r * r)
        P[io, idd] = P[idd, io] = -r * np.sqrt(pen[io] * pen[idd]) / (1.0 - r * r)
        P[0, 0] = 1e-4                          # mu: free (tiny ridge for a stable solve)
        anchored = self.home_prior is not None and self.lambda_home > 0
        P[2 * T + 1, 2 * T + 1] = self.lambda_home if anchored else 1e-4      # eta: anchored, or free as Model 2
        if anchored:
            m[2 * T + 1] = self.home_prior

        Xw = X * w[:, None]
        A = Xw.T @ X + P
        b = Xw.T @ y + P @ m
        theta_hat = linalg.solve(A, b, assume_a="sym")

        resid = y - X @ theta_hat
        self._sigma2_eff = float(np.average(resid ** 2, weights=w))      # per-game variance, as Model 2
        A_inv = linalg.inv(A)
        n = float(N) if sample_weight is None else float(np.sum(sample_weight))
        rss = float(np.sum(w * resid ** 2))
        self._df_eff = float(n_cols - np.sum(A_inv * P))                 # tr(A^-1 X'WX)
        if self.sigma2_rows > 0:
            # conjugate update of s2: flat parameters (mu, and eta when free) each cost a row
            flat = 1.0 if anchored else 2.0
            dev = theta_hat - m
            dev[0] = 0.0
            if not anchored:
                dev[2 * T + 1] = 0.0
            quad = float(dev @ P @ dev)
            self._sigma2_poss = (self.sigma2_rows * self.sigma2_prior + rss + quad) / max(self.sigma2_rows + n - flat - 2.0, 1.0)
        else:
            self._sigma2_poss = rss / max(n - self._df_eff, 1.0)         # as Model 2
        self._prior_mean_vec = m
        return theta_hat, self._sigma2_poss * A_inv
