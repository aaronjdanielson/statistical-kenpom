"""
Prior construction utilities for cross-season continuity models.

The main entry points are:

  load_returning_minutes(season)        → dict[team_id, r_i]
  build_continuity_prior(teams, ...)    → (m, p_diag)

Statistical formulation
-----------------------
For each team i entering season s, let r_i ∈ [0,1] be the fraction of minutes
returned from season s-1.  The prior on the offense and defense effects is:

    o_i^s ~ N( r_i * ô_{i,s-1},  v_o(r_i) )
    d_i^s ~ N( r_i * d̂_{i,s-1},  v_d(r_i) )

where

    v_o(r) = tau_o_hi² - r * (tau_o_hi² - tau_o_lo²)

(linear interpolation from the wide new-team prior at r=0 to the tight
stable-roster prior at r=1).

The prior precision in solver units (matching Model2's convention) is:

    p_i = sigma2_ref / v_i

so that sigma2_eff * P^{-1} = V_prior in the posterior covariance
    Sigma = sigma2_eff * (X'WX + P)^{-1}.

tau values are in pts/100 units.  With sigma2_ref ≈ 196 (14² pts/100):
  • tau_hi = sqrt(sigma2_ref/lambda_team) ≈ 1.4 matches Model2's default ridge
    at r=0 — useful as a "same as Model2 for new teams" reference point.
  • tau_lo ≪ tau_hi anchors stable-roster teams tightly to last season.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

_PORTAL_DB = Path.home() / "Dropbox" / "portal_pivot_db" / "ncaa.db"


# ── Data loading ────────────────────────────────────────────────────────────────

def load_returning_minutes(
    season: int,
    db_path: str | Path = _PORTAL_DB,
) -> dict[int, float]:
    """
    Compute the fraction of minutes returned for each team entering `season`.

    A player "returns" if they appear with MIN > 0 at the same TeamID in both
    `season` and `season - 1`.  Transfer arrivals and true freshmen are NOT
    counted as returning; transferred-out players do not appear in `season`.

    r_i = sum(MIN for same-team returners in season s)
          ─────────────────────────────────────────────
          sum(MIN for all players in season s)

    Coverage: portal_pivot_db has player_summaries from 2018–2026, so returning
    minutes are available for transitions 2018→2019 through 2025→2026.

    Parameters
    ----------
    season  : season end-year being fitted (e.g. 2026 for the 2025-26 season)
    db_path : path to portal_pivot_db/ncaa.db

    Returns
    -------
    dict[team_id → r_i in [0,1]].  Teams with no data in either season
    are absent from the dict and should default to r=0 (weak prior).
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            a.TeamID,
            COALESCE(SUM(a.MIN), 0)   AS returning_min,
            t.total_min
        FROM player_summaries a
        JOIN player_summaries b
            ON  a.PlayerID = b.PlayerID
            AND a.TeamID   = b.TeamID
            AND a.Season   = b.Season + 1
            AND b.MIN > 0
        JOIN (
            SELECT TeamID, SUM(MIN) AS total_min
            FROM   player_summaries
            WHERE  Season = ?
            GROUP  BY TeamID
        ) t ON t.TeamID = a.TeamID
        WHERE a.Season = ? AND a.MIN > 0
        GROUP BY a.TeamID
    """, (season, season))
    rows = cur.fetchall()
    conn.close()

    result: dict[int, float] = {}
    for tid, ret_min, total_min in rows:
        if total_min and total_min > 0:
            result[int(tid)] = min(1.0, float(ret_min) / float(total_min))
    return result


def extract_prev_effects(
    model,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """
    Extract raw (o_i, d_i) effects and their posterior variances from a fitted Model2.

    Returns
    -------
    effects : {team_id: (o_hat, d_hat)}
    variances: {team_id: (var_o, var_d)}  — diagonal of the efficiency posterior covariance
    """
    T     = len(model.teams_)
    theta = model.theta_hat_
    o     = theta[1:1 + T]
    d     = theta[1 + T:1 + 2 * T]

    # Diagonal of the efficiency posterior covariance (Sigma_eff)
    var_diag = np.diag(model._Sigma_eff)
    var_o    = var_diag[1:1 + T]
    var_d    = var_diag[1 + T:1 + 2 * T]

    effects:   dict[int, tuple[float, float]] = {}
    variances: dict[int, tuple[float, float]] = {}
    for idx, tid in enumerate(model.teams_):
        tid = int(tid)
        effects[tid]   = (float(o[idx]), float(d[idx]))
        variances[tid] = (float(var_o[idx]), float(var_d[idx]))

    return effects, variances


# ── Prior construction ──────────────────────────────────────────────────────────

def build_continuity_prior(
    teams: np.ndarray,
    prev_effects: dict[int, tuple[float, float]],
    r_minutes: dict[int, float],
    tau_o_lo: float,
    tau_o_hi: float,
    tau_d_lo: float,
    tau_d_hi: float,
    sigma2_ref: float,
    *,
    prev_var: dict[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the prior mean vector m and prior precision diagonal for the efficiency
    block of the Model2 solver.

    Efficiency block layout: [mu, o_0, …, o_{T-1}, d_0, …, d_{T-1}, eta]
    length = 2T + 2.

    For each team i:
        m[1+i]     = r_i * prev_o_i
        m[1+T+i]   = r_i * prev_d_i
        v_o        = tau_o_hi² - r_i*(tau_o_hi² - tau_o_lo²)
        p_diag[i]  = sigma2_ref / v_o        (prior precision in solver units)

    If prev_var is supplied, the prior variance is inflated by the previous
    season's posterior uncertainty (avoids overconfident priors):
        v_o += r_i² * Var(o_i^{s-1})

    Parameters
    ----------
    teams       : 1-D int array of team IDs in solver canonical order
    prev_effects: {team_id: (o_hat, d_hat)} raw effects from previous season
    r_minutes   : {team_id: r_i}; missing teams default to r=0
    tau_o_lo    : prior std for offense at r=1 (pts/100)
    tau_o_hi    : prior std for offense at r=0 (pts/100)
    tau_d_lo    : prior std for defense at r=1 (pts/100)
    tau_d_hi    : prior std for defense at r=0 (pts/100)
    sigma2_ref  : noise variance estimate used to scale solver precision;
                  pass previous season's model._sigma2_eff
    prev_var    : optional {team_id: (var_o, var_d)} — previous-season posterior
                  variances from extract_prev_effects()

    Returns
    -------
    m       : prior mean, shape (2T+2,)   — zero for mu and eta
    p_diag  : prior precision diagonal, shape (2T+2,) — zero for mu and eta
    """
    T      = len(teams)
    m      = np.zeros(2 * T + 2)
    p_diag = np.zeros(2 * T + 2)

    for idx, tid in enumerate(teams):
        r = float(np.clip(r_minutes.get(int(tid), 0.0), 0.0, 1.0))
        o_hat, d_hat = prev_effects.get(int(tid), (0.0, 0.0))

        m[1 + idx]     = r * o_hat
        m[1 + T + idx] = r * d_hat

        v_o = tau_o_hi ** 2 - r * (tau_o_hi ** 2 - tau_o_lo ** 2)
        v_d = tau_d_hi ** 2 - r * (tau_d_hi ** 2 - tau_d_lo ** 2)

        if prev_var is not None:
            var_o_prev, var_d_prev = prev_var.get(int(tid), (0.0, 0.0))
            v_o += r ** 2 * var_o_prev
            v_d += r ** 2 * var_d_prev

        p_diag[1 + idx]     = sigma2_ref / max(v_o, 1e-10)
        p_diag[1 + T + idx] = sigma2_ref / max(v_d, 1e-10)

    return m, p_diag
