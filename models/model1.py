"""Model 1: KenPom-style multiplicative fixed-point rating system.

Each team has three scalar latent quantities:
    O_i  — adjusted offensive efficiency (pts per 100 poss vs avg defense)
    D_i  — adjusted defensive efficiency (pts per 100 poss allowed vs avg offense)
    R_i  — adjusted tempo (possessions per game vs avg opponent)

These are found by iterating the self-consistency equations from algorithms.md
until convergence.  Uncertainty is quantified by parametric bootstrap:
resample games with replacement, refit, collect draws.

Theta layout (for _summary_from_theta compatibility):
    theta[:T]       = O_0,...,O_{T-1}   (AdjO, already in absolute pts/100)
    theta[T:2T]     = D_0,...,D_{T-1}   (AdjD, already in absolute pts/100)
    theta[2T:3T]    = R_0,...,R_{T-1}   (AdjPace, already in poss/game)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from models.base import BaseModel, KenPomSummary
from models.data import GameRow

logger = logging.getLogger(__name__)

# Home-court adjustments (multiplicative)
_L_OFF  = {"home": 1.014, "neutral": 1.0, "away": 1.0 / 1.014}
_L_DEF  = {"home": 1.0 / 1.014, "neutral": 1.0, "away": 1.014}
_L_PACE = {"home": 1.0, "neutral": 1.0, "away": 1.0}   # venue has minimal pace effect


def _h_to_venue(h: int) -> str:
    if h == 1:
        return "home"
    if h == -1:
        return "away"
    return "neutral"


class Model1(BaseModel):
    """KenPom-style iterative opponent-adjustment + parametric bootstrap."""

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_boot: int = 200,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.n_boot = n_boot

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit_rows(self, rows, season: int) -> "Model1":
        self.season_ = season
        self._rows = list(rows)
        teams = sorted({r.team_id for r in rows} | {r.opp_id for r in rows})
        self.teams_ = np.array(teams, dtype=np.int64)
        self._tidx = {tid: i for i, tid in enumerate(teams)}
        self.theta_hat_ = self._fixed_point(rows)

        # Compute in-sample residual variance so eval.py can add game noise
        # to bootstrap prediction intervals.
        e_off   = np.array([r.pts / r.poss * 100.0 for r in rows])
        pred    = self._predict_from_theta(self.theta_hat_, rows)
        self._sigma2_eff = float(np.mean((e_off - pred) ** 2))

        logger.info("Model1 season=%d  teams=%d  games=%d  σ_eff=%.3f",
                    season, len(teams), len(self._rows) // 2,
                    float(np.sqrt(self._sigma2_eff)))
        return self

    # ------------------------------------------------------------------ #
    # Posterior (bootstrap)                                                #
    # ------------------------------------------------------------------ #

    def sample_posterior(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """
        Draw n bootstrap replicates of theta by resampling games.

        Resample at the game level (both rows of a game move together) so that
        each synthetic dataset has the same number of unique games as the
        original.
        """
        game_ids = np.array(sorted({r.game_id for r in self._rows}))
        game_to_rows: dict[int, list[GameRow]] = {}
        for r in self._rows:
            game_to_rows.setdefault(r.game_id, []).append(r)

        draws = []
        for _ in range(n):
            idx = rng.integers(0, len(game_ids), size=len(game_ids))
            boot_rows: list[GameRow] = []
            for gid in game_ids[idx]:
                boot_rows.extend(game_to_rows[gid])
            draws.append(self._fixed_point(boot_rows))
        return draws

    def _summary_from_theta(self, theta: np.ndarray) -> dict[int, KenPomSummary]:
        T = len(self.teams_)
        O = theta[:T]
        D = theta[T:2 * T]
        R = theta[2 * T:3 * T]
        return {
            int(tid): KenPomSummary(adj_o=float(O[i]), adj_d=float(D[i]), adj_pace=float(R[i]))
            for i, tid in enumerate(self.teams_)
        }

    def _predict_from_theta(self, theta: np.ndarray, rows) -> np.ndarray:
        """Multiplicative prediction: O_i * (D_j / μ) * L(h)."""
        T = len(self.teams_)
        O   = theta[:T]
        D   = theta[T:2 * T]
        mu  = float(O.mean())
        tidx = self._tidx
        out = np.empty(len(rows))
        for k, r in enumerate(rows):
            i = tidx.get(r.team_id)
            j = tidx.get(r.opp_id)
            O_i = float(O[i]) if i is not None else mu
            D_j = float(D[j]) if j is not None else mu
            l   = _L_OFF[_h_to_venue(r.h)]
            out[k] = O_i * (D_j / mu) * l
        return out

    # ------------------------------------------------------------------ #
    # Fixed-point iteration                                                #
    # ------------------------------------------------------------------ #

    def _fixed_point(self, rows: Sequence[GameRow]) -> np.ndarray:
        tidx = self._tidx
        T = len(self.teams_)

        # Pre-compute per-row quantities
        # e_off_raw = pts / poss * 100
        team_idx  = np.array([tidx[r.team_id] for r in rows], dtype=np.int32)
        opp_idx   = np.array([tidx[r.opp_id]  for r in rows], dtype=np.int32)
        e_off_raw = np.array([r.pts / r.poss * 100.0 for r in rows])
        poss_arr  = np.array([r.poss for r in rows])
        venue     = [_h_to_venue(r.h) for r in rows]

        l_off  = np.array([_L_OFF[v]  for v in venue])
        l_def  = np.array([_L_DEF[v]  for v in venue])
        l_pace = np.array([_L_PACE[v] for v in venue])
        w = poss_arr  # weight by possessions

        mu_off  = float(np.average(e_off_raw, weights=w))
        mu_pace = float(np.average(poss_arr, weights=np.ones_like(poss_arr)))

        O = np.full(T, mu_off)
        D = np.full(T, mu_off)
        R = np.full(T, mu_pace)

        for _ in range(self.max_iter):
            O_new = np.zeros(T)
            D_new = np.zeros(T)
            R_new = np.zeros(T)
            W_O   = np.zeros(T)
            W_D   = np.zeros(T)
            W_R   = np.zeros(T)

            # Offense update: team_idx scored e_off_raw against opp_idx defense
            contrib_O = (e_off_raw / l_off) * (mu_off / D[opp_idx]) * w
            np.add.at(O_new, team_idx, contrib_O)
            np.add.at(W_O,   team_idx, w)

            # Defense update: opp_idx allowed e_off_raw to team_idx offense.
            # l_def for opp_idx = L(-h_opp) = L(h_team) = l_off (same value).
            # O[team_idx] = opponent's offensive quality used to normalize.
            contrib_D = (e_off_raw / l_off) * (mu_off / O[team_idx]) * w
            np.add.at(D_new, opp_idx, contrib_D)
            np.add.at(W_D,   opp_idx, w)

            # Pace update: both teams share the same possession count.
            contrib_R = (poss_arr / l_pace) * (mu_pace / R[opp_idx]) * w
            np.add.at(R_new, team_idx, contrib_R)
            np.add.at(W_R,   team_idx, w)

            # Avoid divide-by-zero for teams with no games in bootstrap resample
            mask = W_O > 0
            O_new[mask] /= W_O[mask]
            D_new[mask] /= W_D[mask]
            R_new[mask] /= W_R[mask]
            # Teams with no games keep previous estimate
            O_new[~mask] = O[~mask]
            D_new[~mask] = D[~mask]
            R_new[~mask] = R[~mask]

            # Renormalize to anchor league means
            O_new *= mu_off  / O_new.mean()
            D_new *= mu_off  / D_new.mean()
            R_new *= mu_pace / R_new.mean()

            delta = max(
                np.max(np.abs(O_new - O)),
                np.max(np.abs(D_new - D)),
                np.max(np.abs(R_new - R)),
            )
            O, D, R = O_new, D_new, R_new
            if delta < self.tol:
                break

        return np.concatenate([O, D, R])
