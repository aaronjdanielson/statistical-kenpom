"""Shared interface for all three rating models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KenPomSummary:
    """
    The three-number KenPom-style summary for one team in one season.

    adj_o    — adjusted offensive efficiency (pts per 100 poss vs average defense)
    adj_d    — adjusted defensive efficiency (pts allowed per 100 poss vs average offense)
    adj_pace — adjusted tempo (possessions per game vs average opponent)

    Lower adj_d is better defense; higher adj_o is better offense.
    """
    adj_o: float
    adj_d: float
    adj_pace: float

    @property
    def net_rtg(self) -> float:
        return self.adj_o - self.adj_d


class BaseModel(ABC):
    """
    Contract shared by Models 1-3.

    Core methods every model must implement:
      fit_rows(rows, season)       — estimate parameters from GameRow list
      sample_posterior(n, rng)     — draw n samples from p(θ | data)
      _summary_from_theta(theta)   — θ → {team_id: KenPomSummary}
      _predict_from_theta(theta, rows) → np.ndarray of predicted e_off per row

    Derived public API (implemented here, do not override):
      fit(season, conn)            — load rows from DB, then call fit_rows
      predict_efficiency(rows)     — point predictions at theta_hat_
      predict_interval(rows, ...)  — posterior predictive intervals
      sample_kenpom_summary(n, rng)
      point_summary()
    """

    # Set by fit_rows():
    theta_hat_: np.ndarray
    teams_: np.ndarray
    season_: int

    # ------------------------------------------------------------------ #
    # Abstract methods — must implement in each subclass                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit_rows(self, rows, season: int) -> "BaseModel":
        """Fit on a pre-loaded list of GameRow. Returns self."""
        ...

    @abstractmethod
    def sample_posterior(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Return n draws from the approximate posterior over θ."""
        ...

    @abstractmethod
    def _summary_from_theta(self, theta: np.ndarray) -> dict[int, KenPomSummary]:
        """Map θ → {team_id: KenPomSummary}."""
        ...

    @abstractmethod
    def _predict_from_theta(self, theta: np.ndarray, rows) -> np.ndarray:
        """
        Predicted offensive efficiency (pts/100) for each row, given θ.

        Returns array of length len(rows).  Teams absent from the training
        set fall back to the league baseline (μ for additive models, mean(O)
        for the multiplicative model).
        """
        ...

    # ------------------------------------------------------------------ #
    # Derived public API                                                    #
    # ------------------------------------------------------------------ #

    def fit(self, season: int, conn) -> "BaseModel":
        """Load rows from DB and fit. Thin wrapper around fit_rows."""
        from models.data import load_season_games
        rows = load_season_games(conn, season)
        if not rows:
            raise ValueError(f"No games found for season {season}")
        return self.fit_rows(rows, season)

    def predict_efficiency(self, rows) -> np.ndarray:
        """Point predictions of offensive efficiency (pts/100) for each row."""
        return self._predict_from_theta(self.theta_hat_, rows)

    def predict_interval(
        self,
        rows,
        coverage: float = 0.95,
        n: int = 500,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Posterior predictive interval for offensive efficiency.

        Draws n samples from the posterior, computes predicted efficiency for
        each draw, and returns (lower, mean, upper) arrays.

        The interval integrates parameter uncertainty only; to add irreducible
        game noise, set add_noise=True (not implemented here — callers that
        need coverage of actual outcomes should add σ_eff noise externally).
        """
        if rng is None:
            rng = np.random.default_rng()
        draws = self.sample_posterior(n, rng)
        preds = np.array([self._predict_from_theta(theta, rows) for theta in draws])
        alpha = (1 - coverage) / 2
        lo  = np.quantile(preds, alpha, axis=0)
        hi  = np.quantile(preds, 1 - alpha, axis=0)
        mid = preds.mean(axis=0)
        return lo, mid, hi

    def sample_kenpom_summary(
        self, n: int, rng: np.random.Generator
    ) -> list[dict[int, KenPomSummary]]:
        """
        Hard contract:
            [_summary_from_theta(θ) for θ in sample_posterior(n, rng)]
        """
        return [self._summary_from_theta(theta) for theta in self.sample_posterior(n, rng)]

    def point_summary(self) -> dict[int, KenPomSummary]:
        """KenPom summaries at the point estimate."""
        return self._summary_from_theta(self.theta_hat_)
