"""Predictive evaluation framework for the three rating models.

Workflow
--------
1. Load a full season of GameRows from ncaa.db.
2. Split into train / test by unique game_id (temporal — game_ids are
   approximately date-ordered within a season).
3. Fit a model on the training rows.
4. Predict offensive efficiency (pts/100) for every test row.
5. Compare predictions to actuals and report RMSE, MAE, bias, and the
   coverage of posterior predictive intervals.

The key quantity being evaluated is:

    e_off = pts_scored / poss * 100

for each team-game observation in the hold-out set.

Functions
---------
temporal_split(rows, train_frac)    → (train, test) game-level split
evaluate_season(model_cls, conn, season, ...)  → EvalResult
evaluate_all_seasons(model_cls, conn, seasons, ...) → list[EvalResult]
print_report(results)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence, Type

import numpy as np
from scipy.stats import norm as _norm

from models.data import GameRow, load_season_games

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    model_name: str
    season: int
    n_train_games: int
    n_test_games: int
    n_test_obs: int          # rows = 2 × games (one per team side)
    rmse: float              # root mean squared error on e_off
    mae: float               # mean absolute error on e_off
    bias: float              # mean signed error (predicted − actual)
    coverage_90: float       # fraction of actuals inside 90% posterior PI
    coverage_95: float       # fraction of actuals inside 95% posterior PI
    interval_width_95: float # median width of 95% PI (pts/100)
    sigma_eff: float = 0.0   # model's estimated residual std (for reference)


# --------------------------------------------------------------------------- #
# Train / test split                                                            #
# --------------------------------------------------------------------------- #

def temporal_split(
    rows: list[GameRow],
    train_frac: float = 0.8,
) -> tuple[list[GameRow], list[GameRow]]:
    """
    Split rows into train / test by game_id order.

    Game IDs within a season are approximately date-ordered (RealGM assigns
    them sequentially).  We sort unique game IDs and take the first
    `train_frac` fraction as training, the rest as test.

    Both rows of each game (team A scoring, team B scoring) always land in
    the same split — the test set contains complete games.
    """
    game_ids = sorted({r.game_id for r in rows})
    n_train = max(1, int(len(game_ids) * train_frac))
    train_ids = set(game_ids[:n_train])
    test_ids  = set(game_ids[n_train:])
    train = [r for r in rows if r.game_id in train_ids]
    test  = [r for r in rows if r.game_id in test_ids]
    return train, test


# --------------------------------------------------------------------------- #
# Core evaluation                                                               #
# --------------------------------------------------------------------------- #

def evaluate_season(
    model_cls,
    conn,
    season: int,
    train_frac: float = 0.8,
    n_interval: int = 300,
    rng_seed: int = 0,
    **model_kwargs,
) -> EvalResult:
    """
    Fit model_cls on the training split of `season`, evaluate on the test split.

    Parameters
    ----------
    model_cls : BaseModel subclass (not instance)
    conn      : sqlite3 connection to ncaa.db
    season    : season end-year (e.g. 2023)
    train_frac: fraction of games (by game_id order) used for training
    n_interval: posterior draws for PI coverage calculation
    rng_seed  : reproducibility seed
    model_kwargs: passed to model_cls constructor

    Returns
    -------
    EvalResult with point-prediction and interval-coverage metrics.
    """
    from models.base import BaseModel

    rows = load_season_games(conn, season)
    train_rows, test_rows = temporal_split(rows, train_frac)

    model: BaseModel = model_cls(**model_kwargs)
    model.fit_rows(train_rows, season)

    # ---- Point predictions ----
    actual  = np.array([r.pts / r.poss * 100.0 for r in test_rows])
    pred    = model.predict_efficiency(test_rows)
    resid   = pred - actual
    rmse    = float(np.sqrt(np.mean(resid ** 2)))
    mae     = float(np.mean(np.abs(resid)))
    bias    = float(np.mean(resid))

    # ---- Posterior predictive interval coverage ----
    # Draw posterior samples once; compute both 90% and 95% from the same draws.
    rng = np.random.default_rng(rng_seed)
    draws = model.sample_posterior(n_interval, rng)
    preds = np.array([model._predict_from_theta(theta, test_rows) for theta in draws])
    lo90 = np.quantile(preds, 0.05,  axis=0)
    hi90 = np.quantile(preds, 0.95,  axis=0)
    lo95 = np.quantile(preds, 0.025, axis=0)
    hi95 = np.quantile(preds, 0.975, axis=0)

    # Add irreducible game noise to the PI bounds (parameter uncertainty alone
    # undershoots coverage because actual games have additional noise on top).
    sigma = float(np.sqrt(getattr(model, "_sigma2_eff", 0.0)))
    z90 = 1.645
    z95 = 1.960
    lo90_full = lo90 - z90 * sigma
    hi90_full = hi90 + z90 * sigma
    lo95_full = lo95 - z95 * sigma
    hi95_full = hi95 + z95 * sigma

    cov90 = float(np.mean((actual >= lo90_full) & (actual <= hi90_full)))
    cov95 = float(np.mean((actual >= lo95_full) & (actual <= hi95_full)))
    width95 = float(np.median(hi95_full - lo95_full))

    n_train_games = len({r.game_id for r in train_rows})
    n_test_games  = len({r.game_id for r in test_rows})

    logger.info(
        "%s season=%d  train=%d games  test=%d games  "
        "RMSE=%.2f  MAE=%.2f  bias=%+.2f  cov95=%.1f%%",
        model_cls.__name__, season, n_train_games, n_test_games,
        rmse, mae, bias, cov95 * 100,
    )

    return EvalResult(
        model_name     = model_cls.__name__,
        season         = season,
        n_train_games  = n_train_games,
        n_test_games   = n_test_games,
        n_test_obs     = len(test_rows),
        rmse           = rmse,
        mae            = mae,
        bias           = bias,
        coverage_90    = cov90,
        coverage_95    = cov95,
        interval_width_95 = width95,
        sigma_eff      = sigma,
    )


def evaluate_all_seasons(
    model_cls,
    conn,
    seasons: Sequence[int],
    train_frac: float = 0.8,
    n_interval: int = 300,
    rng_seed: int = 0,
    **model_kwargs,
) -> list[EvalResult]:
    """Run evaluate_season for each season and return results."""
    results = []
    for season in seasons:
        try:
            r = evaluate_season(
                model_cls, conn, season,
                train_frac=train_frac,
                n_interval=n_interval,
                rng_seed=rng_seed,
                **model_kwargs,
            )
            results.append(r)
        except Exception as e:
            logger.warning("evaluate_season failed for %s season=%d: %s",
                           getattr(model_cls, "__name__", str(model_cls)), season, e)
    return results


# --------------------------------------------------------------------------- #
# Reporting                                                                     #
# --------------------------------------------------------------------------- #

def print_report(results: list[EvalResult]) -> None:
    """Print a formatted comparison table."""
    if not results:
        print("No results.")
        return

    header = (
        f"{'Model':>10}  {'Season':>6}  {'Train':>5}  {'Test':>5}  "
        f"{'RMSE':>6}  {'MAE':>6}  {'Bias':>6}  "
        f"{'Cov90':>6}  {'Cov95':>6}  {'PI-Width':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.model_name:>10}  {r.season:>6}  {r.n_train_games:>5}  "
            f"{r.n_test_games:>5}  "
            f"{r.rmse:>6.2f}  {r.mae:>6.2f}  {r.bias:>+6.2f}  "
            f"{r.coverage_90:>6.1%}  {r.coverage_95:>6.1%}  "
            f"{r.interval_width_95:>8.1f}"
        )
    print(sep)

    # Aggregate summary across seasons per model
    from itertools import groupby
    print()
    agg_header = f"{'Model':>10}  {'Seasons':>7}  {'RMSE':>6}  {'MAE':>6}  {'Cov95':>6}"
    print(agg_header)
    print("-" * len(agg_header))
    for name, group in groupby(sorted(results, key=lambda x: x.model_name),
                                key=lambda x: x.model_name):
        g = list(group)
        print(
            f"{name:>10}  {len(g):>7}  "
            f"{np.mean([r.rmse for r in g]):>6.2f}  "
            f"{np.mean([r.mae  for r in g]):>6.2f}  "
            f"{np.mean([r.coverage_95 for r in g]):>6.1%}"
        )


# --------------------------------------------------------------------------- #
# Win probability                                                               #
# --------------------------------------------------------------------------- #

def recency_weights(
    rows: list[GameRow],
    game_dates: dict[int, datetime],
    cutoff_date: datetime,
    half_life_days: float,
) -> np.ndarray:
    """
    Compute exponential recency weights for ``Model2.fit_rows(sample_weight=...)``.

    Each row is assigned weight::

        w_i = exp(-κ · age_i),   κ = log(2) / half_life_days

    where ``age_i = (cutoff_date − game_date).days``.  Rows whose game date is
    at or after the cutoff date receive weight 1.0.

    Parameters
    ----------
    rows          : GameRow list passed to fit_rows
    game_dates    : mapping from game_id to the game's datetime
    cutoff_date   : the prediction date (typically the start of the test window)
    half_life_days: number of days after which a game's weight is halved

    Returns
    -------
    np.ndarray of shape (len(rows),) with values in (0, 1].

    Example
    -------
    >>> sw = recency_weights(train_rows, game_dt, cutoff, half_life_days=45)
    >>> model = Model2()
    >>> model.fit_rows(train_rows, season, sample_weight=sw)
    """
    kappa = np.log(2.0) / half_life_days
    weights = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        age = max(0.0, (cutoff_date - game_dates[r.game_id]).total_seconds() / 86400.0)
        weights[i] = np.exp(-kappa * age)
    return weights


def win_probability(
    model,
    team_id: int,
    opp_id: int,
    *,
    h: int = 0,
    poss: float = 70.0,
    n_draws: int = 500,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Probability that team_id beats opp_id in a game with expected possessions
    ``poss`` and venue indicator ``h`` (+1=home, 0=neutral, -1=away for team_id).

    Uses the Normal approximation

        P(win) = Φ(pred_margin / pred_std)

    where:
      pred_margin = expected point margin (team_id pts − opp_id pts) from
                    posterior mean efficiency predictions.
      pred_std    = combined posterior parameter uncertainty and irreducible
                    per-possession game noise (model's σ_eff).

    Parameters
    ----------
    model     : fitted BaseModel instance
    team_id   : integer team identifier for the team of interest
    opp_id    : integer team identifier for the opponent
    h         : venue for team_id (+1 home, 0 neutral, -1 away)
    poss      : expected possessions per game (used to convert pts/100 → pts)
    n_draws   : posterior draws used to estimate parameter variance
    rng       : numpy Generator; a fresh one is created if None

    Returns
    -------
    float in [0, 1] — probability that team_id wins.
    """
    if rng is None:
        rng = np.random.default_rng()

    season = getattr(model, "season_", 0)
    scale  = poss / 100.0

    row_team = GameRow(game_id=0, season=season,
                       team_id=team_id, opp_id=opp_id,
                       pts=0, poss=poss, h=h)
    row_opp  = GameRow(game_id=0, season=season,
                       team_id=opp_id,  opp_id=team_id,
                       pts=0, poss=poss, h=-h)

    draws = model.sample_posterior(n_draws, rng)
    # margin in actual points for each posterior draw
    margin_draws = np.array([
        (model._predict_from_theta(th, [row_team])[0]
         - model._predict_from_theta(th, [row_opp])[0]) * scale
        for th in draws
    ])

    pred_margin = float(margin_draws.mean())

    # Irreducible noise: two independent scoring processes, each with std
    # σ_eff pts/100, converted to points.
    sigma_eff   = float(np.sqrt(getattr(model, "_sigma2_eff", 0.0)))
    sigma_noise = np.sqrt(2.0) * sigma_eff * scale

    pred_std = float(np.sqrt(margin_draws.var() + sigma_noise ** 2))

    if pred_std <= 0:
        return 0.5 if pred_margin == 0 else float(pred_margin > 0)
    return float(_norm.cdf(pred_margin / pred_std))


# --------------------------------------------------------------------------- #
# Conformal calibration                                                         #
# --------------------------------------------------------------------------- #

_DEFAULT_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def conformal_calibration_scores(
    model,
    test_rows: list[GameRow],
    actual: np.ndarray | None = None,
    n_draws: int = 300,
    rng: np.random.Generator | None = None,
    levels: Sequence[float] = _DEFAULT_LEVELS,
) -> tuple[np.ndarray, dict[float, float]]:
    """
    Compute conformal nonconformity scores and empirical coverage table.

    For each test observation i:
        z_i = (y_i − ŷ_i) / σ_total_i

    where σ_total combines posterior parameter uncertainty (std of posterior
    predictions across draws) with irreducible game noise (σ_eff).  Under a
    correctly-specified Gaussian model, z_i ~ N(0, 1).

    Parameters
    ----------
    model      : fitted BaseModel instance
    test_rows  : GameRow list for test observations
    actual     : observed efficiencies (pts/100); computed from test_rows if None
    n_draws    : posterior draws for σ_param estimation
    rng        : numpy Generator; fresh one created if None
    levels     : nominal coverage levels to evaluate

    Returns
    -------
    z_scores : np.ndarray, shape (n_obs,)
        Normalized residuals.  Values close to N(0,1) indicate good calibration.
    coverage : dict[float, float]
        {nominal_level: empirical_coverage} — compares the model's Gaussian PI
        at each nominal level to the fraction of z_scores inside ±z_{level}.
    """
    if rng is None:
        rng = np.random.default_rng()

    if actual is None:
        actual = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    # Point predictions (MAP)
    pred_map = model.predict_efficiency(test_rows)

    # Posterior prediction std (parameter uncertainty component)
    draws = model.sample_posterior(n_draws, rng)
    preds_draws = np.array([
        model._predict_from_theta(th, test_rows) for th in draws
    ])          # (n_draws, n_obs)
    sigma_param = preds_draws.std(axis=0)          # (n_obs,)

    # Irreducible noise
    sigma_eff   = float(np.sqrt(getattr(model, "_sigma2_eff", 0.0)))
    sigma_total = np.sqrt(sigma_param ** 2 + sigma_eff ** 2)
    sigma_total = np.maximum(sigma_total, 1e-8)    # guard division by zero

    z_scores = (actual - pred_map) / sigma_total

    # Empirical coverage at each nominal level
    coverage: dict[float, float] = {}
    for level in levels:
        z_crit = float(_norm.ppf((1 + level) / 2))
        coverage[level] = float(np.mean(np.abs(z_scores) <= z_crit))

    return z_scores, coverage
