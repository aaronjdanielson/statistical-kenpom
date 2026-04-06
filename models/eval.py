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
from typing import Sequence, Type

import numpy as np

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
                           model_cls.__name__, season, e)
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
