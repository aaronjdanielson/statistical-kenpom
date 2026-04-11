"""NCAA probabilistic team rating models.

Quick start
-----------
The core workflow requires only ``GameRow`` objects — pre-loaded from any
source — and a model.  No database or network access needed.

.. code-block:: python

    from models import Model2, GameRow, win_probability

    # Build rows from your own data source
    rows = [
        GameRow(game_id=1, season=2026, team_id=5, opp_id=8,
                pts=72, poss=70.0, h=1),   # h: +1=home  0=neutral  -1=away
        GameRow(game_id=1, season=2026, team_id=8, opp_id=5,
                pts=68, poss=70.0, h=-1),
        ...
    ]

    # Fit
    model = Model2()
    model.fit_rows(rows, season=2026)

    # KenPom-style ratings for every team
    summary = model.point_summary()          # {team_id: KenPomSummary}
    for tid, s in summary.items():
        print(tid, s.adj_o, s.adj_d, s.net_rtg)

    # Win probability — neutral site, 70 possessions
    p = win_probability(model, team_id=5, opp_id=8)
    print(f"Team 5 wins: {p:.1%}")

    # Posterior uncertainty
    import numpy as np
    rng = np.random.default_rng(0)
    samples = model.sample_kenpom_summary(500, rng)
    adj_o_draws = [s[5].adj_o for s in samples]   # 500 draws of team 5 AdjO

Loading data from ncaa.db (optional)
--------------------------------------
If your environment has a copy of ncaa.db, set the path once:

.. code-block:: bash

    export NCAA_DB_PATH=/path/to/your/ncaa.db

Then in Python:

.. code-block:: python

    from models import open_db, load_season_games
    conn = open_db()                          # uses NCAA_DB_PATH
    rows = load_season_games(conn, season=2026)

Or pass the path directly:

.. code-block:: python

    conn = open_db("/path/to/ncaa.db")

Models
------
Model1  — KenPom-style multiplicative fixed-point iteration, bootstrap posterior
Model2  — Ridge regression with exact Gaussian posterior (recommended)
Model3  — Bilinear ALS extension adding matchup-specific interaction factors
"""
from models.base import BaseModel, KenPomSummary
from models.data import GameRow, load_season_games, open_db, parse_date
from models.model1 import Model1
from models.model2 import Model2
from models.model3 import Model3
from models.model4 import Model4
from models.eval import (
    EvalResult,
    temporal_split,
    evaluate_season,
    evaluate_all_seasons,
    print_report,
    win_probability,
    conformal_calibration_scores,
    recency_weights,
)

__all__ = [
    # ── Data types ──────────────────────────────────────────────────────────
    "GameRow",           # one team's offensive output in one game
    "KenPomSummary",     # (adj_o, adj_d, adj_pace, net_rtg) for one team
    "BaseModel",         # abstract base — use for type hints
    # ── Models ──────────────────────────────────────────────────────────────
    "Model1",            # fixed-point iteration, bootstrap posterior
    "Model2",            # ridge regression, exact Gaussian posterior
    "Model3",            # bilinear ALS (matchup factors)
    "Model4",            # weekly Kalman state-space, dynamic ratings
    # ── Evaluation ──────────────────────────────────────────────────────────
    "EvalResult",
    "temporal_split",
    "evaluate_season",
    "evaluate_all_seasons",
    "print_report",
    "win_probability",
    "conformal_calibration_scores",
    "recency_weights",
    # ── Data loading (optional — only needed for ncaa.db access) ────────────
    "load_season_games",
    "open_db",
    "parse_date",
]
