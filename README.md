# Probabilistic College Basketball Ratings

A Bayesian framework for NCAA basketball team ratings that produces calibrated posterior distributions over team quality — not just point estimates.

> **Key finding:** A ridge prior over team effects is worth approximately three weeks of additional data in early-season forecast accuracy. The resulting posterior predictive intervals are empirically well-calibrated: a 90% PI contains ~90% of actual outcomes.

## Preview

![Rolling OSA RMSE 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/rolling_osa_rmse_2025_26.png)

![Model 1 vs Model 2 scatter](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/scatter_model1_vs_model2.png)

## What this is

Most college basketball rating systems (KenPom, BartTorvik, BPI) assign each team a number. This system assigns each team a **distribution**. The difference matters in two places:

- **Early season** — 5 games per team means high uncertainty. A posterior is honest about that; a point estimate is not.
- **Matchup prediction** — knowing a team scores 112 pts/100 is useful. Knowing they score 112 ± 3 against an opponent allowing 108 ± 4 is a lot more useful.

## Models

| Model | Method | Posterior | Season RMSE |
|-------|--------|-----------|-------------|
| **Model 1** | KenPom-style fixed-point iteration | Parametric bootstrap | 15.10 |
| **Model 2** | Ridge regression (RAPM-style) | Exact Gaussian (Cholesky) | **14.14** |

Evaluation is one-step-ahead: for each week, the model is fit on all prior games only, then evaluated on that week's games. No future data leaks into any prediction.

Model 3 (bilinear interaction term) is implemented and validated for in-sample KenPom rankings (ρ > 0.99 vs ground truth), but excluded from predictive evaluation — the interaction factors overfit a temporal holdout, collapsing to Model 2 under the regularization needed to generalize.

## Repository layout

```
models/          Core model package
  data.py          GameRow, load_season_games, open_db
  base.py          BaseModel ABC, KenPomSummary, predict_interval
  model1.py        Fixed-point iteration + bootstrap posterior
  model2.py        Ridge regression + exact Gaussian posterior
  model3.py        Bilinear interaction (ALS) + hybrid posterior
  eval.py          temporal_split, evaluate_season, EvalResult

scripts/         Visualization and evaluation scripts
  scatter_model_comparison.py      80/20 train/test scatter (3 seasons)
  rolling_one_step_ahead.py        OSA RMSE + bias curves
  rolling_net_rtg_2026.py          Weekly rolling net rating fan
  uncertainty_viz.py               Calibration curve, team fans, pre-game dists

notebooks/
  probabilistic_ratings.ipynb      Full narrative — models → validation → figures

paper/
  probabilistic_ratings.tex        LaTeX manuscript (compiles to PDF)

tests/
  unit/            25 tests — fit, posterior contract, synthetic data
  integration/     119 tests — KenPom validation + predictive eval (2:40 total)

docs/
  models_overview.md
  model1_kenpom_fixed_point.md
  model2_ridge_latent_effects.md
  model3_bilinear_interaction.md

ncaa_scraper/    Data pipeline (RealGM → ncaa.db, kept separate from models/)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run the notebook end-to-end:
```bash
jupyter notebook notebooks/probabilistic_ratings.ipynb
```

## Reproduce figures

```bash
python scripts/scatter_model_comparison.py      # 80/20 scatter, 3 seasons
python scripts/rolling_one_step_ahead.py        # OSA RMSE + bias curves
python scripts/rolling_net_rtg_2026.py          # rolling net rating fan
python scripts/uncertainty_viz.py               # calibration + distributions
```

## Run tests

```bash
pytest tests/                          # unit tests only (~3s)
pytest tests/ -m integration           # full suite including KenPom validation (~2:40)
```

144 tests total, all passing.

## Visualizations

### One-step-ahead forecast quality

![Rolling OSA RMSE 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/rolling_osa_rmse_2025_26.png)

*Model 1 spikes to RMSE 22+ in week 1 (underdetermined system). Model 2 opens at 14.6 and stays there. The gap closes by December as Model 1 accumulates data.*

![Rolling OSA scatter 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/rolling_osa_scatter_2025_26.png)

### Calibration

![Calibration curve 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/calibration_curve_2025_26.png)

*Nearly perfect calibration across all coverage levels. Slight underconfidence (curve above diagonal) is the safe failure mode.*

### Rolling ratings with uncertainty

![Rolling net rating 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/rolling_net_rtg_2025_26.png)

*Posterior mean ± 1σ / ±2σ bands. Uncertainty collapses from ~2 pts/100 in November to ~0.3 pts/100 by January.*

### Pre-game predictive distributions

![Pregame distributions 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/pregame_distributions_2025_26.png)

*Full posterior predictive density for each team in a matchup. Overlap = competitiveness. Dotted lines = actual outcomes.*

### Model comparison scatter

![Model 1 vs Model 2 scatter](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/scatter_model1_vs_model2.png)

### Team uncertainty fans

![Team uncertainty fan 2025-26](https://raw.githubusercontent.com/aaronjdanielson/statistical-kenpom/main/scripts/team_uncertainty_fan_2025_26.png)

## Notes

- `ncaa.db` and caches are in `.gitignore`. The DB lives at `~/Dropbox/kenpom/ncaa.db` by default; override via `KENPOM_DB` env var or `open_db(path=...)`.
- The `models/` package has zero imports from `ncaa_scraper/` — the separation is enforced by tests.
- Season convention: `Year=2026` in the DB means the 2025-26 season.
