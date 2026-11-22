# Project guide history (CLAUDE.md until 2026-11-21)

> Moved here unchanged when CLAUDE.md was cut to the spoke standard. Dated results and
> decisions; superseded where [rolling_ratings.md](rolling_ratings.md) says so (the posterior-sd
> scale, the shrinkage, the rolling model). The "Model summary" table measures team-offence
> efficiency on 2025-26 only, not the next-week margins used since 2026-11-20.


## Environment

- Python: `/Users/aarondanielson/opt/anaconda3/bin/python` (3.9.12)
- pytest: `pytest` (7.1.1)
- DB: **portal_pivot_db's `db/ncaa.db`** (the official database; `models.data` resolves
  `NCAA_DB_PATH` → `$PORTAL_DB_DIR/ncaa.db` (default `~/Dropbox/portal_pivot_db/db`) → this repo's
  own `ncaa.db`). Ratings built from it equal the ones built from this repo's `ncaa.db` exactly
  for 2002–2026 (8,379 team-seasons, checked 2026-11-18) and it carries the season in progress.
  This repo's `ncaa.db` / `intl.db` / `national.db` are frozen copies (ncaa.db since 2026-04);
  every table in them exists in the hub with at least as many rows.
- Package: `ncaa-ratings` (editable install via `pyproject.toml`)

## Place in the portal-pivot ecosystem (user's plan, 2026-11-18)

kenpom is a data-science model like ncaa_rapm: it reads its inputs from portal_pivot_db,
portal_pivot_db's daily chain calls its entry point to update **rolling team ratings for
every league** the hub holds (standard `team_boxscore` + `schedule` views in ncaa.db,
intl.db, national.db, nba.db), and the hub stores the results with provenance. Then this
repo's own databases and scrapers are retired (the hub scrapes). Every repo that still
reads them is listed by `python ~/Dropbox/portal_pivot_db/scripts/check_ecosystem.py --refs kenpom`
(register: `~/Dropbox/portal_pivot_db/ecosystem.toml`) — move those before deleting anything.

## Rolling ratings for every league (built 2026-11-18)

- `models/hub.py` — every league's games from the hub's standard `team_boxscore`
  (NCAA-D1, `INTL-*`, `NT-*`, NBA, GLEAGUE), as GameRows identical to `load_season_games`
  / `load_season_league_games` (checked: NCAA 2002–2026, 40 international leagues).
  NCAA season in progress: + `team_boxscore_espn` rows where RealGM lacks the game.
  Exhibitions (NBA / G League preseason, All-Star) are left out here only; the hub keeps them.
- `scripts/rolling_ratings.py compute [--all | --leagues … --seasons …]` — weekly Model 2
  snapshots per league-season → `output/rolling/<league>/<season>.parquet + .json`; skips a
  league-season whose key (settings + input hash) is unchanged. Last snapshot of a finished
  NCAA season = published `team_ratings.db` exactly.
- `update_ratings.sh` — the entry point the hub's nightly chain runs (`ratings.kenpom_update`,
  `PORTAL_DB_DIR` set); the hub imports builds into `data/models/kenpom_rolling`.
- Tests: `tests/unit/test_hub_loader.py`.
- `scripts/build_intl_team_ratings.py` (international season ratings) reads the hub too since
  2026-11-20 (was this repo's frozen `intl.db`): same method and columns; a full run rebuilds
  the file from empty. `scripts/rolling_intl_team_ratings.py` is the last script that still
  reads this repo's `intl.db`; `rolling_ratings.py` supersedes it and the hub does not import
  its output.
- Where builds are written: `models.hub.out_dir()` — `$PORTAL_HANDOFF_DIR/kenpom/` when the hub
  sets `PORTAL_HANDOFF_DIR` (a spoke stores no data), else `output/` until the hub's imports
  read the handoff directory.
- Open, found 2026-11-20 (not yet changed; they alter published values): `adj_o_sd`,
  `adj_d_sd`, `adj_net_sd` are too small by sqrt(possessions), about 8x — `Sigma =
  _sigma2_eff * inv(A)` uses the per-game residual variance where the per-possession-scale
  variance mean(poss * resid^2) belongs; Model 4 and the Fix A calibration result share the
  scale. One-step-ahead tests put the best `lambda_team` near 100-150 for NCAA and national
  teams, 150-250 international, 500-1000 NBA, ~1500 G League (500 is used everywhere).

## Downstream consumers

`intl.db` and `national.db` (populated by `national_scraper/`) are consumed
by `~/Dropbox/ncaa_roster_prediction/analysis/backfill_intl_summaries.py`
and `backfill_national_gamelogs.py`, which project them into that project's
`roster.db.intl_*` tables for model input. Full cross-repo data map:
[~/Dropbox/ncaa_roster_prediction/docs/DATA_INVENTORY.md](../ncaa_roster_prediction/docs/DATA_INVENTORY.md).

Run tests:
```
pytest tests/
```

## Codebase

```
models/
  data.py       GameRow, load_season_games, open_db, parse_date
  base.py       BaseModel ABC, KenPomSummary
  model1.py     KenPom fixed-point iteration + parametric bootstrap
  model2.py             Ridge (RAPM-style), exact Gaussian posterior; sample_weight for recency
  model2_continuity.py  Model2ContinuityPrior — shift_only=True (Fix B) is production-ready
  model3.py             Bilinear ALS, two-stage (excluded from predictive eval — overfits OOS)
  model4.py             Weekly Kalman state-space, RTS smoother, dynamic ratings
  priors.py             load_returning_minutes, extract_prev_effects, build_continuity_prior
  eval.py               temporal_split, evaluate_season, recency_weights, win_probability,
                        conformal_calibration_scores
  __init__.py           public API for all 4 models + eval functions

scripts/
  recency_half_life_search.py     Phase 1: grid search over recency half-lives
  rolling_one_step_ahead.py       M1 vs M2 rolling OSA eval (produces curves + scatter)
  model_osa_comparison.py         Three-way OSA: M2 static / M2 hl=60 / M4 Kalman
  osa_four_model_comparison.py    Four-way OSA: M1 / M2 static / M2 hl=60 / M4 Kalman
  win_prediction_comparison.py    Four-way win/loss accuracy, Brier, log-loss
  verify_graph_and_rts.py         Two verification tests (graph connectivity + RTS≡M2)
  scatter_model_comparison.py     Scatter plots: M1 vs M2 ratings
  uncertainty_viz.py              Team uncertainty fan chart, calibration curve
  rolling_net_rtg_2026.py         Rolling net rating trajectories
  eval_continuity_prior.py        Fix B experiment: mean-shift prior, stratified eval

docs/
  models_overview.md              Model hierarchy and design principles
  model1_kenpom_fixed_point.md
  model2_ridge_latent_effects.md
  model3_bilinear_interaction.md
  model4_kalman_state_space.md    Equivalence theorem, identification analysis, results
  continuity_prior.md             Cross-season prior: formulation, Fix A/B, results

tests/
  unit/test_models.py             36 tests for M1, M2, M3
  unit/test_model4.py             27 tests for M4 (all passing)
  unit/test_continuity_prior.py   21 tests for Model2ContinuityPrior + build_continuity_prior
```

## Model summary

| Model | RMSE | Bias | Win acc | Notes |
|---|---|---|---|---|
| M1 fixed-point | 14.53 | −1.20 | 68.1% | KenPom clone, heuristic |
| M2 static | 13.84 | −0.91 | 68.7% | Exact ridge posterior |
| M2 hl=60 | **13.87** | −0.70 | **69.0%** | **Best overall** |
| M4 Kalman τ=0.25 | 14.12 | **−0.35** | 68.5% | Best bias; correct UQ |

RMSE and win accuracy evaluated rolling one-step-ahead, 2025-26 season, 20 windows of 7 days.

## Production recommendation

- **Live prediction (weeks 4+):** M2 hl=60 — best RMSE and win accuracy
- **Early season (weeks 1–3):** M2_shift_hl — continuity prior + hl=60; −0.28 RMSE vs baseline
- **Tournament:** M2 static — no drift needed, smallest variance
- **Season trajectory / retrospective:** M4 RTS smoother (`model4.rts_smoother()`)
- **Uncertainty intervals:** M4 Kalman — correctly calibrated (M2 intervals are too tight)

## Key results

### Structural equivalence (Model 2 = RTS smoother, τ→0)

The ridge regression posterior (Model 2) equals the RTS smoother applied to the
Kalman state-space model in the limit τ→0, when the prior is parameterized as
N(0, σ²/λ · I) and observation variance as σ²/poss_k.

Empirically verified on 2025-26 season: max |ΔAdjO| = 0.000 pts/100 (floating-point
agreement across 365 teams). See `scripts/verify_graph_and_rts.py`.

**Critical parameterization:** C₀ = (σ²_eff / λ) · I, not I/λ. The latter is 160×
too tight and causes the filter to ignore observations, driving τ artificially large.

### Graph connectivity (identification)

Early-season ratings are confounded by schedule. The game graph connects slowly:
- Nov 14: 4 components, only 4% of team pairs linked within 3 hops
- Nov 28: first fully connected graph (neutral-site tournaments bridge conferences)
- January+: diameter = 4, >98% of pairs within 3 hops — identification complete

### τ identification

CV-optimal τ = 0.25/week (post prior-fix). MLE-optimal τ is larger (~3–6) because
the marginal likelihood conflates true drift with residual game noise. Use CV.

## Cross-season continuity prior (implemented, Fix B evaluated)

### Core result

The mean shift alone (Fix B: P=λI unchanged, only add λm to the right-hand side)
delivers −0.23 to −0.28 RMSE in early-season windows vs zero-centered ridge.

The improvement comes entirely from shifting the prior mean toward last season's
ratings. Variance modulation (Fix A) is a second-order question; the mean is the
mechanism.

```
θ̂ = (XᵀWX + λI)⁻¹ (XᵀWy + λm)
m_i = r_i · θ̂_{i, s−1}     (returning minutes fraction × prev-season effect)
```

Full details and stratified results: `docs/continuity_prior.md`

### Decomposition (empirically confirmed)

- **Continuity prior** → cross-season level alignment (RMSE/scale)
- **Half-life weighting** → within-season drift correction (bias)
- These are orthogonal mechanisms and stack (M2_shift_hl: −0.277 RMSE vs baseline)

### Production recommendation update

- **Early season (weeks 1–3):** M2_shift_hl — shift-only continuity + hl=60 recency
  - Improvement now exceeds M4 Kalman with no added complexity
  - M4 Kalman still preferred when calibrated UQ is needed

### Fix A result (closed)

Variance modulation improves RMSE by −0.246 on top of Fix B, but destroys posterior
calibration: z_std ≈ 25 (should be 1), 90% coverage ≈ 6% (should be 90%).
The mechanism: tighter τ compresses the posterior covariance. RMSE gain and
calibration failure are inseparable. Fix B is the correct production model.

Full summary vs M2_static early-season RMSE:
  M2_static: 14.753  |  M2_shift_hl (Fix B): 14.476 (−0.277, calibration preserved)
  Fix A best: 14.230  (−0.523, calibration broken)

## Known issues / design decisions

- **Model 3** excluded from OSA comparison — overfits OOS (non-convex ALS, no temporal eval)
- **M4 tournament degradation** — at τ=0.25 the model still over-diffuses state in March
  when the bracket field shrinks; M2 static is better for single-elimination prediction
- **σ²_eff** is taken from a Model 2 pre-fit and held fixed during the Kalman pass;
  joint optimization of σ with the Kalman filter is not implemented
