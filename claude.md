# KenPom NCAA Ratings — Project Guide

## Environment

- Python: `/Users/aarondanielson/opt/anaconda3/bin/python` (3.9.12)
- pytest: `pytest` (7.1.1)
- DB: `~/Dropbox/kenpom/ncaa.db` — seasons 2002–2026, 6305 games in 2026 season
- Package: `ncaa-ratings` (editable install via `pyproject.toml`)

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
