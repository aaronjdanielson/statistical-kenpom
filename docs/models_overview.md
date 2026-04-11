# Rating Models: Overview and Design Principles

This directory documents the three team rating models implemented in `models/`.
Each model produces the same three-number KenPom-style summary (AdjO, AdjD, AdjPace)
and the same posterior sampling interface, but differs in statistical sophistication,
inferential machinery, and the richness of the features it exposes.

---

## The Three-Number Contract

Every model implements the `BaseModel` interface defined in `models/base.py`.
The hard contract is:

```python
sample_kenpom_summary(n, rng) == [_summary_from_theta(θ) for θ in sample_posterior(n, rng)]
```

That is, the KenPom summaries are always derived by mapping posterior draws through
the same `_summary_from_theta` function. This means:

1. Uncertainty in the summaries propagates correctly from the model's posterior.
2. All three models can be compared on the same axis (AdjO, AdjD, AdjPace).
3. Downstream code consuming `KenPomSummary` objects works identically across models.

The `KenPomSummary` dataclass:

```python
@dataclass(frozen=True)
class KenPomSummary:
    adj_o: float       # pts per 100 poss vs avg defense   (higher = better)
    adj_d: float       # pts per 100 poss allowed vs avg offense (lower = better)
    adj_pace: float    # possessions per game vs avg opponent
```

---

## Model Hierarchy

| | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---|---|---|---|
| **Estimator** | Fixed-point iteration | Ridge regression | Ridge + ALS bilinear | Kalman filter + RTS smoother |
| **Statistical foundation** | Heuristic | Penalized likelihood | Penalized likelihood | State-space (linear-Gaussian) |
| **Posterior** | Parametric bootstrap | Exact Gaussian | Laplace (main) + Bootstrap (bilinear) | Kalman filtered / RTS smoothed |
| **Interaction term** | None | None | Rank-k inner product | None |
| **Temporal dynamics** | None | None (optional recency weights) | None | Random-walk drift (τ per week) |
| **KenPom mapping** | Trivial (direct) | μ ± coefficients | Same as Model 2 | Same as Model 2 |
| **Main strength** | Interpretable benchmark | Clean inference, best RMSE | Matchup-specific features | Correct uncertainty; bias reduction; season trajectory |
| **Main limitation** | No likelihood | No drift | Non-convex, more expensive | Causal filter loses to M2 on RMSE after week 3 |

---

## Model Comparison

### Model 1 is the faithful KenPom clone

It implements the same multiplicative fixed-point equations as the original system.
The outputs are adjusted offense, defense, and tempo in the same units as the KenPom website.
Bootstrap resampling over whole games gives approximate uncertainty intervals.

Use Model 1 when you want a direct comparison to published KenPom numbers, or as a
baseline to validate that the data pipeline is working correctly.

### Model 2 is the recommended production model

The additive structure (`e_off = μ + o_i − d_j + η·h`) is a proper statistical model.
Ridge regression gives principled regularization. The exact Gaussian posterior means
standard errors and prediction intervals are computed analytically, not by simulation.

The KenPom summary mapping from Model 2's coefficients is:
- `AdjO_i = μ + o_i` — points per 100 team `i` scores against average defense
- `AdjD_i = μ − d_i` — points per 100 team `i` allows against average offense
- `AdjPace_i = exp(γ₀ + r_i)` — tempo against average opponent

These numbers are interpretable in the same absolute units as KenPom.

Use Model 2 for feature generation, matchup prediction, and any downstream task that
requires calibrated uncertainty on the coefficients.

### Model 3 is the matchup-aware extension

It adds a rank-k bilinear term `aᵢᵀbⱼ` to capture style interactions that scalar
ratings miss. The main effects are estimated by the same ridge solve as Model 2;
the bilinear factors are estimated by alternating least squares.

The centering constraint (`Σ aᵢ = Σ bⱼ = 0`) ensures the interaction has zero mean,
keeping the KenPom summary identical in definition to Model 2.

Use Model 3 when you want the matchup compatibility score `aᵢᵀbⱼ` as a downstream
feature, or when you suspect that the residuals from Model 2 have structured matchup
patterns.

---

## AdjO and AdjD Convention

Following KenPom:
- **Lower AdjD is better** (fewer points allowed per 100 possessions).
- **Higher AdjO is better** (more points scored per 100 possessions).
- **Net rating** = AdjO − AdjD (higher is better).

In the additive model, the defense coefficient `d_j` appears with a *negative* sign
(`μ + o_i − d_j`), so a **positive** `d_j` means team `j` suppresses opponent offense.
The AdjD summary flips this: `AdjD_j = μ − d_j`, so a high `d_j` (good defense) maps
to a low AdjD (as on KenPom).

---

## Data Flow

```
ncaa.db (boxscores + schedules)
        │
        ▼
models/data.py :: load_season_games(conn, season)
        │  Returns list[GameRow]
        │  GameRow = (game_id, season, team_id, opp_id, pts, poss, h)
        ▼
Model{1,2,3}.fit(season, conn)
        │  Estimates theta_hat_ and posterior covariance
        ▼
model.point_summary()                    → dict[team_id, KenPomSummary]
model.sample_kenpom_summary(n, rng)      → list[dict[team_id, KenPomSummary]]
model.sample_posterior(n, rng)           → list[np.ndarray]   (raw theta draws)
```

---

## File Index

| File | Purpose |
|------|---------|
| [model1_kenpom_fixed_point.md](model1_kenpom_fixed_point.md) | Model 1: multiplicative fixed-point algorithm, bootstrap uncertainty |
| [model2_ridge_latent_effects.md](model2_ridge_latent_effects.md) | Model 2: ridge regression, exact Gaussian posterior |
| [model3_bilinear_interaction.md](model3_bilinear_interaction.md) | Model 3: ALS bilinear, Laplace + bootstrap hybrid posterior |
| [model4_kalman_state_space.md](model4_kalman_state_space.md) | Model 4: Kalman state-space, equivalence theorem, identification analysis |
| [continuity_prior.md](continuity_prior.md) | Cross-season continuity prior: returning minutes, mean shift, Fix A/B experiment results |
