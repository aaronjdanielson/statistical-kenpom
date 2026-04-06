# Model 2: Ridge Latent-Effects Rating Model

**File:** `models/model2.py` · **Class:** `Model2`

---

## Purpose

Model 2 is the first model in this hierarchy with a genuine statistical foundation. It is a penalized additive latent-effects model: each team has explicit offense and defense coefficients estimated by ridge regression, and the posterior over those coefficients is an exact Gaussian. This enables analytic uncertainty quantification — no simulation required for the main effects.

It is the recommended base model for downstream feature generation. The coefficients are interpretable, regularized against schedule imbalance, and carry well-defined standard errors.

---

## The Model

### Efficiency

For each team-game observation (team `i` scoring against team `j` in game `g`):

```
e_off_{ig} = μ + o_i - d_j + η · h_g + ε_{ig}

ε_{ig} ~ N(0, σ² / p_g)
```

where:
- `e_off_{ig} = (pts_i / poss_g) × 100` — offensive efficiency in points per 100 possessions
- `μ` — league baseline efficiency
- `o_i` — offensive lift of team `i` above average (positive = better offense)
- `d_j` — defensive suppression by team `j` (positive = better defense, reduces opponent efficiency)
- `η` — home-court effect in pts/100
- `h_g ∈ {+1, 0, −1}` — venue for team `i`: home / neutral / away
- `σ²/p_g` — heteroskedastic noise: fewer possessions → noisier efficiency estimate

This says: your offensive output is determined by your own offensive quality, your opponent's defensive quality, and the venue, plus noise.

### Pace

For each unique game `g`:

```
log(p_g) = γ₀ + r_i + r_j + γ_h · h_g + ξ_g

ξ_g ~ N(0, σ_p²)
```

where:
- `p_g` — estimated possessions (average of both teams' possession estimates)
- `γ₀` — log baseline tempo
- `r_i, r_j` — pace tendency of each team (both contribute additively in log space)
- `γ_h` — venue effect on pace

The log link ensures possessions are always positive and that team pace effects combine multiplicatively in the original scale. A team with `r_i > 0` tends to play at above-average pace regardless of opponent.

---

## Estimation

Both models are ridge regression problems:

```
θ̂ = argmin { (y − Xθ)ᵀ W (y − Xθ) + θᵀ Λ θ }
```

The solution is:

```
θ̂ = (XᵀWX + Λ)⁻¹ XᵀWy
```

### Design Matrix — Efficiency

Each row of `X` corresponds to one team's offensive half of a game. Columns:

| Column | Width | Content |
|--------|-------|---------|
| 0 | 1 | Intercept (μ) |
| 1..T | T | +1 in column `i` for the scoring team |
| T+1..2T | T | −1 in column `j` for the defending team |
| 2T+1 | 1 | h_g (venue indicator) |

Observation weights: `w_k = p_g` (possession count).

Penalty: `λ_team` on all offense and defense columns. The intercept and home-court columns receive a near-zero jitter (`1e-4`) rather than zero, because the intercept column is structurally collinear with the sum of all offense columns — a zero penalty would produce a singular normal matrix.

### Design Matrix — Pace

Each row corresponds to one unique game. Both teams contribute `+1` in their own `r` column:

| Column | Width | Content |
|--------|-------|---------|
| 0 | 1 | Intercept (γ₀) |
| 1..T | T | +1 for each team playing in the game |
| T+1 | 1 | h_g (venue indicator) |

Response: `log(p_g)`. Penalty: `λ_pace` on `r` columns; near-zero jitter on intercept and venue.

---

## Posterior

The Bayesian interpretation of ridge regression under Gaussian priors is:

```
θ | y ~ N(θ̂,  Σ_post)

Σ_post ≈ σ̂² · (XᵀWX + Λ)⁻¹
```

This is the Laplace approximation, which is exact here because the model is linear in `θ` and the prior is Gaussian.

The efficiency and pace posteriors are **independent** (separate design matrices, separate noise terms), so samples are drawn independently from each and concatenated.

### Sampling

To draw `n` samples:

1. Compute `L = cholesky(Σ_post)` once.
2. Draw `Z ~ N(0, I)` of shape `(n, dim)`.
3. Return `θ̂ + Z @ Lᵀ`.

This is fast and exact. No MCMC, no approximation beyond the Gaussian posterior itself.

---

## Parameter Vector Layout

Efficiency block (length `2T + 2`):

```
[0]         = μ
[1 .. T]    = o_0, ..., o_{T-1}
[T+1 .. 2T] = d_0, ..., d_{T-1}
[2T+1]      = η
```

Pace block (length `T + 2`), concatenated after efficiency:

```
[0]         = γ₀
[1 .. T]    = r_0, ..., r_{T-1}
[T+1]       = γ_h
```

Full `theta_hat_` length: `3T + 4`.

---

## KenPom Summary Mapping

The three KenPom numbers are derived from the coefficient vector as follows:

```
AdjO_i    = μ + o_i             (points per 100 poss team i scores vs avg defense, d=0)
AdjD_i    = μ − d_i             (points per 100 poss team i allows vs avg offense, o=0)
AdjPace_i = exp(γ₀ + r_i)      (possessions per game team i plays vs avg opponent, r_opp=0)
```

**Why `μ − d_i` for defense?** The convention is that lower AdjD is better (fewer points allowed). The coefficient `d_j` appears with a negative sign in the model (`μ + o_i − d_j`), so a larger `d_j` means team `j` is better at suppressing offense. Writing AdjD as `μ − d_i` converts this to the KenPom convention where AdjD is "points per 100 allowed," and a good defense has a low value.

**Why `exp(γ₀ + r_i)` for pace?** The pace model is log-linear, so a team's expected possessions against an average opponent (where `r_opp = 0`) is `exp(γ₀ + r_i)`. This keeps AdjPace strictly positive.

These mappings are identical to Model 3. Any downstream code consuming `KenPomSummary` objects works identically across both models.

---

## Uncertainty Quantities

For any linear combination of coefficients `c = aᵀθ` (e.g., a single coefficient `o_i`, a contrast `o_i − d_j`, or a matchup prediction `μ + o_i − d_j`):

```
Var(c) = aᵀ Σ_post a

95% CI: ĉ ± 1.96 · sqrt(Var(c))
```

This enables exact standard errors on:
- Individual team offense/defense ratings
- Head-to-head efficiency contrasts (`o_i − d_j`)
- Net rating `o_i − d_i`
- Any linear combination of team effects

Pace uncertainty is computed separately from the pace posterior `Σ_pace`.

---

## Predictive Distribution

For a future game between team `i` (home) and team `j`, the predictive distribution over offensive efficiency is:

```
ẽ_ij | data ~ N(μ + o_i − d_j + η·h,  xᵀ Σ_post x + σ²_new)
```

The predictive variance has two additive components:
- **Parameter uncertainty:** `xᵀ Σ_post x` — how uncertain are we about the team effects?
- **Irreducible game noise:** `σ²_new ≈ σ̂²` — even if we knew the parameters exactly, games are noisy.

To simulate a game outcome:
1. Draw `p̃ ~ lognormal(γ₀ + r_i + r_j + γ_h·h, σ_p²)` (or from the pace posterior predictive)
2. Draw `ẽ_ij, ẽ_ji` from the efficiency posteriors
3. Compute `ỹ_ij = p̃ · ẽ_ij / 100`, `ỹ_ji = p̃ · ẽ_ji / 100`

This gives a joint predictive distribution over score, margin, total, and win probability.

---

## Strengths and Limitations

**Strengths:**
- Exact Gaussian posterior — no MCMC, no approximation errors.
- Ridge penalty provides principled regularization tunable via `λ_team` and `λ_pace`.
- Standard errors available analytically for any linear combination.
- Additive structure makes coefficients directly reusable as features.
- Possessions-weighted estimation is the statistically correct treatment of efficiency rates.
- Pace and efficiency are estimated independently, so their posteriors are cleanly separable.

**Limitations:**
- Assumes the linear additive decomposition is correct — no interaction effects between specific offense-defense pairs.
- Ridge penalty treats all teams symmetrically; does not incorporate prior information about team quality (e.g., preseason expectations, conference strength).
- The Laplace posterior is exact for the efficiency block but assumes the Gaussian noise model is correct; if efficiency residuals are heavy-tailed, the posterior may understate uncertainty.
- `λ_team` is a single scalar penalty — both strong and weak teams are shrunk at the same rate.

---

## Default Hyperparameters

| Parameter     | Default | Meaning                                      |
|---------------|---------|----------------------------------------------|
| `lambda_team` | 100.0   | Ridge penalty on offense/defense effects     |
| `lambda_pace` | 50.0    | Ridge penalty on pace effects                |

These values produce ratings with reasonable shrinkage on a full season of D1 data (~350 teams, ~5,500 games). They have not been cross-validated and could be tuned if predictive accuracy on held-out games is the objective.
