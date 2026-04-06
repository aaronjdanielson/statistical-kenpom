# Model 1: KenPom-Style Multiplicative Fixed-Point Rating System

**File:** `models/model1.py` · **Class:** `Model1`

---

## Purpose

Model 1 is a direct implementation of the KenPom opponent-adjustment algorithm. It is the conceptual baseline: a schedule-adjusted rating system that produces three scalar quantities per team without reference to any explicit statistical model. Its outputs are interpretable in absolute units (points per 100 possessions, possessions per game), anchored to the league average, and comparable across seasons.

The goal here is not to be the most powerful estimator. It is to be the faithful translation of the classical algorithm into code, with uncertainty attached via bootstrap rather than pretending the estimates are exact.

---

## What Gets Estimated

For each team `i`, three latent quantities:

- **O_i** — adjusted offensive efficiency: points per 100 possessions team `i` would score against an average defense
- **D_i** — adjusted defensive efficiency: points per 100 possessions team `i` would allow against an average offense
- **R_i** — adjusted tempo: possessions per game team `i` would play against an average opponent

All three are calibrated so that the unweighted league mean equals the actual league mean:

```
mean(O) = mean(D) = μ_off    (national pts/100 poss average)
mean(R) = μ_tempo             (national possessions/game average)
```

---

## The Algorithm

The estimates satisfy a system of self-consistency equations. Team `i`'s adjusted offense should equal the possession-weighted average of its raw efficiency, after dividing out the home-court factor and multiplying by the ratio `μ_off / D_j` (opponent quality normalization):

```
O_i = Σ_g  w_g · (e_off_{ig} / L(h_g)) · (μ_off / D_{j(g)})
      ────────────────────────────────────────────────────────
                        Σ_g  w_g
```

Defense and tempo are symmetric:

```
D_i = Σ_g  w_g · (e_off_{ig} / L(h_g)) · (μ_off / O_{j(g)})   [accumulated at opp index]
      ────────────────────────────────────────────────────────

R_i = Σ_g  w_g · (p_g / L_pace(h_g)) · (μ_tempo / R_{j(g)})
      ──────────────────────────────────────────────────────
                        Σ_g  w_g
```

Because `O` appears in the `D` equation and vice versa, these cannot be solved in closed form. The algorithm iterates:

1. **Initialize:** `O_i = D_i = μ_off`, `R_i = μ_tempo` for all teams.
2. **Update offense:** recompute all `O_i` using current `D`.
3. **Update defense:** recompute all `D_i` using current `O`.
4. **Update tempo:** recompute all `R_i` using current `R`.
5. **Renormalize** so league means remain anchored.
6. **Repeat** until `max(|ΔO|, |ΔD|, |ΔR|) < tol`.

In practice this converges in 30–80 iterations to tolerance `1e-6`.

### Location Adjustments

Home court advantage is applied multiplicatively. The values used are:

| Venue   | L_off  | L_def (= 1/L_off) |
|---------|--------|-------------------|
| Home    | 1.014  | 0.986             |
| Neutral | 1.000  | 1.000             |
| Away    | 0.986  | 1.014             |

These divide out the venue effect so that a team scoring well at home gets less credit than if the same performance happened on a neutral floor.

Pace adjustments are all 1.0 (venue has negligible effect on total possessions).

### Weights

Observations are weighted by possession count (`w_g = p_g`). This discounts fast games (fewer possessions = noisier efficiency estimate) relative to slower games with more evidence. It is the natural observation weight for efficiency rates.

---

## Implementation Details

**Theta layout** (length `3T`):

```
theta[:T]    = O_0, ..., O_{T-1}   (absolute pts/100)
theta[T:2T]  = D_0, ..., D_{T-1}  (absolute pts/100)
theta[2T:3T] = R_0, ..., R_{T-1}  (absolute poss/game)
```

This is passed directly to `_summary_from_theta`, making the KenPom mapping trivial: `AdjO = O_i`, `AdjD = D_i`, `AdjPace = R_i`. No transformation needed.

**Vectorized updates** use `numpy.add.at` to accumulate contributions per team index without a Python loop over teams. The per-row quantities (`e_off_raw`, `l_off`, `l_def`, `l_pace`, `w`) are all precomputed before the iteration loop.

**Bootstrap mask:** in each bootstrap resample, some teams may appear zero times (their games were not drawn). Rather than crashing or producing NaN, those teams retain their previous estimate for that iteration.

---

## Uncertainty: Parametric Bootstrap

Model 1 has no formal likelihood, so there are no standard errors or posterior distributions in the classical sense. The bootstrap is the right approach.

**Procedure:**
1. Collect all unique game IDs from the season.
2. Draw a bootstrap sample by resampling game IDs with replacement (game-level, so both rows of each game move together — this preserves the bipartite structure of the schedule graph).
3. Refit the fixed-point system on the resampled dataset.
4. Repeat `n_boot` times.

Each bootstrap draw produces one `theta` vector, which `sample_kenpom_summary` maps to `{team_id: KenPomSummary}` via `_summary_from_theta`. The collection of `n` draws is the approximate posterior over `(O, D, R)`.

**Why game-level resampling?** Resampling individual rows would break the score symmetry — team `i`'s offense and team `j`'s defense in the same game would be included independently, distorting both estimates. Resampling whole games keeps the data coherent.

---

## KenPom Summary Mapping

```
AdjO_i    = O_i          (already absolute pts/100, league-mean-anchored)
AdjD_i    = D_i          (already absolute pts/100, league-mean-anchored)
AdjPace_i = R_i          (already absolute poss/game, league-mean-anchored)
```

This is the trivial mapping. Model 1 estimates the KenPom summary directly, so `_summary_from_theta` is just an indexing operation.

---

## Strengths and Limitations

**Strengths:**
- Exact reproduction of the KenPom algorithmic idea.
- Outputs are in absolute, interpretable units with a clear meaning.
- Fast convergence; cheap to bootstrap.
- No distributional assumptions required.

**Limitations:**
- No explicit likelihood, so fit quality cannot be measured directly.
- Regularization is implicit (the iterative weighted averaging acts as a form of smoothing), not principled or tunable.
- Bootstrap uncertainty is a reasonable approximation but is not a true posterior — it reflects schedule variation, not model uncertainty.
- The multiplicative formulation means AdjO and AdjD do not decompose additively, which makes combining them with other covariates awkward.

For downstream use as features, Model 1's outputs are the most interpretable but the least statistically coherent. They serve well as a benchmark and as a sanity check against the other two models.

---

## Default Hyperparameters

| Parameter  | Default | Meaning                              |
|------------|---------|--------------------------------------|
| `max_iter` | 500     | Maximum fixed-point iterations       |
| `tol`      | 1e-6    | Convergence threshold (max |Δ|)      |
| `n_boot`   | 200     | Bootstrap draws for posterior        |
