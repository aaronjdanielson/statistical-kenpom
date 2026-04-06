# Model 3: Additive + Bilinear (Hoff-Style) Interaction Model

**File:** `models/model3.py` · **Class:** `Model3`

---

## Purpose

Model 3 extends Model 2 with a structured interaction term that captures matchup-specific effects. The core idea is that two teams can be unusually well or poorly matched beyond what their scalar offense and defense ratings would predict. A team with an elite perimeter defense might suppress high-volume three-point offenses more than average, and vice versa. The bilinear term gives the model a mechanism to represent this.

The interaction is represented as a low-rank inner product `aᵢ · bⱼ`, where `aᵢ ∈ ℝᵏ` is an offensive style vector for team `i` and `bⱼ ∈ ℝᵏ` is a defensive style vector for team `j`. Their inner product is a scalar compatibility score for the specific offense-defense matchup.

This follows the approach of Hoff (2005) for relational data, applied to basketball game efficiency.

---

## The Model

### Efficiency

```
e_off_{ig} = μ + o_i − d_j + aᵢᵀbⱼ + η · h_g + ε_{ig}

ε_{ig} ~ N(0, σ² / p_g)
```

The `aᵢᵀbⱼ` term adds a matchup-specific adjustment on top of the main effects. If `aᵢᵀbⱼ > 0`, offense `i` tends to outperform its main-effect prediction against defense `j`. If `< 0`, the opposite.

### Pace

Identical to Model 2:

```
log(p_g) = γ₀ + r_i + r_j + γ_h · h_g + ξ_g
```

The bilinear interaction is only applied to the offense-defense relationship, not to pace. Pace is a property of how both teams play, not a matchup-specific interaction in the same sense.

---

## Centering Constraint

The bilinear term is centered by enforcing:

```
Σᵢ aᵢ = 0   (element-wise)
Σⱼ bⱼ = 0   (element-wise)
```

These constraints are applied at each ALS iteration step: after updating `a` or `b`, subtract the column mean.

**Why centering matters:** Without it, the bilinear term has a nonzero mean across teams, which would partially absorb the intercept `μ`. The centering constraint ensures that `aᵢᵀbⱼ` has zero mean when `i` and `j` are drawn uniformly, so the main effects `(μ, o, d)` retain their interpretation as league-average-relative quantities.

**Consequence:** The KenPom summary mapping is identical to Model 2:

```
AdjO_i    = μ + o_i
AdjD_i    = μ − d_i
AdjPace_i = exp(γ₀ + r_i)
```

The bilinear term is excluded from the summary. This means that `_summary_from_theta` ignores the `a, b` portion of `theta_hat_` entirely. The three-number summary is invariant to the bilinear factors — it summarizes the team's quality against an average opponent, and the centering constraint makes this well-defined.

---

## Estimation: Alternating Least Squares (ALS)

The objective is non-convex in `(o, d, a, b)` jointly because of the product `aᵢᵀbⱼ`. ALS decomposes it into a sequence of convex subproblems that each have closed-form ridge solutions.

**Each iteration:**

1. **Fix `a`, `b` → solve for main effects.**
   Subtract the current interaction from the response: `y_adj = e_off − aᵢᵀbⱼ`.
   Solve the same ridge system as Model 2 to get `(μ, o, d, η)`.

2. **Compute main-effect residuals.**
   `resid = e_off − (μ + oᵢ − dⱼ + η·h)`.
   This is what the bilinear term needs to explain.

3. **Fix `b`, update each `aᵢ` independently.**
   For team `i`'s offense, the problem is:
   ```
   aᵢ = argmin Σ_{g: team=i} w_g (resid_g − aᵢᵀb_{j(g)})² + λ_ab ‖aᵢ‖²
   ```
   This is a weighted ridge problem with solution:
   ```
   aᵢ = (BᵀWB + λ_ab I)⁻¹ BᵀW resid
   ```
   where `B` is the matrix of `b_{j(g)}` vectors for the games team `i` played.
   Solved independently per team, `O(T · k³)` total.
   Then center: `a ← a − mean(a)`.

4. **Fix `a_new`, update each `bⱼ` independently.**
   Symmetric to step 3, accumulating residuals against the defensive side.
   Then center: `b ← b − mean(b)`.

5. **Check convergence:** `max(|Δθ_main|, |Δa|, |Δb|) < tol`. If not, repeat.

ALS is not guaranteed to find the global optimum (the problem is non-convex), but it reliably finds a good local optimum in practice, especially when `λ_ab` is large enough to prevent degenerate solutions.

---

## Identifiability

The bilinear term `aᵢᵀbⱼ` is not uniquely identified. If `R` is any orthogonal matrix, then:

```
aᵢᵀbⱼ = (Raᵢ)ᵀ(Rbⱼ)
```

The factor vectors can be rotated freely without changing the inner products or the model fit. This means that the individual entries of `aᵢ` and `bⱼ` are not directly interpretable — only the interaction scores `aᵢᵀbⱼ` are.

For uncertainty quantification and feature extraction:
- Report and use `aᵢᵀbⱼ` (the matchup compatibility score), not the raw factor vectors.
- Standard errors on `aᵢᵀbⱼ` are meaningful; standard errors on individual coordinates of `aᵢ` are not, unless a canonical orientation is imposed.
- The norms `‖aᵢ‖` and `‖bⱼ‖` have some meaning (offensive/defensive "style complexity") but must be interpreted with caution.

---

## Parameter Vector Layout

Main block (length `2T + 2`, same as Model 2):
```
[0]         = μ
[1..T]      = o_0, ..., o_{T-1}
[T+1..2T]   = d_0, ..., d_{T-1}
[2T+1]      = η
```

Pace block (length `T + 2`, same as Model 2):
```
[0]         = γ₀
[1..T]      = r_0, ..., r_{T-1}
[T+1]       = γ_h
```

Bilinear factors (length `2kT`):
```
[0..kT]     = a flattened (T×k, row-major)
[kT..2kT]   = b flattened (T×k, row-major)
```

Full `theta_hat_` length: `3T + 4 + 2kT`.

---

## Uncertainty: Hybrid Posterior

The posterior is hybrid because the main effects and the bilinear factors require different approximation strategies.

### Main Effects and Pace — Laplace (Exact Gaussian)

After ALS convergence, the main-effect Hessian at the MAP estimate is the same ridge normal matrix as in Model 2:

```
H = XᵀWX + Λ
Σ_main ≈ σ̂² · H⁻¹
```

This is the Laplace approximation to the posterior over `(μ, o, d, η)` conditional on the converged `(a, b)`. Draws are taken by Cholesky sampling, exactly as in Model 2.

### Bilinear Factors — Parametric Bootstrap

The bilinear factors `(a, b)` cannot be sampled analytically because:
1. Each `aᵢ` is estimated from only team `i`'s games, so the factors are coupled through the residual structure in a complicated way.
2. The rotational non-identifiability means the factor distribution is not well-approximated by a simple Gaussian around the MAP estimate.

The parametric bootstrap is more appropriate:

1. Compute MAP fitted values from `(θ̂_main, â, b̂)`.
2. For each bootstrap draw `s = 1, ..., n`:
   a. Simulate new efficiency observations by adding Gaussian noise scaled by `σ̂_eff / sqrt(p_g / p̄)`.
   b. Rebuild synthetic `GameRow` objects with the simulated scores.
   c. Refit ALS on the synthetic data with the same `λ_ab` and convergence settings.
   d. Record `(a_s, b_s)`.
3. Return the collection of bootstrap draws.

The noise scaling `σ̂_eff / sqrt(p_g / p̄)` reflects the heteroskedasticity: high-possession games should be simulated with lower noise.

### Combined Draws

Each posterior draw `θ^(s)` concatenates:
- One Gaussian draw from `N(θ̂_main, Σ_main)`
- One Gaussian draw from `N(θ̂_pace, Σ_pace)`
- One bootstrap draw `(a_s, b_s)` from the parametric bootstrap

These three components are **independent** by construction (separate likelihoods, separate estimation steps).

---

## Matchup Compatibility Score

The primary downstream use of the bilinear factors is the matchup compatibility score:

```
compatibility_{ij} = aᵢᵀbⱼ
```

This is the expected additional efficiency of offense `i` against defense `j` beyond the main-effect prediction. It is the correct quantity to use as a downstream feature because it is:
1. Invariant to rotations of `a` and `b`.
2. Centered at zero (by the centering constraint, the mean compatibility across all pairs is zero).
3. Interpretable in the same units as efficiency (pts/100).

Delta-method variance of `aᵢᵀbⱼ` (treating `a` and `b` as random via the bootstrap):

```
Var(aᵢᵀbⱼ) ≈ b̂ⱼᵀ Var(aᵢ) b̂ⱼ + âᵢᵀ Var(bⱼ) âᵢ + 2 b̂ⱼᵀ Cov(aᵢ, bⱼ) âᵢ
```

In practice, the bootstrap distribution over `compatibility_{ij}` draws is the empirical approximation to this quantity and can be used directly for intervals.

---

## Feature Library

After fitting Model 3, the recommended feature set for downstream prediction:

**Main-effect features (identical to Model 2):**
- `o_i` — offensive quality above average
- `d_i` — defensive quality above average
- `r_i` — pace tendency
- `o_i − d_i` — net rating

**Matchup-level composites:**
- `o_i − d_j` — expected offensive advantage of team `i` vs defense `j`
- `o_j − d_i` — expected offensive advantage of team `j` vs defense `i`
- `(o_i − d_j) − (o_j − d_i)` — net expected edge
- `r_i + r_j` — expected combined tempo

**Interaction features (Model 3 only):**
- `aᵢᵀbⱼ` — compatibility score for offense `i` vs defense `j`
- `aⱼᵀbᵢ` — compatibility score for offense `j` vs defense `i`
- `‖aᵢ‖` — offensive style complexity
- `‖bⱼ‖` — defensive style complexity

The interaction features are highest value when the rank `k` is large enough to capture real style variation but small enough to be well-estimated.

---

## Strengths and Limitations

**Strengths:**
- Captures matchup-specific effects that scalar ratings cannot represent.
- Main effects retain their interpretation and are computed by the same procedure as Model 2.
- Centering constraint keeps the three-number KenPom summary well-defined and comparable to Models 1 and 2.
- The interaction score `aᵢᵀbⱼ` is a principled, rotation-invariant downstream feature.
- ALS is fast and reliable with moderate ridge penalties.

**Limitations:**
- Non-convex optimization: ALS finds a local optimum. Different random initializations can produce different factor coordinates (though the same interaction scores, up to the bootstrap variation).
- The bilinear term requires enough matchup repetition to be estimated reliably. With a typical schedule of ~30 games per team, low-rank factors with `k ≤ 5` are feasible; higher ranks are likely over-parameterized.
- Bootstrap uncertainty for the bilinear factors is more expensive than the analytic Gaussian uncertainty for main effects — each bootstrap replicate refits the full ALS.
- The main-effect Laplace posterior is conditioned on `(â, b̂)`, ignoring joint uncertainty between main effects and interaction factors. This is an approximation.

---

## Default Hyperparameters

| Parameter     | Default | Meaning                                          |
|---------------|---------|--------------------------------------------------|
| `rank`        | 3       | Dimensionality k of interaction vectors a, b     |
| `lambda_team` | 100.0   | Ridge penalty on offense/defense main effects    |
| `lambda_pace` | 50.0    | Ridge penalty on pace effects                    |
| `lambda_ab`   | 50.0    | Ridge penalty on bilinear factors a_i, b_j       |
| `max_iter`    | 100     | Maximum ALS iterations                           |
| `tol`         | 1e-5    | ALS convergence threshold                        |
| `n_boot`      | 200     | Bootstrap replicates for bilinear uncertainty    |

`rank = 3` is a conservative default. For a full season of D1 data with ~350 teams, ranks up to 5–8 are typically identifiable given the schedule structure. Larger ranks should be validated by checking out-of-sample efficiency prediction.
