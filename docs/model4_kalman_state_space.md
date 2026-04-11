# Model 4: Weekly Kalman State-Space Ratings

**Status:** complete and validated  
**Related scripts:** `scripts/model_osa_comparison.py`, `scripts/verify_graph_and_rts.py`  
**Tests:** `tests/unit/test_model4.py` (27 tests, all passing)

---

## Overview

Model 4 extends the ridge regression framework of Model 2 to a *dynamic* setting:
team offensive and defensive effects are allowed to drift week-to-week as random walks,
while the league-wide intercept μ and home advantage η remain static.
The model is estimated by a Kalman forward filter plus RTS smoother operating on
weekly observation batches.

This document covers:

1. The model specification and its exact correspondence to Model 2
2. A formal equivalence theorem (Model 2 = RTS smoother, τ→0)
3. The identification problem: schedule confounding and graph connectivity
4. Empirical one-step-ahead comparison across all four models
5. Practical guidance on when to use each model

---

## 1. Model Specification

### Observation equation

For each team-game row *(i scores against j)*:

```
y_k = μ + o_i(t) − d_j(t) + η · h_k + ε_k
ε_k ~ N(0, σ²_eff / poss_k)
```

where `o_i(t)` and `d_j(t)` are the offense and defense effects of teams i and j
at the time of game k, and `poss_k` is the possession count (possession-weighted
WLS matching Model 2).

### State equation

Between weeks t and t+1, each team's effects evolve as:

```
o_i(t+1) = o_i(t) + ω_o,i       ω_o,i ~ N(0, τ_o²)
d_i(t+1) = d_i(t) + ω_d,i       ω_d,i ~ N(0, τ_d²)
```

The state vector at week t is `θ_t = (o_1, …, o_T, d_1, …, d_T) ∈ ℝ^{2T}`.

### Prior

The initial prior over the state is:

```
θ_0 ~ N(0, C_0)    where    C_0 = (σ²_eff / λ) · I
```

This matches Model 2's ridge prior: `N(0, σ²_eff / λ · I)`.
**Parameterization note:** the prior precision is `λ / σ²_eff`, not `λ`.
Using `C_0 = I/λ` (prior precision = λ) is wrong by a factor of σ²_eff ≈ 160
and causes the filter to treat all teams as near-identical — effectively ignoring observations.

### Fixed parameters

μ and η are pre-estimated from a Model 2 fit on the full season, then held fixed
during the Kalman pass. σ²_eff is also taken from Model 2. The Kalman filter
then only tracks the (o, d) block, making the state space manageable.

---

## 2. The Equivalence Theorem

### Proposition (Model 2 = RTS smoother, τ→0)

> **In a linear-Gaussian team efficiency model with ridge prior N(0, σ²/λ · I)
> and possession-weighted observation variance σ²/poss_k, the MAP estimator
> (Model 2 ridge regression) equals the RTS smoother applied to the Kalman
> state-space model in the limit τ→0, when μ and η are fixed at the same values.**

*Proof sketch.*  When τ_o = τ_d = 0, the process noise covariance Q = 0 and the
state does not evolve between steps.  The Kalman forward filter then reduces to
sequential Bayesian updating on a static parameter θ:

```
After week t:
  C_t^{-1} = C_0^{-1} + Σ_{k ≤ t}  H_k' (σ²/poss_k)^{-1} H_k
            = (λ/σ²) I  +  (1/σ²) Σ_{k ≤ t}  poss_k · H_k' H_k
            = (1/σ²) [ λI + X_t' W_t X_t ]
```

where X_t is the design matrix of all games up to week t and W_t = diag(poss_k/σ²).
This is exactly the ridge precision matrix after seeing games 1:t.

The RTS smoother at τ=0 propagates no information backward (Q=0 means the gain
matrix G_t = C_t F' R_{t+1}^{-1} → I), so the smoothed state equals the
filtered state at every step, which equals the ridge posterior on the full season.

**Empirically verified:** with τ = 10⁻⁴ on the 2025-26 season (6305 games, 365 teams),
max |ΔAdjO| = 0.000 pts/100 (floating-point agreement) between RTS smoother
and Model 2. *(See `scripts/verify_graph_and_rts.py`, Test 2.)*

### Corollary (Filter error = identification loss)

For τ > 0, the **forward Kalman filter** (causal, sees only games 1:t) produces a
different — and generally worse — estimate than the RTS smoother or Model 2.
The gap is exactly the information missing from the *causal game subgraph* at time t:

```
E[||θ̂_filter(t) − θ_MAP||²]  ∝  (information missing from G_{1:t})
```

This gap is large in the early season (sparse, disconnected graph) and shrinks as
the graph becomes fully connected. The forward filter is penalized for not yet having
seen the future games that would resolve the current ranking ambiguity.

---

## 3. Game Graph Connectivity

### Why connectivity matters

A team's efficiency cannot be estimated accurately until it is connected (through game
chains) to the rest of the league.  With only 2–3 games per team per week, the early-
season game graph is sparse and fragmented.  As more games are played the graph
approaches full connectivity, and every team is linked to every other within a few hops.

This is the *identification problem*: even with a perfectly specified model, the
causal filter cannot distinguish "team A is strong" from "team A played weak opponents"
until it accumulates enough cross-schedule connections.

### Empirical connectivity (2025-26 season)

| Date | Train games | Components | LCC% | Diameter | ≤2 hops | ≤3 hops |
|---|---|---|---|---|---|---|
| Nov 14 |  676 |  4 | 96% | 6.0 | 14.5% |  4.2% |
| Nov 21 | 1032 |  2 | 99% | 5.0 | 33.9% | 68.7% |
| Nov 28 | 1425 |  1 | 100% | 4.0 | 55.2% | 92.1% |
| Dec 05 | 1748 |  1 | 100% | 4.0 | 68.7% | 97.8% |
| Jan 09 | 3149 |  1 | 100% | 4.0 | 87.5% | 98.2% |
| Feb 06 | 4503 |  1 | 100% | 4.0 | 95.7% | 99.8% |
| Mar 27 | 6291 |  1 | 100% | 4.0 | 96.5% | 99.9% |

Key transitions:

- **Week 1–2:** 4 disconnected components; less than 5% of team pairs linked within 3 hops.
  Efficiency estimates for unconnected components are anchored only to the prior.
- **Week 3 (Nov 28):** first fully connected graph; early neutral-site tournaments have
  bridged the major conferences.
- **January onward:** diameter stabilizes at 4; >98% of pairs reachable in ≤3 hops.
  Identification is essentially complete for well-connected teams.

*(Source: `scripts/verify_graph_and_rts.py`, Test 1 — scipy sparse BFS on the
undirected game adjacency matrix.)*

---

## 4. Identification Problem and Schedule Confounding

### The fundamental confound

The forward Kalman filter must rank teams using only games played so far.
But whether a team's high AdjO is *real* (genuine offensive strength) or *apparent*
(weak opponents, easy early schedule) cannot be resolved until those opponents play
other common opponents.

Formally: the causal subgraph G_{1:t} has sparse inter-conference edges in November.
The model correctly assigns uncertainty — but the *point estimate* is confounded.
This confounding is not a bug; it is the irreducible cost of causal estimation.

### Why τ was overestimated (pre-fix)

With the wrong prior C_0 = I/λ (too tight by 160×), the filter ignored new observations:
the posterior was dominated by the prior regardless of what games occurred.
To reconcile the (ignored) observations with a fixed μ/η, the MLE optimizer drove
τ upward — large drift was the only way to let the state "escape" the prior.

After the fix (C_0 = σ²/λ · I), optimal τ dropped from ~3–6 to **τ ≈ 0.25** per week.
At σ²_eff = 160 and ~2 games/team/week, τ = 0.25 implies a signal-to-noise ratio
per week of roughly:

```
SNR ≈ τ² / (σ²_eff / poss_k · n_games_per_team)
    ≈ 0.0625 / (160/70 · 2)
    ≈ 0.014   (1.4% of variance is true drift)
```

This is plausible: real teams do drift, but slowly relative to game-to-game noise.

### RMSE decomposition

The one-step-ahead RMSE can be decomposed as:

```
RMSE²(t) = σ²_irreducible               (irreducible game noise)
          + σ²_param(t)                  (estimation variance, decreases with data)
          + σ²_identification(t)         (causal filter loss vs RTS smoother)
          + σ²_drift(t)                  (true week-to-week drift missed by static model)
```

- **Static models** (M1, M2 static) have zero σ²_drift term by assumption, but pay
  σ²_identification = 0 because they use the full-season solve (equivalent to RTS).
- **Kalman forward filter** has a σ²_identification term from the causal subgraph — 
  this is large early and small late.
- **RTS smoother** (non-causal, uses full season) eliminates σ²_identification at the
  cost of not being usable for live prediction.

---

## 5. One-Step-Ahead Empirical Comparison

All four models evaluated on the 2025-26 season (20 weekly windows, ≥300 training games
before first prediction). Methodology: fit on all games before week W, predict games
in week W, compute RMSE/MAE/bias on offensive efficiency (pts/100 poss).

| Model | RMSE | MAE | Bias | Notes |
|---|---|---|---|---|
| M1 (KenPom fixed-point) | **15.10** | 11.91 | −1.45 | Heuristic baseline |
| M2 static (ridge) | 14.14 | 11.24 | −1.18 | Exact posterior |
| M2 hl=60 (recency) | **13.87** | — | −0.96 | **Best RMSE** |
| M4 Kalman (τ=0.25) | 14.12 | — | **−0.35** | **Best bias** |

### Reading the table

**M1 vs M2 static (+0.96 RMSE):** The fixed-point algorithm is a heuristic approximation
of the same WLS problem that M2 solves exactly. The gap is structural and consistent —
M1 is never better than M2 except in late tournament weeks when sample sizes are tiny.
M1's bias is also worse: fixed-point damping shrinks extreme estimates too aggressively.

**M2 static vs M2 hl=60 (−0.27 RMSE):** Exponential recency weighting with a 60-day
half-life reduces RMSE by 0.27 and bias by 0.22. Small but consistent improvement;
this is the best single model for production use.

**M2 hl=60 vs M4 Kalman (+0.25 RMSE, −0.61 bias):** The state-space model reduces
bias by 63% (−0.96 → −0.35) by tracking season-long evolution. But the RMSE cost
of +0.25 pts/100 means the variance reduction from better trend estimation does not
offset the added parameter uncertainty. MSE = Bias² + Variance; the bias² term
is not large enough to pay for the added variance.

**The early-season exception:** M4 wins in weeks 1–3 when the graph is sparse and
the Kalman prior dominates. In week 1, M4 RMSE ≈ M2 RMSE − 0.5. This advantage
disappears by week 4 as the graph connects.

**Tournament degradation:** In March (conference tournaments + NCAA tournament),
both models converge as the field shrinks and team quality is concentrated. M4
does not catastrophically degrade with the corrected prior.

### Calibration

Model 4's prediction intervals are better calibrated than Model 2's:
z-score standard deviation σ_z ≈ 1.13 vs M2's σ_z > 1.3 (overconfident intervals).
The Kalman posterior correctly inflates uncertainty in early-season and after the
Kalman gain absorbs new observations.

---

## 6. Practical Guidance

### Which model to use

| Phase | Recommended model | Reason |
|---|---|---|
| Weeks 1–3 (Nov) | M4 Kalman | Prior dominates; Kalman uncertainty quantification is correct |
| Weeks 4+ (Dec–Feb) | M2 hl=60 | Better RMSE; recency captures moderate drift |
| Tournament (March) | M2 static | Stable ratings; no drift needed for single-elimination |
| Historical analysis | M4 RTS smoother | Non-causal; retroactively identifies trajectory |
| Uncertainty intervals | M4 Kalman | Correctly calibrated; M2 intervals are too tight |

### Model 4 as an analysis tool

The RTS smoother (Model 4, non-causal) is the most useful artifact from Model 4.
Call `model4.rts_smoother()` to get a weekly trajectory of ratings that uses the
full season context at every time step. This answers: "What was Duke's true offensive
strength in mid-January, given everything we now know about their opponents?"

The forward filter (`point_summary_trajectory()`) answers the causal question: "What
did we know about Duke in mid-January based only on games played so far?"

These two answers diverge most in the early season and converge by March.

---

## 7. Implementation Notes

### Key parameters

| Parameter | Value | How set |
|---|---|---|
| σ²_eff | ~160 (pts/100)² | From Model 2 residual variance |
| λ_team | 100.0 | Same as Model 2 |
| λ_pace | 50.0 | Same as Model 2 |
| τ_o, τ_d | 0.25/week | MLE on season (post prior-fix) |
| C_0 | (σ²_eff/λ) · I ≈ 1.6 · I | Matched to Model 2 ridge prior |
| Obs. noise | σ²_eff / poss_k | Matched to Model 2 WLS weights |

### Correctness checks (verified)

1. **RTS(τ→0) = Model 2**: max |ΔAdjO| = 0.000 pts/100 on 365 teams, full season.
2. **Filter uncertainty shrinks after observation**: `diag(C_filtered) ≤ diag(R_predicted)`.
3. **Empty week leaves state unchanged** (only adds Q to covariance).
4. **Terminal smoothed state = terminal filtered state** (no future data to propagate back).
5. **Team ordering preserved** under all τ values (sanity check on synthetic data).
6. **Optimised log-lik ≥ fixed-τ log-lik** (MLE is at least as good as any fixed point).

All 27 unit tests (`tests/unit/test_model4.py`) pass.

---

## 8. Theoretical Significance

The equivalence theorem (Section 2) establishes that Model 2 and Model 4 are not
two different models — they are the *same model* viewed from two angles:

- **Model 2** solves the batch posterior over the full season at once.
- **Model 4 (RTS, τ→0)** reaches the same posterior by sequential Bayesian updating
  followed by backward smoothing.

This means every result from Model 2 has an exact sequential interpretation:
the Kalman gain at time t tells you *which games were most informative*, and
the smoother gain tells you *how much future games revised past estimates*.

The forward filter deviation from Model 2 — which shrinks as the game graph connects —
is a formal measure of **identification incompleteness**: how much information is
still missing from the causal subgraph at any point in the season.

> "The right amount of temporal modeling for NCAA basketball ratings is small.
> Simple exponential recency weighting captures the available drift signal.
> Full state-space dynamics add correct uncertainty quantification and bias reduction,
> but at a variance cost that exceeds the benefit for point prediction.
> The most practically useful artifact of the state-space model is not the forward
> filter but the retroactive RTS smoother, which provides the ground truth trajectory
> that no causal model can access in real time."
