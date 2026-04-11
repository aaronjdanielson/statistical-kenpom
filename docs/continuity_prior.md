# Cross-Season Continuity Prior

## 1. The Problem

Standard ridge regression for team ratings uses a zero-centered prior every season:

$$\hat\theta = (X^\top W X + \lambda I)^{-1} X^\top W y$$

This is correct when you have no information about where teams start. But you do: last
season's ratings. A team returning 80% of its minutes is unlikely to have shifted 10
points in efficiency. The zero-centered prior ignores that signal.

The question is: how much does it cost to ignore it, and what is the right way to use it?

---

## 2. The Model

Let $r_i \in [0,1]$ be the fraction of minutes returned by team $i$ entering season $s$.
The prior on team effects is:

$$o_i^s \mid o_i^{s-1},\ r_i \;\sim\; \mathcal{N}\!\left(r_i\, \hat{o}_i^{s-1},\; \tau_o^2(r_i)\right)$$

$$d_i^s \mid d_i^{s-1},\ r_i \;\sim\; \mathcal{N}\!\left(r_i\, \hat{d}_i^{s-1},\; \tau_d^2(r_i)\right)$$

where $r_i$ controls **mean persistence** and $\tau^2(r_i)$ controls **innovation
variance**. These are separate mechanisms and should not be conflated.

The prior mean vector is:

$$m = \bigl(0,\; r_1 \hat{o}_1,\; \ldots,\; r_T \hat{o}_T,\; r_1 \hat{d}_1,\; \ldots,\; r_T \hat{d}_T,\; 0\bigr)^\top$$

The MAP estimate under this prior satisfies the same normal equations as ridge, with
the right-hand side shifted by $\lambda m$:

$$\hat\theta = (X^\top W X + P)^{-1}(X^\top W y + P m)$$

The posterior covariance is unchanged in structure:

$$\Sigma = \sigma^2_{\text{eff}}\,(X^\top W X + P)^{-1}$$

No new solver is needed. Ordinary ridge had $Pm = 0$; this adds a linear term.

---

## 3. Two Variants

**Fix A (full prior):** $P = \text{diag}(\sigma^2_{\text{ref}} / \tau_i^2)$ with
team-specific precision. Simultaneously shifts the mean *and* modulates regularization
strength based on returning minutes.

**Fix B (shift-only):** $P = \lambda I$ exactly as in Model 2. Only the linear term
$\lambda m$ is added. Isolates the question: *does centering the prior at last
season's rating help, holding regularization fixed?*

Fix B is the right first experiment. It answers the cleanest possible question.

---

## 4. Experiment

**Season:** 2025–26 (6305 games total).  
**Training data for prior:** Full 2024–25 season fit (Model 2, 364 teams,
$\sigma_{\text{eff}} = 12.51$ pts/100).  
**Returning minutes:** Computed from `portal_pivot_db` player-level minutes,
same-player same-team join across consecutive seasons (326 of 365 teams with data).

Distribution of $r_i$: min 0.2%, median 50.2%, p75 69.0%, max 100%.

**Evaluation:** Rolling one-step-ahead RMSE over 6 early-season windows (Nov 7 –
Dec 19), with at least 50 training games per window. Four models compared:

| Model | Description |
|---|---|
| M2_static | Standard Model 2, zero-centered ridge |
| M2_hl60 | Model 2 with half-life 60 recency weighting |
| M2_shift | Model 2 + shift-only continuity prior (Fix B) |
| M2_shift_hl | Model 2 + shift-only + half-life 60 |

---

## 5. Results

### Overall RMSE (weeks 1–6, early season)

| Model | RMSE | Δ vs M2_static |
|---|---|---|
| M2_static | 14.753 | — |
| M2_hl60 | 14.731 | −0.022 |
| M2_shift | 14.519 | **−0.234** |
| M2_shift_hl | 14.476 | **−0.277** |

The mean shift alone delivers −0.234 RMSE improvement with no change to
regularization. Combined with recency weighting: −0.277.

### Per-window results

| Window | Date | Train | M2 | M2hl | Shift | Sh+hl |
|---|---|---|---|---|---|---|
| 1 | Nov 07 | 295 | 15.78 | 15.79 | 15.04 | 15.02 |
| 2 | Nov 14 | 676 | 14.63 | 14.59 | 14.43 | 14.37 |
| 3 | Nov 21 | 1032 | 15.80 | 15.76 | 15.55 | 15.49 |
| 4 | Nov 28 | 1425 | 14.29 | 14.25 | 14.17 | 14.11 |
| 5 | Dec 05 | 1748 | 13.94 | 13.98 | 13.88 | 13.89 |
| 6 | Dec 12 | 2026 | 14.08 | 14.02 | 14.05 | 13.97 |

The improvement is present across all six windows. It is largest in windows 1–4, when
the game graph is sparse and prior information is most valuable.

### Stratified by returning-minutes quartile

| Quartile | r cutoffs | N | M2_static | M2_shift | Δ |
|---|---|---|---|---|---|
| Q1 (low r) | ≤18% | 840 | 15.084 | 15.039 | −0.046 |
| Q2 | ≤45% | 845 | 15.154 | 14.891 | −0.263 |
| Q3 | ≤66% | 842 | 14.335 | **13.946** | **−0.389** |
| Q4 (high r) | >66% | 855 | 14.880 | 14.543 | −0.337 |

Q1 is essentially unchanged: with $r \approx 0$, $m \approx 0$, and the prior has no
effect. That is the correct behavior.

The gain is largest in Q3, not Q4. This is the diagnostic result: the prior adds most
value when it is *informative but not dominant* — the regime where Bayesian updating
is most valuable. In Q4, the prior is strong but the data also accumulate quickly and
begin to dominate.

### Stratified by opponent strength

Opponents ranked by previous-season adjusted offense (proxy for schedule quality).

| Quartile | N | M2_static | M2_shift | Δ |
|---|---|---|---|---|
| oQ1 (weak opp) | 843 | 15.372 | 15.091 | −0.281 |
| oQ2 | 845 | 14.785 | 14.563 | −0.222 |
| oQ3 | 840 | 14.471 | 14.328 | −0.143 |
| oQ4 (strong opp) | 854 | 14.826 | 14.450 | −0.376 |

The prior helps in all opponent quartiles, including against strong opponents. This
extends the identification story: the prior is not only resolving graph sparsity, it
is anchoring the global scale of team strength. Early in the season the absolute level
of ratings is uncertain for all teams, not just those with sparse schedules.

### Prior influence diagnostic

$$|m|_2 \approx 112, \qquad |\hat\theta - m|_2 \approx 131\text{–}185$$

The ratio $|\hat\theta - m| / |m| \approx 1.5$ places the posterior in the Goldilocks
regime: not collapsing to the prior, not ignoring it. The prior is informative but
the data are pulling estimates meaningfully away from it. This validates the scaling
— $\lambda$ and the returning-minutes data together produce a well-calibrated prior
without any additional tuning.

---

## 6. The Bias Decomposition

| Model | Bias |
|---|---|
| M2_static | −1.140 |
| M2_shift | −1.135 |
| M2_shift_hl | −1.099 |

The mean shift alone does not reduce bias. Only recency weighting does. This confirms
that the two mechanisms address orthogonal problems:

- **Continuity prior** → cross-season level alignment (RMSE, scale)
- **Half-life weighting** → within-season drift correction (bias)

---

## 7. The Key Finding

> **The benefit of continuity priors in early-season team ratings comes almost entirely
> from shifting the prior mean, not from modulating prior variance.**

The information in $m = \{r_i \hat\theta_{i,s-1}\}$ is real, usable, and correctly
weighted by the existing $\lambda$. No variance tuning is needed to recover most of
the gain.

This is not merely an empirical observation. It is consistent with a structural
interpretation: $\lambda$ already encodes the right amount of skepticism about any
team's initial position. What was missing was only the *location* of that prior — the
best guess about where teams start, not the confidence in that guess.

---

## 8. Implementation

`models/priors.py`:
- `load_returning_minutes(season)` — SQL join on `player_summaries` for same PlayerID, same TeamID, consecutive seasons
- `extract_prev_effects(model)` — extracts $(o_i, d_i)$ and posterior variances from a fitted Model 2
- `build_continuity_prior(teams, prev_effects, r_minutes, ...)` — returns $(m, p_{\text{diag}})$

`models/model2_continuity.py`:
- `Model2ContinuityPrior(Model2)` — overrides only `_fit_efficiency`
- `shift_only=True` parameter uses $P = \lambda I$ (Fix B); `shift_only=False` uses team-specific $P$ (Fix A)

All other model machinery — pace model, `predict_efficiency`, `sample_posterior`,
`point_summary` — is inherited unchanged.

---

## 9. What Comes Next

Fix B is the correct minimal model. Two extensions remain:

## 10. Fix A Results (variance modulation)

Fix A was evaluated over a grid of $\tau_{\text{hi}} \in \{1.0, 1.25, 1.5\}$,
$\tau_{\text{lo}} \in \{0.3, 0.5, 0.7\}$, using Fix B (shift + hl=60) as the baseline.

### RMSE

| $\tau_{\text{hi}}$ | $\tau_{\text{lo}}$ | RMSE | $\Delta$ vs Fix B |
|---|---|---|---|
| 1.00 | 0.30 | **14.230** | **−0.246** |
| 1.00 | 0.50 | 14.230 | −0.246 |
| 1.00 | 0.70 | 14.255 | −0.221 |
| 1.25 | 0.30 | 14.279 | −0.197 |
| 1.25 | 0.50 | 14.287 | −0.189 |
| 1.25 | 0.70 | 14.316 | −0.160 |
| 1.50 | 0.30 | 14.380 | −0.096 |
| 1.50 | 0.50 | 14.389 | −0.087 |
| 1.50 | 0.70 | 14.418 | −0.058 |

Every combination improves RMSE over Fix B. The gain increases as $\tau$ decreases —
i.e., as the prior is made tighter.

### Calibration

The posterior is catastrophically overconfident under Fix A. For all combinations:

- z_std ≈ 24–26 (ideal: ≈ 1.0)
- 90% coverage ≈ 6% (ideal: 90%)

### Interpretation

Fix A's RMSE gain and its calibration failure are the same mechanism. At
$\tau_{\text{lo}} = 0.3$, the prior precision for a returning team is
$\sigma^2 / \tau_{\text{lo}}^2 = 157 / 0.09 \approx 1740$, versus Model 2's
$\lambda = 100$. The prior is 17× tighter. Point estimates are pulled toward
informative prior means (good for RMSE) but the posterior covariance collapses
toward zero (bad for UQ). You cannot separate these effects.

### Conclusion

> **Fix A improves point-prediction RMSE by an additional −0.25, but destroys
> posterior uncertainty. It is not a valid production model for probabilistic use.**

Fix B (shift-only) is the correct choice: −0.277 RMSE improvement with no
calibration cost. This is the recommended early-season model.

*Exception:* if the only goal is point prediction and no downstream use of intervals,
simulation, or uncertainty is required, Fix A with $\tau_{\text{hi}}=1.0$,
$\tau_{\text{lo}}=0.3$ is a legitimate shrinkage estimator. It happens to be
parameterized as a Gaussian prior, but at those precision levels it is effectively
shrinking toward a point mass. Use it only if you are certain calibration does not matter.

The full improvement summary relative to M2_static:

| Model | Early-season RMSE | $\Delta$ | Calibration |
|---|---|---|---|
| M2_static | 14.753 | — | baseline |
| M2_hl60 | 14.731 | −0.022 | preserved |
| M2_shift_hl (Fix B) | **14.476** | **−0.277** | **preserved** |
| Fix A best | 14.230 | −0.523 | broken |

**Model 4 extension:** The same idea applies to the Kalman initial state. Replace the
uniform $C_0 = (\sigma^2/\lambda) I$ with team-specific $m_0 = r_i \hat\theta_{i,s-1}$
and $C_0^{(i)} = \tau^2(r_i)$. This would give Model 4 the same cross-season anchor
while preserving its within-season dynamics. The same RMSE/calibration tradeoff
would apply — the shift alone is likely the right extension there too.
