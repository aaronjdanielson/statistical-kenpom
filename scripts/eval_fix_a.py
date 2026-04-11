"""
Early-season OSA evaluation: Fix A — variance modulation.

Fix A enables team-specific precision in the prior:

    P_ii = sigma2_ref / tau_i²(r_i)
    tau²(r) = tau_hi² − r · (tau_hi² − tau_lo²)

The question: does modulating prior variance (on top of the mean shift) help?

Baseline: M2_shift_hl (Fix B — shift-only + hl=60), RMSE = 14.476.

Grid search
-----------
tau_hi ∈ {1.0, 1.25, 1.5}  — prior std at r=0 (new team)
tau_lo ∈ {0.3, 0.5, 0.7}   — prior std at r=1 (full return)

Constraint: tau_hi ≤ sqrt(sigma2/lambda) ≈ 1.25 to avoid weakening
regularization for new teams vs Model2. tau_hi=1.5 is included as a
boundary check but should hurt Q1 teams.

Calibration tracking
--------------------
For each (tau_lo, tau_hi) combination, we check whether variance modulation
improves or degrades predictive calibration:
  - z_std: std of (pred - actual) / predictive_std  (should be ≈ 1.0)
  - coverage_90: fraction of actuals within 90% predictive interval
  - coverage_50: fraction of actuals within 50% predictive interval

Predictive std is approximated by posterior sampling (100 draws).

Outputs
-------
  fix_a_grid.png     — heatmap: ΔRMSE vs (tau_lo, tau_hi), baseline = Fix B
  fix_a_calib.png    — calibration panel: z_std and coverage for each combo
"""
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games, parse_date, GameRow
from models.model2 import Model2
from models.model2_continuity import Model2ContinuityPrior
from models.priors import extract_prev_effects, load_returning_minutes
from models.eval import recency_weights

# ── Config ─────────────────────────────────────────────────────────────────────
SEASON      = 2026
PREV_SEASON = 2025
WEEK_STEP   = 7
MIN_TRAIN   = 50
EVAL_WEEKS  = 6
RECENCY_HL  = 60
N_CALIB     = 100        # posterior samples for calibration

TAU_HI_GRID = [1.0, 1.25, 1.5]
TAU_LO_GRID = [0.3, 0.5, 0.7]

GRID_OUT  = Path(__file__).parent / "fix_a_grid.png"
CALIB_OUT = Path(__file__).parent / "fix_a_calib.png"

# ── Load data ──────────────────────────────────────────────────────────────────
conn = open_db()
cur  = conn.cursor()
cur.execute("""
    SELECT DISTINCT s.GameID, s.Date
    FROM schedules s JOIN boxscores b ON s.GameID = b.GameID
    WHERE s.Year = ? ORDER BY s.GameID
""", (SEASON,))
game_order = cur.fetchall()
conn.close()

game_dt  = {gid: parse_date(ds) for gid, ds in game_order}
all_gids = [gid for gid, _ in game_order]

all_rows = load_season_games(open_db(), SEASON)
row_by_gid: dict[int, list[GameRow]] = defaultdict(list)
for r in all_rows:
    row_by_gid[r.game_id].append(r)

prev_rows = load_season_games(open_db(), PREV_SEASON)

season_start = min(game_dt.values())
season_end   = max(game_dt.values())

# ── Build early-season windows ─────────────────────────────────────────────────
windows = []
d = season_start
while d <= season_end and len(windows) < EVAL_WEEKS:
    d_next = d + timedelta(days=WEEK_STEP)
    train  = {g for g in all_gids if game_dt[g] < d}
    pred   = {g for g in all_gids if d <= game_dt[g] < d_next}
    if len(train) >= MIN_TRAIN and pred:
        windows.append((d, train, pred))
    d = d_next

print(f"Season {SEASON}: {len(all_gids)} games total, {len(windows)} early-season windows")

# ── Fit prior season → extract effects ────────────────────────────────────────
print(f"\nFitting Model2 on full season {PREV_SEASON}...", end=" ", flush=True)
m_prev = Model2(lambda_team=100.0, lambda_pace=50.0)
m_prev.fit_rows(prev_rows, PREV_SEASON)
prev_effects, prev_var = extract_prev_effects(m_prev)
sigma2_prev = float(m_prev._sigma2_eff)
print(f"done  ({len(m_prev.teams_)} teams, σ_eff={sigma2_prev**0.5:.2f})")

print(f"Loading returning minutes for season {SEASON}...", end=" ", flush=True)
r_minutes = load_returning_minutes(SEASON)
print(f"done  ({len(r_minutes)} teams with data)")

# ── Baseline: Fix B (shift-only + hl=60) ──────────────────────────────────────
print(f"\nComputing Fix B baseline across {len(windows)} windows...")
baseline_rmse_windows = []
for win_dt, train_gids, pred_gids in windows:
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
    actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])
    sw         = recency_weights(train_rows, game_dt, win_dt, half_life_days=RECENCY_HL)
    m_base = Model2ContinuityPrior(
        prev_effects=prev_effects, r_minutes=r_minutes,
        sigma2_prev=sigma2_prev, prev_var=prev_var,
        shift_only=True, lambda_team=100.0, lambda_pace=50.0,
    )
    m_base.fit_rows(train_rows, SEASON, sample_weight=sw)
    resid = m_base.predict_efficiency(test_rows) - actual
    baseline_rmse_windows.append(float(np.sqrt(np.mean(resid ** 2))))

baseline_rmse = float(np.mean(baseline_rmse_windows))
print(f"Fix B baseline RMSE: {baseline_rmse:.3f}")

# ── Grid search ───────────────────────────────────────────────────────────────
# Results: dict[(tau_lo, tau_hi)] → (rmse, delta_rmse, z_std, cov50, cov90)
grid_results: dict[tuple, dict] = {}

total_combos = len(TAU_HI_GRID) * len(TAU_LO_GRID)
print(f"\nGrid search: {total_combos} combinations × {len(windows)} windows × "
      f"{N_CALIB} calib samples\n")

header = f"{'tau_hi':>6}  {'tau_lo':>6}  {'RMSE':>7}  {'ΔRMSE':>7}  "
header += f"{'z_std':>6}  {'cov50':>6}  {'cov90':>6}"
print(header)
print("-" * len(header))

rng = np.random.default_rng(42)

for tau_hi in TAU_HI_GRID:
    for tau_lo in TAU_LO_GRID:
        if tau_lo >= tau_hi:
            continue   # degenerate: tau_lo must be < tau_hi

        rmse_list, z_list, cov50_list, cov90_list = [], [], [], []

        for win_dt, train_gids, pred_gids in windows:
            train_rows = [r for g in train_gids for r in row_by_gid[g]]
            test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
            actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])
            sw         = recency_weights(train_rows, game_dt, win_dt,
                                         half_life_days=RECENCY_HL)

            m_fa = Model2ContinuityPrior(
                prev_effects=prev_effects,
                r_minutes=r_minutes,
                sigma2_prev=sigma2_prev,
                prev_var=prev_var,
                shift_only=False,           # Fix A: team-specific P
                tau_o_lo=tau_lo,
                tau_o_hi=tau_hi,
                tau_d_lo=tau_lo,
                tau_d_hi=tau_hi,
                lambda_team=100.0,
                lambda_pace=50.0,
            )
            m_fa.fit_rows(train_rows, SEASON, sample_weight=sw)

            pred_point = m_fa.predict_efficiency(test_rows)
            resid = pred_point - actual
            rmse_list.append(float(np.sqrt(np.mean(resid ** 2))))

            # Calibration via posterior sampling (uses predict_interval from base)
            draws = m_fa.sample_posterior(N_CALIB, rng)   # list of theta arrays
            pred_draws = np.array([
                m_fa._predict_from_theta(th, test_rows) for th in draws
            ])  # shape (N_CALIB, n_test)

            pred_std = pred_draws.std(axis=0)
            valid    = pred_std > 0.01
            if valid.any():
                z_scores = (pred_point[valid] - actual[valid]) / pred_std[valid]
                z_list.append(float(z_scores.std()))

                lo50 = np.percentile(pred_draws, 25, axis=0)[valid]
                hi50 = np.percentile(pred_draws, 75, axis=0)[valid]
                lo90 = np.percentile(pred_draws,  5, axis=0)[valid]
                hi90 = np.percentile(pred_draws, 95, axis=0)[valid]
                cov50_list.append(float(np.mean((actual[valid] >= lo50) &
                                                 (actual[valid] <= hi50))))
                cov90_list.append(float(np.mean((actual[valid] >= lo90) &
                                                 (actual[valid] <= hi90))))

        rmse     = float(np.mean(rmse_list))
        delta    = rmse - baseline_rmse
        z_std    = float(np.mean(z_list))    if z_list    else float("nan")
        cov50    = float(np.mean(cov50_list)) if cov50_list else float("nan")
        cov90    = float(np.mean(cov90_list)) if cov90_list else float("nan")

        grid_results[(tau_lo, tau_hi)] = dict(
            rmse=rmse, delta=delta, z_std=z_std, cov50=cov50, cov90=cov90,
        )

        flag = " ←" if delta < -0.02 else ("  ↑" if delta > 0.02 else "")
        print(f"  {tau_hi:>6.2f}  {tau_lo:>6.2f}  {rmse:>7.3f}  {delta:>+7.3f}  "
              f"{z_std:>6.3f}  {cov50:>6.3f}  {cov90:>6.3f}{flag}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nBaseline (Fix B shift+hl):  {baseline_rmse:.3f}")
best_key = min(grid_results, key=lambda k: grid_results[k]["rmse"])
best     = grid_results[best_key]
print(f"Best Fix A:   tau_hi={best_key[1]:.2f}, tau_lo={best_key[0]:.2f}  "
      f"RMSE={best['rmse']:.3f}  Δ={best['delta']:+.3f}  "
      f"z_std={best['z_std']:.3f}  cov90={best['cov90']:.3f}")

# ── Figures ────────────────────────────────────────────────────────────────────
BG    = "#0f0f1a"
GRID  = "#1a1a3a"
SPINE = "#333355"
TICK  = "#aaaacc"

# Figure 1: ΔRMSE heatmap
tau_lo_arr = sorted(set(k[0] for k in grid_results))
tau_hi_arr = sorted(set(k[1] for k in grid_results))

delta_mat  = np.full((len(tau_hi_arr), len(tau_lo_arr)), np.nan)
for (tlo, thi), res in grid_results.items():
    i = tau_hi_arr.index(thi)
    j = tau_lo_arr.index(tlo)
    delta_mat[i, j] = res["delta"]

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.tick_params(colors=TICK, labelsize=10)

vmax = max(0.1, np.nanmax(np.abs(delta_mat)))
im = ax.imshow(delta_mat, aspect="auto", cmap="RdYlGn_r",
               vmin=-vmax, vmax=vmax, origin="lower")

ax.set_xticks(range(len(tau_lo_arr)))
ax.set_xticklabels([f"{v:.2f}" for v in tau_lo_arr], color=TICK)
ax.set_yticks(range(len(tau_hi_arr)))
ax.set_yticklabels([f"{v:.2f}" for v in tau_hi_arr], color=TICK)
ax.set_xlabel("τ_lo  (prior std at r=1, tight end)", color=TICK)
ax.set_ylabel("τ_hi  (prior std at r=0, new teams)", color=TICK)
ax.set_title(f"Fix A vs Fix B baseline  ·  ΔRMSE  (green = better than Fix B)\n"
             f"baseline = {baseline_rmse:.3f}  ·  early-season wks 1–{EVAL_WEEKS}",
             color="white", fontsize=10, fontweight="bold")

for i in range(len(tau_hi_arr)):
    for j in range(len(tau_lo_arr)):
        v = delta_mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=TICK, labelsize=8)
cb.set_label("ΔRMSE vs Fix B", color=TICK, fontsize=9)

plt.tight_layout()
plt.savefig(GRID_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved grid → {GRID_OUT}")

# Figure 2: calibration panel
n_combos = len(grid_results)
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.patch.set_facecolor(BG)
ax_zstd, ax_cov50, ax_cov90 = axes2

combo_labels = [f"hi={thi:.2f}\nlo={tlo:.2f}"
                for (tlo, thi) in sorted(grid_results)]
z_vals   = [grid_results[k]["z_std"]  for k in sorted(grid_results)]
c50_vals = [grid_results[k]["cov50"]  for k in sorted(grid_results)]
c90_vals = [grid_results[k]["cov90"]  for k in sorted(grid_results)]
x        = np.arange(len(combo_labels))

for ax, vals, target, title, ylabel in [
    (ax_zstd,  z_vals,   1.0,  "z-score std  (ideal = 1.0)",    "z-std"),
    (ax_cov50, c50_vals, 0.50, "50% coverage  (ideal = 0.50)",  "coverage"),
    (ax_cov90, c90_vals, 0.90, "90% coverage  (ideal = 0.90)",  "coverage"),
]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", axis="y")
    ax.bar(x, vals, color="#a8ff78", alpha=0.80, edgecolor=SPINE)
    ax.axhline(target, color="#ff9f1c", lw=1.5, linestyle="--", label=f"ideal {target}")
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, color=TICK, fontsize=7)
    ax.set_ylabel(ylabel, color=TICK, fontsize=9)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.legend(labelcolor="white", facecolor="#111133", edgecolor=SPINE, fontsize=8)

fig2.suptitle(
    f"Fix A calibration diagnostics  ·  {SEASON} wks 1–{EVAL_WEEKS}  "
    f"·  {N_CALIB} posterior samples",
    color="white", fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig(CALIB_OUT, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved calibration → {CALIB_OUT}")
