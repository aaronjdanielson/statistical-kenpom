"""
Phase 1 experiment: exponential recency weighting half-life grid search.

For each half-life in HALF_LIVES (plus the static baseline at ∞):
  1. Run the rolling one-step-ahead evaluation (fit before week W, predict week W)
     with Model 2 weighted by exp(-κ · age_days).
  2. Record per-window RMSE, MAE, bias.
  3. Compute conformal calibration scores on the full-season 80/20 split.

Produces two figures:
  recency_osa_curves.png   — RMSE / bias curves over the season for each half-life
  recency_summary.png      — aggregate RMSE, MAE, bias, and calibration vs half-life
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games, parse_date, GameRow
from models.model2 import Model2
from models.eval import temporal_split, recency_weights, conformal_calibration_scores

# ── Config ────────────────────────────────────────────────────────────────────
SEASON      = 2026
WEEK_STEP   = 7
MIN_TRAIN   = 300
# None = static baseline (no recency weighting)
HALF_LIVES  = [None, 120, 90, 60, 45, 30, 14]
CURVES_OUT  = Path(__file__).parent / "recency_osa_curves.png"
SUMMARY_OUT = Path(__file__).parent / "recency_summary.png"

# ── Load data ─────────────────────────────────────────────────────────────────
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

season_start = min(game_dt.values())
season_end   = max(game_dt.values())

# ── Build weekly windows ──────────────────────────────────────────────────────
windows = []
d = season_start
while d <= season_end:
    d_next = d + timedelta(days=WEEK_STEP)
    train  = {g for g in all_gids if game_dt[g] < d}
    pred   = {g for g in all_gids if d <= game_dt[g] < d_next}
    if len(train) >= MIN_TRAIN and pred:
        windows.append((d, train, pred))
    d = d_next

print(f"Season {SEASON}: {len(all_gids)} games, {len(windows)} windows")

# ── Rolling OSA evaluation ────────────────────────────────────────────────────
# results[hl] = [(date, rmse, mae, bias), ...]
results: dict = {hl: [] for hl in HALF_LIVES}

for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
    actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    for hl in HALF_LIVES:
        m = Model2()
        if hl is None:
            m.fit_rows(train_rows, SEASON)
        else:
            sw = recency_weights(train_rows, game_dt, win_dt, half_life_days=hl)
            m.fit_rows(train_rows, SEASON, sample_weight=sw)

        pred   = m.predict_efficiency(test_rows)
        resid  = pred - actual
        rmse   = float(np.sqrt(np.mean(resid ** 2)))
        mae    = float(np.mean(np.abs(resid)))
        bias   = float(np.mean(resid))
        results[hl].append((win_dt, rmse, mae, bias))

    if w_idx % 4 == 0 or w_idx == len(windows) - 1:
        static_rmse = results[None][-1][1]
        best_hl  = min((hl for hl in HALF_LIVES if hl is not None),
                       key=lambda h: results[h][-1][1])
        best_rmse = results[best_hl][-1][1]
        print(f"  {w_idx+1:>2}/{len(windows)}  {win_dt:%b %d}  "
              f"train={len(train_gids):>4}  static={static_rmse:.2f}  "
              f"best(hl={best_hl})={best_rmse:.2f}", flush=True)

# ── Aggregate stats ───────────────────────────────────────────────────────────
print(f"\n{'Half-life':>12}  {'RMSE':>7}  {'MAE':>7}  {'Bias':>7}")
print("-" * 42)
agg = {}
for hl in HALF_LIVES:
    pts = results[hl]
    all_rmse = [p[1] for p in pts]
    all_mae  = [p[2] for p in pts]
    all_bias = [p[3] for p in pts]
    label = "∞ (static)" if hl is None else f"{hl} days"
    rmse_mean = float(np.mean(all_rmse))
    mae_mean  = float(np.mean(all_mae))
    bias_mean = float(np.mean(all_bias))
    agg[hl] = (rmse_mean, mae_mean, bias_mean)
    flag = "  ← baseline" if hl is None else ""
    print(f"  {label:>10}  {rmse_mean:>7.3f}  {mae_mean:>7.3f}  {bias_mean:>+7.3f}{flag}")

# ── Calibration check at each half-life ──────────────────────────────────────
print("\nCalibration (80/20 split, 90% PI coverage):")
train_rows_cal, test_rows_cal = temporal_split(all_rows, 0.80)
cutoff_cal = max(game_dt[r.game_id] for r in train_rows_cal)
cal_coverage = {}

for hl in HALF_LIVES:
    m = Model2()
    if hl is None:
        m.fit_rows(train_rows_cal, SEASON)
    else:
        sw = recency_weights(train_rows_cal, game_dt, cutoff_cal, half_life_days=hl)
        m.fit_rows(train_rows_cal, SEASON, sample_weight=sw)

    z, cov = conformal_calibration_scores(
        m, test_rows_cal, n_draws=200, rng=np.random.default_rng(0)
    )
    cal_coverage[hl] = cov
    label = "∞ (static)" if hl is None else f"{hl} days"
    print(f"  {label:>10}  z_std={z.std():.3f}  "
          f"cov@90%={cov[0.90]:.1%}  cov@95%={cov[0.95]:.1%}")

# ── Figure 1: OSA curves ──────────────────────────────────────────────────────
BG, GRID, SPINE, TICK = "#0f0f1a", "#1a1a3a", "#333355", "#aaaacc"
cmap_lines = plt.get_cmap("plasma")
n_hl = len(HALF_LIVES)

fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.patch.set_facecolor(BG)
for ax in [ax_rmse, ax_bias]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")

for idx, hl in enumerate(HALF_LIVES):
    pts = results[hl]
    dates = [p[0] for p in pts]
    rmses = [p[1] for p in pts]
    biases = [p[3] for p in pts]
    color = "white" if hl is None else cmap_lines(idx / (n_hl - 1))
    lw    = 2.5    if hl is None else 1.8
    ls    = "--"   if hl is None else "-"
    label = "∞  static" if hl is None else f"hl={hl}d"
    ax_rmse.plot(dates, rmses,  color=color, lw=lw, linestyle=ls, label=label)
    ax_bias.plot(dates, biases, color=color, lw=lw, linestyle=ls, label=label)

ax_rmse.set_ylabel("Weekly RMSE  (pts/100)", color=TICK)
ax_rmse.set_title(
    f"One-step-ahead RMSE  ·  static vs exponential recency  ·  2025-26",
    color="white", fontsize=11, fontweight="bold",
)
ax_rmse.legend(loc="upper right", labelcolor="white", facecolor="#111133",
               edgecolor=SPINE, fontsize=8, ncol=4)

ax_bias.axhline(0, color="#555577", lw=0.9, linestyle=":")
ax_bias.set_ylabel("Weekly bias  (pred − actual,  pts/100)", color=TICK)
ax_bias.set_title("Bias — target: collapse toward 0",
                  color="white", fontsize=11, fontweight="bold")
ax_bias.legend(loc="upper right", labelcolor="white", facecolor="#111133",
               edgecolor=SPINE, fontsize=8, ncol=4)
ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_bias.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved curves → {CURVES_OUT}")

# ── Figure 2: Summary vs half-life ───────────────────────────────────────────
numeric_hls = [hl for hl in HALF_LIVES if hl is not None]
x_labels    = ["static\n(∞)"] + [f"{h}d" for h in numeric_hls]
x_pos       = np.arange(len(HALF_LIVES))

rmse_vals  = [agg[hl][0] for hl in HALF_LIVES]
mae_vals   = [agg[hl][1] for hl in HALF_LIVES]
bias_vals  = [agg[hl][2] for hl in HALF_LIVES]
cov90_vals = [cal_coverage[hl][0.90] for hl in HALF_LIVES]

fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
fig2.patch.set_facecolor(BG)

panel_data = [
    (axes[0, 0], rmse_vals,  "RMSE  (pts/100)",         "lower = better"),
    (axes[0, 1], mae_vals,   "MAE  (pts/100)",           "lower = better"),
    (axes[1, 0], bias_vals,  "Mean bias  (pts/100)",     "target = 0"),
    (axes[1, 1], cov90_vals, "Empirical 90% PI coverage","target = 0.90"),
]

for ax, vals, ylabel, subtitle in panel_data:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", axis="y")

    colors = ["white"] + [
        cmap_lines(i / max(len(numeric_hls) - 1, 1))
        for i in range(len(numeric_hls))
    ]
    bars = ax.bar(x_pos, vals, color=colors, alpha=0.85, width=0.6)

    # Highlight minimum (for RMSE/MAE) or closest to target
    if "bias" in ylabel.lower():
        best_idx = int(np.argmin(np.abs(vals)))
        target_line = 0.0
        ax.axhline(target_line, color="#555577", lw=1.0, linestyle=":")
    elif "coverage" in ylabel.lower():
        best_idx = int(np.argmin(np.abs(np.array(vals) - 0.90)))
        ax.axhline(0.90, color="#555577", lw=1.0, linestyle=":", label="nominal 90%")
    else:
        best_idx = int(np.argmin(vals))
    bars[best_idx].set_edgecolor("#a8ff78")
    bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, color=TICK, fontsize=8)
    ax.set_ylabel(ylabel, color=TICK, fontsize=9)
    ax.set_title(subtitle, color="#aaaacc", fontsize=8)
    ax.yaxis.label.set_color(TICK)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom",
                color=TICK, fontsize=7.5)

fig2.suptitle(
    f"Recency half-life grid search  ·  Model 2 (ridge)  ·  2025-26\n"
    f"green border = best value  ·  white bar = static baseline",
    color="white", fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(SUMMARY_OUT, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved summary → {SUMMARY_OUT}")
