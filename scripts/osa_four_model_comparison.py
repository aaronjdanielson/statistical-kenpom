"""
Four-model one-step-ahead comparison.

M1 (KenPom fixed-point), M2 static, M2 hl=60, M4 Kalman — evaluated on the same
20 weekly windows so all four appear on the same axes.

Outputs
-------
  osa_four_model_curves.png    — weekly RMSE and bias
  osa_four_model_summary.png   — aggregate bar chart (RMSE / MAE / bias)
"""
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games, parse_date, GameRow
from models.model1 import Model1
from models.model2 import Model2
from models.model4 import Model4
from models.eval import recency_weights

# ── Config ─────────────────────────────────────────────────────────────────────
SEASON             = 2026
WEEK_STEP          = 7
MIN_TRAIN          = 300
MIN_HYPEROPT_TRAIN = 2000
RECENCY_HL         = 60
TAU_GRID           = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
CURVES_OUT         = Path(__file__).parent / "osa_four_model_curves.png"
SUMMARY_OUT        = Path(__file__).parent / "osa_four_model_summary.png"

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

season_start = min(game_dt.values())
season_end   = max(game_dt.values())

# ── Build windows ──────────────────────────────────────────────────────────────
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

# ── CV-tune Model 4 τ ─────────────────────────────────────────────────────────
m4_tau = 1.0   # default if window never reached
for tune_dt, tune_train, _ in windows:
    if len(tune_train) >= MIN_HYPEROPT_TRAIN:
        inner_gids  = sorted(tune_train)
        n_tr        = int(0.80 * len(inner_gids))
        inner_tr    = [r for g in inner_gids[:n_tr]  for r in row_by_gid[g]]
        inner_va    = [r for g in inner_gids[n_tr:]  for r in row_by_gid[g]]
        actual_va   = np.array([r.pts / r.poss * 100.0 for r in inner_va])

        print(f"\nTuning τ on {tune_dt:%b %d} "
              f"({n_tr} inner-train / {len(inner_gids)-n_tr} inner-val):")
        best_tau, best_rmse = 1.0, float("inf")
        for tau in TAU_GRID:
            m = Model4(lambda_team=100.0, lambda_pace=50.0,
                       tau_o=tau, tau_d=tau,
                       optimize_hyperparams=False, week_step=WEEK_STEP)
            m.fit_rows(inner_tr, SEASON, game_dates=game_dt)
            pred_va  = m.predict_efficiency(inner_va)
            rmse_cv  = float(np.sqrt(np.mean((pred_va - actual_va) ** 2)))
            flag = "  ←" if rmse_cv < best_rmse else ""
            if rmse_cv < best_rmse:
                best_rmse = rmse_cv
                best_tau  = tau
            print(f"  τ={tau:.2f}  val_RMSE={rmse_cv:.3f}{flag}")

        m4_tau = best_tau
        print(f"  Selected τ = {m4_tau}\n")
        break

# ── Rolling OSA evaluation ─────────────────────────────────────────────────────
MODELS = ["M1", "M2_static", "M2_hl60", "M4_kalman"]
results: dict[str, list] = {k: [] for k in MODELS}

print(f"{'Win':>3}  {'Date':>6}  {'Train':>5}  "
      f"{'M1':>6}  {'M2':>6}  {'M2hl':>6}  {'M4':>6}")
print("-" * 52)

for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
    actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    m1 = Model1()
    m1.fit_rows(train_rows, SEASON)

    m2s = Model2()
    m2s.fit_rows(train_rows, SEASON)

    sw  = recency_weights(train_rows, game_dt, win_dt, half_life_days=RECENCY_HL)
    m2r = Model2()
    m2r.fit_rows(train_rows, SEASON, sample_weight=sw)

    m4 = Model4(lambda_team=100.0, lambda_pace=50.0,
                tau_o=m4_tau, tau_d=m4_tau,
                optimize_hyperparams=False, week_step=WEEK_STEP)
    m4.fit_rows(train_rows, SEASON, game_dates=game_dt)

    for label, model in [("M1", m1), ("M2_static", m2s),
                         ("M2_hl60", m2r), ("M4_kalman", m4)]:
        resid = model.predict_efficiency(test_rows) - actual
        results[label].append((
            win_dt,
            float(np.sqrt(np.mean(resid ** 2))),
            float(np.mean(np.abs(resid))),
            float(np.mean(resid)),
        ))

    print(f"{w_idx+1:>3}  {win_dt:%b %d}  {len(train_gids):>5}  "
          f"{results['M1'][-1][1]:>6.2f}  "
          f"{results['M2_static'][-1][1]:>6.2f}  "
          f"{results['M2_hl60'][-1][1]:>6.2f}  "
          f"{results['M4_kalman'][-1][1]:>6.2f}", flush=True)

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n{'Model':>12}  {'RMSE':>7}  {'MAE':>7}  {'Bias':>7}")
print("-" * 40)
agg: dict[str, tuple] = {}
for label in MODELS:
    pts    = results[label]
    rmse_m = float(np.mean([p[1] for p in pts]))
    mae_m  = float(np.mean([p[2] for p in pts]))
    bias_m = float(np.mean([p[3] for p in pts]))
    agg[label] = (rmse_m, mae_m, bias_m)
    print(f"  {label:>12}  {rmse_m:>7.3f}  {mae_m:>7.3f}  {bias_m:>+7.3f}")

# ── Styling ────────────────────────────────────────────────────────────────────
BG    = "#0f0f1a"
GRID  = "#1a1a3a"
SPINE = "#333355"
TICK  = "#aaaacc"

C_M1     = "#e07070"   # red-ish — M1 baseline
C_STATIC = "#888899"   # grey    — M2 static
C_HL60   = "#ff9f1c"   # amber   — M2 recency
C_KAL    = "#4cc9f0"   # cyan    — M4 Kalman

plot_cfg = [
    ("M1",       C_M1,     "--", 1.4, "M1  fixed-point"),
    ("M2_static", C_STATIC, "--", 1.6, "M2  static"),
    ("M2_hl60",  C_HL60,   "-",  1.8, f"M2  recency hl={RECENCY_HL}d"),
    ("M4_kalman", C_KAL,   "-",  2.2, f"M4  Kalman  τ={m4_tau:.2f}"),
]

# ── Figure 1: curves ───────────────────────────────────────────────────────────
fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.patch.set_facecolor(BG)

for ax in (ax_rmse, ax_bias):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")

for key, color, ls, lw, label in plot_cfg:
    pts    = results[key]
    dates  = [p[0] for p in pts]
    rmses  = [p[1] for p in pts]
    biases = [p[3] for p in pts]
    ax_rmse.plot(dates, rmses,  color=color, lw=lw, ls=ls, label=label)
    ax_bias.plot(dates, biases, color=color, lw=lw, ls=ls, label=label)

ax_rmse.set_ylabel("Weekly RMSE  (pts/100)", color=TICK)
ax_rmse.set_title(
    "One-step-ahead RMSE  ·  2025-26  ·  four models",
    color="white", fontsize=11, fontweight="bold",
)
ax_rmse.legend(loc="upper right", labelcolor="white",
               facecolor="#111133", edgecolor=SPINE, fontsize=9)

ax_bias.axhline(0, color="#555577", lw=0.9, linestyle=":")
ax_bias.set_ylabel("Weekly bias  (pred − actual,  pts/100)", color=TICK)
ax_bias.set_title("Bias", color="white", fontsize=11, fontweight="bold")
ax_bias.legend(loc="upper right", labelcolor="white",
               facecolor="#111133", edgecolor=SPINE, fontsize=9)
ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_bias.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved curves → {CURVES_OUT}")

# ── Figure 2: summary bar chart ────────────────────────────────────────────────
x_labels   = ["M1\nfixed-pt", "M2\nstatic", f"M2\nhl={RECENCY_HL}d",
               f"M4\nKalman"]
bar_colors = [C_M1, C_STATIC, C_HL60, C_KAL]
x_pos      = np.arange(4)

rmse_vals = [agg[k][0] for k in MODELS]
mae_vals  = [agg[k][1] for k in MODELS]
bias_vals = [agg[k][2] for k in MODELS]

fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
fig2.patch.set_facecolor(BG)

panel_data = [
    (axes[0], rmse_vals, "RMSE  (pts/100)",      "lower = better"),
    (axes[1], mae_vals,  "MAE  (pts/100)",        "lower = better"),
    (axes[2], bias_vals, "Mean bias  (pts/100)", "target = 0"),
]

for ax, vals, ylabel, subtitle in panel_data:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", axis="y")

    bars = ax.bar(x_pos, vals, color=bar_colors, alpha=0.85, width=0.55)

    if "bias" in ylabel.lower():
        best_idx = int(np.argmin(np.abs(vals)))
        ax.axhline(0.0, color="#555577", lw=1.0, linestyle=":")
    else:
        best_idx = int(np.argmin(vals))

    bars[best_idx].set_edgecolor("#a8ff78")
    bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, color=TICK, fontsize=9)
    ax.set_ylabel(ylabel, color=TICK, fontsize=9)
    ax.set_title(subtitle, color=TICK, fontsize=8)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if val >= 0 else -0.04),
            f"{val:.3f}",
            ha="center", va="bottom", color=TICK, fontsize=8,
        )

fig2.suptitle(
    f"OSA model comparison  ·  2025-26  ·  all four models\n"
    f"green border = best  ·  M4 τ={m4_tau:.2f}/week",
    color="white", fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(SUMMARY_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
print(f"Saved summary → {SUMMARY_OUT}")
