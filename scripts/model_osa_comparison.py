"""
Three-way one-step-ahead (OSA) comparison.

Models compared
---------------
  M2_static   : Model 2 ridge regression, no recency weighting (baseline)
  M2_hl60     : Model 2 with exponential recency, half-life = 60 days
                (best from recency_half_life_search.py)
  M4_kalman   : Model 4 weekly Kalman filter, fixed hyperparams

Methodology
-----------
For each 7-day prediction window W (after ≥ MIN_TRAIN training games):
  1. Fit every model on all games BEFORE week W.
  2. Predict offensive efficiency (pts / 100 poss) for games IN week W.
  3. Record RMSE, MAE, bias.

Model 4 hyperparameter strategy
--------------------------------
Optimising τ_o, τ_d, σ, μ, η by L-BFGS-B inside every window would
dominate runtime.  Instead we tune once on the first window large enough
for a stable fit (≥ MIN_HYPEROPT_TRAIN games), then reuse those fixed
values for all subsequent windows via optimize_hyperparams=False.
This mimics realistic deployment: hyperparams are estimated offline on a
held-out historical season, then locked during live prediction.

Outputs
-------
  osa_comparison_curves.png    — weekly RMSE and bias for all three models
  osa_comparison_summary.png   — aggregate bar chart (RMSE / MAE / bias / calib)
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
from models.model4 import Model4
from models.eval import recency_weights, temporal_split, conformal_calibration_scores

# ── Config ─────────────────────────────────────────────────────────────────────
SEASON            = 2026
WEEK_STEP         = 7
MIN_TRAIN          = 300   # minimum training games before first prediction window
MIN_HYPEROPT_TRAIN = 2000  # minimum games before Model 4 hyperparameter tuning
                           # (need enough data for τ to be well-identified with ~700 params)
RECENCY_HL        = 60     # days — best half-life from grid search
CURVES_OUT        = Path(__file__).parent / "osa_comparison_curves.png"
SUMMARY_OUT       = Path(__file__).parent / "osa_comparison_summary.png"

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

# ── Build weekly windows ───────────────────────────────────────────────────────
windows = []
d = season_start
while d <= season_end:
    d_next   = d + timedelta(days=WEEK_STEP)
    train    = {g for g in all_gids if game_dt[g] < d}
    pred     = {g for g in all_gids if d <= game_dt[g] < d_next}
    if len(train) >= MIN_TRAIN and pred:
        windows.append((d, train, pred))
    d = d_next

print(f"Season {SEASON}: {len(all_gids)} games, {len(windows)} windows")

# ── Tune Model 4 hyperparameters by cross-validating τ ────────────────────────
# MLE-optimal τ (from marginal log-likelihood) tends to be large because it
# conflates irreducible game noise with true team drift.  Instead we:
#   1. Find the first window with ≥ MIN_HYPEROPT_TRAIN training games.
#   2. Split that training window 80/20 (inner split).
#   3. Grid-search τ ∈ TAU_GRID, fit Model4 on the 80% inner train,
#      evaluate RMSE on the 20% inner val.
#   4. Pick the τ with the lowest val RMSE.
# σ, μ, η are taken from Model2 fit (held fixed during the τ search).

TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

print(f"\nTuning Model 4 τ (CV over {TAU_GRID}) "
      f"— need ≥ {MIN_HYPEROPT_TRAIN} training games...")
m4_tau_o, m4_tau_d = None, None

for tune_dt, tune_train, _ in windows:
    if len(tune_train) >= MIN_HYPEROPT_TRAIN:
        tune_rows = [r for g in tune_train for r in row_by_gid[g]]

        # Inner 80/20 temporal split for CV
        inner_gids = sorted(tune_train)
        n_inner_tr = int(0.80 * len(inner_gids))
        inner_tr_gids = set(inner_gids[:n_inner_tr])
        inner_va_gids = set(inner_gids[n_inner_tr:])
        inner_tr = [r for g in inner_tr_gids for r in row_by_gid[g]]
        inner_va = [r for g in inner_va_gids for r in row_by_gid[g]]
        inner_actual = np.array([r.pts / r.poss * 100.0 for r in inner_va])

        # Get σ, μ, η from a Model2 fit (held fixed during τ search)
        m2_ref = Model2(lambda_team=100.0, lambda_pace=50.0)
        m2_ref.fit_rows(inner_tr, SEASON)
        N_ref  = len(m2_ref.teams_)
        sigma0 = float(np.sqrt(m2_ref._sigma2_eff))
        mu0    = float(m2_ref.theta_hat_[0])
        eta0   = float(m2_ref.theta_hat_[2 * N_ref + 1])

        best_tau, best_rmse = None, float("inf")
        print(f"  CV on {tune_dt:%b %d} window  "
              f"({len(inner_tr_gids)} inner-train / {len(inner_va_gids)} inner-val games)")
        for tau in TAU_GRID:
            m4_cv = Model4(
                lambda_team=100.0, lambda_pace=50.0,
                tau_o=tau, tau_d=tau,
                optimize_hyperparams=False,
                week_step=WEEK_STEP,
            )
            m4_cv.fit_rows(inner_tr, SEASON, game_dates=game_dt)
            pred_cv = m4_cv.predict_efficiency(inner_va)
            rmse_cv = float(np.sqrt(np.mean((pred_cv - inner_actual) ** 2)))
            flag = ""
            if rmse_cv < best_rmse:
                best_rmse = rmse_cv
                best_tau  = tau
                flag = "  ←"
            print(f"    τ={tau:.2f}  val_RMSE={rmse_cv:.3f}{flag}")

        m4_tau_o = m4_tau_d = best_tau
        print(f"  Selected τ = {m4_tau_o}  (val RMSE = {best_rmse:.3f})\n")
        break

if m4_tau_o is None:
    print(f"  WARNING: never reached {MIN_HYPEROPT_TRAIN} games — "
          "using default τ=1.0")
    m4_tau_o = m4_tau_d = 1.0

# ── Rolling OSA evaluation ─────────────────────────────────────────────────────
# results[key] = [(date, rmse, mae, bias), ...]
MODELS = ["M2_static", "M2_hl60", "M4_kalman"]
results: dict[str, list] = {k: [] for k in MODELS}

for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
    actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    # ── M2 static ────────────────────────────────────────────────────────
    m2s = Model2()
    m2s.fit_rows(train_rows, SEASON)

    # ── M2 hl=60 ─────────────────────────────────────────────────────────
    m2r = Model2()
    sw  = recency_weights(train_rows, game_dt, win_dt, half_life_days=RECENCY_HL)
    m2r.fit_rows(train_rows, SEASON, sample_weight=sw)

    # ── M4 Kalman (fixed hyperparams) ─────────────────────────────────────
    m4 = Model4(
        lambda_team=100.0,
        lambda_pace=50.0,
        tau_o=m4_tau_o,
        tau_d=m4_tau_d,
        optimize_hyperparams=False,
        week_step=WEEK_STEP,
    )
    m4.fit_rows(train_rows, SEASON, game_dates=game_dt)

    for label, model in [("M2_static", m2s), ("M2_hl60", m2r), ("M4_kalman", m4)]:
        pred  = model.predict_efficiency(test_rows)
        resid = pred - actual
        results[label].append((
            win_dt,
            float(np.sqrt(np.mean(resid ** 2))),
            float(np.mean(np.abs(resid))),
            float(np.mean(resid)),
        ))

    if w_idx % 4 == 0 or w_idx == len(windows) - 1:
        static_rmse = results["M2_static"][-1][1]
        hl60_rmse   = results["M2_hl60"][-1][1]
        kal_rmse    = results["M4_kalman"][-1][1]
        print(
            f"  {w_idx+1:>2}/{len(windows)}  {win_dt:%b %d}  "
            f"train={len(train_gids):>4}  "
            f"static={static_rmse:.2f}  "
            f"hl60={hl60_rmse:.2f}  "
            f"kalman={kal_rmse:.2f}",
            flush=True,
        )

# ── Aggregate stats ────────────────────────────────────────────────────────────
print(f"\n{'Model':>12}  {'RMSE':>7}  {'MAE':>7}  {'Bias':>7}")
print("-" * 46)
agg: dict[str, tuple] = {}
for label in MODELS:
    pts      = results[label]
    rmse_m   = float(np.mean([p[1] for p in pts]))
    mae_m    = float(np.mean([p[2] for p in pts]))
    bias_m   = float(np.mean([p[3] for p in pts]))
    agg[label] = (rmse_m, mae_m, bias_m)
    flag = "  ← baseline" if label == "M2_static" else ""
    print(f"  {label:>12}  {rmse_m:>7.3f}  {mae_m:>7.3f}  {bias_m:>+7.3f}{flag}")

# ── Calibration on 80/20 season split ─────────────────────────────────────────
print("\nCalibration (80/20 temporal split, conformal z-scores):")
train_cal, test_cal = temporal_split(all_rows, 0.80)
cutoff_cal = max(game_dt[r.game_id] for r in train_cal)
cal_cov: dict[str, dict] = {}

models_cal = {
    "M2_static": Model2(),
    "M2_hl60":   Model2(),
    "M4_kalman": Model4(
        lambda_team=100.0, lambda_pace=50.0,
        tau_o=m4_tau_o, tau_d=m4_tau_d,
        optimize_hyperparams=False, week_step=WEEK_STEP,
    ),
}
models_cal["M2_static"].fit_rows(train_cal, SEASON)
sw_cal = recency_weights(train_cal, game_dt, cutoff_cal, half_life_days=RECENCY_HL)
models_cal["M2_hl60"].fit_rows(train_cal, SEASON, sample_weight=sw_cal)
models_cal["M4_kalman"].fit_rows(train_cal, SEASON, game_dates=game_dt)

for label, model in models_cal.items():
    z, cov = conformal_calibration_scores(
        model, test_cal, n_draws=200, rng=np.random.default_rng(0)
    )
    cal_cov[label] = cov
    print(f"  {label:>12}  z_std={z.std():.3f}  "
          f"cov@90%={cov[0.90]:.1%}  cov@95%={cov[0.95]:.1%}")

# ── Figure 1: OSA curves ───────────────────────────────────────────────────────
BG, GRID, SPINE, TICK = "#0f0f1a", "#1a1a3a", "#333355", "#aaaacc"
C_STATIC = "#888899"     # grey   — static baseline
C_HL60   = "#ff9f1c"     # amber  — recency
C_KAL    = "#4cc9f0"     # cyan   — Kalman

fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.patch.set_facecolor(BG)
for ax in (ax_rmse, ax_bias):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")

plot_cfg = [
    ("M2_static", C_STATIC, "--", 1.6, "M2  static (baseline)"),
    ("M2_hl60",   C_HL60,   "-",  1.8, f"M2  recency hl={RECENCY_HL}d"),
    ("M4_kalman", C_KAL,    "-",  2.2, "M4  Kalman"),
]

for key, color, ls, lw, label in plot_cfg:
    pts   = results[key]
    dates = [p[0] for p in pts]
    rmses = [p[1] for p in pts]
    biases = [p[3] for p in pts]
    ax_rmse.plot(dates, rmses,  color=color, lw=lw, ls=ls, label=label)
    ax_bias.plot(dates, biases, color=color, lw=lw, ls=ls, label=label)

ax_rmse.set_ylabel("Weekly RMSE  (pts/100)", color=TICK)
ax_rmse.set_title(
    f"One-step-ahead RMSE  ·  2025-26  ·  three models",
    color="white", fontsize=11, fontweight="bold",
)
ax_rmse.legend(loc="upper right", labelcolor="white", facecolor="#111133",
               edgecolor=SPINE, fontsize=9)

ax_bias.axhline(0, color="#555577", lw=0.9, linestyle=":")
ax_bias.set_ylabel("Weekly bias  (pred − actual,  pts/100)", color=TICK)
ax_bias.set_title("Bias", color="white", fontsize=11, fontweight="bold")
ax_bias.legend(loc="upper right", labelcolor="white", facecolor="#111133",
               edgecolor=SPINE, fontsize=9)
ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_bias.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved curves → {CURVES_OUT}")

# ── Figure 2: Summary bar chart ────────────────────────────────────────────────
x_labels = ["M2\nstatic", f"M2\nhl={RECENCY_HL}d", "M4\nKalman"]
x_pos    = np.arange(3)
bar_colors = [C_STATIC, C_HL60, C_KAL]

rmse_vals  = [agg[k][0] for k in MODELS]
mae_vals   = [agg[k][1] for k in MODELS]
bias_vals  = [agg[k][2] for k in MODELS]
cov90_vals = [cal_cov[k][0.90] for k in MODELS]

fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.patch.set_facecolor(BG)

panel_data = [
    (axes[0, 0], rmse_vals,  "RMSE  (pts/100)",           "lower = better"),
    (axes[0, 1], mae_vals,   "MAE  (pts/100)",             "lower = better"),
    (axes[1, 0], bias_vals,  "Mean bias  (pts/100)",       "target = 0"),
    (axes[1, 1], cov90_vals, "Empirical 90% PI coverage",  "target = 0.90"),
]

for ax, vals, ylabel, subtitle in panel_data:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", axis="y")

    bars = ax.bar(x_pos, vals, color=bar_colors, alpha=0.85, width=0.5)

    if "bias" in ylabel.lower():
        best_idx = int(np.argmin(np.abs(vals)))
        ax.axhline(0.0, color="#555577", lw=1.0, linestyle=":")
    elif "coverage" in ylabel.lower():
        best_idx = int(np.argmin(np.abs(np.array(vals) - 0.90)))
        ax.axhline(0.90, color="#555577", lw=1.0, linestyle=":")
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
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom", color=TICK, fontsize=8,
        )

fig2.suptitle(
    f"OSA model comparison  ·  2025-26\n"
    f"green border = best  ·  "
    f"M4 τ_o={m4_tau_o:.2f}  τ_d={m4_tau_d:.2f}",
    color="white", fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(SUMMARY_OUT, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved summary → {SUMMARY_OUT}")
