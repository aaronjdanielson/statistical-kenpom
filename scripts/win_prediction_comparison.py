"""
Four-model game win/loss prediction comparison.

For each weekly prediction window:
  1. Fit all four models on games BEFORE the window.
  2. For each game IN the window, predict the winner.
  3. Track accuracy (did we pick the right team?), win probability, and Brier score.

Win probability is computed from the Gaussian margin model:
  P(A wins) = Φ( (pred_eff_A − pred_eff_B) / (√2 · σ_eff) )
where σ_eff is the model's residual std (pts/100).  This marginalises out
possession count because the poss/100 factor cancels in numerator and denominator.

Outputs
-------
  win_pred_curves.png    — per-window accuracy and Brier score curves
  win_pred_summary.png   — aggregate bar chart
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
from scipy.stats import norm as _norm

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
CURVES_OUT         = Path(__file__).parent / "win_pred_curves.png"
SUMMARY_OUT        = Path(__file__).parent / "win_pred_summary.png"

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

# ── CV-tune Model 4 τ (same grid as OSA comparison) ──────────────────────────
m4_tau = 0.25   # default; grid search below updates this
for tune_dt, tune_train, _ in windows:
    if len(tune_train) >= MIN_HYPEROPT_TRAIN:
        inner_gids = sorted(tune_train)
        n_tr       = int(0.80 * len(inner_gids))
        inner_tr   = [r for g in inner_gids[:n_tr] for r in row_by_gid[g]]
        inner_va   = [r for g in inner_gids[n_tr:]  for r in row_by_gid[g]]
        actual_va  = np.array([r.pts / r.poss * 100.0 for r in inner_va])

        best_tau, best_rmse = m4_tau, float("inf")
        print(f"CV-tuning τ on {tune_dt:%b %d} window:")
        for tau in TAU_GRID:
            m = Model4(lambda_team=100.0, lambda_pace=50.0,
                       tau_o=tau, tau_d=tau,
                       optimize_hyperparams=False, week_step=WEEK_STEP)
            m.fit_rows(inner_tr, SEASON, game_dates=game_dt)
            rmse_cv = float(np.sqrt(np.mean(
                (m.predict_efficiency(inner_va) - actual_va) ** 2)))
            flag = "  ←" if rmse_cv < best_rmse else ""
            if rmse_cv < best_rmse:
                best_rmse = rmse_cv
                best_tau  = tau
            print(f"  τ={tau:.2f}  val_RMSE={rmse_cv:.3f}{flag}")

        m4_tau = best_tau
        print(f"  Selected τ = {m4_tau}\n")
        break

# ── Win-prediction helpers ─────────────────────────────────────────────────────

def pair_games(game_ids, row_by_gid):
    """Return list of (row_A, row_B) pairs for each complete game.

    row_A is arbitrarily the first row; row_B is the second.
    Both are required for a valid pairing — skip games with ≠ 2 rows.
    """
    pairs = []
    for gid in game_ids:
        rows = row_by_gid[gid]
        if len(rows) == 2:
            pairs.append((rows[0], rows[1]))
    return pairs


def win_metrics(model, game_pairs):
    """Compute accuracy, Brier score, and log-loss for a fitted model.

    For each game pair (row_A, row_B):
      - pred_A = expected pts/100 for row_A.team_id vs row_A.opp_id
      - pred_B = expected pts/100 for row_B.team_id vs row_B.opp_id
      - margin = pred_A − pred_B  (pts/100)
      - p_win_A = Φ(margin / (√2 · σ_eff))
      - actual: A wins if row_A.pts > row_B.pts

    Returns (accuracy, brier, logloss, n_games).
    """
    sigma_eff = float(np.sqrt(getattr(model, "_sigma2_eff", 196.0)))
    denom = np.sqrt(2.0) * sigma_eff   # std of margin in pts/100 units

    row_As = [p[0] for p in game_pairs]
    row_Bs = [p[1] for p in game_pairs]

    pred_A = model.predict_efficiency(row_As)
    pred_B = model.predict_efficiency(row_Bs)

    p_win = _norm.cdf((pred_A - pred_B) / denom)   # P(row_A's team wins)
    actual_win = np.array([
        1 if a.pts > b.pts else 0
        for a, b in game_pairs
    ], dtype=float)

    # Skip ties (very rare in basketball)
    tied = np.array([a.pts == b.pts for a, b in game_pairs])
    mask = ~tied
    p_win    = p_win[mask]
    actual_win = actual_win[mask]

    acc    = float(np.mean((p_win >= 0.5) == (actual_win == 1)))
    brier  = float(np.mean((p_win - actual_win) ** 2))
    p_clip = np.clip(p_win, 1e-7, 1 - 1e-7)
    logloss = float(-np.mean(
        actual_win * np.log(p_clip) + (1 - actual_win) * np.log(1 - p_clip)
    ))
    return acc, brier, logloss, int(mask.sum())


# ── Rolling win-prediction evaluation ─────────────────────────────────────────
MODELS = ["M1", "M2_static", "M2_hl60", "M4_kalman"]
results: dict[str, list] = {k: [] for k in MODELS}

print(f"\n{'Win':>3}  {'Date':>6}  {'Train':>5}  {'Games':>5}  "
      f"{'M1 Acc':>7}  {'M2 Acc':>7}  {'M2hl Acc':>8}  {'M4 Acc':>7}")
print("-" * 62)

for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    pairs      = pair_games(sorted(pred_gids), row_by_gid)
    if not pairs:
        continue

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
        acc, brier, logloss, n = win_metrics(model, pairs)
        results[label].append((win_dt, acc, brier, logloss, n))

    print(f"{w_idx+1:>3}  {win_dt:%b %d}  {len(train_gids):>5}  "
          f"{results['M1'][-1][4]:>5}  "
          f"{results['M1'][-1][1]:>7.1%}  "
          f"{results['M2_static'][-1][1]:>7.1%}  "
          f"{results['M2_hl60'][-1][1]:>8.1%}  "
          f"{results['M4_kalman'][-1][1]:>7.1%}", flush=True)

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n{'Model':>12}  {'Accuracy':>9}  {'Brier':>7}  {'LogLoss':>8}  {'Games':>6}")
print("-" * 50)
agg: dict[str, dict] = {}
for label in MODELS:
    pts   = results[label]
    # Weighted mean across windows by number of games
    ns     = np.array([p[4] for p in pts], dtype=float)
    accs   = np.array([p[1] for p in pts])
    briers = np.array([p[2] for p in pts])
    lls    = np.array([p[3] for p in pts])
    total  = ns.sum()
    acc_w  = float(np.average(accs,   weights=ns))
    brier_w= float(np.average(briers, weights=ns))
    ll_w   = float(np.average(lls,    weights=ns))
    agg[label] = dict(acc=acc_w, brier=brier_w, logloss=ll_w, n=int(total))
    print(f"  {label:>12}  {acc_w:>9.2%}  {brier_w:>7.4f}  {ll_w:>8.4f}  {total:>6.0f}")

# ── Styling ────────────────────────────────────────────────────────────────────
BG    = "#0f0f1a"
GRID  = "#1a1a3a"
SPINE = "#333355"
TICK  = "#aaaacc"
C_M1     = "#e07070"
C_STATIC = "#888899"
C_HL60   = "#ff9f1c"
C_KAL    = "#4cc9f0"

plot_cfg = [
    ("M1",        C_M1,     "--", 1.4, "M1  fixed-point"),
    ("M2_static", C_STATIC, "--", 1.6, "M2  static"),
    ("M2_hl60",   C_HL60,   "-",  1.8, f"M2  recency hl={RECENCY_HL}d"),
    ("M4_kalman", C_KAL,    "-",  2.2, f"M4  Kalman  τ={m4_tau:.2f}"),
]

# ── Figure 1: curves ───────────────────────────────────────────────────────────
fig, (ax_acc, ax_brier) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.patch.set_facecolor(BG)

for ax in (ax_acc, ax_brier):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")

for key, color, ls, lw, label in plot_cfg:
    pts    = results[key]
    dates  = [p[0] for p in pts]
    accs   = [p[1] for p in pts]
    briers = [p[2] for p in pts]
    ax_acc.plot(dates,   accs,   color=color, lw=lw, ls=ls, label=label)
    ax_brier.plot(dates, briers, color=color, lw=lw, ls=ls, label=label)

# Naive 50% baseline on accuracy
ax_acc.axhline(0.5, color="#555577", lw=0.9, linestyle=":")
ax_acc.set_ylabel("Win prediction accuracy", color=TICK)
ax_acc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax_acc.set_title(
    "Game win/loss accuracy  ·  2025-26  ·  four models",
    color="white", fontsize=11, fontweight="bold",
)
ax_acc.legend(loc="lower right", labelcolor="white",
              facecolor="#111133", edgecolor=SPINE, fontsize=9)

ax_brier.set_ylabel("Brier score  (lower = better)", color=TICK)
ax_brier.set_title("Brier score", color="white", fontsize=11, fontweight="bold")
ax_brier.legend(loc="upper right", labelcolor="white",
                facecolor="#111133", edgecolor=SPINE, fontsize=9)
ax_brier.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_brier.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved curves → {CURVES_OUT}")

# ── Figure 2: summary bar chart ────────────────────────────────────────────────
x_labels   = ["M1\nfixed-pt", "M2\nstatic", f"M2\nhl={RECENCY_HL}d",
               f"M4\nKalman"]
bar_colors = [C_M1, C_STATIC, C_HL60, C_KAL]
x_pos      = np.arange(4)

acc_vals    = [agg[k]["acc"]     for k in MODELS]
brier_vals  = [agg[k]["brier"]   for k in MODELS]
ll_vals     = [agg[k]["logloss"] for k in MODELS]

fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
fig2.patch.set_facecolor(BG)

panel_data = [
    (axes[0], acc_vals,   "Win accuracy",      "higher = better", True),
    (axes[1], brier_vals, "Brier score",        "lower = better",  False),
    (axes[2], ll_vals,    "Log-loss",           "lower = better",  False),
]

for ax, vals, ylabel, subtitle, higher_better in panel_data:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", axis="y")

    bars = ax.bar(x_pos, vals, color=bar_colors, alpha=0.85, width=0.55)
    best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor("#a8ff78")
    bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, color=TICK, fontsize=9)
    ax.set_ylabel(ylabel, color=TICK, fontsize=9)
    ax.set_title(subtitle, color=TICK, fontsize=8)

    for bar, val in zip(bars, vals):
        fmt = f"{val:.1%}" if "accuracy" in ylabel.lower() else f"{val:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003,
            fmt,
            ha="center", va="bottom", color=TICK, fontsize=8,
        )

    if "accuracy" in ylabel.lower():
        ax.axhline(0.5, color="#555577", lw=0.9, linestyle=":",
                   label="random baseline")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

fig2.suptitle(
    f"Win/loss prediction  ·  2025-26  ·  all four models\n"
    f"green border = best  ·  M4 τ={m4_tau:.2f}/week",
    color="white", fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(SUMMARY_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
print(f"Saved summary → {SUMMARY_OUT}")
