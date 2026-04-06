"""
Rolling one-step-ahead evaluation: Model 1 vs Model 2.

For each week W in the season:
  1. Fit on all games BEFORE week W  (true out-of-sample)
  2. Predict offensive efficiency for games IN week W
  3. Collect residuals

This simulates actual deployment — the model only ever sees games
that already happened before making a prediction.

Produces two figures:
  (A) rolling_osa_scatter_YYYY.png   — predicted vs actual scatter,
        left = Model 1, right = Model 2, colored by residual magnitude
  (B) rolling_osa_rmse_YYYY.png      — RMSE / MAE / bias curves over
        the season for both models, with ±1σ shading
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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games, GameRow
from models.model1 import Model1
from models.model2 import Model2

# ── Config ────────────────────────────────────────────────────────────────────
SEASON      = 2026       # 2025-26 season
WEEK_STEP   = 7          # days per prediction window
MIN_TRAIN   = 300        # minimum cumulative games before first prediction
SCATTER_OUT = Path(__file__).parent / "rolling_osa_scatter_2025_26.png"
RMSE_OUT    = Path(__file__).parent / "rolling_osa_rmse_2025_26.png"

# ── Load data ─────────────────────────────────────────────────────────────────
conn = open_db()
cur  = conn.cursor()
cur.execute("""
    SELECT DISTINCT s.GameID, s.Date
    FROM schedules s JOIN boxscores b ON s.GameID = b.GameID
    WHERE s.Year = ?
    ORDER BY s.GameID
""", (SEASON,))
game_order = cur.fetchall()
conn.close()

def parse_date(s):
    for fmt in ("%b %d, %Y", "%b %d,%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(s)

game_dt = {gid: parse_date(ds) for gid, ds in game_order}
all_gids = [gid for gid, _ in game_order]

all_rows = load_season_games(open_db(), SEASON)
row_by_gid: dict[int, list[GameRow]] = defaultdict(list)
for r in all_rows:
    row_by_gid[r.game_id].append(r)

# ── Build weekly windows ──────────────────────────────────────────────────────
season_start = min(game_dt.values())
season_end   = max(game_dt.values())

windows = []   # (window_start_dt, train_gids, predict_gids)
d = season_start
while d <= season_end:
    d_next = d + timedelta(days=WEEK_STEP)
    train_gids   = {gid for gid in all_gids if game_dt[gid] < d}
    predict_gids = {gid for gid in all_gids if d <= game_dt[gid] < d_next}
    if len(train_gids) >= MIN_TRAIN and predict_gids:
        windows.append((d, train_gids, predict_gids))
    d = d_next

print(f"Season {SEASON}: {len(all_gids)} games, {len(windows)} prediction windows")

# ── Rolling evaluation ────────────────────────────────────────────────────────
# Per window: store (date, actual, pred1, pred2)
records = []   # (window_dt, actual_arr, pred1_arr, pred2_arr)

for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for gid in train_gids for r in row_by_gid[gid]]
    test_rows  = [r for gid in pred_gids  for r in row_by_gid[gid]]

    m1 = Model1(); m1.fit_rows(train_rows, SEASON)
    m2 = Model2(); m2.fit_rows(train_rows, SEASON)

    actual = np.array([r.pts / r.poss * 100.0 for r in test_rows])
    pred1  = m1.predict_efficiency(test_rows)
    pred2  = m2.predict_efficiency(test_rows)

    records.append((win_dt, actual, pred1, pred2))

    n_pred = len({r.game_id for r in test_rows})
    rmse1  = float(np.sqrt(np.mean((pred1 - actual) ** 2)))
    rmse2  = float(np.sqrt(np.mean((pred2 - actual) ** 2)))
    print(f"  {w_idx+1:>2}/{len(windows)}  {win_dt.strftime('%b %d')}  "
          f"train={len(train_gids):>4}  pred={n_pred:>3} games  "
          f"RMSE1={rmse1:.2f}  RMSE2={rmse2:.2f}", flush=True)

# ── Aggregate for scatter ─────────────────────────────────────────────────────
all_actual = np.concatenate([r[1] for r in records])
all_pred1  = np.concatenate([r[2] for r in records])
all_pred2  = np.concatenate([r[3] for r in records])

print(f"\nTotal test observations: {len(all_actual)}")
for name, pred in [("Model 1", all_pred1), ("Model 2", all_pred2)]:
    res = pred - all_actual
    print(f"  {name}  RMSE={np.sqrt(np.mean(res**2)):.3f}  "
          f"MAE={np.mean(np.abs(res)):.3f}  bias={np.mean(res):+.3f}")

# ── Figure A: Scatter ─────────────────────────────────────────────────────────
fig_s, axes = plt.subplots(1, 2, figsize=(15, 7))
fig_s.patch.set_facecolor("#0f0f1a")

for ax, (model_name, pred) in zip(axes, [
    ("Model 1  (fixed-point)", all_pred1),
    ("Model 2  (ridge)",       all_pred2),
]):
    ax.set_facecolor("#0f0f1a")
    resid  = pred - all_actual
    rmse   = float(np.sqrt(np.mean(resid ** 2)))
    mae    = float(np.mean(np.abs(resid)))
    bias   = float(np.mean(resid))
    absres = np.abs(resid)

    sc = ax.scatter(
        all_actual, pred,
        c=absres, cmap="plasma", vmin=0, vmax=30,
        s=4, alpha=0.45, linewidths=0, zorder=3,
    )
    lims = [60, 155]
    ax.plot(lims, lims, color="#88aaff", linewidth=1.2,
            linestyle="--", alpha=0.8, zorder=4)

    cb = fig_s.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("|error|  pts/100", color="#aaaacc", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#aaaacc", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaaacc")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual  (pts / 100 poss)", color="#aaaacc", fontsize=10)
    ax.set_ylabel("Predicted  (pts / 100 poss)", color="#aaaacc", fontsize=10)
    ax.set_title(
        f"2025-26  ·  {model_name}\n"
        f"one-step-ahead  ({len(windows)} weekly windows)",
        color="white", fontsize=10, fontweight="bold",
    )
    ax.text(
        0.03, 0.97,
        f"RMSE {rmse:.2f}   MAE {mae:.2f}   bias {bias:+.2f}",
        transform=ax.transAxes, color="#ccccff", fontsize=9,
        va="top", bbox=dict(fc="#111133", ec="none", alpha=0.75),
    )
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.grid(color="#1a1a3a", linewidth=0.5)

fig_s.suptitle(
    "One-step-ahead prediction  —  fit on all prior games, predict next week\n"
    "every dot is one team-game observation in the 2025-26 season",
    color="white", fontsize=11, fontweight="bold", y=1.02,
)
plt.tight_layout()
fig_s.savefig(SCATTER_OUT, dpi=150, bbox_inches="tight",
              facecolor=fig_s.get_facecolor())
print(f"\nSaved scatter → {SCATTER_OUT}")

# ── Figure B: RMSE / MAE / bias over season ───────────────────────────────────
dates  = [r[0] for r in records]
rmse1s = [float(np.sqrt(np.mean((r[2] - r[1]) ** 2))) for r in records]
rmse2s = [float(np.sqrt(np.mean((r[3] - r[1]) ** 2))) for r in records]
mae1s  = [float(np.mean(np.abs(r[2] - r[1]))) for r in records]
mae2s  = [float(np.mean(np.abs(r[3] - r[1]))) for r in records]
bias1s = [float(np.mean(r[2] - r[1])) for r in records]
bias2s = [float(np.mean(r[3] - r[1])) for r in records]

# Cumulative RMSE (all predictions up to each window)
cum_rmse1, cum_rmse2 = [], []
all_r1, all_r2 = [], []
for r in records:
    all_r1.extend((r[2] - r[1]).tolist())
    all_r2.extend((r[3] - r[1]).tolist())
    cum_rmse1.append(float(np.sqrt(np.mean(np.array(all_r1) ** 2))))
    cum_rmse2.append(float(np.sqrt(np.mean(np.array(all_r2) ** 2))))

C1 = "#00bfff"   # Model 1 — sky blue
C2 = "#ff6b35"   # Model 2 — orange

fig_r = plt.figure(figsize=(14, 10))
fig_r.patch.set_facecolor("#0f0f1a")
gs = GridSpec(3, 1, figure=fig_r, hspace=0.45)

# Panel 1: weekly RMSE
ax1 = fig_r.add_subplot(gs[0])
ax1.set_facecolor("#0f0f1a")
ax1.plot(dates, rmse1s, color=C1, linewidth=1.8, label="Model 1 (fixed-point)", zorder=3)
ax1.plot(dates, rmse2s, color=C2, linewidth=1.8, label="Model 2 (ridge)", zorder=3)
ax1.fill_between(dates, rmse1s, rmse2s,
                 where=[r2 < r1 for r1, r2 in zip(rmse1s, rmse2s)],
                 color=C2, alpha=0.15, label="Model 2 better")
ax1.fill_between(dates, rmse1s, rmse2s,
                 where=[r1 < r2 for r1, r2 in zip(rmse1s, rmse2s)],
                 color=C1, alpha=0.15, label="Model 1 better")
ax1.set_ylabel("Weekly RMSE  (pts/100)", color="#aaaacc", fontsize=9)
ax1.set_title("Weekly RMSE  —  one-step-ahead predictions", color="white",
              fontsize=10, fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.3, fontsize=8,
           labelcolor="white", facecolor="#111133", edgecolor="#333355")

# Panel 2: cumulative RMSE (converges to season total)
ax2 = fig_r.add_subplot(gs[1])
ax2.set_facecolor("#0f0f1a")
ax2.plot(dates, cum_rmse1, color=C1, linewidth=2.2,
         label="Model 1  cumulative", zorder=3)
ax2.plot(dates, cum_rmse2, color=C2, linewidth=2.2,
         label="Model 2  cumulative", zorder=3)
ax2.set_ylabel("Cumulative RMSE  (pts/100)", color="#aaaacc", fontsize=9)
ax2.set_title("Cumulative RMSE  —  all one-step-ahead predictions to date",
              color="white", fontsize=10, fontweight="bold")
ax2.legend(loc="upper right", framealpha=0.3, fontsize=8,
           labelcolor="white", facecolor="#111133", edgecolor="#333355")

# Panel 3: bias over season
ax3 = fig_r.add_subplot(gs[2])
ax3.set_facecolor("#0f0f1a")
ax3.plot(dates, bias1s, color=C1, linewidth=1.8,
         label="Model 1 bias  (pred − actual)", zorder=3)
ax3.plot(dates, bias2s, color=C2, linewidth=1.8,
         label="Model 2 bias", zorder=3)
ax3.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
ax3.set_ylabel("Bias  (pts/100)", color="#aaaacc", fontsize=9)
ax3.set_title("Weekly bias  —  positive = over-prediction",
              color="white", fontsize=10, fontweight="bold")
ax3.legend(loc="upper right", framealpha=0.3, fontsize=8,
           labelcolor="white", facecolor="#111133", edgecolor="#333355")

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.grid(color="#1a1a3a", linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color("#333355")

fig_r.suptitle(
    "2025-26  One-step-ahead forecast quality  —  Model 1 vs Model 2",
    color="white", fontsize=12, fontweight="bold", y=1.01,
)
plt.savefig(RMSE_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig_r.get_facecolor())
print(f"Saved RMSE curves → {RMSE_OUT}")
