"""
Scatter plots comparing Model 1 vs Model 2 predictive accuracy.

Two panels per season:
  Left  — Model 1 (fixed-point)   predicted vs actual e_off
  Right — Model 2 (ridge)         predicted vs actual e_off

Color = residual magnitude; diagonal = perfect prediction.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games
from models.eval import temporal_split
from models.model1 import Model1
from models.model2 import Model2

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS    = [2023, 2024, 2025]
TRAIN_FRAC = 0.80
OUT_PATH   = Path(__file__).parent / "scatter_model1_vs_model2.png"

conn = open_db()

fig = plt.figure(figsize=(15, 5 * len(SEASONS)))
fig.patch.set_facecolor("#0f0f1a")
gs  = GridSpec(len(SEASONS), 2, figure=fig, hspace=0.42, wspace=0.28)

for row, season in enumerate(SEASONS):
    rows = load_season_games(conn, season)
    train_rows, test_rows = temporal_split(rows, TRAIN_FRAC)
    actual = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    m1 = Model1(); m1.fit_rows(train_rows, season)
    m2 = Model2(); m2.fit_rows(train_rows, season)

    pred1 = m1.predict_efficiency(test_rows)
    pred2 = m2.predict_efficiency(test_rows)

    n_train = len({r.game_id for r in train_rows})
    n_test  = len({r.game_id for r in test_rows})

    for col, (model_name, pred) in enumerate([("Model 1  (fixed-point)", pred1),
                                               ("Model 2  (ridge)", pred2)]):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#0f0f1a")

        resid  = pred - actual
        rmse   = float(np.sqrt(np.mean(resid ** 2)))
        mae    = float(np.mean(np.abs(resid)))
        bias   = float(np.mean(resid))
        absres = np.abs(resid)

        sc = ax.scatter(
            actual, pred,
            c=absres, cmap="plasma", vmin=0, vmax=30,
            s=5, alpha=0.55, linewidths=0, zorder=3,
        )

        # Identity line
        lims = [max(actual.min(), pred.min()) - 5,
                min(actual.max(), pred.max()) + 5]
        ax.plot(lims, lims, color="#88aaff", linewidth=1.0, linestyle="--",
                alpha=0.7, zorder=4, label="perfect")

        # Colorbar
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("|error|  pts/100", color="#aaaacc", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="#aaaacc", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaaacc")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual  (pts / 100 poss)", color="#aaaacc", fontsize=9)
        ax.set_ylabel("Predicted  (pts / 100 poss)", color="#aaaacc", fontsize=9)
        ax.set_title(
            f"{season}  ·  {model_name}\n"
            f"train {n_train} games → test {n_test} games",
            color="white", fontsize=9.5, fontweight="bold",
        )

        stats = (f"RMSE {rmse:.2f}   MAE {mae:.2f}   "
                 f"bias {bias:+.2f}")
        ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                color="#ccccff", fontsize=8, va="top",
                bbox=dict(fc="#111133", ec="none", alpha=0.7))

        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.grid(color="#1a1a3a", linewidth=0.5)

conn.close()
fig.suptitle(
    "Model 1 vs Model 2 — predicted vs actual offensive efficiency\n"
    "train on first 80% of season  |  test on final 20%",
    color="white", fontsize=12, fontweight="bold", y=1.01,
)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_PATH}")
