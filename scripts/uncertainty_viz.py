"""
Three uncertainty visualizations for Model 2 (ridge).

Fig 1  calibration_curve.png
    Nominal vs empirical PI coverage across all coverage levels 0–100%.
    Perfect calibration = diagonal.  Overconfident = below; underconfident = above.

Fig 2  team_uncertainty_fan.png
    For the top 12 teams by final net_rtg, show posterior mean ± 1σ / ± 2σ
    as shaded fan bands through the season.  Uncertainty shrinks as games
    accumulate; late-season intervals are tight.

Fig 3  pregame_distributions.png
    For 6 marquee games, show the posterior predictive distribution of
    predicted offensive efficiency for each team side as a density / violin,
    with the actual result marked.  Captures what the model "believed"
    going into each game.
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
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games, parse_date, GameRow
from models.model2 import Model2
from models.eval import temporal_split

SEASON   = 2026
N_POST   = 300
RNG_SEED = 42
WEEK_STEP = 7
MIN_TRAIN = 300
TOP_N     = 12
OUT_DIR   = Path(__file__).parent

# ── Load ──────────────────────────────────────────────────────────────────────
conn = open_db()
cur  = conn.cursor()
cur.execute("""
    SELECT DISTINCT s.GameID, s.Date
    FROM schedules s JOIN boxscores b ON s.GameID = b.GameID
    WHERE s.Year = ? ORDER BY s.GameID
""", (SEASON,))
game_order = cur.fetchall()

cur.execute("SELECT TeamID, School FROM teams_years WHERE year = ?", (SEASON,))
team_names = dict(cur.fetchall())

all_rows = load_season_games(conn, SEASON)
conn.close()


game_dt  = {gid: parse_date(ds) for gid, ds in game_order}
all_gids = [gid for gid, _ in game_order]

row_by_gid: dict[int, list[GameRow]] = defaultdict(list)
for r in all_rows:
    row_by_gid[r.game_id].append(r)

rng = np.random.default_rng(RNG_SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Calibration curve (80/20 train/test split)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building calibration curve …")

train_rows, test_rows = temporal_split(all_rows, 0.80)

m_cal = Model2()
m_cal.fit_rows(train_rows, SEASON)

actual_cal = np.array([r.pts / r.poss * 100.0 for r in test_rows])

rng_cal = np.random.default_rng(RNG_SEED)
draws   = m_cal.sample_posterior(N_POST, rng_cal)
preds   = np.array([m_cal._predict_from_theta(th, test_rows) for th in draws])  # (N_POST, N_test)

sigma = float(np.sqrt(m_cal._sigma2_eff))

levels   = np.linspace(0.02, 0.99, 60)
emp_cov  = []
for lv in levels:
    alpha = (1 - lv) / 2
    lo = np.quantile(preds, alpha, axis=0) - abs(np.sqrt(2) * sigma * np.sqrt(-2 * np.log(alpha * np.sqrt(2 * np.pi) * sigma + 1e-12) + 1e-6))
    # Simpler: Gaussian noise convolution
    z  = float(np.abs(np.quantile(np.random.default_rng(0).standard_normal(100000), alpha)))
    lo = np.quantile(preds, alpha, axis=0) - z * sigma
    hi = np.quantile(preds, 1 - alpha, axis=0) + z * sigma
    emp_cov.append(float(np.mean((actual_cal >= lo) & (actual_cal <= hi))))

fig1, ax = plt.subplots(figsize=(7, 7))
fig1.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

ax.plot([0, 1], [0, 1], color="#555588", linewidth=1.5,
        linestyle="--", label="Perfect calibration", zorder=2)
ax.fill_between([0, 1], [0, 0], [1, 1],
                color="#222244", alpha=0.5)

ax.plot(levels, emp_cov, color="#00bfff", linewidth=2.5,
        label="Model 2 (ridge)", zorder=3)

# Shade over/under-confidence regions
ax.fill_between(levels, levels, emp_cov,
                where=[e > n for e, n in zip(emp_cov, levels)],
                color="#00ff88", alpha=0.15, label="Underconfident (PI too wide)")
ax.fill_between(levels, levels, emp_cov,
                where=[e < n for e, n in zip(emp_cov, levels)],
                color="#ff4466", alpha=0.15, label="Overconfident (PI too narrow)")

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Nominal coverage  (1 − α)", color="#aaaacc", fontsize=11)
ax.set_ylabel("Empirical coverage  (fraction of actuals inside PI)", color="#aaaacc", fontsize=11)
ax.set_title("Calibration curve — posterior predictive intervals\n"
             "2025-26 season  ·  80% train / 20% test  ·  Model 2 (ridge)",
             color="white", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.4, fontsize=9,
          labelcolor="white", facecolor="#111133", edgecolor="#333355")
ax.tick_params(colors="#aaaacc", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#333355")
ax.grid(color="#1a1a3a", linewidth=0.6, linestyle="--")

# Annotate some reference levels
for lv in [0.50, 0.80, 0.90, 0.95]:
    idx = np.argmin(np.abs(np.array(levels) - lv))
    ax.annotate(f"{emp_cov[idx]:.0%} actual\n@ {lv:.0%} nominal",
                xy=(lv, emp_cov[idx]),
                xytext=(lv - 0.22, emp_cov[idx] + 0.04),
                color="#ccccff", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555588", lw=0.8))

plt.tight_layout()
fig1.savefig(OUT_DIR / "calibration_curve_2025_26.png", dpi=150,
             bbox_inches="tight", facecolor=fig1.get_facecolor())
print("  Saved calibration_curve_2025_26.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Team uncertainty fan (weekly rolling, top 12 teams)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building team uncertainty fan …")

season_start = min(game_dt.values())
season_end   = max(game_dt.values())

milestones = []
d = season_start
while d <= season_end + timedelta(days=1):
    gids = {gid for gid in all_gids if game_dt[gid] < d}
    if len(gids) >= MIN_TRAIN:
        milestones.append((d - timedelta(days=1), gids))
    d += timedelta(days=WEEK_STEP)
milestones.append((season_end, set(all_gids)))
# deduplicate
seen = set(); uniq = []
for dt, gs in milestones:
    if dt not in seen:
        seen.add(dt); uniq.append((dt, gs))
milestones = uniq

# For each milestone fit and collect posterior mean ± std per team
# results[tid] = [(date, mean_net, std_net), ...]
fan_results: dict[int, list] = defaultdict(list)
rng_fan = np.random.default_rng(RNG_SEED)

for m_idx, (cutoff, gids) in enumerate(milestones):
    rows_here = [r for gid in gids for r in row_by_gid[gid]]
    m = Model2()
    m.fit_rows(rows_here, SEASON)
    T = len(m.teams_)
    draws_here = m.sample_posterior(200, rng_fan)
    # net_rtg draws: (N, T)
    net = np.array([th[1:T+1] + th[T+1:2*T+1] for th in draws_here])
    net_mean = net.mean(axis=0)
    net_std  = net.std(axis=0)
    for t_pos, tid in enumerate(m.teams_):
        fan_results[int(tid)].append(
            (cutoff, float(net_mean[t_pos]), float(net_std[t_pos]))
        )
    print(f"  {m_idx+1}/{len(milestones)}  {cutoff.strftime('%b %d')}  games={len(gids)}", flush=True)

# Pick top TOP_N teams by final net_rtg
final_net = {tid: pts[-1][1] for tid, pts in fan_results.items()}
top_tids  = sorted(final_net, key=final_net.get, reverse=True)[:TOP_N]

fig2, axes2 = plt.subplots(3, 4, figsize=(18, 11), sharey=False)
fig2.patch.set_facecolor("#0f0f1a")
axes2_flat = axes2.flatten()

cmap2  = plt.get_cmap("tab20")
colors2 = [cmap2(i / TOP_N) for i in range(TOP_N)]

for ax, tid, color in zip(axes2_flat, top_tids, colors2):
    ax.set_facecolor("#0f0f1a")
    pts   = fan_results[tid]
    dates = [p[0] for p in pts]
    mn    = np.array([p[1] for p in pts])
    sd    = np.array([p[2] for p in pts])
    name  = team_names.get(tid, f"Team {tid}")
    rank  = top_tids.index(tid) + 1

    ax.fill_between(dates, mn - 2*sd, mn + 2*sd,
                    color=color, alpha=0.18, label="±2σ")
    ax.fill_between(dates, mn - sd,   mn + sd,
                    color=color, alpha=0.40, label="±1σ")
    ax.plot(dates, mn, color=color, linewidth=2.0, zorder=3, label="mean")

    # Final estimate annotation
    ax.annotate(f"σ={sd[-1]:.2f}",
                xy=(dates[-1], mn[-1]), xytext=(-36, 6),
                textcoords="offset points",
                color="#cccccc", fontsize=7.5)

    ax.set_title(f"#{rank}  {name}", color=color, fontsize=9, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(colors="#888899", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2a44")
    ax.grid(color="#1a1a3a", linewidth=0.4, linestyle="--")
    ax.axhline(0, color="#444466", linewidth=0.6, linestyle=":")

    if ax == axes2_flat[0]:
        ax.legend(loc="upper left", framealpha=0.4, fontsize=7,
                  labelcolor="white", facecolor="#111133", edgecolor="#333355")

fig2.suptitle(
    f"2025-26  Team Net Rating Uncertainty  —  top {TOP_N} teams\n"
    "posterior mean  ·  shading = ±1σ / ±2σ  ·  Model 2 (ridge)",
    color="white", fontsize=12, fontweight="bold",
)
fig2.text(0.5, 0.01, "Uncertainty shrinks as games accumulate; "
          "early season σ reflects sparse schedule, "
          "not team quality volatility.",
          ha="center", color="#888899", fontsize=8.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig(OUT_DIR / "team_uncertainty_fan_2025_26.png", dpi=150,
             bbox_inches="tight", facecolor=fig2.get_facecolor())
print("  Saved team_uncertainty_fan_2025_26.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Pre-game posterior predictive distributions for marquee games
# ═══════════════════════════════════════════════════════════════════════════════
print("Building pre-game distributions …")

# Find 6 interesting games: high-profile matchups near tournament time
# Strategy: pick late-season games between top-25 teams
top25_tids = set(sorted(final_net, key=final_net.get, reverse=True)[:25])

cur2 = open_db().cursor()
cur2.execute("""
    SELECT s.GameID, s.Date, s.TeamID, s.Versus
    FROM schedules s
    JOIN boxscores b ON s.GameID = b.GameID
    WHERE s.Year = ?
    ORDER BY s.GameID
""", (SEASON,))
sched_rows = cur2.fetchall()

# Build: game_id -> (team_id_home, team_id_away, date)
game_meta: dict[int, dict] = {}
for gid, ds, tid, versus in sched_rows:
    if gid not in game_meta:
        game_meta[gid] = {"teams": [], "date": parse_date(ds)}
    game_meta[gid]["teams"].append(tid)

# Find games with both teams in top 25, after Jan 1
elite_games = [
    (gid, meta) for gid, meta in game_meta.items()
    if (len(meta["teams"]) >= 2
        and len(set(meta["teams"]) & top25_tids) >= 2
        and meta["date"] >= datetime(2026, 1, 15))
]
elite_games.sort(key=lambda x: x[1]["date"])

# Pick 6 spread across the season
step = max(1, len(elite_games) // 6)
showcase = [elite_games[i * step] for i in range(6)][:6]

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
fig3.patch.set_facecolor("#0f0f1a")

COLORS_SIDE = ["#00bfff", "#ff6b35"]

for ax, (gid, meta) in zip(axes3.flatten(), showcase):
    ax.set_facecolor("#0f0f1a")

    # Fit on all games strictly before this game's date
    train_gids = {g for g in all_gids if game_dt[g] < meta["date"]}
    if len(train_gids) < MIN_TRAIN:
        ax.text(0.5, 0.5, "insufficient training data",
                ha="center", va="center", color="#666677",
                transform=ax.transAxes)
        continue

    t_rows = [r for g in train_gids for r in row_by_gid[g]]
    m_gm   = Model2()
    m_gm.fit_rows(t_rows, SEASON)

    test_here = row_by_gid[gid]
    actual_gm = {r.team_id: r.pts / r.poss * 100.0 for r in test_here}

    rng_gm = np.random.default_rng(RNG_SEED)
    draws_gm = m_gm.sample_posterior(N_POST, rng_gm)
    sigma_gm  = float(np.sqrt(m_gm._sigma2_eff))

    x_min, x_max = 70, 150
    xs = np.linspace(x_min, x_max, 300)

    for side_idx, r in enumerate(test_here[:2]):
        tid  = r.team_id
        name = team_names.get(tid, f"T{tid}")

        # Posterior predictive: parameter uncertainty + residual noise
        param_preds = np.array([
            float(m_gm._predict_from_theta(th, [r])[0]) for th in draws_gm
        ])
        # Convolve with Gaussian noise N(0, sigma^2)
        noise   = np.random.default_rng(RNG_SEED + side_idx).normal(
            0, sigma_gm, size=len(param_preds))
        pp_draws = param_preds + noise

        kde  = gaussian_kde(pp_draws, bw_method=0.25)
        dens = kde(xs)

        color = COLORS_SIDE[side_idx]
        ax.fill_between(xs, dens, alpha=0.35, color=color)
        ax.plot(xs, dens, color=color, linewidth=1.8, label=name)

        # Mark actual
        act_val = actual_gm.get(tid)
        if act_val is not None:
            ax.axvline(act_val, color=color, linewidth=1.5,
                       linestyle=":", alpha=0.9)
            ax.annotate(f"{act_val:.0f}",
                        xy=(act_val, kde(act_val)[0] * 0.5),
                        xytext=(4, 0), textcoords="offset points",
                        color=color, fontsize=7.5, va="center")

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Offensive efficiency  (pts / 100 poss)", color="#aaaacc", fontsize=8)
    ax.set_ylabel("Posterior density", color="#aaaacc", fontsize=8)
    ax.set_title(
        f"{meta['date'].strftime('%b %d, %Y')}",
        color="white", fontsize=9, fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.4, fontsize=8,
              labelcolor="white", facecolor="#111133", edgecolor="#333355")
    ax.tick_params(colors="#888899", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2a44")
    ax.grid(color="#1a1a3a", linewidth=0.4, linestyle="--", axis="y")

    # Note: dotted line = actual result
    ax.text(0.02, 0.97, "dotted = actual result",
            transform=ax.transAxes, color="#666688",
            fontsize=6.5, va="top")

fig3.suptitle(
    "2025-26  Pre-game Posterior Predictive Distributions  —  top-25 matchups\n"
    "fit on all prior games  ·  shading = posterior density  ·  Model 2 (ridge)",
    color="white", fontsize=11, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig3.savefig(OUT_DIR / "pregame_distributions_2025_26.png", dpi=150,
             bbox_inches="tight", facecolor=fig3.get_facecolor())
print("  Saved pregame_distributions_2025_26.png")
print("\nDone.")
