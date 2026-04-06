"""
Rolling net rating plot for 2025-26 NCAA season.

Fits Model 2 (ridge) on all games up to each weekly date milestone,
records posterior mean and std of net_rtg (AdjO - AdjD) per team, and plots
the top TOP_N teams by final net_rtg with ±1σ ribbons.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games
from models.model2 import Model2

# ── Config ───────────────────────────────────────────────────────────────────
SEASON    = 2026
TOP_N     = 15
N_POST    = 100
RNG_SEED  = 42
WEEK_STEP = 7        # advance milestone by this many days each step
MIN_GAMES = 100      # minimum cumulative games before first fit
OUT_PATH  = Path(__file__).parent / "rolling_net_rtg_2025_26.png"

# ── Load data ─────────────────────────────────────────────────────────────────
conn = open_db()

cur = conn.cursor()
cur.execute("""
    SELECT DISTINCT s.GameID, s.Date
    FROM schedules s
    JOIN boxscores b ON s.GameID = b.GameID
    WHERE s.Year = ?
    ORDER BY s.GameID
""", (SEASON,))
game_order = cur.fetchall()   # [(game_id, date_str), ...]

cur.execute("SELECT TeamID, School FROM teams_years WHERE year = ?", (SEASON,))
team_names = dict(cur.fetchall())

all_rows = load_season_games(conn, SEASON)
conn.close()

def parse_date(s):
    for fmt in ("%b %d, %Y", "%b %d,%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {s!r}")

# Map game_id → parsed date
game_dt = {gid: parse_date(ds) for gid, ds in game_order}
all_game_ids_sorted = [gid for gid, _ in game_order]

# Map game_id → rows
from models.data import GameRow
row_by_gid: dict[int, list[GameRow]] = {}
for r in all_rows:
    row_by_gid.setdefault(r.game_id, []).append(r)

# ── Build weekly date milestones ──────────────────────────────────────────────
season_start = min(game_dt.values())
season_end   = max(game_dt.values())

milestones = []
d = season_start + timedelta(days=WEEK_STEP)
while d <= season_end + timedelta(days=1):
    gids_so_far = [gid for gid in all_game_ids_sorted if game_dt[gid] < d]
    if len(gids_so_far) >= MIN_GAMES:
        milestones.append((d - timedelta(days=1), set(gids_so_far)))
    d += timedelta(days=WEEK_STEP)

# Always include exact season end
final_gids = set(all_game_ids_sorted)
milestones.append((season_end, final_gids))
# Deduplicate by date
seen_dates = set()
unique_milestones = []
for dt, gids in milestones:
    if dt not in seen_dates:
        seen_dates.add(dt)
        unique_milestones.append((dt, gids))
milestones = unique_milestones

print(f"Season {SEASON}: {len(all_game_ids_sorted)} games, {len(milestones)} milestones")

# ── Rolling fit ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
# results[tid] = [(date, mean_net, std_net), ...]
results: dict[int, list[tuple[datetime, float, float]]] = {}

for m_idx, (cutoff_dt, gids_so_far) in enumerate(milestones):
    rows_so_far = [r for gid in gids_so_far for r in row_by_gid.get(gid, [])]

    model = Model2()
    model.fit_rows(rows_so_far, SEASON)

    T = len(model.teams_)
    draws = model.sample_posterior(N_POST, rng)

    # net_rtg = AdjO - AdjD = (μ+o_i) - (μ-d_i) = o_i + d_i
    net_draws = np.array([th[1:T+1] + th[T+1:2*T+1] for th in draws])  # (N_POST, T)
    net_mean  = net_draws.mean(axis=0)
    net_std   = net_draws.std(axis=0)

    for t_pos, tid in enumerate(model.teams_):
        results.setdefault(int(tid), []).append(
            (cutoff_dt, float(net_mean[t_pos]), float(net_std[t_pos]))
        )

    n = len(gids_so_far)
    print(f"  {m_idx+1}/{len(milestones)}  cutoff={cutoff_dt.strftime('%b %d')}  games={n}", flush=True)

# ── Select top-N teams by final net_rtg ──────────────────────────────────────
final_net = {tid: pts[-1][1] for tid, pts in results.items()}
top_tids   = sorted(final_net, key=final_net.get, reverse=True)[:TOP_N]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

cmap   = plt.get_cmap("tab20")
colors = [cmap(i / TOP_N) for i in range(TOP_N)]

label_entries = []
for tid, color in zip(top_tids, colors):
    pts   = results[tid]
    dates = [p[0] for p in pts]
    means = np.array([p[1] for p in pts])
    stds  = np.array([p[2] for p in pts])
    name  = team_names.get(tid, f"Team {tid}")

    ax.plot(dates, means, color=color, linewidth=1.7, zorder=3)
    ax.fill_between(dates, means - stds, means + stds,
                    color=color, alpha=0.18, zorder=2)
    label_entries.append((name, color, means[-1]))

# Sorted right-side labels
label_entries.sort(key=lambda x: x[2], reverse=True)
x_start = min(p[0] for pts in results.values() for p in pts)
x_end   = max(p[0] for pts in results.values() for p in pts)
x_span  = x_end - x_start
ax.set_xlim(x_start, x_end + x_span * 0.20)

# Clamp y-axis to ±3σ of final values to suppress early-season noise
final_vals = [v for _, _, v in label_entries]
yc = np.array(final_vals)
ax.set_ylim(yc.min() - 8, yc.max() + 8)

# Right-side labels with nudging to avoid overlap
MIN_GAP = 1.1   # pts on y-axis
placed: list[float] = []
for name, color, val in label_entries:
    y = val
    for py in placed:
        if abs(y - py) < MIN_GAP:
            y = min(py - MIN_GAP, y)
    placed.append(y)
    # connector tick from line end to label
    ax.annotate(
        name, xy=(x_end, val), xytext=(10, 0), textcoords="offset points",
        color=color, fontsize=8.0, va="center", fontweight="bold",
        annotation_clip=False,
    )

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
fig.autofmt_xdate(rotation=30, ha="right")

for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_color("#cccccc")
for spine in ax.spines.values():
    spine.set_color("#333355")

ax.set_ylabel("Net Rating  (AdjO − AdjD,  pts / 100 poss)", color="#cccccc", fontsize=10)
ax.set_title(
    f"2025-26  Rolling Net Rating  —  top {TOP_N} teams at season end  "
    f"(ribbon = ±1σ posterior)",
    color="white", fontsize=12, fontweight="bold", pad=12,
)
ax.grid(axis="y", color="#222244", linewidth=0.6, linestyle="--")
ax.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
ax.tick_params(colors="#cccccc", labelsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT_PATH}")
