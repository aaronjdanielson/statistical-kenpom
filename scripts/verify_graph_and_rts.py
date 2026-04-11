"""
Two verification tests for the identification story.

Test 1 — Game graph connectivity over the season
-------------------------------------------------
Claim: at week 3 the graph is sparse with small connected components;
by week 16 virtually every pair of teams is linked within 2-3 hops.
Computes per-window: #components, largest-component fraction, mean
shortest-path within largest component, and fraction of pairs at dist ≤ 2.

Test 2 — RTS smoother with τ→0 recovers Model 2
-------------------------------------------------
Claim: the RTS smoother conditioning on the full season with τ→0 is
formally equivalent to Model 2's batch ridge solve (same linear-Gaussian
structure, same prior, same observations).  With τ→0, Q≈0 and the Kalman
forward-filter + RTS smoother reduce to sequential Bayesian updating on a
static parameter, which equals the batch posterior.

Both models fix μ and η from the same Model 2 pre-fit, so the comparison
is on the (o, d) block alone.
"""
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.data import open_db, load_season_games, parse_date, GameRow
from models.model2 import Model2
from models.model4 import Model4

SEASON    = 2026
WEEK_STEP = 7
MIN_TRAIN = 300

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

game_dt   = {gid: parse_date(ds) for gid, ds in game_order}
all_gids  = [gid for gid, _ in game_order]
all_rows  = load_season_games(open_db(), SEASON)
row_by_gid: dict[int, list[GameRow]] = defaultdict(list)
for r in all_rows:
    row_by_gid[r.game_id].append(r)

season_start = min(game_dt.values())
season_end   = max(game_dt.values())

windows = []
d = season_start
while d <= season_end:
    d_next = d + timedelta(days=WEEK_STEP)
    train  = {g for g in all_gids if game_dt[g] < d}
    pred   = {g for g in all_gids if d <= game_dt[g] < d_next}
    if len(train) >= MIN_TRAIN and pred:
        windows.append((d, train, pred))
    d = d_next

all_teams = sorted({r.team_id for r in all_rows} | {r.opp_id for r in all_rows})
tidx = {t: i for i, t in enumerate(all_teams)}
N = len(all_teams)

print("=" * 70)
print("TEST 1 — Game graph connectivity over the season")
print("=" * 70)
print(f"\n{'Date':>8}  {'Train':>5}  {'#comp':>5}  {'LCC%':>5}  "
      f"{'diam':>5}  {'dist≤2%':>7}  {'dist≤3%':>7}")
print("-" * 55)

graph_rows = []
for win_dt, train_gids, _ in windows:
    # Build adjacency for the training graph
    edges_i, edges_j = [], []
    seen = set()
    for gid in train_gids:
        for r in row_by_gid[gid]:
            a, b = tidx[r.team_id], tidx[r.opp_id]
            key  = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                edges_i.append(a); edges_j.append(b)
                edges_i.append(b); edges_j.append(a)

    A = sp.csr_matrix(
        (np.ones(len(edges_i)), (edges_i, edges_j)),
        shape=(N, N),
    )
    n_comp, labels = csgraph.connected_components(A, directed=False)

    # Largest connected component
    lcc_label = np.bincount(labels).argmax()
    lcc_nodes = np.where(labels == lcc_label)[0]
    lcc_frac  = len(lcc_nodes) / N

    # Shortest paths within LCC (BFS, unweighted)
    A_lcc = A[lcc_nodes][:, lcc_nodes]
    dist  = csgraph.shortest_path(A_lcc, directed=False, unweighted=True)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    mean_d = float(finite.mean()) if len(finite) else float("nan")
    diam   = float(finite.max())  if len(finite) else float("nan")

    # Fraction of LCC pairs within 2 and 3 hops
    n_lcc = len(lcc_nodes)
    n_pairs = n_lcc * (n_lcc - 1)
    frac2 = float(np.sum(finite <= 2) / n_pairs) if n_pairs else 0.0
    frac3 = float(np.sum(finite <= 3) / n_pairs) if n_pairs else 0.0

    graph_rows.append((win_dt, n_comp, lcc_frac, diam, frac2, frac3))
    print(f"{win_dt:%b %d}  {len(train_gids):>5}  {n_comp:>5}  "
          f"{lcc_frac:>4.0%}  {diam:>5.1f}  {frac2:>7.1%}  {frac3:>7.1%}")

# ── Test 2 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2 — RTS smoother with τ→0 vs Model 2 (full season)")
print("=" * 70)

# Fit Model 2 on the full season (ground truth)
m2_full = Model2(lambda_team=100.0, lambda_pace=50.0)
m2_full.fit_rows(all_rows, SEASON)
T   = len(m2_full.teams_)
mu2 = float(m2_full.theta_hat_[0])
o2  = m2_full.theta_hat_[1:1 + T]          # offense effects
d2  = m2_full.theta_hat_[1 + T:1 + 2 * T]  # defense effects

print(f"\nModel 2 fit: {T} teams,  μ={mu2:.3f},  "
      f"σ_eff={float(np.sqrt(m2_full._sigma2_eff)):.3f}")

# Fit Model 4 with near-zero τ (static limit), no hyperopt
# Model 4 internally bootstraps μ/η from a Model 2 pre-fit, so they align.
TAU_STATIC = 1e-4

m4_static = Model4(
    lambda_team=100.0,
    lambda_pace=50.0,
    tau_o=TAU_STATIC,
    tau_d=TAU_STATIC,
    optimize_hyperparams=False,
    week_step=WEEK_STEP,
)
m4_static.fit_rows(all_rows, SEASON, game_dates=game_dt)

# RTS smoother on the full season
smoothed = m4_static.rts_smoother()

# Extract final smoothed state — use the last time step
_, final_summary = smoothed[-1]

# Compare AdjO and AdjD for all teams
m2_summary = m2_full.point_summary()
common_tids = sorted(set(final_summary.keys()) & set(m2_summary.keys()))

adj_o_m2  = np.array([m2_summary[t].adj_o for t in common_tids])
adj_d_m2  = np.array([m2_summary[t].adj_d for t in common_tids])
adj_o_rts = np.array([final_summary[t].adj_o for t in common_tids])
adj_d_rts = np.array([final_summary[t].adj_d for t in common_tids])

diff_o = adj_o_rts - adj_o_m2
diff_d = adj_d_rts - adj_d_m2

print(f"\nComparison across {len(common_tids)} teams  (RTS smoother τ={TAU_STATIC} vs Model 2):")
print(f"  AdjO  mean_diff={diff_o.mean():+.4f}  "
      f"max_abs={np.abs(diff_o).max():.4f}  "
      f"RMSE={np.sqrt(np.mean(diff_o**2)):.4f}")
print(f"  AdjD  mean_diff={diff_d.mean():+.4f}  "
      f"max_abs={np.abs(diff_d).max():.4f}  "
      f"RMSE={np.sqrt(np.mean(diff_d**2)):.4f}")

# Show the 5 largest discrepancies
print("\nLargest 5 AdjO discrepancies (RTS − M2):")
top5 = np.argsort(np.abs(diff_o))[-5:][::-1]
for idx in top5:
    t = common_tids[idx]
    print(f"  team {t:>6}  M2={adj_o_m2[idx]:.2f}  "
          f"RTS={adj_o_rts[idx]:.2f}  diff={diff_o[idx]:+.4f}")

# Correlation
corr_o = float(np.corrcoef(adj_o_m2, adj_o_rts)[0, 1])
corr_d = float(np.corrcoef(adj_d_m2, adj_d_rts)[0, 1])
print(f"\nCorrelation:  AdjO r={corr_o:.6f}   AdjD r={corr_d:.6f}")
print(f"\nVerdict: {'PASS' if np.abs(diff_o).max() < 1.0 else 'FAIL (gap > 1 pt/100)'} "
      f"— RTS(τ→0) {'≈' if np.abs(diff_o).max() < 0.5 else '!='} Model 2  "
      f"(max |ΔAdjO| = {np.abs(diff_o).max():.3f} pts/100)")
