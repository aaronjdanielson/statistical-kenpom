"""
Early-season OSA evaluation: continuity prior (Fix B — shift-only).

Fix B isolates the single question:
  "If we keep the same shrinkage as Model 2, does centering the prior at
   last year's rating (scaled by returning minutes) help early-season prediction?"

Four models evaluated on weeks 1–EVAL_WEEKS:

  M2_static   : zero-centered ridge, λI precision (baseline)
  M2_hl60     : recency-weighted ridge (current best overall)
  M2_shift    : shift-only continuity prior — same P=λI as M2_static,
                prior mean m_i = r_i · ô_{i,s-1}
  M2_shift_hl : shift prior + recency weighting (combined)

Diagnostics per window:
  |m|₂           : prior mean norm — how far from zero the prior sits
  |θ̂ − m|₂     : how far the posterior moved from the prior
  These show whether the prior is influential or overwhelmed by the data.

Stratified analysis:
  (1) returning-minutes quartile
  (2) opponent-strength quartile (AdjO of opponents in test window)
  Key interaction: benefit largest when r_i high AND schedule weakly identified.

Outputs
-------
  continuity_curves.png    — weekly RMSE/bias for all four models
  continuity_summary.png   — aggregate bars + stratified panels
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
CURVES_OUT  = Path(__file__).parent / "continuity_curves.png"
SUMMARY_OUT = Path(__file__).parent / "continuity_summary.png"

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

# ── Load returning minutes ─────────────────────────────────────────────────────
print(f"Loading returning minutes for season {SEASON}...", end=" ", flush=True)
r_minutes = load_returning_minutes(SEASON)
print(f"done  ({len(r_minutes)} teams with data)")
r_vals = np.array(list(r_minutes.values()))
print(f"  r_i: min={r_vals.min():.1%}  p25={np.percentile(r_vals,25):.1%}  "
      f"median={np.median(r_vals):.1%}  p75={np.percentile(r_vals,75):.1%}  "
      f"max={r_vals.max():.1%}")

# ── Rolling evaluation ─────────────────────────────────────────────────────────
MODELS = ["M2_static", "M2_hl60", "M2_shift", "M2_shift_hl"]
results:     dict[str, list] = {k: [] for k in MODELS}
diagnostics: list[dict]      = []   # per-window prior influence

print(f"\n{'Win':>3}  {'Date':>6}  {'Train':>5}  {'Games':>5}  "
      f"{'M2':>6}  {'M2hl':>6}  {'Shift':>6}  {'Sh+hl':>6}  "
      f"{'|m|':>6}  {'|θ-m|':>6}")
print("-" * 72)

for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
    actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    # M2 static
    m2s = Model2(lambda_team=100.0, lambda_pace=50.0)
    m2s.fit_rows(train_rows, SEASON)

    # M2 hl=60
    sw  = recency_weights(train_rows, game_dt, win_dt, half_life_days=RECENCY_HL)
    m2r = Model2(lambda_team=100.0, lambda_pace=50.0)
    m2r.fit_rows(train_rows, SEASON, sample_weight=sw)

    # M2 shift-only (Fix B)
    m_sh = Model2ContinuityPrior(
        prev_effects=prev_effects,
        r_minutes=r_minutes,
        sigma2_prev=sigma2_prev,
        prev_var=prev_var,
        shift_only=True,
        lambda_team=100.0,
        lambda_pace=50.0,
    )
    m_sh.fit_rows(train_rows, SEASON)

    # M2 shift + hl=60
    m_sh_hl = Model2ContinuityPrior(
        prev_effects=prev_effects,
        r_minutes=r_minutes,
        sigma2_prev=sigma2_prev,
        prev_var=prev_var,
        shift_only=True,
        lambda_team=100.0,
        lambda_pace=50.0,
    )
    m_sh_hl.fit_rows(train_rows, SEASON, sample_weight=sw)

    for label, model in [
        ("M2_static",   m2s),
        ("M2_hl60",     m2r),
        ("M2_shift",    m_sh),
        ("M2_shift_hl", m_sh_hl),
    ]:
        resid = model.predict_efficiency(test_rows) - actual
        results[label].append((
            win_dt,
            float(np.sqrt(np.mean(resid ** 2))),
            float(np.mean(np.abs(resid))),
            float(np.mean(resid)),
        ))

    # Prior influence diagnostics (on shift model)
    T = len(m_sh.teams_)
    prior_m_vec = np.zeros(2 * T)
    theta_od    = np.empty(2 * T)
    for idx, tid in enumerate(m_sh.teams_):
        r_i = float(r_minutes.get(int(tid), 0.0))
        o_hat, d_hat = prev_effects.get(int(tid), (0.0, 0.0))
        prior_m_vec[idx]     = r_i * o_hat
        prior_m_vec[T + idx] = r_i * d_hat
        theta_od[idx]     = float(m_sh.theta_hat_[1 + idx])
        theta_od[T + idx] = float(m_sh.theta_hat_[1 + T + idx])

    m_norm     = float(np.linalg.norm(prior_m_vec))
    theta_diff = float(np.linalg.norm(theta_od - prior_m_vec))
    diagnostics.append(dict(
        win_dt=win_dt, m_norm=m_norm, theta_diff=theta_diff,
        n_train=len(train_gids), n_test=len(test_rows),
    ))

    print(f"{w_idx+1:>3}  {win_dt:%b %d}  {len(train_gids):>5}  "
          f"{len(test_rows):>5}  "
          f"{results['M2_static'][-1][1]:>6.2f}  "
          f"{results['M2_hl60'][-1][1]:>6.2f}  "
          f"{results['M2_shift'][-1][1]:>6.2f}  "
          f"{results['M2_shift_hl'][-1][1]:>6.2f}  "
          f"{m_norm:>6.1f}  {theta_diff:>6.1f}", flush=True)

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n{'Model':>14}  {'RMSE':>7}  {'MAE':>7}  {'Bias':>7}")
print("-" * 44)
agg = {}
for label in MODELS:
    pts    = results[label]
    rmse_m = float(np.mean([p[1] for p in pts]))
    mae_m  = float(np.mean([p[2] for p in pts]))
    bias_m = float(np.mean([p[3] for p in pts]))
    agg[label] = (rmse_m, mae_m, bias_m)
    print(f"  {label:>14}  {rmse_m:>7.3f}  {mae_m:>7.3f}  {bias_m:>+7.3f}")

# ── Stratified analysis ────────────────────────────────────────────────────────
# Build per-row (team_id, r_i, resid_shift, resid_base) from all windows
per_row: list[tuple] = []
for w_idx, (win_dt, train_gids, pred_gids) in enumerate(windows):
    train_rows = [r for g in train_gids for r in row_by_gid[g]]
    test_rows  = [r for g in pred_gids  for r in row_by_gid[g]]
    actual     = np.array([r.pts / r.poss * 100.0 for r in test_rows])

    m2s = Model2(lambda_team=100.0, lambda_pace=50.0)
    m2s.fit_rows(train_rows, SEASON)

    m_sh = Model2ContinuityPrior(
        prev_effects=prev_effects, r_minutes=r_minutes,
        sigma2_prev=sigma2_prev, shift_only=True,
    )
    m_sh.fit_rows(train_rows, SEASON)

    # Opponent strength proxy: previous-season AdjO of the opponent
    for row, act, pb, ps in zip(
        test_rows, actual,
        m2s.predict_efficiency(test_rows),
        m_sh.predict_efficiency(test_rows),
    ):
        tid = int(row.team_id)
        oid = int(row.opp_id)
        opp_prev = prev_effects.get(oid, (0.0, 0.0))
        opp_strength = float(m_prev.theta_hat_[0]) + opp_prev[0]   # AdjO of opp
        per_row.append((
            r_minutes.get(tid, 0.0),
            (ps - act) ** 2,
            (pb - act) ** 2,
            opp_strength,
        ))

per_arr  = np.array(per_row)
r_all    = per_arr[:, 0]
mse_sh   = per_arr[:, 1]
mse_base = per_arr[:, 2]
opp_str  = per_arr[:, 3]

q_bounds = np.quantile(r_all, [0.0, 0.25, 0.50, 0.75, 1.0])
qlabels  = ["Q1 low r", "Q2", "Q3", "Q4 high r"]

print(f"\nStratified RMSE by returning-minutes quartile "
      f"(r: ≤{q_bounds[1]:.0%} / ≤{q_bounds[2]:.0%} / ≤{q_bounds[3]:.0%}):")
print(f"  {'Quartile':>10}  {'N':>6}  "
      f"{'M2_base':>9}  {'M2_shift':>9}  {'Δ':>7}")
q_base_rmse, q_shift_rmse = [], []
for q in range(4):
    lo, hi = q_bounds[q], q_bounds[q + 1]
    mask = (r_all >= lo) & (r_all <= hi) if q == 3 else (r_all >= lo) & (r_all < hi)
    n  = mask.sum()
    rb = float(np.sqrt(mse_base[mask].mean())) if n else float("nan")
    rs = float(np.sqrt(mse_sh[mask].mean()))   if n else float("nan")
    q_base_rmse.append(rb)
    q_shift_rmse.append(rs)
    flag = " ←" if (rs - rb) < -0.05 else ""
    print(f"  {qlabels[q]:>10}  {n:>6}  {rb:>9.3f}  {rs:>9.3f}  "
          f"{rs-rb:>+7.3f}{flag}")

# Opponent-strength quartile (low opponent = easy schedule = weakly identified)
oq_bounds = np.quantile(opp_str, [0.0, 0.25, 0.50, 0.75, 1.0])
oq_labels = ["oQ1 weak opp", "oQ2", "oQ3", "oQ4 strong opp"]
print(f"\nStratified RMSE by opponent strength (prev-season AdjO):")
print(f"  {'Quartile':>14}  {'N':>6}  "
      f"{'M2_base':>9}  {'M2_shift':>9}  {'Δ':>7}")
for q in range(4):
    lo, hi = oq_bounds[q], oq_bounds[q + 1]
    mask = (opp_str >= lo) & (opp_str <= hi) if q == 3 \
           else (opp_str >= lo) & (opp_str < hi)
    n  = mask.sum()
    rb = float(np.sqrt(mse_base[mask].mean())) if n else float("nan")
    rs = float(np.sqrt(mse_sh[mask].mean()))   if n else float("nan")
    flag = " ←" if (rs - rb) < -0.05 else ""
    print(f"  {oq_labels[q]:>14}  {n:>6}  {rb:>9.3f}  {rs:>9.3f}  "
          f"{rs-rb:>+7.3f}{flag}")

# ── Figures ────────────────────────────────────────────────────────────────────
BG    = "#0f0f1a"
GRID  = "#1a1a3a"
SPINE = "#333355"
TICK  = "#aaaacc"
C_BASE  = "#888899"
C_HL60  = "#ff9f1c"
C_SHIFT = "#a8ff78"
C_SH_HL = "#4cc9f0"

plot_cfg = [
    ("M2_static",   C_BASE,  "--", 1.6, "M2 static"),
    ("M2_hl60",     C_HL60,  "--", 1.6, f"M2 hl={RECENCY_HL}d"),
    ("M2_shift",    C_SHIFT, "-",  2.2, "M2 shift-only (Fix B)"),
    ("M2_shift_hl", C_SH_HL, "-",  1.8, "M2 shift + hl=60"),
]

# Figure 1: weekly curves
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.patch.set_facecolor(BG)
ax_rmse, ax_bias, ax_diag = axes

for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")

for key, color, ls, lw, label in plot_cfg:
    pts   = results[key]
    dates = [p[0] for p in pts]
    ax_rmse.plot(dates, [p[1] for p in pts], color=color, lw=lw, ls=ls, label=label)
    ax_bias.plot(dates, [p[3] for p in pts], color=color, lw=lw, ls=ls, label=label)

ax_rmse.set_ylabel("RMSE  (pts/100)", color=TICK)
ax_rmse.set_title(
    f"Early-season OSA  ·  {SEASON}  (shift-only, Fix B)  ·  prior from {PREV_SEASON}",
    color="white", fontsize=11, fontweight="bold",
)
ax_rmse.legend(loc="upper right", labelcolor="white",
               facecolor="#111133", edgecolor=SPINE, fontsize=9)

ax_bias.axhline(0, color="#555577", lw=0.9, linestyle=":")
ax_bias.set_ylabel("Bias  (pts/100)", color=TICK)
ax_bias.set_title("Bias", color="white", fontsize=11, fontweight="bold")
ax_bias.legend(loc="upper right", labelcolor="white",
               facecolor="#111133", edgecolor=SPINE, fontsize=9)

# Prior influence panel
d_dates  = [d["win_dt"]    for d in diagnostics]
m_norms  = [d["m_norm"]    for d in diagnostics]
th_diffs = [d["theta_diff"] for d in diagnostics]
ax_diag.plot(d_dates, m_norms,  color=C_SHIFT, lw=1.8, ls="-",  label="|m|₂  prior norm")
ax_diag.plot(d_dates, th_diffs, color=C_SH_HL, lw=1.8, ls="--", label="|θ̂−m|₂  posterior drift")
ax_diag.set_ylabel("L₂ norm  (pts/100)", color=TICK)
ax_diag.set_title("Prior influence: how far does the posterior move from the prior?",
                  color="white", fontsize=10, fontweight="bold")
ax_diag.legend(loc="upper right", labelcolor="white",
               facecolor="#111133", edgecolor=SPINE, fontsize=9)
ax_diag.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_diag.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

plt.tight_layout()
plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved curves → {CURVES_OUT}")

# Figure 2: stratified summary
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor(BG)

def _strat_panel(ax, base_vals, shift_vals, x_labels, title):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", axis="y")
    x   = np.arange(len(x_labels))
    w   = 0.35
    b1  = ax.bar(x - w/2, base_vals,  w, color=C_BASE,  alpha=0.85, label="M2 static")
    b2  = ax.bar(x + w/2, shift_vals, w, color=C_SHIFT, alpha=0.85, label="M2 shift")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, color=TICK, fontsize=8)
    ax.set_ylabel("RMSE  (pts/100)", color=TICK, fontsize=9)
    ax.set_title(title, color=TICK, fontsize=8)
    ax.legend(labelcolor="white", facecolor="#111133", edgecolor=SPINE, fontsize=8)

_strat_panel(axes2[0], q_base_rmse, q_shift_rmse, qlabels,
             "by returning-minutes quartile")
_strat_panel(axes2[1],
             [float(np.sqrt(mse_base[(opp_str >= oq_bounds[q]) &
              (opp_str < oq_bounds[q+1] if q < 3 else opp_str <= oq_bounds[4])].mean()))
              for q in range(4)],
             [float(np.sqrt(mse_sh[(opp_str >= oq_bounds[q]) &
              (opp_str < oq_bounds[q+1] if q < 3 else opp_str <= oq_bounds[4])].mean()))
              for q in range(4)],
             oq_labels, "by opponent strength (prev-season AdjO)")

fig2.suptitle(
    f"Continuity prior Fix B (shift-only)  ·  {SEASON} weeks 1–{EVAL_WEEKS}"
    f"  ·  prior from {PREV_SEASON}",
    color="white", fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig(SUMMARY_OUT, dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
print(f"Saved summary → {SUMMARY_OUT}")
