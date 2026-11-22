# Rolling team ratings for every league (production, since 2026-11-21)

`scripts/rolling_ratings.py`, run nightly by the hub through `update_ratings.sh`. One file
pair per league-season (NCAA-D1, ~150 international leagues, national-team competitions, NBA,
G League): weekly snapshots of every team's offence, defence, net rating and pace, each with
its posterior sd, fitted on the games played up to the snapshot date.

## The model: `Model2Chain` (`models/model2_chain.py`)

Model 2's ridge regression with an exact Gaussian posterior, made one link of a chain over
a league's seasons (an AR(1) state-space across seasons, a conjugate update within one):

    e_k     = mu + o_i - d_j + eta*h + eps,     eps ~ N(0, s2 / poss_k)
    o_{i,s} = rho * o_{i,s-1} + u_o,  d likewise;  (u_o, u_d) ~ N(0, s2/lambda * [[1, r], [r, 1]])
    eta_s   = eta_{s-1} + v,  v ~ N(0, 0.25^2);    s2 starts from last season's (weight 200 rows)

- A team starts the season at `rho` x its final rating of the league's most recent earlier
  season (up to 4 back, `rho^gap` and a wider prior across a gap), never from a season still
  being played. A team with no earlier season starts at league average with the chain's
  stationary prior, penalty `lambda (1 - rho^2)`; that matches the best prior-free shrinkage
  found by one-step-ahead tests in every league type.
- `r`, the prior correlation of a team's offence and defence (both "good" when positive),
  improves totals and early-season margins.
- The home effect starts from the league's previous estimate, so strong teams hosting weak
  ones in November no longer inflate it (NCAA: 7.7 a week in, against ~3.1 at season end).
- Posterior covariance `s2 * (X'WX + P)^-1` with `s2` on the per-possession scale. Until
  2026-11-20 every kenpom sd used the per-game variance and was ~8x too small.

| League type | lambda | rho (carry-over) | r (offence-defence) |
|---|---|---|---|
| NCAA-D1 | 500 | 0.9 | 0.5 |
| International | 500 | 0.8 | 0.5 |
| National teams | 250 | 0.7 | 0.5 |
| NBA | 1500 | 0.6 | 0.4 |
| G League | 1500 | 0.5 | 0.0 |

Chosen 2026-11-20 by predicting each next week's games from the games before it, over every
season in the hub; a grid chosen on seasons to 2015 picked the same or neighbouring settings
on 2016 onward. Tried and not adopted (gain under 0.02): widening a prior by last season's
posterior variance, recency half-lives of 60 or 120 days, a lower start for newcomers, home
anchors 4–5x weaker or stronger.

## Evaluation (2026-11-21, from the stored files)

Next-week margin per 100 possessions, games where both teams already have a rating; old =
Model 2 with lambda 500 in every league (layout v1), new = `Model2Chain` (layout v2).

| League type | games | old RMSE | new RMSE | first quarter of season, old → new |
|---|---|---|---|---|
| NCAA-D1 | 118,688 | 17.35 | 16.28 | 19.20 → 16.82 |
| International | 212,153 | 17.18 | 16.77 | 17.68 → 16.87 |
| National teams | 4,195 | 20.80 | 19.41 | 21.00 → 18.76 |
| NBA | 36,520 | 12.89 | 12.82 | 12.81 → 12.62 |
| G League | 10,146 | 13.37 | 13.24 | 13.67 → 13.51 |

The gain is the same before and after 2016. Calibration of the stored rating sd: (final
rating − an earlier snapshot) / sqrt(sd_t² − sd_final²) has spread 0.90–1.04 (should be 1;
the old files gave 10–12). Predictive margin intervals are too wide: 80% intervals cover
84–87% (see Known gaps in the README).

## Low overlap

`models/overlap.py`. Inside a group of teams linked by games the games set the scale;
between groups that never met only the prior does. Each snapshot row carries
`n_components`, `component` (0 = the largest group) and `pairs_met_share`; each `.json` carries
`low_overlap`: several unlinked groups, or the two most weakly linked halves of the largest
group (spectral bisection) joined by under 30 games and under 20% of the smaller half's
games. Flagged today: 70 international league-seasons (Eurocup, FIBA Europe Cup, Champions
League, cups), 30 national-team (qualifying groups), 1 G League; no NCAA or NBA season.

## Files (agreed with the hub: changes go through the hub session first)

`$PORTAL_HANDOFF_DIR/kenpom/rolling/<league_id>/<season>.parquet`: league_id, season,
snapshot_date, snapshot_idx, team_id, adj_o, adj_d, adj_net, adj_pace, adj_o_sd, adj_d_sd,
adj_net_sd, n_games, n_games_espn, games_elapsed, home_effect, n_components, component,
pairs_met_share, prior_adj_o, prior_adj_d (null: started at league average).

`.json`: `key` (layout `rolling-ratings-v2`, model, settings, inputs hash, `prior_from` = the
previous season's build), `created`, `code_commit`, `code` (tree_sha256, commit, dirty),
`environment`, counts, `low_overlap` and its statistics, `final` (mu, home effect and its sd,
noise variance: where the next season starts). A league-season is refitted only when its key
changes; refitting a season changes the keys of the league's later seasons.

`$PORTAL_HANDOFF_DIR/kenpom/last_run.json`: every run's built / unchanged / failed and the
failures. Any failure exits 1, so the hub's nightly step fails.

## Changing the model

1. Build everything into a scratch directory: `PORTAL_HANDOFF_DIR=<scratch> ./update_ratings.sh --all`.
2. Compare with the stored files (next-week margins, sd calibration), and agree any new
   column with the hub session (its import skips builds with columns it does not know).
3. Swap the scratch `rolling/` into the handoff before the 06:45 nightly; then run
   `./update_ratings.sh --all` against the handoff and expect 0 built. On 2026-11-21 the
   nightly ran between a layout change and its rebuild: 45 of 46 league-seasons failed.
