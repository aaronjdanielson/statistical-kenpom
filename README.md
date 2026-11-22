# kenpom: probabilistic team ratings for every league in portal_pivot_db

A spoke of portal_pivot_db (the spoke standard: `~/Dropbox/portal_pivot_db/docs/SPOKES.md`;
how to set up, test and run it: `spoke.toml`). It reads box scores from the hub and hands
ratings back; the hub stores and serves them (`kenpom_rolling`, `ncaa_team_ratings`,
`intl_team_ratings`).

## Question

How good is each team, at each end of the floor, on any date of a season, and how sure
are we? Offence, defence, net rating and pace per 100 possessions, each with a posterior
sd, for NCAA-D1, ~150 international leagues, national-team competitions, NBA and G League.

## Status

- **Rolling ratings** (nightly; the hub runs `update_ratings.sh`): `Model2Chain`, Model 2's
  exact Gaussian ridge chained over a league's seasons (a team starts from last season's
  rating, offence and defence correlated, home effect anchored), shrinkage per league type.
  Since 2026-11-21. Design and evaluation: [docs/rolling_ratings.md](docs/rolling_ratings.md).
- Next-week margin error per 100 possessions, old (Model 2, lambda 500 everywhere) → new:
  NCAA 17.35 → 16.28 (first quarter of the season 19.20 → 16.82), international
  17.18 → 16.77, national teams 20.80 → 19.41, NBA 12.89 → 12.82, G League 13.37 → 13.24.
- Posterior sds are the right size since 2026-11-20 (they were ~8x too small): rating
  z-scores have spread 0.90–1.04.
- **Season files** (`team_ratings.db`, `intl_team_ratings.db`; built by hand with
  `scripts/build_team_ratings.py`, `build_intl_team_ratings.py`): Model 2, lambda 500, one fit
  per season, corrected sds. The hub imports them by hand.
- Every run writes `last_run.json` beside its outputs and exits 1 if anything failed.
  Tests: `.venv/bin/python -m pytest tests/unit -q` (131, no hub or network needed).

## Known gaps

- Predictive margin intervals are too wide (80% intervals cover 84–87%): the two teams'
  efficiencies in one game are positively correlated, and the model treats them as
  independent. A game-level random effect would fix it (it cancels in margins and doubles
  in totals).
- Ratings are static within a season: every game up to a snapshot counts equally. Recency
  weights gained under 0.02; a within-season state-space (`models/model4.py`) is the
  principled route and is not in production.
- Season files and rolling ratings use different models, so a finished season's last
  rolling snapshot no longer equals its season rating.
- Low-overlap league-seasons (cups, qualifying groups; flagged `low_overlap`): ratings
  compare unlinked groups by the prior, not by games.
- `team_ratings.db` holds an NCAA 2002 season fitted on 25 games (the hub's to drop).
- This repo still holds frozen copies of the hub's data (`ncaa.db`, `intl.db`,
  `national.db`, backups, `output/`) and its old scrapers. They are to be retired, but
  other repos read them first (`check_ecosystem.py --refs kenpom`); deletion needs the user.
- Waivers in `spoke.toml`: R4 (output layout) and R7 (the package is named `models`).

## Documents

- [docs/rolling_ratings.md](docs/rolling_ratings.md): the production model, settings,
  evaluation, low-overlap flag, output files, how to change the model safely.
- [docs/models_overview.md](docs/models_overview.md) and `docs/model1…4_*.md`: the four
  research models (fixed point, ridge, bilinear, Kalman).
- [docs/continuity_prior.md](docs/continuity_prior.md): the returning-minutes prior (research).
- [docs/research_writeup.md](docs/research_writeup.md): the README until 2026-11-21, with figures.
- [docs/project_history.md](docs/project_history.md): the project guide until 2026-11-21, dated results.
