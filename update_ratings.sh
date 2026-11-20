#!/usr/bin/env bash
# update_ratings.sh — daily rolling team ratings for every league in portal_pivot_db.
#
# The single entry point portal_pivot_db's nightly chain calls (its step
# ratings.kenpom_update, command KENPOM_DAILY); portal_pivot_db then imports the
# builds (ratings.kenpom_rolling_import). Refits only the league-seasons in progress
# whose games changed (scripts/rolling_ratings.py; unchanged ones are skipped), so a
# night with no new games does nothing.
#
#   ./update_ratings.sh                 # league-seasons in progress
#   ./update_ratings.sh --all           # every league-season (backfill, ~minutes)
#   ./update_ratings.sh --leagues NCAA-D1 --seasons 2027
#
# Environment: PORTAL_DB_DIR (portal_pivot_db's db/ directory; default
# ~/Dropbox/portal_pivot_db/db), KENPOM_PYTHON (default nba_draft_prediction's .venv,
# which has numpy, scipy and pandas). Output goes to stdout and logs/update_ratings_<date>.log.
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
PY=${KENPOM_PYTHON:-$HOME/Dropbox/nba_draft_prediction/.venv/bin/python}
cd "$REPO"
mkdir -p logs
LOG="logs/update_ratings_$(date +%Y%m%d).log"
exec > >(tee -a "$LOG") 2>&1
echo "======== $(date)  update_ratings.sh $*  (PORTAL_DB_DIR=${PORTAL_DB_DIR:-default})"
set +e
"$PY" scripts/rolling_ratings.py compute "$@"
STATUS=$?
set -e
echo "======== $(date)  done (exit $STATUS)"
exit "$STATUS"
