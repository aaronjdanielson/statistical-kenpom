#!/usr/bin/env python3
"""
NCAA data pipeline for building team efficiency ratings (Models 1-3).

Steps:
  1  Teams     → ncaa.db :: teams_years
  2  Schedules → ncaa.db :: schedules
  3  Boxscores → ncaa.db :: boxscores

Usage:
  python -m ncaa_scraper.pipeline run --steps 1,2,3 --years 2002-2026
  python -m ncaa_scraper.pipeline status
"""
import argparse
import logging
import signal
import sqlite3
import sys

from ncaa_scraper.config import (
    DATA_DB_PATH, DB_PATH, HEADERS, RATE_LIMIT_RPS, RATE_LIMIT_JITTER, OUTPUT_DIR,
)
from ncaa_scraper.db import Database
from ncaa_scraper.scrapers.client import RealGMClient
from ncaa_scraper.scrapers.checkpoint import Checkpoint
from ncaa_scraper.scrapers.steps import step1_teams, step2_schedules, step3_boxscores

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_fmt)
_fh = logging.FileHandler(OUTPUT_DIR / "pipeline.log")
_fh.setFormatter(logging.Formatter(_fmt))
logging.getLogger().addHandler(_fh)

logger = logging.getLogger(__name__)


def _sigint_handler(sig, frame):
    logger.info("Interrupted — progress saved to checkpoint, exiting cleanly")
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint_handler)


def cmd_run(args):
    steps = [int(s) for s in args.steps.split(",")]
    years = _parse_years(args.years)
    checkpoint = Checkpoint(DB_PATH)
    client = RealGMClient(HEADERS, RATE_LIMIT_RPS, RATE_LIMIT_JITTER)  # used for steps 1+2 only
    db = Database(DATA_DB_PATH)

    teams = None

    try:
        if 1 in steps:
            teams = step1_teams.run_step1(client, checkpoint, db, years)

        if 2 in steps:
            if teams is None:
                teams = _load_teams_from_db(db)
            step2_schedules.run_step2(client, checkpoint, db, teams)

        if 3 in steps:
            step3_boxscores.run_step3(checkpoint, db, workers=args.workers)

    except Exception as e:
        logger.error("Pipeline aborted: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        client.close()
        db.close()


def cmd_status(args):
    checkpoint = Checkpoint(DB_PATH)
    for step, label in [
        ("step1", "Teams    "),
        ("step2", "Schedules"),
        ("step3", "Boxscores"),
    ]:
        done = checkpoint.done_count(step)
        errors = checkpoint.pending_count(step)
        print(f"{label}: {done:6d} done, {errors:4d} errors")

    # Also show DB row counts if the DB exists
    if DATA_DB_PATH.exists():
        conn = sqlite3.connect(str(DATA_DB_PATH))
        for table in ("teams_years", "schedules", "boxscores"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  db.{table}: {n:,} rows")
        conn.close()


def _parse_years(s: str) -> list[int]:
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                return list(range(int(parts[0]), int(parts[1]) + 1))
            except ValueError:
                pass
    return [int(y) for y in s.split(",")]


def _load_teams_from_db(db: Database) -> list[dict]:
    """Load team-year rows from DB when resuming after step1."""
    rows = db._conn.execute(
        "SELECT TeamID, year, School, TeamCode, ConferenceCode, ConferenceID FROM teams_years"
    ).fetchall()
    if not rows:
        logger.warning("teams_years table is empty; run step 1 first")
        return []
    cols = ["TeamID", "year", "School", "TeamCode", "ConferenceCode", "ConferenceID"]
    return [dict(zip(cols, r)) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="NCAA scraper pipeline")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run scraping steps")
    run_p.add_argument("--steps", default="1,2,3")
    run_p.add_argument("--years", default="2002-2026")
    run_p.add_argument("--workers", type=int, default=3,
                       help="Parallel workers for step 3 boxscores (default 3)")
    run_p.set_defaults(func=cmd_run)

    status_p = sub.add_parser("status", help="Show checkpoint status")
    status_p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
