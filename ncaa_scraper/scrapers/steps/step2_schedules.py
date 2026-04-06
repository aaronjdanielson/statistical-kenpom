"""Step 2: Scrape team schedules to collect GameIDs."""
import logging

from ncaa_scraper.config import REALGM_BASE
from ncaa_scraper.parsers.schedule import parse_schedule_page

logger = logging.getLogger(__name__)


def run_step2(client, checkpoint, db, teams):
    """
    Scrape schedule pages for all team-years, write to schedules table.

    Args:
        client: RealGMClient instance.
        checkpoint: Checkpoint instance.
        db: Database instance.
        teams: List of team dicts from step1.
    """
    total = len(teams)
    game_ids: set[int] = set()

    for i, team in enumerate(teams):
        team_id = team["TeamID"]
        year = team["year"]
        key = f"sched_{team_id}_{year}"
        if checkpoint.is_done("step2", key):
            continue
        url = (
            f"{REALGM_BASE}/ncaa/conferences/"
            f"{team['ConferenceCode']}/{team['ConferenceID']}/"
            f"{team['TeamCode']}/{team_id}/schedule/{year}"
        )
        try:
            html = client.get(url)
            rows = parse_schedule_page(html, team_id, year)
            db.insert_schedules(rows)
            for r in rows:
                game_ids.add(r["GameID"])
            checkpoint.mark_done("step2", key)
            if (i + 1) % 100 == 0:
                logger.info("Step2: %d/%d team-years done", i + 1, total)
        except Exception as e:
            logger.error("Step2: %s failed: %s", url, e)
            checkpoint.mark_error("step2", key)

    logger.info("Step2 complete: %d unique games", len(game_ids))
