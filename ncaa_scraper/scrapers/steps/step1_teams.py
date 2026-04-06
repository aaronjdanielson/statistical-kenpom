"""Step 1: Scrape team listings per year."""
import logging

from ncaa_scraper.config import REALGM_BASE, YEARS
from ncaa_scraper.parsers.teams import parse_teams_page

logger = logging.getLogger(__name__)


def run_step1(client, checkpoint, db, years=None):
    """
    Scrape team listings for each year, write to teams_years table.

    Returns list of all team dicts (feeds directly into step2).
    """
    if years is None:
        years = YEARS

    all_teams = []
    for year in years:
        key = f"teams_{year}"
        if checkpoint.is_done("step1", key):
            logger.info("Step1: %d already done, skipping", year)
            continue
        url = f"{REALGM_BASE}/ncaa/teams/{year}"
        try:
            html = client.get(url)
            teams = parse_teams_page(html, year)
            db.insert_teams(teams)
            all_teams.extend(teams)
            checkpoint.mark_done("step1", key)
            logger.info("Step1: %d → %d teams", year, len(teams))
        except Exception as e:
            logger.error("Step1: %s failed: %s", url, e)
            checkpoint.mark_error("step1", key)

    logger.info("Step1 complete: %d team-years scraped", len(all_teams))
    return all_teams
