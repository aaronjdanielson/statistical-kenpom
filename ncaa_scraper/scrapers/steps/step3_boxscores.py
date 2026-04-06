"""Step 3: Scrape team boxscores (parallel workers)."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ncaa_scraper.config import REALGM_BASE, HEADERS, RATE_LIMIT_RPS, RATE_LIMIT_JITTER
from ncaa_scraper.parsers.boxscore import parse_boxscore_page
from ncaa_scraper.scrapers.client import RealGMClient

logger = logging.getLogger(__name__)


def run_step3(checkpoint, db, workers: int = 1):
    """
    Scrape boxscores for all games in the schedules table.

    Each worker gets its own RealGMClient session (curl-cffi sessions are not
    thread-safe). The DB and checkpoint both use internal locks.

    Args:
        checkpoint: Checkpoint instance.
        db: Database instance.
        workers: Number of parallel workers (default 1; safe up to ~5).
    """
    game_queue = db.game_queue(checkpoint)
    total = len(game_queue)
    logger.info("Step3: %d boxscores to fetch across %d worker(s)", total, workers)

    if total == 0:
        logger.info("Step3: nothing to do")
        return

    done_count = 0
    error_count = 0
    counter_lock = threading.Lock()
    t0 = time.monotonic()

    def fetch(gid: int, date_slug: str, versus: str, season: int, client: RealGMClient) -> None:
        nonlocal done_count, error_count
        url = f"{REALGM_BASE}/ncaa/boxscore/{date_slug}/{versus}/{gid}"
        try:
            html = client.get(url)
            rows = parse_boxscore_page(html, gid, season)
            if len(rows) != 2:
                logger.warning("Step3: GameID %d returned %d rows (expected 2)", gid, len(rows))
            db.insert_boxscores(rows)
            checkpoint.mark_done("step3", f"box_{gid}")
            with counter_lock:
                done_count += 1
                d = done_count
        except Exception as e:
            logger.error("Step3: GameID %d failed: %s", gid, e)
            checkpoint.mark_error("step3", f"box_{gid}")
            with counter_lock:
                error_count += 1
                d = done_count

        with counter_lock:
            n = done_count + error_count
        if n % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = done_count / elapsed if elapsed > 0 else 0
            remaining = total - n
            eta_h = (remaining / rate / 3600) if rate > 0 else 0
            logger.info(
                "Step3: %d/%d done, %d errors | %.2f req/s | ETA %.1fh",
                done_count, total, error_count, rate, eta_h,
            )

    # One client per worker — each has its own curl-cffi session
    clients = [
        RealGMClient(HEADERS, RATE_LIMIT_RPS, RATE_LIMIT_JITTER)
        for _ in range(workers)
    ]

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(fetch, gid, slug, versus, season, clients[i % workers])
                for i, (gid, slug, versus, season) in enumerate(game_queue)
            ]
            for f in as_completed(futures):
                f.result()  # re-raises any unexpected exception
    finally:
        for c in clients:
            c.close()

    logger.info("Step3 complete: %d fetched, %d errors", done_count, error_count)
