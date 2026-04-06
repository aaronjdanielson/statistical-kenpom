from pathlib import Path

OUTPUT_DIR    = Path("~/Dropbox/kenpom/output").expanduser()
DB_PATH       = Path("~/Dropbox/kenpom/checkpoints.db").expanduser()
DATA_DB_PATH  = Path("~/Dropbox/kenpom/ncaa.db").expanduser()

YEARS = list(range(2002, 2027))

REALGM_BASE = "https://basketball.realgm.com"

RATE_LIMIT_RPS    = 0.8
RATE_LIMIT_JITTER = 0.4

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
