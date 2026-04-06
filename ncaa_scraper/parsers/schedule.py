"""Parser for NCAA team schedule pages."""
import re
from bs4 import BeautifulSoup


def _safe_int(val: str) -> int | None:
    try:
        return int(str(val).strip().replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_schedule_page(html: str, team_id: int, year: int) -> list[dict]:
    """
    Parse NCAA team schedule page.

    URL: https://basketball.realgm.com/ncaa/conferences/{ConfCode}/{ConfID}/{TeamCode}/{TeamID}/schedule/{Year}

    Returns list of dicts:
        GameID(int), Date(str), Versus(str), Location(str: H/A/N),
        WL(str), TeamScore(int|None), OppScore(int|None), TeamID(int), Year(int)
    """
    soup = BeautifulSoup(html, "lxml")
    results = []
    seen_game_ids = set()

    # Game/boxscore links pattern
    GAME_LINK_RE = re.compile(r"/ncaa/(?:boxscore|scores)/([^/]+)/([^/]+)/(\d+)")

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Look for a game link in any cell
        game_link = None
        for cell in cells:
            a = cell.find("a", href=GAME_LINK_RE)
            if a:
                game_link = a
                break

        if game_link is None:
            continue

        href = game_link["href"]
        m = GAME_LINK_RE.search(href)
        if not m:
            continue

        try:
            game_id = int(m.group(3))
        except ValueError:
            continue

        date_slug = m.group(1)   # URL slug, e.g. "October-26-2025" — used to build boxscore URL
        versus = m.group(2)

        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)

        date_str = date_slug  # fallback

        # Determine location from opponent cell text
        # Look for opponent cell (usually has @ or *)
        opp_cell_text = ""
        for cell in cells:
            text = cell.get_text(strip=True)
            if "@" in text or "*" in text:
                opp_cell_text = text
                break

        if "@" in opp_cell_text:
            location = "A"
        elif "*" in opp_cell_text:
            location = "N"
        else:
            location = "H"

        # Try to find the actual opponent name from versus string
        # Versus is the URL slug like "Duke-at-North-Carolina"

        # Extract score info: look for W/L indicators and scores
        wl = ""
        team_score = None
        opp_score = None

        cell_texts = [c.get_text(strip=True) for c in cells]

        # Find header row to map columns
        table = row.find_parent("table")
        if table:
            header_row = table.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
                col_map = {h: i for i, h in enumerate(headers)}

                def get_cell(name, default=""):
                    idx = col_map.get(name)
                    if idx is not None and idx < len(cell_texts):
                        return cell_texts[idx]
                    return default

                date_str = get_cell("Date") or date_str
                wl_text = get_cell("W/L") or get_cell("WL") or get_cell("Result") or get_cell("Score")

                # Parse W/L and scores from text like "W 75-60" or "L 55-70"
                score_match = re.search(r"([WL])\s*(\d+)\s*-\s*(\d+)", wl_text)
                if score_match:
                    wl = score_match.group(1)
                    team_score = int(score_match.group(2))
                    opp_score = int(score_match.group(3))
                else:
                    wl = wl_text[:1] if wl_text else ""

        results.append({
            "GameID": game_id,
            "Date": date_str,
            "DateSlug": date_slug,
            "Versus": versus,
            "Location": location,
            "WL": wl,
            "TeamScore": team_score,
            "OppScore": opp_score,
            "TeamID": team_id,
            "Year": year,
        })

    return results
