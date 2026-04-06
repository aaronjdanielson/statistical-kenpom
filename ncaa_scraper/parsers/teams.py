"""Parser for NCAA teams listing pages."""
import re
from bs4 import BeautifulSoup


def parse_teams_page(html: str, year: int) -> list[dict]:
    """
    Parse NCAA teams page and return list of team dicts.

    URL: https://basketball.realgm.com/ncaa/teams/{year}

    Returns list of dicts with keys:
        year, ConferenceCode, ConferenceID(int), TeamCode, TeamID(int), School
    """
    soup = BeautifulSoup(html, "lxml")
    seen = set()
    results = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match /ncaa/conferences/{ConferenceCode}/{ConferenceID}/{TeamCode}/{TeamID}/
        if not href.startswith("/ncaa/conferences/"):
            continue
        parts = href.strip("/").split("/")
        # parts: ['ncaa', 'conferences', ConferenceCode, ConferenceID, TeamCode, TeamID, ...]
        if len(parts) < 6:
            continue
        try:
            conf_id = int(parts[3])
            team_id = int(parts[5])
        except (ValueError, IndexError):
            continue

        conf_code = parts[2]
        team_code = parts[4]
        school = a.get_text(strip=True)

        # Strip AP/coaches poll rank prefix, e.g. "#1 Duke" → "Duke"
        school = re.sub(r"^#?\d+\s+", "", school).strip()

        # Skip empty school names or purely numeric names (likely rank rows)
        if not school or school.isdigit():
            continue

        dedup_key = (team_id, year)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        results.append({
            "year": year,
            "ConferenceCode": conf_code,
            "ConferenceID": conf_id,
            "TeamCode": team_code,
            "TeamID": team_id,
            "School": school,
        })

    return results
