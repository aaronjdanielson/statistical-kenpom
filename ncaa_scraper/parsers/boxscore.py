"""Parser for NCAA boxscore pages."""
import re
from bs4 import BeautifulSoup


def _safe_int(val: str) -> int:
    try:
        return int(str(val).strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0


def _safe_float(val: str) -> float | None:
    s = str(val).strip()
    if s in ("", "-", "N/A", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_combined(val: str) -> tuple[int, int]:
    """Parse 'FGM-A' style combined stat (e.g. '21-57') into (made, attempted)."""
    s = str(val).strip()
    if "-" in s:
        parts = s.split("-")
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
    return 0, 0


def parse_boxscore_page(html: str, game_id: int, season: int = 0) -> list[dict]:
    """
    Parse NCAA boxscore page.

    URL: https://basketball.realgm.com/ncaa/boxscore/{date}/{versus}/{GameID}

    Args:
        html: Page HTML.
        game_id: Numeric game identifier.
        season: Season end-year (e.g. 2026), passed in from schedules.

    Returns list of 2 dicts (one per team):
        GameID(int), TeamID(int), TeamCode(str), Season(int), Home(int),
        Minutes(int), FGM(int), FGA(int), FG3M(int), FG3A(int),
        FTM(int), FTA(int), OREB(int), DREB(int), REB(int),
        AST(int), PF(int), STL(int), TO(int), BLK(int), PTS(int),
        POSS(float)  — estimated via FGA - OREB + TO + 0.44*FTA
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    TEAM_CONF_RE = re.compile(r"/ncaa/conferences/([^/]+)/(\d+)/([^/]+)/(\d+)/")

    # --- Extract team IDs from the score header h2 ---
    # The score h2 contains both team links, e.g.:
    # <h2>...<a href="/ncaa/conferences/.../Maine/10/...">Maine</a> 62, ...<a href="...">Duke</a> 96</h2>
    team_order = []  # list of (team_code, team_id) in score display order
    for h2 in soup.find_all("h2"):
        links = h2.find_all("a", href=TEAM_CONF_RE)
        if len(links) >= 2:
            for a in links:
                m = TEAM_CONF_RE.search(a["href"])
                if m:
                    try:
                        tid = int(m.group(4))
                    except ValueError:
                        tid = 0
                    team_order.append((m.group(3), tid))
            break

    # --- Find the two player-stat tables ---
    # Identified by headers containing "Player" and "PTS" and "FGM-A"
    player_tables = []
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        if not header_row:
            continue
        hdrs = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        if "Player" in hdrs and "PTS" in hdrs and "FGM-A" in hdrs:
            player_tables.append((table, hdrs))

    for idx, (table, hdrs) in enumerate(player_tables[:2]):
        col_map = {h: i for i, h in enumerate(hdrs)}

        # Assign team from score header order
        team_code = ""
        team_id_val = 0
        if idx < len(team_order):
            team_code, team_id_val = team_order[idx]

        # home/away: second team in the score h2 is listed second (typically home)
        home = 1 if idx == 1 else 0

        # Find the totals row: last data row where minutes column has a numeric value
        # Rows after the team players: "Team" row (col1 = "Team"), then totals, then percentages
        totals_row = None
        min_col = col_map.get("Min", 4)
        pts_col = col_map.get("PTS", len(hdrs) - 1)

        data_rows = table.find_all("tr")[1:]  # skip header
        for row in reversed(data_rows):
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) <= min_col:
                continue
            min_text = cells[min_col]
            if ":" in min_text:  # player row (e.g. "30:00")
                continue
            # Try parsing as integer (total minutes row)
            try:
                int(min_text)
                totals_row = cells
                break
            except ValueError:
                continue

        if totals_row is None:
            continue

        def get_col(name, default="0"):
            i = col_map.get(name)
            if i is not None and i < len(totals_row):
                return totals_row[i]
            return default

        minutes = _safe_int(get_col("Min"))

        fgm_val = get_col("FGM-A")
        fg3m_val = get_col("3PM-A")
        ftm_val = get_col("FTM-A")

        fgm, fga = _parse_combined(fgm_val)
        fg3m, fg3a = _parse_combined(fg3m_val)
        ftm, fta = _parse_combined(ftm_val)

        oreb = _safe_int(get_col("Off"))
        dreb = _safe_int(get_col("Def"))
        reb = _safe_int(get_col("Reb"))
        ast = _safe_int(get_col("Ast"))
        pf = _safe_int(get_col("PF"))
        stl = _safe_int(get_col("STL"))
        to = _safe_int(get_col("TO"))
        blk = _safe_int(get_col("BLK"))
        pts = _safe_int(get_col("PTS"))

        poss = round(fga - oreb + to + 0.44 * fta, 3)

        results.append({
            "GameID": game_id,
            "TeamID": team_id_val,
            "TeamCode": team_code,
            "Season": season,
            "Home": home,
            "Minutes": minutes,
            "FGM": fgm,
            "FGA": fga,
            "FG3M": fg3m,
            "FG3A": fg3a,
            "FTM": ftm,
            "FTA": fta,
            "OREB": oreb,
            "DREB": dreb,
            "REB": reb,
            "AST": ast,
            "PF": pf,
            "STL": stl,
            "TO": to,
            "BLK": blk,
            "PTS": pts,
            "POSS": poss,
        })

    return results
