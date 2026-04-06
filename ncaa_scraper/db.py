"""SQLite database for NCAA game data.

Schema
------
teams_years  — one row per (team, season)
schedules    — one row per (game, team) from each team's schedule page
boxscores    — one row per (game, team) with full box stats and possession estimate

All inserts are INSERT OR IGNORE so the pipeline is safe to resume.
WAL mode enables concurrent reads from the analysis layer while scraping writes.
"""
import sqlite3
import threading
from pathlib import Path


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS teams_years (
                TeamID         INTEGER NOT NULL,
                year           INTEGER NOT NULL,
                School         TEXT,
                TeamCode       TEXT,
                ConferenceCode TEXT,
                ConferenceID   INTEGER,
                PRIMARY KEY (TeamID, year)
            );

            CREATE TABLE IF NOT EXISTS schedules (
                GameID     INTEGER NOT NULL,
                TeamID     INTEGER NOT NULL,
                Year       INTEGER,
                Date       TEXT,
                DateSlug   TEXT,
                Versus     TEXT,
                Location   TEXT,
                WL         TEXT,
                TeamScore  INTEGER,
                OppScore   INTEGER,
                PRIMARY KEY (GameID, TeamID)
            );
            CREATE INDEX IF NOT EXISTS idx_schedules_year
                ON schedules(Year);

            CREATE TABLE IF NOT EXISTS boxscores (
                GameID   INTEGER NOT NULL,
                TeamID   INTEGER NOT NULL,
                TeamCode TEXT,
                Season   INTEGER,
                Home     INTEGER,
                Minutes  INTEGER,
                FGM      INTEGER,
                FGA      INTEGER,
                FG3M     INTEGER,
                FG3A     INTEGER,
                FTM      INTEGER,
                FTA      INTEGER,
                OREB     INTEGER,
                DREB     INTEGER,
                REB      INTEGER,
                AST      INTEGER,
                PF       INTEGER,
                STL      INTEGER,
                TOV      INTEGER,
                BLK      INTEGER,
                PTS      INTEGER,
                POSS     REAL,
                PRIMARY KEY (GameID, TeamID)
            );
            CREATE INDEX IF NOT EXISTS idx_boxscores_season
                ON boxscores(Season);
            CREATE INDEX IF NOT EXISTS idx_boxscores_team
                ON boxscores(TeamID);
        """)
        self._conn.commit()

    def insert_teams(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """INSERT OR IGNORE INTO teams_years
               (TeamID, year, School, TeamCode, ConferenceCode, ConferenceID)
               VALUES (:TeamID, :year, :School, :TeamCode, :ConferenceCode, :ConferenceID)""",
            rows,
        )
        self._conn.commit()

    def insert_schedules(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """INSERT OR IGNORE INTO schedules
               (GameID, TeamID, Year, Date, DateSlug, Versus, Location, WL, TeamScore, OppScore)
               VALUES (:GameID, :TeamID, :Year, :Date, :DateSlug, :Versus,
                       :Location, :WL, :TeamScore, :OppScore)""",
            rows,
        )
        self._conn.commit()

    def insert_boxscores(self, rows: list[dict]) -> None:
        if not rows:
            return
        # Rename TO -> TOV to avoid SQL keyword ambiguity
        normed = [{**r, "TOV": r.pop("TO")} if "TO" in r else r for r in rows]
        with self._lock:
            self._conn.executemany(
                """INSERT OR IGNORE INTO boxscores
                   (GameID, TeamID, TeamCode, Season, Home, Minutes,
                    FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB,
                    AST, PF, STL, TOV, BLK, PTS, POSS)
                   VALUES (:GameID, :TeamID, :TeamCode, :Season, :Home, :Minutes,
                           :FGM, :FGA, :FG3M, :FG3A, :FTM, :FTA, :OREB, :DREB, :REB,
                           :AST, :PF, :STL, :TOV, :BLK, :PTS, :POSS)""",
                normed,
            )
            self._conn.commit()

    def game_queue(self, checkpoint) -> list[tuple[int, str, str, int]]:
        """Return (GameID, DateSlug, Versus, Year) for games not yet checkpointed."""
        cur = self._conn.execute(
            """SELECT GameID, DateSlug, Versus, Year
               FROM schedules
               WHERE DateSlug != '' AND Versus != ''
               GROUP BY GameID
               ORDER BY Year, GameID"""
        )
        return [
            (gid, slug, versus, year)
            for gid, slug, versus, year in cur.fetchall()
            if not checkpoint.is_done("step3", f"box_{gid}")
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
