"""Database schema and management for Polish Football Attendance prediction."""
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "attendance.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stadium_capacities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    club_name TEXT NOT NULL,
    stadium_name TEXT NOT NULL,
    city TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    valid_from TEXT NOT NULL, -- YYYY-MM-DD
    valid_to TEXT NOT NULL,   -- YYYY-MM-DD
    capacity INTEGER NOT NULL,
    roof_pct REAL DEFAULT 1.0, -- 0.0 to 1.0
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_stadium_club ON stadium_capacities (club_name, valid_from, valid_to);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY, -- 90minut id_mecz
    season TEXT NOT NULL,         -- e.g. '2022/2023'
    league_tier INTEGER NOT NULL, -- 1=Ekstraklasa, 2=1 Liga, 3=2 Liga
    league_name TEXT NOT NULL,
    matchday INTEGER,             -- kolejka
    match_date TEXT,              -- YYYY-MM-DD
    match_time TEXT,              -- HH:MM
    match_datetime TEXT,          -- ISO format YYYY-MM-DD HH:MM:SS
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    stadium_reported TEXT,
    city_reported TEXT,
    attendance INTEGER,
    capacity_resolved INTEGER,
    occupancy_pct REAL,           -- attendance / capacity_resolved
    referee TEXT,
    news_url TEXT,
    is_outlier INTEGER DEFAULT 0, -- 1 if flagged (boycott, closed doors, etc.)
    outlier_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches (home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_matches_season_tier ON matches (season, league_tier);

CREATE TABLE IF NOT EXISTS weather (
    match_id INTEGER PRIMARY KEY REFERENCES matches(match_id),
    temperature_c REAL,
    precipitation_mm REAL,
    rain_mm REAL,
    weather_code INTEGER,
    wind_speed_kmh REAL,
    cloud_cover_pct REAL,
    source TEXT DEFAULT 'open-meteo',
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS club_locations (
    club_name TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL
);
"""


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get SQLite database connection with row factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")

