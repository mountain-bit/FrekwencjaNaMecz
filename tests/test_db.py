"""Tests for SQLite database schema and operations."""
import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.db.schema import init_db, get_connection


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    init_db(temp_path)
    yield temp_path
    try:
        if temp_path.exists():
            temp_path.unlink()
    except Exception:
        pass


def test_init_db_creates_tables(temp_db):
    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "matches" in tables
    assert "stadium_capacities" in tables
    assert "weather" in tables
    assert "club_locations" in tables
    conn.close()


def test_insert_and_retrieve_match(temp_db):
    conn = get_connection(temp_db)
    with conn:
        conn.execute(
            """
            INSERT INTO matches (match_id, season, league_tier, league_name, matchday, match_date, match_time, match_datetime, home_team, away_team, attendance, capacity_resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1001, "2023/2024", 1, "Ekstraklasa", 1, "2023-07-22", "17:30", "2023-07-22T17:30:00", "Legia Warszawa", "ŁKS Łódź", 26500, 31103)
        )
    row = conn.execute("SELECT * FROM matches WHERE match_id = 1001").fetchone()
    assert row is not None
    assert row["home_team"] == "Legia Warszawa"
    assert row["attendance"] == 26500
    assert row["capacity_resolved"] == 31103
    conn.close()


def test_weather_foreign_key(temp_db):
    conn = get_connection(temp_db)
    # Inserting weather for non-existent match should fail with FK constraint error
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO weather (match_id, temperature_c) VALUES (?, ?)",
                (999999, 18.5)
            )
    conn.close()
