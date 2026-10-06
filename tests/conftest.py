from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

SCHEMA = """
CREATE TABLE players (player_id INTEGER PRIMARY KEY, first_name TEXT, family_name TEXT);
CREATE TABLE teams (team_id INTEGER PRIMARY KEY, abbr TEXT, city TEXT, name TEXT);
CREATE TABLE lineup_stats (
    season TEXT, team TEXT, lineup_uid TEXT, poss INTEGER, net_rating REAL, shrunk REAL
);
"""


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A small on-disk database with the table names the real build uses."""
    path = tmp_path / "fixture.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO players VALUES (?, ?, ?)",
        [(1, "Jayson", "Tatum"), (2, "Jaylen", "Brown"), (3, "Nikola", "Jokic")],
    )
    con.executemany(
        "INSERT INTO teams VALUES (?, ?, ?, ?)",
        [(10, "BOS", "Boston", "Celtics"), (11, "DEN", "Denver", "Nuggets")],
    )
    con.executemany(
        "INSERT INTO lineup_stats VALUES (?, ?, ?, ?, ?, ?)",
        [("2025-26", "BOS", f"u{i}", 10 + i * 20, float(i), i / 2) for i in range(60)],
    )
    con.commit()
    con.close()
    return path


TABLES = frozenset({"players", "teams", "lineup_stats"})
