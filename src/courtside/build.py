"""Merge the sibling projects' databases into one read-only file for Courtside.

Two source databases overlap (``games``, ``players``, ``teams``, ``player_game_stats``), so a
merge is only safe if they actually agree. ``build`` therefore runs a list of consistency
checks first and refuses to build when any of them finds a difference, instead of silently
preferring one side. It copies each table with its original DDL and indexes, verifies row
counts, records where every table came from, and never opens a source for writing.

The output is deterministic: the same sources give the same bytes, so a hash identifies
a build and no timestamp is stored inside the database.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LINEUP = "lineup"
TACTICAL = "tactical"


class BuildError(RuntimeError):
    """Sources are missing a table, disagree, or the copy lost rows."""


@dataclass(frozen=True)
class BuildSpec:
    """What to copy from where, and what must hold before copying.

    Each check is ``(label, sql)`` where ``sql`` returns one number, the count of
    disagreements, and must return 0. SQL may reference the ``lineup`` and ``tactical``
    schemas. Table names here are constants from this file, never user input, which is why
    they are interpolated into SQL below.
    """

    lineup_tables: tuple[str, ...]
    tactical_tables: tuple[str, ...]
    checks: tuple[tuple[str, str], ...] = ()


def _identical(table: str) -> tuple[str, str]:
    return (
        f"{table} identical in both sources",
        f"SELECT (SELECT count(*) FROM (SELECT * FROM {LINEUP}.{table} "
        f"EXCEPT SELECT * FROM {TACTICAL}.{table})) + "
        f"(SELECT count(*) FROM (SELECT * FROM {TACTICAL}.{table} "
        f"EXCEPT SELECT * FROM {LINEUP}.{table}))",
    )


NBA_SPEC = BuildSpec(
    # Lineup tables come from nba-lineup-analytics, the authority for everything lineup-shaped.
    lineup_tables=(
        "games",
        "teams",
        "players",
        "lineup_stats",
        "lineup_bootstrap",
        "player_rapm",
        "player_season",
        "noise_model",
        "coverage_season",
        "coverage_team",
    ),
    # player_game_stats comes from the tactical optimizer: a strict superset (adds steals,
    # blocks, oreb, dreb); the check below proves it agrees on every shared column.
    tactical_tables=(
        "player_game_stats",
        "contracts",
        "cap_rules",
        "play_types",
        "player_positions",
        "player_play_type_stats",
        "team_play_type_stats",
    ),
    checks=(
        _identical("games"),
        _identical("players"),
        _identical("teams"),
        (
            "player_game_stats: same rows and same values on shared columns",
            f"""SELECT
                (SELECT count(*) FROM (SELECT game_id, player_id FROM {LINEUP}.player_game_stats
                    EXCEPT SELECT game_id, player_id FROM {TACTICAL}.player_game_stats))
              + (SELECT count(*) FROM (SELECT game_id, player_id FROM {TACTICAL}.player_game_stats
                    EXCEPT SELECT game_id, player_id FROM {LINEUP}.player_game_stats))
              + (SELECT count(*) FROM {LINEUP}.player_game_stats a
                    JOIN {TACTICAL}.player_game_stats b USING (game_id, player_id)
                    WHERE a.team_id IS NOT b.team_id OR a.is_starter IS NOT b.is_starter
                       OR a.minutes IS NOT b.minutes OR a.points IS NOT b.points
                       OR a.rebounds IS NOT b.rebounds OR a.assists IS NOT b.assists
                       OR a.fgm IS NOT b.fgm OR a.fga IS NOT b.fga
                       OR a.fg3m IS NOT b.fg3m OR a.fg3a IS NOT b.fg3a
                       OR a.ftm IS NOT b.ftm OR a.fta IS NOT b.fta
                       OR a.turnovers IS NOT b.turnovers OR a.plus_minus IS NOT b.plus_minus
                       OR a.rebounds_off IS NOT b.oreb)""",
        ),
        (
            "every contract belongs to a known player",
            f"SELECT count(*) FROM {TACTICAL}.contracts "
            f"WHERE player_id NOT IN (SELECT player_id FROM {LINEUP}.players)",
        ),
    ),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _uri(path: Path, query: str = "") -> str:
    return f"{path.resolve().as_uri()}{query}"


def build(
    out_path: Path,
    lineup_db: Path,
    tactical_db: Path,
    spec: BuildSpec = NBA_SPEC,
) -> dict[str, Any]:
    """Build ``out_path`` and return a manifest.

    Raises ``BuildError`` on any problem, and an existing output is left untouched.
    """
    for label, p in ((LINEUP, lineup_db), (TACTICAL, tactical_db)):
        if not p.is_file():
            raise BuildError(f"{label} source database not found: {p}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.unlink(missing_ok=True)

    source_hashes = {LINEUP: sha256_file(lineup_db), TACTICAL: sha256_file(tactical_db)}
    con = sqlite3.connect(_uri(tmp), uri=True)
    try:
        con.execute(f"ATTACH DATABASE ? AS {LINEUP}", (_uri(lineup_db, "?mode=ro&immutable=1"),))
        con.execute(
            f"ATTACH DATABASE ? AS {TACTICAL}", (_uri(tactical_db, "?mode=ro&immutable=1"),)
        )
        _require_tables(con, spec)
        passed = _run_checks(con, spec)

        con.execute(
            "CREATE TABLE provenance (table_name TEXT PRIMARY KEY, source TEXT NOT NULL, "
            "source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL, rows INTEGER NOT NULL)"
        )
        tables: dict[str, dict[str, Any]] = {}
        files = {LINEUP: lineup_db.name, TACTICAL: tactical_db.name}
        for alias, names in ((LINEUP, spec.lineup_tables), (TACTICAL, spec.tactical_tables)):
            for name in names:
                rows = _copy_table(con, alias, name)
                tables[name] = {"rows": rows, "source": alias}
                con.execute(
                    "INSERT INTO provenance VALUES (?, ?, ?, ?, ?)",
                    (name, alias, files[alias], source_hashes[alias], rows),
                )
        con.commit()
        con.execute(f"DETACH DATABASE {LINEUP}")
        con.execute(f"DETACH DATABASE {TACTICAL}")
        con.execute("VACUUM")
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    os.replace(tmp, out_path)

    return {
        "out_file": out_path.name,
        "out_sha256": sha256_file(out_path),
        "out_bytes": out_path.stat().st_size,
        "sources": {
            LINEUP: {"file": lineup_db.name, "sha256": source_hashes[LINEUP]},
            TACTICAL: {"file": tactical_db.name, "sha256": source_hashes[TACTICAL]},
        },
        "tables": tables,
        "checks_passed": passed,
    }


def _require_tables(con: sqlite3.Connection, spec: BuildSpec) -> None:
    for alias, names in ((LINEUP, spec.lineup_tables), (TACTICAL, spec.tactical_tables)):
        for name in names:
            found = con.execute(
                f"SELECT 1 FROM {alias}.sqlite_master WHERE type = 'table' AND name = ?", (name,)
            ).fetchone()
            if found is None:
                raise BuildError(f"{alias} database has no table {name!r}")


def _run_checks(con: sqlite3.Connection, spec: BuildSpec) -> list[str]:
    passed: list[str] = []
    for label, sql in spec.checks:
        try:
            bad = con.execute(sql).fetchone()[0]
        except sqlite3.Error as err:
            raise BuildError(f"consistency check could not run ({label}): {err}") from err
        if bad:
            raise BuildError(f"sources disagree: {label} ({bad} differing rows)")
        passed.append(label)
    return passed


def _copy_table(con: sqlite3.Connection, alias: str, name: str) -> int:
    ddl = con.execute(
        f"SELECT sql FROM {alias}.sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()[0]
    con.execute(ddl)
    con.execute(f'INSERT INTO main."{name}" SELECT * FROM {alias}."{name}" ORDER BY rowid')
    for (index_sql,) in con.execute(
        f"SELECT sql FROM {alias}.sqlite_master "
        "WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL ORDER BY name",
        (name,),
    ).fetchall():
        con.execute(index_sql)
    copied = con.execute(f'SELECT count(*) FROM main."{name}"').fetchone()[0]
    expected = con.execute(f'SELECT count(*) FROM {alias}."{name}"').fetchone()[0]
    if copied != expected:
        raise BuildError(f"{name}: copied {copied} rows but the source has {expected}")
    return int(copied)
