"""Layers 2 and 3, each tested ALONE, plus the executor limits and the full pipeline."""

from __future__ import annotations

import hashlib
import sqlite3
import time
import tracemalloc
from pathlib import Path

import pytest
from attacks import ABUSE_ATTACKS, WRITE_ATTACKS
from conftest import TABLES

from courtside.db import execute, install_authorizer, open_readonly, run_query
from courtside.errors import QueryError, QueryFailed


def _fingerprint(db_path: Path) -> tuple[str, list[str], int]:
    """Bytes of the db file, names of files beside it, and a live row count."""
    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    siblings = sorted(p.name for p in db_path.parent.iterdir())
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT count(*) FROM players").fetchone()[0]
    finally:
        con.close()
    return digest, siblings, rows


def _fire(con: sqlite3.Connection, sql: str) -> None:
    try:
        con.execute(sql)
        con.commit()
    except (sqlite3.Error, ValueError):
        pass


# --- layer 2 alone: authorizer on a connection that CAN write ------------------------------
@pytest.mark.parametrize("sql", WRITE_ATTACKS)
def test_layer2_authorizer_alone_stops_writes(
    sql: str, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(db_path.parent)
    before = _fingerprint(db_path)
    con = sqlite3.connect(db_path)  # read-write on purpose; no validator, no ro flag
    install_authorizer(con)
    _fire(con, sql)
    con.close()
    assert _fingerprint(db_path) == before


# --- layer 3 alone: read-only handle with NO authorizer ------------------------------------
@pytest.mark.parametrize("sql", WRITE_ATTACKS)
def test_layer3_readonly_handle_alone_stops_writes(
    sql: str, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(db_path.parent)
    before = _fingerprint(db_path)
    con = open_readonly(db_path, authorizer=False)
    _fire(con, sql)
    con.close()
    assert _fingerprint(db_path) == before


def test_open_readonly_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_readonly(tmp_path / "missing.db")


def test_readonly_handle_really_is_readonly(db_path: Path) -> None:
    con = open_readonly(db_path, authorizer=False)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM players")
    con.close()


def test_authorizer_denies_reading_the_schema_table(db_path: Path) -> None:
    con = open_readonly(db_path)
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("SELECT * FROM sqlite_master").fetchall()
    con.close()


def test_authorizer_still_allows_ordinary_reads(db_path: Path) -> None:
    con = open_readonly(db_path)
    assert con.execute("SELECT count(*) FROM players").fetchone()[0] == 3
    con.close()


# --- executor limits ------------------------------------------------------------------------
def test_row_cap_truncates_and_says_so(db_path: Path) -> None:
    con = open_readonly(db_path)
    res = execute(con, "SELECT * FROM lineup_stats", max_rows=10)
    assert len(res.rows) == 10 and res.truncated is True
    assert res.columns[:2] == ["season", "team"]


def test_no_truncation_flag_when_under_the_cap(db_path: Path) -> None:
    con = open_readonly(db_path)
    res = execute(con, "SELECT * FROM players", max_rows=10)
    assert len(res.rows) == 3 and res.truncated is False


def test_long_cells_are_clipped(db_path: Path) -> None:
    con = open_readonly(db_path)
    res = execute(con, "SELECT printf('%5000s', 'x') AS big", max_cell_chars=50)
    assert len(res.rows[0][0]) <= 60


def test_time_limit_kills_a_runaway_query(db_path: Path) -> None:
    con = open_readonly(db_path)
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    start = time.monotonic()
    with pytest.raises(QueryFailed) as exc:
        execute(con, sql, timeout_s=0.5)
    assert time.monotonic() - start < 5
    assert exc.value.kind == "timeout"


def test_timeout_does_not_poison_the_next_query(db_path: Path) -> None:
    con = open_readonly(db_path)
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    with pytest.raises(QueryFailed):
        execute(con, sql, timeout_s=0.3)
    assert execute(con, "SELECT count(*) FROM players").rows == [(3,)]


def test_sql_errors_become_model_readable_failures(db_path: Path) -> None:
    con = open_readonly(db_path)
    with pytest.raises(QueryFailed) as exc:
        execute(con, "SELECT no_such_column FROM players")
    assert "no_such_column" in str(exc.value)


def test_memory_is_bounded_before_clipping_not_after(db_path: Path) -> None:
    """40 rows x 20 columns x 90 KB is 72 MB if every cell is held before it is clipped."""
    cols = ", ".join(f"printf('%90000s', 'x') AS c{i}" for i in range(20))
    sql = f"WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 40) SELECT {cols} FROM r"
    con = open_readonly(db_path)
    tracemalloc.start()
    res = execute(con, sql, max_rows=1000, max_cell_chars=50)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(res.rows) > 0
    assert peak < 20_000_000, f"peak {peak / 1e6:.0f} MB: cells were held unclipped"


def test_total_output_budget_truncates_wide_results(db_path: Path) -> None:
    con = open_readonly(db_path)
    res = execute(con, "SELECT * FROM lineup_stats", max_rows=1000, max_total_chars=500)
    assert res.truncated is True and 0 < len(res.rows) < 60


def test_single_cell_cannot_exceed_the_length_limit(db_path: Path) -> None:
    con = open_readonly(db_path)
    with pytest.raises(QueryFailed):
        # 180,000 bytes, built under printf's own width cap, exceeds MAX_VALUE_BYTES (100,000).
        execute(con, "SELECT replace(printf('%90000s', ''), ' ', 'ab')")


def test_column_count_is_capped(db_path: Path) -> None:
    con = open_readonly(db_path)
    sql = "SELECT " + ", ".join(str(i) for i in range(300))
    with pytest.raises(QueryFailed):
        execute(con, sql)


def test_a_sort_heavy_query_still_respects_the_time_limit(db_path: Path) -> None:
    """Review finding 3: a big final sort is one VM step. Measured: the limit still holds,
    because the rows feeding the sort are produced under the progress handler."""
    con = open_readonly(db_path)
    sql = (
        "SELECT a.poss FROM lineup_stats a, lineup_stats b, lineup_stats c, lineup_stats d "
        "ORDER BY d.net_rating DESC, a.poss"
    )
    start = time.monotonic()
    with pytest.raises(QueryFailed) as exc:
        execute(con, sql, timeout_s=0.5)
    assert time.monotonic() - start < 3.0
    assert exc.value.kind == "timeout"


def test_authorizer_denies_reads_outside_the_main_database() -> None:
    from courtside.db import _authorize

    assert (
        _authorize(sqlite3.SQLITE_READ, "players", "first_name", "main", None) == sqlite3.SQLITE_OK
    )
    assert (
        _authorize(sqlite3.SQLITE_READ, "players", "first_name", "temp", None)
        == sqlite3.SQLITE_DENY
    )
    assert (
        _authorize(sqlite3.SQLITE_READ, "players", "first_name", "evil", None)
        == sqlite3.SQLITE_DENY
    )


@pytest.mark.parametrize(
    "call",
    ["hex('a')", "unhex('41')", "random()", "randomblob(4)", "zeroblob(4)", "quote('a')",
     "sqlite_version()", "json_extract('{}', '$')", "load_extension('x')", "readfile('x')"],
)  # fmt: skip
def test_layer2_alone_refuses_functions_outside_the_allowlist(call: str, db_path: Path) -> None:
    """The authorizer is the SECOND wall for functions: prove it without layer 1 in front."""
    con = sqlite3.connect(db_path)
    install_authorizer(con)
    with pytest.raises(sqlite3.DatabaseError):
        con.execute(f"SELECT {call}").fetchall()
    con.close()


def test_unexpected_exceptions_become_query_failures(db_path: Path) -> None:
    class Boom:
        def execute(self, *_: object) -> None:
            raise OverflowError("boom")

        def set_progress_handler(self, *_: object) -> None:
            return None

    with pytest.raises(QueryFailed):
        execute(Boom(), "SELECT 1")  # type: ignore[arg-type]


# --- the full pipeline ------------------------------------------------------------------------
@pytest.mark.parametrize("sql", WRITE_ATTACKS + ABUSE_ATTACKS)
def test_pipeline_refuses_every_attack(sql: str, db_path: Path) -> None:
    before = _fingerprint(db_path)
    con = open_readonly(db_path)
    with pytest.raises(QueryError) as exc:
        run_query(con, sql, allowed_tables=TABLES, timeout_s=0.5, max_rows=100)
    con.close()
    assert _fingerprint(db_path) == before
    # Refused for the intended reason, not because an unrelated table name happened to be wrong.
    if "nonexistent_table" not in sql:
        assert exc.value.kind != "unknown_table", sql


def test_pipeline_runs_a_good_query(db_path: Path) -> None:
    con = open_readonly(db_path)
    res = run_query(
        con,
        "SELECT family_name FROM players WHERE player_id = 1",
        allowed_tables=TABLES,
    )
    assert res.rows == [("Tatum",)]
