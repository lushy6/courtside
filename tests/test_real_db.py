"""The safety stack against the REAL merged database, not the 3-table fixture.

Review finding: every other safety test runs on a tiny fixture, so anything specific to the
production schema (views, virtual tables, more tables than the catalog) went unproven. These
tests copy data/courtside_nba.db to a temp dir and replay the whole attack corpus against it.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest
from attacks import ABUSE_ATTACKS, WRITE_ATTACKS

from courtside.db import list_tables, open_readonly, run_query
from courtside.errors import QueryError

REAL = Path(__file__).resolve().parents[1] / "data" / "courtside_nba.db"
pytestmark = pytest.mark.skipif(not REAL.exists(), reason="run scripts/build_courtside_db.py")


@pytest.fixture(scope="module")
def real_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dst = tmp_path_factory.mktemp("real") / "courtside_nba.db"
    shutil.copyfile(REAL, dst)
    return dst


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_database_has_only_plain_tables(real_copy: Path) -> None:
    """No views, virtual tables, FTS or triggers: nothing that widens what a SELECT can reach."""
    con = sqlite3.connect(real_copy)
    kinds = {r[0] for r in con.execute("SELECT DISTINCT type FROM sqlite_master")}
    assert kinds <= {"table", "index"}, kinds
    sqls = [r[0] or "" for r in con.execute("SELECT sql FROM sqlite_master WHERE type = 'table'")]
    assert not any("VIRTUAL" in s.upper() for s in sqls)


def test_catalog_helper_lists_every_table_in_the_file(real_copy: Path) -> None:
    con = open_readonly(real_copy)
    tables = list_tables(con)
    assert {"lineup_stats", "players", "contracts", "provenance"} <= set(tables)
    raw = sqlite3.connect(real_copy)
    expected = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(tables) == expected


@pytest.mark.parametrize("sql", WRITE_ATTACKS + ABUSE_ATTACKS)
def test_whole_attack_corpus_is_refused_on_the_real_schema(sql: str, real_copy: Path) -> None:
    before = _sha(real_copy)
    con = open_readonly(real_copy)
    tables = list_tables(con)
    with pytest.raises(QueryError):
        run_query(con, sql, allowed_tables=tables, timeout_s=0.5, max_rows=100)
    con.close()
    assert _sha(real_copy) == before


def test_a_realistic_question_runs_on_the_real_data(real_copy: Path) -> None:
    con = open_readonly(real_copy)
    res = run_query(
        con,
        "SELECT team, count(*) AS lineups, sum(distinguishable) AS separable "
        "FROM lineup_stats WHERE poss >= 100 GROUP BY team ORDER BY lineups DESC LIMIT 3",
        allowed_tables=list_tables(con),
    )
    assert res.columns == ["team", "lineups", "separable"] and len(res.rows) == 3
