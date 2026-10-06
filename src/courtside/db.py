"""Layers 2 and 3 of the SQL safety stack, and the bounded executor.

Layer 2  a SQLite authorizer that allows only reads, selects and allowlisted functions.
Layer 3  a read-only, immutable file handle.

Each is designed to stop a write on its own (tests/test_db.py runs every write attack
against each layer alone). The executor adds a row cap, a wall-clock limit and a cell
clip, so a ``SELECT *`` over a 400k-row table cannot flood the model's context.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtside.errors import QueryFailed
from courtside.validate import ALLOWED_FUNCTIONS, validate_select

_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    elapsed_s: float


def _authorize(action: int, arg1: str | None, arg2: str | None, *_: object) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and (arg1 or "").lower().startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in ALLOWED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def install_authorizer(con: sqlite3.Connection) -> None:
    """Layer 2. Deny everything except reads, selects and allowlisted functions."""
    con.set_authorizer(_authorize)


def open_readonly(path: str | Path, *, authorizer: bool = True) -> sqlite3.Connection:
    """Layer 3 (and layer 2 unless ``authorizer=False``, which exists for the layer tests).

    ``mode=ro`` refuses writes at the file-handle level; ``immutable=1`` tells SQLite the
    file cannot change under it, so it takes no locks and never writes a journal.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Database file not found: {p}")
    con = sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    # A read-only handle protects THIS file but would still let ATTACH create a new one
    # (found by the layer-3-alone test), so cap attached databases at zero. The length
    # cap bounds any single string or blob an expression can build.
    con.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
    if authorizer:
        install_authorizer(con)
    return con


def _clip(value: Any, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


def execute(
    con: sqlite3.Connection,
    sql: str,
    *,
    max_rows: int = 1000,
    timeout_s: float = 3.0,
    max_cell_chars: int = 200,
) -> QueryResult:
    """Run ``sql`` with a row cap and a wall-clock limit. Does not validate; see ``run_query``."""
    start = time.monotonic()
    deadline = start + timeout_s
    # Called every 1000 SQLite VM steps; returning non-zero aborts the statement.
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        cur = con.execute(sql)
        fetched = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description] if cur.description else []
        cur.close()
    except sqlite3.OperationalError as err:
        if "interrupt" in str(err).lower():
            raise QueryFailed(
                "timeout",
                f"The query ran longer than {timeout_s:g}s and was stopped. "
                "Add filters, a LIMIT, or aggregate in SQL.",
            ) from err
        raise QueryFailed("sql_error", f"SQLite error: {err}") from err
    except (sqlite3.DatabaseError, sqlite3.Warning, sqlite3.ProgrammingError, ValueError) as err:
        raise QueryFailed("sql_error", f"SQLite error: {err}") from err
    finally:
        con.set_progress_handler(None, 0)

    truncated = len(fetched) > max_rows
    rows = [tuple(_clip(v, max_cell_chars) for v in row) for row in fetched[:max_rows]]
    return QueryResult(columns, rows, truncated, time.monotonic() - start)


def run_query(
    con: sqlite3.Connection,
    sql: str,
    *,
    allowed_tables: Iterable[str] | None = None,
    max_rows: int = 1000,
    timeout_s: float = 3.0,
    max_cell_chars: int = 200,
) -> QueryResult:
    """The full pipeline the agent's ``run_sql`` tool calls: validate, then execute bounded."""
    safe = validate_select(sql, allowed_tables)
    return execute(con, safe, max_rows=max_rows, timeout_s=timeout_s, max_cell_chars=max_cell_chars)
