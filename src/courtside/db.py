"""Layers 2 and 3 of the SQL safety stack, and the bounded executor.

Layer 2  a SQLite authorizer that allows only reads, selects and allowlisted functions.
Layer 3  a read-only, immutable file handle with limits on attached databases, value
         length and column count.

Each is designed to stop a write on its own (tests/test_db.py runs every write attack
against each layer alone). The executor adds a row cap, a wall-clock limit, a cell clip and
a total-output budget, so a ``SELECT *`` over a 400k-row table cannot flood the model's
context, and a wide result cannot balloon memory before it is clipped.
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

# One value may not exceed this many bytes. Real cells here are short; a larger limit only
# helps an expression build a huge string. Rows are clipped as they stream, so memory is
# bounded by (columns x this) per row, not (rows x columns x this).
MAX_VALUE_BYTES = 100_000
MAX_COLUMNS = 100


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    elapsed_s: float


def _authorize(
    action: int, arg1: str | None, arg2: str | None, dbname: str | None = None, *_: object
) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ:
        if (arg1 or "").lower().startswith("sqlite_"):
            return sqlite3.SQLITE_DENY
        if dbname not in (None, "main"):
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
    # (found by the layer-3-alone test), so cap attached databases at zero.
    con.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    con.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, MAX_COLUMNS)
    if authorizer:
        install_authorizer(con)
    return con


def list_tables(con: sqlite3.Connection) -> list[str]:
    """Every table in the file, for use as the validator's catalog.

    Passing the real list (not a hand-written subset) means an error message can never name
    a table the validator would have hidden. The authorizer denies reading sqlite_master, so
    it is lifted for this one constant query and restored in ``finally``.
    """
    con.set_authorizer(None)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    finally:
        install_authorizer(con)
    return [r[0] for r in rows]


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
    max_total_chars: int = 100_000,
) -> QueryResult:
    """Run ``sql`` with row, time and output limits. Does not validate; see ``run_query``.

    Rows are pulled one at a time and clipped immediately, so a row is never held at full
    width after the next is fetched. ``truncated`` is set when any limit stops the listing.
    """
    start = time.monotonic()
    deadline = start + timeout_s
    # Called every 1000 SQLite VM steps; returning non-zero aborts the statement.
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    rows: list[tuple[Any, ...]] = []
    columns: list[str] = []
    truncated = False
    used = 0
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        for raw in cur:
            if len(rows) >= max_rows:
                truncated = True
                break
            row = tuple(_clip(v, max_cell_chars) for v in raw)
            size = sum(len(str(v)) for v in row)
            if rows and used + size > max_total_chars:
                truncated = True
                break
            rows.append(row)
            used += size
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
    except (OverflowError, MemoryError, RecursionError) as err:
        raise QueryFailed(
            "resource", f"The query exceeded a resource limit ({type(err).__name__})."
        ) from err
    finally:
        con.set_progress_handler(None, 0)

    return QueryResult(columns, rows, truncated, time.monotonic() - start)


def run_query(
    con: sqlite3.Connection,
    sql: str,
    *,
    allowed_tables: Iterable[str] | None = None,
    max_rows: int = 1000,
    timeout_s: float = 3.0,
    max_cell_chars: int = 200,
    max_total_chars: int = 100_000,
) -> QueryResult:
    """The full pipeline the agent's ``run_sql`` tool calls: validate, then execute bounded."""
    safe = validate_select(sql, allowed_tables)
    return execute(
        con,
        safe,
        max_rows=max_rows,
        timeout_s=timeout_s,
        max_cell_chars=max_cell_chars,
        max_total_chars=max_total_chars,
    )
