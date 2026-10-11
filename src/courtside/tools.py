"""Small, bounded tool surface for a future model adapter and agent loop.

Arguments are untrusted model output. Only documented identifiers are accepted,
and SQL still passes all three read-only safety layers before execution.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from courtside.db import QueryResult, run_query
from courtside.errors import QueryError
from courtside.semantic import SemanticLayer


def _result(query: QueryResult) -> dict[str, Any]:
    return {
        "ok": True,
        "columns": query.columns,
        "rows": [list(row) for row in query.rows],
        "truncated": query.truncated,
    }


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"kind": kind, "message": message}}


class ToolRuntime:
    """Expose only schema inspection, distinct samples, and bounded SELECTs."""

    def __init__(self, connection: sqlite3.Connection, layer: SemanticLayer) -> None:
        self.connection = connection
        self.layer = layer

    def call(self, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            return _error("invalid_arguments", "tool arguments must be an object")
        try:
            if name == "describe_table":
                table = arguments.get("name")
                if not isinstance(table, str):
                    return _error("invalid_arguments", "name must be a table name")
                return {"ok": True, "text": self.layer.describe_table(table, self.connection)}

            if name == "sample_values":
                table = arguments.get("table")
                column = arguments.get("column")
                if not isinstance(table, str) or not isinstance(column, str):
                    return _error("invalid_arguments", "table and column must be names")
                if table not in self.layer.table_names or column not in self.layer.columns(table):
                    return _error("invalid_arguments", "table or column is not documented")
                # Names come from the reviewed semantic layer, never from the raw arguments.
                sql = (
                    f'SELECT DISTINCT "{column}" FROM "{table}" '
                    f'WHERE "{column}" IS NOT NULL ORDER BY "{column}"'
                )
                return _result(
                    run_query(
                        self.connection,
                        sql,
                        allowed_tables=self.layer.table_names,
                        max_rows=20,
                    )
                )

            if name == "run_sql":
                query = arguments.get("query")
                if not isinstance(query, str):
                    return _error("invalid_arguments", "query must be a SQL string")
                return _result(
                    run_query(
                        self.connection,
                        query,
                        allowed_tables=self.layer.table_names,
                    )
                )

            return _error("unknown_tool", f"Unknown tool: {name}")
        except QueryError as error:
            return _error(error.kind, error.message)
