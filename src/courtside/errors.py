"""Errors the SQL safety stack raises.

Every one carries a short machine-readable ``kind`` and a message written to be read by
the model, because the agent loop feeds it straight back as a tool result. A rejection
that only said "invalid" would give the model nothing to correct.
"""

from __future__ import annotations


class QueryError(Exception):
    """Base class: the query did not run to a result."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


class QueryRejected(QueryError):
    """The validator refused the query before it reached the database."""


class QueryFailed(QueryError):
    """The query reached the database and did not produce a result (error or time limit)."""
