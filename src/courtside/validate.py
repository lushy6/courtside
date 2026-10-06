"""Layer 1 of the SQL safety stack: a parser allowlist.

Generated SQL is untrusted input. This layer parses it and admits exactly one read-only
SELECT (optionally with CTEs or set operations) that touches only known tables and
calls only known functions. It is deliberately the strictest and the most informative
layer: its messages teach the model what it may write.

It is NOT the only layer. ``courtside.db`` adds a SQLite authorizer and a read-only file
handle, and each of the three is meant to stop a write on its own. This module therefore
does not try to be clever about cost: a valid but expensive SELECT passes here and is
stopped by the executor's time limit.
"""

from __future__ import annotations

from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from courtside.errors import QueryRejected

# Scalar, aggregate and window functions a stats question can reasonably need. Anything
# else (load_extension, readfile, writefile, randomblob, zeroblob, fts3_tokenizer, ...) is
# refused. The SQLite authorizer in courtside.db enforces the same list a second time.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        # aggregates
        "count", "sum", "avg", "min", "max", "total", "group_concat", "median",
        "percentile", "percentile_cont", "percentile_disc",
        # scalar
        "abs", "round", "coalesce", "ifnull", "nullif", "iif", "sign", "length", "lower",
        "upper", "substr", "substring", "trim", "ltrim", "rtrim", "replace", "instr",
        "typeof", "printf", "format", "like", "glob", "likelihood", "unlikely", "likely",
        "floor", "ceil", "ceiling", "sqrt", "power", "pow", "ln", "log", "log10", "log2",
        "exp", "mod", "char", "unicode",
        # dates
        "date", "time", "datetime", "julianday", "unixepoch", "strftime",
        # window
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag",
        "lead", "first_value", "last_value", "nth_value",
    }
)  # fmt: skip

_FORBIDDEN_NODES: tuple[type[exp.Expr], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

_SET_OPERATIONS: tuple[type[exp.Expr], ...] = (exp.Union, exp.Intersect, exp.Except)


def validate_select(sql: str, allowed_tables: Iterable[str] | None = None) -> str:
    """Return ``sql`` unchanged if it is a single safe SELECT, else raise ``QueryRejected``.

    ``allowed_tables`` is the catalog. When given, any other table is rejected with a
    message listing the tables that do exist, which catches hallucinated names before the
    database does. CTE names are not tables and are exempt.

    The original text is returned, not a regenerated string, so what runs is what the
    author wrote; the two later layers exist precisely so a parser disagreement here
    cannot become a write.
    """
    text = sql.strip()
    if not text:
        raise QueryRejected("empty", "The query is empty. Write one SELECT statement.")

    try:
        parsed = sqlglot.parse(text, read="sqlite")
    except SqlglotError as err:
        raise QueryRejected("parse_error", f"Could not parse the SQL: {err}") from err

    statements = [s for s in parsed if s is not None]
    if not statements:
        raise QueryRejected("empty", "The query is empty. Write one SELECT statement.")
    if len(statements) > 1:
        raise QueryRejected(
            "multiple_statements", "Only one statement is allowed per query. Send one SELECT."
        )

    root = statements[0]
    if not isinstance(root, (exp.Select, *_SET_OPERATIONS)):
        raise QueryRejected(
            "not_select",
            "Only read-only SELECT queries are allowed (WITH ... SELECT is fine). "
            "This database cannot be modified.",
        )
    for select in root.find_all(exp.Select):
        if not select.expressions:
            # sqlglot parses a bare "SELECT" leniently; SQLite would reject it later.
            raise QueryRejected(
                "parse_error", "The SELECT lists no columns. Write SELECT <columns>."
            )
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise QueryRejected(
                "not_select", f"{type(node).__name__} is not allowed. Only SELECT is permitted."
            )

    _check_functions(root)
    _check_tables(root, allowed_tables)
    return text


def _check_functions(root: exp.Expr) -> None:
    for node in root.find_all(exp.Anonymous):
        name = str(node.name).lower()
        if name not in ALLOWED_FUNCTIONS:
            raise QueryRejected(
                "forbidden_function",
                f"Function '{name}' is not allowed. Allowed functions include "
                "count, sum, avg, min, max, round, coalesce, substr, lower, upper, "
                "strftime and the standard window functions.",
            )


def _check_tables(root: exp.Expr, allowed_tables: Iterable[str] | None) -> None:
    catalog = None if allowed_tables is None else {t.lower() for t in allowed_tables}
    cte_names = {str(cte.alias).lower() for cte in root.find_all(exp.CTE)}

    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            # FROM some_function(...): a table-valued function such as pragma_table_info.
            raise QueryRejected(
                "forbidden_function",
                "Table-valued functions are not allowed. Query the tables directly.",
            )
        name = table.name.lower()
        if table.db:
            raise QueryRejected(
                "forbidden_table",
                f"Schema-qualified table '{table.db}.{table.name}' is not allowed. "
                f"Use the bare table name.",
            )
        if name in cte_names:
            continue
        if name.startswith(("sqlite_", "pragma_")):
            raise QueryRejected(
                "forbidden_table",
                f"'{table.name}' is an internal table. Use describe_table to see the schema.",
            )
        if catalog is not None and name not in catalog:
            raise QueryRejected(
                "unknown_table",
                f"Unknown table '{table.name}'. Available tables: {', '.join(sorted(catalog))}.",
            )
