"""Layer 1: the parser allowlist."""

from __future__ import annotations

import pytest
from attacks import ABUSE_ATTACKS, WRITE_ATTACKS
from conftest import TABLES

from courtside.errors import QueryRejected
from courtside.validate import validate_select

# Valid SELECTs that are expensive rather than dangerous. Layer 1 cannot judge cost,
# so it admits them and the executor's time limit is what stops them (test_db.py).
RESOURCE_ONLY = {
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r",
    "SELECT count(*) FROM lineup_stats a, lineup_stats b, lineup_stats c, lineup_stats d, lineup_stats e",
}

GOOD = [
    "SELECT * FROM players",
    "select first_name, family_name from players where player_id = 1",
    "SELECT team, count(*) AS n FROM lineup_stats GROUP BY team HAVING n > 1 ORDER BY n DESC LIMIT 5",
    "SELECT p.first_name FROM players p JOIN teams t ON t.team_id = p.player_id",
    "WITH big AS (SELECT * FROM lineup_stats WHERE poss > 100) SELECT lineup_uid FROM big",
    "SELECT first_name FROM players UNION SELECT city FROM teams",
    "SELECT lineup_uid, rank() OVER (ORDER BY shrunk DESC) AS r FROM lineup_stats",
    "SELECT CASE WHEN poss >= 100 THEN 'ok' ELSE 'thin' END FROM lineup_stats",
    "SELECT round(avg(net_rating), 2), min(poss), max(poss), sum(poss) FROM lineup_stats",
    "SELECT * FROM (SELECT * FROM players) WHERE player_id IN (SELECT player_id FROM players)",
    "SELECT 1",
    "SELECT * FROM players;",
    "SELECT first_name || ' ' || family_name AS name FROM players",
    "SELECT strftime('%Y', '2025-10-01')",
    "SELECT * FROM PLAYERS",
    "SELECT * FROM players WHERE first_name = 'a--b' AND family_name <> 'x/*y*/'",
    # every function family the system prompt tells the model it may use
    "SELECT coalesce(NULL, 1), ifnull(NULL, 2), nullif(1, 1), iif(1 > 0, 'y', 'n')",
    "SELECT abs(-1), round(2.567, 1), length('abc'), lower('A'), upper('a'), trim(' a ')",
    "SELECT substr('abcdef', 2, 3), replace('aXb', 'X', '-'), instr('abc', 'b'), typeof(1)",
    "SELECT CAST(poss AS INTEGER), CAST(net_rating AS TEXT) FROM lineup_stats",
    "SELECT group_concat(first_name, ',') FROM players",
    "SELECT date('2025-10-21'), datetime('2025-10-21 10:00:00'), julianday('2025-10-21')",
    "SELECT lag(poss) OVER (ORDER BY lineup_uid), lead(poss) OVER (ORDER BY lineup_uid) FROM lineup_stats",
    "SELECT row_number() OVER (PARTITION BY team ORDER BY poss DESC) FROM lineup_stats",
    "SELECT ntile(4) OVER (ORDER BY poss), percent_rank() OVER (ORDER BY poss) FROM lineup_stats",
    "SELECT sqrt(poss), power(poss, 2), ln(poss), floor(poss), ceil(poss) FROM lineup_stats",
    "SELECT team FROM lineup_stats WHERE team LIKE 'B%' OR team GLOB 'D*'",
]

# sqlglot turns these into typed nodes, so a check that only looks at unrecognised function
# names never sees them. Layer 1 must still refuse every one (layer 2 is a second wall, not
# the first).
TYPED_FUNCTIONS_NOT_ALLOWED = [
    "SELECT hex(first_name) FROM players",
    "SELECT unhex('41')",
    "SELECT random()",
    "SELECT first_name FROM players ORDER BY random()",
    "SELECT json_extract('{}', '$')",
]


@pytest.mark.parametrize("sql", GOOD)
def test_accepts_plain_selects(sql: str) -> None:
    assert validate_select(sql, allowed_tables=TABLES).strip()


@pytest.mark.parametrize("sql", WRITE_ATTACKS)
def test_rejects_every_write_attack(sql: str) -> None:
    with pytest.raises(QueryRejected):
        validate_select(sql, allowed_tables=TABLES)


@pytest.mark.parametrize("sql", [s for s in ABUSE_ATTACKS if s not in RESOURCE_ONLY])
def test_rejects_every_abuse_attack(sql: str) -> None:
    with pytest.raises(QueryRejected):
        validate_select(sql, allowed_tables=TABLES)


@pytest.mark.parametrize("sql", sorted(RESOURCE_ONLY))
def test_admits_resource_only_queries_by_design(sql: str) -> None:
    assert validate_select(sql, allowed_tables=TABLES)


def test_unknown_table_error_lists_what_exists() -> None:
    with pytest.raises(QueryRejected) as exc:
        validate_select("SELECT * FROM nope", allowed_tables=TABLES)
    msg = str(exc.value)
    assert "nope" in msg and "lineup_stats" in msg and "players" in msg


def test_cte_names_are_not_mistaken_for_unknown_tables() -> None:
    sql = "WITH big AS (SELECT * FROM lineup_stats) SELECT * FROM big"
    assert validate_select(sql, allowed_tables=TABLES)


def test_function_name_case_cannot_bypass_the_allowlist() -> None:
    with pytest.raises(QueryRejected):
        validate_select("SELECT LoAd_ExTeNsIoN('x')", allowed_tables=TABLES)


def test_rejection_carries_a_machine_readable_kind() -> None:
    with pytest.raises(QueryRejected) as exc:
        validate_select("DROP TABLE players", allowed_tables=TABLES)
    assert exc.value.kind == "not_select"
    with pytest.raises(QueryRejected) as exc:
        validate_select("SELECT 1; SELECT 2", allowed_tables=TABLES)
    assert exc.value.kind == "multiple_statements"
    with pytest.raises(QueryRejected) as exc:
        validate_select("SELECT * FROM nope", allowed_tables=TABLES)
    assert exc.value.kind == "unknown_table"
    with pytest.raises(QueryRejected) as exc:
        validate_select("SELECT load_extension('x')", allowed_tables=TABLES)
    assert exc.value.kind == "forbidden_function"


def test_no_allowlist_means_table_check_is_skipped() -> None:
    assert validate_select("SELECT * FROM anything")


@pytest.mark.parametrize("sql", TYPED_FUNCTIONS_NOT_ALLOWED)
def test_typed_sqlglot_functions_are_checked_against_the_allowlist(sql: str) -> None:
    with pytest.raises(QueryRejected) as exc:
        validate_select(sql, allowed_tables=TABLES)
    assert exc.value.kind == "forbidden_function"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 -- note",
        "SELECT 1 /* note */",
        "SELECT /* a */ 1",
        "SELECT 1 /* /* */ ; DROP TABLE players; /* */",
    ],
)
def test_comments_are_refused_so_two_parsers_cannot_disagree_about_them(sql: str) -> None:
    with pytest.raises(QueryRejected) as exc:
        validate_select(sql, allowed_tables=TABLES)
    assert exc.value.kind in {"comments", "multiple_statements", "parse_error"}


def test_comment_markers_inside_string_literals_are_data_not_comments() -> None:
    assert validate_select("SELECT 'a--b', 'c/*d*/e'")


def test_pathologically_deep_nesting_is_a_rejection_not_a_crash() -> None:
    sql = "SELECT " + "(" * 3000 + "1" + ")" * 3000
    with pytest.raises(QueryRejected) as exc:
        validate_select(sql, allowed_tables=TABLES)
    assert exc.value.kind in {"too_complex", "parse_error"}


def test_very_long_input_is_rejected_cheaply() -> None:
    with pytest.raises(QueryRejected) as exc:
        validate_select("SELECT 1 WHERE " + " OR ".join(["1=1"] * 20000))
    assert exc.value.kind == "too_complex"
