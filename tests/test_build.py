"""The merged-database build: consistency checks, fidelity, determinism, source safety."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from courtside.build import LINEUP, TACTICAL, BuildError, BuildSpec, _identical, build, sha256_file
from courtside.db import open_readonly, run_query

SPEC = BuildSpec(
    lineup_tables=("players", "lineup_stats"),
    tactical_tables=("contracts",),
    checks=(
        _identical("players"),
        (
            "every contract belongs to a known player",
            f"SELECT count(*) FROM {TACTICAL}.contracts "
            f"WHERE player_id NOT IN (SELECT player_id FROM {LINEUP}.players)",
        ),
    ),
)

PLAYERS_DDL = "CREATE TABLE players (player_id INTEGER PRIMARY KEY, name TEXT NOT NULL)"


def _make(path: Path, statements: list[str]) -> Path:
    con = sqlite3.connect(path)
    for s in statements:
        con.execute(s)
    con.commit()
    con.close()
    return path


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    lineup = _make(
        tmp_path / "lineup.db",
        [
            PLAYERS_DDL,
            "INSERT INTO players VALUES (1, 'Tatum'), (2, 'Brown')",
            "CREATE TABLE lineup_stats (uid TEXT, poss INTEGER, net REAL)",
            "CREATE INDEX ix_lineup_uid ON lineup_stats (uid)",
            "INSERT INTO lineup_stats VALUES ('a', 100, 1.5), ('b', 40, -3.0)",
        ],
    )
    tactical = _make(
        tmp_path / "tactical.db",
        [
            PLAYERS_DDL,
            "INSERT INTO players VALUES (1, 'Tatum'), (2, 'Brown')",
            "CREATE TABLE contracts (player_id INTEGER, salary INTEGER)",
            "INSERT INTO contracts VALUES (1, 58000000), (2, 53000000)",
        ],
    )
    return lineup, tactical


def test_builds_every_table_with_matching_row_counts(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "out" / "merged.db"
    manifest = build(out, *sources, spec=SPEC)
    con = sqlite3.connect(out)
    assert con.execute("SELECT count(*) FROM players").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM lineup_stats").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM contracts").fetchone()[0] == 2
    assert manifest["tables"]["contracts"] == {"rows": 2, "source": TACTICAL}
    assert manifest["tables"]["players"]["source"] == LINEUP
    assert manifest["out_sha256"] == sha256_file(out)
    assert len(manifest["checks_passed"]) == 2


def test_provenance_table_records_origin_and_hash(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "merged.db"
    manifest = build(out, *sources, spec=SPEC)
    con = sqlite3.connect(out)
    row = con.execute(
        "SELECT source, source_file, source_sha256, rows FROM provenance "
        "WHERE table_name = 'contracts'"
    ).fetchone()
    assert row == ("tactical", "tactical.db", manifest["sources"][TACTICAL]["sha256"], 2)


def test_original_ddl_and_indexes_survive(sources: tuple[Path, Path], tmp_path: Path) -> None:
    out = tmp_path / "merged.db"
    build(out, *sources, spec=SPEC)
    con = sqlite3.connect(out)
    pk = [r[1] for r in con.execute("PRAGMA table_info(players)") if r[5]]
    assert pk == ["player_id"]
    indexes = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")]
    assert "ix_lineup_uid" in indexes


def test_build_is_deterministic(sources: tuple[Path, Path], tmp_path: Path) -> None:
    a = build(tmp_path / "a.db", *sources, spec=SPEC)["out_sha256"]
    b = build(tmp_path / "b.db", *sources, spec=SPEC)["out_sha256"]
    assert a == b


def test_sources_are_never_modified(sources: tuple[Path, Path], tmp_path: Path) -> None:
    before = [sha256_file(p) for p in sources]
    names_before = sorted(p.name for p in tmp_path.iterdir())
    build(tmp_path / "merged.db", *sources, spec=SPEC)
    assert [sha256_file(p) for p in sources] == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*names_before, "merged.db"])


def test_disagreeing_sources_fail_loudly_and_leave_no_output(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    con = sqlite3.connect(sources[1])
    con.execute("UPDATE players SET name = 'Tatum Jr.' WHERE player_id = 1")
    con.commit()
    con.close()
    out = tmp_path / "merged.db"
    with pytest.raises(BuildError, match="players identical"):
        build(out, *sources, spec=SPEC)
    assert not out.exists() and not out.with_name("merged.db.tmp").exists()


def test_orphan_contract_fails(sources: tuple[Path, Path], tmp_path: Path) -> None:
    con = sqlite3.connect(sources[1])
    con.execute("INSERT INTO contracts VALUES (99, 1)")
    con.commit()
    con.close()
    with pytest.raises(BuildError, match="known player"):
        build(tmp_path / "merged.db", *sources, spec=SPEC)


def test_missing_table_is_named(sources: tuple[Path, Path], tmp_path: Path) -> None:
    spec = BuildSpec(lineup_tables=("players", "nope"), tactical_tables=())
    with pytest.raises(BuildError, match="nope"):
        build(tmp_path / "merged.db", *sources, spec=spec)


def test_missing_source_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="lineup source"):
        build(tmp_path / "o.db", tmp_path / "gone.db", tmp_path / "gone2.db", spec=SPEC)


def test_a_failed_rebuild_keeps_the_previous_good_output(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "merged.db"
    build(out, *sources, spec=SPEC)
    good = sha256_file(out)
    con = sqlite3.connect(sources[1])
    con.execute("INSERT INTO contracts VALUES (99, 1)")
    con.commit()
    con.close()
    with pytest.raises(BuildError):
        build(out, *sources, spec=SPEC)
    assert sha256_file(out) == good


def test_merged_db_works_with_the_safety_pipeline(
    sources: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "merged.db"
    build(out, *sources, spec=SPEC)
    con = open_readonly(out)
    res = run_query(
        con,
        "SELECT p.name, c.salary FROM players p JOIN contracts c USING (player_id) "
        "ORDER BY c.salary DESC",
        allowed_tables={"players", "contracts", "lineup_stats", "provenance"},
    )
    assert res.rows[0] == ("Tatum", 58000000)
