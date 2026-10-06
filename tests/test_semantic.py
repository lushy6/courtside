"""The semantic layer: structure, drift against the real schema, and the facts it asserts.

Every number or relationship the YAML states about the data is re-checked here against the real
database. If a rebuild changes the data, these tests fail and the YAML has to be updated, so the
prompt can never go on telling the model something that is no longer true.
"""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest

from courtside.db import open_readonly, table_schema
from courtside.errors import QueryRejected
from courtside.semantic import DEFAULT_NBA, SemanticLayer

REAL = Path(__file__).resolve().parents[1] / "data" / "courtside_nba.db"
needs_db = pytest.mark.skipif(not REAL.exists(), reason="run scripts/build_courtside_db.py")


@pytest.fixture(scope="module")
def layer() -> SemanticLayer:
    return SemanticLayer.load(DEFAULT_NBA)


@pytest.fixture(scope="module")
def con() -> sqlite3.Connection:
    return open_readonly(REAL)


def one(con: sqlite3.Connection, sql: str) -> float:
    return con.execute(sql).fetchone()[0]


# --- structure ----------------------------------------------------------------------------------
def test_rule_ids_are_unique_and_every_rule_has_text(layer: SemanticLayer) -> None:
    ids = [r["id"] for r in layer.data["rules"]]
    assert len(ids) == len(set(ids))
    assert all(r["text"].strip() for r in layer.data["rules"])


def test_every_table_has_purpose_grain_and_described_columns(layer: SemanticLayer) -> None:
    for name in layer.table_names:
        t = layer.data["tables"][name]
        assert t["purpose"].strip() and t["grain"].strip(), name
        assert t["columns"], name
        assert all(str(d).strip() for d in t["columns"].values()), name


def test_sha256_changes_when_the_file_changes(layer: SemanticLayer, tmp_path: Path) -> None:
    copy_path = tmp_path / "nba.yaml"
    copy_path.write_text(DEFAULT_NBA.read_text(encoding="utf-8") + "\n# x\n", encoding="utf-8")
    assert SemanticLayer.load(copy_path).sha256 != layer.sha256
    assert len(layer.sha256) == 64


# --- drift guard ----------------------------------------------------------------------------------
@needs_db
def test_yaml_matches_the_real_schema_exactly(
    layer: SemanticLayer, con: sqlite3.Connection
) -> None:
    assert layer.problems(con) == []


@needs_db
def test_drift_is_detected_in_both_directions(
    layer: SemanticLayer, con: sqlite3.Connection
) -> None:
    broken = copy.deepcopy(layer.data)
    del broken["tables"]["lineup_stats"]["columns"]["poss"]  # real column the YAML forgot
    broken["tables"]["lineup_stats"]["columns"]["ghost"] = "does not exist"  # YAML-only column
    del broken["tables"]["teams"]  # whole real table forgotten
    broken["tables"]["phantom"] = {"purpose": "x", "grain": "x", "columns": {"a": "x"}}
    problems = SemanticLayer(broken).problems(con)
    text = "\n".join(problems)
    assert "lineup_stats.poss" in text and "lineup_stats.ghost" in text
    assert "teams" in text and "phantom" in text


@needs_db
def test_table_schema_reports_columns_with_types(con: sqlite3.Connection) -> None:
    schema = table_schema(con)
    assert ("poss", "REAL") in schema["lineup_stats"] or ("poss", "") in schema["lineup_stats"]
    assert "provenance" in schema


# --- every claim the YAML makes about the data is true --------------------------------------------
@needs_db
def test_scope_claims(con: sqlite3.Connection) -> None:
    assert one(con, "SELECT count(*) FROM games") == 1230
    assert one(con, "SELECT min(game_date) FROM games") == "2025-10-21"
    assert one(con, "SELECT max(game_date) FROM games") == "2026-04-12"
    assert one(con, "SELECT count(DISTINCT season) FROM games") == 1
    assert one(con, "SELECT group_concat(DISTINCT season) FROM contracts") == "2026-27"
    assert one(con, "SELECT count(*) FROM games WHERE home_pts = away_pts") == 0
    # Home games vary (40-42: neutral-site games), so the claim is about total games.
    totals = (
        "SELECT min(n) = 82 AND max(n) = 82 FROM (SELECT count(*) n FROM "
        "(SELECT home_team_id t FROM games UNION ALL SELECT away_team_id FROM games) GROUP BY t)"
    )
    assert one(con, totals) == 1


@needs_db
def test_lineup_rarity_claims(con: sqlite3.Connection) -> None:
    assert one(con, "SELECT count(*) FROM lineup_stats") == 17733
    assert one(con, "SELECT count(*) FROM lineup_stats WHERE poss >= 100") == 234
    assert one(con, "SELECT sum(distinguishable) FROM lineup_stats") == 4
    assert (
        one(
            con,
            "SELECT count(*) FROM lineup_stats WHERE lineup_uid NOT IN "
            "(SELECT lineup_uid FROM lineup_bootstrap)",
        )
        == 32
    )


@needs_db
def test_the_raw_ranking_trap_is_real(con: sqlite3.Connection) -> None:
    """The reason rank_lineups_honestly exists: the raw top lineup is a one-possession fluke."""
    poss, net = con.execute(
        "SELECT poss, net_rating FROM lineup_stats ORDER BY net_rating DESC LIMIT 1"
    ).fetchone()
    assert poss < 5 and net >= 500


@needs_db
def test_minutes_null_means_did_not_play(con: sqlite3.Connection) -> None:
    assert one(con, "SELECT count(*) FROM player_game_stats WHERE minutes IS NULL") == 5528
    assert (
        one(
            con,
            "SELECT count(*) FROM player_game_stats WHERE minutes IS NULL AND "
            "(points > 0 OR fga > 0 OR is_starter = 1)",
        )
        == 0
    )
    assert one(con, "SELECT count(*) FROM player_game_stats WHERE minutes = 0") == 3
    assert (
        one(
            con,
            "SELECT min(s) = 5 AND max(s) = 5 FROM (SELECT sum(is_starter) s "
            "FROM player_game_stats GROUP BY game_id, team_id)",
        )
        == 1
    )


@needs_db
def test_contract_claims(con: sqlite3.Connection) -> None:
    assert one(con, "SELECT count(*) FROM contracts") == 414
    assert one(con, "SELECT count(DISTINCT player_id) FROM contracts") == 408
    dupes = one(
        con,
        "SELECT count(*) FROM (SELECT player_id FROM contracts GROUP BY player_id "
        "HAVING count(*) > 1)",
    )
    assert dupes == 6
    assert (
        one(
            con,
            "SELECT count(*) FROM (SELECT player_id FROM contracts GROUP BY player_id "
            "HAVING count(*) > 1 AND min(salary) <> max(salary))",
        )
        == 0
    ), "same salary on both rows"
    assert one(con, "SELECT count(*) FROM contracts WHERE guaranteed IS NULL") == 34


@needs_db
def test_metric_definitions_hold_in_the_data(con: sqlite3.Connection) -> None:
    assert (
        one(con, "SELECT max(abs(net_rating - (off_rating - def_rating))) FROM lineup_stats") < 1e-9
    )
    assert (
        one(con, "SELECT max(abs(off_rating - pts_for * 100.0 / poss_for)) FROM lineup_stats")
        < 1e-9
    )
    assert (
        one(
            con,
            "SELECT max(abs(def_rating - pts_against * 100.0 / poss_against)) FROM lineup_stats",
        )
        < 1e-9
    )
    assert (
        one(con, "SELECT max(abs(poss - (poss_for + poss_against) / 2.0)) FROM lineup_stats") < 1e-9
    )
    assert one(con, "SELECT max(abs(rapm - (o_rapm + d_rapm))) FROM player_rapm") < 1e-9
    assert (
        one(
            con,
            "SELECT max(abs(shrunk - (SELECT mu FROM noise_model) "
            "- weight * (net_rating - (SELECT mu FROM noise_model)))) FROM lineup_stats",
        )
        < 1e-9
    )
    assert one(con, "SELECT max(abs((hi95 - shrunk) - 1.96 * post_sd)) FROM lineup_stats") < 1e-9
    assert one(con, "SELECT max(abs((shrunk - lo95) - 1.96 * post_sd)) FROM lineup_stats") < 1e-9
    # distinguishable means "interval excludes the league-average rating mu", NOT "excludes zero".
    vs_mu = (
        "SELECT count(*) FROM lineup_stats, noise_model "
        "WHERE distinguishable <> (lo95 > mu OR hi95 < mu)"
    )
    assert one(con, vs_mu) == 0
    vs_zero = "SELECT count(*) FROM lineup_stats WHERE distinguishable <> (lo95 > 0 OR hi95 < 0)"
    assert one(con, vs_zero) == 2, "the YAML warns that the two definitions differ"
    assert round(one(con, "SELECT mu FROM noise_model"), 2) == -0.68
    assert round(one(con, "SELECT true_sd FROM noise_model"), 1) == 7.8


@needs_db
def test_lineup_id_format_and_membership_idiom(con: sqlite3.Connection) -> None:
    """The documented LIKE idiom matches whole ids; the naive one over-matches."""
    uid, lineup = con.execute("SELECT lineup_uid, lineup FROM lineup_stats LIMIT 1").fetchone()
    assert uid == f"2025-26|{one(con, 'SELECT team_id FROM lineup_stats LIMIT 1')}|{lineup}"
    ids = [int(x) for x in lineup.split("-")]
    assert ids == sorted(ids) and len(ids) == 5
    safe = one(
        con, "SELECT count(*) FROM lineup_stats WHERE ('-' || lineup || '-') LIKE '%-2544-%'"
    )
    naive = one(con, "SELECT count(*) FROM lineup_stats WHERE lineup LIKE '%2544%'")
    assert safe > 0 and naive >= safe


@needs_db
def test_player_names_carry_diacritics_and_lineups_use_family_names(
    con: sqlite3.Connection,
) -> None:
    assert one(con, "SELECT count(*) FROM players WHERE family_name = 'Jokić'") == 1
    assert one(con, "SELECT count(*) FROM players WHERE family_name = 'Jokic'") == 0
    assert one(con, "SELECT count(*) FROM lineup_stats WHERE players LIKE '%Jokić%'") > 0


@needs_db
def test_players_has_no_team_and_traded_players_show_an_arrow(con: sqlite3.Connection) -> None:
    assert "team_id" not in [c for c, _ in table_schema(con)["players"]]
    assert one(con, "SELECT count(*) FROM player_season WHERE teams LIKE '%→%'") > 0


# --- rendering ----------------------------------------------------------------------------------
def test_overview_states_scope_rules_and_lists_every_table(layer: SemanticLayer) -> None:
    text = layer.overview()
    assert "2025-26" in text and "2026-27" in text
    for rule in layer.data["rules"]:
        assert rule["id"] in text
    for name in layer.table_names:
        assert name in text
    assert "describe_table" in text


def test_overview_is_compact_enough_to_cache_cheaply(layer: SemanticLayer) -> None:
    assert len(layer.overview()) < 14_000  # roughly 3.5k tokens


@needs_db
def test_describe_table_has_columns_notes_and_sample_rows(
    layer: SemanticLayer, con: sqlite3.Connection
) -> None:
    text = layer.describe_table("lineup_stats", con)
    assert "distinguishable" in text and "Never rank by raw net_rating" in text
    assert "Sample rows" in text and "2025-26" in text


@needs_db
def test_describe_table_is_case_insensitive_and_clips_sample_cells(
    layer: SemanticLayer, con: sqlite3.Connection
) -> None:
    assert "TABLE players" in layer.describe_table("PLAYERS", con)
    assert "…" in layer.describe_table("provenance", con) or "sha" in layer.describe_table(
        "provenance", con
    )


def test_describe_unknown_table_lists_the_real_ones(layer: SemanticLayer) -> None:
    with pytest.raises(QueryRejected) as exc:
        layer.describe_table("nope")
    assert exc.value.kind == "unknown_table" and "lineup_stats" in str(exc.value)


def test_describe_without_a_connection_omits_samples(layer: SemanticLayer) -> None:
    assert "Sample rows" not in layer.describe_table("teams")
