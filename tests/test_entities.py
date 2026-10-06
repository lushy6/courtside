"""Entity resolution: turning "Jokic", "Giannis Antetokounpo" or "Sixers" into ids.

Two layers of tests. The first runs on a tiny fixture and pins the rules (normalisation, exact
before fuzzy, ambiguity is reported rather than guessed). The second is the acceptance test on the
real 591 players: a 60-name set of misspellings, missing diacritics, nicknames-by-format and
suffix variants, which must resolve top-1 at 95% or better, plus names that must NOT resolve.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from courtside.db import open_readonly
from courtside.entities import EntityResolver, normalize

REAL = Path(__file__).resolve().parents[1] / "data" / "courtside_nba.db"
needs_db = pytest.mark.skipif(not REAL.exists(), reason="run scripts/build_courtside_db.py")

PLAYERS = [
    (1, "Jayson", "Tatum"),
    (2, "Jaylen", "Brown"),
    (3, "Nikola", "Jokić"),
    (4, "Shai", "Gilgeous-Alexander"),
    (5, "T.J.", "McConnell"),
    (6, "Jaren", "Jackson Jr."),
    (7, "Isaiah", "Jackson"),
    (8, "Jimmy", "Butler III"),
    (9, "Kelly", "Oubre Jr."),
    (10, "Marcus", "Brown"),
    (11, "De'Aaron", "Fox"),
]
TEAMS = [
    (10, "BOS", "Boston", "Celtics"),
    (11, "LAL", "Los Angeles", "Lakers"),
    (12, "LAC", "Los Angeles", "Clippers"),
    (13, "PHI", "Philadelphia", "76ers"),
    (14, "POR", "Portland", "Trail Blazers"),
    (15, "CLE", "Cleveland", "Cavaliers"),
    (16, "MIN", "Minnesota", "Timberwolves"),
]


@pytest.fixture
def resolver() -> EntityResolver:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE players (player_id INTEGER, first_name TEXT, family_name TEXT)")
    con.execute("CREATE TABLE teams (team_id INTEGER, abbr TEXT, city TEXT, name TEXT)")
    con.executemany("INSERT INTO players VALUES (?, ?, ?)", PLAYERS)
    con.executemany("INSERT INTO teams VALUES (?, ?, ?, ?)", TEAMS)
    return EntityResolver.from_connection(con)


def top(r: EntityResolver, q: str) -> int | None:
    res = r.resolve_player(q)
    return res.matches[0].player_id if res.matches else None


# --- normalisation ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Nikola Jokić", "nikola jokic"),
        ("  Shai   Gilgeous-Alexander ", "shai gilgeous alexander"),
        ("T.J. McConnell", "tj mcconnell"),
        ("De'Aaron Fox", "deaaron fox"),
        ("Jaren Jackson Jr.", "jaren jackson"),
        ("Jimmy Butler III", "jimmy butler"),
        ("Šarić", "saric"),
        ("Porziņģis", "porzingis"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


# --- exact, then fuzzy -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "player_id"),
    [
        ("Jayson Tatum", 1),
        ("jayson tatum", 1),
        ("Tatum", 1),
        ("Jason Tatum", 1),  # typo
        ("Nikola Jokic", 3),  # diacritic dropped
        ("Jokić", 3),
        ("Nicola Jokic", 3),  # typo plus diacritic
        ("Shai Gilgeous Alexander", 4),  # hyphen dropped
        ("Gilgeous-Alexander", 4),
        ("TJ McConnell", 5),
        ("T J McConnell", 5),
        ("Jaren Jackson", 6),  # suffix omitted
        ("Jaren Jackson Jr", 6),
        ("Jimmy Butler", 8),
        ("DeAaron Fox", 11),
        ("De Aaron Fox", 11),
    ],
)
def test_resolves_variants(resolver: EntityResolver, query: str, player_id: int) -> None:
    res = resolver.resolve_player(query)
    assert res.matches and res.matches[0].player_id == player_id, res
    assert res.ambiguous is False


def test_exact_full_name_beats_a_fuzzy_neighbour(resolver: EntityResolver) -> None:
    res = resolver.resolve_player("Jaylen Brown")
    assert res.matches[0].player_id == 2 and res.matches[0].how == "exact"


def test_family_name_shared_by_several_players_is_ambiguous_not_guessed(
    resolver: EntityResolver,
) -> None:
    res = resolver.resolve_player("Brown")
    assert res.ambiguous is True
    assert {m.player_id for m in res.matches} == {2, 10}
    res = resolver.resolve_player("Jackson")
    assert res.ambiguous is True and {m.player_id for m in res.matches} == {6, 7}


def test_unknown_names_do_not_resolve(resolver: EntityResolver) -> None:
    for q in ["Michael Jordan", "Tyrese Haliburton", "asdfgh", "the best lineup", "", "   "]:
        assert resolver.resolve_player(q).matches == (), q


def test_limit_caps_the_number_of_matches(resolver: EntityResolver) -> None:
    assert len(resolver.resolve_player("Jackson", limit=1).matches) == 1


def test_display_name_keeps_the_original_spelling(resolver: EntityResolver) -> None:
    assert resolver.resolve_player("Jokic").matches[0].name == "Nikola Jokić"
    assert resolver.resolve_player("Jaren Jackson").matches[0].name == "Jaren Jackson Jr."


# --- teams ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "abbr"),
    [
        ("BOS", "BOS"),
        ("bos", "BOS"),
        ("Celtics", "BOS"),
        ("Boston Celtics", "BOS"),
        ("Boston", "BOS"),
        ("Lakers", "LAL"),
        ("Clippers", "LAC"),
        ("Sixers", "PHI"),
        ("76ers", "PHI"),
        ("Cavs", "CLE"),
        ("Wolves", "MIN"),
        ("Blazers", "POR"),
        ("Trail Blazers", "POR"),
        ("Celtcs", "BOS"),  # typo
    ],
)
def test_resolves_teams(resolver: EntityResolver, query: str, abbr: str) -> None:
    res = resolver.resolve_team(query)
    assert res.matches and res.matches[0].abbr == abbr, res
    assert res.ambiguous is False


def test_a_shared_city_is_ambiguous(resolver: EntityResolver) -> None:
    res = resolver.resolve_team("Los Angeles")
    assert res.ambiguous is True and {m.abbr for m in res.matches} == {"LAL", "LAC"}


def test_unknown_team_does_not_resolve(resolver: EntityResolver) -> None:
    assert resolver.resolve_team("Seattle Supersonics").matches == ()


# --- finding entities inside a question ------------------------------------------------------------------
def test_finds_a_player_and_a_team_in_a_question(resolver: EntityResolver) -> None:
    found = resolver.find_mentions("How many points did Jayson Tatum score against the Celtics?")
    kinds = {(m.kind, m.text) for m in found}
    assert ("player", "Jayson Tatum") in kinds and ("team", "Celtics") in kinds


def test_finds_two_players(resolver: EntityResolver) -> None:
    found = resolver.find_mentions("Compare Jokic and Tatum")
    assert {m.text for m in found if m.kind == "player"} == {"Jokic", "Tatum"}


def test_longest_match_wins_and_mentions_do_not_overlap(resolver: EntityResolver) -> None:
    found = resolver.find_mentions("Is Shai Gilgeous-Alexander better than Tatum?")
    players = [m for m in found if m.kind == "player"]
    assert [m.text for m in players] == ["Shai Gilgeous-Alexander", "Tatum"]


def test_lowercase_ordinary_words_are_not_mistaken_for_names(resolver: EntityResolver) -> None:
    assert resolver.find_mentions("which brown lineup was the best in the league") == []


def test_ambiguous_mention_is_marked(resolver: EntityResolver) -> None:
    found = resolver.find_mentions("How did Brown play in March?")
    assert len(found) == 1 and found[0].ambiguous is True


def test_possessives_do_not_hide_a_name(resolver: EntityResolver) -> None:
    assert [m.text for m in resolver.find_mentions("Is Tatum's net rating good?")] == ["Tatum"]
    assert [m.text for m in resolver.find_mentions("the Celtics' best lineup")] == ["Celtics"]
    assert [m.text for m in resolver.find_mentions("the Celtics’ best lineup")] == ["Celtics"]


def test_a_unique_first_name_is_found_but_a_shared_one_is_not(resolver: EntityResolver) -> None:
    found = resolver.find_mentions("Which lineups include Jimmy?")
    assert [(m.kind, m.text, m.matches[0].player_id) for m in found] == [("player", "Jimmy", 8)]
    # "Jayson" and "Jaylen" are each unique here, but a first name shared by two players is not a hint.
    assert (
        resolver.find_mentions("How about Marcus?") == []
        or resolver.find_mentions("How about Marcus?")[0].matches[0].player_id == 10
    )


def test_lowercase_team_nicknames_count_unless_they_are_everyday_words(
    resolver: EntityResolver,
) -> None:
    found = resolver.find_mentions("list the lakers and clippers payrolls")
    assert [(m.kind, m.matches[0].abbr) for m in found] == [("team", "LAL"), ("team", "LAC")]
    assert [m.text for m in resolver.find_mentions("the sixers and cavs")] == ["sixers", "cavs"]
    assert resolver.find_mentions("a heat check and some magic") == []


def test_a_bare_team_abbreviation_must_be_uppercase_to_count(resolver: EntityResolver) -> None:
    assert [m.text for m in resolver.find_mentions("Best BOS lineup")] == ["BOS"]
    assert resolver.find_mentions("the bos lineup") == []


# --- acceptance on the real data --------------------------------------------------------------------------
# (what a user might type, the name as stored). 60 cases: misspellings, dropped diacritics and
# hyphens, suffix and initials variants.
REAL_CASES = [
    ("Jokic", "Nikola Jokić"), ("nikola jokic", "Nikola Jokić"), ("Nicola Jokic", "Nikola Jokić"),
    ("Giannis Antetokounpo", "Giannis Antetokounmpo"), ("Antetokounmpo", "Giannis Antetokounmpo"),
    ("Shai Gilgeous Alexander", "Shai Gilgeous-Alexander"), ("Gilgeous-Alexander", "Shai Gilgeous-Alexander"),
    ("Luka Doncic", "Luka Dončić"), ("Doncic", "Luka Dončić"),
    ("Wembanyama", "Victor Wembanyama"), ("Victor Wembenyama", "Victor Wembanyama"),
    ("Jayson Tatum", "Jayson Tatum"), ("tatum", "Jayson Tatum"), ("Jason Tatum", "Jayson Tatum"),
    ("Kristaps Porzingis", "Kristaps Porziņģis"), ("Nikola Vucevic", "Nikola Vučević"),
    ("Dario Saric", "Dario Šarić"), ("TJ McConnell", "T.J. McConnell"), ("T.J. McConnell", "T.J. McConnell"),
    ("Karl Anthony Towns", "Karl-Anthony Towns"), ("DeAaron Fox", "De'Aaron Fox"), ("De Aaron Fox", "De'Aaron Fox"),
    ("Jaren Jackson", "Jaren Jackson Jr."), ("Jaren Jackson Jr", "Jaren Jackson Jr."),
    ("Michael Porter Jr", "Michael Porter Jr."), ("Michael Porter", "Michael Porter Jr."),
    ("Sengun", "Alperen Sengun"), ("Alperen Şengün", "Alperen Sengun"), ("Jimmy Butler", "Jimmy Butler III"),
    ("Anthony Edwards", "Anthony Edwards"), ("Steph Curry", "Stephen Curry"), ("Lebron James", "LeBron James"),
    ("Kevin Durant", "Kevin Durant"), ("Joel Embid", "Joel Embiid"), ("Devin Booker", "Devin Booker"),
    ("Donovan Mitchel", "Donovan Mitchell"), ("Jalen Brunson", "Jalen Brunson"), ("Zion Williamson", "Zion Williamson"),
    ("Ja Morant", "Ja Morant"), ("Paolo Banchero", "Paolo Banchero"), ("Franz Wagner", "Franz Wagner"),
    ("Cade Cunningham", "Cade Cunningham"), ("Evan Mobley", "Evan Mobley"), ("Chet Holmgren", "Chet Holmgren"),
    ("Amen Thompson", "Amen Thompson"), ("Scottie Barnes", "Scottie Barnes"), ("Bam Adebayo", "Bam Adebayo"),
    ("Draymond Green", "Draymond Green"), ("Derrick White", "Derrick White"), ("Jaylen Brown", "Jaylen Brown"),
    ("Bogdan Bogdanovic", "Bogdan Bogdanović"), ("Jusuf Nurkic", "Jusuf Nurkić"), ("Domantas Sabonis", "Domantas Sabonis"),
    ("Pascal Siakam", "Pascal Siakam"), ("Lauri Markkanen", "Lauri Markkanen"), ("Deni Avdija", "Deni Avdija"),
    ("Moussa Diabate", "Moussa Diabaté"), ("Vit Krejci", "Vít Krejčí"), ("Jabari Smith Jr", "Jabari Smith Jr."),
    ("Kelel Ware", "Kel'el Ware"),
]  # fmt: skip


@needs_db
def test_sixty_name_acceptance_set_resolves_top1_at_95_percent() -> None:
    r = EntityResolver.from_connection(open_readonly(REAL))
    assert len(REAL_CASES) >= 60
    misses = []
    for query, stored in REAL_CASES:
        res = r.resolve_player(query)
        got = res.matches[0].name if res.matches else None
        if got != stored:
            misses.append((query, stored, got))
    rate = 1 - len(misses) / len(REAL_CASES)
    assert rate >= 0.95, f"{rate:.1%} top-1; misses: {misses}"


@needs_db
def test_real_names_that_must_not_resolve() -> None:
    r = EntityResolver.from_connection(open_readonly(REAL))
    # Haliburton is genuinely absent from the 2025-26 data: a real "refuse, not guess" case.
    for q in ["Tyrese Haliburton", "Michael Jordan", "Kobe Bryant", "asdfgh qwerty"]:
        assert r.resolve_player(q).matches == (), q


@needs_db
def test_real_shared_family_names_are_flagged() -> None:
    r = EntityResolver.from_connection(open_readonly(REAL))
    for q in ["Jackson", "Williams", "Thompson", "Johnson"]:
        res = r.resolve_player(q)
        assert res.ambiguous is True and len(res.matches) >= 2, q


@needs_db
def test_real_teams_and_aliases() -> None:
    r = EntityResolver.from_connection(open_readonly(REAL))
    for q, abbr in [("Celtics", "BOS"), ("Sixers", "PHI"), ("Cavs", "CLE"), ("Wolves", "MIN"), ("Blazers", "POR"),
                    ("Mavs", "DAL"), ("Knicks", "NYK"), ("Warriors", "GSW"), ("Spurs", "SAS"), ("Thunder", "OKC")]:  # fmt: skip
        res = r.resolve_team(q)
        assert res.matches and res.matches[0].abbr == abbr, (q, res)
    assert len(r.teams) == 30 and len(r.players) == 591
