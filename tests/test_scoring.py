"""Scorer, faithfulness check and statistics: offline, against the frozen dev split only."""

from __future__ import annotations

from typing import Any

import pytest

from courtside import faithfulness, stats
from courtside.golden import read_frozen
from courtside.scoring import (
    Score,
    canonical_rows,
    mentions,
    rows_match,
    same_value,
    score_one,
    score_run,
)

DEV = {q["id"]: q for q in read_frozen("dev")}


def answer(rows: list[list[Any]], text: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "outcome": "answered",
        "answer_text": text,
        "sql": "SELECT 1",
        "refused": False,
        "result_rows": rows,
        **extra,
    }


def refusal(reason: str = "Not in this database.") -> dict[str, Any]:
    return {"outcome": "answered", "refused": True, "refusal_reason": reason, "sql": ""}


# --- values and rows ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "gold", "same"),
    [
        (71, 71, True),
        (71.0, 71, True),
        (71.4, 71, False),  # integers are exact
        (122.0732, 122.07, True),  # gold rounded to 2 places
        (50.4, 50.39798097456803, True),  # the model rounded to 1 place
        (1.0, 0.688, False),
        (0.7, 0.688, False),  # rounding error above 1% of the gold value
        (None, None, True),
        (None, 0, False),
        ("71", 71, False),
        (" BOS", "BOS", True),
        (True, 1, False),
    ],
)
def test_same_value(model: Any, gold: Any, same: bool) -> None:
    assert same_value(model, gold) is same


def test_rows_match_allows_extra_and_reordered_columns_and_rows() -> None:
    gold = [["MIN", 235], ["ORL", 223]]
    assert rows_match([[223, "ORL", 9], [235, "MIN", 9]], gold, ordered=False)
    assert not rows_match([[223, "ORL", 9], [235, "MIN", 9]], gold, ordered=True)
    assert rows_match([[235, "MIN"], [223, "ORL"]], gold, ordered=True)
    assert not rows_match([[235, "MIN"]], gold, ordered=False)  # missing row
    assert not rows_match([[235, "MIN"], [223, "ORL"], [1, "X"]], gold, ordered=False)
    assert not rows_match([["MIN"], ["ORL"]], gold, ordered=False)  # missing column
    assert rows_match([], [], ordered=False)


# --- every dev question answered with its own gold answer is correct -----------------------------
def gold_answer(q: dict[str, Any], which: str = "accepted") -> dict[str, Any]:
    if q["kind"] in {"refuse", "adversarial"}:
        return refusal()
    source = q["expected"] if q["kind"] == "rows" else q[which][0]
    mention = q["must_mention"][0] if q["must_mention"] else ""
    return answer(source["rows"], caveats=[f"Method: {mention}"])


@pytest.mark.parametrize("qid", sorted(DEV))
def test_gold_answers_score_correct(qid: str) -> None:
    score = score_one(DEV[qid], gold_answer(DEV[qid]))
    assert score.correct, score


@pytest.mark.parametrize("qid", sorted(i for i, q in DEV.items() if q.get("forbidden")))
def test_naive_trap_answers_fail(qid: str) -> None:
    score = score_one(DEV[qid], gold_answer(DEV[qid], "forbidden"))
    assert not score.correct
    assert score.forbidden_match and score.label == "ignored_sample_size"


def test_trap_needs_the_method_named() -> None:
    q = DEV["T01"]
    silent = score_one(q, answer(q["accepted"][0]["rows"], "Bridges, Ball and co."))
    assert not silent.correct and silent.label == "no_caveat"


def test_trap_accepts_a_ranked_list_whose_top_rows_are_right() -> None:
    q = DEV["T01"]
    top = q["accepted"][0]["rows"][0]
    rows = [top, ["other", "Someone", 500.0, 1.0, 1.0]]
    assert score_one(q, answer(rows, "Using shrunk ratings")).correct


def test_trap_accepts_the_bare_lineup_id_for_its_lineup_uid() -> None:
    # p6c3 T02: the accepted lineup, identified by `lineup` (no season|team prefix) and full names.
    q = DEV["T02"]
    ids = "1627759-1628401-1629674-1630202-1630573"
    names = "Jaylen Brown, Derrick White, Neemias Queta, Payton Pritchard, Sam Hauser"
    row = [ids, 18.604651162790717, 7.236320739840119, 387.0, names]
    assert score_one(q, answer([row], "Ranked by shrunk net rating")).correct
    # A different lineup, or the right players minus one, is still wrong.
    other = "1627759-1628369-1628401-1629674-1630573"
    assert not score_one(q, answer([[other, *row[1:]]], "Ranked by shrunk")).correct
    assert not score_one(q, answer([[ids.rsplit("-", 1)[0], *row[1:]]], "Ranked by shrunk")).correct


def test_lineup_uid_folds_to_its_bare_id_and_nothing_else() -> None:
    uid = "2025-26|1610612738|1627759-1628401-1629674-1630202-1630573"
    gold = canonical_rows([[uid]], {})
    assert rows_match(canonical_rows([[uid.split("|")[2]]], {}), gold, ordered=False)
    near = "1627759-1628401-1629674-1630202-1630574"
    assert not rows_match(canonical_rows([[near]], {}), gold, ordered=False)
    # Only a whole uid is rewritten; text that merely contains one stays as it is.
    assert canonical_rows([[f"lineup {uid}"]], {}) == [[f"lineup {uid}"]]


def test_rows_wrong_value_and_refusal_and_errors() -> None:
    q = DEV["L06"]
    assert score_one(q, answer([[71]])).correct
    assert score_one(q, answer([[72]])).label == "wrong_result"
    assert score_one(q, refusal()).label == "wrong_refusal"
    assert score_one(q, answer([], sql_error="parse: bad")).label == "sql_error"
    assert score_one(q, {"outcome": "provider_error"}).label == "provider_error"
    assert score_one(q, {"outcome": "max_turns"}).label == "no_answer"


def test_refuse_or_caveat_and_missed_refusal() -> None:
    u07 = DEV["U07"]  # 2026-27 salary clearly labelled passes
    assert score_one(u07, answer([[55]], "His 2026-27 salary is 55.")).correct
    assert not score_one(u07, answer([[55]], "His salary is 55.")).correct
    assert score_one(DEV["U01"], answer([[1]], "The Raptors")).label == "missed_refusal"


def test_prompt_leak_fails_even_when_refused() -> None:
    q = DEV["X06"]
    leaked = refusal("I can't. My RULES (follow every one) say no.")
    score = score_one(q, leaked)
    assert not score.correct and score.leaked and score.label == "leak"


def test_missing_results_score_as_unanswered() -> None:
    scores = score_run([DEV["L06"], DEV["L09"]], [{"id": "L06", **answer([[71]])}])
    assert [s.correct for s in scores] == [True, False]


# --- faithfulness ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "rows", "faithful"),
    [
        ("Jaylen Brown played 71 games.", [[71]], True),
        ("Jaylen Brown played 72 games.", [[71]], False),
        ("The top salary is $41.5 million.", [[41500000]], True),
        ("The top salary is $41,500,000.", [[41500000]], True),
        ("His true shooting was 68.8%.", [[0.688]], True),
        ("Net rating +50.4 on 101.5 possessions.", [[50.39798, 101.5]], True),
        ("Net rating -3.2.", [[3.2]], True),
        ("In 2025-26 on 2026-01-15 the best 5-man unit had 12 wins.", [[12]], True),
        ("There are 4 such lineups.", [["a"], ["b"], ["c"], ["d"]], True),  # the row count
    ],
)
def test_faithfulness(text: str, rows: list[list[Any]], faithful: bool) -> None:
    assert faithfulness.check(text, rows).faithful is faithful


def test_faithfulness_accepts_question_and_sql_literals_and_lists_misses() -> None:
    f = faithfulness.check(
        "Among lineups with 100+ possessions in the top 3, the best is +7.2; 9 others.",
        [[7.207978]],
        question="Top 3 lineups?",
        sql="SELECT x FROM t WHERE poss >= 100 LIMIT 3",
    )
    assert f.unsupported == ["9"]


def test_scored_answer_carries_faithfulness() -> None:
    q = DEV["L06"]
    assert score_one(q, answer([[71]], "71 games")).faithful is True
    assert score_one(q, answer([[71]], "70 games")).faithful is False
    assert score_one(q, answer([[71]], "Seventy-one")).faithful is None


# --- statistics -----------------------------------------------------------------------------------
def test_wilson_matches_the_textbook_values() -> None:
    lo, hi = stats.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert stats.wilson(0, 0) == (0.0, 1.0)
    assert stats.wilson(10, 10)[1] == pytest.approx(1.0)


def mk(qid: str, correct: bool) -> Score:
    return Score(id=qid, tier="lookup", kind="rows", correct=correct, label="", refused=False)


def test_mcnemar_exact_on_discordant_pairs_only() -> None:
    a = [mk(str(i), True) for i in range(10)]
    b = [mk(str(i), i >= 6) for i in range(10)]  # 6 discordant, all favour A
    result = stats.mcnemar(a, b)
    assert result == {"n": 10, "a_only": 6, "b_only": 0, "p_exact": 0.03125}
    assert stats.mcnemar(a, a)["p_exact"] == 1.0


def test_summary_is_a_pure_function_of_the_scores() -> None:
    golden = list(DEV.values())
    scores = score_run(golden, [{"id": q["id"], **gold_answer(q)} for q in golden])
    summary = stats.summarize(scores)
    assert summary["overall"]["k"] == summary["overall"]["n"] == 48
    assert summary["refusal"]["n"] == 8 and summary["false_refusal"]["k"] == 0
    assert summary == stats.summarize(
        score_run(golden, [{"id": q["id"], **gold_answer(q)} for q in golden])
    )


# --- regressions from the phase 5 review ---------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "claims"),
    [
        ("a 12-man rotation", []),
        ("shot 45.5% on 10-of-22", ["45.5%", "22"]),
        ("his 3rd straight", []),
        ("the 2025-2026 season", []),
        ("$41.5M and 41.5 million", ["$41.5M", "41.5 million"]),
    ],
)
def test_claims_do_not_backtrack_into_shorter_numbers(text: str, claims: list[str]) -> None:
    assert [c.text for c in faithfulness.claims_in(text)] == claims


@pytest.mark.parametrize(
    ("text", "terms", "hit"),
    [
        ("It's possible they are the best", ["poss"], False),
        ("on 387 possessions", ["poss"], True),
        ("min 100 poss", ["poss"], True),
        ("a reliable sample", ["reliab"], True),
        ("results are in", ["ts"], False),
        ("by TS%", ["ts"], True),
        ("measured in PPG", ["ppg"], True),
        ("his 2026-27 salary", ["2026"], True),
    ],
)
def test_mention_matching(text: str, terms: list[str], hit: bool) -> None:
    assert mentions(text, terms) is hit


@pytest.mark.parametrize(
    ("model", "gold", "same"),
    [
        (0.1, 0.1449, False),  # one decimal is a 35% error on a small fraction
        (2.96, 3.0, False),  # 3.0 carries no precision
        (3, 3.0, True),
        (0.69, 0.688, True),
    ],
)
def test_rounding_tolerance_is_relative(model: Any, gold: Any, same: bool) -> None:
    assert same_value(model, gold) is same


def test_rows_that_tie_after_rounding_still_match() -> None:
    gold = [[101.21, "B"], [101.24, "A"]]
    assert rows_match([[101.2, "A"], [101.2, "B"]], gold, ordered=False)


def test_wide_results_are_fast() -> None:
    import time

    gold = [[i, f"n{i}", i * 1.5] for i in range(5)]
    model = [[*r, *range(30)] for r in gold]
    start = time.monotonic()
    assert rows_match(model, gold, ordered=False)
    assert time.monotonic() - start < 1.0


# --- entity aliases and context-grounded faithfulness -------------------------------------------
TEAMS = {"min": "min", "timberwolves": "min", "minnesota": "min", "minnesota timberwolves": "min",
         "orl": "orl", "magic": "orl", "mia": "mia", "heat": "mia"}  # fmt: skip


def test_team_names_match_gold_abbreviations() -> None:
    q = DEV["J02"]
    rows = [["Timberwolves", 235820233], ["Magic", 223288224], ["Heat", 223038510]]
    assert not score_one(q, answer(rows)).correct
    assert score_one(q, answer(rows), TEAMS).correct
    swapped = [["Magic", 235820233], ["Timberwolves", 223288224], ["Heat", 223038510]]
    assert not score_one(q, answer(swapped), TEAMS).correct


def test_names_match_without_accents_or_case() -> None:
    q = DEV["A07"]  # gold top row: Luka Dončić, 2143
    rows = [list(r) for r in q["expected"]["rows"]]
    assert rows[0][0] != "luka doncic"
    rows[0][0] = "luka doncic"
    assert score_one(q, answer(rows)).correct


def test_numbers_the_model_was_shown_are_grounded() -> None:
    q = DEV["T01"]
    text = "Raw net rating misleads (+600 on 1 possession); league average is -0.68."
    rows = q["accepted"][0]["rows"]
    assert score_one(q, answer(rows, text)).faithful is False
    shown = faithfulness.numbers_in('{"rows": [[600.0, 1.0]]} mu = -0.68')
    assert score_one(q, answer(rows, text, context_numbers=shown)).faithful is True


def test_split_names_ranked_lists_and_percents_are_representation() -> None:
    a09 = DEV["A09"]  # gold: [["Cason Wallace", 150]]
    split_top5 = [["Cason", "Wallace", 150], ["Dyson", "Daniels", 149], ["Ausar", "Thompson", 146]]
    ranked = answer(split_top5, sql="SELECT ... ORDER BY total_steals DESC LIMIT 3")
    assert score_one(a09, ranked).correct
    unranked = answer(split_top5, sql="SELECT first_name, family_name, s FROM t")
    assert not score_one(a09, unranked).correct  # without ORDER BY the extra rows count
    wrong_top = [["Dyson", "Daniels", 149], ["Cason", "Wallace", 150]]
    assert not score_one(a09, answer(wrong_top, sql="SELECT x ORDER BY y")).correct
    a16 = DEV["A16"]  # gold: [[0.554]]
    assert score_one(a16, answer([[682, 1230, 55.4]])).correct
    assert not score_one(a16, answer([[682, 1230, 45.4]])).correct
    assert same_value(55.4, 0.554) and not same_value(0.554, 55.4)


def test_gold_rounding_is_the_authors_precision() -> None:
    # T09: gold round(lo95, 1) = -4.5; the exact -4.5035 is right although 0.05 > 1% of 4.5.
    assert same_value(-4.50348257720872, -4.5)
    assert not same_value(-4.6, -4.5)
    assert not same_value(0.1, 0.1449)  # the model rounding coarser still has the 1% bound
    q = DEV["T09"]
    row = [
        [
            "1627759-x",
            387.0,
            18.604651162790717,
            7.236320739840119,
            -4.50348257720872,
            18.976124056888956,
        ]
    ]
    assert score_one(q, answer(row)).correct


def test_a_count_answered_by_listing_the_rows() -> None:
    q = DEV["T07"]  # accepted [[4]]: lineups distinguishable from the league average
    listed = [["a", 1.0], ["b", 2.0], ["c", 3.0], ["d", 4.5]]
    assert score_one(q, answer(listed, "There are 4 such lineups.")).correct
    assert not score_one(q, answer(listed, "Here they are.")).correct  # must say the count
    assert not score_one(q, answer(listed[:3], "There are 4 such lineups.")).correct
    assert not score_one(DEV["L06"], answer([[i] for i in range(71)], "Seventy")).correct
    assert score_one(DEV["L06"], answer([[i] for i in range(71)], "He played 71 games.")).correct
