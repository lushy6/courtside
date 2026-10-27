"""Scorer, faithfulness check and statistics: offline, against the frozen dev split only."""

from __future__ import annotations

from typing import Any

import pytest

from courtside import faithfulness, stats
from courtside.golden import read_frozen
from courtside.scoring import Score, rows_match, same_value, score_one, score_run

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
        (0.7, 0.688, True),
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
