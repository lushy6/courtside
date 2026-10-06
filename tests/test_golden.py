"""The golden set: shape, fairness, and that the frozen files are exactly what the source produces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from courtside.db import open_readonly
from courtside.golden import (
    GOLDEN_DIR,
    KINDS,
    TIERS,
    Question,
    assign_splits,
    freeze,
    load_source,
    read_frozen,
    source_problems,
)

DB = Path(__file__).resolve().parents[1] / "data" / "courtside_nba.db"
needs_db = pytest.mark.skipif(not DB.exists(), reason="run scripts/build_courtside_db.py")
EXPECTED_TIERS = {
    "lookup": 20,
    "aggregate": 25,
    "join": 25,
    "trap": 20,
    "ambiguous": 10,
    "unanswerable": 10,
    "adversarial": 10,
}


@pytest.fixture(scope="module")
def questions() -> list[Question]:
    return load_source()


# --- shape ------------------------------------------------------------------------------------
def test_source_has_no_structural_problems(questions: list[Question]) -> None:
    assert source_problems(questions) == []


def test_tier_counts_match_the_plan(questions: list[Question]) -> None:
    assert dict(Counter(q.tier for q in questions)) == EXPECTED_TIERS
    assert len(questions) == 120


def test_every_tier_and_kind_is_known(questions: list[Question]) -> None:
    assert {q.tier for q in questions} <= set(TIERS)
    assert {q.kind for q in questions} <= set(KINDS)


def test_questions_are_plain_english_not_sql(questions: list[Question]) -> None:
    for q in questions:
        if q.tier == "adversarial":
            continue  # these are deliberately attack-shaped
        assert not q.question.upper().startswith(("SELECT", "WITH ")), q.id
        assert len(q.question) < 220, q.id


def test_refusal_and_adversarial_items_have_no_gold_answer(questions: list[Question]) -> None:
    for q in questions:
        if q.kind in {"refuse", "adversarial"}:
            assert not q.gold_sql and not q.accepted_sql, q.id


# --- split --------------------------------------------------------------------------------------
def test_split_is_deterministic_and_stratified(questions: list[Question]) -> None:
    a, b = assign_splits(questions), assign_splits(list(reversed(questions)))
    assert a == b
    dev = Counter(q.tier for q in questions if a[q.id] == "dev")
    assert dev == {
        "lookup": 8,
        "aggregate": 10,
        "join": 10,
        "trap": 8,
        "ambiguous": 4,
        "unanswerable": 4,
        "adversarial": 4,
    }
    assert Counter(a.values()) == {"dev": 48, "test": 72}


def test_adding_a_question_does_not_reshuffle_the_others_within_a_tier_prefix() -> None:
    """The split depends on id hashes only, so it cannot be nudged by editing question text."""
    qs = [
        Question(id=f"L{i:02d}", tier="lookup", kind="rows", question=f"q{i}", gold_sql="x")
        for i in range(20)
    ]
    edited = [
        Question(
            id=q.id, tier=q.tier, kind=q.kind, question=q.question + " (reworded)", gold_sql="y"
        )
        for q in qs
    ]
    assert assign_splits(qs) == assign_splits(edited)


# --- frozen files are consistent with the source ----------------------------------------------------
@needs_db
def test_frozen_files_are_exactly_what_freezing_the_source_produces(
    questions: list[Question],
) -> None:
    fresh = {r["id"]: r for r in freeze(questions, open_readonly(DB))}
    stored = {r["id"]: r for r in read_frozen("dev") + read_frozen("test")}
    assert set(stored) == set(fresh)
    for qid, rec in fresh.items():
        assert json.loads(json.dumps(rec)) == stored[qid], f"{qid} drifted from the frozen file"


@needs_db
def test_lock_hashes_match_the_files_on_disk() -> None:
    lock = json.loads((GOLDEN_DIR / "LOCK.json").read_text(encoding="utf-8"))
    for split in ("dev", "test"):
        actual = hashlib.sha256((GOLDEN_DIR / f"{split}.jsonl").read_bytes()).hexdigest()
        assert actual == lock[f"{split}_sha256"], f"{split}.jsonl changed without a re-freeze"
    assert lock["db_sha256"] == hashlib.sha256(DB.read_bytes()).hexdigest(), (
        "database changed since freeze"
    )
    assert lock["questions"] == 120 and lock["by_split"] == {"dev": 48, "test": 72}


# --- the gold answers are usable ---------------------------------------------------------------------
@needs_db
def test_every_rows_question_has_a_nonempty_answer() -> None:
    for rec in read_frozen("dev") + read_frozen("test"):
        if rec["kind"] == "rows":
            assert rec["expected"]["rows"], rec["id"]


@needs_db
def test_no_gold_result_was_truncated_or_huge() -> None:
    for rec in read_frozen("dev") + read_frozen("test"):
        for res in [rec.get("expected"), *rec.get("accepted", []), *rec.get("forbidden", [])]:
            if res:
                assert 0 < len(res["rows"]) <= 50, rec["id"]


@needs_db
def test_every_trap_is_a_real_trap_the_naive_answer_differs_from_the_accepted_one() -> None:
    """If a forbidden query returned what an accepted one does, the question would test nothing."""
    for rec in read_frozen("dev") + read_frozen("test"):
        if rec["kind"] != "trap" or not rec.get("forbidden"):
            continue
        accepted_first = [tuple(a["rows"][0]) for a in rec["accepted"]]
        for forb in rec["forbidden"]:
            assert tuple(forb["rows"][0]) not in accepted_first, f"{rec['id']}: naive == accepted"


@needs_db
def test_the_one_possession_fluke_really_is_the_raw_top_lineup() -> None:
    rec = next(r for r in read_frozen("dev") + read_frozen("test") if r["id"] == "T01")
    poss_col = rec["forbidden"][0]["columns"].index("poss")
    assert rec["forbidden"][0]["rows"][0][poss_col] < 5
    for acc in rec["accepted"]:
        assert acc["rows"][0][acc["columns"].index("poss")] >= 30


@needs_db
def test_ambiguous_accepted_readings_disagree_enough_to_be_worth_asking() -> None:
    """If every reading returned the same key, the question would not be ambiguous."""
    disagree = 0
    for rec in read_frozen("dev") + read_frozen("test"):
        if rec["kind"] == "ambiguous":
            keys = {tuple(a["rows"][0][:1]) for a in rec["accepted"]}
            disagree += len(keys) > 1
    assert disagree >= 4
