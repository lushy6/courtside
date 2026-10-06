"""The golden question set: authored by hand, frozen by execution, split deterministically.

Questions are written in ``eval/golden/source.yaml`` BEFORE any model has seen them. Freezing runs
each gold SQL against the real database and stores the result, so scoring never depends on a
model's opinion. The split is stratified by tier and decided by a hash of the question id, so it
cannot be tuned: the same ids always land in the same split, and the test split is locked by a
hash (``LOCK.json``) that a test enforces.

Question kinds and how each is meant to be scored (the scorer lands in phase 5):

``rows``         the model's final SQL result must contain the gold rows (extra columns allowed,
                 column order free, row order free unless ``order_matters``; floats within
                 tolerance).
``trap``         a naive query gives a confident wrong answer. ``accepted`` lists the legitimate
                 answers (there may be more than one defensible method) and ``forbidden`` the naive
                 one; the answer must be accepted, not forbidden, and mention what the method hid.
``ambiguous``    several defensible readings. The answer must match one ``accepted`` reading AND
                 state which metric or reading it used (``must_mention``).
``refuse``       the database cannot answer. The model must decline with a reason, not invent.
``adversarial``  a request to change data, leak the prompt or run forbidden SQL. The database
                 must be unchanged and nothing forbidden may execute.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from courtside.db import list_tables, run_query

TIERS = ("lookup", "aggregate", "join", "trap", "ambiguous", "unanswerable", "adversarial")
KINDS = ("rows", "trap", "ambiguous", "refuse", "adversarial")
TIER_PREFIX = {
    "lookup": "L",
    "aggregate": "A",
    "join": "J",
    "trap": "T",
    "ambiguous": "M",
    "unanswerable": "U",
    "adversarial": "X",
}
REFUSE_CATEGORIES = (
    "other_season",
    "prediction",
    "playoffs",
    "missing_data",
    "missing_player",
    "wrong_season",
    "write_request",
    "prompt_leak",
)
DEV_SHARE = 0.4
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval" / "golden" / "source.yaml"
GOLDEN_DIR = ROOT / "eval" / "golden"


@dataclass(frozen=True)
class Question:
    id: str
    tier: str
    kind: str
    question: str
    gold_sql: str = ""
    order_matters: bool = False
    notes: str = ""
    accepted_sql: tuple[str, ...] = ()
    forbidden_sql: tuple[str, ...] = ()
    must_mention: tuple[str, ...] = ()
    must_not_mention: tuple[str, ...] = ()
    refuse_category: str = ""
    refuse_or_caveat: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def load_source(path: Path = SOURCE) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = []
    for item in raw:
        tier = item["tier"]
        kind = item.get("kind") or _default_kind(tier)
        questions.append(
            Question(
                id=item["id"],
                tier=tier,
                kind=kind,
                question=" ".join(str(item["question"]).split()),
                gold_sql=" ".join(str(item.get("gold_sql", "")).split()),
                order_matters=bool(item.get("order_matters", False)),
                notes=" ".join(str(item.get("notes", "")).split()),
                accepted_sql=tuple(" ".join(s.split()) for s in item.get("accepted_sql", [])),
                forbidden_sql=tuple(" ".join(s.split()) for s in item.get("forbidden_sql", [])),
                must_mention=tuple(item.get("must_mention", [])),
                must_not_mention=tuple(item.get("must_not_mention", [])),
                refuse_category=item.get("refuse_category", ""),
                refuse_or_caveat=tuple(item.get("refuse_or_caveat", [])),
            )
        )
    return questions


def _default_kind(tier: str) -> str:
    return {
        "trap": "trap",
        "ambiguous": "ambiguous",
        "unanswerable": "refuse",
        "adversarial": "adversarial",
    }.get(tier, "rows")


# --- structure checks (no database needed) ------------------------------------------------------
def source_problems(questions: list[Question]) -> list[str]:
    found: list[str] = []
    ids = [q.id for q in questions]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        found.append(f"duplicate id {dup}")
    texts = [q.question.lower() for q in questions]
    for dup in sorted({t for t in texts if texts.count(t) > 1}):
        found.append(f"duplicate question text: {dup[:60]}")
    for q in questions:
        if q.tier not in TIERS:
            found.append(f"{q.id}: unknown tier {q.tier}")
            continue
        if not q.id.startswith(TIER_PREFIX[q.tier]):
            found.append(f"{q.id}: id prefix does not match tier {q.tier}")
        if q.kind not in KINDS:
            found.append(f"{q.id}: unknown kind {q.kind}")
        if q.kind == "rows" and not q.gold_sql:
            found.append(f"{q.id}: rows question has no gold_sql")
        if q.kind in {"trap", "ambiguous"} and not q.accepted_sql:
            found.append(f"{q.id}: {q.kind} question needs accepted_sql")
        if q.kind == "trap" and not (q.forbidden_sql or q.gold_sql or q.must_mention):
            found.append(f"{q.id}: trap question needs a forbidden_sql, gold_sql or must_mention")
        if q.kind == "ambiguous" and not q.must_mention:
            found.append(f"{q.id}: ambiguous question must require the answer to state its metric")
        if q.kind == "refuse" and q.refuse_category not in REFUSE_CATEGORIES:
            found.append(
                f"{q.id}: refuse_category {q.refuse_category!r} not in {REFUSE_CATEGORIES}"
            )
        if q.kind == "adversarial" and q.refuse_category not in REFUSE_CATEGORIES:
            found.append(f"{q.id}: adversarial needs a refuse_category")
        for sql in (q.gold_sql, *q.accepted_sql, *q.forbidden_sql):
            if sql and sql.upper().replace(" ", "").startswith("SELECT*"):
                found.append(f"{q.id}: gold SQL must name its columns, not SELECT *")
    return found


# --- split --------------------------------------------------------------------------------------
def assign_splits(questions: list[Question]) -> dict[str, str]:
    """Stratified by tier: within a tier, the first ceil(40%) by hash of id are dev, rest test."""
    splits: dict[str, str] = {}
    for tier in TIERS:
        ids = sorted(
            (q.id for q in questions if q.tier == tier),
            key=lambda i: hashlib.sha256(i.encode()).hexdigest(),
        )
        n_dev = math.ceil(len(ids) * DEV_SHARE)
        for position, qid in enumerate(ids):
            splits[qid] = "dev" if position < n_dev else "test"
    return splits


# --- freezing -----------------------------------------------------------------------------------
def _run(con: sqlite3.Connection, sql: str) -> dict[str, Any]:
    res = run_query(
        con,
        sql,
        allowed_tables=list_tables(con),
        max_rows=1000,
        timeout_s=10.0,
        max_total_chars=10**7,
    )
    if res.truncated:
        raise ValueError(f"gold SQL result was truncated: {sql[:80]}")
    return {"columns": res.columns, "rows": [list(r) for r in res.rows]}


def freeze(questions: list[Question], con: sqlite3.Connection) -> list[dict[str, Any]]:
    """Execute every gold, accepted and forbidden query; one JSON-ready record per question."""
    splits = assign_splits(questions)
    frozen = []
    for q in questions:
        rec: dict[str, Any] = {
            "id": q.id,
            "tier": q.tier,
            "kind": q.kind,
            "split": splits[q.id],
            "question": q.question,
            "order_matters": q.order_matters,
            "notes": q.notes,
            "gold_sql": q.gold_sql,
            "accepted_sql": list(q.accepted_sql),
            "forbidden_sql": list(q.forbidden_sql),
            "must_mention": list(q.must_mention),
            "must_not_mention": list(q.must_not_mention),
            "refuse_category": q.refuse_category,
            "refuse_or_caveat": list(q.refuse_or_caveat),
        }
        if q.gold_sql:
            rec["expected"] = _run(con, q.gold_sql)
        if q.accepted_sql:
            rec["accepted"] = [_run(con, s) for s in q.accepted_sql]
        if q.forbidden_sql:
            rec["forbidden"] = [_run(con, s) for s in q.forbidden_sql]
        frozen.append(rec)
    return frozen


def write_frozen(frozen: list[dict[str, Any]], out_dir: Path = GOLDEN_DIR) -> dict[str, str]:
    """Write dev.jsonl and test.jsonl; return their sha256 for the lock file."""
    hashes = {}
    for split in ("dev", "test"):
        lines = [
            json.dumps(r, sort_keys=True, ensure_ascii=False) for r in frozen if r["split"] == split
        ]
        path = out_dir / f"{split}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def read_frozen(split: str, out_dir: Path = GOLDEN_DIR) -> list[dict[str, Any]]:
    path = out_dir / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
