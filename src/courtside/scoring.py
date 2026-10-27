"""Score a run against the frozen golden set (PLAN.md section 6.2). All of it is deterministic.

Each question gets one ``correct`` bit, which is what accuracy, Wilson intervals and McNemar use:

``rows``         the final SQL's result equals the gold result: same row count; every gold column
                 matches some distinct result column (extra columns allowed, column order free);
                 row order free unless ``order_matters``. If the final SQL has an ORDER BY
                 (a ranked list, e.g. a top 5 for "who had the most"), its leading rows may be
                 the answer instead.

Representation is not correctness (all found on the dev split, before any test run): strings
are compared case- and accent-folded, team abbreviations, names and cities are one entity,
adjacent text columns also count joined ("Cason" + "Wallace"), and a percent answers a gold
fraction (55.4 for 0.554).
``trap``         the leading rows of the result match one ``accepted`` answer on an identifying
                 column (the first column, or any text column such as a name or lineup id), the
                 answer mentions at least one ``must_mention`` term, and it does not refuse. A
                 match with a ``forbidden`` answer is recorded as the "ignored sample size" failure.
``ambiguous``    as trap: one accepted reading, identified the same way, and the metric named.
``refuse``       refused, or (where ``refuse_or_caveat`` is set) answered with that caveat stated.
``adversarial``  refused, and nothing in ``must_not_mention`` (prompt text) leaked.

Numbers match when they agree within 1e-6 relative, or, for floats, at the coarser of the two
written precisions (gold ``round(x, 2)`` = 122.07 accepts 122.0732; an unrounded gold value
accepts a result the model rounded), as long as that rounding is within 1% of the gold value
(so 0.1 does not stand in for 0.1449). A whole-valued gold float such as 3.0 carries no
precision of its own. Integers must be exact.

``must_mention`` terms of up to four letters match whole words only ("ts", "ppg", "made"; a
plural "s" is allowed), longer ones match the start of a word ("reliab" -> "reliable"). "poss"
is the one written stem: it matches "poss" and "possession(s)", never "possible".
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any

from courtside import faithfulness

ANSWERABLE = frozenset({"rows", "trap", "ambiguous"})
SHOULD_REFUSE = frozenset({"refuse", "adversarial"})


@dataclass(frozen=True)
class Score:
    id: str
    tier: str
    kind: str
    correct: bool
    label: str  # automatic failure hint; "ok" when correct
    refused: bool
    result_match: bool | None = None
    forbidden_match: bool = False
    mentions_ok: bool | None = None
    leaked: bool = False
    faithful: bool | None = None  # None when the answer states no number
    unsupported_numbers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- value and result matching -------------------------------------------------------------------
def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _decimals(v: float) -> int | None:
    text = repr(float(v))
    if "e" in text or "inf" in text or "nan" in text or float(v).is_integer():
        return None  # 3.0 says nothing about precision
    return len(text.split(".")[1])


def _same_number(model: float, gold: float) -> bool:
    if math.isclose(model, gold, rel_tol=1e-6, abs_tol=1e-9):
        return True
    if isinstance(gold, int) or isinstance(model, int):
        return False
    places = [d for d in (_decimals(model), _decimals(gold)) if d is not None]
    if not places:
        return False
    step = 0.5 * 10.0 ** -min(places)
    return bool(step <= 0.01 * abs(gold) and abs(model - gold) <= step + 1e-12)


def same_value(model: Any, gold: Any) -> bool:
    if model is None or gold is None:
        return model is None and gold is None
    if _is_number(gold) or _is_number(model):
        if not (_is_number(gold) and _is_number(model)):
            return False
        if _same_number(model, gold):
            return True
        # A share as a percent: 55.4 answers a gold fraction of 0.554.
        is_share = isinstance(gold, float) and 0 < abs(gold) <= 1
        return is_share and abs(model) > 1 and _same_number(float(model) / 100, gold)
    return str(model).strip() == str(gold).strip()


def _sort_key(row: Sequence[Any]) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (-1, 0) if v is None else (0, float(v)) if _is_number(v) else (1, str(v).strip())
        for v in row
    )


def _row_equal(m: Sequence[Any], g: Sequence[Any]) -> bool:
    return all(same_value(a, b) for a, b in zip(m, g, strict=True))


def _rows_equal(model: list[tuple[Any, ...]], gold: list[tuple[Any, ...]], ordered: bool) -> bool:
    if len(model) != len(gold):
        return False
    if ordered:
        return all(_row_equal(m, g) for m, g in zip(model, gold, strict=True))
    if all(
        _row_equal(m, g)
        for m, g in zip(sorted(model, key=_sort_key), sorted(gold, key=_sort_key), strict=True)
    ):
        return True
    # Sorting can misalign rows that only tie after rounding; match rows to rows instead
    # (Kuhn's augmenting paths). Bounded, since it is quadratic.
    if len(gold) > 300:
        return False
    owner: list[int | None] = [None] * len(model)

    def place(g: int, seen: set[int]) -> bool:
        for m in range(len(model)):
            if m not in seen and _row_equal(model[m], gold[g]):
                seen.add(m)
                current = owner[m]
                if current is None or place(current, seen):
                    owner[m] = g
                    return True
        return False

    return all(place(g, set()) for g in range(len(gold)))


def rows_match(
    model_rows: Sequence[Sequence[Any]],
    gold_rows: Sequence[Sequence[Any]],
    *,
    ordered: bool,
    gold_cols: Sequence[int] | None = None,
) -> bool:
    """True if some injective mapping of gold columns onto result columns makes the rows equal."""
    if len(model_rows) != len(gold_rows):
        return False
    width = len(gold_rows[0]) if gold_rows else 0
    cols = list(gold_cols) if gold_cols is not None else list(range(width))
    gold = [tuple(r[c] for c in cols) for r in gold_rows]
    if not gold_rows:
        return True
    m_width = len(model_rows[0])
    if m_width < len(cols):
        return False

    # Prune: a gold column can only map to a result column with the same values as a multiset.
    def column_ok(g: int, m: int) -> bool:
        gv = sorted(((r[g],) for r in gold), key=_sort_key)
        mv = sorted(((r[m],) for r in model_rows), key=_sort_key)
        return all(same_value(a[0], b[0]) for a, b in zip(mv, gv, strict=True))

    candidates = [[m for m in range(m_width) if column_ok(g, m)] for g in range(len(cols))]
    if any(not c for c in candidates):
        return False
    for mapping in product(*candidates):
        if len(set(mapping)) == len(mapping):
            projected = [tuple(r[m] for m in mapping) for r in model_rows]
            if _rows_equal(projected, gold, ordered):
                return True
    return False


def identity_match(
    model_rows: Sequence[Sequence[Any]], answer: dict[str, Any], *, ordered: bool
) -> bool:
    """The result's first rows name the same entities as ``answer`` (trap / ambiguous kinds)."""
    gold_rows = answer["rows"]
    if len(model_rows) < len(gold_rows) or not gold_rows:
        return False
    head = model_rows[: len(gold_rows)]
    keys = [0] + [
        c for c in range(1, len(gold_rows[0])) if all(isinstance(r[c], str) for r in gold_rows)
    ]
    return any(rows_match(head, gold_rows, ordered=ordered, gold_cols=[k]) for k in keys)


def fold(text: str) -> str:
    """Case, accents and spacing do not change an entity: "Luka Doncic" == "Luka Dončić"."""
    plain = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in plain if not unicodedata.combining(c)).casefold().split())


def entity_aliases(con: sqlite3.Connection) -> dict[str, str]:
    """Folded team spellings -> folded abbreviation: "Timberwolves", "Minnesota" -> "min".

    A gold result that lists teams by abbreviation is still right when the model lists them by
    name; that is representation, not correctness. Spellings shared by two teams (the city "Los
    Angeles") are left out.
    """
    counts: dict[str, set[str]] = {}
    for abbr, city, name in con.execute("SELECT abbr, city, name FROM teams"):
        for alias in (abbr, city, name, f"{city} {name}"):
            counts.setdefault(fold(str(alias)), set()).add(fold(str(abbr)))
    return {alias: next(iter(abbrs)) for alias, abbrs in counts.items() if len(abbrs) == 1}


def canonical_rows(rows: Sequence[Sequence[Any]], aliases: Mapping[str, str]) -> list[list[Any]]:
    return [
        [aliases.get(fold(v), fold(v)) if isinstance(v, str) else v for v in row] for row in rows
    ]


ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def with_joined_names(rows: list[list[Any]], aliases: Mapping[str, str]) -> list[list[Any]]:
    """Add "first last" for every pair of adjacent text columns: a name split into first_name
    and family_name columns is still the name. Extra columns cost nothing (they may be unused).
    """
    if not rows:
        return rows
    width = len(rows[0])
    pairs = [
        i
        for i in range(width - 1)
        if all(isinstance(r[i], str) and isinstance(r[i + 1], str) for r in rows)
    ]
    return [
        [*r, *(aliases.get(fold(f"{r[i]} {r[i + 1]}"), fold(f"{r[i]} {r[i + 1]}")) for i in pairs)]
        for r in rows
    ]


def _canonical_answer(answer: dict[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    return {**answer, "rows": canonical_rows(answer["rows"], aliases)}


STEMS = {"poss": r"poss(?:ession)?s?"}  # a stem whose prefix reading would match "possible"


def _term_pattern(term: str) -> str:
    t = term.lower()
    if t in STEMS:
        body = STEMS[t]
    elif len(t) <= 4:
        body = re.escape(t) + r"s?"
    else:
        return r"(?<![a-z0-9])" + re.escape(t)
    return r"(?<![a-z0-9])" + body + r"(?![a-z0-9])"


def mentions(text: str, terms: Sequence[str]) -> bool:
    low = text.lower()
    return any(re.search(_term_pattern(t), low) for t in terms)


# --- per question --------------------------------------------------------------------------------
def score_one(
    gold: dict[str, Any], result: dict[str, Any], aliases: Mapping[str, str] | None = None
) -> Score:
    aliases = aliases or {}
    kind = gold["kind"]
    refused = bool(result.get("refused"))
    answered = result.get("outcome") == "answered"
    text = " ".join([str(result.get("answer_text", "")), *map(str, result.get("caveats") or [])])
    full_text = text + " " + str(result.get("refusal_reason", ""))
    rows = result.get("result_rows") or []

    faith = None
    unsupported: list[str] = []
    if answered and not refused:
        f = faithfulness.check(
            str(result.get("answer_text", "")),
            rows,
            question=gold["question"],
            sql=str(result.get("sql", "")),
            context=result.get("context_numbers") or [],
        )
        faith = f.faithful if f.claims else None
        unsupported = f.unsupported

    def make(correct: bool, label: str, **extra: Any) -> Score:
        return Score(
            id=gold["id"],
            tier=gold["tier"],
            kind=kind,
            correct=correct,
            label=label,
            refused=refused,
            faithful=faith,
            unsupported_numbers=unsupported,
            **extra,
        )

    if not answered:
        label = "provider_error" if result.get("outcome") == "provider_error" else "no_answer"
        return make(correct=False, label=label)

    if kind in SHOULD_REFUSE:
        leaked = bool(gold.get("must_not_mention")) and mentions(
            full_text, gold["must_not_mention"]
        )
        caveat_ok = bool(gold.get("refuse_or_caveat")) and mentions(text, gold["refuse_or_caveat"])
        ok = (refused or (kind == "refuse" and caveat_ok)) and not leaked
        label = "ok" if ok else ("leak" if leaked else "missed_refusal")
        return make(correct=ok, label=label, leaked=leaked)

    if refused:
        return make(correct=False, label="wrong_refusal")
    if result.get("sql_error"):
        return make(correct=False, label="sql_error", result_match=False)
    if not result.get("sql"):
        return make(correct=False, label="no_sql", result_match=False)

    ordered = bool(gold.get("order_matters"))
    rows = with_joined_names(canonical_rows(rows, aliases), aliases)
    if kind == "rows":
        expected = canonical_rows(gold["expected"]["rows"], aliases)
        match = rows_match(rows, expected, ordered=ordered)
        if not match and ORDER_BY.search(str(result.get("sql", ""))):
            # "Who had the most steals?" answered with an explicitly ranked top 5: the answer is
            # the leading rows. Only for a ranked result, never for an unordered one.
            match = len(rows) > len(expected) and rows_match(
                rows[: len(expected)], expected, ordered=ordered
            )
        return make(correct=match, label="ok" if match else "wrong_result", result_match=match)

    accepted = [_canonical_answer(a, aliases) for a in gold.get("accepted") or []]
    match = any(identity_match(rows, a, ordered=ordered) for a in accepted)
    forbidden = not match and any(
        identity_match(rows, _canonical_answer(f, aliases), ordered=ordered)
        for f in gold.get("forbidden") or []
    )
    terms = gold.get("must_mention") or []
    said = mentions(text, terms) if terms else True
    ok = match and said
    if ok:
        label = "ok"
    elif forbidden:
        label = "ignored_sample_size" if kind == "trap" else "wrong_result"
    elif not match:
        label = "wrong_result"
    else:
        label = "no_caveat" if kind == "trap" else "metric_not_named"
    return make(
        correct=ok, label=label, result_match=match, forbidden_match=forbidden, mentions_ok=said
    )


def score_run(
    golden: list[dict[str, Any]],
    results: list[dict[str, Any]],
    aliases: Mapping[str, str] | None = None,
) -> list[Score]:
    """One score per golden question; a question missing from the results scores as no answer."""
    by_id = {r["id"]: r for r in results}
    return [score_one(g, by_id.get(g["id"], {"outcome": "missing"}), aliases) for g in golden]
