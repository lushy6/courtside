"""Deterministic faithfulness: every number the answer states must come from the query result.

A number in ``answer_text`` is supported when it matches, at the precision it is written with,
a numeric cell of the final result, the result's row count, or a literal in the question or in
the final SQL (a threshold such as "at least 100 possessions" is grounded in the query, not
invented). "41.5 million", "$41,500,000" and "41.5M" are read as the same value; "68.8%" also
matches a stored 0.688. A mismatch is a failure even when the SQL was right.

Not checked, on purpose: ISO dates and season labels ("2026-01-15", "2025-26"), which describe
scope rather than claim results, and numbers inside hyphenated compounds ("5-man", "3-point"),
which are adjectives. Unsupported numbers are listed so a failure can be checked by hand.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

_SCOPE = re.compile(r"\b\d{4}-\d{2}(?:\d{2})?(?:-\d{2})?\b")  # dates, 2025-26, 2025-2026
# The (?!\d)(?!\.\d)(?!,\d) guards stop backtracking into a shorter number ("12-man" -> "1").
_NUMBER = re.compile(
    r"(?<![\w.])[-+−]?\$?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?"
    r"(?!\d)(?!\.\d)(?!,\d)(?!-[A-Za-z])(?!(?:st|nd|rd|th)\b)"
    r"(?:\s?(?P<pct>%|percent\b))?"
    r"(?:\s?(?P<mult>million\b|billion\b|thousand\b|[MBK]\b))?"
)
_MULTIPLIER = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9}


@dataclass(frozen=True)
class Claim:
    text: str
    value: float
    decimals: int
    percent: bool
    multiplier: float


@dataclass(frozen=True)
class Faithfulness:
    claims: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def faithful(self) -> bool:
        return not self.unsupported


def claims_in(text: str) -> list[Claim]:
    found = []
    for m in _NUMBER.finditer(_SCOPE.sub(" ", text)):
        frac = m.group("frac") or ""
        mult = _MULTIPLIER.get((m.group("mult") or "").lower(), 1.0)
        found.append(
            Claim(
                text=m.group(0).strip(),
                value=float(m.group("int").replace(",", "") + frac),
                decimals=max(len(frac) - 1, 0),
                percent=m.group("pct") is not None,
                multiplier=mult,
            )
        )
    return found


def _numbers(values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]


def _supports(claim: Claim, value: float) -> bool:
    tolerance = 0.5 * 10.0**-claim.decimals * claim.multiplier + 1e-9
    target = claim.value * claim.multiplier
    candidates = [abs(value)]
    if claim.percent:
        candidates.append(abs(value) * 100)  # a share stored as a fraction
    return any(abs(c - target) <= tolerance for c in candidates)


def check(
    answer_text: str, rows: Sequence[Sequence[Any]], *, question: str = "", sql: str = ""
) -> Faithfulness:
    pool = _numbers(cell for row in rows for cell in row)
    pool.append(float(len(rows)))
    pool += [c.value * c.multiplier for c in claims_in(question)]
    pool += [float(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", sql)]
    claims = claims_in(answer_text)
    unsupported = [c.text for c in claims if not any(_supports(c, v) for v in pool)]
    return Faithfulness([c.text for c in claims], unsupported)
