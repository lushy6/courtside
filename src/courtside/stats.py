"""Wilson intervals, exact McNemar, and the per-run summary that score.py prints and stores.

The summary splits ``metrics`` (a pure function of the golden set and the answers, so a replay
from the cassette must reproduce them exactly) from ``cost`` (tokens, turns, seconds; wall-clock
time differs on replay and is never part of the reproducibility check).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from scipy.stats import binomtest

from courtside.golden import TIERS
from courtside.scoring import ANSWERABLE, SHOULD_REFUSE, Score

ROW_TIERS = ("lookup", "aggregate", "join")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 6) if n else None,
        "ci95": [round(lo, 6), round(hi, 6)],
    }


def mcnemar(a: Sequence[Score], b: Sequence[Score]) -> dict[str, Any]:
    """Exact McNemar on the questions both runs scored; b_only/a_only are the discordant pairs."""
    sa = {s.id: s.correct for s in a}
    sb = {s.id: s.correct for s in b}
    common = sorted(sa.keys() & sb.keys())
    a_only = sum(1 for i in common if sa[i] and not sb[i])
    b_only = sum(1 for i in common if sb[i] and not sa[i])
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5).pvalue)
    return {"n": len(common), "a_only": a_only, "b_only": b_only, "p_exact": round(p, 6)}


def summarize(scores: Sequence[Score]) -> dict[str, Any]:
    def count(pred: Any) -> tuple[int, int]:
        chosen = [s for s in scores if pred(s)]
        return sum(s.correct for s in chosen), len(chosen)

    by_tier = {t: rate(*count(lambda s, t=t: s.tier == t)) for t in TIERS}
    answerable = [s for s in scores if s.kind in ANSWERABLE]
    with_numbers = [s for s in scores if s.faithful is not None]
    return {
        "overall": rate(*count(lambda s: True)),
        "execution": rate(*count(lambda s: s.tier in ROW_TIERS)),
        "by_tier": by_tier,
        "refusal": rate(*count(lambda s: s.kind in SHOULD_REFUSE)),
        "false_refusal": rate(sum(s.refused for s in answerable), len(answerable)),
        "faithfulness": rate(sum(bool(s.faithful) for s in with_numbers), len(with_numbers)),
        "leaks": sum(s.leaked for s in scores),
        "labels": dict(sorted(Counter(s.label for s in scores).items())),
        "per_question": {s.id: s.correct for s in scores},
    }


def cost(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(results) or 1
    total = {
        k: sum(int(r.get(k, 0) or 0) for r in results)
        for k in ("input_tokens", "output_tokens", "cached_input_tokens", "turns")
    }
    return {
        **total,
        "questions": len(results),
        "mean_turns": round(total["turns"] / n, 3),
        "mean_tool_calls": round(sum(len(r.get("tool_calls") or []) for r in results) / n, 3),
        "mean_seconds": round(sum(float(r.get("elapsed_s", 0.0)) for r in results) / n, 3),
    }
