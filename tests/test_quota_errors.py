"""HTTP quota metadata survives diagnostic truncation; only exceeded daily quotas pause."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from courtside.llm import HttpTransport, RateLimited, RetryingTransport
from courtside.runner import DailyLimitReached, QuotaTransport


def gemini_error(quota_ids: list[str], retry: str | None) -> dict[str, Any]:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.Help",
            "links": [
                {"description": "Quota documentation", "url": "https://example.invalid/PerDay"}
            ],
        },
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": quota_id,
                    "quotaDimensions": {"location": "global", "model": "fixture"},
                    "quotaValue": "10",
                }
                for quota_id in quota_ids
            ],
        },
    ]
    if retry is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
    return {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }


def http_quota(
    body: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> QuotaTransport:
    payload = json.dumps(body).encode()
    responses = [
        urllib.error.HTTPError(
            "https://example.invalid", 429, "quota", Message(), io.BytesIO(payload)
        ),
        b'{"ok": true}',
    ]

    def urlopen(*args: Any, **kwargs: Any) -> io.BytesIO:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(response)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return QuotaTransport(HttpTransport(), "fixture", daily_limit=100, usage_dir=tmp_path)


@pytest.mark.parametrize("retry", [None, "15s", "120s"])
def test_daily_violation_pauses_even_with_a_short_or_missing_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: str | None
) -> None:
    quota = http_quota(
        gemini_error(["GenerateRequestsPerDayPerProjectPerModel"], retry), tmp_path, monkeypatch
    )
    with pytest.raises(DailyLimitReached):
        RetryingTransport(quota, sleep=lambda _: None).post("https://example.invalid", {}, {})
    assert quota.requests == 1


@pytest.mark.parametrize("retry,wait", [(None, 3.0), ("15s", 16.0)])
def test_minute_violation_retries_despite_a_daily_quota_help_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: str | None, wait: float
) -> None:
    body = gemini_error(["GenerateRequestsPerMinutePerProjectPerModel"], retry)
    if retry:
        assert json.dumps(body).index("retryDelay") > 500
    quota = http_quota(body, tmp_path, monkeypatch)
    waits: list[float] = []
    assert RetryingTransport(quota, sleep=waits.append).post("https://example.invalid", {}, {}) == {
        "ok": True
    }
    assert waits == [wait]


def test_a_long_retry_hint_alone_does_not_make_a_minute_violation_daily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quota = http_quota(
        gemini_error(["GenerateRequestsPerMinutePerProjectPerModel"], "900s"), tmp_path, monkeypatch
    )
    with pytest.raises(RateLimited) as raised:
        quota.post("https://example.invalid", {}, {})
    assert raised.value.retry_after_s == 900.0


def test_any_daily_violation_pauses_when_multiple_quotas_are_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = gemini_error(
        ["GenerateRequestsPerMinutePerProjectPerModel", "GenerateRequestsPerDayPerProjectPerModel"],
        "15s",
    )
    quota = http_quota(body, tmp_path, monkeypatch)
    with pytest.raises(DailyLimitReached):
        quota.post("https://example.invalid", {}, {})


@pytest.mark.parametrize(
    "description,daily",
    [
        ("Generate content requests per day limit exceeded on FreeTier", True),
        ("Generate content requests per minute limit exceeded on FreeTier", False),
    ],
)
def test_gemini_message_alone_identifies_the_period_when_details_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, description: str, daily: bool
) -> None:
    # Wording from the real 2026-10-26 cap (limit: 500, gemini-3.5-flash-lite), details dropped.
    body = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for metric: generativelanguage.googleapis.com/"
            f"generate_content_free_tier_requests, limit: 500, description: {description}",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    quota = http_quota(body, tmp_path, monkeypatch)
    if daily:
        with pytest.raises(DailyLimitReached):
            quota.post("https://example.invalid", {}, {})
    else:
        with pytest.raises(RateLimited):
            quota.post("https://example.invalid", {}, {})


@pytest.mark.parametrize(
    "period,daily",
    [
        ("tokens per day (TPD)", True),
        ("requests per day (RPD)", True),
        ("tokens per minute (TPM)", False),
    ],
)
def test_groq_identifies_the_exceeded_period_from_the_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, period: str, daily: bool
) -> None:
    body = {
        "error": {
            "message": f"Rate limit reached for model fixture on {period}: Limit 200000. Please try again in 2s.",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
    quota = http_quota(body, tmp_path, monkeypatch)
    if daily:
        with pytest.raises(DailyLimitReached):
            quota.post("https://example.invalid", {}, {})
    else:
        waits: list[float] = []
        assert RetryingTransport(quota, sleep=waits.append).post(
            "https://example.invalid", {}, {}
        ) == {"ok": True}
        assert waits == [3.0]
