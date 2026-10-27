"""The eval runner: daily quota, resume without repeated calls, replay reproduces the scores.

A scripted HTTP transport speaks Groq's wire format, so the real adapter, cassette, retry and
agent code all run; only the network is fake.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from courtside import runner, scoring, stats
from courtside.cassette import Cassette
from courtside.golden import read_frozen
from courtside.llm import GroqModel, RetryingTransport
from courtside.runner import DailyLimitReached, QuotaTransport

pytestmark = pytest.mark.skipif(not runner.DB.exists(), reason="run scripts/build_courtside_db.py")

IDS = ["L06", "L09", "J02"]
DEV = {q["id"]: q for q in read_frozen("dev")}


def groq_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    call = {
        "id": f"c_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
    return {
        "choices": [
            {"message": {"content": "", "tool_calls": [call]}, "finish_reason": "tool_calls"}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        "model": "fake",
    }


class FakeGroq:
    """Turn 1: run the gold SQL. Turn 2 (after a tool result): final_answer with it."""

    def __init__(self) -> None:
        self.bodies: list[str] = []

    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(json.dumps(body, sort_keys=True))
        question = next(m["content"] for m in body["messages"] if m["role"] == "user")
        gold = next(q for q in DEV.values() if question.startswith(q["question"]))
        if body["messages"][-1]["role"] != "tool":
            return groq_call("run_sql", {"query": gold["gold_sql"]})
        return groq_call(
            "final_answer",
            {"answer_text": "See the result.", "sql": gold["gold_sql"], "refused": False},
        )


def test_quota_counts_per_day_and_stops_at_the_limit(tmp_path: Path) -> None:
    fake = FakeGroq()
    clock = [1_000_000.0]
    slept: list[float] = []
    quota = QuotaTransport(
        fake, "groq", daily_limit=2, rpm=30, usage_dir=tmp_path,
        clock=lambda: clock[0], sleep=slept.append,
    )  # fmt: skip
    body = {"messages": [{"role": "user", "content": DEV["L06"]["question"]}]}
    quota.post("u", {}, body)
    quota.post("u", {}, body)
    assert slept == [2.0]  # 30 rpm = one request every 2 s
    with pytest.raises(DailyLimitReached):
        quota.post("u", {}, body)
    assert quota.used_today() == 2 and len(fake.bodies) == 2
    clock[0] += 86_400  # next UTC day
    quota.post("u", {}, body)
    assert quota.used_today() == 1


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int) -> tuple[Any, FakeGroq]:
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(runner, "CASSETTES", tmp_path / "cassettes")
    fake = FakeGroq()
    quota = QuotaTransport(fake, "groq", daily_limit=limit, usage_dir=tmp_path / "usage")
    cassette = Cassette(tmp_path / "cassettes" / "run.jsonl", "record", RetryingTransport(quota))
    model = GroqModel(model="fake", transport=cassette, key="k")
    agent = runner.build_agent(model, "C", max_turns=8, max_output_tokens=256)
    return agent, fake


def test_interrupted_run_resumes_without_repeating_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions = [DEV[i] for i in IDS]
    results = tmp_path / "results" / "run" / "results.jsonl"
    # Each question takes 2 calls; a limit of 3 stops halfway through the second question.
    agent, first = setup(tmp_path, monkeypatch, limit=3)
    assert runner.run_questions(agent, questions, results, log=lambda _: None) == (
        "paused_daily_limit"
    )
    assert [r["id"] for r in runner.read_results(results)] == ["L06"]

    usage = next((tmp_path / "usage").iterdir())
    usage.write_text(json.dumps({"requests": 0}), encoding="utf-8")  # a new day
    agent, second = setup(tmp_path, monkeypatch, limit=100)
    assert runner.run_questions(agent, questions, results, log=lambda _: None) == "complete"

    calls = first.bodies + second.bodies
    assert len(calls) == 6 and len(set(calls)) == 6  # 3 questions x 2 turns, none repeated
    done = runner.read_results(results)
    assert [r["id"] for r in done] == IDS
    assert all(s.correct for s in scoring.score_run(questions, done))

    # Replaying the cassette alone reproduces the scores exactly, without any network.
    manifest = {
        "run_id": "run", "split": "dev", "config": "C", "provider": "groq", "model": "fake",
        "question_ids": IDS, "max_turns": 8, "max_output_tokens": 256, "db_sha256": "x",
    }  # fmt: skip
    runner.write_json(tmp_path / "results" / "run" / "manifest.json", manifest)
    replayed = runner.replay_results("run")
    original = runner.score_report("run")["metrics"]
    assert stats.summarize(scoring.score_run(questions, replayed)) == original
    for again, first_time in zip(replayed, done, strict=True):
        for key in runner.REPLAY_FIELDS:
            assert again[key] == first_time[key], (again["id"], key)


def test_retry_errors_drops_provider_errors_before_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results" / "run" / "results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text(
        json.dumps({"id": "L06", "outcome": "provider_error"}) + "\n", encoding="utf-8"
    )
    agent, fake = setup(tmp_path, monkeypatch, limit=100)
    runner.run_questions(agent, [DEV["L06"]], results, log=lambda _: None)
    assert fake.bodies == []  # kept: a recorded provider error is a result
    runner.run_questions(agent, [DEV["L06"]], results, retry_errors=True, log=lambda _: None)
    assert len(fake.bodies) == 2
    assert [r["outcome"] for r in runner.read_results(results)] == ["answered"]


def test_resume_refuses_a_changed_setup(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    fixed = {k: 1 for k in runner.FIXED}
    runner.write_json(path, runner.open_manifest(path, fixed))
    assert len(runner.open_manifest(path, fixed)["sessions"]) == 2
    with pytest.raises(SystemExit, match="prompt_sha256"):
        runner.open_manifest(path, {**fixed, "prompt_sha256": 2})
    with pytest.raises(SystemExit, match="code_sha256"):
        runner.open_manifest(path, {**fixed, "code_sha256": 2})
    reopened = runner.open_manifest(path, {**fixed, "code_sha256": 2}, allow_code_change=True)
    assert reopened["sessions"][-1]["code_sha256"] == 2
    with pytest.raises(SystemExit, match="prompt_sha256"):  # the flag covers code only
        runner.open_manifest(
            path, {**fixed, "code_sha256": 2, "prompt_sha256": 2}, allow_code_change=True
        )


def test_test_split_needs_the_phase_7_flag() -> None:
    with pytest.raises(SystemExit):
        runner.run_main(["--split", "test"])


def test_default_run_id() -> None:
    assert runner.default_run_id("dev", "D", "groq", "qwen/qwen3.8-27b", 2) == (
        "dev-D-groq-qwen3.8-27b-r2"
    )


def test_a_provider_daily_cap_pauses_instead_of_recording_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from courtside.llm import RateLimited

    class Capped(FakeGroq):
        def post(
            self, url: str, headers: Mapping[str, str], body: dict[str, Any]
        ) -> dict[str, Any]:
            raise RateLimited(
                "Rate limit reached for model on tokens per day (TPD): Limit 200000", 700.0
            )

    monkeypatch.setattr(runner, "RESULTS", tmp_path / "results")
    quota = QuotaTransport(Capped(), "groq", daily_limit=100, usage_dir=tmp_path / "usage")
    cassette = Cassette(
        tmp_path / "c.jsonl", "record", RetryingTransport(quota, sleep=lambda s: None)
    )
    agent = runner.build_agent(
        GroqModel(model="fake", transport=cassette, key="k"),
        "C",
        max_turns=8,
        max_output_tokens=256,
    )
    results = tmp_path / "results" / "run" / "results.jsonl"
    status = runner.run_questions(agent, [DEV["L06"]], results, log=lambda _: None)
    assert status == "paused_daily_limit" and runner.read_results(results) == []
    assert quota.requests == 1  # no retry loop against a daily cap


def test_a_per_minute_429_that_mentions_daily_quotas_is_retried_not_paused(tmp_path: Path) -> None:
    from courtside.llm import RateLimited

    class MinuteThenOk(FakeGroq):
        calls = 0

        def post(
            self, url: str, headers: Mapping[str, str], body: dict[str, Any]
        ) -> dict[str, Any]:
            MinuteThenOk.calls += 1
            if MinuteThenOk.calls == 1:
                raise RateLimited(
                    "quota GenerateRequestsPerDayPerProjectPerModel; per minute", 15.0
                )
            return super().post(url, headers, body)

    quota = QuotaTransport(MinuteThenOk(), "gemini", daily_limit=100, usage_dir=tmp_path)
    retry = RetryingTransport(quota, sleep=lambda s: None)
    body = {"messages": [{"role": "user", "content": DEV["L06"]["question"]}]}
    assert "choices" in retry.post("u", {}, body)  # waited 15 s and retried
    assert quota.requests == 2
