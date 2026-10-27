"""Run one configuration over a golden split, resumably, and score runs (PLAN.md phase 5).

    python eval/run.py --provider groq --model qwen/qwen3.8-27b --config D --split dev
    python eval/score.py dev-D-groq-qwen3.8-27b-r1 [more runs] [--compare A B] [--verify-replay]

A run lives in ``eval/results/<run_id>/``: ``results.jsonl`` (one line per question, appended as
it finishes) and ``manifest.json`` (everything needed to trace a number back: git SHA, model,
config, prompt hash, semantic-layer, DB and golden-set hashes). Its provider calls go to
``eval/cassettes/<run_id>.jsonl``. Re-running the same command resumes: finished questions are
skipped and a question cut off mid-conversation replays its recorded turns, so no call is
repeated. A resume refuses to continue if anything that defines the run has changed.

Free tiers are capped per day, so every real HTTP attempt is counted per provider and UTC day in
``eval/results/_usage/`` and the run pauses (exit code 3) at ``--daily-limit``; run it again
tomorrow. ``--rpm`` spaces requests out; HTTP 429s are waited out by ``RetryingTransport``.

The test split is frozen until phase 7 and needs ``--frozen-test`` to run at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from courtside import __version__, scoring, stats
from courtside.agent import CONFIGS, Agent
from courtside.cassette import Cassette, CassetteMiss
from courtside.db import open_readonly
from courtside.golden import GOLDEN_DIR, read_frozen
from courtside.llm import (
    ChatModel,
    HttpTransport,
    RateLimited,
    RetryingTransport,
    Transport,
    make_model,
)
from courtside.semantic import SemanticLayer

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "courtside_nba.db"
SEMANTIC = ROOT / "semantic" / "nba.yaml"
RESULTS = ROOT / "eval" / "results"
CASSETTES = ROOT / "eval" / "cassettes"
USAGE = RESULTS / "_usage"
EXIT_PAUSED = 3
DAILY_WAIT_S = 300.0  # a retry hint longer than this is a daily cap, not a per-minute one
PER_DAY = re.compile(r"per day|PerDay|\(TPD\)|\(RPD\)", re.IGNORECASE)

# Fields that define a run; a resume with any of them changed would mix two experiments.
FIXED = (
    "run_id", "split", "config", "provider", "model", "rep", "prompt_sha256",
    "semantic_sha256", "db_sha256", "golden_sha256", "question_ids", "max_turns",
    "max_output_tokens", "code_sha256", "base_cassette",
)  # fmt: skip
# The code that shapes requests and runs the loop; a change to it changes the experiment.
CODE = ("agent.py", "llm.py", "tools.py", "db.py", "validate.py", "entities.py", "semantic.py")
REPLAY_FIELDS = ("outcome", "turns", "tool_calls", "sql", "answer_text", "refused", "result_rows")


class DailyLimitReached(RuntimeError):
    """The provider's daily request budget is spent. Not a ProviderError: nothing may swallow it."""


class QuotaTransport:
    """Counts every real HTTP attempt per provider and UTC day, and spaces requests out."""

    def __init__(
        self,
        inner: Transport,
        provider: str,
        *,
        daily_limit: int,
        rpm: float = 0.0,
        usage_dir: Path = USAGE,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.provider = provider
        self.daily_limit = daily_limit
        self.rpm = rpm
        self.usage_dir = usage_dir
        self.clock = clock
        self.sleep = sleep
        self.requests = 0  # made by this process
        self._last = float("-inf")

    def _path(self) -> Path:
        day = datetime.fromtimestamp(self.clock(), UTC).date().isoformat()
        return self.usage_dir / f"{self.provider}-{day}.json"

    def used_today(self) -> int:
        path = self._path()
        return int(json.loads(path.read_text(encoding="utf-8"))["requests"]) if path.exists() else 0

    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        used = self.used_today()
        if used >= self.daily_limit:
            raise DailyLimitReached(
                f"{self.provider}: {used} requests today (UTC), limit {self.daily_limit}"
            )
        if self.rpm > 0:
            wait = self._last + 60.0 / self.rpm - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"requests": used + 1}), encoding="utf-8")  # count before
        self.requests += 1
        try:
            return self.inner.post(url, headers, body)
        except RateLimited as error:
            # The provider's own daily cap (Groq: tokens per day; Gemini: requests per day).
            # Waiting minutes per question would only turn the rest of the run into errors.
            # A per-minute 429 can still mention per-day quotas in its details (Gemini lists
            # them); only a long retry hint means today's budget is really gone.
            hint = error.retry_after_s
            if PER_DAY.search(str(error)) and (hint is None or hint > DAILY_WAIT_S):
                raise DailyLimitReached(f"{self.provider}: {str(error)[:600]}") from error
            raise


# --- hashes and manifest -------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SCORER = ("scoring.py", "faithfulness.py", "stats.py")


def code_sha256(files: tuple[str, ...] = CODE) -> str:
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode() + b"/" + (Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def prompt_sha256(agent: Agent) -> str:
    tools = json.dumps([asdict(t) for t in agent.tools], sort_keys=True)
    return hashlib.sha256((agent.system + "\n" + tools).encode("utf-8")).hexdigest()


def git_state() -> tuple[str, bool]:
    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    sha = git("rev-parse", "HEAD") or "unknown"
    dirty = [
        line
        for line in git("status", "--porcelain").splitlines()
        if not re.search(r"eval/(results|cassettes)/", line)  # the run's own outputs
    ]
    return sha, bool(dirty)


def default_run_id(split: str, config: str, provider: str, model: str, rep: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", model.split("/")[-1]).strip("-")
    return f"{split}-{config}-{provider}-{slug}-r{rep}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def open_manifest(
    path: Path, fixed: dict[str, Any], *, allow_code_change: bool = False
) -> dict[str, Any]:
    """Create the manifest, or reopen it for a resume if nothing that defines the run changed.

    ``allow_code_change`` lets a resume continue when only ``code_sha256`` differs (for a
    change that does not alter requests, e.g. recording a new result field). The new hash is
    logged on the session, so the manifest still says exactly which code produced what.
    """
    sha, dirty = git_state()
    session: dict[str, Any] = {"started_at": now(), "git_sha": sha, "git_dirty": dirty}
    if path.exists():
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        changed = [k for k in FIXED if manifest.get(k) != fixed.get(k)]
        if allow_code_change and changed == ["code_sha256"]:
            session["code_sha256"] = fixed["code_sha256"]
            changed = []
        if changed:
            raise SystemExit(
                f"{path.parent.name}: cannot resume, {', '.join(changed)} changed since the run "
                "started. Use a new --run-id (or --rep) for the changed setup."
            )
    else:
        manifest = {
            **fixed,
            "created_at": session["started_at"],
            "git_sha": sha,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "courtside": __version__,
            "sessions": [],
        }
    manifest["sessions"].append(session)
    return manifest


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- running -------------------------------------------------------------------------------------
def run_questions(
    agent: Agent,
    questions: list[dict[str, Any]],
    results_path: Path,
    *,
    retry_errors: bool = False,
    log: Callable[[str], None] = print,
) -> str:
    """Ask each question not yet in ``results_path``; "complete" or "paused_daily_limit"."""
    done = read_results(results_path)
    if retry_errors:
        kept = [r for r in done if r.get("outcome") != "provider_error"]
        if len(kept) != len(done):
            results_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                encoding="utf-8",
                newline="\n",
            )
        done = kept
    finished = {r["id"] for r in done}
    todo = [q for q in questions if q["id"] not in finished]
    log(f"{len(finished)} done, {len(todo)} to ask")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    for q in todo:
        try:
            result = agent.ask(q["question"])
        except DailyLimitReached as error:
            log(f"paused: {error}")
            return "paused_daily_limit"
        record = {"id": q["id"], "tier": q["tier"], "kind": q["kind"], **result.to_dict()}
        with results_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        log(f"{q['id']}: {result.outcome} ({result.turns} turns, {result.elapsed_s:.1f} s)")
    return "complete"


def select(split: str, ids: str | None, limit: int | None) -> list[dict[str, Any]]:
    questions = read_frozen(split)
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        unknown = sorted(set(wanted) - {q["id"] for q in questions})
        if unknown:
            raise SystemExit(f"not in the {split} split: {', '.join(unknown)}")
        questions = [q for q in questions if q["id"] in wanted]
    return questions[:limit] if limit else questions


def build_agent(model: ChatModel, config: str, *, max_turns: int, max_output_tokens: int) -> Agent:
    con = open_readonly(DB)
    return Agent(
        model,
        con,
        SemanticLayer.load(),
        CONFIGS[config],
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
    )


def run_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one Courtside config over a golden split.")
    p.add_argument("--provider", default="groq", choices=["groq", "gemini"])
    p.add_argument("--model", default=None)
    p.add_argument("--config", default="D", choices=sorted(CONFIGS))
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--frozen-test", action="store_true", help="allow --split test (phase 7 only)")
    p.add_argument("--rep", type=int, default=1, help="repetition number (new calls per rep)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--ids", default=None, help="comma-separated question ids (use a --run-id)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--daily-limit", type=int, default=900, help="real requests per UTC day")
    p.add_argument("--rpm", type=float, default=20.0, help="max requests per minute (0 = off)")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--retry-errors", action="store_true", help="re-ask provider_error questions")
    p.add_argument(
        "--base-cassette",
        default=None,
        metavar="RUN_ID",
        help="start from that run's recorded calls: identical requests replay, so only "
        "conversations a change actually alters make new calls (a controlled comparison)",
    )
    p.add_argument(
        "--allow-code-change",
        action="store_true",
        help="resume although code_sha256 changed (only for changes that keep requests identical)",
    )
    args = p.parse_args(argv)
    if args.split == "test" and not args.frozen_test:
        p.error("the test split is frozen until phase 7; pass --frozen-test only then")
    if (args.ids or args.limit) and not args.run_id:
        p.error("a subset run needs its own --run-id")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # live progress in logs

    questions = select(args.split, args.ids, args.limit)
    quota = QuotaTransport(
        HttpTransport(), args.provider, daily_limit=args.daily_limit, rpm=args.rpm
    )
    probe = make_model(args.provider, args.model, quota)
    run_id = args.run_id or default_run_id(
        args.split, args.config, args.provider, probe.model, args.rep
    )
    cassette_path = CASSETTES / f"{run_id}.jsonl"
    if args.base_cassette and not cassette_path.exists():
        base = CASSETTES / f"{args.base_cassette}.jsonl"
        if not base.exists():
            p.error(f"no cassette for base run {args.base_cassette}")
        shutil.copyfile(base, cassette_path)
    cassette = Cassette(cassette_path, "record", RetryingTransport(quota))
    model = make_model(args.provider, probe.model, cassette)
    agent = build_agent(
        model, args.config, max_turns=args.max_turns, max_output_tokens=args.max_output_tokens
    )
    out = RESULTS / run_id
    fixed = {
        "run_id": run_id,
        "split": args.split,
        "config": args.config,
        "provider": args.provider,
        "model": model.model,
        "rep": args.rep,
        "prompt_sha256": prompt_sha256(agent),
        "code_sha256": code_sha256(),
        "base_cassette": args.base_cassette,
        "semantic_sha256": sha256_file(SEMANTIC),
        "db_sha256": sha256_file(DB),
        "golden_sha256": sha256_file(GOLDEN_DIR / f"{args.split}.jsonl"),
        "question_ids": [q["id"] for q in questions],
        "max_turns": args.max_turns,
        "max_output_tokens": args.max_output_tokens,
    }
    manifest = open_manifest(out / "manifest.json", fixed, allow_code_change=args.allow_code_change)
    write_json(out / "manifest.json", manifest)
    print(f"run {run_id}: {len(questions)} questions, {quota.used_today()} requests used today")
    try:
        status = run_questions(
            agent, questions, out / "results.jsonl", retry_errors=args.retry_errors
        )
    except KeyboardInterrupt:
        status = "interrupted"
    session = manifest["sessions"][-1]
    session.update(
        finished_at=now(),
        status=status,
        http_requests=quota.requests,
        cassette_hits=cassette.hits,
        cassette_new=cassette.misses,
    )
    manifest["status"] = status
    manifest["answered"] = len(read_results(out / "results.jsonl"))
    manifest["db_sha256_after"] = sha256_file(DB)  # the safety claim: never modified
    write_json(out / "manifest.json", manifest)
    print(f"{status}: {manifest['answered']}/{len(questions)} in {out.relative_to(ROOT)}")
    return 0 if status == "complete" else EXIT_PAUSED


# --- scoring -------------------------------------------------------------------------------------
def load_run(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    out = RESULTS / run_id
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(manifest["question_ids"])
    golden = [q for q in read_frozen(manifest["split"]) if q["id"] in wanted]
    return manifest, golden, read_results(out / "results.jsonl")


@cache
def aliases() -> dict[str, str]:
    con = open_readonly(DB)
    try:
        return scoring.entity_aliases(con)
    finally:
        con.close()


def differing(replayed: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[str]:
    """Questions whose replay is not the recorded conversation (see REPLAY_FIELDS)."""
    original = {r["id"]: r for r in results}
    out = []
    for r in replayed:
        first = original[r["id"]]
        # an unrecorded provider error ends the replay at an earlier turn
        keys = ("outcome",) if first.get("outcome") == "provider_error" else REPLAY_FIELDS
        if any(r.get(k) != first.get(k) for k in keys):
            out.append(r["id"])
    return out


def refresh_results(run_id: str) -> str:
    """Rewrite results.jsonl from a cassette replay, if the replay is the same conversation.

    Lets a scorer that needs a newly recorded field (``context_numbers``) use old runs without
    a single API call. Wall-clock time and provider errors are kept from the original.
    """
    results = load_run(run_id)[2]
    replayed = replay_results(run_id)
    bad = differing(replayed, results)
    if bad:
        return f"{run_id}: not refreshed, replay differs on {', '.join(bad)}"
    merged = []
    for first, again in zip(results, replayed, strict=True):
        if first.get("outcome") == "provider_error":
            merged.append(first)
            continue
        keep: dict[str, Any] = {
            k: first[k] for k in ("id", "tier", "kind", "elapsed_s") if k in first
        }
        merged.append({**first, **again, **keep})
    path = RESULTS / run_id / "results.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in merged),
        encoding="utf-8",
        newline="\n",
    )
    return f"{run_id}: refreshed {len(merged)} results from the cassette"


def score_report(run_id: str) -> dict[str, Any]:
    manifest, golden, results = load_run(run_id)
    scores = scoring.score_run(golden, results, aliases())
    return {
        "run_id": run_id,
        "complete": len(results) >= len(golden),
        "db_unchanged": manifest.get("db_sha256_after", manifest["db_sha256"])
        == manifest["db_sha256"],
        "scorer_sha256": code_sha256(SCORER),
        "metrics": stats.summarize(scores),
        "cost": stats.cost(results),
        "scores": [s.to_dict() for s in scores],
    }


def replay_results(run_id: str) -> list[dict[str, Any]]:
    """Re-ask every answered question from the run's cassette alone; no network is possible."""
    manifest, golden, results = load_run(run_id)

    class NoNetwork:
        def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> Any:
            raise AssertionError("replay tried to reach the network")

    cassette = Cassette(CASSETTES / f"{run_id}.jsonl", "replay", NoNetwork())
    model = make_model(manifest["provider"], manifest["model"], cassette)
    setattr(model, "key", "replay")  # noqa: B010 -- no .env needed; replay mode never posts
    agent = build_agent(
        model,
        manifest["config"],
        max_turns=manifest["max_turns"],
        max_output_tokens=manifest["max_output_tokens"],
    )
    by_id = {q["id"]: q for q in golden}
    replayed = []
    for r in results:
        try:
            again = agent.ask(by_id[r["id"]]["question"]).to_dict()
        except CassetteMiss:  # a run-time provider error is not in the cassette either
            again = {"outcome": "provider_error", "error": "not in cassette"}
        replayed.append({"id": r["id"], **again})
    return replayed


def score_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score Courtside runs.")
    p.add_argument("runs", nargs="*", help="run ids under eval/results/")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="exact McNemar")
    p.add_argument("--verify-replay", action="store_true", help="re-run from cassette, compare")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="re-derive results from the cassette (offline) to add fields newer code records",
    )
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    status = 0
    for run_id in args.runs:
        if args.refresh:
            print(refresh_results(run_id))
        report = score_report(run_id)
        write_json(RESULTS / run_id / "score.json", report)
        m = report["metrics"]

        def pct(r: dict[str, Any]) -> str:
            lo, hi = r["ci95"]
            return f"{r['k']}/{r['n']} [{lo:.0%}, {hi:.0%}]"

        print(f"\n{run_id}{'' if report['complete'] else '  (INCOMPLETE)'}")
        print(f"  overall      {pct(m['overall'])}")
        print(f"  execution    {pct(m['execution'])}   (lookup + aggregate + join)")
        for tier, r in m["by_tier"].items():
            print(f"    {tier:<12} {pct(r)}")
        print(f"  refusal      {pct(m['refusal'])}   false refusals {pct(m['false_refusal'])}")
        print(f"  faithfulness {pct(m['faithfulness'])}   leaks {m['leaks']}")
        print(f"  db unchanged {report['db_unchanged']}   labels {m['labels']}")
        c = report["cost"]
        print(
            f"  tokens in {c['input_tokens']} out {c['output_tokens']}, "
            f"{c['mean_turns']} turns/q, {c['mean_seconds']} s/q"
        )
        if args.verify_replay:
            golden = load_run(run_id)[1]
            replayed = replay_results(run_id)
            differ = differing(replayed, load_run(run_id)[2])
            same_metrics = stats.summarize(scoring.score_run(golden, replayed, aliases())) == m
            same = same_metrics and not differ
            print(
                f"  replay from cassette: metrics identical {same_metrics}, "
                f"questions that differ {differ or 'none'}"
            )
            status = status or (0 if same else 1)
    if args.compare:
        partial = [r for r in args.compare if len(load_run(r)[2]) < len(load_run(r)[1])]
        if partial:
            # Unasked questions score as wrong, so a partial run would "lose" every one of them.
            print(f"\nMcNemar skipped: {', '.join(partial)} is incomplete; finish it first.")
            return status or 1
        a, b = (scoring.score_run(*load_run(r)[1:], aliases()) for r in args.compare)
        result = stats.mcnemar(a, b)
        print(
            f"\nMcNemar {args.compare[0]} vs {args.compare[1]}: n={result['n']}, "
            f"only A right {result['a_only']}, only B right {result['b_only']}, "
            f"exact p = {result['p_exact']}"
        )
    return status
