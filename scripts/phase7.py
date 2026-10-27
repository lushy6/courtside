"""Phase 7: freeze everything a test result depends on, then run the pre-registered test runs.

    python scripts/phase7.py freeze            # write eval/FREEZE.json (once, then commit it)
    python scripts/phase7.py check             # exit 1 if anything drifted from FREEZE.json
    python scripts/phase7.py run --lane gemini # A-D x 3 reps on flash-lite
    python scripts/phase7.py run --lane groq   # qwen, config D x 3 reps

A lane runs its queue in order and is safe to stop and restart at any time: the runner skips
finished questions and replays a half-finished conversation from its cassette. On a daily-cap
pause (exit 3) the lane sleeps until the provider's quota day resets. Provider errors saved in a
completed run are re-asked (``--retry-errors``) before the lane moves on. Before every run the
lane re-checks the freeze, so nothing a result depends on can change between runs unnoticed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtside.llm import make_model  # noqa: E402
from courtside.runner import (  # noqa: E402
    CODE,
    DB,
    RESULTS,
    SCORER,
    SEMANTIC,
    build_agent,
    code_sha256,
    default_run_id,
    git_state,
    prompt_sha256,
    read_results,
    sha256_file,
)

FREEZE = ROOT / "eval" / "FREEZE.json"
LOCK = ROOT / "eval" / "golden" / "LOCK.json"
CONFIGS = ("A", "B", "C", "D")
MAX_TURNS, MAX_OUTPUT_TOKENS = 8, 2048
# Rep-major, so if the calendar runs out there is a complete rep of every config first.
LANES: dict[str, dict[str, Any]] = {
    "gemini": {
        "model": "gemini-3.5-flash-lite",
        "runs": [(c, rep) for rep in (1, 2, 3) for c in CONFIGS],
        "reset": ZoneInfo("America/Los_Angeles"),  # Google resets at midnight Pacific
        "rpm": 8,
    },
    "groq": {
        "model": "qwen/qwen3.8-27b",
        # Config D, pre-registered here before any test run (not picked after seeing A-D).
        "runs": [("D", rep) for rep in (1, 2, 3)],
        "reset": UTC,  # Groq's token cap resets at midnight UTC
        "rpm": 8,
    },
}


def identity() -> dict[str, Any]:
    """Every hash a test result depends on, computed from the working tree."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    prompts = {}
    for c in CONFIGS:
        agent = build_agent(
            make_model("gemini", transport=_NoNetwork()),
            c,
            max_turns=MAX_TURNS,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        prompts[c] = prompt_sha256(agent)
    return {
        "prompt_sha256": prompts,
        "code_sha256": code_sha256(CODE),
        "scorer_sha256": code_sha256(SCORER),
        "semantic_sha256": sha256_file(SEMANTIC),
        "db_sha256": sha256_file(DB),
        "test_sha256": lock["test_sha256"],
        "dev_sha256": lock["dev_sha256"],
        "max_turns": MAX_TURNS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "lanes": {
            name: {"model": lane["model"], "runs": [f"{c} r{rep}" for c, rep in lane["runs"]]}
            for name, lane in LANES.items()
        },
    }


class _NoNetwork:
    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("hashing a prompt never calls a provider")


def drift() -> list[str]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))["identity"]
    now = identity()
    return [k for k in frozen if frozen[k] != now.get(k)]


def freeze() -> int:
    if FREEZE.exists():
        print(f"{FREEZE.relative_to(ROOT)} exists; a freeze is written once")
        return 1
    sha, dirty = git_state()
    if dirty:
        print("commit first: the freeze records a clean git SHA")
        return 1
    record = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "identity": identity(),
    }
    FREEZE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {FREEZE.relative_to(ROOT)} at {sha[:7]}")
    return 0


def check() -> int:
    changed = drift()
    print("freeze intact" if not changed else f"DRIFT from FREEZE.json: {', '.join(changed)}")
    return 1 if changed else 0


def seconds_to_reset(zone: Any, now: datetime | None = None) -> float:
    """Seconds until 3 minutes past the next midnight in the provider's quota zone."""
    local = (now or datetime.now(UTC)).astimezone(zone)
    nxt = (local + timedelta(days=1)).replace(hour=0, minute=3, second=0, microsecond=0)
    return (nxt - local).total_seconds()


def log(lane: str, text: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {lane}: {text}", flush=True)


def provider_errors(run_id: str) -> int:
    path = RESULTS / run_id / "results.jsonl"
    return sum(r.get("outcome") == "provider_error" for r in read_results(path))


def run_lane(name: str, *, max_failures: int = 5) -> int:
    lane = LANES[name]
    model = lane["model"]
    for config, rep in lane["runs"]:
        run_id = default_run_id("test", config, name, model, rep)
        failures = retries = 0
        retry_errors = False
        while True:
            changed = drift()
            if changed:
                log(name, f"stopping before {run_id}: drift from FREEZE.json in {changed}")
                return 1
            cmd = [
                sys.executable, str(ROOT / "eval" / "run.py"),
                "--provider", name, "--model", model, "--config", config,
                "--split", "test", "--frozen-test", "--rep", str(rep),
                "--rpm", str(lane["rpm"]), "--daily-limit", "480",
            ]  # fmt: skip
            if retry_errors:
                cmd.append("--retry-errors")
            log(name, f"{run_id}: start{' (retry errors)' if retry_errors else ''}")
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code == 3:
                wait = seconds_to_reset(lane["reset"])
                log(name, f"{run_id}: paused on the daily cap, sleeping {wait / 3600:.1f} h")
                time.sleep(wait)
                continue
            if code != 0:
                failures += 1
                log(name, f"{run_id}: exit {code} ({failures}/{max_failures})")
                if failures >= max_failures:
                    log(name, f"{run_id}: giving up; fix the cause and restart the lane")
                    return code
                time.sleep(600)
                continue
            errors = provider_errors(run_id)
            if errors and retries < 3:
                retries += 1
                retry_errors = True
                log(name, f"{run_id}: {errors} provider errors, re-asking ({retries}/3)")
                continue
            log(name, f"{run_id}: complete ({errors} provider errors left)")
            break
    log(name, "lane complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--lane", required=True, choices=sorted(LANES))
    args = p.parse_args(argv)
    if args.cmd == "freeze":
        return freeze()
    if args.cmd == "check":
        return check()
    if not FREEZE.exists():
        print("no FREEZE.json: run `freeze` and commit it first")
        return 1
    return run_lane(args.lane)


if __name__ == "__main__":
    raise SystemExit(main())
