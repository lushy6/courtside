"""Export one recorded dev run as the demo's data: every question's turns, answer, rows and score.

    python scripts/export_demo.py dev-D-flash-lite-p6c3

The demo replays recorded answers; it never calls a model. The run is re-asked offline from its
cassette through a recording wrapper, which captures the whole conversation (results.jsonl only
keeps tool names). Replay needs the code the run was recorded with, so a run whose code hash
differs from the current code is refused. Only the dev split can be exported: the frozen test
split must not be published before phase 7.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from typing import Any

from courtside import runner
from courtside.cassette import Cassette, CassetteMiss
from courtside.llm import Completion, Message, ToolSpec, make_model

OUT = runner.ROOT / "app" / "public" / "demo.json"
PREVIEW_ROWS = 8
NOTES_CHARS = 1600  # describe_table notes, trimmed for the page
TOOL_RESULT = re.compile(r"<tool_result>\n(.*)\n</tool_result>", re.DOTALL)


class Recorder:
    """Passes calls through and keeps the last conversation and completion."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.messages: list[Message] = []
        self.last: Completion | None = None

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool = False,
        force_tool: str | None = None,
        max_output_tokens: int = 2048,
    ) -> Completion:
        self.messages = list(messages)
        completion: Completion = self.inner.complete(
            messages,
            tools,
            require_tool=require_tool,
            force_tool=force_tool,
            max_output_tokens=max_output_tokens,
        )
        self.last = completion
        return completion


def tool_output(text: str) -> dict[str, Any]:
    match = TOOL_RESULT.search(text)
    try:
        data: dict[str, Any] = json.loads(match.group(1)) if match else {}
    except json.JSONDecodeError:
        data = {}
    return data


def turns(recorder: Recorder) -> list[dict[str, Any]]:
    """Each tool call the model made, with what it was shown back (rows capped for the page)."""
    transcript = list(recorder.messages)
    if recorder.last is not None:
        transcript.append(recorder.last.message)
    steps = []
    for i, m in enumerate(transcript):
        if m.role != "assistant":
            continue
        # Pair by position, not id: Gemini numbers calls per turn, so every turn has a call_0.
        replies = []
        for later in transcript[i + 1 :]:
            if later.role != "tool":
                break
            replies.append(later)
        shown = {r.tool_call_id: r.text for r in replies}
        for call in m.tool_calls:
            if call.name == "final_answer":
                continue
            out = tool_output(shown.get(call.id, ""))
            rows = out.get("rows") or []
            steps.append(
                {
                    "tool": call.name,
                    "args": call.arguments,
                    "ok": bool(out.get("ok")),
                    "error": (out.get("error") or {}).get("message", ""),
                    "columns": out.get("columns") or [],
                    "rows": rows[:PREVIEW_ROWS],
                    "row_count": len(rows) + int(out.get("rows_not_shown", 0) or 0),
                    "text": str(out.get("text") or "")[:NOTES_CHARS],
                }
            )
    return steps


def expected(gold: dict[str, Any]) -> list[dict[str, Any]]:
    answers = [gold["expected"]] if gold.get("expected") else list(gold.get("accepted") or [])
    return [{"columns": a["columns"], "rows": a["rows"][:3]} for a in answers]


def export(run_id: str) -> dict[str, Any]:
    manifest, golden, results = runner.load_run(run_id)
    if manifest["split"] != "dev":
        raise SystemExit("only dev runs may be published; the test split is frozen until phase 7")
    if manifest.get("code_sha256") != runner.code_sha256():
        # Not fatal by itself: a transport-only change (e.g. 429 handling in llm.py) moves the hash
        # without changing a single request. The replay below is the real check: any request the
        # cassette lacks raises CassetteMiss, and any differing SQL or answer aborts the export.
        print(
            f"note: {run_id} was recorded with other code (code_sha256 differs); "
            "exporting only if every question replays identically"
        )
    cassette = Cassette(runner.CASSETTES / f"{run_id}.jsonl", "replay")
    model = make_model(manifest["provider"], manifest["model"], cassette)
    setattr(model, "key", "replay")  # noqa: B010 -- replay mode never posts
    recorder = Recorder(model)
    agent = runner.build_agent(
        recorder,
        manifest["config"],
        max_turns=manifest["max_turns"],
        max_output_tokens=manifest["max_output_tokens"],
    )
    report = runner.score_report(run_id)
    scores = {s["id"]: s for s in report["scores"]}
    by_id = {r["id"]: r for r in results}
    questions = []
    for gold in golden:
        recorded = by_id[gold["id"]]
        if recorded["outcome"] == "provider_error":
            recorder.messages, recorder.last = [], None  # never recorded: no turns to show
        else:
            try:
                replayed = agent.ask(gold["question"])
            except CassetteMiss:
                raise SystemExit(
                    f"{gold['id']}: this code sends a request the cassette never recorded (the "
                    "prompt or agent changed since the run). Replay it from a worktree at the "
                    "run's git_sha; see docs/HANDOFF.md."
                ) from None
            if replayed.sql != recorded["sql"] or replayed.answer_text != recorded["answer_text"]:
                raise SystemExit(f"{gold['id']}: replay differs from the recorded run")
        score = scores[gold["id"]]
        questions.append(
            {
                "id": gold["id"],
                "tier": gold["tier"],
                "question": gold["question"],
                "turns": turns(recorder),
                "answer": recorded["answer_text"],
                "caveats": recorded["caveats"],
                "refused": recorded["refused"],
                "refusal_reason": recorded["refusal_reason"],
                "sql": recorded["sql"],
                "sql_error": recorded["sql_error"],
                "columns": recorded["result_columns"],
                "rows": recorded["result_rows"][:50],
                "row_count": len(recorded["result_rows"]),
                "correct": score["correct"],
                "label": score["label"],
                "faithful": score["faithful"],
                "unsupported_numbers": score["unsupported_numbers"],
                "expected": expected(gold),
                "turn_count": recorded["turns"],
                "tokens": recorded["input_tokens"] + recorded["output_tokens"],
                "seconds": round(recorded["elapsed_s"], 1),
            }
        )
    metrics = report["metrics"]
    return {
        "run_id": run_id,
        "provider": manifest["provider"],
        "model": manifest["model"],
        "config": manifest["config"],
        "git_sha": manifest["git_sha"],
        "recorded_at": manifest["created_at"],
        "overall": metrics["overall"],
        "questions": questions,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = export(argv[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    count = len(data["questions"])
    print(f"wrote {OUT.relative_to(runner.ROOT)}: {count} questions from {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
