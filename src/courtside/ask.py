"""Ask one question from the command line.

    python -m courtside.ask --provider groq --config D "How many games did Jaylen Brown play in?"
    python -m courtside.ask --provider gemini --cassette runs/try.jsonl "..."  # replays for free

Prints the answer, the SQL, its result and token usage. Keys come from ``.env`` (never printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from courtside.agent import CONFIGS, Agent
from courtside.cassette import Cassette
from courtside.db import open_readonly
from courtside.llm import RetryingTransport, Transport, make_model
from courtside.semantic import SemanticLayer

DB = Path(__file__).resolve().parents[2] / "data" / "courtside_nba.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Courtside agent one question.")
    parser.add_argument("question")
    parser.add_argument("--provider", default="groq", choices=["groq", "gemini"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default="D", choices=sorted(CONFIGS))
    parser.add_argument("--cassette", default=None, help="JSONL file to record to / replay from")
    parser.add_argument("--replay", action="store_true", help="fail instead of calling the API")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )  # model text is not limited to the console code page

    transport: Transport = RetryingTransport()
    if args.cassette:
        transport = Cassette(args.cassette, "replay" if args.replay else "record", transport)
    model = make_model(args.provider, args.model, transport)
    con = open_readonly(DB)
    agent = Agent(model, con, SemanticLayer.load(), CONFIGS[args.config])
    result = agent.ask(args.question)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(
            f"[{result.provider}/{result.model}, config {result.config}] outcome: {result.outcome}"
        )
        if result.error:
            print(f"error: {result.error}")
        if result.refused:
            print(f"REFUSED: {result.refusal_reason}")
        print(f"answer: {result.answer_text}")
        if result.caveats:
            print("caveats: " + " | ".join(result.caveats))
        print(f"sql: {result.sql}")
        if result.sql_error:
            print(f"sql error: {result.sql_error}")
        elif result.result_columns:
            print(
                "result: "
                + json.dumps([result.result_columns, *result.result_rows[:10]], default=str)
            )
        print(
            f"turns {result.turns}, tools {result.tool_calls}, tokens in {result.input_tokens} "
            f"(cached {result.cached_input_tokens}) out {result.output_tokens}, "
            f"{result.elapsed_s:.1f} s"
        )
    return 0 if result.outcome == "answered" else 1


if __name__ == "__main__":
    sys.exit(main())
