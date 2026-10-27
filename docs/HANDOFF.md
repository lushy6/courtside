# Handoff: Courtside phase 4 (LLM adapter + agent), in progress 2026-10-26

Paste this into a new Claude Code session to continue:

> Continue Courtside phase 4 in C:\Users\birbu\Projects\Sports\_Projects\Shared\courtside. Read docs/HANDOFF.md
> and docs/PLAN.md first. Finish the open items in HANDOFF.md, run ruff/mypy/pytest, commit, and update my memory notes.

## State (uncommitted; nothing from phase 4 is committed yet)

New files: `src/courtside/llm.py` (Message/ToolSpec/Completion types, HttpTransport, RetryingTransport,
GroqModel, GeminiModel, make_model, .env loading), `src/courtside/cassette.py` (record/replay/live, key =
sha256(url+body), never stores headers), `src/courtside/agent.py` (configs A-D, final_answer tool, loop,
entity hints, executes the final SQL through run_query), `src/courtside/ask.py` (CLI:
`.venv/Scripts/python.exe -m courtside.ask --provider gemini|groq --config A-D --cassette runs/x.jsonl "question"`).
ruff + ruff format + mypy strict are clean as of the last check. **No tests written yet for these modules.**

Keys: `.env` has GEMINI_API_KEY and GROQ_API_KEY (verified working; never print them). Smoke cassettes in `eval/cassettes/smoke/` (committed;
headers are never stored, so no secrets).

## Findings so far (keep; they go in the README / RESULTS later)

- Groq's CDN returns HTTP 403 code 1010 for urllib's default User-Agent -> HttpTransport sends `courtside/0.1`.
- Free Groq `openai/gpt-oss-120b` cap is **8,000 tokens/minute**; one agent turn is ~3.2-3.6k tokens, so ~2 turns/min.
  RetryingTransport parses "try again in 15.7s" / Gemini "retryDelay" and waits; also retries 500/502/503/504.
- Bug found + fixed: `Cassette` defines `__len__`, so an EMPTY cassette is falsy and `transport or default()` silently
  dropped it (nothing recorded). Now `is not None` checks. **Add a regression test.**
- gpt-oss-120b on Groq (dev question L06 "How many games did Jaylen Brown play in?", gold 71): it resolved the player,
  ran the right SQL and saw 71, then returned an EMPTY `stop` turn instead of calling final_answer; with
  tool_choice required/forced it gets HTTP 400 `tool_use_failed` (non-deterministic: once it produced text).
  Not the schema (all-string schema fails too). Passing back `reasoning` did not fix it reliably. Handled in code as an
  empty turn -> nudge + forced final_answer, but it still hits max_turns. Next: try `qwen/qwen3.8-27b` on Groq, or
  accept a plain-text final answer as fallback (parse sql from the last successful run_sql) and REPORT it as a
  model-format failure, not silently.
- Gemini: `gemini-2.5-flash` is closed to new users (404) -> default is now `gemini-3.8-flash`. Got a 503 "high
  demand" once and a 120 s read timeout once.

## Open items, in order

1. DONE: read timeouts map to ProviderError(status=408), retried as transient.
2. DONE: Groq qwen/qwen3.8-27b answered L06 correctly (71) in 2 turns, cassette eval/cassettes/smoke/groq_qwen3.8-27b_L06.jsonl replays offline. gemini-3.8-flash was 503 for 9 min; retry later.
3. Tests (no network): adapter request/response fixtures for Groq + Gemini (tool calls, tool results, Gemini
   consecutive functionResponse merge, raw parts echoed back, usage parsing, stop reasons, tool_use_failed mapping),
   cassette (record -> replay identical, replay miss raises, headers/key never written, empty cassette not dropped),
   RetryingTransport with a fake sleep (429 hint, 503 backoff, gives up past max_wait), agent with a scripted fake
   model (loop config C/D: run_sql then final_answer; single-shot A/B only gets final_answer; refusal; max_turns;
   empty-turn -> forced final_answer; tool results delimited as data; entity hints only in D).
4. Commit phase 4 to main (solo private repo; commit to main, see memory). Update docs/PLAN.md status and memory
   `courtside_project.md`.
5. Phase 5 (eval harness run.py / score.py) only after that.

Rules: never tune or run on the test split before phase 7; dev split only. Report null results plainly.
