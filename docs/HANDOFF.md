# Handoff: Courtside after phase 4 (2026-10-26)

Paste this into a new Claude Code session to continue:

> Continue Courtside in C:\Users\birbu\Projects\Sports\_Projects\Shared\courtside. Read docs/HANDOFF.md and
> docs/PLAN.md first, then do the "Next" list in HANDOFF.md, starting with phase 5. Dev split only.

## Phase 4: done (committed 7dd119b)

- `src/courtside/llm.py`: provider-agnostic `Message` / `ToolSpec` / `Completion`; `GroqModel`, `GeminiModel`;
  `HttpTransport` (explicit User-Agent, timeouts -> 408); `RetryingTransport` (429 hints, 408/5xx backoff);
  keys from `.env` (GEMINI_API_KEY, GROQ_API_KEY), only ever sent as headers.
- `src/courtside/cassette.py`: record / replay / live, keyed by sha256(url + body); headers never stored.
- `src/courtside/agent.py`: configs A-D (PLAN 6.3), one `final_answer` tool for every provider and config, the
  final SQL is executed through the three safety layers, tool results delimited as data, entity hints in D only.
- `src/courtside/ask.py`: `.venv/Scripts/python.exe -m courtside.ask --provider groq --model qwen/qwen3.8-27b
  --config D --cassette eval/cassettes/smoke/x.jsonl "question"` (add `--replay` to run offline).
- Tests: `tests/test_llm.py`, `tests/test_agent.py` (offline; fake transport / fake model over the real DB).
- Done-when met: Groq `qwen/qwen3.8-27b`, config D, dev question L06 -> 71 (gold), 2 turns, 6.7 s; replays from
  `eval/cassettes/smoke/groq_qwen3.8-27b_L06.jsonl` in 0.0 s.

## Findings to carry into RESULTS / README

- Groq's CDN returns HTTP 403 (code 1010) for urllib's default User-Agent.
- Free Groq limits bite on tokens per minute: gpt-oss-120b 8,000 TPM, and one agent turn is ~3.2-3.7k tokens.
- An empty `Cassette` is falsy (it has `__len__`), so `transport or default()` silently dropped it and nothing was
  recorded; fixed with `is not None`, regression test mutation-checked.
- `openai/gpt-oss-120b` (Groq) on L06: found the player, ran the right SQL, saw 71, then returned an empty `stop`
  turn; forced `final_answer` gives HTTP 400 `tool_use_failed`, non-deterministically. Not the schema; passing back
  `reasoning` did not fix it reliably. Cassette kept: `eval/cassettes/smoke/groq_gpt-oss-120b_L06_failed.jsonl`.
  This is a model / tool-format failure to REPORT, not to hide.
- Gemini: `gemini-2.5-flash` closed to new users (404); default is `gemini-3.8-flash`, which was 503 "high demand"
  for 9 minutes of retries on 2026-10-26. Not yet run end to end.

## Next

1. Retry Gemini end to end (`--provider gemini`); if 3.8-flash stays overloaded try `gemini-3.8-flash-lite` or
   `gemini-flash-lite-latest`. Echoed thought signatures are tested offline only so far.
2. Decide the two free models for the ablation: likely qwen3.8-27b (Groq) + one Gemini Flash; record gpt-oss as a
   failed candidate with its reason.
3. Phase 5: `eval/run.py` (one config over a split -> results/<run_id>.jsonl + manifest.json with git SHA, model,
   config, prompt hash, semantic sha256, DB sha256, golden hash; daily request counter; resumable via the cassette)
   and `eval/score.py` (execution accuracy per PLAN 6.2, refusal accuracy, faithfulness, Wilson CIs, McNemar).
   Done when: re-scoring a cassette reproduces identical numbers; an interrupted run resumes without repeat calls.
4. Then phase 6 (iterate on dev only) and 7 (frozen test runs). Never touch the test split before phase 7.
5. Claude adapter for 7b: load the `claude-api` skill first; hard $20 cap.
