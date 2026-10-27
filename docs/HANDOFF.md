# Handoff: Courtside after phase 5 (2026-10-26)

Paste this into a new Claude Code session to continue:

> Continue Courtside in C:\Users\birbu\Projects\Sports\_Projects\Shared\courtside. Read docs/HANDOFF.md and
> docs/PLAN.md first, then do the "Next" list in HANDOFF.md, starting with phase 6. Dev split only.

## Phase 5: done (595d4ee .. this commit, pushed to origin/main)

- `eval/run.py` (`courtside.runner`): one config over a split -> `eval/results/<run_id>/{results.jsonl,
  manifest.json}`, provider calls in `eval/cassettes/<run_id>.jsonl`. Resumable: finished questions are skipped, a
  question cut off mid-conversation replays its recorded turns. The manifest pins git SHA, model, config,
  prompt / semantic / DB / golden / code hashes; a resume refuses if any changed (`--allow-code-change` only for
  code changes that keep requests identical, logged per session). `--split test` needs `--frozen-test`.
- Daily caps: our own counter per provider and UTC day (`--daily-limit`, `eval/results/_usage/`, gitignored), and
  a provider 429 that names a per-day limit (Groq TPD, Gemini RPD) both PAUSE the run (exit 3). Re-run tomorrow;
  add `--retry-errors` to re-ask questions saved as provider errors.
- `eval/score.py RUN [RUN..]`: per-kind scoring (rules in `scoring.py` docstring), Wilson 95% CIs, faithfulness,
  labels; `--compare A B` = exact McNemar; `--verify-replay` re-runs the whole run from the cassette offline and
  checks metrics AND each question's outcome/turns/tools/SQL/answer/rows; `--refresh` re-derives results from the
  cassette (adds fields newer code records) and refuses if the replay is a different conversation.
  `score.json` carries `scorer_sha256`.
- Done-when met: `--verify-replay` reproduced identical metrics and per-question outputs on three real runs;
  an interrupted run resumes with zero repeated calls (tested: 6 calls for 6 turns).
- Review: independent ECC code review of 595d4ee, 3 high + 4 medium findings, all reproduced, fixed, regression
  tested (number regex backtracking, "poss" matching "possible", 1% rounding bound, tie-safe row matching,
  only Groq `tool_use_failed` 400s recorded, code hash in the run identity, stronger replay check).

## Scorer changes made on the DEV split from hand-checked answers (before any test run)

Every "wrong" answer in the first runs was read by hand. Representation, not correctness, was fixed in the scorer;
each rule is narrow and tested both ways. These are the only scorer changes after seeing model output:

- strings case/accent-folded; team abbr / name / unique city / city+name are one entity (J02 "Timberwolves")
- adjacent text columns also count joined (A09/A20 first and last name in two columns)
- a result with ORDER BY may answer with its leading rows (A09, J01 top-5 lists); never without ORDER BY
- a percent answers a gold fraction in (0, 1] (A16 55.4 for 0.554)
- the gold's own rounding is its stated precision; only a model rounding coarser must stay within 1% (T09)
- a 1x1 count gold matches a result with exactly that many rows if the answer states the number (T07)
- faithfulness grounds numbers in everything the model was shown (prompt, question, every tool result,
  recorded by the agent as `context_numbers`), not only the final rows. Arithmetic the model did itself
  (a difference "$12.5M") is still flagged: known, accepted strictness.
- Known strictness kept: `must_mention` terms are locked with the golden set; M07 named its metric
  ("points per 100 possessions") in words not on the list and fails.

## Dev results so far (config D, dev split, scorer as of this commit)

| Run | Overall | Lookup+agg+join | Trap | Refusal | Faithful | Notes |
|---|---|---|---|---|---|---|
| dev-D-gemini-gemini-3.5-flash-lite-r1 | 39/48 [68, 90] | 26/28 | 6/8 | 6/8 | 35/36 | 6 x max_turns, no final_answer |
| dev-D-groq-qwen3.8-27b-r1 | 19/48 | 19/28 | 0/8 | 0/8 | 19/20 | 28 provider errors = Groq daily token cap; 19/20 of answered correct |

Dev numbers are for iterating, not for the README or resume.

## Model decision (made 2026-10-26, user: "your call, go with A")

- **Ablation A-D runs on `gemini-3.5-flash-lite` (Gemini, pinned ID).** 248 requests in one day with no 429;
  its real daily cap is still unknown (the 429 will name it; the run pauses on it).
- **Second free model: Groq `qwen/qwen3.8-27b`, best config only.** Free cap is 200,000 TOKENS per day; a config
  C/D question costs ~9.1k tokens (2.8k system prompt resent every turn), so ~20 questions/day; a 72-question
  test run takes ~3-4 days. Run it in parallel with the Gemini runs.
- Failed candidates (report them): Groq `openai/gpt-oss-120b` (empty stop turn, then HTTP 400
  `tool_use_failed`) and `openai/gpt-oss-20b` (HTTP 400 `output_parse_failed` on turn 2); cassettes in
  `eval/cassettes/smoke/*_failed.jsonl`. Gemini `3.8-flash` and `3.7-flash`: HTTP 503 "high demand" for 15+ min on
  2026-10-26. `gemini-3.5-flash`: free tier is 20 requests/day (quota is billed as `gemini-3.6-flash`); its
  partial run `dev-D-gemini-gemini-3.5-flash-r1` stopped after 4 questions and is kept as evidence only.
- Expected phase 7 calendar: ~3-5 days at $0 (4 configs x 3 reps on flash-lite if its cap allows ~900
  requests/day; qwen best config in parallel).

## Findings to carry into RESULTS / README

- Groq's CDN returns HTTP 403 (code 1010) for urllib's default User-Agent.
- Free-tier limits are the schedule: Groq qwen 200k tokens/day; Gemini 3.5-flash 20 requests/day.
- An empty `Cassette` is falsy (`__len__`), so `transport or default()` silently dropped it (fixed, tested).
- gpt-oss-120b / -20b tool-format failures above. Model / tool-format failures to REPORT, not hide.
- flash-lite's dominant dev failure: 6 of 48 questions hit `max_turns` (8) still querying, never answering.

## Next

1. **Phase 6, iterate on dev only.** First lever, from the failure list: force `final_answer` on the last allowed
   turn (and maybe say how many turns remain), so a model that keeps exploring still answers. Then re-run dev D
   on flash-lite as a new run id (`--rep 2` or a descriptive `--run-id`), score, `--compare` with r1 (McNemar).
   Log each change and its dev result. Never tune on the test split; do not edit the golden set (locked).
2. Resume the Groq dev run when its token cap resets (Groq resets on a rolling window; try after 2026-10-27 03:00
   UTC): `python eval/run.py --provider groq --model qwen/qwen3.8-27b --config D --split dev --retry-errors
   --allow-code-change`, then `python eval/score.py dev-D-groq-qwen3.8-27b-r1 --refresh --verify-replay`.
   (`--allow-code-change`: agent.py now records `context_numbers`; requests are identical.)
3. Dev runs of configs A, B, C on flash-lite to see the ablation shape before freezing (still dev only).
4. Phase 7: freeze prompt + scorer (record SHAs), then `--split test --frozen-test`, 4 configs x 3 reps on
   flash-lite, best config x 3 on qwen; RESULTS.md with Wilson CIs, McNemar, failure taxonomy.
5. Claude adapter for 7b: load the `claude-api` skill first; hard $20 cap.
6. Optional: a lock around the usage counter if two runs ever share a provider (today they never do).
7. Cassettes are ~5 MB per dev D run and are committed; phase 7 adds ~15 runs. Consider gzip (`.jsonl.gz`,
   transparent in `Cassette`) before then to keep the public repo small.
