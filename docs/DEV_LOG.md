# Phase 6 dev log

One entry per change, with its dev-split result. Dev only: the test split is untouched until phase 7, and the
golden set is locked. Model: Gemini `gemini-3.5-flash-lite`, config D, unless stated. Numbers are dev numbers for
choosing changes; they are not results and are not quoted outside this file.

How a change is measured:

- A change that leaves earlier requests byte-identical (an agent change that only affects later turns) is run
  with `--base-cassette <previous run>`: unchanged conversations replay exactly and only the altered ones call
  the API. That is a controlled counterfactual: same sampled trajectories, one difference.
- A change to the prompt (semantic layer, system text) alters every request, so it is a fresh run, and its
  difference from the previous run mixes the change with run-to-run sampling noise. Read those with that in mind.
- Each comparison is an exact McNemar on the paired per-question outcomes (`eval/score.py --compare A B`).
  With n = 48, a handful of flips cannot reach significance; say so instead of claiming an effect.

## Baseline: `dev-D-gemini-gemini-3.5-flash-lite-r1` (commit 595d4ee code, scorer at ae1c3aa)

39/48 overall; lookup + aggregate + join 26/28; trap 6/8; refusal 6/8; faithfulness 35/36; 0 leaks.
Failures: 6 x `max_turns` (still querying after 8 turns: J25, T01, T02, M04, U06, U08), A24 and J25-type
double counting of traded players' contracts, M04/M08 picked their own minimum-attempt thresholds,
M07 named its metric in words not on the locked `must_mention` list.

## Change 1 (139e79d): force `final_answer` on the last allowed turn

On the last turn a loop config gets "This is your last turn: call final_answer now with your best answer from
what you have seen." and `final_answer` is forced.

Run `dev-D-flash-lite-p6c1`, controlled (`--base-cassette` r1, 6 new API calls, 193 replayed).
**39 -> 43/48.** All 6 stuck questions now answer: T01, T02, U06, U08 correctly; J25 and M04 wrongly (J25 is the
traded-player double count, M04 a 300-FGA minimum vs the gold's 500). McNemar vs r1: 4 vs 0 discordant,
exact p = 0.125. Direction is clear, significance is not reachable with 4 flips. **Kept.**

## Change 2 (b0ba360): contracts rule covers league-wide aggregates

A24 and J25 averaged / summed raw contract rows although the rule warned about the 6 traded players; it read as
"never SUM salary per player". It now says any total, average or count over players not split by team counts
each player once (`SELECT DISTINCT player_id, salary, ...`). Following it reproduces both gold answers.

Run `dev-D-flash-lite-p6c2`, fresh (the prompt changed; 248 requests). **43 -> 42/48**; McNemar vs p6c1:
2 vs 1 discordant, exact p = 1.0, i.e. noise. Per question:

- A24 fixed (its target): `AVG(salary) FROM (SELECT DISTINCT player_id, salary FROM contracts)`.
- J25 still wrong: the answer itself says "70 unique players, 2 appear twice" and then sums the raw rows.
- T04 newly wrong, unrelated: filtered `poss >= 100` AND ranked by `shrunk`, a hybrid that is not one of the
  locked accepted readings. Fresh sampling, not the change.
- U06 newly wrong, unrelated: the text correctly says Haliburton is not in the data, but `refused` stayed
  false. The scorer reads the structured flag on purpose (deterministic); a correct decline without the flag
  fails.

**Kept**: it fixes its target mechanically; the other flips are sampling noise on unrelated questions.

## Change 3 (bd145e2): define `refused` as any cannot-answer reply

Target U06 (correct decline in the text, `refused` left false). The `refused` field description in the
final_answer tool now says: true if the data cannot or must not answer (another season, a prediction, an
entity or statistic not in the data, a write), and must be true whenever answer_text says the answer is not
available. A prompt change, so a fresh run: `dev-D-flash-lite-p6c3`.

Run history: it took two quota days. The first pause was false (a per-minute 429 whose details also list
per-day quotas; fixed in 12c1080, then replaced by QuotaFailure classification in llm.py). The second was
real: flash-lite's free tier is 500 requests/day. The 23 remaining questions ran on 2026-10-27 after the
Pacific-midnight reset, with `--allow-code-change` (transport only: request bodies, prompt and semantic hashes
unchanged). `--verify-replay`: metrics and every question identical.

**42 -> 45/48** [83, 98] (44/48 before the scorer rule below); refusal **8/8** (was 7/8), false refusals
**0/40**, trap 8/8, faithfulness 38/40, 0 leaks. McNemar vs p6c2: 0 vs 3 discordant, exact p = 0.25, not
significant. (Vs the r1 baseline: 0 vs 6, p = 0.031, but that spans three changes plus fresh sampling, so it
is dev evidence for the direction, not a result.) Per question, hand-checked:

- U06 fixed (its target): "Tyrese Haliburton is not found in the player directory", `refused` true.
- T04 now right and M04 now right (500-FGA minimum, matching the gold): fresh sampling, not the change.
- T02 first scored "wrong": **a scorer false negative, not a model error**. The model ranked by `shrunk` and
  returned exactly the accepted lineup (Brown, White, Queta, Pritchard, Hauser; shrunk 7.236, 387 poss), and
  the answer also gives the other accepted reading. But its identifying columns were the bare `lineup` (player
  ids without the `2025-26|1610612738|` prefix of `lineup_uid`) and a GROUP_CONCAT of full names, while the
  gold has `lineup_uid` and surnames. **Scorer rule added (user approved 2026-10-27):** a `lineup_uid` folds
  to its bare lineup id on both sides (the bare id is unique in the DB: 17,733 rows, 17,733 distinct). Tested
  both ways (a different or four-player lineup still fails; only a whole uid is rewritten). Rescored every dev
  run: only p6c3 changes (T02, 44 -> 45); r1, p6c1, p6c2 and the qwen / flash runs keep their counts.
- Still wrong: J25 (sums raw contract rows; the answer text even states the correct de-duplicated total),
  M07 (metric named in words not on the locked list), M08 (100-3PA minimum vs the gold's reading).

**Kept**: it fixes its target with no false refusals, and the other flips are noise or the scorer.

## Candidates for the next changes (not done yet)

- Lineup ranking hybrids (T04): **rejected as a change.** Ranking by `shrunk` within `poss >= 100` is a
  defensible method; it is just not one of the two locked accepted readings. Rewording the rule to push the
  model onto exactly those readings would be teaching to the benchmark. Recorded instead as a limitation of
  the golden set: trap questions accept two methods, and a third defensible one scores as wrong. Report it.
- Run-to-run noise is now the dominant term (fresh runs move 1-3 questions either way). Before freezing, run
  the final dev config twice to see the spread, then stop iterating.
