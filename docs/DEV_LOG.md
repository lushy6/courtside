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

## Change 3: define `refused` as any cannot-answer reply (bd145e2), RUN INCOMPLETE

Target U06 (correct decline in the text, `refused` left false). The `refused` field description in the
final_answer tool now says: true if the data cannot or must not answer (another season, a prediction, an
entity or statistic not in the data, a write), and must be true whenever answer_text says the answer is not
available. A prompt change, so a fresh run: `dev-D-flash-lite-p6c3`.

**Status: 25/48 asked, paused.** The first pause was false (a per-minute 429 whose details also list per-day
quotas; fixed in 12c1080, which now also needs a retry hint over 5 minutes). The second is real: flash-lite's
free tier is 500 requests/day, not the ~1000 assumed. Resume after the quota resets (midnight Pacific) with
the same command plus `--allow-code-change`; the 25 finished questions replay from the cassette for free.
The code exception is transport only: after the pre-push review, 429s are classified from Google's
QuotaFailure violations (and Groq's message) in llm.py, which moves the code fingerprint; request bodies,
prompt and semantic hashes are unchanged, and all 25 recorded questions replay identically. **No result yet**: do not
quote the McNemar printed on the partial run (p = 1.5e-05 was the unasked questions scoring as wrong;
`--compare` now refuses partial runs). Judge it on the refusal tier and the false-refusal rate (0/40 so far).

## Candidates for the next changes (not done yet)

- Lineup ranking hybrids (T04): **rejected as a change.** Ranking by `shrunk` within `poss >= 100` is a
  defensible method; it is just not one of the two locked accepted readings. Rewording the rule to push the
  model onto exactly those readings would be teaching to the benchmark. Recorded instead as a limitation of
  the golden set: trap questions accept two methods, and a third defensible one scores as wrong. Report it.
- Run-to-run noise is now the dominant term (fresh runs move 1-3 questions either way). Before freezing, run
  the final dev config twice to see the spread, then stop iterating.
