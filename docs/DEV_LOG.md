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

## Candidates for the next changes (not done yet)

- Declines without the flag (U06): the prompt says to refuse "with refused=true"; consider saying that a
  question about an entity or statistic the data does not contain is a refusal, so the flag is set. Measure on
  the refusal tier only; watch false refusals (0/40 so far).
- Lineup ranking hybrids (T04): the rank_lineups_honestly rule offers two methods; the model mixed them. Saying
  "choose ONE: rank all lineups by shrunk, or filter poss >= 100 and rank by net_rating" is a wording fix, but
  it is close to teaching the golden set's accepted readings. Only do it if it is the honest statement of the
  rule (it is: mixing a filter with shrinkage double-penalises small samples).
- Run-to-run noise is now the dominant term (fresh runs move 1-3 questions either way). Before freezing, run
  the final dev config twice to see the spread, then stop iterating.
