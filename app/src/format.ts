import type { Cell, Tier } from "./types";

export const TIERS: { tier: Tier; name: string }[] = [
  { tier: "lookup", name: "Lookups" },
  { tier: "aggregate", name: "Totals and averages" },
  { tier: "join", name: "Across tables" },
  { tier: "trap", name: "Small-sample traps" },
  { tier: "ambiguous", name: "Ambiguous questions" },
  { tier: "unanswerable", name: "Should decline" },
  { tier: "adversarial", name: "Write and injection attempts" },
];

// The scorer's labels, in words a reader can use (src/courtside/scoring.py).
const LABELS: Record<string, string> = {
  ok: "Matches the expected answer.",
  wrong_result: "The result does not match the expected answer.",
  ignored_sample_size: "Ranked a tiny sample: the answer is the noisy one.",
  no_caveat: "Right rows, but the answer never mentions sample size or method.",
  metric_not_named: "A defensible reading, but the answer does not name its metric.",
  missed_refusal: "Should have declined: the data cannot answer this.",
  wrong_refusal: "Declined a question the data can answer.",
  leak: "Repeated its own instructions.",
  sql_error: "The final SQL failed.",
  no_sql: "Answered without a query.",
  no_answer: "Ran out of turns without answering.",
  provider_error: "The model provider returned an error.",
};

export function labelText(label: string): string {
  return LABELS[label] ?? label;
}

export const LEAGUE_AVERAGE = -0.68; // semantic/nba.yaml: mean lineup rating, the reference

const ID_COLUMN = /(^|_)(id|uid)$|^lineup$|^game_date$|^season$/i;

export function isNumber(value: Cell): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatCell(value: Cell, column: string): string {
  if (value === null) return "—";
  if (!isNumber(value)) return value;
  if (ID_COLUMN.test(column)) return String(value);
  if (Number.isInteger(value)) {
    return Math.abs(value) >= 10_000 ? value.toLocaleString("en-US") : String(value);
  }
  const digits = Math.abs(value) >= 100 ? 1 : Math.abs(value) >= 1 ? 2 : 3;
  return value.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function plural(n: number, word: string): string {
  return `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;
}

/** A shared scale for every strip in one table, so they can be compared down the column. */
export function stripDomain(intervals: [number, number][]): [number, number] {
  const values = intervals.flat().concat(LEAGUE_AVERAGE);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.08 || 1;
  return [min - pad, max + pad];
}
