// The shape written by scripts/export_demo.py. Values are whatever SQLite returned.
export type Cell = string | number | null;

export interface Turn {
  tool: "run_sql" | "describe_table" | "sample_values";
  args: Record<string, string>;
  ok: boolean;
  error: string;
  columns: string[];
  rows: Cell[][];
  row_count: number;
  text: string;
}

export interface Expected {
  columns: string[];
  rows: Cell[][];
}

export type Tier =
  | "lookup"
  | "aggregate"
  | "join"
  | "trap"
  | "ambiguous"
  | "unanswerable"
  | "adversarial";

export interface Question {
  id: string;
  tier: Tier;
  question: string;
  turns: Turn[];
  answer: string;
  caveats: string[];
  refused: boolean;
  refusal_reason: string;
  sql: string;
  sql_error: string;
  columns: string[];
  rows: Cell[][];
  row_count: number;
  correct: boolean;
  label: string;
  faithful: boolean | null;
  unsupported_numbers: string[];
  expected: Expected[];
  turn_count: number;
  tokens: number;
  seconds: number;
}

export interface Rate {
  k: number;
  n: number;
  rate: number | null;
  ci95: [number, number];
}

export interface Demo {
  run_id: string;
  provider: string;
  model: string;
  config: string;
  git_sha: string;
  recorded_at: string;
  overall: Rate;
  questions: Question[];
}
