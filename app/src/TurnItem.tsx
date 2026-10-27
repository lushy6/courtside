import { type CSSProperties, useState } from "react";
import { ResultTable } from "./ResultTable";
import { plural } from "./format";
import type { Turn } from "./types";

function describe(turn: Turn) {
  if (turn.tool === "run_sql") {
    // The gist line is what tells six "Ran a query" turns apart at a glance.
    return (
      <>
        Ran a query
        <code className="turn-gist">{String(turn.args.query ?? "").replace(/\s+/g, " ")}</code>
      </>
    );
  }
  if (turn.tool === "describe_table") {
    return (
      <>
        Read the notes for <code>{turn.args.name}</code>
      </>
    );
  }
  return (
    <>
      Looked up values of{" "}
      <code>
        {turn.args.table}.{turn.args.column}
      </code>
    </>
  );
}

function outcome(turn: Turn): string {
  if (!turn.ok) return "Error";
  if (turn.tool === "describe_table") return "Notes and samples";
  return plural(turn.row_count, turn.tool === "sample_values" ? "value" : "row");
}

export function TurnItem({ turn, index }: { turn: Turn; index: number }) {
  const [open, setOpen] = useState(false);
  const id = `turn-${index}`;
  return (
    <li
      className={`turn${open ? " is-open" : ""}${turn.ok ? "" : " has-error"}`}
      style={{ "--i": index } as CSSProperties}
    >
      <button
        type="button"
        className="turn-head"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="turn-no">Turn {index + 1}</span>
        <span className="turn-what">{describe(turn)}</span>
        <span className={`turn-out${turn.ok ? "" : " is-error"}`}>{outcome(turn)}</span>
      </button>
      {open && (
        <div className="turn-inner" id={id}>
          {turn.tool === "run_sql" && <pre className="sql">{turn.args.query}</pre>}
          {!turn.ok && <p className="error-text">{turn.error}</p>}
          {turn.ok && turn.tool === "describe_table" && <pre className="notes">{turn.text}</pre>}
          {turn.ok && turn.tool !== "describe_table" && (
            <ResultTable
              columns={turn.columns}
              rows={turn.rows}
              total={turn.row_count}
              caption={`What turn ${index + 1} returned`}
            />
          )}
        </div>
      )}
    </li>
  );
}
