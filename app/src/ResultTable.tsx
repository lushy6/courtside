import { type ReactNode } from "react";
import { IntervalStrip } from "./IntervalStrip";
import { LEAGUE_AVERAGE, formatCell, isNumber, plural, stripDomain } from "./format";
import type { Cell } from "./types";

interface Props {
  columns: string[];
  rows: Cell[][];
  total?: number;
  caption?: string;
}

const POINT_COLUMNS = ["shrunk", "net_rating"];
const LONG_ID = /^lineup(_uid)?$/i; // five player ids joined: useful to a query, noise to a reader

export function ResultTable({ columns, rows, total, caption }: Props) {
  if (columns.length === 0) return null;
  const lo = columns.indexOf("lo95");
  const hi = columns.indexOf("hi95");
  const point = POINT_COLUMNS.map((c) => columns.indexOf(c)).find((i) => i >= 0) ?? -1;
  const intervals = lo >= 0 && hi >= 0 ? rows.filter((r) => isNumber(r[lo]) && isNumber(r[hi])) : [];
  const domain =
    intervals.length > 0
      ? stripDomain(intervals.map((r) => [r[lo] as number, r[hi] as number]))
      : null;
  const numeric = columns.map((_, c) => rows.some((r) => isNumber(r[c])));
  // The strip goes right after the columns that name the row, before the numbers, so it is
  // on screen beside the lineup it describes instead of past a horizontal scroll.
  const firstNumber = numeric.indexOf(true);
  const stripAt = firstNumber >= 0 ? firstNumber : columns.length;
  const count = total ?? rows.length;

  const withStrip = (cells: ReactNode[], strip: ReactNode) =>
    domain ? [...cells.slice(0, stripAt), strip, ...cells.slice(stripAt)] : cells;

  return (
    <>
      <div className="table-wrap">
        <table>
          {caption && <caption className="visually-hidden">{caption}</caption>}
          <thead>
            <tr>
              {withStrip(
                columns.map((c, i) => (
                  <th
                    key={c + i}
                    scope="col"
                    className={numeric[i] ? "num" : LONG_ID.test(c) ? "long-id" : undefined}
                  >
                    {c}
                  </th>
                )),
                <th key="strip" scope="col">
                  95% interval
                </th>,
              )}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={columns.length + (domain ? 1 : 0)}>No rows.</td>
              </tr>
            )}
            {rows.map((row, r) => (
              <tr key={r}>
                {withStrip(
                  row.map((cell, c) => {
                    const text = formatCell(cell, columns[c]);
                    return LONG_ID.test(columns[c]) ? (
                      <td key={c} className="long-id" title={text}>
                        {text}
                      </td>
                    ) : (
                      <td key={c} className={numeric[c] ? "num" : undefined}>
                        {text}
                      </td>
                    );
                  }),
                  <td key="strip">
                    {isNumber(row[lo]) && isNumber(row[hi]) ? (
                      <IntervalStrip
                        lo={row[lo]}
                        hi={row[hi]}
                        point={point >= 0 && isNumber(row[point]) ? row[point] : null}
                        domain={domain!}
                      />
                    ) : (
                      "—"
                    )}
                  </td>,
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {count > rows.length && (
        <p className="table-note">
          Showing {rows.length} of {plural(count, "row")}.
        </p>
      )}
      {domain && (
        <p className="strip-legend">
          <svg width="18" height="12" aria-hidden="true">
            <line x1="9" x2="9" y1="0" y2="12" stroke="currentColor" strokeDasharray="2 2" />
          </svg>
          Dashed line: league-average lineup ({formatCell(LEAGUE_AVERAGE, "")} per 100 possessions). A band
          that crosses it is not distinguishable from average.
        </p>
      )}
    </>
  );
}
