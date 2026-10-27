import { ResultTable } from "./ResultTable";
import { TurnItem } from "./TurnItem";
import { TIERS, labelText, plural } from "./format";
import type { Question } from "./types";

export function Detail({ q }: { q: Question }) {
  const tierName = TIERS.find((t) => t.tier === q.tier)?.name ?? q.tier;
  return (
    <article aria-labelledby="question-title">
      <h2 id="question-title" className="question">
        {q.question}
      </h2>

      <p className="verdict">
        <span className={`verdict-word ${q.correct ? "is-correct" : "is-wrong"}`}>
          {q.correct ? "Correct" : "Wrong"}
        </span>
        <span className="verdict-why">{labelText(q.label)}</span>
      </p>

      {q.turns.length > 0 && (
        <section className="section" aria-labelledby="turns-title">
          <h3 id="turns-title">What the model did</h3>
          {/* key: a new question replays the arrival of its turns */}
          <ol className="turns" key={q.id}>
            {q.turns.map((turn, i) => (
              <TurnItem key={i} turn={turn} index={i} />
            ))}
          </ol>
        </section>
      )}

      <section className="section" aria-labelledby="answer-title">
        <h3 id="answer-title">{q.refused ? "It declined" : "Its answer"}</h3>
        <p className={`answer${q.refused ? " is-refusal" : ""}`}>
          {q.refused ? q.refusal_reason || q.answer : q.answer || "No answer."}
        </p>
        {q.caveats.length > 0 && (
          <ul className="caveats">
            {q.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        )}
        {q.faithful === false && (
          <p className="numbers-note">
            Not found in anything the model was shown: {q.unsupported_numbers.join(", ")}.
          </p>
        )}
      </section>

      {q.sql && (
        <section className="section" aria-labelledby="sql-title">
          <h3 id="sql-title">The query behind it</h3>
          <pre className="sql">{q.sql}</pre>
          {q.sql_error ? (
            <p className="error-text">{q.sql_error}</p>
          ) : (
            <div className="result-gap">
              <ResultTable
                columns={q.columns}
                rows={q.rows}
                total={q.row_count}
                caption="The rows the answer is scored on"
              />
            </div>
          )}
        </section>
      )}

      {!q.correct && q.expected.length > 0 && (
        <section className="section" aria-labelledby="expected-title">
          <h3 id="expected-title">
            {q.expected.length > 1 ? "Accepted answers" : "Expected answer"}
          </h3>
          <div className="expected">
            {q.expected.map((e, i) => (
              <ResultTable key={i} columns={e.columns} rows={e.rows} caption="An accepted answer" />
            ))}
          </div>
        </section>
      )}

      <section className="section" aria-labelledby="facts-title">
        <h3 id="facts-title" className="visually-hidden">
          Run facts
        </h3>
        <dl className="facts">
          <div>
            <dt>Kind of question</dt>
            <dd>{tierName}</dd>
          </div>
          <div>
            <dt>Model turns</dt>
            <dd>{q.turn_count}</dd>
          </div>
          <div>
            <dt>Tokens</dt>
            <dd>{q.tokens.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Time</dt>
            <dd>{q.seconds} s</dd>
          </div>
          <div>
            <dt>Question</dt>
            <dd>{q.id}</dd>
          </div>
        </dl>
      </section>
      <span className="visually-hidden" aria-live="polite">
        {`${q.id}: ${q.correct ? "correct" : "wrong"}, ${plural(q.turns.length, "turn")}`}
      </span>
    </article>
  );
}
