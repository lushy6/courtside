import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Detail } from "./Detail";
import { TIERS } from "./format";
import type { Demo, Question } from "./types";

type Load = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; demo: Demo };

const SWITCH_MS = 110;

function fromHash(questions: Question[]): string {
  const id = decodeURIComponent(window.location.hash.slice(1));
  return questions.some((q) => q.id === id) ? id : questions[0].id;
}

export default function App() {
  const [load, setLoad] = useState<Load>({ state: "loading" });

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}demo.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`demo.json returned HTTP ${r.status}`);
        return r.json() as Promise<Demo>;
      })
      .then((demo) => setLoad({ state: "ready", demo }))
      .catch((e: unknown) =>
        setLoad({ state: "error", message: e instanceof Error ? e.message : String(e) }),
      );
  }, []);

  if (load.state === "loading") return <Shell />;
  if (load.state === "error") {
    return (
      <Shell>
        <p className="detail error-text">
          The recorded answers could not be loaded ({load.message}). Run
          scripts/export_demo.py to create app/public/demo.json.
        </p>
      </Shell>
    );
  }
  return <Explorer demo={load.demo} />;
}

function Shell({ children, demo }: { children?: ReactNode; demo?: Demo }) {
  return (
    <>
      <header className="masthead">
        <h1 className="wordmark">Courtside</h1>
        <p className="intro">
          Questions about the 2025-26 NBA regular season, answered by a language model that writes
          SQL against a read-only database. Each answer shows the queries it ran, the rows it saw,
          and whether it was right.
        </p>
        {demo && (
          <p className="provenance">
            Recorded answers from one development run of <strong>{demo.model}</strong> (setup{" "}
            {demo.config}): <strong>{demo.overall.k} of {demo.overall.n} correct</strong>. These are
            the questions used while building it, not the held-out test set, whose results are not
            published yet.
          </p>
        )}
      </header>
      {children}
    </>
  );
}

function Explorer({ demo }: { demo: Demo }) {
  const { questions } = demo;
  const [current, setCurrent] = useState(() => fromHash(questions));
  const [switching, setSwitching] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const order = useMemo(
    () => TIERS.flatMap(({ tier }) => questions.filter((q) => q.tier === tier)),
    [questions],
  );

  const select = useCallback((id: string, animate: boolean) => {
    window.clearTimeout(timer.current);
    history.replaceState(null, "", `#${id}`);
    if (!animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setSwitching(false);
      setCurrent(id);
      return;
    }
    setSwitching(true);
    timer.current = window.setTimeout(() => {
      setCurrent(id);
      setSwitching(false);
      window.scrollTo({ top: 0 });
    }, SWITCH_MS);
  }, []);

  // j / k and the arrow keys move through questions. Used constantly, so no animation at all.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest("select, input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      const step = e.key === "j" || e.key === "ArrowDown" ? 1 : e.key === "k" || e.key === "ArrowUp" ? -1 : 0;
      if (!step) return;
      e.preventDefault();
      const at = order.findIndex((q) => q.id === current);
      const next = order[Math.min(Math.max(at + step, 0), order.length - 1)];
      if (next.id !== current) select(next.id, false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [order, current, select]);

  // A pasted link or Back changes the hash without a reload. select() uses replaceState,
  // which fires no hashchange, so this never loops.
  useEffect(() => {
    const onHash = () => setCurrent(fromHash(questions));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [questions]);

  // Keys, links and Back all land here, so the rail always shows where you are.
  useEffect(() => {
    document.getElementById(`pick-${current}`)?.scrollIntoView({ block: "nearest" });
  }, [current]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const q = questions.find((x) => x.id === current) ?? questions[0];

  return (
    <Shell demo={demo}>
      <div className="picker">
        <label htmlFor="question-select">Question</label>
        <select
          id="question-select"
          value={current}
          onChange={(e) => select(e.target.value, true)}
        >
          {TIERS.map(({ tier, name }) => (
            <optgroup key={tier} label={name}>
              {questions
                .filter((x) => x.tier === tier)
                .map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.correct ? "Correct" : "Wrong"}: {x.question}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
      </div>
      <div className="layout">
        <nav className="rail" aria-label="Questions">
          <p className="rail-hint">
            <kbd>j</kbd> <kbd>k</kbd> to move between questions
          </p>
          {TIERS.map(({ tier, name }) => {
            const group = questions.filter((x) => x.tier === tier);
            if (group.length === 0) return null;
            const right = group.filter((x) => x.correct).length;
            return (
              <section className="tier" key={tier} aria-label={name}>
                <h2>
                  {name}, {right} of {group.length}
                </h2>
                <ul>
                  {group.map((x) => (
                    <li key={x.id}>
                      <button
                        type="button"
                        id={`pick-${x.id}`}
                        className="pick"
                        aria-current={x.id === current}
                        onClick={() => select(x.id, true)}
                      >
                        <span
                          className={`mark ${x.correct ? "is-correct" : "is-wrong"}`}
                          aria-label={x.correct ? "Correct" : "Wrong"}
                        >
                          {x.correct ? "✓" : "✕"}
                        </span>
                        <span>{x.question}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
        </nav>
        <main className={`detail${switching ? " is-switching" : ""}`}>
          <Detail q={q} />
        </main>
      </div>
      <footer className="footer">
        Run {demo.run_id}, commit {demo.git_sha.slice(0, 7)}. Source, evaluation method and
        golden questions:{" "}
        <a href="https://github.com/lushy6/courtside">github.com/lushy6/courtside</a>.
      </footer>
    </Shell>
  );
}
