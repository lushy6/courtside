# Courtside: bounded, read-only sports SQL

A Python toolkit for safely exploring an offline NBA SQLite database. It combines SQL validation, database-level read-only enforcement, a documented semantic layer, entity resolution, and a frozen question set for future text-to-SQL evaluation.

**Current release:** the database, safety layers, semantic definitions, entity resolver, golden questions, and local tool runtime are implemented. **An LLM adapter, natural-language chat interface, model evaluation results, and hosted demo are not implemented.** This project demonstrates database safety and evaluation infrastructure; it does not claim measured text-to-SQL model accuracy.

## Architecture

```text
Untrusted SQL / tool arguments
  -> SQL AST validation and allowlisted names/functions
  -> SQLite authorizer
  -> immutable, read-only SQLite connection
  -> bounded execution: time, rows, columns, values, output budget
  -> structured results or explicit rejection
```

The semantic YAML explains table grains, metrics, and uncertainty. Its assertions are tested against the bundled database so that schema and metric changes cannot silently invalidate the documentation.

## Run locally

Requires Python 3.12+. From a fresh checkout:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python examples/read_only_query.py
```

The example reads the bundled database, prints a descriptive lineup summary, and demonstrates rejection of a write attempt. It makes no network or model calls and does not modify the database.

## Verify

```sh
python -m ruff check .
python -m mypy src
python -m pytest -o addopts="" -q
```

On October 17, 2026, the local review ran **484 passing tests**, including attacks against individual safety layers and the real bundled database. Lint and strict source typing passed. CI repeats these checks; the run status is available in the Actions tab.

## Evaluation design

The versioned question set contains 120 items, split into 48 development and 72 held-out test questions. It includes lookups, aggregation, joins, ambiguous questions, unanswerable questions, statistical traps, and adversarial requests. Hash locks detect changes to the frozen files. These are **benchmark design facts, not model performance results**.

See [the implementation status](docs/PLAN.md), [external checks](docs/GOLDEN_CHECKS.md), and [the semantic layer](semantic/nba.yaml).

## Data and scope

The bundled NBA database and manifest record source hashes, row counts, and build checks. `scripts/build_courtside_db.py` can rebuild it from the lineup-analytics and tactical-optimizer source databases, using explicit `--lineup-db` and `--tactical-db` paths. The source databases are not changed.

Sports statistics are descriptive. Small samples and uncertainty affect comparisons. No prediction, betting edge, production authentication, or general-purpose SQL sandbox is claimed. The safety policy is deliberately restricted to this documented SQLite schema.

## Portfolio context

By Matias Birbuet. The repository history records AI-assisted development. The implemented engineering work to discuss is the SQL boundary, deterministic database merge, semantic drift tests, entity matching, and frozen evaluation design. See the commit history for attribution.
