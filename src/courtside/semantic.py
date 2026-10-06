"""Load the hand-written semantic layer and render it for the model.

The YAML (``semantic/nba.yaml``) is the knowledge the model cannot get from column names:
what the metrics mean, which numbers are noise, and the traps in this particular data. This
module turns it into two prompt fragments: ``overview()`` goes in the cached system prompt, and
``describe_table()`` backs the agent's table-detail tool. ``problems()`` compares the YAML with a
live database, so a schema change that the YAML does not know about is caught by a test instead
of showing up as a confidently wrong answer.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from courtside.db import execute, table_schema
from courtside.errors import QueryRejected

DEFAULT_NBA = Path(__file__).resolve().parents[2] / "semantic" / "nba.yaml"


@dataclass(frozen=True)
class SemanticLayer:
    data: dict[str, Any]
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str = DEFAULT_NBA) -> SemanticLayer:
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(data=data, path=p)

    @property
    def sha256(self) -> str:
        """Hash of the YAML file, recorded in every run manifest so results trace to a version."""
        if self.path is None:
            raise ValueError("this layer was built from a dict and has no file to hash")
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def table_names(self) -> list[str]:
        return list(self.data["tables"])

    def columns(self, table: str) -> dict[str, str]:
        return dict(self.data["tables"][table]["columns"])

    # --- drift ------------------------------------------------------------------------------
    def problems(self, con: sqlite3.Connection) -> list[str]:
        """Disagreements between the YAML and the database; empty means they match exactly."""
        real = table_schema(con)
        found: list[str] = []
        documented = set(self.data["tables"])
        for name in sorted(set(real) - documented):
            found.append(f"table {name} exists in the database but is not documented")
        for name in sorted(documented - set(real)):
            found.append(f"table {name} is documented but does not exist in the database")
        for name in sorted(documented & set(real)):
            db_cols = {c for c, _ in real[name]}
            doc_cols = set(self.data["tables"][name]["columns"])
            for col in sorted(db_cols - doc_cols):
                found.append(f"column {name}.{col} exists but is not documented")
            for col in sorted(doc_cols - db_cols):
                found.append(f"column {name}.{col} is documented but does not exist")
        return found

    # --- rendering --------------------------------------------------------------------------
    def overview(self) -> str:
        """The stable system-prompt fragment: scope, rules, metrics and a table index."""
        d = self.data
        lines = [f"DATABASE {d['database']}: {_one_line(d['summary'])}", "", "SCOPE"]
        for key in ("stats", "contracts"):
            lines.append(f"- {key}: {d['scope'][key]}")
        lines.append(f"- refuse when: {_one_line(d['scope']['refuse_when'])}")
        lines += ["", "RULES (follow every one)"]
        for i, rule in enumerate(d["rules"], 1):
            lines.append(f"{i}. [{rule['id']}] {_one_line(rule['text'])}")
        lines += ["", "METRICS"]
        for name, text in d["metrics"].items():
            lines.append(f"- {name}: {_one_line(text)}")
        lines += ["", "TABLES (use describe_table(name) for columns and sample rows)"]
        for name, t in d["tables"].items():
            lines.append(f"- {name} [{t['grain']}]: {_one_line(t['purpose'])}")
        return "\n".join(lines)

    def describe_table(
        self, name: str, con: sqlite3.Connection | None = None, sample_rows: int = 3
    ) -> str:
        """Columns, notes and (given a connection) a few real rows, for the table-detail tool."""
        key = name.lower()
        if key not in self.data["tables"]:
            raise QueryRejected(
                "unknown_table",
                f"Unknown table '{name}'. Available tables: {', '.join(self.table_names)}.",
            )
        t = self.data["tables"][key]
        types = {}
        if con is not None:
            types = dict(table_schema(con).get(key, []))
        lines = [f"TABLE {key}: {_one_line(t['purpose'])}", f"Grain: {t['grain']}"]
        if t.get("notes"):
            lines.append("Notes:")
            lines += [f"- {_one_line(n)}" for n in t["notes"]]
        lines.append("Columns:")
        for col, desc in t["columns"].items():
            typ = f" ({types[col]})" if types.get(col) else ""
            lines.append(f"- {col}{typ}: {_one_line(desc)}")
        if con is not None and sample_rows > 0:
            res = execute(con, f'SELECT * FROM "{key}" LIMIT {int(sample_rows)}', max_cell_chars=40)
            lines.append(f"Sample rows (first {len(res.rows)}):")
            lines.append(" | ".join(res.columns))
            lines += [" | ".join("NULL" if v is None else str(v) for v in row) for row in res.rows]
        return "\n".join(lines)


def _one_line(text: object) -> str:
    return " ".join(str(text).split())
