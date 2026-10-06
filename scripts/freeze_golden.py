"""Freeze the golden set: execute every gold query and write dev.jsonl, test.jsonl and LOCK.json.

    python scripts/freeze_golden.py            # refuses to overwrite a locked test split
    python scripts/freeze_golden.py --force    # deliberate re-freeze; say why in the commit message

The test split is meant to be frozen once and used for the final runs only. If the questions or the
database change, the new test.jsonl hash no longer matches LOCK.json; re-freezing is allowed but is
an explicit, reviewable act, not something that happens by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from courtside.db import open_readonly
from courtside.golden import (
    GOLDEN_DIR,
    SOURCE,
    assign_splits,
    freeze,
    load_source,
    source_problems,
    write_frozen,
)

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "courtside_nba.db"
LOCK = GOLDEN_DIR / "LOCK.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite a locked test split")
    args = ap.parse_args()

    questions = load_source()
    problems = source_problems(questions)
    if problems:
        print("SOURCE PROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    if LOCK.exists() and not args.force:
        locked = json.loads(LOCK.read_text(encoding="utf-8"))
        current = hashlib.sha256((GOLDEN_DIR / "test.jsonl").read_bytes()).hexdigest()
        if current == locked["test_sha256"]:
            print(f"test split is locked (sha256 {current[:12]}...).")
            print("Re-run with --force to re-freeze, and say why in the commit message.")
            return 1

    frozen = freeze(questions, open_readonly(DB))
    hashes = write_frozen(frozen)
    splits = assign_splits(questions)
    lock = {
        "questions": len(questions),
        "by_tier": dict(Counter(q.tier for q in questions)),
        "by_split": dict(Counter(splits.values())),
        "dev_sha256": hashes["dev"],
        "test_sha256": hashes["test"],
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "db_sha256": hashlib.sha256(DB.read_bytes()).hexdigest(),
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"froze {len(questions)} questions: {lock['by_split']}")
    print(f"test sha256 {hashes['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
