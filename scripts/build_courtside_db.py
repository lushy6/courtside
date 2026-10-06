"""Build data/courtside_nba.db from the two sibling NBA projects.

    python scripts/build_courtside_db.py
    python scripts/build_courtside_db.py --lineup-db X --tactical-db Y --out Z

Writes the database plus a ``.manifest.json`` beside it (hashes of the inputs and the output,
row counts, and the checks that passed). Exits non-zero, and writes nothing, if the sources
disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from courtside.build import BuildError, build

REPO = Path(__file__).resolve().parents[1]
PROJECTS = REPO.parents[1]  # ...\Sports\_Projects
DEFAULT_LINEUP = PROJECTS / "NBA" / "nba-lineup-analytics" / "data" / "nba.db"
DEFAULT_TACTICAL = PROJECTS / "NBA" / "nba-tactical-optimizer" / "data" / "nba.db"
DEFAULT_OUT = REPO / "data" / "courtside_nba.db"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineup-db", type=Path, default=DEFAULT_LINEUP)
    ap.add_argument("--tactical-db", type=Path, default=DEFAULT_TACTICAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        manifest = build(args.out, args.lineup_db, args.tactical_db)
    except BuildError as err:
        print(f"BUILD FAILED: {err}", file=sys.stderr)
        return 1

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(t["rows"] for t in manifest["tables"].values())
    size_mb = manifest["out_bytes"] / 1e6
    print(f"built {args.out} ({size_mb:.1f} MB, {len(manifest['tables'])} tables, {total:,} rows)")
    for label in manifest["checks_passed"]:
        print(f"  ok: {label}")
    print(f"sha256 {manifest['out_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
