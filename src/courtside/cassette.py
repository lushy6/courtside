"""Record and replay provider calls, so a run can be re-scored offline and resumed for free.

Sampling cannot be fixed on these APIs, so reproducibility comes from keeping every response.
A cassette is a JSONL file of ``{"key", "url", "request", "response"}`` lines. The key is the
SHA-256 of the URL and the request body; headers (which carry the API key) are never part of
the key and never written.

Modes:
- ``replay``: every call must already be in the cassette; a missing one raises ``CassetteMiss``.
- ``record``: calls already in the cassette are replayed, new ones go to the network and are
  appended. An interrupted run therefore resumes without repeating any call.
- ``live``: pass-through, nothing stored.

Because each agent turn's request contains the previous responses, replaying turn 1 yields the
same turn-2 request, and a whole multi-turn conversation replays from one file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from courtside.llm import HttpTransport, Transport

Mode = Literal["replay", "record", "live"]


class CassetteMiss(KeyError):
    """Replay mode met a request that was never recorded."""


def request_key(url: str, body: Mapping[str, Any]) -> str:
    canonical = json.dumps({"url": url, "body": body}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Cassette:
    def __init__(
        self, path: Path | str, mode: Mode = "record", inner: Transport | None = None
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.inner = inner if inner is not None else HttpTransport()
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry["response"]

    def __len__(self) -> int:
        return len(self._entries)

    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "live":
            return self.inner.post(url, headers, body)
        key = request_key(url, body)
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        if self.mode == "replay":
            raise CassetteMiss(f"request not in cassette {self.path.name} (key {key[:12]})")
        response = self.inner.post(url, headers, body)  # errors propagate and are not stored
        self.misses += 1
        self._entries[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            line = {"key": key, "url": url, "request": body, "response": response}
            f.write(json.dumps(line, sort_keys=True) + "\n")
        return response
