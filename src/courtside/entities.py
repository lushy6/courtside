"""Resolve names typed by a person to the ids stored in the database.

The model is good at SQL and bad at knowing that "Jokic" is ``player_id`` 203999 stored as
"Nikola Jokić". A wrong guess here turns a correct query into a confident wrong answer, so the
rules are conservative:

* exact normalised full name first, then exact family name, then fuzzy matching;
* a family name shared by several players is reported as **ambiguous**, never silently resolved;
* an unknown name resolves to nothing rather than to the nearest neighbour (the 2025-26 data
  has no Tyrese Haliburton, and the right answer to a question about him is to say so).

Equally-scored candidates are ordered by how much they played, so the top result is the most
likely one even when the caller must still be told it is ambiguous.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

from courtside.db import execute

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_NO_DECOMPOSITION = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ı": "i", "æ": "ae", "ß": "ss"})
_PUNCT = re.compile(r"[.'’`´]")
_SEPARATORS = re.compile(r"[-_/,]+")

FUZZY_FULL = 85  # token-aware ratio on full names
FUZZY_FAMILY = 88  # single-token query against family names (stricter: shorter strings)
MIN_FUZZY_LEN = 5  # never fuzzy-match tiny strings such as team abbreviations
MENTION_FUZZY = 92

# Nicknames people use for teams, mapped to the stored nickname.
TEAM_ALIASES = {
    "sixers": "76ers",
    "cavs": "cavaliers",
    "mavs": "mavericks",
    "wolves": "timberwolves",
    "twolves": "timberwolves",
    "blazers": "trail blazers",
    "dubs": "warriors",
    "clips": "clippers",
    "pels": "pelicans",
    "grizz": "grizzlies",
}

# Team nicknames that are also ordinary English words, so lower-case use is not a team mention.
_EVERYDAY_TEAM_WORDS = frozenset({"heat", "magic", "jazz", "kings", "suns", "nets", "thunder"})

# Capitalised words that start questions or name concepts, and must never be read as people.
_STOPWORDS = frozenset(
    "a an the how what which who whom whose when where why compare best worst top most least "
    "highest lowest list show give tell average total per game games season team teams lineup "
    "lineups player players points rebounds assists minutes salary contract contracts net rating "
    "is are was were did does do has have had and or vs versus than with without in on of for "
    "to by at from".split()  # noqa: SIM905  a word list reads better than 60 quoted strings
)


def normalize(text: str) -> str:
    """Lowercase, strip diacritics, periods, apostrophes and generational suffixes."""
    decomposed = unicodedata.normalize("NFKD", text.lower().translate(_NO_DECOMPOSITION))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    plain = _PUNCT.sub("", plain)
    plain = _SEPARATORS.sub(" ", plain)
    tokens = [t for t in plain.split() if t]
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


PlayerHow = Literal["exact", "family", "fuzzy", "first_name"]
TeamHow = Literal["exact", "fuzzy"]


@dataclass(frozen=True)
class PlayerMatch:
    player_id: int
    name: str
    score: float
    how: PlayerHow


@dataclass(frozen=True)
class TeamMatch:
    team_id: int
    abbr: str
    name: str
    score: float
    how: TeamHow


@dataclass(frozen=True)
class PlayerResolution:
    query: str
    matches: tuple[PlayerMatch, ...]
    ambiguous: bool


@dataclass(frozen=True)
class TeamResolution:
    query: str
    matches: tuple[TeamMatch, ...]
    ambiguous: bool


@dataclass(frozen=True)
class Mention:
    kind: Literal["player", "team"]
    text: str  # as written in the question
    matches: tuple[PlayerMatch, ...] | tuple[TeamMatch, ...]
    ambiguous: bool
    start: int  # token index, inclusive
    end: int  # token index, exclusive


@dataclass(frozen=True)
class _Player:
    player_id: int
    name: str
    full: str
    family: str
    first: str
    prominence: float


@dataclass(frozen=True)
class _Team:
    team_id: int
    abbr: str
    city: str
    nickname: str

    @property
    def name(self) -> str:
        return f"{self.city} {self.nickname}"


class EntityResolver:
    def __init__(self, players: list[_Player], teams: list[_Team]) -> None:
        self.players = players
        self.teams = teams
        self._by_full: dict[str, list[_Player]] = {}
        self._by_family: dict[str, list[_Player]] = {}
        self._by_first: dict[str, list[_Player]] = {}
        for p in players:
            self._by_full.setdefault(p.full, []).append(p)
            self._by_family.setdefault(p.family, []).append(p)
            self._by_first.setdefault(p.first, []).append(p)
        self._full_keys = list(self._by_full)
        self._family_keys = list(self._by_family)

        # Every way a person might write a team: abbreviation, city, nickname, "city nickname".
        # A city shared by two teams (Los Angeles) maps to both, which is what makes it ambiguous.
        keyed: dict[str, dict[int, _Team]] = {}
        for t in teams:
            for key in (t.abbr, t.city, t.nickname, t.name):
                keyed.setdefault(normalize(key), {})[t.team_id] = t
        for alias, target in TEAM_ALIASES.items():
            if target in keyed:
                keyed.setdefault(alias, {}).update(keyed[target])
        self._team_keys: dict[str, list[_Team]] = {k: list(v.values()) for k, v in keyed.items()}
        self._nickname_keys = {normalize(t.nickname) for t in teams} | {
            a for a, target in TEAM_ALIASES.items() if target in keyed
        }

    # --- construction -----------------------------------------------------------------------
    @classmethod
    def from_connection(cls, con: sqlite3.Connection) -> EntityResolver:
        raw = execute(
            con,
            "SELECT player_id, first_name, family_name FROM players",
            max_rows=100_000,
            max_total_chars=10_000_000,
        ).rows
        prominence = _prominence(con)
        players = [
            _Player(
                player_id=int(pid),
                name=f"{first} {family}",
                full=normalize(f"{first} {family}"),
                family=normalize(str(family)),
                first=normalize(str(first)),
                prominence=prominence.get(int(pid), 0.0),
            )
            for pid, first, family in raw
        ]
        team_rows = execute(con, "SELECT team_id, abbr, city, name FROM teams", max_rows=1000).rows
        teams = [
            _Team(int(tid), str(abbr), str(city), str(name)) for tid, abbr, city, name in team_rows
        ]
        return cls(players, teams)

    # --- players ----------------------------------------------------------------------------
    def resolve_player(self, query: str, limit: int = 5) -> PlayerResolution:
        q = normalize(query)
        if not q:
            return PlayerResolution(query, (), False)
        scored: list[tuple[float, PlayerHow, _Player]] = []

        for p in self._by_full.get(q, []):
            scored.append((100.0, "exact", p))
        if not scored:
            for p in self._by_family.get(q, []):
                scored.append((92.0, "family", p))
        if not scored:
            scored = self._fuzzy_players(q)
        if not scored and " " not in q:
            for p in self._by_first.get(q, []):
                scored.append((70.0, "first_name", p))

        scored.sort(key=lambda s: (-s[0], -s[2].prominence, s[2].name))
        matches = tuple(
            PlayerMatch(p.player_id, p.name, round(score, 1), how)
            for score, how, p in scored[:limit]
        )
        ambiguous = bool(scored) and sum(1 for s in scored if s[0] >= scored[0][0] - 3) > 1
        return PlayerResolution(query, matches, ambiguous)

    def _fuzzy_players(self, q: str) -> list[tuple[float, PlayerHow, _Player]]:
        found: dict[int, tuple[float, PlayerHow, _Player]] = {}

        def add(score: float, players: list[_Player]) -> None:
            for p in players:
                if p.player_id not in found or found[p.player_id][0] < score:
                    found[p.player_id] = (score, "fuzzy", p)

        if len(q) >= MIN_FUZZY_LEN:
            for key, score, _ in process.extract(
                q, self._full_keys, scorer=_name_ratio, score_cutoff=FUZZY_FULL, limit=10
            ):
                add(score, self._by_full[key])
            if " " not in q:
                for key, score, _ in process.extract(
                    q, self._family_keys, scorer=fuzz.ratio, score_cutoff=FUZZY_FAMILY, limit=10
                ):
                    add(score, self._by_family[key])
        return list(found.values())

    # --- teams ------------------------------------------------------------------------------
    def resolve_team(self, query: str, limit: int = 5) -> TeamResolution:
        q = normalize(query)
        if not q:
            return TeamResolution(query, (), False)
        hits: list[tuple[float, TeamHow, _Team]] = [
            (100.0, "exact", t) for t in self._team_keys.get(q, [])
        ]
        if not hits and len(q) >= MIN_FUZZY_LEN:
            best: dict[int, tuple[float, TeamHow, _Team]] = {}
            for key, score, _ in process.extract(
                q,
                [k for k in self._team_keys if len(k) >= MIN_FUZZY_LEN],
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_FULL,
                limit=5,
            ):
                for t in self._team_keys[key]:
                    if t.team_id not in best or best[t.team_id][0] < score:
                        best[t.team_id] = (score, "fuzzy", t)
            hits = list(best.values())
        hits.sort(key=lambda h: (-h[0], h[2].abbr))
        matches = tuple(
            TeamMatch(t.team_id, t.abbr, t.name, round(s, 1), how) for s, how, t in hits[:limit]
        )
        ambiguous = bool(hits) and sum(1 for h in hits if h[0] >= hits[0][0] - 3) > 1
        return TeamResolution(query, matches, ambiguous)

    # --- free text --------------------------------------------------------------------------
    def find_mentions(self, text: str) -> list[Mention]:
        """Entities named in a question, longest match first, never overlapping.

        These are hints for the model, not facts, so the rules favour precision: a one-word
        player or team must be capitalised in the question (or be an upper-case abbreviation),
        which keeps "which brown lineup" from naming a person.
        """
        words = [w for w in (_strip(t) for t in text.split()) if w]
        mentions: list[Mention] = []
        i = 0
        while i < len(words):
            hit = None
            for n in range(min(4, len(words) - i), 0, -1):
                hit = self._mention_at(words, i, n)
                if hit:
                    break
            if hit:
                mentions.append(hit)
                i = hit.end
            else:
                i += 1
        return mentions

    def _mention_at(self, words: list[str], i: int, n: int) -> Mention | None:
        window = words[i : i + n]
        phrase = " ".join(window)
        key = normalize(phrase)
        capitalised = True
        if n == 1:
            w = window[0]
            if w.lower() in _STOPWORDS or len(w) < 2:
                return None
            is_abbr = w.isupper() and 2 <= len(w) <= 4
            capitalised = w[0].isupper() or w[0].isdigit()
            # Lower-case is fine for a team nickname that is not also an everyday word.
            lowercase_team_ok = key in self._nickname_keys and key not in _EVERYDAY_TEAM_WORDS
            if not capitalised and not lowercase_team_ok:
                return None
        else:
            is_abbr = False
            if any(w.lower() in _STOPWORDS for w in (window[0], window[-1])):
                return None

        team = self.resolve_team(phrase)
        if team.matches and team.matches[0].how == "exact":
            abbr_only = key == normalize(team.matches[0].abbr)
            if not abbr_only or is_abbr:
                return Mention("team", phrase, team.matches, team.ambiguous, i, i + n)
        if n == 1 and not capitalised:
            return None
        player = self.resolve_player(phrase)
        if player.matches:
            top = player.matches[0]
            if n == 1:
                ok = (top.how in {"exact", "family"} and len(key) >= 3) or (
                    top.how == "first_name" and not player.ambiguous and len(key) >= 4
                )
            else:
                ok = top.score >= MENTION_FUZZY and top.how != "first_name"
            if ok:
                return Mention("player", phrase, player.matches, player.ambiguous, i, i + n)
        return None


def _name_ratio(a: str, b: str, **_: object) -> float:
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))


def _strip(token: str) -> str:
    """Drop surrounding punctuation and a possessive ("Tatum's", "Celtics'", "Heat’s")."""
    word = token.strip(".,;:?!()\"'’[]{}")
    for possessive in ("'s", "’s"):
        if word.lower().endswith(possessive):
            return word[: -len(possessive)]
    return word


def _prominence(con: sqlite3.Connection) -> dict[int, float]:
    """Possessions played per player, to order equally-scored candidates (0 if unavailable)."""
    try:
        rows = execute(
            con,
            "SELECT player_id, poss FROM player_season",
            max_rows=100_000,
            max_total_chars=10_000_000,
        ).rows
    except Exception:  # noqa: BLE001  the table is optional; the resolver works without it
        return {}
    return {int(pid): float(poss or 0.0) for pid, poss in rows}
