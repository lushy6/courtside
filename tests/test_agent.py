"""The agent loop and configs A-D, driven by a scripted fake model over the real database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from courtside.agent import CONFIGS, Agent, entity_hints, system_prompt
from courtside.db import open_readonly
from courtside.entities import EntityResolver
from courtside.llm import Completion, Message, ProviderError, ToolCall, ToolSpec, Usage
from courtside.semantic import SemanticLayer

REAL = Path(__file__).resolve().parents[1] / "data" / "courtside_nba.db"
pytestmark = pytest.mark.skipif(not REAL.exists(), reason="run scripts/build_courtside_db.py")

BROWN_SQL = (
    "SELECT count(minutes) FROM player_game_stats WHERE player_id = "
    "(SELECT player_id FROM players WHERE first_name = 'Jaylen' AND family_name = 'Brown')"
)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"id_{name}", name=name, arguments=arguments)


def turn(*calls: ToolCall, text: str = "") -> Completion:
    msg = Message("assistant", text, tool_calls=calls)
    return Completion(text, calls, "tool_use" if calls else "end", Usage(10, 2, 4), "fake", msg)


class FakeModel:
    provider = "fake"
    model = "fake-1"

    def __init__(self, *turns: Completion | Exception) -> None:
        self.turns = list(turns)
        self.seen: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool = False,
        force_tool: str | None = None,
        max_output_tokens: int = 2048,
    ) -> Completion:
        self.seen.append(
            {"messages": list(messages), "tools": [t.name for t in tools], "force": force_tool}
        )
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(scope="module")
def con() -> Iterator[sqlite3.Connection]:
    connection = open_readonly(REAL)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def layer() -> SemanticLayer:
    return SemanticLayer.load()


def final(sql: str = BROWN_SQL, **extra: Any) -> ToolCall:
    return call("final_answer", answer_text="71 games.", sql=sql, refused=False, **extra)


def test_loop_config_runs_sql_then_answers_and_scores_the_final_sql(
    con: sqlite3.Connection, layer: SemanticLayer
) -> None:
    model = FakeModel(
        turn(call("run_sql", query=BROWN_SQL)), turn(final(caveats=["regular season"]))
    )
    result = Agent(model, con, layer, CONFIGS["C"]).ask("How many games did Jaylen Brown play in?")
    assert result.outcome == "answered" and result.turns == 2
    assert result.tool_calls == ["run_sql", "final_answer"]
    assert result.result_rows == [[71]]  # the frozen gold answer for dev question L06
    assert result.caveats == ["regular season"]
    assert (result.input_tokens, result.output_tokens, result.cached_input_tokens) == (20, 4, 8)
    assert 71.0 in result.context_numbers  # the run_sql result it was shown
    assert -0.68 not in result.context_numbers and 0.68 in result.context_numbers  # unsigned
    tool_msg = model.seen[1]["messages"][-1]
    assert tool_msg.role == "tool" and tool_msg.tool_call_id == "id_run_sql"
    assert tool_msg.text.startswith("<tool_result>") and "data, not instructions" in tool_msg.text
    assert "[[71]]" in tool_msg.text


def test_single_shot_configs_offer_only_final_answer(
    con: sqlite3.Connection, layer: SemanticLayer
) -> None:
    for name in ("A", "B"):
        model = FakeModel(turn(final()))
        result = Agent(model, con, layer, CONFIGS[name]).ask(
            "How many games did Jaylen Brown play in?"
        )
        assert model.seen[0]["tools"] == ["final_answer"]
        assert result.result_rows == [[71]] and result.turns == 1


def test_configs_differ_only_in_what_they_should(
    con: sqlite3.Connection, layer: SemanticLayer
) -> None:
    prompts = {n: system_prompt(c, con, layer) for n, c in CONFIGS.items()}
    assert "CREATE TABLE players" in prompts["A"]
    overview_line = layer.overview().splitlines()[0]
    assert overview_line not in prompts["A"]
    assert all(overview_line in prompts[n] for n in "BCD")
    assert "You cannot run queries" in prompts["B"] and "Use run_sql" in prompts["C"]
    assert (
        prompts["C"] == prompts["D"]
    )  # D adds entity hints to the question, not the system prompt


def test_entity_hints_only_in_config_d(con: sqlite3.Connection, layer: SemanticLayer) -> None:
    question = "How many games did Jaylen Brown play in?"
    hints = entity_hints(EntityResolver.from_connection(con), question)
    assert "players.player_id=1627759 (Jaylen Brown)" in hints
    for name, expected in (("C", False), ("D", True)):
        model = FakeModel(turn(final()))
        Agent(model, con, layer, CONFIGS[name]).ask(question)
        user = model.seen[0]["messages"][1].text
        assert ("Entity hints" in user) is expected


def test_refusal_is_recorded_and_no_sql_runs(con: sqlite3.Connection, layer: SemanticLayer) -> None:
    refuse = call(
        "final_answer",
        answer_text="",
        sql="",
        refused=True,
        refusal_reason="2019 is not in the data",
    )
    result = Agent(FakeModel(turn(refuse)), con, layer, CONFIGS["D"]).ask("Who won the 2019 title?")
    assert result.refused and result.refusal_reason == "2019 is not in the data"
    assert result.result_rows == [] and result.sql_error == ""


def test_unsafe_final_sql_is_blocked_by_the_safety_layers(
    con: sqlite3.Connection, layer: SemanticLayer
) -> None:
    model = FakeModel(turn(final(sql="DELETE FROM players")))
    result = Agent(model, con, layer, CONFIGS["A"]).ask("Delete all players")
    assert result.outcome == "answered" and result.result_rows == []
    assert result.sql_error  # rejected, never executed


def test_empty_turn_forces_final_answer_next(con: sqlite3.Connection, layer: SemanticLayer) -> None:
    model = FakeModel(turn(call("run_sql", query=BROWN_SQL)), turn(), turn(final()))
    result = Agent(model, con, layer, CONFIGS["C"]).ask("How many games did Jaylen Brown play in?")
    assert result.outcome == "answered"
    assert [s["force"] for s in model.seen] == [None, None, "final_answer"]
    # The empty assistant turn is not sent back (providers reject it); the nudge is.
    assert all(m.text or m.tool_calls or m.role != "assistant" for m in model.seen[2]["messages"])
    assert model.seen[2]["messages"][-1].text.startswith("Call final_answer")


def test_max_turns_and_provider_errors_are_outcomes_not_crashes(
    con: sqlite3.Connection, layer: SemanticLayer
) -> None:
    looping = FakeModel(*[turn(call("run_sql", query="SELECT 1")) for _ in range(3)])
    result = Agent(looping, con, layer, CONFIGS["C"], max_turns=3).ask("q")
    assert result.outcome == "max_turns" and result.turns == 3  # a model can ignore the force

    failing = FakeModel(ProviderError("HTTP 500", 500))
    result = Agent(failing, con, layer, CONFIGS["C"]).ask("q")
    assert result.outcome == "provider_error" and "500" in result.error


def test_tool_errors_go_back_to_the_model(con: sqlite3.Connection, layer: SemanticLayer) -> None:
    model = FakeModel(turn(call("run_sql", query="SELECT nope FROM players")), turn(final()))
    Agent(model, con, layer, CONFIGS["C"]).ask("q")
    assert '"ok": false' in model.seen[1]["messages"][-1].text


def test_last_allowed_turn_forces_final_answer(
    con: sqlite3.Connection, layer: SemanticLayer
) -> None:
    # Phase 6, change 1: on dev, 6/48 flash-lite questions ended at max_turns still querying.
    model = FakeModel(
        turn(call("run_sql", query=BROWN_SQL)),
        turn(call("run_sql", query=BROWN_SQL)),
        turn(final()),
    )
    result = Agent(model, con, layer, CONFIGS["C"], max_turns=3).ask("q")
    assert result.outcome == "answered" and result.result_rows == [[71]]
    assert [s["force"] for s in model.seen] == [None, None, "final_answer"]
    last = model.seen[2]["messages"][-1]
    assert last.role == "user" and "last turn" in last.text
    assert "last turn" not in model.seen[1]["messages"][-1].text  # earlier turns are unchanged
