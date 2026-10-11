"""Model-facing tools preserve the read-only boundary and bounded outputs."""

from __future__ import annotations

from courtside.db import open_readonly
from courtside.semantic import SemanticLayer
from courtside.tools import ToolRuntime


def _runtime(db_path):
    layer = SemanticLayer(
        {
            "tables": {
                "players": {
                    "purpose": "Fixture players",
                    "grain": "one player",
                    "columns": {
                        "player_id": "ID",
                        "first_name": "First name",
                        "family_name": "Family name",
                    },
                }
            }
        }
    )
    return ToolRuntime(open_readonly(db_path), layer)


def test_tool_runtime_runs_bounded_select_and_reports_rejected_write(db_path):
    runtime = _runtime(db_path)
    result = runtime.call("run_sql", {"query": "SELECT player_id FROM players ORDER BY player_id"})
    assert result["ok"] is True
    assert result["columns"] == ["player_id"]
    assert result["rows"] == [[1], [2], [3]]
    assert result["truncated"] is False

    rejected = runtime.call("run_sql", {"query": "DELETE FROM players"})
    assert rejected["ok"] is False
    assert rejected["error"]["kind"]


def test_tool_runtime_rejects_unknown_identifiers_and_bad_arguments(db_path):
    runtime = _runtime(db_path)
    assert runtime.call("sample_values", {"table": "players", "column": "family_name"})["rows"] == [
        ["Brown"], ["Jokic"], ["Tatum"]
    ]
    for arguments in (
        {"table": "players", "column": "missing"},
        {"table": 'players"; DROP TABLE players; --', "column": "family_name"},
    ):
        response = runtime.call("sample_values", arguments)
        assert response["ok"] is False
        assert response["error"]["kind"] == "invalid_arguments"
    assert runtime.call("run_sql", {"query": 123})["ok"] is False
    assert runtime.call("run_sql", ["SELECT 1"])["ok"] is False  # type: ignore[arg-type]
    assert runtime.call("unknown", {})["ok"] is False


def test_tool_runtime_describes_only_documented_tables(db_path):
    runtime = _runtime(db_path)
    described = runtime.call("describe_table", {"name": "players"})
    assert described["ok"] is True
    assert "player_id" in described["text"]
    assert runtime.call("describe_table", {"name": "teams"})["ok"] is False
