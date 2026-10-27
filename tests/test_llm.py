"""Adapters, cassette and retry, all offline: fixtures shaped like real provider responses."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from courtside.cassette import Cassette, CassetteMiss, request_key
from courtside.llm import (
    TRANSIENT_STATUS,
    GeminiModel,
    GroqModel,
    HttpTransport,
    Message,
    ProviderError,
    RateLimited,
    RetryingTransport,
    ToolCall,
    ToolSpec,
    _retry_hint,
    make_model,
)

TOOL = ToolSpec(
    "run_sql", "Run a query.", {"type": "object", "properties": {"query": {"type": "string"}}}
)


class Scripted:
    """A transport that returns (or raises) prepared responses and remembers every request."""

    def __init__(self, *responses: dict[str, Any] | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((url, dict(headers), json.loads(json.dumps(body))))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


GROQ_TOOL_CALL = {
    "model": "qwen/qwen3.8-27b",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_1",
                        "type": "function",
                        "function": {"name": "run_sql", "arguments": '{"query": "SELECT 1"}'},
                    }
                ],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 64},
    },
}

GEMINI_TOOL_CALL = {
    "modelVersion": "gemini-3.8-flash",
    "candidates": [
        {
            "finishReason": "STOP",
            "content": {
                "role": "model",
                "parts": [
                    {"text": "thinking", "thought": True},
                    {
                        "functionCall": {"name": "run_sql", "args": {"query": "SELECT 1"}},
                        "thoughtSignature": "sig==",
                    },
                ],
            },
        }
    ],
    "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 5, "thoughtsTokenCount": 3},
}


# --- Groq -----------------------------------------------------------------------------------------


def test_groq_parses_tool_calls_usage_and_stop_reason() -> None:
    model = GroqModel(model="qwen/qwen3.8-27b", transport=Scripted(GROQ_TOOL_CALL), key="k")
    out = model.complete([Message("user", "q")], [TOOL], require_tool=True)
    assert out.tool_calls == (ToolCall("fc_1", "run_sql", {"query": "SELECT 1"}),)
    assert out.stop_reason == "tool_use"
    assert (out.usage.input_tokens, out.usage.output_tokens, out.usage.cached_input_tokens) == (
        100,
        7,
        64,
    )


def test_groq_request_shapes_tools_results_and_tool_choice() -> None:
    transport = Scripted(GROQ_TOOL_CALL, GROQ_TOOL_CALL)
    model = GroqModel(transport=transport, key="secret")
    first = model.complete(
        [Message("system", "s"), Message("user", "q")], [TOOL], require_tool=True
    )
    history = [
        Message("user", "q"),
        first.message,
        Message("tool", "rows", tool_call_id="fc_1", tool_name="run_sql"),
    ]
    model.complete(history, [TOOL], force_tool="run_sql")
    url, headers, body = transport.requests[1]
    assert url.endswith("/chat/completions")
    assert headers == {"Authorization": "Bearer secret"}
    assert (
        body["messages"][1]["tool_calls"][0]["id"] == "fc_1"
    )  # the assistant turn is sent back as-is
    assert body["messages"][2] == {"role": "tool", "tool_call_id": "fc_1", "content": "rows"}
    assert body["tool_choice"] == {"type": "function", "function": {"name": "run_sql"}}
    assert transport.requests[0][2]["tool_choice"] == "required"
    assert "secret" not in json.dumps(body)  # the key is only ever a header


def test_groq_tool_use_failed_is_an_empty_turn_not_an_error() -> None:
    error = ProviderError('HTTP 400: {"error":{"code":"tool_use_failed"}}', status=400)
    out = GroqModel(transport=Scripted(error), key="k").complete([Message("user", "q")], [TOOL])
    assert out.tool_calls == () and out.text == "" and out.stop_reason == "other"
    with pytest.raises(ProviderError):  # any other 400 still raises
        GroqModel(transport=Scripted(ProviderError("HTTP 400: bad", 400)), key="k").complete(
            [Message("user", "q")], [TOOL]
        )


def test_unparseable_arguments_are_kept_not_guessed() -> None:
    bad = json.loads(json.dumps(GROQ_TOOL_CALL))
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    out = GroqModel(transport=Scripted(bad), key="k").complete([Message("user", "q")], [TOOL])
    assert out.tool_calls[0].arguments == {"_unparseable_arguments": "{not json"}


# --- Gemini ---------------------------------------------------------------------------------------


def test_gemini_parses_calls_skips_thoughts_and_echoes_signatures() -> None:
    transport = Scripted(GEMINI_TOOL_CALL, GEMINI_TOOL_CALL)
    model = GeminiModel(transport=transport, key="secret")
    out = model.complete(
        [Message("system", "rules"), Message("user", "q")], [TOOL], require_tool=True
    )
    assert out.text == ""  # the thought part is not answer text
    assert out.tool_calls[0].name == "run_sql" and out.tool_calls[0].arguments == {
        "query": "SELECT 1"
    }
    assert out.stop_reason == "tool_use"
    assert out.usage.output_tokens == 8  # candidates + thoughts
    history = [
        Message("user", "q"),
        out.message,
        Message("tool", "r1", tool_call_id="call_0", tool_name="run_sql"),
        Message("tool", "r2", tool_call_id="call_1", tool_name="run_sql"),
    ]
    model.complete(history, [TOOL], force_tool="run_sql")
    url, headers, body = transport.requests[1]
    assert url.endswith("/gemini-3.8-flash:generateContent")
    assert headers == {"x-goog-api-key": "secret"} and "secret" not in url
    assert body["contents"][1]["parts"][1]["thoughtSignature"] == "sig=="  # sent back verbatim
    assert [p["functionResponse"]["response"]["result"] for p in body["contents"][2]["parts"]] == [
        "r1",
        "r2",
    ]
    assert len(body["contents"]) == 3  # consecutive tool results share one user turn
    assert body["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["run_sql"],
    }
    assert transport.requests[0][2]["systemInstruction"]["parts"][0]["text"] == "rules"


def test_gemini_blocked_prompt_is_a_refusal() -> None:
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}}
    out = GeminiModel(transport=Scripted(blocked), key="k").complete([Message("user", "q")], [])
    assert out.stop_reason == "refusal" and "SAFETY" in out.text and out.tool_calls == ()


def test_make_model_keeps_an_empty_cassette(tmp_path: Path) -> None:
    # Regression: Cassette defines __len__, so an empty one is falsy and `transport or default`
    # silently replaced it, and nothing was recorded.
    cassette = Cassette(tmp_path / "c.jsonl", "record", inner=Scripted())
    assert len(cassette) == 0
    model = make_model("groq", transport=cassette)
    assert isinstance(model, GroqModel) and model.transport is cassette


# --- cassette -------------------------------------------------------------------------------------


def test_cassette_records_then_replays_without_network_or_secrets(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    inner = Scripted({"answer": 1})
    rec = Cassette(path, "record", inner=inner)
    assert rec.post("https://x/y", {"Authorization": "Bearer TOPSECRET"}, {"q": 1}) == {"answer": 1}
    assert rec.post("https://x/y", {"Authorization": "Bearer TOPSECRET"}, {"q": 1}) == {"answer": 1}
    assert (rec.misses, rec.hits, len(inner.requests)) == (1, 1, 1)  # the repeat was not re-sent
    assert "TOPSECRET" not in path.read_text(encoding="utf-8")

    replay = Cassette(
        path, "replay", inner=Scripted()
    )  # an empty script: any network call would fail
    assert replay.post("https://x/y", {}, {"q": 1}) == {"answer": 1}
    with pytest.raises(CassetteMiss):
        replay.post("https://x/y", {}, {"q": 2})


def test_cassette_key_ignores_headers_and_key_order() -> None:
    assert request_key("u", {"a": 1, "b": 2}) == request_key("u", {"b": 2, "a": 1})
    assert request_key("u", {"a": 1}) != request_key("u", {"a": 2})


def test_cassette_does_not_store_failed_calls(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    cassette = Cassette(path, "record", inner=Scripted(ProviderError("HTTP 500", 500)))
    with pytest.raises(ProviderError):
        cassette.post("u", {}, {"q": 1})
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_cassette_records_a_deterministic_400_so_the_conversation_replays(tmp_path: Path) -> None:
    # Groq's tool_use_failed 400 makes the agent nudge and continue, so the path depends on it.
    path = tmp_path / "run.jsonl"
    error = ProviderError("HTTP 400: tool_use_failed", 400)
    messages = [Message("user", "x")]
    recorded = GroqModel(transport=Cassette(path, "record", inner=Scripted(error)), key="k")
    assert recorded.complete(messages, []).tool_calls == ()  # an empty turn, as live
    replay = Cassette(path, "replay", inner=Scripted())
    replayed = GroqModel(transport=replay, key="k").complete(messages, [])
    assert replayed.tool_calls == () and replay.hits == 1
    with pytest.raises(ProviderError) as raised:
        replay.post(
            recorded.url,
            {},
            recorded.request_body(messages, [], require_tool=False, max_output_tokens=2048),
        )
    assert raised.value.status == 400 and "tool_use_failed" in str(raised.value)


# --- retry ----------------------------------------------------------------------------------------


def test_retry_waits_for_the_hint_then_succeeds() -> None:
    waits: list[float] = []
    inner = Scripted(
        RateLimited("slow down", retry_after_s=15.7), ProviderError("busy", 503), {"ok": True}
    )
    transport = RetryingTransport(inner, sleep=waits.append)
    assert transport.post("u", {}, {}) == {"ok": True}
    assert waits == [16.7, 10.0]  # hint + 1 s, then 5 s * 2**1 backoff for the 503


def test_retry_gives_up_on_long_waits_and_on_non_transient_errors() -> None:
    long_wait = RetryingTransport(Scripted(RateLimited("daily cap", 3600.0)), sleep=lambda _: None)
    with pytest.raises(RateLimited):
        long_wait.post("u", {}, {})
    bad_request = RetryingTransport(Scripted(ProviderError("HTTP 400", 400)), sleep=lambda _: None)
    with pytest.raises(ProviderError):
        bad_request.post("u", {}, {})


@pytest.mark.parametrize(
    "drop",
    [
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        ConnectionResetError(10054, "An existing connection was forcibly closed"),
        urllib.error.URLError("getaddrinfo failed"),
    ],
)
def test_a_dropped_connection_is_retried_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, drop: Exception
) -> None:
    # p6c3-rep2 died on http.client.RemoteDisconnected, which is not a URLError.
    def urlopen(*_: Any, **__: Any) -> Any:
        raise drop

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(ProviderError) as caught:
        HttpTransport().post("https://example.invalid", {}, {})
    assert caught.value.status in TRANSIENT_STATUS

    waits: list[float] = []
    transport = RetryingTransport(Scripted(caught.value, {"ok": True}), sleep=waits.append)
    assert transport.post("u", {}, {}) == {"ok": True}
    assert waits == [5.0]


def test_retry_hint_parsing() -> None:
    assert _retry_hint("Please try again in 15.6975s. Need more") == pytest.approx(15.6975)
    assert _retry_hint("try again in 1m2.5s") == pytest.approx(62.5)
    assert _retry_hint('"retryDelay": "21s"') == 21.0
    assert _retry_hint("no hint") is None


def test_cassette_does_not_freeze_other_400s(tmp_path: Path) -> None:
    # A bad key or a quota precondition is also a 400; freezing it would block --retry-errors.
    path = tmp_path / "run.jsonl"
    rec = Cassette(
        path, "record", inner=Scripted(ProviderError("HTTP 400: API key not valid", 400))
    )
    with pytest.raises(ProviderError):
        rec.post("u", {}, {"q": 1})
    assert len(Cassette(path, "replay")) == 0
