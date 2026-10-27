"""Provider-agnostic chat interface with tool calls, and the Gemini and Groq adapters.

The agent loop is written once against ``ChatModel.complete``. Each adapter translates the
common shapes (``Message``, ``ToolSpec``, ``ToolCall``) to its provider's wire format and back,
and normalises usage and the stop reason. Requests go through a ``Transport``; wrapping it in a
``Cassette`` (``courtside.cassette``) records or replays every call, which is how runs are made
reproducible without control over sampling.

API keys travel only in request headers, never in URLs or request bodies, so a cassette (which
stores URL, body and response) never contains a secret.

A Claude adapter is planned for the one paid comparison run (PLAN.md phase 7b) and will be
written against the official SDK then; nothing here depends on it.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["end", "tool_use", "max_tokens", "refusal", "other"]

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
NO_RESPONSE = 599  # no HTTP status at all: the connection failed or was dropped
# 408 = our own read timeout
TRANSIENT_STATUS = frozenset({408, 500, 502, 503, 504, NO_RESPONSE})


class ProviderError(RuntimeError):
    """The provider returned an error or a response this adapter cannot read."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimited(ProviderError):
    """HTTP 429, with the retry hint and evidence of an exceeded daily quota."""

    def __init__(
        self, message: str, retry_after_s: float | None = None, *, daily: bool | None = None
    ) -> None:
        super().__init__(message, status=429)
        self.retry_after_s = retry_after_s
        self.daily = _daily_quota(message) if daily is None else daily


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""  # role == "tool": which call this answers
    tool_name: str = ""  # role == "tool": Gemini matches results by name
    # Assistant turns keep the provider's own payload so the same adapter can send it back
    # verbatim (Gemini 3 rejects a follow-up whose function-call parts lost their signatures).
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: StopReason
    usage: Usage
    model: str
    message: Message  # the assistant turn, ready to append to the conversation


class Transport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], body: dict[str, Any]
    ) -> dict[str, Any]: ...


class HttpTransport:
    """Plain JSON POST over urllib; maps HTTP errors to ``ProviderError`` / ``RateLimited``."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s

    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        # An explicit User-Agent: Groq's CDN answers urllib's default one with HTTP 403 (code 1010).
        base = {"Content-Type": "application/json", "User-Agent": "courtside/0.1"}
        request = urllib.request.Request(url, data=data, headers={**base, **headers}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                parsed: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return parsed
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 429:
                retry = _float_or_none(error.headers.get("retry-after")) or _retry_hint(detail)
                # Parse metadata before trimming the diagnostic: RetryInfo often comes last.
                raise RateLimited(
                    f"rate limited: {detail[:500]}", retry, daily=_daily_quota(detail)
                ) from error
            raise ProviderError(f"HTTP {error.code}: {detail[:500]}", status=error.code) from error
        except urllib.error.URLError as error:
            raise ProviderError(f"network error: {error.reason}", status=NO_RESPONSE) from error
        except (http.client.HTTPException, ConnectionError) as error:
            # A connection the server closed mid-request (RemoteDisconnected) is not a URLError.
            raise ProviderError(f"connection dropped: {error!r}", status=NO_RESPONSE) from error
        except TimeoutError as error:  # a slow body read raises this directly, not URLError
            raise ProviderError(f"no response within {self.timeout_s:.0f} s", status=408) from error


class RetryingTransport:
    """Waits out HTTP 429s and retries, so free-tier per-minute caps slow a run instead of failing.

    Also retries 500/502/503/504 ("model overloaded") and dropped connections with backoff.
    Uses the provider's hint when there is one (a Retry-After header, Groq's "try again in 15.7s",
    Gemini's "retryDelay": "15s"), otherwise exponential backoff. Gives up after ``max_retries``
    or when a single wait would exceed ``max_wait_s``. The delay alone does not identify
    which quota was exceeded; the runner handles explicit daily quota failures separately.
    """

    def __init__(
        self,
        inner: Transport | None = None,
        *,
        max_retries: int = 6,
        max_wait_s: float = 90.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner if inner is not None else HttpTransport()
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.sleep = sleep
        self.waited_s = 0.0

    def post(self, url: str, headers: Mapping[str, str], body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return self.inner.post(url, headers, body)
            except ProviderError as error:
                if isinstance(error, RateLimited):
                    wait = (error.retry_after_s or 2.0 * 2**attempt) + 1.0
                elif error.status in TRANSIENT_STATUS:  # overloaded / gateway errors pass quickly
                    wait = 5.0 * 2**attempt
                else:
                    raise
                if attempt == self.max_retries or wait > self.max_wait_s:
                    raise
                self.sleep(wait)
                self.waited_s += wait
        raise AssertionError("unreachable")


class ChatModel(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool = False,
        force_tool: str | None = None,
        max_output_tokens: int = 2048,
    ) -> Completion: ...


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines from ``.env`` (no quoting rules beyond stripping); real env vars win."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in list(values):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def api_key(name: str) -> str:
    value = os.environ.get(name) or load_env().get(name, "")
    if not value:
        raise ProviderError(f"{name} is not set (put it in .env at the repo root)")
    return value


def _float_or_none(text: str | None) -> float | None:
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _retry_hint(detail: str) -> float | None:
    """Seconds to wait, from a 429 body: Groq "try again in 1m2.5s", Gemini "retryDelay"."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", detail)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', detail)
    return float(m.group(1)) if m else None


def _daily_quota(detail: str) -> bool:
    """Recognize exceeded quotas, not daily limits merely mentioned in help or metadata."""
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = {"error": {"message": detail}}
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return False
    details = error.get("details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict) or item.get("@type") != (
                "type.googleapis.com/google.rpc.QuotaFailure"
            ):
                continue
            violations = item.get("violations")
            if not isinstance(violations, list):
                continue
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                # Google QuotaFailure entries describe limits actually exceeded.
                period = " ".join(
                    str(violation.get(k, "")) for k in ("quotaId", "quotaMetric", "description")
                )
                if re.search(r"per[ _-]?day|\bdaily\b|\b(?:TPD|RPD)\b", period, re.IGNORECASE):
                    return True
    # Groq names the exceeded period in error.message, without QuotaFailure details. Gemini's
    # message also lists exceeded quotas ("... requests per day limit exceeded"), a fallback for
    # when details are missing or the body was cut short.
    message = error.get("message")
    return isinstance(message, str) and bool(
        re.search(
            r"\brate limit reached\b[^\n]*?\bon (?:tokens|requests) per day\b"
            r"|\bper day limit exceeded\b",
            message,
            re.IGNORECASE,
        )
    )


def _parse_arguments(raw: object) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string (Groq) or an object (Gemini); never string-match."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {"_unparseable_arguments": raw}
        return parsed if isinstance(parsed, dict) else {"_non_object_arguments": parsed}
    return {}


# --- Groq (OpenAI-compatible chat completions) ---------------------------------------------------


@dataclass
class GroqModel:
    model: str = "openai/gpt-oss-120b"
    transport: Transport = field(default_factory=HttpTransport)
    key: str | None = None
    provider: str = "groq"
    url: str = "https://api.groq.com/openai/v1/chat/completions"

    def request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool,
        force_tool: str | None = None,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant" and m.raw is not None:
                wire.append(m.raw)
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.text}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ]
                wire.append(entry)
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
            else:
                wire.append({"role": m.role, "content": m.text})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "max_completion_tokens": max_output_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if force_tool:
                body["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
            else:
                body["tool_choice"] = "required" if require_tool else "auto"
        return body

    def parse(self, response: dict[str, Any]) -> Completion:
        try:
            choice = response["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected Groq response: {str(response)[:300]}") from error
        calls = tuple(
            ToolCall(
                id=str(c.get("id", "")),
                name=str(c["function"]["name"]),
                arguments=_parse_arguments(c["function"].get("arguments")),
            )
            for c in msg.get("tool_calls") or []
        )
        finish = choice.get("finish_reason")
        stop: StopReason = {
            "stop": "end",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "refusal",
        }.get(str(finish), "other")  # type: ignore[assignment]
        if calls and stop == "end":
            stop = "tool_use"
        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        text = msg.get("content") or ""
        raw = {"role": "assistant", "content": text}
        if msg.get("tool_calls"):
            raw["tool_calls"] = msg["tool_calls"]
        return Completion(
            text=text,
            tool_calls=calls,
            stop_reason=stop,
            usage=Usage(
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                int(details.get("cached_tokens", 0) or 0),
            ),
            model=str(response.get("model", self.model)),
            message=Message(role="assistant", text=text, tool_calls=calls, raw=raw),
        )

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool = False,
        force_tool: str | None = None,
        max_output_tokens: int = 2048,
    ) -> Completion:
        body = self.request_body(
            messages,
            tools,
            require_tool=require_tool,
            force_tool=force_tool,
            max_output_tokens=max_output_tokens,
        )
        key = self.key or api_key("GROQ_API_KEY")
        try:
            response = self.transport.post(self.url, {"Authorization": f"Bearer {key}"}, body)
        except ProviderError as error:
            # With tool_choice "required", Groq rejects a turn in which the model wrote text
            # instead of calling a tool (HTTP 400 tool_use_failed). That is model behaviour, not a
            # transport failure: report a turn without tool calls and let the agent ask again.
            if error.status == 400 and "tool_use_failed" in str(error):
                empty = Message(role="assistant", text="")
                return Completion("", (), "other", Usage(), self.model, empty)
            raise
        return self.parse(response)


# --- Gemini (generateContent) --------------------------------------------------------------------


@dataclass
class GeminiModel:
    model: str = "gemini-3.8-flash"
    transport: Transport = field(default_factory=HttpTransport)
    key: str | None = None
    provider: str = "gemini"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.model}:generateContent"

    def request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool,
        force_tool: str | None = None,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        system = [m.text for m in messages if m.role == "system"]
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "assistant":
                if m.raw is not None:
                    contents.append(m.raw)
                    continue
                parts: list[dict[str, Any]] = [{"text": m.text}] if m.text else []
                parts += [
                    {"functionCall": {"name": c.name, "args": c.arguments}} for c in m.tool_calls
                ]
                contents.append({"role": "model", "parts": parts})
            elif m.role == "tool":
                part = {"functionResponse": {"name": m.tool_name, "response": {"result": m.text}}}
                # Consecutive tool results belong in one user turn, as Gemini expects.
                if (
                    contents
                    and contents[-1]["role"] == "user"
                    and all("functionResponse" in p for p in contents[-1]["parts"])
                ):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            else:
                contents.append({"role": "user", "parts": [{"text": m.text}]})
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in tools
                    ]
                }
            ]
            calling: dict[str, Any] = {"mode": "ANY" if require_tool or force_tool else "AUTO"}
            if force_tool:
                calling["allowedFunctionNames"] = [force_tool]
            body["toolConfig"] = {"functionCallingConfig": calling}
        return body

    def parse(self, response: dict[str, Any]) -> Completion:
        candidates = response.get("candidates") or []
        usage_meta = response.get("usageMetadata") or {}
        usage = Usage(
            int(usage_meta.get("promptTokenCount", 0)),
            int(usage_meta.get("candidatesTokenCount", 0))
            + int(usage_meta.get("thoughtsTokenCount", 0)),
            int(usage_meta.get("cachedContentTokenCount", 0)),
        )
        if not candidates:
            block = (response.get("promptFeedback") or {}).get("blockReason", "no candidates")
            # The prompt itself was blocked; there is no model turn to keep.
            message = Message(role="assistant", text="")
            return Completion(f"[blocked: {block}]", (), "refusal", usage, self.model, message)
        candidate = candidates[0]
        content = candidate.get("content") or {"role": "model", "parts": []}
        parts = content.get("parts") or []
        text = "".join(str(p["text"]) for p in parts if "text" in p and not p.get("thought"))
        calls = tuple(
            ToolCall(
                id=f"call_{i}",
                name=str(p["functionCall"]["name"]),
                arguments=_parse_arguments(p["functionCall"].get("args")),
            )
            for i, p in enumerate(parts)
            if "functionCall" in p
        )
        finish = str(candidate.get("finishReason", ""))
        stop: StopReason
        if calls:
            stop = "tool_use"
        elif finish == "STOP":
            stop = "end"
        elif finish == "MAX_TOKENS":
            stop = "max_tokens"
        elif finish in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
            stop = "refusal"
        else:
            stop = "other"
        raw = {"role": "model", "parts": parts}
        return Completion(
            text=text,
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            model=str(response.get("modelVersion", self.model)),
            message=Message(role="assistant", text=text, tool_calls=calls, raw=raw),
        )

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        require_tool: bool = False,
        force_tool: str | None = None,
        max_output_tokens: int = 2048,
    ) -> Completion:
        body = self.request_body(
            messages,
            tools,
            require_tool=require_tool,
            force_tool=force_tool,
            max_output_tokens=max_output_tokens,
        )
        key = self.key or api_key("GEMINI_API_KEY")
        return self.parse(self.transport.post(self.url, {"x-goog-api-key": key}, body))


def default_transport() -> Transport:
    return RetryingTransport(HttpTransport())


def make_model(
    provider: str, model: str | None = None, transport: Transport | None = None
) -> ChatModel:
    t = transport if transport is not None else default_transport()  # an empty Cassette is falsy
    if provider == "groq":
        return GroqModel(model=model or GroqModel.model, transport=t)
    if provider == "gemini":
        return GeminiModel(model=model or GeminiModel.model, transport=t)
    raise ValueError(f"unknown provider {provider!r} (expected 'gemini' or 'groq')")
