"""Streaming Responses adapter for explicitly authorized ChatGPT plan usage.

The OAuth owner supplies a current access token for every request. This adapter has
no API-key fallback and never uses a private ChatGPT endpoint. The public direct
route requires full input history, developer instructions, namespaced functions,
``store=False`` and ``stream=True``. See the Sign in with ChatGPT model/inference
and preview-limitations guides at https://developers.openai.com/siwc/.

Text streams immediately. Function calls are read from the authoritative completed
response, never from partial argument deltas, and are released only after every
call has been checked. A failed or interrupted response cannot execute a tool.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, aclosing
from typing import Any, cast

import httpx

from agent.providers.llm.base import Event, Message, TextDelta, ToolCall
from agent.tools import Tool

RESPONSES_URL = "https://api.openai.com/v1/responses"
TOOL_NAMESPACE = "tel_agent"
TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)
MAX_STREAM_SECONDS = 300.0
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024
MAX_TOOL_CALLS = 64
MAX_ARGUMENT_BYTES = 256 * 1024


class ChatGPTPlanError(RuntimeError):
    """A safe diagnostic, without upstream message bodies or bearer credentials."""

    def __init__(
        self, code: str, *, status_code: int | None = None, request_id: str | None = None
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        detail = f"ChatGPT plan request failed: {code}"
        if status_code is not None:
            detail += f" (HTTP {status_code})"
        if request_id is not None:
            detail += f" [request {request_id}]"
        super().__init__(detail)


class ChatGPTPlanLLM:
    """Streams text and completed tool calls using an OAuth token supplier."""

    def __init__(
        self,
        model: str,
        token_supplier: Callable[[], Awaitable[str]],
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("A ChatGPT plan model is required")
        self._model = model
        self._token_supplier = token_supplier
        # The caller retains ownership of an injected client.
        self._client = client

    async def stream(
        self, messages: list[Message], tools: Sequence[Tool] | None = None
    ) -> AsyncGenerator[Event, None]:
        """Yield text promptly; release calls only after successful completion."""
        offered = list(tools or ())
        payload: dict[str, Any] = {
            "model": self._model,
            "input": _input(messages),
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if offered:
            payload["tools"] = [
                {
                    "type": "namespace",
                    "name": TOOL_NAMESPACE,
                    "description": "Tools available to this Tel-Agent conversation.",
                    "tools": [
                        {
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                            # Preserve existing schemas with optional properties.
                            "strict": False,
                        }
                        for tool in offered
                    ],
                }
            ]
        try:
            token = await self._token_supplier()
        except Exception as exc:
            # Import only on this error path; keep auth recovery out of this adapter.
            from agent.chatgpt_auth import ChatGPTAuthError

            code = _safe_value(exc.code) if isinstance(exc, ChatGPTAuthError) else None
            raise ChatGPTPlanError(code or "authentication_unavailable") from None
        if (
            not isinstance(token, str)
            or len(token) > 16384
            or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token)
        ):
            raise ChatGPTPlanError("authentication_unavailable")

        async with AsyncExitStack() as stack:
            client = self._client
            if client is None:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False, trust_env=False)
                )
            # Explicitly close nested generators on aclose(), as well as task cancel.
            reader = self._read(client, payload, token, {tool.name for tool in offered})
            async with aclosing(reader):
                async for event in reader:
                    yield event

    async def _read(
        self, client: httpx.AsyncClient, payload: dict[str, Any], token: str, names: set[str]
    ) -> AsyncGenerator[Event, None]:
        request_id: str | None = None
        status_code: int | None = None
        try:
            async with client.stream(
                "POST",
                RESPONSES_URL,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
                timeout=TIMEOUT,
                follow_redirects=False,
            ) as response:
                status_code = response.status_code
                request_id = _safe_value(
                    response.headers.get("x-request-id"), token, code=False
                )
                if not 200 <= status_code < 300:
                    http_code = await _http_error(response, token)
                    raise ChatGPTPlanError(
                        http_code, status_code=status_code, request_id=request_id
                    )

                async with aclosing(_events(response)) as events:
                    async for event in events:
                        kind = event.get("type")
                        if kind in {"response.failed", "response.incomplete", "error"}:
                            result = event.get("response")
                            detail = result if isinstance(result, dict) else event
                            error = detail.get("error")
                            code = error.get("code") if isinstance(error, dict) else None
                            if kind == "error":
                                code = event.get("code") or code
                            fallback = (
                                "response_incomplete"
                                if kind == "response.incomplete"
                                else "response_failed"
                            )
                            raise ChatGPTPlanError(
                                _safe_value(code, token) or fallback,
                                status_code=status_code,
                                request_id=request_id,
                            )
                        if kind in {"response.output_text.delta", "response.refusal.delta"}:
                            delta = event.get("delta")
                            if not isinstance(delta, str):
                                raise ChatGPTPlanError("invalid_stream")
                            if delta:
                                yield TextDelta(delta)
                        elif kind == "response.completed":
                            calls = _completed_calls(event.get("response"), names)
                            for call in calls:
                                yield call
                            return
                raise ChatGPTPlanError("stream_interrupted")
        except ChatGPTPlanError as exc:
            # Attach diagnostics to local parse/limit failures too, without chaining
            # a JSON/HTTP exception that may include content or Authorization.
            raise ChatGPTPlanError(
                exc.code,
                status_code=exc.status_code if exc.status_code is not None else status_code,
                request_id=exc.request_id or request_id,
            ) from None
        except (httpx.HTTPError, UnicodeError):
            raise ChatGPTPlanError(
                "transport_error", status_code=status_code, request_id=request_id
            ) from None


def _input(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate the existing chat history without changing it or losing call IDs.

    Every offered function lives in the same namespace, so its metadata can be
    reconstructed after the shared reply loop stores an ordinary ToolCall.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if "response_items" in message:
            items = message["response_items"]
            if role != "assistant" or not isinstance(items, list):
                raise ChatGPTPlanError("invalid_history")
            if any(not isinstance(item, dict) for item in items):
                raise ChatGPTPlanError("invalid_history")
            result.extend(items)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ChatGPTPlanError("invalid_history")
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.get("content", ""),
                }
            )
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise ChatGPTPlanError("invalid_history")
        if "content" in message:
            result.append(
                {
                    "role": "developer" if role == "system" else role,
                    "content": message["content"],
                }
            )
        calls = message.get("tool_calls") or []
        if calls and role != "assistant":
            raise ChatGPTPlanError("invalid_history")
        for call in calls:
            function = call.get("function")
            if not isinstance(function, dict):
                raise ChatGPTPlanError("invalid_history")
            call_id, name, arguments = (
                call.get("id"),
                function.get("name"),
                function.get("arguments"),
            )
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(name, str)
                or not name
                or not isinstance(arguments, str)
                or call.get("namespace", TOOL_NAMESPACE) != TOOL_NAMESPACE
            ):
                raise ChatGPTPlanError("invalid_history")
            result.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "namespace": TOOL_NAMESPACE,
                    "arguments": arguments,
                }
            )
    return result


def _completed_calls(response: Any, names: set[str]) -> list[ToolCall]:
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise ChatGPTPlanError("invalid_completion")
    output = response.get("output")
    if not isinstance(output, list):
        raise ChatGPTPlanError("invalid_completion")
    calls: list[ToolCall] = []
    identifiers: set[str] = set()
    for item in output:
        if not isinstance(item, dict):
            raise ChatGPTPlanError("invalid_completion")
        if item.get("type") in ("message", "reasoning"):
            continue
        if item.get("type") != "function_call":
            raise ChatGPTPlanError("unsupported_output")
        identifier, name, arguments = (
            item.get("call_id"),
            item.get("name"),
            item.get("arguments"),
        )
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(name, str)
            or name not in names
            or item.get("namespace") != TOOL_NAMESPACE
            or not isinstance(arguments, str)
            or item.get("status") not in (None, "completed")
        ):
            raise ChatGPTPlanError("invalid_tool_call")
        if len(arguments.encode("utf-8")) > MAX_ARGUMENT_BYTES or len(calls) >= MAX_TOOL_CALLS:
            raise ChatGPTPlanError("tool_call_limit_exceeded")
        # Arguments remain raw for the tool-specific validator, but a broken JSON
        # object must not release even an earlier, well-formed call in this response.
        try:
            parsed = json.loads(arguments, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            raise ChatGPTPlanError("invalid_tool_arguments") from None
        if not isinstance(parsed, dict):
            raise ChatGPTPlanError("invalid_tool_arguments")
        identifiers.add(identifier)
        calls.append(
            ToolCall(
                id=identifier,
                name=name,
                arguments=arguments,
                response_items=output if not calls else None,
            )
        )
    return calls


def _safe_value(value: Any, token: str | None = None, *, code: bool = True) -> str | None:
    """Keep bounded machine diagnostics, never prose or credential echoes."""
    pattern = r"[a-z][a-z0-9_]{0,95}" if code else r"[A-Za-z0-9_-]{1,128}"
    if (
        not isinstance(value, str)
        or (token and token in value)
        or not re.fullmatch(pattern, value)
    ):
        return None
    return value


async def _http_error(response: httpx.Response, token: str) -> str:
    body = bytearray()
    async for chunk in _chunks(response):
        if len(body) + len(chunk) > MAX_ERROR_BYTES:
            return "http_error"
        body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return "http_error"
    error = value.get("error") if isinstance(value, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return _safe_value(code, token) or "http_error"


async def _events(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """Parse bounded UTF-8 SSE events across arbitrary network chunk boundaries."""
    buffered = bytearray()
    data: list[bytes] = []
    event_bytes = 0
    total = 0
    async for chunk in _chunks(response):
        total += len(chunk)
        if total > MAX_STREAM_BYTES:
            raise ChatGPTPlanError("stream_limit_exceeded")
        buffered.extend(chunk)
        while (newline := buffered.find(b"\n")) >= 0:
            raw = bytes(buffered[:newline])
            del buffered[: newline + 1]
            line = raw.removesuffix(b"\r")
            event_bytes += len(raw) + 1
            if event_bytes > MAX_EVENT_BYTES:
                raise ChatGPTPlanError("event_limit_exceeded")
            if not line:
                if data:
                    yield _event(b"\n".join(data))
                data = []
                event_bytes = 0
            elif line.startswith(b"data:"):
                data.append(line[5:].removeprefix(b" "))
        if len(buffered) + event_bytes > MAX_EVENT_BYTES:
            raise ChatGPTPlanError("event_limit_exceeded")
    # SSE dispatch requires the blank line; EOF never completes a partial event.
    if buffered or data:
        raise ChatGPTPlanError("stream_interrupted")


def _event(data: bytes) -> dict[str, Any]:
    if data == b"[DONE]":
        # A Chat Completions marker is not successful Responses completion.
        raise ChatGPTPlanError("stream_interrupted")
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ChatGPTPlanError("invalid_stream") from None
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ChatGPTPlanError("invalid_stream")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite numbers are not JSON")


async def _chunks(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Bound both idle reads and an endless stream of small keepalive events.

    Deadlines surround network reads only, never a yield into caller code: an async
    generator must not leave a task-cancelling timeout active while it is suspended.
    """
    deadline = time.monotonic() + MAX_STREAM_SECONDS
    # httpx annotates this async generator as AsyncIterator, although its
    # implementation supports aclose(). Keep that cleanup explicit on cancel.
    iterator = cast(AsyncGenerator[bytes, None], response.aiter_bytes())
    async with aclosing(iterator):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChatGPTPlanError("stream_timeout")
            try:
                async with asyncio.timeout(min(20.0, remaining)):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise ChatGPTPlanError("stream_timeout") from None
            yield chunk
