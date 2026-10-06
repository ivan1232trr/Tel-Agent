"""Mocked public Responses transport: no real token or inference request is used."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from agent.providers.llm import chatgpt_plan as module
from agent.providers.llm.base import TextDelta, ToolCall
from agent.providers.llm.chatgpt_plan import ChatGPTPlanError, ChatGPTPlanLLM
from agent.tools import TAKE_MESSAGE

TOKEN = "test-oauth-access-token"  # noqa: S105 - synthetic fixture


async def _token() -> str:
    return TOKEN


def _event(kind: str, **values) -> bytes:
    return ("data: " + json.dumps({"type": kind, **values}) + "\n\n").encode()


def _complete(*output) -> bytes:
    return _event(
        "response.completed", response={"status": "completed", "output": list(output)}
    )


def _call(**values) -> dict:
    return {
        "type": "function_call",
        "id": "fc_server_item",
        "call_id": "call_stable",
        "name": TAKE_MESSAGE.name,
        "namespace": module.TOOL_NAMESPACE,
        "arguments": '{"name":"Pat","reason":"A question"}',
        "status": "completed",
        **values,
    }


class RecordingStream(httpx.AsyncByteStream):
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.closed = False
        self.read = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.read += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def _collect(body: bytes, *, tools=None):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        return [
            event
            async for event in ChatGPTPlanLLM("account-model", _token, client=client).stream(
                [{"role": "user", "content": "hello"}], tools
            )
        ]


async def test_request_uses_public_route_oauth_history_and_namespaced_functions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-key")
    monkeypatch.setenv("LLM_API_KEY", "another-unused-key")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=_complete())

    messages = [
        {"role": "system", "content": "Keep it short."},
        {"role": "user", "content": "Take a message."},
        {
            "role": "assistant",
            "content": "I'll note that.",
            "tool_calls": [
                {
                    "id": "call_from_history",
                    "type": "function",
                    "function": {"name": TAKE_MESSAGE.name, "arguments": '{"name":"Pat"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_from_history", "content": "Saved."},
    ]
    original = copy.deepcopy(messages)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ChatGPTPlanLLM("account-model", _token, client=client)
        assert [event async for event in provider.stream(messages, [TAKE_MESSAGE])] == []
        assert not client.is_closed
    assert messages == original
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["accept"] == "text/event-stream"
    body = json.loads(request.content)
    assert set(body) == {"model", "input", "store", "stream", "tools", "include"}
    assert body["model"] == "account-model"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["input"] == [
        {"role": "developer", "content": "Keep it short."},
        {"role": "user", "content": "Take a message."},
        {"role": "assistant", "content": "I'll note that."},
        {
            "type": "function_call",
            "call_id": "call_from_history",
            "namespace": "tel_agent",
            "name": TAKE_MESSAGE.name,
            "arguments": '{"name":"Pat"}',
        },
        {"type": "function_call_output", "call_id": "call_from_history", "output": "Saved."},
    ]
    namespace = body["tools"][0]
    assert namespace["type"] == "namespace"
    assert namespace["name"] == "tel_agent"
    assert namespace["tools"] == [
        {
            "type": "function",
            "name": TAKE_MESSAGE.name,
            "description": TAKE_MESSAGE.description,
            "parameters": TAKE_MESSAGE.parameters,
            "strict": False,
        }
    ]


async def test_current_token_is_requested_for_each_inference():
    count = 0
    headers = []

    async def current():
        nonlocal count
        count += 1
        return f"refreshed-{count}"

    def handler(request):
        headers.append(request.headers["authorization"])
        assert "tools" not in json.loads(request.content)
        return httpx.Response(200, content=_complete())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ChatGPTPlanLLM("model", current, client=client)
        for _ in range(2):
            assert [event async for event in provider.stream([])] == []
    assert headers == ["Bearer refreshed-1", "Bearer refreshed-2"]


async def test_text_is_not_buffered_and_aclose_closes_response_immediately():
    source = RecordingStream(
        [_event("response.output_text.delta", delta="one"), _complete(), b"unread"]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    ) as client:
        stream = ChatGPTPlanLLM("model", _token, client=client).stream([])
        assert await anext(stream) == TextDelta("one")
        assert source.read == 1
        await stream.aclose()
        assert source.closed
        assert source.read == 1
        assert not client.is_closed


async def test_task_cancellation_closes_the_upstream_response():
    waiting = asyncio.Event()

    class WaitingStream(RecordingStream):
        async def __aiter__(self):
            waiting.set()
            await asyncio.Event().wait()
            yield b"never"

    source = WaitingStream([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    ) as client:
        stream = ChatGPTPlanLLM("model", _token, client=client).stream([])
        pending = asyncio.create_task(anext(stream))
        await waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert source.closed
        assert not client.is_closed


async def test_fragmented_multiline_sse_handles_comments_crlf_and_utf8():
    body = (
        b": heartbeat\r\n\r\nevent: response.output_text.delta\r\n"
        b'data: {"type":"response.output_text.delta",\r\n'
        + 'data: "delta":"Grüße"}\r\n\r\n'.encode()
        + _event("response.refusal.delta", delta="Sorry.")
        + _complete()
    )
    source = RecordingStream([body[index : index + 1] for index in range(len(body))])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    ) as client:
        events = [
            event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])
        ]
    assert events == [TextDelta("Grüße"), TextDelta("Sorry.")]
    assert source.closed


async def test_only_complete_calls_are_released_and_call_id_is_not_item_id():
    body = (
        _event("response.output_item.added", output_index=0, item=_call(arguments=""))
        + _event(
            "response.function_call_arguments.delta", delta='{"na', item_id="fc_server_item"
        )
        + _event("response.output_text.delta", delta="Checking.")
        + _event("response.function_call_arguments.done", arguments=_call()["arguments"])
        + _event("response.output_item.done", output_index=0, item=_call())
        + _complete(_call(), _call(call_id="second_call", id="second_item"))
    )
    events = await _collect(body, tools=[TAKE_MESSAGE])
    assert events == [
        TextDelta("Checking."),
        ToolCall(id="call_stable", name=TAKE_MESSAGE.name, arguments=_call()["arguments"]),
        ToolCall(id="second_call", name=TAKE_MESSAGE.name, arguments=_call()["arguments"]),
    ]


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", "error"])
async def test_failure_after_text_and_tool_deltas_never_releases_a_call(terminal):
    code = "subscription_sharing_usage_limit_exceeded"
    failure = (
        {"code": code, "message": f"Private prompt and {TOKEN}"}
        if terminal == "error"
        else {"response": {"error": {"code": code, "message": f"Private prompt and {TOKEN}"}}}
    )
    source = RecordingStream(
        [
            _event("response.output_text.delta", delta="Checking."),
            _event("response.output_item.done", item=_call()),
            _event(terminal, **failure),
            _complete(_call()),
        ]
    )
    collected = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"x-request-id": "req_safe"}, stream=source
            )
        )
    ) as client:
        with pytest.raises(ChatGPTPlanError) as caught:
            async for event in ChatGPTPlanLLM("m", _token, client=client).stream(
                [], [TAKE_MESSAGE]
            ):
                collected.append(event)
    assert collected == [TextDelta("Checking.")]
    assert caught.value.code == code
    assert caught.value.status_code == 200
    assert caught.value.request_id == "req_safe"
    assert TOKEN not in str(caught.value)
    assert "Private" not in str(caught.value)
    assert source.closed
    assert source.read == 3


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"", "stream_interrupted"),
        (b"data: [DONE]\n\n", "stream_interrupted"),
        (b"data: not-json\n\n", "invalid_stream"),
        (b"data: []\n\n", "invalid_stream"),
        (b"data: {}\n\n", "invalid_stream"),
        (b"data: \xff\n\n", "invalid_stream"),
        (_complete().rstrip(), "stream_interrupted"),
        (
            _event("response.completed", response={"status": "incomplete", "output": []}),
            "invalid_completion",
        ),
        (_event("response.completed", response={"status": "completed"}), "invalid_completion"),
        (_event("response.output_text.delta", delta=23), "invalid_stream"),
        (_event("response.output_item.done", item=_call()), "stream_interrupted"),
    ],
)
async def test_malformed_truncated_and_non_successful_streams_fail_closed(body, code):
    with pytest.raises(ChatGPTPlanError) as caught:
        await _collect(body, tools=[TAKE_MESSAGE])
    assert caught.value.code == code


@pytest.mark.parametrize(
    "change",
    [
        {"call_id": ""},
        {"call_id": None},
        {"namespace": "foreign"},
        {"namespace": None},
        {"name": "not_offered"},
        {"arguments": "{broken"},
        {"arguments": "[]"},
        {"arguments": None},
        {"status": "incomplete"},
    ],
)
async def test_all_calls_are_validated_before_even_the_first_is_released(change):
    collected = []
    body = _complete(
        _call(), _call(call_id="second", **{k: v for k, v in change.items() if k != "call_id"})
    )
    if "call_id" in change:
        body = _complete(_call(), _call(**change))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(ChatGPTPlanError):
            async for event in ChatGPTPlanLLM("m", _token, client=client).stream(
                [], [TAKE_MESSAGE]
            ):
                collected.append(event)
    assert collected == []


async def test_duplicate_call_ids_are_rejected():
    with pytest.raises(ChatGPTPlanError, match="invalid_tool_call"):
        await _collect(_complete(_call(), _call()), tools=[TAKE_MESSAGE])


async def test_unoffered_tools_are_rejected():
    with pytest.raises(ChatGPTPlanError, match="invalid_tool_call"):
        await _collect(_complete(_call()))


@pytest.mark.parametrize("status", [401, 403, 429, 503])
async def test_http_failures_preserve_safe_diagnostics_without_body_content(status):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                status,
                headers={"x-request-id": "req_public_diagnostic"},
                json={
                    "error": {
                        "code": "subscription_sharing_usage_unavailable",
                        "message": TOKEN,
                    }
                },
            )
        )
    ) as client:
        with pytest.raises(ChatGPTPlanError) as caught:
            [event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])]
    assert caught.value.code == "subscription_sharing_usage_unavailable"
    assert caught.value.status_code == status
    assert caught.value.request_id == "req_public_diagnostic"
    assert TOKEN not in str(caught.value)
    assert not hasattr(caught.value, "response")
    assert not hasattr(caught.value, "request")


@pytest.mark.parametrize(
    "body", [{"detail": TOKEN}, {"error": TOKEN}, [TOKEN], {"error": {"code": TOKEN}}]
)
async def test_unstructured_errors_and_credential_echoes_are_redacted(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(403, headers={"x-request-id": TOKEN}, json=body)
        )
    ) as client:
        with pytest.raises(ChatGPTPlanError) as caught:
            [event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])]
    assert caught.value.code == "http_error"
    assert caught.value.request_id is None
    assert TOKEN not in str(caught.value)


async def test_redirects_are_not_followed_even_with_a_redirecting_injected_client():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": "https://elsewhere.test/collect"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        with pytest.raises(ChatGPTPlanError) as caught:
            [event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])]
    assert caught.value.status_code == 307
    assert len(seen) == 1


async def test_transport_exceptions_are_sanitized_without_retry():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadError(f"Private prompt {TOKEN}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ChatGPTPlanError) as caught:
            [event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])]
    assert caught.value.code == "transport_error"
    assert TOKEN not in str(caught.value)
    assert caught.value.__suppress_context__
    assert count == 1


@pytest.mark.parametrize("limit", ["MAX_STREAM_BYTES", "MAX_EVENT_BYTES", "MAX_ERROR_BYTES"])
async def test_oversized_streams_events_and_error_bodies_are_bounded(monkeypatch, limit):
    monkeypatch.setattr(module, limit, 32)
    source = RecordingStream([b"data: " + b"x" * 40, b"never-read"])
    status = 403 if limit == "MAX_ERROR_BYTES" else 200
    expected = {
        "MAX_STREAM_BYTES": "stream_limit_exceeded",
        "MAX_EVENT_BYTES": "event_limit_exceeded",
        "MAX_ERROR_BYTES": "http_error",
    }[limit]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, stream=source))
    ) as client:
        with pytest.raises(ChatGPTPlanError, match=expected):
            [event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])]
    assert source.closed
    assert source.read == 1


@pytest.mark.parametrize("limit", ["MAX_ARGUMENT_BYTES", "MAX_TOOL_CALLS"])
async def test_tool_call_resources_are_bounded(monkeypatch, limit):
    monkeypatch.setattr(module, limit, 1)
    with pytest.raises(ChatGPTPlanError, match="tool_call_limit_exceeded"):
        await _collect(_complete(_call(), _call(call_id="second")), tools=[TAKE_MESSAGE])


@pytest.mark.parametrize(
    "token", ["", "contains a space", "contains\nnewline", "not-ascii-ü", None]
)
async def test_missing_or_unsafe_tokens_fail_before_any_request(token):
    async def invalid():
        return token

    def forbidden(request):
        pytest.fail("No request may be made without a usable OAuth token")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(ChatGPTPlanError, match="authentication_unavailable"):
            [event async for event in ChatGPTPlanLLM("m", invalid, client=client).stream([])]


async def test_reply_loop_preserves_reasoning_and_namespace_without_shared_state():
    from agent.reply import reply
    from agent.tools import Tool

    executed = []
    requests = []
    reasoning = {
        "type": "reasoning",
        "id": "rs_opaque",
        "summary": [],
        "encrypted_content": "opaque",
    }
    output_message = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Checking.", "annotations": []}],
    }
    complete_call = _call()
    output = [reasoning, output_message, complete_call]

    async def run(arguments):
        executed.append(arguments)
        return "Saved."

    tool = Tool(TAKE_MESSAGE.name, TAKE_MESSAGE.description, TAKE_MESSAGE.parameters, run)

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                content=_event("response.output_text.delta", delta="Checking.")
                + _complete(*output),
            )
        return httpx.Response(
            200, content=_event("response.output_text.delta", delta="Done.") + _complete()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = ChatGPTPlanLLM("m", _token, client=client)
        answer = "".join(
            [part async for part in reply("Take a message", provider=provider, tools=[tool])]
        )
        assert answer == "Checking.Done."
        assert len(executed) == 1
        assert requests[1]["input"][-4:-1] == output
        assert requests[1]["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_stable",
            "output": "Saved.",
        }
        # Even reusing this instance cannot leak the previous conversation's context.
        [
            event
            async for event in provider.stream(
                [{"role": "user", "content": "New conversation"}]
            )
        ]
        assert requests[2]["input"] == [{"role": "user", "content": "New conversation"}]


async def test_failed_reply_never_executes_tool_even_after_output_item_done():
    from agent.reply import reply
    from agent.tools import Tool

    executed = []

    async def run(arguments):
        executed.append(arguments)
        return "Saved."

    tool = Tool(TAKE_MESSAGE.name, TAKE_MESSAGE.description, TAKE_MESSAGE.parameters, run)
    body = _event("response.output_item.done", item=_call()) + _event(
        "response.failed",
        response={"error": {"code": "subscription_sharing_usage_unavailable"}},
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        provider = ChatGPTPlanLLM("m", _token, client=client)
        with pytest.raises(ChatGPTPlanError):
            [part async for part in reply("Take a message", provider=provider, tools=[tool])]
    assert executed == []


async def test_auth_errors_keep_safe_recovery_code_and_never_use_an_api_key(monkeypatch):
    from agent.chatgpt_auth import ChatGPTAuthError

    monkeypatch.setenv("LLM_API_KEY", "should-not-be-used")

    async def expired():
        raise ChatGPTAuthError("invalid_grant", "Sign in again")

    def forbidden(request):
        pytest.fail("An auth error must never invoke another billing path")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(ChatGPTPlanError) as caught:
            [event async for event in ChatGPTPlanLLM("m", expired, client=client).stream([])]
    assert caught.value.code == "invalid_grant"


async def test_unexpected_auth_exception_text_is_not_exposed():
    async def broken():
        raise RuntimeError(f"Private diagnostic {TOKEN}")

    with pytest.raises(ChatGPTPlanError) as caught:
        [event async for event in ChatGPTPlanLLM("m", broken).stream([])]
    assert caught.value.code == "authentication_unavailable"
    assert TOKEN not in str(caught.value)
    assert caught.value.__suppress_context__


async def test_owned_client_is_closed_when_the_consumer_stops(monkeypatch):
    source = RecordingStream([_event("response.output_text.delta", delta="first"), _complete()])
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    )
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client)
    stream = ChatGPTPlanLLM("m", _token).stream([])
    assert await anext(stream) == TextDelta("first")
    await stream.aclose()
    assert client.is_closed
    assert source.closed


async def test_overall_stream_deadline_closes_even_a_stream_of_keepalives(monkeypatch):
    monkeypatch.setattr(module, "MAX_STREAM_SECONDS", 0)
    source = RecordingStream([b": still waiting\n\n"] * 10)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    ) as client:
        with pytest.raises(ChatGPTPlanError, match="stream_timeout"):
            [event async for event in ChatGPTPlanLLM("m", _token, client=client).stream([])]
    assert source.closed
    assert source.read == 0


@pytest.mark.parametrize("item", [{"type": "tool_search_call"}, [], {"type": []}])
async def test_unsupported_or_invalid_completed_output_is_not_success(item):
    with pytest.raises(ChatGPTPlanError):
        await _collect(_complete(item))


async def test_non_json_numbers_in_arguments_are_not_released():
    with pytest.raises(ChatGPTPlanError, match="invalid_tool_arguments"):
        await _collect(_complete(_call(arguments='{"value":NaN}')), tools=[TAKE_MESSAGE])


async def test_incomplete_without_error_code_has_specific_failure():
    with pytest.raises(ChatGPTPlanError, match="response_incomplete"):
        await _collect(_event("response.incomplete", response={"status": "incomplete"}))
