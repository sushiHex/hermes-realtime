from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from typing import cast
from urllib.request import ProxyHandler, Request

import pytest

import hermes_realtime.providers.ollama as ollama_module
from hermes_realtime.conversation import (
    ActiveTaskSummary,
    ConversationInferenceRequest,
    ConversationMessage,
)
from hermes_realtime.providers import OllamaStreamingInference


class LineResponse:
    def __init__(self, payloads: tuple[dict[str, object], ...]) -> None:
        self._lines = deque(
            json.dumps(payload).encode("utf-8") + b"\n" for payload in payloads
        )
        self.closed = False

    def readline(self, limit: int = -1, /) -> bytes:
        if self.closed or not self._lines:
            return b""
        return self._lines.popleft()

    def close(self) -> None:
        self.closed = True


def _snapshot() -> ConversationInferenceRequest:
    return ConversationInferenceRequest(
        revision=3,
        messages=(
            ConversationMessage(role="user", text="What changed?"),
            ConversationMessage(role="assistant", text="The bridge is complete."),
        ),
        active_tasks=(
            ActiveTaskSummary(task_id="task_review", objective="Review the release"),
        ),
        updates=(),
    )


@pytest.mark.asyncio
async def test_ollama_streams_bounded_speakable_segments_without_tool_schema() -> None:
    requests: list[tuple[Request, float]] = []
    response = LineResponse(
        (
            {"message": {"content": "First sentence. Second"}, "done": False},
            {"message": {"content": " sentence!"}, "done": False},
            {"message": {"content": " Final fragment"}, "done": True},
        )
    )

    def open_request(request: Request, timeout: float) -> LineResponse:
        requests.append((request, timeout))
        return response

    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="hermes-4.3-36b-iq4xs-16k:latest",
        open_request=open_request,
        max_segment_chars=64,
    )

    segments = [
        segment
        async for segment in inference.stream(_snapshot(), turn_id="turn_1")
    ]

    assert segments == ["First sentence.", "Second sentence!", "Final fragment"]
    assert response.closed
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.data is not None
    body = json.loads(cast(bytes, request.data))
    assert timeout == 30.0
    assert body["model"] == "hermes-4.3-36b-iq4xs-16k:latest"
    assert body["stream"] is True
    assert body["messages"] == [
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "The bridge is complete."},
        {
            "role": "system",
            "content": (
                "Active background work:\n"
                "- task_review: Review the release"
            ),
        },
    ]
    assert "tools" not in body
    assert "tool_choice" not in body


def test_ollama_keeps_ordered_list_markers_with_their_items() -> None:
    inference = object.__new__(OllamaStreamingInference)
    inference._max_segment_chars = 4096
    source = "1. First item remains whole. 2. Second item remains whole."

    segments, remaining = inference._extract_segments(source, final=True)

    assert segments == ["1. First item remains whole.", "2. Second item remains whole."]
    assert remaining == ""


def test_ollama_buffers_an_incremental_ordered_list_marker() -> None:
    inference = object.__new__(OllamaStreamingInference)
    inference._max_segment_chars = 4096

    assert inference._extract_segments("1. ", final=False) == ([], "1. ")


def test_ollama_does_not_split_inside_markdown_emphasis() -> None:
    inference = object.__new__(OllamaStreamingInference)
    inference._max_segment_chars = 4096
    source = "Try *The Mitchells vs. the Machines*, then *Holes*."

    assert inference._extract_segments(source, final=True) == ([source], "")


def test_ollama_buffers_sentence_punctuation_inside_open_emphasis() -> None:
    inference = object.__new__(OllamaStreamingInference)
    inference._max_segment_chars = 4096
    source = "Try *The Mitchells vs. "

    assert inference._extract_segments(source, final=False) == ([], source)


def test_ollama_rejects_non_loopback_endpoint_before_request() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OllamaStreamingInference(
            base_url="http://example.com:11434",
            model="model",
        )


def test_ollama_urlopen_explicitly_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = LineResponse(())
    handlers: list[object] = []

    class Opener:
        def open(self, request: Request, timeout: float) -> LineResponse:
            del request, timeout
            return response

    def build_opener(*selected: object) -> Opener:
        handlers.extend(selected)
        return Opener()

    monkeypatch.setattr(ollama_module, "build_opener", build_opener, raising=False)
    request = Request("http://127.0.0.1:11434/api/chat")

    assert OllamaStreamingInference._urlopen(request, 1.0) is response
    proxy_handlers = [handler for handler in handlers if isinstance(handler, ProxyHandler)]
    assert len(proxy_handlers) == 1
    assert vars(proxy_handlers[0])["proxies"] == {}
    assert [type(handler).__name__ for handler in handlers].count("_NoRedirectHandler") == 1


@pytest.mark.asyncio
async def test_ollama_retains_and_closes_response_when_cancelled_during_open() -> None:
    open_started = threading.Event()
    release_open = threading.Event()
    response = LineResponse(
        ({"message": {"content": "late"}, "done": True},)
    )

    def open_request(request: Request, timeout: float) -> LineResponse:
        del request, timeout
        open_started.set()
        assert release_open.wait(timeout=5)
        return response

    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="model",
        open_request=open_request,
    )

    async def consume() -> list[str]:
        return [
            segment
            async for segment in inference.stream(_snapshot(), turn_id="turn_open")
        ]

    stream = asyncio.create_task(consume())
    assert await asyncio.to_thread(open_started.wait, 1)
    stream.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stream

    cleanup = asyncio.create_task(inference.cancel("turn_open"))
    await asyncio.sleep(0)
    assert not cleanup.done()
    assert not response.closed
    release_open.set()
    await asyncio.wait_for(cleanup, timeout=1)

    assert response.closed
    await inference.close()


@pytest.mark.asyncio
async def test_ollama_cancel_waits_for_owned_blocked_reader_thread() -> None:
    read_started = threading.Event()
    close_called = threading.Event()
    allow_read_finish = threading.Event()

    class BlockingReadResponse(LineResponse):
        def readline(self, limit: int = -1, /) -> bytes:
            read_started.set()
            assert close_called.wait(timeout=5)
            assert allow_read_finish.wait(timeout=5)
            return b""

        def close(self) -> None:
            super().close()
            close_called.set()

    response = BlockingReadResponse(())
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="model",
        open_request=lambda _request, _timeout: response,
    )

    async def consume() -> list[str]:
        return [
            segment
            async for segment in inference.stream(_snapshot(), turn_id="turn_read")
        ]

    stream = asyncio.create_task(consume())
    assert await asyncio.to_thread(read_started.wait, 1)
    cancelling = asyncio.create_task(inference.cancel("turn_read"))
    assert await asyncio.to_thread(close_called.wait, 1)
    await asyncio.sleep(0)
    assert not cancelling.done()

    allow_read_finish.set()
    await asyncio.wait_for(cancelling, timeout=1)
    assert await asyncio.wait_for(stream, timeout=1) == []
    await inference.close()


@pytest.mark.asyncio
async def test_ollama_cancel_ignores_reader_failure_caused_by_response_close() -> None:
    second_read_started = threading.Event()
    close_called = threading.Event()
    first_segment = asyncio.Event()

    class CloseRaceResponse(LineResponse):
        def readline(self, limit: int = -1, /) -> bytes:
            if self._lines:
                return self._lines.popleft()
            second_read_started.set()
            assert close_called.wait(timeout=5)
            raise AttributeError("closed response file pointer")

        def close(self) -> None:
            super().close()
            close_called.set()

    responses = deque(
        (
            CloseRaceResponse(
                ({"message": {"content": "First sentence."}, "done": False},)
            ),
            LineResponse(
                ({"message": {"content": "Recovered."}, "done": True},)
            ),
        )
    )
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="model",
        open_request=lambda _request, _timeout: responses.popleft(),
    )

    async def consume_first() -> list[str]:
        segments = []
        async for segment in inference.stream(_snapshot(), turn_id="turn_close_race"):
            segments.append(segment)
            first_segment.set()
        return segments

    stream = asyncio.create_task(consume_first())
    await asyncio.wait_for(first_segment.wait(), timeout=1)
    assert await asyncio.to_thread(second_read_started.wait, 1)
    stream.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stream
    await asyncio.wait_for(inference.cancel("turn_close_race"), timeout=1)

    recovered = [
        segment
        async for segment in inference.stream(_snapshot(), turn_id="turn_recovered")
    ]
    assert recovered == ["Recovered."]
    await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["cancel", "close"])
async def test_ollama_cleanup_survives_caller_cancellation(
    operation_name: str,
) -> None:
    read_started = threading.Event()
    close_started = threading.Event()
    allow_close = threading.Event()
    release_read = threading.Event()

    class BlockingCleanupResponse(LineResponse):
        close_calls = 0

        def readline(self, limit: int = -1, /) -> bytes:
            read_started.set()
            assert release_read.wait(timeout=5)
            return b""

        def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            assert allow_close.wait(timeout=5)
            super().close()
            release_read.set()

    response = BlockingCleanupResponse(())
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="model",
        open_request=lambda _request, _timeout: response,
    )
    turn_id = f"turn_{operation_name}_owner"

    async def consume() -> list[str]:
        return [
            segment
            async for segment in inference.stream(_snapshot(), turn_id=turn_id)
        ]

    async def cleanup() -> None:
        if operation_name == "cancel":
            await inference.cancel(turn_id)
        else:
            await inference.close()

    stream = asyncio.create_task(consume())
    assert await asyncio.to_thread(read_started.wait, 1)
    caller = asyncio.create_task(cleanup())
    assert await asyncio.to_thread(close_started.wait, 1)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    retry = asyncio.create_task(cleanup())
    await asyncio.sleep(0.05)
    close_calls_before_release = response.close_calls
    allow_close.set()
    await asyncio.wait_for(retry, timeout=1)
    assert await asyncio.wait_for(stream, timeout=1) == []
    assert close_calls_before_release == 1
    assert response.close_calls == 1
    await inference.close()


@pytest.mark.asyncio
async def test_ollama_failed_response_close_remains_owned_across_retries() -> None:
    read_started = threading.Event()
    release_read = threading.Event()

    class FailingCloseResponse(LineResponse):
        close_calls = 0

        def readline(self, limit: int = -1, /) -> bytes:
            read_started.set()
            assert release_read.wait(timeout=5)
            return b""

        def close(self) -> None:
            self.close_calls += 1
            release_read.set()
            raise OSError("response close failed")

    response = FailingCloseResponse(())
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="model",
        open_request=lambda _request, _timeout: response,
    )

    async def consume() -> list[str]:
        return [
            segment
            async for segment in inference.stream(_snapshot(), turn_id="turn_close_fail")
        ]

    stream = asyncio.create_task(consume())
    assert await asyncio.to_thread(read_started.wait, 1)
    with pytest.raises(OSError, match="response close failed"):
        await inference.cancel("turn_close_fail")
    with pytest.raises(OSError, match="response close failed"):
        await stream
    with pytest.raises(OSError, match="response close failed"):
        await inference.cancel("turn_close_fail")
    with pytest.raises(OSError, match="response close failed"):
        await inference.close()
    with pytest.raises(OSError, match="response close failed"):
        await inference.close()
    assert response.close_calls == 1
