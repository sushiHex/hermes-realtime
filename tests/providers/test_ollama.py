from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from typing import cast
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request

import pytest

import hermes_realtime.providers.ollama as ollama_module
from hermes_realtime.conversation import (
    ActiveTaskSummary,
    ConversationInferenceRequest,
    ConversationMessage,
)
from hermes_realtime.conversation.output_style import OUTPUT_STYLES, OutputStyleSelection
from hermes_realtime.conversation.streaming import ConversationPromptUpdate
from hermes_realtime.memory import BuiltinMemorySnapshot
from hermes_realtime.providers import OllamaStreamingInference


@pytest.mark.parametrize("style", OUTPUT_STYLES)
@pytest.mark.asyncio
async def test_output_style_and_reference_kinds_bind_to_the_sent_prompt(style, capsys) -> None:
    selection = OutputStyleSelection()
    selection.select(style)
    requests = []

    def open_request(request, timeout):
        requests.append(json.loads(request.data))
        return LineResponse(({"message": {"content": "Synthetic answer."}, "done": True},))

    snapshot = ConversationInferenceRequest(
        revision=2,
        messages=(ConversationMessage("user", "Older topic."),
                  ConversationMessage("assistant", "Earlier answer."),
                  ConversationMessage("user", "Current question.")),
        active_tasks=(ActiveTaskSummary("task_1", "Synthetic objective."),),
        memory=BuiltinMemorySnapshot(memory="Synthetic memory.", user="Synthetic user."),
        updates=(ConversationPromptUpdate(sequence=1, task_id="task_2", status="completed",
                                          text="Synthetic finding."),),
    )
    inference = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test",
                                        open_request=open_request, output_style=selection.get)
    assert [part async for part in inference.stream(snapshot, turn_id="turn_style")]
    sent = requests[0]["messages"]
    assert [row["role"] for row in sent] == ["system", "user", "assistant", "user"]
    assert sent[1:] == [{"role": row.role, "content": row.text} for row in snapshot.messages]
    assert sent[0]["content"].startswith(f"Output style: {style}.\n")
    sections = json.loads(sent[0]["content"].split("\n\nReference data:\n", 1)[1])
    assert sections["memory"]["memory"] == "Synthetic memory."
    assert sections["active_work"] == [{"task_id": "task_1", "objective": "Synthetic objective."}]
    assert sections["updates"][0]["text"] == "Synthetic finding."
    report = json.loads(capsys.readouterr().out.split("[ollama-prompt] ")[1])
    assert report["reference_kinds"] == ["memory", "active_work", "updates"]
    assert report["output_style"] == style
    assert "Synthetic" not in json.dumps(report)


def test_ollama_renders_memory_as_separate_untrusted_context() -> None:
    snapshot = ConversationInferenceRequest(
        revision=1,
        messages=(
            ConversationMessage(role="user", text="Earlier question."),
            ConversationMessage(role="assistant", text="Earlier answer."),
            ConversationMessage(role="user", text="What do I prefer?"),
        ),
        active_tasks=(),
        memory=BuiltinMemorySnapshot(memory="Prefers concise replies.", user="Ari"),
    )

    copied = OllamaStreamingInference._trusted_snapshot(snapshot)
    messages = OllamaStreamingInference._messages(copied)

    assert len(messages) == 4
    assert messages[1:] == [
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "What do I prefer?"},
    ]
    assert messages[0]["role"] == "system"
    assert "untrusted reference data" in messages[0]["content"]
    payload = messages[0]["content"].split("\n\nReference data:\n", 1)[1]
    assert json.loads(payload) == {
        "work_state": "none",
        "memory": {"memory": "Prefers concise replies.", "user": "Ari", "truncated": False},
    }



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
    assert body["messages"][1:] == [
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "The bridge is complete."},
    ]
    sections = json.loads(body["messages"][0]["content"].split("\n\nReference data:\n", 1)[1])
    assert sections["active_work"] == [
        {"task_id": "task_review", "objective": "Review the release"}
    ]
    assert "tools" not in body
    assert "tool_choice" not in body


def test_ollama_renders_the_interrupted_flag_as_a_fixed_suffix_only_when_set() -> None:
    snapshot = ConversationInferenceRequest(
        revision=2,
        messages=(
            ConversationMessage(role="assistant", text="Cut off here.", interrupted=True),
            ConversationMessage(role="assistant", text="I said [speech interrupted]"),
        ),
        active_tasks=(),
        updates=(),
    )

    assert OllamaStreamingInference._messages(snapshot)[1:] == [
        {
            "role": "assistant",
            "content": "Cut off here." + ollama_module._INTERRUPTED_SPEECH_SUFFIX,
        },
        {"role": "assistant", "content": "I said [speech interrupted]"},
    ]
    assert ollama_module._INTERRUPTED_SPEECH_SUFFIX == " [speech interrupted]"


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


@pytest.mark.asyncio
async def test_ollama_sends_an_explicit_context_window_and_reports_the_prompt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bodies: list[dict[str, object]] = []

    def open_request(request: Request, timeout: float) -> LineResponse:
        del timeout
        bodies.append(json.loads(cast(bytes, request.data)))
        return LineResponse(
            (
                {"message": {"content": "Short reply."}, "done": False},
                {"message": {"content": ""}, "done": True, "prompt_eval_count": 31},
            )
        )

    default = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="model", open_request=open_request
    )
    assert [segment async for segment in default.stream(_snapshot(), turn_id="turn_1")] == [
        "Short reply."
    ]
    explicit = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="model", open_request=open_request,
        num_ctx=32_768,
    )
    assert [segment async for segment in explicit.stream(_snapshot(), turn_id="turn_2")] == [
        "Short reply."
    ]

    assert [body["options"] for body in bodies] == [{"num_ctx": 16_384}, {"num_ctx": 32_768}]
    assert all(body.get("shift") is False for body in bodies)
    assert all("truncate" not in body for body in bodies)
    sent = cast(list[dict[str, str]], bodies[0]["messages"])
    content = "".join(message["content"] for message in sent)
    reports = [
        json.loads(line.removeprefix("[ollama-prompt] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[ollama-prompt] ")
    ]
    assert reports[0] == {
        "output_style": "default", "reference_kinds": ["active_work"],
        "messages": len(sent),
        "num_ctx": 16_384,
        "prompt_bytes": len(content.encode("utf-8")),
        "prompt_chars": len(content),
        "prompt_eval_count": 31,
        "version": 1,
    }
    assert reports[1]["num_ctx"] == 32_768
    assert len(reports) == 2


@pytest.mark.asyncio
async def test_ollama_overflow_refusal_does_not_retry_or_emit_and_releases_turn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[Request] = []
    refusal = HTTPError(
        "http://127.0.0.1:11434/api/chat", 400,
        "the prompt is longer than the context window", {}, None,
    )
    response = LineResponse((
        {"message": {"content": "Next turn reply."}, "done": True},
    ))

    def open_request(request: Request, timeout: float) -> LineResponse:
        del timeout
        requests.append(request)
        if len(requests) == 1:
            raise refusal
        return response

    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="model", open_request=open_request,
        max_active_streams=1,
    )
    emitted: list[str] = []
    with pytest.raises(HTTPError) as caught:
        async for segment in inference.stream(_snapshot(), turn_id="turn_refused"):
            emitted.append(segment)
    assert caught.value is refusal
    assert emitted == []
    assert len(requests) == 1
    assert capsys.readouterr().out == ""

    # Let the failed iterator's scheduled cleanup take its next loop turn.
    await asyncio.sleep(0)
    assert inference._openings == {}
    assert inference._opening_cleanups == {}
    assert inference._responses == {}
    assert inference._cancelled_turns == set()
    assert [part async for part in inference.stream(_snapshot(), turn_id="turn_next")] == [
        "Next turn reply."
    ]
    assert len(requests) == 2
    assert response.closed
    await inference.close()


@pytest.mark.parametrize(
    ("num_ctx", "error"),
    [(cast(int, True), TypeError), (cast(int, 16_384.0), TypeError), (2047, ValueError),
     (262_145, ValueError)],
)
def test_ollama_context_window_is_an_exact_bounded_integer(
    num_ctx: int, error: type[Exception]
) -> None:
    assert OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="model", num_ctx=2048
    )._num_ctx == 2048
    assert OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="model", num_ctx=262_144
    )._num_ctx == 262_144
    with pytest.raises(error, match="num_ctx"):
        OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="model", num_ctx=num_ctx)


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
