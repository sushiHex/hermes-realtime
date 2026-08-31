"""The advertised response-line bound must constrain the read, not just the parse."""

import io
import json

import pytest

from hermes_realtime.conversation import (
    ActiveTaskSummary,
    ConversationInferenceRequest,
    ConversationMessage,
)
from hermes_realtime.providers.ollama import (
    _MAX_RESPONSE_LINE_BYTES,
    OllamaStreamingInference,
)


def _snapshot() -> ConversationInferenceRequest:
    return ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage(role="user", text="What changed?"),),
        active_tasks=(
            ActiveTaskSummary(task_id="task_review", objective="Review the release"),
        ),
        updates=(),
    )


class RecordingResponse:
    """Records each bound while preserving real ``readline(size)`` semantics."""

    def __init__(self, lines: list[bytes]) -> None:
        self._stream = io.BytesIO(b"".join(lines))
        self.limits: list[int] = []
        self.closed = False

    @property
    def remaining_bytes(self) -> int:
        return len(self._stream.getbuffer()) - self._stream.tell()

    def readline(self, limit: int = -1, /) -> bytes:
        self.limits.append(limit)
        if self.closed:
            return b""
        return self._stream.readline(limit)

    def close(self) -> None:
        self.closed = True


def _valid_record_with_size(total_bytes: int, *, newline: bool) -> bytes:
    prefix = b'{"message":{"content":""},"done":true,"padding":"'
    suffix = b'"}' + (b"\n" if newline else b"")
    assert total_bytes >= len(prefix) + len(suffix)
    return prefix + b"a" * (total_bytes - len(prefix) - len(suffix)) + suffix


@pytest.mark.asyncio
async def test_reader_requests_at_most_one_byte_past_the_advertised_bound() -> None:
    # An unbounded readline() would materialise a newline-free record in full
    # before the size check ever ran, so the bound could not prevent it.
    response = RecordingResponse(
        [json.dumps({"message": {"content": "Hello."}, "done": True}).encode("utf-8") + b"\n"]
    )
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="test-model",
        open_request=lambda _request, _timeout: response,
    )

    async for _segment in inference.stream(_snapshot(), turn_id="turn_001"):
        pass

    assert response.limits
    assert set(response.limits) == {_MAX_RESPONSE_LINE_BYTES + 1}


@pytest.mark.parametrize("newline", [False, True], ids=["eof", "newline"])
@pytest.mark.asyncio
async def test_record_at_the_exact_byte_bound_is_accepted(newline: bool) -> None:
    response = RecordingResponse(
        [_valid_record_with_size(_MAX_RESPONSE_LINE_BYTES, newline=newline)]
    )
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="test-model",
        open_request=lambda _request, _timeout: response,
    )

    segments = [
        segment async for segment in inference.stream(_snapshot(), turn_id="turn_001")
    ]

    assert segments == []
    assert response.remaining_bytes == 0
    assert set(response.limits) == {_MAX_RESPONSE_LINE_BYTES + 1}


@pytest.mark.asyncio
async def test_overlong_newline_free_record_is_rejected_before_parsing() -> None:
    # The size-respecting fake leaves the unread overflow buffered, proving that
    # rejection occurs after MAX + 1 bytes rather than materialising the record.
    record = b"{" + b"a" * (_MAX_RESPONSE_LINE_BYTES * 2)
    response = RecordingResponse([record])
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="test-model",
        open_request=lambda _request, _timeout: response,
    )

    with pytest.raises(ValueError, match="exceeds supported size"):
        async for _segment in inference.stream(_snapshot(), turn_id="turn_001"):
            pass

    assert response.limits == [_MAX_RESPONSE_LINE_BYTES + 1]
    assert response.remaining_bytes == len(record) - (_MAX_RESPONSE_LINE_BYTES + 1)
