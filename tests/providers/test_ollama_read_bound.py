"""The advertised response-line bound must constrain the read, not just the parse."""

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
    """Records the exact limit each read requests."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self.limits: list[int] = []
        self.closed = False

    def readline(self, limit: int = -1, /) -> bytes:
        self.limits.append(limit)
        if self.closed or not self._lines:
            return b""
        return self._lines.pop(0)

    def close(self) -> None:
        self.closed = True


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


@pytest.mark.asyncio
async def test_overlong_newline_free_record_is_rejected_before_parsing() -> None:
    # A record that fills the whole allowance without terminating is either
    # overlong or truncated; either way it must not be parsed.
    response = RecordingResponse([b"{" + b"a" * _MAX_RESPONSE_LINE_BYTES])
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model="test-model",
        open_request=lambda _request, _timeout: response,
    )

    with pytest.raises(ValueError, match="exceeds supported size"):
        async for _segment in inference.stream(_snapshot(), turn_id="turn_001"):
            pass
