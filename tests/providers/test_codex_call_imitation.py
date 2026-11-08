"""Codex refuses model text that imitates an advertised tool call (#311)."""

import pytest

from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from tests.providers.test_codex_app_server import (
    FakeCodexTransport,
    FakeWorkToolHandler,
    _collect_stream,
)

IMITATIONS = [
    'start_work(objective="check the logs")',
    'Okay, I can help with that. functions.start_work(objective="check the logs")',
    '{"name": "start_work", "arguments": {"objective": "check the logs"}}',
    '{"task_status": {}}',
    "<function=search_knowledge>",
    # Complete mid-stream, before the final flush.
    'Then I run start_work(objective="x"). Another sentence follows here. And one more to end.',
]


def _inference(text: str, *, bound: bool) -> CodexAppServerStreamingInference:
    transport = FakeCodexTransport(assistant_delta=text)
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    if bound:
        inference.bind_work_tools(FakeWorkToolHandler())
    return inference


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [True, False], ids=["work_tools", "conversation_only"])
@pytest.mark.parametrize("text", IMITATIONS)
async def test_imitation_segment_is_never_yielded(text, bound) -> None:
    inference = _inference(text, bound=bound)
    spoken: list[object] = []
    try:
        with pytest.raises(RuntimeError, match="tool-call imitation"):
            async for segment in inference.stream(
                _request(), turn_id="turn_imitation",
            ):
                spoken.append(segment)
    finally:
        await inference.close()
    assert all("start_work" not in str(s) and "search_knowledge" not in str(s)
               and "task_status" not in str(s) for s in spoken)


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [True, False], ids=["work_tools", "conversation_only"])
async def test_ordinary_mention_is_spoken(bound) -> None:
    text = "I can start_work for that, and the task status is fine."
    inference = _inference(text, bound=bound)
    try:
        assert " ".join(await _collect_stream(inference, "turn_mention")) == text
    finally:
        await inference.close()


def _request():
    from hermes_realtime.conversation.context import ConversationMessage
    from hermes_realtime.conversation.streaming import ConversationInferenceRequest

    return ConversationInferenceRequest(
        revision=1, messages=(ConversationMessage("user", "Please check the logs."),),
        active_tasks=(),
    )
