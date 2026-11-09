"""Ollama refuses model text that imitates an advertised tool call (#311)."""

import json

import pytest

import hermes_realtime.providers.ollama as ollama
from hermes_realtime.conversation.streaming import ModelOutputRefused
from tests.providers.test_ollama_work_tools import (
    Response,
    adapter,
    collect,
    decision_frame,
    request,
)
from tests.support.ollama_metadata import supported_ollama_metadata

IMITATIONS = [
    'Okay, running it. start_work(objective="check the logs")',
    '{"name": "start_work", "arguments": {"objective": "check the logs"}}',
    '{"start_work": {"objective": "check the logs"}}',
    "<function=task_status>",
    # Complete mid-stream, before the final flush.
    'Then I run start_work(objective="x"). Another sentence follows here. And one more to end.',
]


def _evidence(capsys) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("[ollama-work] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[ollama-work] ")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", IMITATIONS)
async def test_structured_reply_imitation_is_never_yielded(text, capsys) -> None:
    inference, handler, _, response = adapter([decision_frame({"kind": "reply", "text": text})])
    spoken: list[object] = []
    with pytest.raises(ModelOutputRefused, match="call_imitation") as caught:
        async for segment in inference.stream(request(), turn_id="turn_test"):
            spoken.append(segment)
    assert type(caught.value) is ModelOutputRefused
    assert spoken == []
    assert handler.calls == []
    assert response.closed
    assert _evidence(capsys) == [{"category": "call_imitation", "kind": "start", "attempts": 0}]
    await inference.close()


@pytest.mark.asyncio
async def test_structured_reply_mentioning_a_tool_name_is_spoken() -> None:
    text = "I can start_work for that, and the task status is fine."
    inference, _, _, _ = adapter([decision_frame({"kind": "reply", "text": text})])
    try:
        assert " ".join(await collect(inference)) == text
    finally:
        await inference.close()


def _unbound(frames: list[dict[str, object]]) -> ollama.OllamaStreamingInference:
    response = Response(frames)
    return ollama.OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test",
        open_request=lambda req, timeout: response,
        metadata_client=supported_ollama_metadata(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", IMITATIONS)
async def test_no_tools_stream_imitation_is_never_yielded(text) -> None:
    prefix = "Okay, I am glad to help with that right now. "
    inference = _unbound([
        {"message": {"content": prefix}, "done": False},
        {"message": {"content": text + " "}, "done": True},
    ])
    spoken: list[object] = []
    with pytest.raises(ModelOutputRefused, match="call_imitation") as caught:
        async for segment in inference.stream(request(), turn_id="turn_test"):
            spoken.append(segment)
    assert type(caught.value) is ModelOutputRefused
    assert all("start_work" not in str(s) and "task_status" not in str(s) for s in spoken)
    await inference.close()


@pytest.mark.asyncio
async def test_no_tools_stream_ordinary_text_is_spoken() -> None:
    text = "I can start_work for that. The task status is fine."
    inference = _unbound([{"message": {"content": text}, "done": True}])
    try:
        assert " ".join(await collect(inference)) == text
    finally:
        await inference.close()
