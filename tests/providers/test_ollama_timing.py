from __future__ import annotations

import json
from collections import deque
from typing import cast
from urllib.request import Request

import pytest

from hermes_realtime.conversation import ConversationInferenceRequest, ConversationMessage
from hermes_realtime.providers import OllamaStreamingInference
from tests.support.ollama_metadata import supported_ollama_metadata

pytestmark = pytest.mark.usefixtures("deny_uninjected_ollama_metadata")

_METRIC_NAMES = (
    "load_duration_ns",
    "prompt_eval_duration_ns",
    "eval_duration_ns",
    "total_duration_ns",
)
_SOURCE_NAMES = tuple(name.removesuffix("_ns") for name in _METRIC_NAMES)
_MAX_DURATION_NS = 86_400_000_000_000


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._lines = deque((json.dumps(payload).encode("utf-8") + b"\n",))

    def readline(self, limit: int = -1, /) -> bytes:
        del limit
        return self._lines.popleft() if self._lines else b""

    def close(self) -> None:
        pass


def _inference(payload: dict[str, object]) -> OllamaStreamingInference:
    def open_request(request: Request, timeout: float) -> _Response:
        del request, timeout
        return _Response(payload)

    return OllamaStreamingInference(
        metadata_client=supported_ollama_metadata("synthetic-model"),
        base_url="http://127.0.0.1:11434",
        model="synthetic-model",
        open_request=open_request,
    )


def _snapshot() -> ConversationInferenceRequest:
    return ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage(role="user", text="Synthetic question."),),
        active_tasks=(),
        updates=(),
    )


def _report(output: str) -> dict[str, object]:
    lines = [line for line in output.splitlines() if line.startswith("[ollama-prompt] ")]
    assert len(lines) == 1
    return cast(dict[str, object], json.loads(lines[0].removeprefix("[ollama-prompt] ")))


@pytest.mark.asyncio
async def test_final_completion_durations_are_bound_to_their_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = dict(zip(_METRIC_NAMES, (1, 20, 300, 321), strict=True))
    payload = {
        "message": {"content": "Synthetic answer."},
        "done": True,
        **dict(zip(_SOURCE_NAMES, expected.values(), strict=True)),
    }

    assert [part async for part in _inference(payload).stream(_snapshot(), turn_id="timing")]
    report = _report(capsys.readouterr().out)

    assert {name: report[name] for name in _METRIC_NAMES} == expected
    assert report["timing_invalid_count"] == 0
    assert report["timing_invalid_fields"] == []


@pytest.mark.parametrize(
    "bad_value",
    [None, True, False, 1.0, "1", -1, _MAX_DURATION_NS + 1, [], {}],
)
@pytest.mark.asyncio
async def test_invalid_completion_duration_is_reported_without_blocking_speech(
    bad_value: object, capsys: pytest.CaptureFixture[str]
) -> None:
    payload: dict[str, object] = {
        "message": {"content": "Synthetic answer."},
        "done": True,
        "load_duration": bad_value,
        "prompt_eval_duration": 20,
        "eval_duration": 300,
        "total_duration": 321,
    }

    assert [part async for part in _inference(payload).stream(_snapshot(), turn_id="timing")]
    report = _report(capsys.readouterr().out)

    assert report["load_duration_ns"] is None
    assert report["prompt_eval_duration_ns"] == 20
    assert report["eval_duration_ns"] == 300
    assert report["total_duration_ns"] == 321
    assert report["timing_invalid_count"] == 1
    assert report["timing_invalid_fields"] == ["load_duration"]
    assert "Synthetic answer" not in json.dumps(report)


@pytest.mark.parametrize("field", _SOURCE_NAMES)
@pytest.mark.parametrize("bad_value", [True, _MAX_DURATION_NS + 1])
@pytest.mark.asyncio
async def test_each_completion_duration_rejects_bad_type_and_overbound_values(
    field: str, bad_value: object, capsys: pytest.CaptureFixture[str]
) -> None:
    payload: dict[str, object] = {
        "message": {"content": "Synthetic answer."},
        "done": True,
        "load_duration": 1,
        "prompt_eval_duration": 20,
        "eval_duration": 300,
        "total_duration": 321,
    }
    payload[field] = bad_value

    assert [part async for part in _inference(payload).stream(_snapshot(), turn_id="timing")]
    report = _report(capsys.readouterr().out)

    assert report[f"{field}_ns"] is None
    assert report["timing_invalid_count"] == 1
    assert report["timing_invalid_fields"] == [field]


@pytest.mark.parametrize("field", _SOURCE_NAMES)
@pytest.mark.asyncio
async def test_missing_completion_duration_is_unknown_not_a_stream_failure(
    field: str, capsys: pytest.CaptureFixture[str]
) -> None:
    payload: dict[str, object] = {
        "message": {"content": "Synthetic answer."},
        "done": True,
        "load_duration": 1,
        "prompt_eval_duration": 20,
        "eval_duration": 300,
        "total_duration": 321,
    }
    del payload[field]

    assert [part async for part in _inference(payload).stream(_snapshot(), turn_id="timing")]
    report = _report(capsys.readouterr().out)

    assert report[f"{field}_ns"] is None
    assert report["timing_invalid_count"] == 0
    assert report["timing_invalid_fields"] == []
