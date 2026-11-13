"""A failed or interrupted result is bounded in the unit the browser counts: UTF-16 code units.

The shared fixture holds reasons and the exact text the server must publish for each;
``web/tests/legacy-result-units.test.ts`` parses the same texts with the browser's validator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.conversation.tasks import TaskTerminalOutcome
from hermes_realtime.conversation.updates import UpdatePolicyInput
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.protocol.limits import truncate_to_utf16_units, utf16_units
from hermes_realtime.result_limits import MAX_LEGACY_RESULT_CHARS

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (ROOT / "tests/fixtures/legacy_result_utf16.json").read_text(encoding="ascii")
)
ASTRAL = "\U0001F600"


def _expand(segments: list[list[object]]) -> str:
    return "".join(str(text) * int(str(repeat)) for text, repeat in segments)


def _javascript_length(text: str) -> int:
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


async def _published_result(status: str, reason: str) -> str:
    projection = BrowserEventProjection()
    await ProactiveCompletionUpdatePolicy(projection=projection).decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(task_id="task_astral", status=status, reason=reason),
            context=ConversationContextStore().snapshot(),
        )
    )
    event = projection.events_after(0)[-1]
    assert event.kind == "task_result"
    text = event.data["text"]
    assert type(text) is str
    return text


def test_fixture_is_bound_to_the_shared_limit() -> None:
    assert FIXTURE["limit_utf16_units"] == MAX_LEGACY_RESULT_CHARS
    assert len(FIXTURE["cases"]) >= 5


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "interrupted"])
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
async def test_host_publishes_a_result_the_browser_bound_admits(
    status: str, case: dict[str, list[list[object]]]
) -> None:
    text = await _published_result(status, _expand(case["reason"]))

    assert _javascript_length(text) <= MAX_LEGACY_RESULT_CHARS
    assert text == _expand(case["text"])


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_projection_refuses_a_noncompleted_result_over_the_unit_bound(status: str) -> None:
    projection = BrowserEventProjection()
    data = {"status": status, "taskId": "task_astral"}

    accepted = projection.publish("task_result", data | {"text": ASTRAL * 512})
    assert accepted.data["text"] == ASTRAL * 512
    with pytest.raises(ValueError, match="text"):
        projection.publish("task_result", data | {"text": ASTRAL * 512 + "x"})
    with pytest.raises(ValueError, match="text"):
        projection.publish("task_result", data | {"text": ASTRAL * 1024})


@pytest.mark.parametrize(
    "text",
    ["", "x", ASTRAL, "x" + ASTRAL, "\ud800", "\U0001f600", "a\ud800" + ASTRAL, "\uffff\U00010000"],
)
def test_unit_count_is_the_javascript_string_length(text: str) -> None:
    assert utf16_units(text) == _javascript_length(text)


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("abc", 3, "abc"),
        ("abc", 2, "ab"),
        ("abc", 0, ""),
        (ASTRAL * 3, 6, ASTRAL * 3),
        (ASTRAL * 3, 5, ASTRAL * 2),
        (ASTRAL * 3, 1, ""),
        ("x" + ASTRAL * 2, 4, "x" + ASTRAL),
        ("\ud800\ud800", 1, "\ud800"),
    ],
)
def test_truncation_keeps_the_longest_prefix_of_whole_characters(
    text: str, limit: int, expected: str
) -> None:
    assert truncate_to_utf16_units(text, limit) == expected


@pytest.mark.parametrize("limit", [-1, True, 1.0])
def test_truncation_refuses_an_invalid_limit(limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        truncate_to_utf16_units("abc", limit)  # type: ignore[arg-type]
