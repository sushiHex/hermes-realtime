"""A refused task result leaves one content-free line; the real producer never triggers it."""

from __future__ import annotations

import json
import random

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.conversation.tasks import TaskTerminalOutcome
from hermes_realtime.conversation.updates import UpdatePolicyInput
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.protocol.limits import utf16_units
from hermes_realtime.result_limits import MAX_COMPLETED_RESULT_CHARS, MAX_LEGACY_RESULT_CHARS

MARKER = "[task-result-projection] "
ASTRAL = "\U0001F600"
SENTINEL = "SENTINEL-result-text"
TASK_ID = "task_sentinel_identity"


def _refusal_lines(output: str) -> list[dict[str, object]]:
    lines = [line for line in output.splitlines() if line.startswith(MARKER)]
    return [json.loads(line.removeprefix(MARKER)) for line in lines]


@pytest.mark.parametrize(
    ("status", "text", "category", "units", "limit"),
    [
        ("failed", SENTINEL + ASTRAL * 512, "result_text_units", 1025, 1024),
        ("interrupted", SENTINEL + "x" * 2000, "result_text_units", 1025, 1024),
        ("completed", SENTINEL + ASTRAL * 4096, "result_text_scalars", 4097, 4096),
        ("completed", SENTINEL + "\ud800", "result_text_unicode", len(SENTINEL) + 1, 4096),
        ("failed", " \n\t", "result_text_blank", 3, 1024),
        ("failed", 7, "result_text_type", 0, 1024),
    ],
    ids=["astral", "basic-plane", "scalars", "unicode", "blank", "type"],
)
def test_a_refused_result_leaves_one_line_without_its_text(
    capsys: pytest.CaptureFixture[str],
    status: str,
    text: object,
    category: str,
    units: int,
    limit: int,
) -> None:
    projection = BrowserEventProjection()

    with pytest.raises(ValueError, match="task result text is invalid"):
        projection.publish(
            "task_result",
            {"status": status, "taskId": TASK_ID, "text": text},  # type: ignore[dict-item]
        )

    output = capsys.readouterr().out
    assert _refusal_lines(output) == [
        {
            "category": category,
            "kind": "task_result",
            "limit": limit,
            "status": status,
            "units": units,
            "version": 1,
        }
    ]
    assert SENTINEL not in output and TASK_ID not in output and ASTRAL not in output
    assert len(output) < 200
    assert projection.events_after(0) == ()


@pytest.mark.parametrize(
    ("status", "text"),
    [
        ("failed", ASTRAL * 512),
        ("interrupted", "x" * MAX_LEGACY_RESULT_CHARS),
        ("completed", ASTRAL * MAX_COMPLETED_RESULT_CHARS),
    ],
    ids=["failed", "interrupted", "completed"],
)
def test_an_admitted_result_leaves_no_line(
    capsys: pytest.CaptureFixture[str], status: str, text: str
) -> None:
    projection = BrowserEventProjection()

    projection.publish("task_result", {"status": status, "taskId": TASK_ID, "text": text})

    assert capsys.readouterr().out == ""


def test_other_task_result_refusals_keep_their_own_error_and_leave_no_text_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    projection = BrowserEventProjection()

    with pytest.raises(ValueError, match="status"):
        projection.publish("task_result", {"status": "done", "taskId": TASK_ID, "text": "x"})

    assert _refusal_lines(capsys.readouterr().out) == []


@pytest.mark.asyncio
async def test_the_host_producer_never_reaches_the_refusal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Seeded reasons across the reason bound: every one is published within the unit bound."""
    generator = random.Random(526)
    # run_12345678 is replaced by a longer phrase before the text is bounded.
    alphabet = ("x", " ", ASTRAL, "\u00e9", "\uffff", "\U0010ffff", ".", " run_12345678 ")
    reasons = [ASTRAL * 1024, "x" * 1024, (ASTRAL + " ") * 512, "x" + ASTRAL * 1023]
    for _ in range(400):
        length = generator.choice((1, 2, 511, 512, 513, 1000, 1023, 1024))
        reason = "".join(generator.choice(alphabet) for _ in range(length))[:1024]
        reasons.append(reason if reason.strip() else "x")

    for status in ("failed", "interrupted"):
        for reason in reasons:
            projection = BrowserEventProjection()
            completion = TaskTerminalOutcome(task_id=TASK_ID, status=status, reason=reason)
            await ProactiveCompletionUpdatePolicy(projection=projection).decide(
                UpdatePolicyInput(
                    sequence=1,
                    completion=completion,
                    context=ConversationContextStore().snapshot(),
                )
            )
            text = projection.events_after(0)[-1].data["text"]
            assert type(text) is str and utf16_units(text) <= MAX_LEGACY_RESULT_CHARS

    assert MARKER not in capsys.readouterr().out
