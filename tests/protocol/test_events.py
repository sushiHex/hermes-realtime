from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    Durability,
    WorkCompletedEvent,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    parse_event,
)

NOW = datetime(2026, 7, 19, tzinfo=UTC)
BASE = {
    "protocol_version": "0.1",
    "event_id": "evt_001",
    "session_id": "session_001",
    "sequence": 1,
    "timestamp": NOW,
}


def test_work_dispatch_request_round_trips_through_json() -> None:
    event = WorkDispatchRequestedEvent(
        **BASE,
        type="work.dispatch.requested",
        task_id="task_001",
        utterance_id="utterance_001",
        payload={
            "objective": "Compare two realtime speech providers",
            "durability": Durability.EPHEMERAL,
            "progress_reporting": "material_only",
            "completion_reporting": "proactive",
        },
    )

    parsed = parse_event(event.model_dump_json())

    assert parsed == event
    assert parsed.payload.objective == "Compare two realtime speech providers"


def test_accepted_dispatch_acknowledgment_requires_run_id() -> None:
    with pytest.raises(ValidationError, match="run_id"):
        WorkDispatchAcknowledgedEvent(
            **BASE,
            type="work.dispatch.acknowledged",
            task_id="task_001",
            payload={"accepted": True},
        )


def test_rejected_dispatch_acknowledgment_requires_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        WorkDispatchAcknowledgedEvent(
            **BASE,
            type="work.dispatch.acknowledged",
            task_id="task_001",
            payload={"accepted": False},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"accepted": True, "run_id": "run_001", "reason": "contradictory"},
        {"accepted": False, "run_id": "run_claimed", "reason": "rejected"},
    ],
)
def test_dispatch_acknowledgment_rejects_contradictory_outcome(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="must not be present"):
        WorkDispatchAcknowledgedPayload.model_validate(payload)


def test_completed_work_round_trips_with_authoritative_run_id() -> None:
    event = WorkCompletedEvent(
        **BASE,
        type="work.completed",
        task_id="task_001",
        run_id="deleg_001",
        payload={
            "status": "completed",
            "summary": "The comparison finished with one material recommendation.",
        },
    )

    parsed = parse_event(event.model_dump_json())

    assert parsed == event
    assert parsed.payload.summary.startswith("The comparison finished")


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "completed"},
        {"status": "completed", "summary": "done", "reason": "contradictory"},
        {"status": "failed"},
        {"status": "interrupted", "summary": "partial"},
    ],
)
def test_work_completion_requires_unambiguous_terminal_evidence(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        WorkCompletedEvent(
            **BASE,
            type="work.completed",
            task_id="task_001",
            run_id="deleg_001",
            payload=payload,  # type: ignore[arg-type]
        )


def test_task_cancellation_requires_task_id() -> None:
    with pytest.raises(ValidationError, match="task_id"):
        ControlCancelEvent(
            **BASE,
            type="control.cancel",
            payload={"scope": CancelScope.TASK},
        )


def test_speech_cancellation_rejects_task_id() -> None:
    with pytest.raises(ValidationError, match="task_id"):
        ControlCancelEvent(
            **BASE,
            type="control.cancel",
            task_id="task_001",
            payload={"scope": CancelScope.SPEECH},
        )


def test_cancel_acknowledgment_round_trips_signaled_runs() -> None:
    event = ControlCancelAcknowledgedEvent(
        **BASE,
        type="control.cancel.acknowledged",
        request_event_id="evt_cancel_001",
        scope=CancelScope.TASK,
        task_id="task_001",
        payload={"accepted": True, "signaled_run_ids": ["deleg_001"]},
    )

    assert parse_event(event.model_dump_json()) == event


@pytest.mark.parametrize(
    "payload",
    [
        {"accepted": True, "signaled_run_ids": []},
        {
            "accepted": True,
            "signaled_run_ids": ["deleg_001"],
            "reason": "contradictory",
        },
        {"accepted": False},
        {
            "accepted": False,
            "signaled_run_ids": ["deleg_claimed"],
            "reason": "not running",
        },
    ],
)
def test_cancel_acknowledgment_requires_unambiguous_signal_evidence(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ControlCancelAcknowledgedPayload.model_validate(payload)


def test_unknown_protocol_version_fails_closed() -> None:
    data = {
        **BASE,
        "protocol_version": "9.9",
        "type": "control.cancel",
        "payload": {"scope": "speech"},
    }

    with pytest.raises(ValidationError, match="protocol_version"):
        parse_event(data)


def test_unknown_event_type_fails_closed() -> None:
    data = {
        **BASE,
        "type": "assistant.did-something-surprising",
        "payload": {},
    }

    with pytest.raises(ValidationError):
        parse_event(data)


def test_protocol_rejects_text_that_could_exceed_bridge_line_limit() -> None:
    with pytest.raises(ValidationError):
        WorkDispatchRequestedEvent(
            **BASE,
            type="work.dispatch.requested",
            task_id="task_001",
            utterance_id="utterance_001",
            payload={"objective": "x" * 8193},
        )

    with pytest.raises(ValidationError):
        WorkCompletedEvent(
            **BASE,
            type="work.completed",
            task_id="task_001",
            run_id="run_001",
            payload={"status": "completed", "summary": "x" * 8193},
        )


def test_timestamp_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timestamp"):
        ControlCancelEvent(
            **{**BASE, "timestamp": datetime(2026, 7, 19)},
            type="control.cancel",
            payload={"scope": CancelScope.SPEECH},
        )
