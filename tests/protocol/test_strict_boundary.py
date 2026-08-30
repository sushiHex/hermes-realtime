"""The bridge is the trust boundary: malformed events fail closed, never repair."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from hermes_realtime.protocol import Durability, parse_event

NOW = datetime(2026, 7, 19, tzinfo=UTC).isoformat()


def _request(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "protocol_version": "0.1",
        "type": "work.dispatch.requested",
        "event_id": "evt_001",
        "session_id": "session_001",
        "sequence": 1,
        "timestamp": NOW,
        "task_id": "task_001",
        "utterance_id": "utterance_001",
        "payload": {"objective": "Compare two options"},
    }
    event.update(overrides)
    return event


def test_parse_event_accepts_the_canonical_wire_form() -> None:
    parsed = parse_event(json.dumps(_request()))

    assert parsed.sequence == 1
    assert parsed.event_id == "evt_001"


@pytest.mark.parametrize("coerced", ["1", 1.0, True])
def test_parse_event_rejects_a_coerced_sequence(coerced: object) -> None:
    # Pydantic's lax default would convert each of these to int(1) and admit a
    # malformed event as though the peer had sent a real sequence.
    with pytest.raises(ValidationError):
        parse_event(json.dumps(_request(sequence=coerced)))


def test_parse_event_rejects_a_coerced_accepted_flag() -> None:
    acknowledgment = {
        "protocol_version": "0.1",
        "type": "work.dispatch.acknowledged",
        "event_id": "evt_002",
        "session_id": "session_001",
        "sequence": 2,
        "timestamp": NOW,
        "task_id": "task_001",
        "payload": {"accepted": "true", "run_id": "deleg_001"},
    }

    with pytest.raises(ValidationError):
        parse_event(json.dumps(acknowledgment))


@pytest.mark.parametrize(
    "field",
    ["event_id", "session_id", "task_id", "utterance_id"],
)
def test_parse_event_rejects_identifiers_that_require_normalization(field: str) -> None:
    # Trimming would alias " evt_001 " onto "evt_001" and let two distinct wire
    # spellings claim one identity across the bridge.
    with pytest.raises(ValidationError):
        parse_event(json.dumps(_request(**{field: " evt_001 "})))


def test_parse_event_still_accepts_enum_values_in_their_wire_form() -> None:
    # The string *is* the wire representation of these enums, so accepting it is
    # the contract rather than a coercion. Guards against over-tightening.
    parsed = parse_event(
        json.dumps(
            _request(payload={"objective": "Compare two options", "durability": "durable"})
        )
    )

    assert parsed.payload.durability is Durability.DURABLE
