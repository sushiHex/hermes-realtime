"""Memory is a negotiated, bounded data stream, separate from work events."""

import pytest
from pydantic import ValidationError

from hermes_realtime.protocol import parse_voice_event


def _snapshot() -> dict[str, object]:
    return dict(protocol_version="0.4", type="voice_memory_snapshot",
                conversation_id="conv", generation=0, revision=0,
                memory="", user="", truncated=False)


def test_memory_request_snapshot_and_refusal_round_trip() -> None:
    request = dict(protocol_version="0.4", type="voice_memory",
                   conversation_id="conv", generation=0)
    refusal = dict(request, type="voice_memory_refused", category="not_ready")
    for value in (request, _snapshot(), refusal):
        assert parse_voice_event(value).model_dump() == value


@pytest.mark.parametrize("field,value", [
    ("generation", True), ("revision", -1), ("revision", True),
    ("truncated", 1), ("memory", None), ("user", 4),
    ("protocol_version", "0.3"), ("extra", "unknown"),
])
def test_memory_wire_rejects_inexact_or_unknown_fields(field: str, value: object) -> None:
    raw = _snapshot()
    raw[field] = value
    with pytest.raises(ValidationError):
        parse_voice_event(raw)


@pytest.mark.parametrize("field", ["memory", "user"])
def test_memory_wire_bounds_utf8_bytes(field: str) -> None:
    raw = _snapshot()
    raw[field] = "\u00e9" * 2048
    assert parse_voice_event(raw).model_dump()[field] == raw[field]
    raw[field] += "a"  # type: ignore[operator]
    with pytest.raises(ValidationError):
        parse_voice_event(raw)


@pytest.mark.parametrize("field,value", [
    ("generation", -1), ("generation", True), ("generation", 2**53),
    ("protocol_version", "0.3"), ("extra", "unknown"),
])
def test_memory_subscription_requires_exact_bounded_fields(field: str, value: object) -> None:
    request = dict(
        protocol_version="0.4", type="voice_memory",
        conversation_id="conv", generation=0,
    )
    request[field] = value
    with pytest.raises(ValidationError):
        parse_voice_event(request)


def test_memory_refusal_requires_a_known_bounded_category() -> None:
    refusal = dict(
        protocol_version="0.4", type="voice_memory_refused",
        conversation_id="conv", generation=0, category="not_ready",
    )
    assert parse_voice_event(refusal).model_dump() == refusal
    for category in ("Nope", "x" * 33, 3):
        with pytest.raises(ValidationError):
            parse_voice_event(dict(refusal, category=category))
