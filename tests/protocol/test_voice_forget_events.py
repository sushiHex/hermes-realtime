"""Deletion stays capability-gated on the existing strict bridge version."""

import pytest
from pydantic import ValidationError

from hermes_realtime.protocol import VOICE_FORGET_CAPABILITY, parse_voice_event


def test_forget_events_round_trip() -> None:
    assert VOICE_FORGET_CAPABILITY == "voice_forget"
    request = dict(protocol_version="0.3", type="voice_forget",
                   conversation_id="synthetic", generation=3)
    for raw in (request, dict(request, type="voice_forget_ack", state="pending"),
                dict(request, type="voice_forget_ack", state="complete"),
                dict(request, type="voice_forget_refused", category="not_ready")):
        assert parse_voice_event(raw).model_dump() == raw


@pytest.mark.parametrize("kind", ["voice_forget", "voice_forget_ack", "voice_forget_refused"])
@pytest.mark.parametrize("field,value", [
    ("protocol_version", "0.4"), ("generation", True), ("generation", -1),
    ("generation", 2**53), ("conversation_id", ""), ("extra", "unknown"),
])
def test_forget_events_refuse_inexact_fields(kind: str, field: str, value: object) -> None:
    raw = dict(protocol_version="0.3", type=kind, conversation_id="synthetic", generation=3)
    if kind == "voice_forget_ack":
        raw["state"] = "pending"
    if kind == "voice_forget_refused":
        raw["category"] = "not_ready"
    raw[field] = value  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        parse_voice_event(raw)


@pytest.mark.parametrize("state", ["learned", "deleted", True, None, "", "x" * 100])
def test_forget_ack_accepts_only_pending_or_complete(state: object) -> None:
    with pytest.raises(ValidationError):
        parse_voice_event(dict(protocol_version="0.3", type="voice_forget_ack",
                               conversation_id="synthetic", generation=0, state=state))
