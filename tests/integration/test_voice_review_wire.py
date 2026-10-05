"""Review traffic stays on the private voice bridge."""

import pytest

from hermes_realtime.protocol import VoiceReviewEvent, parse_voice_event


def test_review_wire_is_strict_and_bounded() -> None:
    event = VoiceReviewEvent(
        protocol_version="0.4",
        type="voice_review",
        conversation_id="voice_1",
        generation=0,
        seq_from=0,
        seq_through=9,
        memory=True,
        skills=True,
        closing=False,
    )
    assert parse_voice_event(event.model_dump_json()) == event
    with pytest.raises(ValueError):
        parse_voice_event(event.model_dump() | {"objective": "speak"})
    for field in ("memory", "skills", "closing"):
        with pytest.raises(ValueError):
            parse_voice_event(event.model_dump() | {field: 1})
