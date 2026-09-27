from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from hermes_realtime.companion.integrity import MAX_BATCH_ROWS, REFUSAL_CATEGORIES
from hermes_realtime.protocol import (
    BRIDGE_CAPABILITIES,
    BRIDGE_PROTOCOL_VERSION,
    VOICE_ARCHIVE_CAPABILITY,
    VOICE_INTEGRITY_REFUSALS,
    VOICE_REFUSAL_CATEGORIES,
    VOICE_TRANSIENT_REFUSALS,
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceArchiveRow,
    parse_event,
    parse_voice_event,
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seq": 3,
        "role": "user",
        "text": " Hello ",
        "interrupted": False,
        "ts": 1_700_000_000.5,
        "gap_before": [1, 2],
    }
    row.update(overrides)
    return row


def _archive(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "protocol_version": "0.2",
        "type": "voice_archive",
        "conversation_id": "conv_1-a",
        "generation": 0,
        "seq_from": 1,
        "seq_through": 3,
        "rows": [_row()],
    }
    event.update(overrides)
    return event


def _range(kind: str, **overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "protocol_version": "0.2",
        "type": kind,
        "conversation_id": "conv",
        "generation": 0,
        "seq_from": 0,
        "seq_through": 4,
    }
    event.update(overrides)
    return event


def test_the_bridge_is_version_0_2_and_offers_only_voice_archive() -> None:
    assert BRIDGE_PROTOCOL_VERSION == "0.2"
    assert VOICE_ARCHIVE_CAPABILITY == "voice_archive"
    assert frozenset({"voice_archive"}) == BRIDGE_CAPABILITIES


def test_a_voice_archive_round_trips_exactly_and_never_normalizes_text() -> None:
    event = parse_voice_event(json.dumps(_archive()))

    assert type(event) is VoiceArchiveEvent
    assert event.rows[0] == VoiceArchiveRow(
        seq=3, role="user", text=" Hello ", interrupted=False, ts=1_700_000_000.5,
        gap_before=(1, 2),
    )
    assert parse_voice_event(event.model_dump_json()) == event
    assert parse_voice_event(_archive()) == event


def test_acknowledgment_and_refusal_round_trip() -> None:
    ack = parse_voice_event(json.dumps(_range("voice_archive_ack")))
    refused = parse_voice_event(json.dumps(_range("voice_archive_refused", category="partition")))

    assert type(ack) is VoiceArchiveAckEvent
    assert type(refused) is VoiceArchiveRefusedEvent
    assert refused.category == "partition"
    assert parse_voice_event(ack.model_dump_json()) == ack
    assert parse_voice_event(refused.model_dump_json()) == refused


def test_the_refusal_categories_are_exactly_the_companions() -> None:
    assert VOICE_REFUSAL_CATEGORIES == REFUSAL_CATEGORIES


def test_transient_and_integrity_refusals_partition_the_companions_categories() -> None:
    assert VOICE_TRANSIENT_REFUSALS | VOICE_INTEGRITY_REFUSALS == REFUSAL_CATEGORIES
    assert not VOICE_TRANSIENT_REFUSALS & VOICE_INTEGRITY_REFUSALS


def test_every_category_that_can_mean_the_archive_is_wrong_fences() -> None:
    # The batch, the archive's evidence, the quarantine and the retired generation.
    assert {
        "invalid", "partition", "identity", "conflict", "capacity", "mismatch", "missing",
        "over_cap", "rotated", "lineage", "count", "recovery", "drift", "quarantined",
        "tombstoned",
    } <= VOICE_INTEGRITY_REFUSALS
    assert {"not_ready", "lease_held", "conversations"} <= VOICE_TRANSIENT_REFUSALS


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(_archive(protocol_version="0.1"), id="version-0.1"),
        pytest.param(_archive(extra=1), id="extra-field"),
        pytest.param(_archive(conversation_id="has space"), id="bad-conversation-id"),
        pytest.param(_archive(conversation_id="c" * 65), id="long-conversation-id"),
        pytest.param(_archive(conversation_id=""), id="empty-conversation-id"),
        pytest.param(_archive(generation=-1), id="negative-generation"),
        pytest.param(_archive(generation=True), id="boolean-generation"),
        pytest.param(_archive(generation="0"), id="string-generation"),
        pytest.param(_archive(seq_through=2**53), id="seq-past-the-identity-bound"),
        pytest.param(_archive(seq_from=4), id="reversed-range"),
        pytest.param(_archive(rows=[]), id="no-rows"),
        pytest.param(_archive(rows=[_row()] * (MAX_BATCH_ROWS + 1)), id="too-many-rows"),
        pytest.param(_archive(rows=[_row(role="system")]), id="unknown-role"),
        pytest.param(_archive(rows=[_row(text="")]), id="empty-text"),
        pytest.param(_archive(rows=[_row(text="x" * 65_537)]), id="text-past-the-bound"),
        pytest.param(_archive(rows=[_row(interrupted=0)]), id="integer-flag"),
        pytest.param(_archive(rows=[_row(ts=1_700_000_000)]), id="integer-timestamp"),
        pytest.param(_archive(rows=[_row(ts=-0.5)]), id="negative-timestamp"),
        pytest.param(_archive(rows=[_row(ts="1.0")]), id="string-timestamp"),
        pytest.param(_archive(rows=[_row(gap_before=[1])]), id="short-gap"),
        pytest.param(_archive(rows=[_row(gap_before=[1, 2, 3])]), id="long-gap"),
        pytest.param(_archive(rows=[_row(gap_before=[-1, 2])]), id="negative-gap"),
        pytest.param(_archive(rows=[{k: v for k, v in _row().items() if k != "gap_before"}]),
                     id="missing-gap-field"),
        pytest.param(_archive(rows=[_row(extra=1)]), id="extra-row-field"),
        pytest.param(_range("voice_archive_ack", category="x"), id="ack-with-category"),
        pytest.param(_range("voice_archive_refused"), id="refusal-without-category"),
        pytest.param(_range("voice_archive_refused", category=""), id="empty-category"),
        pytest.param(_range("voice_archive_refused", category="x" * 33), id="long-category"),
        pytest.param(_range("voice_archive_refused", category="Not_Ready"), id="upper-category"),
        pytest.param(_range("voice_archive_refused", category=7), id="integer-category"),
        pytest.param(
            {k: v for k, v in _range("voice_archive_ack").items() if k != "protocol_version"},
            id="no-protocol-version",
        ),
        pytest.param(_range("voice_archive_ack", seq_from=5), id="reversed-ack"),
        pytest.param(_range("voice_forget"), id="unknown-type"),
    ],
)
def test_every_malformed_voice_event_is_refused(event: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_voice_event(json.dumps(event))


def test_a_refusal_category_is_a_bounded_string_on_the_wire_not_a_closed_list() -> None:
    # A newer companion may add a category: it still parses, and the sender classifies it.
    refused = parse_voice_event(json.dumps(_range("voice_archive_refused", category="novel")))

    assert type(refused) is VoiceArchiveRefusedEvent
    assert refused.category == "novel"
    assert "novel" not in VOICE_TRANSIENT_REFUSALS | VOICE_INTEGRITY_REFUSALS


def test_a_voice_event_names_its_protocol_version_explicitly() -> None:
    with pytest.raises(ValidationError):
        VoiceArchiveAckEvent(  # type: ignore[call-arg]
            type="voice_archive_ack", conversation_id="c", generation=0, seq_from=0,
            seq_through=0,
        )


def test_a_non_finite_timestamp_is_refused() -> None:
    raw = json.dumps(_archive()).replace("1700000000.5", "Infinity")
    with pytest.raises(ValidationError):
        parse_voice_event(raw)


def test_work_and_voice_parsers_never_accept_each_others_events() -> None:
    with pytest.raises(ValidationError):
        parse_event(json.dumps(_archive()))
    work = {
        "protocol_version": "0.1",
        "type": "control.cancel",
        "event_id": "evt_1",
        "session_id": "session_1",
        "sequence": 0,
        "timestamp": "2026-09-26T00:00:00Z",
        "payload": {"scope": "turn"},
    }
    parse_event(json.dumps(work))
    with pytest.raises(ValidationError):
        parse_voice_event(json.dumps(work))
