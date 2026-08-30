from __future__ import annotations

import pytest

from hermes_realtime.client import BrowserEventProjection


def test_capture_status_reservation_survives_concurrent_publication() -> None:
    projection = BrowserEventProjection(capacity=2)
    reservation = projection.reserve_capture_status()

    projection.publish("notification_queued", {})
    with pytest.raises(RuntimeError, match="capacity"):
        projection.publish("typed_input_admitted", {"inputSequence": 1})

    status = projection.publish_capture_status(
        reservation,
        {
            "available": True,
            "captureState": "active",
            "consentVersion": "realtime-evidence-consent-v1",
            "disclosureDigest": "a" * 64,
            "retentionHours": 24,
        },
    )
    assert status.kind == "capture_status"


def test_revocation_status_cannot_be_starved() -> None:
    projection = BrowserEventProjection(capacity=2)
    consent_status = projection.reserve_capture_status()
    revoke_status = projection.reserve_capture_status()

    with pytest.raises(RuntimeError, match="capacity"):
        projection.publish("notification_queued", {})

    projection.publish_capture_status(
        consent_status,
        {
            "available": True,
            "captureState": "active",
            "consentVersion": "realtime-evidence-consent-v1",
            "disclosureDigest": "a" * 64,
            "retentionHours": 24,
        },
    )
    terminal = projection.publish_capture_status(
        revoke_status,
        {
            "available": True,
            "captureState": "revoked_purging",
            "consentVersion": "realtime-evidence-consent-v1",
            "disclosureDigest": "a" * 64,
            "retentionHours": 24,
        },
    )
    assert terminal.sequence == 2


def test_consent_can_atomically_retain_four_capture_status_slots() -> None:
    projection = BrowserEventProjection(capacity=5)

    reservations = projection.reserve_capture_status_slots(4)
    projection.publish("notification_queued", {})
    with pytest.raises(RuntimeError, match="capacity"):
        projection.publish("typed_input_admitted", {"inputSequence": 1})

    assert len(reservations) == 4
    for reservation, state in zip(
        reservations,
        ("active", "faulted", "revoked_purging", "idle"),
        strict=True,
    ):
        projection.publish_capture_status(
            reservation,
            {
                "available": True,
                "captureState": state,
                "consentVersion": "realtime-evidence-consent-v1",
                "disclosureDigest": "a" * 64,
                "retentionHours": 24,
            },
        )


def test_capture_status_reservation_ownership_is_validated_without_consuming() -> None:
    from hermes_realtime.evidence.models import ProjectionReservation

    projection = BrowserEventProjection(capacity=2)
    reservation = projection.reserve_capture_status()
    foreign = BrowserEventProjection(capacity=2).reserve_capture_status()
    forged = object.__new__(ProjectionReservation)

    projection.validate_capture_status_reservation(reservation)
    for rejected in (foreign, forged):
        with pytest.raises(RuntimeError, match="stale or foreign"):
            projection.validate_capture_status_reservation(rejected)

    published = projection.publish_capture_status(
        reservation,
        {
            "available": True,
            "captureState": "idle",
            "consentVersion": "realtime-evidence-consent-v1",
            "disclosureDigest": "a" * 64,
            "retentionHours": 24,
        },
    )
    assert published.kind == "capture_status"


def test_failed_consent_can_release_four_status_slots_atomically() -> None:
    projection = BrowserEventProjection(capacity=4)
    reservations = projection.reserve_capture_status_slots(4)

    projection.release_capture_status_reservations(reservations)

    for index in range(4):
        event = projection.publish("notification_queued", {})
        assert event.sequence == index + 1


def test_projection_accepts_generated_assistant_text() -> None:
    projection = BrowserEventProjection()

    event = projection.publish(
        "assistant_text_generated",
        {
            "text": "Complete answer.",
            "turnId": "turn_001",
            "turnGeneration": 1,
            "segmentId": "segment_1",
        },
    )

    assert event.kind == "assistant_text_generated"


def test_projection_is_bounded_sequenced_and_rejects_private_handles() -> None:
    projection = BrowserEventProjection(capacity=2, clock=lambda: 12.5)

    first = projection.publish("notification_queued", {})
    projection.publish("typed_input_admitted", {"inputSequence": 1})

    assert (first.sequence, first.monotonic_ms) == (1, 12_500.0)
    assert [event.sequence for event in projection.events_after(0)] == [1, 2]
    for handle in (
        "deleg_private_handle",
        "run_12345678",
        "subagent_public_opaque_handle",
    ):
        with pytest.raises(ValueError, match="private"):
            projection.publish("task_state", {"detail": handle})
    with pytest.raises(RuntimeError, match="capacity"):
        projection.publish("session_stopped", {})


def test_projection_rejects_integers_that_browser_json_cannot_represent_exactly() -> None:
    projection = BrowserEventProjection()

    with pytest.raises(ValueError, match="integer"):
        projection.publish("typed_input_admitted", {"inputSequence": 1 << 53})


def test_projection_returns_bounded_polling_batches() -> None:
    projection = BrowserEventProjection(capacity=64)
    for sequence in range(40):
        projection.publish("typed_input_admitted", {"inputSequence": sequence + 1})

    assert len(projection.events_after(0)) == 32
    assert [event.sequence for event in projection.events_after(32)] == list(range(33, 41))


def test_projection_reset_starts_a_fresh_session_sequence() -> None:
    projection = BrowserEventProjection(capacity=2)
    projection.publish("notification_queued", {})
    projection.publish("session_stopped", {})

    projection.reset()
    replacement = projection.publish("notification_queued", {})

    assert replacement.sequence == 1
    assert projection.events_after(0) == (replacement,)


def test_projection_reserves_multiple_slots_before_partial_publication() -> None:
    projection = BrowserEventProjection(capacity=2)
    projection.publish("notification_queued", {})

    with pytest.raises(RuntimeError, match="capacity"):
        projection.ensure_capacity(2)
    assert [event.kind for event in projection.events_after(0)] == ["notification_queued"]


def test_projection_bounds_aggregate_event_string_data() -> None:
    projection = BrowserEventProjection()

    with pytest.raises(ValueError, match="aggregate"):
        projection.publish(
            "task_state",
            {"first": "a" * 4096, "second": "b" * 4096, "third": "c"},
        )


def test_projection_admits_bounded_partial_transcript_event() -> None:
    projection = BrowserEventProjection()

    event = projection.publish(
        "transcript_partial",
        {"role": "user", "text": "words still changing"},
    )

    assert event.kind == "transcript_partial"
    assert dict(event.data) == {"role": "user", "text": "words still changing"}


def test_projection_admits_server_confirmed_voice_input_readiness() -> None:
    projection = BrowserEventProjection()

    event = projection.publish(
        "voice_input_ready",
        {"generation": 1, "mediaIncarnation": 4},
    )

    assert event.kind == "voice_input_ready"
    assert dict(event.data) == {"generation": 1, "mediaIncarnation": 4}


def test_projection_requires_exact_session_speech_runtime() -> None:
    projection = BrowserEventProjection()
    data: dict[str, str | int | bool | None] = {
        "conversationProfile": "natural_v1",
        "mode": "microphone_or_typed",
        "sttModel": "moonshine-v2-small",
        "sttProvider": "moonshine",
        "ttsModel": "kokoro-v1.0.onnx",
        "ttsProvider": "kokoro",
    }

    assert dict(projection.publish("session_ready", data).data) == data
    with pytest.raises(ValueError, match="speech runtime"):
        projection.publish("session_ready", {**data, "sttModel": ""})


@pytest.mark.parametrize(
    "data",
    (
        {},
        {"generation": 1},
        {"generation": 0, "mediaIncarnation": 4},
        {"generation": True, "mediaIncarnation": 4},
        {"generation": 1, "mediaIncarnation": 0},
        {"generation": 1, "mediaIncarnation": True},
        {"generation": 1, "mediaIncarnation": 4, "extra": 2},
    ),
)
def test_projection_rejects_invalid_voice_input_readiness(data: dict[str, object]) -> None:
    projection = BrowserEventProjection()

    with pytest.raises((TypeError, ValueError)):
        projection.publish("voice_input_ready", data)  # type: ignore[arg-type]


def test_projection_admits_server_voice_activity_boundaries() -> None:
    projection = BrowserEventProjection()

    started = projection.publish("voice_activity_started", {})
    ended = projection.publish("voice_activity_ended", {})

    assert [started.kind, ended.kind] == ["voice_activity_started", "voice_activity_ended"]


def test_projection_validates_task_result_shape_before_publication() -> None:
    projection = BrowserEventProjection()

    event = projection.publish(
        "task_result",
        {
            "status": "completed",
            "taskId": "task_release_check",
            "text": "The full release result.",
        },
    )
    assert event.kind == "task_result"

    with pytest.raises(ValueError, match="invalid shape"):
        projection.publish("task_result", {"text": "missing authority"})
    with pytest.raises(ValueError, match="status"):
        projection.publish(
            "task_result",
            {
                "status": "active",
                "taskId": "task_release_check",
                "text": "Not terminal.",
            },
        )


def test_projection_admits_speech_verifier_advisories() -> None:
    projection = BrowserEventProjection()

    suppressed = projection.publish_advisory("barge_in_non_speech_suppressed", {})
    unavailable = projection.publish_advisory(
        "barge_in_verifier_unavailable",
        {"reason": "classification_error"},
    )

    assert suppressed is not None and suppressed.kind == "barge_in_non_speech_suppressed"
    assert unavailable is not None and unavailable.kind == "barge_in_verifier_unavailable"


def test_projection_admits_turn_scoped_speech_timing_events() -> None:
    projection = BrowserEventProjection()

    timing = projection.publish_advisory(
        "speech_timing",
        {
            "turnId": "turn_001",
            "turnGeneration": 7,
            "chunkId": "kokoro_abc123",
            "streamId": "speech_def456",
            "sampleRate": 48_000,
            "timingSource": "estimated",
            "timings": "0,5,0,12000;6,11,12000,24000",
        },
    )
    completed = projection.publish(
        "assistant_turn_completed",
        {"turnId": "turn_001", "turnGeneration": 7},
    )
    interrupted = projection.publish(
        "assistant_turn_interrupted",
        {"turnId": "turn_002", "turnGeneration": 8},
    )

    assert timing is not None and timing.kind == "speech_timing"
    assert completed.kind == "assistant_turn_completed"
    assert interrupted.kind == "assistant_turn_interrupted"


def test_projection_validates_content_free_knowledge_timing() -> None:
    projection = BrowserEventProjection()
    data = {
        "turnId": "session_7_media_3_utterance_11",
        "route": "current_fact",
        "backend": "ddgs",
        "outcome": "usable",
        "sampleCount": 9,
        "lastMs": 1200,
        "p50Ms": 900,
        "p95Ms": 1800,
        "lookupElapsedMs": 1500,
        "lookupBlockingMs": 500,
        "lookupOverlapMs": 1000,
        "recoveryUsed": False,
    }

    event = projection.publish_advisory("knowledge_timing", data)

    assert event is not None and dict(event.data) == data
    health_data = {
        **data,
        "lookupClosed": False,
        "lookupDetachedCalls": 2,
        "lookupDetachedCallsTotal": 5,
        "lookupSaturationEvents": 1,
    }
    health_event = projection.publish_advisory("knowledge_timing", health_data)
    assert health_event is not None and dict(health_event.data) == health_data
    with pytest.raises(ValueError, match="knowledge timing shape"):
        projection.publish_advisory(
            "knowledge_timing",
            {**data, "lookupDetachedCalls": 2},
        )
    with pytest.raises(ValueError, match="knowledge timing shape"):
        projection.publish_advisory("knowledge_timing", {**data, "query": "private"})
    with pytest.raises(ValueError, match="knowledge timing milliseconds"):
        projection.publish_advisory("knowledge_timing", {**data, "p95Ms": -1})

def test_advisory_partial_drops_without_latching_projection_overflow() -> None:
    projection = BrowserEventProjection(capacity=1)
    projection.publish("notification_queued", {})

    assert (
        projection.publish_advisory(
            "transcript_partial",
            {"role": "user", "text": "droppable words"},
        )
        is None
    )

    projection.acknowledge_through(1)
    final = projection.publish(
        "transcript_final",
        {"role": "user", "text": "authoritative words"},
    )
    assert final.sequence == 2


def test_advisory_drops_without_latching_while_reservations_are_outstanding() -> None:
    # Regression: publish_advisory guarded only on len(events) while
    # ensure_capacity also counts retained capture-status reservations, so a
    # droppable advisory fell through to publish() and permanently latched the
    # projection closed for the rest of the session.
    projection = BrowserEventProjection(capacity=3)
    projection.reserve_capture_status_slots(2)
    projection.publish("notification_queued", {})

    assert (
        projection.publish_advisory(
            "transcript_partial",
            {"role": "user", "text": "droppable words"},
        )
        is None
    )

    projection.acknowledge_through(1)
    final = projection.publish(
        "transcript_final",
        {"role": "user", "text": "authoritative words"},
    )
    assert final.sequence == 2
