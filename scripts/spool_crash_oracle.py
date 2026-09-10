"""Pin the seven synthetic source events used by the crash matrix.

This is a deliberately bounded fixture oracle, not a general payload parser. It
imports neither the candidate's models nor the crash driver. Byte comparison
requires every field and its exact JSON type, including explicit null values.
"""

from __future__ import annotations

from scripts.evidence_protocol_oracle import canonical_json_bytes


def _source_events() -> frozenset[bytes]:
    events = []
    for prefix, offset in (("10000000", 0), ("20000000", 100)):
        epoch = f"{prefix}-0000-4000-8000-000000000003"
        session = f"{prefix}-0000-4000-8000-000000000004"
        binding = f"{prefix}-0000-4000-8000-000000000006"
        consent = {
            "consent_epoch_id": epoch,
            "consent_version": "realtime-evidence-consent-v1",
            "disclosure_digest": "cd" * 32,
        }
        specifications = [
            (
                1,
                offset + 1,
                "session_opened",
                {
                    **consent,
                    "binding_id": binding,
                    "retention_hours": 24,
                    "microphone_accepted": False,
                    "typed_accepted": True,
                    "predecessor_session_id": None,
                },
            ),
            (
                2,
                offset + 2,
                "binding_opened",
                {
                    "binding_id": binding,
                    "binding_generation": 1,
                    "microphone_available": False,
                    "typed_available": True,
                },
            ),
        ]
        if offset == 0:
            specifications.extend(
                [
                    (
                        3,
                        10,
                        "turn_opened",
                        {
                            "evidence_turn_id": "10000000-0000-4000-8000-000000000007",
                            "turn_kind": "user_response",
                            "utterance_id": "10000000-0000-4000-8000-000000000008",
                        },
                    ),
                    (
                        3,
                        20,
                        "binding_closed",
                        {"binding_id": binding, "close_reason": "client_closed"},
                    ),
                    (4, 21, "session_seal_requested", {**consent, "final_event_sequence": 4}),
                ]
            )
        for sequence, event_index, kind, payload in specifications:
            events.append(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "installation_id": "10000000-0000-4000-8000-000000000001",
                        "producer_instance_id": "10000000-0000-4000-8000-000000000002",
                        "logical_session_id": session,
                        "event_id": f"40000000-0000-4000-8000-{event_index:012d}",
                        "event_sequence": sequence,
                        "event_kind": kind,
                        "payload": payload,
                    }
                )
            )
    return frozenset(events)


_SOURCE_EVENTS = _source_events()


def validate_spool_snapshot_v1(snapshot: object) -> None:
    if canonical_json_bytes(snapshot) not in _SOURCE_EVENTS:
        raise ValueError("storage event differs from the pinned synthetic source")
