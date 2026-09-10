"""Pin the seven synthetic source events used by the crash matrix.

This is a deliberately bounded fixture oracle, not a general payload parser. It
imports neither the candidate's models nor the crash driver. Byte comparison
requires every field and its exact JSON type, including explicit null values.
"""

from __future__ import annotations

import hashlib

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


def initialization_digest_v1(owner: str, stage: str) -> str:
    """Commit the exact V1 bytes supplied to each initialization write."""
    if owner == "root_marker":
        image = (
            b'{"manifestVersion":1,"owner":"hermes-realtime-evidence",'
            b'"rootId":"50000000-0000-4000-8000-000000000001"}\n'
        )
    elif owner == "sentinel":
        prefix = b"HRELOCK1\x00\x00\x00\x01"
        pending = (
            (1).to_bytes(8, "big")
            + b"\x03"
            + bytes(7)
            + bytes.fromhex("50000000000040008000000000000002")
            + bytes(184)
        )
        clear = bytes(216)
        image = prefix + bytes(4)
        for slot in (pending, clear):
            image += slot + hashlib.sha256(prefix + slot).digest()
    else:
        raise ValueError("initialization owner differs")
    if stage == "create":
        image = b""
    elif stage == "partial_write":
        image = image[: len(image) // 2]
    elif stage not in {"full_write", "flush"}:
        raise ValueError("initialization boundary differs")
    return hashlib.sha256(image).hexdigest()


def erasure_receipt_digest_v1(reason: str, event_count: int, *, rollback: bool = False) -> str:
    """Pin all thirteen tombstone columns to the fixture's accepted authority."""
    if reason == "unclean_epoch" and event_count in {2, 3}:
        prefix = "20000000" if rollback else "10000000"
        row = [
            "50000000-0000-4000-8000-000000000001",
            "consent_epoch",
            f"{prefix}-0000-4000-8000-000000000003",
            "unclean_epoch",
            None,
            None,
            None,
            None,
            event_count,
            None,
            "2026-08-08T02:00:01.000000Z",
            1,
            event_count,
        ]
    elif reason == "revoked" and event_count == 2 and not rollback:
        row = [
            "40000000-0000-4000-8000-000000000090",
            "consent_epoch",
            "10000000-0000-4000-8000-000000000003",
            "revoked",
            7,
            "ab" * 32,
            None,
            None,
            2,
            2,
            "2026-08-08T02:00:00.000000Z",
            1,
            2,
        ]
    else:
        raise ValueError("erasure source authority differs")
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()
