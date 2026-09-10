"""Pin the seven synthetic source events used by the crash matrix.

This is a deliberately bounded fixture oracle, not a general payload parser. It
imports neither the candidate's models nor the crash driver. Byte comparison
requires every field and its exact JSON type, including explicit null values.
"""

from __future__ import annotations

import hashlib
from uuid import UUID

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
        image = _sentinel_image_v1(
            (1, 3, "50000000-0000-4000-8000-000000000002"), (0, 0, None)
        )
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


def erasure_request_digest_v1(state: str) -> str:
    if state not in {"pending", "logical_deleted"}:
        raise ValueError("erasure request state differs")
    finalized = state == "logical_deleted"
    row = [
        "40000000-0000-4000-8000-000000000090",
        "consent_epoch",
        "10000000-0000-4000-8000-000000000003",
        "2026-08-08T00:00:01.000000Z",
        "revoked",
        state,
        7,
        "ab" * 32,
        None,
        None,
        2,
        2 if finalized else None,
        None,
        1 if finalized else None,
        2 if finalized else None,
    ]
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


_Slot = tuple[int, int, str | None]
_PREFIX = b"HRELOCK1\x00\x00\x00\x01"
_STATES = ("clear", "full_purge_pending", "clock_rollback_purge_pending", "first_create_pending")


def _sentinel_image_v1(a: _Slot, b: _Slot) -> bytes:
    image = _PREFIX + bytes(4)
    for generation, state, identifier in (a, b):
        if not 0 <= state <= 3 or generation < 0:
            raise ValueError("sentinel slot differs")
        if state == 0:
            if identifier is not None:
                raise ValueError("clear sentinel has authority")
            raw_id = bytes(16)
        else:
            if generation == 0 or identifier is None or UUID(identifier).version != 4:
                raise ValueError("pending sentinel has no version four authority")
            raw_id = UUID(identifier).bytes
        body = generation.to_bytes(8, "big") + bytes([state]) + bytes(7) + raw_id + bytes(184)
        image += body + hashlib.sha256(_PREFIX + body).digest()
    return image


def sentinel_state_v1(raw: bytes) -> str:
    """Require the matrix's two intact canonical slots, independently of runtime."""
    if type(raw) is not bytes or len(raw) != 512:
        raise ValueError("sentinel image size differs")
    slots: list[_Slot] = []
    for offset in (16, 264):
        body = raw[offset : offset + 216]
        identifier = body[16:32]
        slots.append(
            (
                int.from_bytes(body[:8], "big"),
                body[8],
                None if identifier == bytes(16) else str(UUID(bytes=identifier)),
            )
        )
    a, b = slots
    if raw != _sentinel_image_v1(a, b) or a[0] == b[0]:
        raise ValueError("sentinel bytes or generation order differs")
    return _STATES[max(slots, key=lambda slot: slot[0])[1]]


def checkpoint_sentinel_digest_v1(index: int, clock: str, *, after: bool) -> str | None:
    """Pin both slots across each fixture transition, including the predecessor."""
    if not 0 <= index <= 40 or clock not in {"caught-up", "regressed"}:
        raise ValueError("sentinel checkpoint differs")
    if 9 <= index <= 17:
        return None
    a: _Slot = (1, 3, "50000000-0000-4000-8000-000000000002")
    b: _Slot = (2, 0, None)
    if 18 <= index <= 21 and not after:
        b = (0, 0, None)
    elif 23 <= index <= 27:
        a = (5, 3, "50000000-0000-4000-8000-000000000003")
        b = (6 if after or index == 27 else 4, 0, None)
    elif 29 <= index <= 31:
        a = (3, 2, "50000000-0000-4000-8000-000000000003")
        b = (4 if after else 2, 0, None)
    elif index == 28 and after and clock == "regressed":
        a = (3, 2, "50000000-0000-4000-8000-000000000001")
        b = (4, 0, None)
    elif index >= 32:
        a = (3, 1, "40000000-0000-4000-8000-000000000096")
        b = (4 if after or index == 40 else 2, 0, None)
    return hashlib.sha256(_sentinel_image_v1(a, b)).hexdigest()
