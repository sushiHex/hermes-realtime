"""Independent V1 JSON and HRE1 protocol oracle for qualification readers.

This module deliberately imports no candidate runtime. Fixed null-tag and
previous-hash vectors pin the byte format separately from the implementation
whose durable records are being qualified.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from uuid import UUID

_KINDS = frozenset(
    {
        "session_opened",
        "binding_opened",
        "turn_opened",
        "user_final_accepted",
        "command_routed",
        "assistant_segment_generated",
        "assistant_chunk_transport_confirmed_full",
        "turn_snapshot",
        "turn_settled",
        "binding_closed",
        "session_seal_requested",
        "session_tainted",
    }
)


def canonical_json_bytes(document: object) -> bytes:
    def inspect(value: object) -> None:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("protocol object keys must be strings")
            for item in value.values():
                inspect(item)
        elif type(value) is list:
            for item in value:
                inspect(item)
        elif type(value) not in {str, int, bool, type(None)}:
            raise ValueError("protocol JSON contains a noncanonical value")

    inspect(document)
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hre1_record_hash(
    *,
    installation_id: str,
    producer_instance_id: str,
    event_id: str,
    logical_session_id: str,
    event_sequence: int,
    event_kind: str,
    recorded_at_utc: str,
    payload_hash: str,
    previous_hash: str | None,
) -> str:
    """Hash the fixed HRE1 framing; lengths and integers are unsigned big endian."""
    frame = bytearray(b"HRE1\x00\x00\x00\x01")
    for identifier in (installation_id, producer_instance_id, event_id, logical_session_id):
        if (
            type(identifier) is not str
            or re.fullmatch(
                "[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                identifier,
            )
            is None
        ):
            raise ValueError("protocol UUID is not canonical version four")
        frame.extend(UUID(identifier).bytes)
    if type(event_sequence) is not int or not 1 <= event_sequence <= 9216:
        raise ValueError("protocol sequence is outside its bound")
    if type(event_kind) is not str or event_kind not in _KINDS:
        raise ValueError("protocol event kind is unknown")
    if (
        type(recorded_at_utc) is not str
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
            recorded_at_utc,
        )
        is None
    ):
        raise ValueError("protocol timestamp spelling differs")
    datetime.strptime(recorded_at_utc, "%Y-%m-%dT%H:%M:%S.%fZ")
    frame.extend(event_sequence.to_bytes(8, "big"))
    for value in (event_kind, recorded_at_utc):
        raw = value.encode("utf-8")
        frame.extend(len(raw).to_bytes(4, "big"))
        frame.extend(raw)
    for digest in (payload_hash,) if previous_hash is None else (payload_hash, previous_hash):
        if type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("protocol digest spelling differs")
    frame.extend(bytes.fromhex(payload_hash))
    frame.append(0 if previous_hash is None else 1)
    if previous_hash is not None:
        frame.extend(bytes.fromhex(previous_hash))
    return hashlib.sha256(frame).hexdigest()
