"""Fixed protocol vectors, independent of candidate runtime helpers."""

from __future__ import annotations

import pytest

from scripts.evidence_protocol_oracle import canonical_json_bytes, hre1_record_hash


@pytest.mark.parametrize(
    ("sequence", "kind", "previous", "expected"),
    [
        (
            1,
            "session_opened",
            None,
            "5fd1e46b71e2fd3bbd673b01da1bc32b1293be6870d364c1ac0ffa2e2d5a6172",
        ),
        (
            3,
            "user_final_accepted",
            "11" * 32,
            "af58a6ea26369267612d6185f91cd2aab884cfdd01d49cfbbf0c595a27784c71",
        ),
    ],
)
def test_hre1_fixed_null_and_previous_hash_vectors(sequence, kind, previous, expected) -> None:
    assert (
        hre1_record_hash(
            installation_id="00000000-0000-4000-8000-000000000001",
            producer_instance_id="00000000-0000-4000-8000-000000000002",
            event_id="00000000-0000-4000-8000-000000000003",
            logical_session_id="00000000-0000-4000-8000-000000000004",
            event_sequence=sequence,
            event_kind=kind,
            recorded_at_utc="2026-08-08T00:00:00.000000Z",
            payload_hash="00" * 32,
            previous_hash=previous,
        )
        == expected
    )


def test_canonical_payload_has_sorted_utf8_keys_and_no_terminal_newline() -> None:
    assert canonical_json_bytes({"z": [None, True, 2], "a": "é"}) == (
        b'{"a":"\xc3\xa9","z":[null,true,2]}'
    )


@pytest.mark.parametrize("value", [{"v": 1.0}, {"v": float("nan")}, {1: "v"}, (1, 2)])
def test_canonical_payload_refuses_float_and_foreign_types(value) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes(value)
