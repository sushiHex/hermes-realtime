from __future__ import annotations

import hashlib
import json

import pytest

from hermes_realtime.companion.integrity import (
    EXPECTED_HEADER,
    MAX_ARCHIVE_ROWS,
    MAX_BATCH_ROWS,
    MAX_TEXT_CHARS,
    MESSAGE_COLUMNS,
    VOICE_SOURCE,
    ArchiveRefusal,
    Fingerprint,
    Header,
    Identity,
    ProjectedRow,
    Projection,
    VoiceBatch,
    VoiceRow,
    canonical_row,
    check_partition,
    expected_after,
    expected_row_values,
    extend,
    genesis,
    header_bytes,
    plan_archive,
    platform_message_id,
    project,
    split_batch,
    validate_conversation_id,
    voice_metadata,
)

_CONVERSATION = "conv_a"


def _row(seq: int, *, generation: int = 0, role: str = "user", text: str = "hello",
         interrupted: bool = False, timestamp: float = 1_700_000_000.25,
         gap_before: tuple[int, int] | None = None) -> VoiceRow:
    return VoiceRow(
        identity=Identity(generation, seq),
        role=role,
        text=text,
        interrupted=interrupted,
        timestamp=timestamp,
        gap_before=gap_before,
    )


def _batch(start: int, stop: int, *, generation: int = 0) -> tuple[VoiceRow, ...]:
    return tuple(
        _row(seq, generation=generation, role="user" if seq % 2 == 0 else "assistant",
             text=f"row {seq}", interrupted=seq % 4 == 3)
        for seq in range(start, stop)
    )


def _projection(rows: tuple[VoiceRow, ...], header: Header = EXPECTED_HEADER) -> Projection:
    values = [expected_row_values(_CONVERSATION, row) for row in rows]
    return project(
        {
            "parent_session_id": header.parent_session_id,
            "source": header.source,
            "ended_at": None if header.ended_at_is_null else 1.0,
            "end_reason": header.end_reason,
        },
        values,
        MAX_ARCHIVE_ROWS,
        has_children=False,
    )


# --- canonical serialization ---------------------------------------------------------------


def test_canonical_row_is_sorted_compact_typed_json() -> None:
    values = expected_row_values(_CONVERSATION, _row(0))
    encoded = canonical_row(values)
    decoded = json.loads(encoded)
    assert list(decoded) == sorted(MESSAGE_COLUMNS)
    assert encoded == json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
    assert decoded["timestamp"] == ["f", (1_700_000_000.25).hex()]
    assert decoded["content"] == ["s", "hello"]
    assert decoded["tool_calls"] == ["n"]
    assert decoded["active"] == ["i", 1]


@pytest.mark.parametrize(
    ("left", "right"),
    [(1, 1.0), (1, "1"), (None, ""), (b"a", "a"), (0, None)],
    ids=["int-float", "int-str", "null-empty", "bytes-str", "zero-null"],
)
def test_canonical_row_distinguishes_storage_types(left: object, right: object) -> None:
    base = expected_row_values(_CONVERSATION, _row(0))
    assert canonical_row(base | {"token_count": left}) != canonical_row(
        base | {"token_count": right}
    )


def test_canonical_row_requires_exactly_the_message_columns() -> None:
    values = expected_row_values(_CONVERSATION, _row(0))
    with pytest.raises(ValueError):
        canonical_row({key: value for key, value in values.items() if key != "compacted"})
    with pytest.raises(ValueError):
        canonical_row(values | {"id": 7})
    with pytest.raises(TypeError):
        canonical_row(values | {"content": ["not", "a", "column", "value"]})


def test_expected_row_values_name_the_voice_identity() -> None:
    row = _row(3, generation=2, role="assistant", interrupted=True)
    values = expected_row_values(_CONVERSATION, row)
    assert values["platform_message_id"] == "voice:conv_a:2:3"
    assert platform_message_id(_CONVERSATION, row.identity) == "voice:conv_a:2:3"
    assert voice_metadata(row) == {
        "voice": {"gap_before": None, "gen": 2, "interrupted": True, "seq": 3}
    }
    # Hermes stores display metadata with json.dumps' default separators.
    assert values["display_metadata"] == json.dumps(voice_metadata(row))
    assert values["active"] == 1 and values["compacted"] == 0
    assert values["role"] == "assistant" and values["content"] == "hello"


# --- header and chain ----------------------------------------------------------------------


def test_expected_header_is_an_open_voice_session_without_lineage() -> None:
    assert Header(
        parent_session_id=None, source=VOICE_SOURCE, ended_at_is_null=True, end_reason=None
    ) == EXPECTED_HEADER
    assert VOICE_SOURCE == "hermes-realtime-voice"


@pytest.mark.parametrize(
    "changed",
    [
        Header(parent_session_id="other", source=VOICE_SOURCE, ended_at_is_null=True,
               end_reason=None),
        Header(parent_session_id=None, source="cli", ended_at_is_null=True, end_reason=None),
        Header(parent_session_id=None, source=VOICE_SOURCE, ended_at_is_null=False,
               end_reason=None),
        Header(parent_session_id=None, source=VOICE_SOURCE, ended_at_is_null=True,
               end_reason="compression"),
    ],
    ids=["parent", "source", "ended", "end-reason"],
)
def test_every_header_field_changes_the_genesis(changed: Header) -> None:
    assert header_bytes(changed) != header_bytes(EXPECTED_HEADER)
    assert genesis(changed) != genesis(EXPECTED_HEADER)


def test_the_chain_is_sha256_of_the_previous_link_and_the_row_digest() -> None:
    rows = [canonical_row(expected_row_values(_CONVERSATION, row)) for row in _batch(0, 3)]
    link = hashlib.sha256(header_bytes(EXPECTED_HEADER)).digest()
    assert genesis(EXPECTED_HEADER) == Fingerprint(0, link.hex())
    for row in rows:
        link = hashlib.sha256(link + hashlib.sha256(row).digest()).digest()
    assert extend(genesis(EXPECTED_HEADER), rows) == Fingerprint(3, link.hex())


def test_expected_after_precomputes_the_projection_fingerprint() -> None:
    rows = _batch(0, 6)
    committed = expected_after(genesis(EXPECTED_HEADER), _CONVERSATION, rows[:2])
    pending = expected_after(committed, _CONVERSATION, rows[2:])
    assert pending == _projection(rows).fingerprint()
    assert committed == _projection(rows[:2]).fingerprint()


def test_reordering_rows_changes_the_fingerprint() -> None:
    rows = _batch(0, 4)
    swapped = (rows[2], rows[1], rows[0], rows[3])
    assert _projection(rows).fingerprint() != _projection(swapped).fingerprint()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("content", "changed"),
        ("display_metadata", json.dumps({"voice": {"gen": 0, "interrupted": False, "seq": 3}})),
        ("active", 0),
        ("compacted", 1),
        ("display_kind", "voice"),
        ("timestamp", 1.0),
    ],
)
def test_any_column_change_in_any_row_changes_the_fingerprint(column: str, value: object) -> None:
    rows = _batch(0, 4)
    values = [expected_row_values(_CONVERSATION, row) for row in rows]
    header = {"parent_session_id": None, "source": VOICE_SOURCE, "ended_at": None,
              "end_reason": None}
    baseline = project(header, values, MAX_ARCHIVE_ROWS, has_children=False).fingerprint()
    values[3] = values[3] | {column: value}
    assert project(header, values, MAX_ARCHIVE_ROWS, has_children=False).fingerprint() != baseline


def test_fingerprints_are_exact_and_bounded() -> None:
    with pytest.raises(ValueError):
        Fingerprint(0, "A" * 64)
    with pytest.raises(ValueError):
        Fingerprint(-1, "a" * 64)
    with pytest.raises(ValueError):
        Fingerprint(2**53, "a" * 64)
    with pytest.raises(TypeError):
        Fingerprint(True, "a" * 64)
    # A batch that would pass the cap still has a pending fingerprint, so it can be refused.
    assert Fingerprint(MAX_ARCHIVE_ROWS + 1, "a" * 64).count == MAX_ARCHIVE_ROWS + 1


# --- projection bound ----------------------------------------------------------------------


def test_a_projection_over_the_cap_is_refused_never_truncated() -> None:
    rows = _batch(0, 5)
    values = [expected_row_values(_CONVERSATION, row) for row in rows]
    header = {"parent_session_id": None, "source": VOICE_SOURCE, "ended_at": None,
              "end_reason": None}
    assert len(project(header, values, 5, has_children=False).rows) == 5
    with pytest.raises(ArchiveRefusal) as refusal:
        project(header, values, 4, has_children=False)
    assert refusal.value.category == "over_cap"


def test_a_session_that_has_become_a_parent_is_a_lineage_refusal() -> None:
    values = [expected_row_values(_CONVERSATION, row) for row in _batch(0, 2)]
    header = {"parent_session_id": None, "source": VOICE_SOURCE, "ended_at": None,
              "end_reason": None}
    # A child leaves the archive's own rows and header untouched: the chain cannot see it.
    with pytest.raises(ArchiveRefusal) as refusal:
        project(header, values, MAX_ARCHIVE_ROWS, has_children=True)
    assert refusal.value.category == "lineage"
    with pytest.raises(ArchiveRefusal) as refusal:
        project(header, [], MAX_ARCHIVE_ROWS, has_children=True)
    assert refusal.value.category == "lineage"
    with pytest.raises(TypeError):
        project(header, values, MAX_ARCHIVE_ROWS, has_children=0)  # type: ignore[arg-type]


def test_a_projection_keeps_each_rows_platform_message_id() -> None:
    projection = _projection(_batch(0, 2))
    assert [row.platform_message_id for row in projection.rows] == [
        "voice:conv_a:0:0",
        "voice:conv_a:0:1",
    ]
    assert type(projection.rows[0]) is ProjectedRow


# --- row validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"role": "system"},
        {"role": "user", "interrupted": True},
        {"text": "   "},
        {"text": "x" * (MAX_TEXT_CHARS + 1)},
        {"text": "bad \ud800 surrogate"},
        {"timestamp": float("nan")},
        {"timestamp": -1.0},
    ],
    ids=["role", "user-interrupted", "blank", "long", "surrogate", "nan", "negative"],
)
def test_invalid_voice_rows_are_rejected(arguments: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "identity": Identity(0, 0),
        "role": "assistant",
        "text": "fine",
        "interrupted": False,
        "timestamp": 1.0,
    } | arguments
    with pytest.raises(ValueError):
        VoiceRow(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        {"interrupted": 1},
        {"timestamp": 1},
        {"text": b"bytes"},
        {"identity": (0, 0)},
    ],
    ids=["interrupted-int", "timestamp-int", "text-bytes", "identity-tuple"],
)
def test_voice_rows_require_exact_types(arguments: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "identity": Identity(0, 0),
        "role": "assistant",
        "text": "fine",
        "interrupted": False,
        "timestamp": 1.0,
    } | arguments
    with pytest.raises(TypeError):
        VoiceRow(**fields)  # type: ignore[arg-type]


def test_identities_are_exact_bounded_and_ordered() -> None:
    assert Identity(0, 5) < Identity(1, 0) < Identity(1, 1)
    with pytest.raises(TypeError):
        Identity(True, 0)
    with pytest.raises(ValueError):
        Identity(0, -1)
    with pytest.raises(ValueError):
        Identity(2**53, 0)


@pytest.mark.parametrize("conversation", ["", "a:b", "x" * 65, "sp ace", "voice/1"])
def test_conversation_ids_are_bounded_and_colon_free(conversation: str) -> None:
    with pytest.raises(ValueError):
        validate_conversation_id(conversation)


def test_a_valid_conversation_id_is_returned() -> None:
    assert validate_conversation_id("conv_A-1") == "conv_A-1"
    with pytest.raises(TypeError):
        validate_conversation_id(7)  # type: ignore[arg-type]


# --- batch identity ------------------------------------------------------------------------


def _category(error: pytest.ExceptionInfo[ArchiveRefusal]) -> str:
    return error.value.category


def _span(start: int, stop: int, *, generation: int = 0) -> VoiceBatch:
    """A gapless batch covering seq ``start`` to ``stop - 1``."""
    return VoiceBatch(generation, start, stop - 1, _batch(start, stop, generation=generation))


def _gapped() -> VoiceBatch:
    """Rows 0-3, then user row 6 carrying the overflow gap 4-5, then 7: covers 0-7."""
    rows = (*_batch(0, 4), _row(6, gap_before=(4, 5)), _row(7, role="assistant"))
    return VoiceBatch(0, 0, 7, rows)


def test_the_first_batch_starts_at_seq_zero() -> None:
    split = split_batch(None, _span(0, 3))
    assert split.duplicates == () and split.new == _batch(0, 3)
    with pytest.raises(ArchiveRefusal) as refusal:
        split_batch(None, _span(1, 3))
    assert _category(refusal) == "identity"


def test_a_batch_continues_the_cursor_contiguously() -> None:
    split = split_batch(Identity(0, 3), _span(2, 6))
    assert split.duplicates == _batch(2, 4)
    assert split.new == _batch(4, 6)
    with pytest.raises(ArchiveRefusal) as refusal:
        split_batch(Identity(0, 3), _span(5, 6))
    assert _category(refusal) == "identity"


def test_a_fully_acknowledged_batch_is_all_duplicates() -> None:
    split = split_batch(Identity(0, 9), _span(2, 6))
    assert split.duplicates == _batch(2, 6) and split.new == ()


def test_a_new_generation_starts_at_seq_zero() -> None:
    split = split_batch(Identity(0, 9), _span(0, 2, generation=1))
    assert split.new == _batch(0, 2, generation=1)
    with pytest.raises(ArchiveRefusal) as refusal:
        split_batch(Identity(0, 9), _span(1, 2, generation=1))
    assert _category(refusal) == "identity"


def test_a_batch_may_continue_the_cursor_through_a_leading_gap() -> None:
    gapped = VoiceBatch(0, 4, 7, (_row(6, gap_before=(4, 5)), _row(7, role="assistant")))
    split = split_batch(Identity(0, 3), gapped)
    assert split.new == gapped.rows
    # The gap claims seq 4, which the committed cursor already covers.
    with pytest.raises(ArchiveRefusal) as refusal:
        split_batch(Identity(0, 4), gapped)
    assert _category(refusal) == "identity"


# --- the partition: rows plus their gaps cover [seq_from, seq_through] exactly -------------


def test_rows_and_gaps_that_partition_the_range_are_accepted() -> None:
    assert check_partition(_gapped()) == _gapped()
    assert check_partition(_span(3, 5)) == _span(3, 5)
    leading = VoiceBatch(0, 0, 2, (_row(2, gap_before=(0, 1)),))
    assert check_partition(leading) == leading


@pytest.mark.parametrize(
    "batch",
    [
        VoiceBatch(0, 0, 2, (_row(0), _row(2))),
        VoiceBatch(0, 0, 1, (_row(1), _row(0))),
        VoiceBatch(0, 0, 1, (_row(0), _row(0))),
        VoiceBatch(0, 0, 0, (_row(0, generation=1),)),
        VoiceBatch(0, 1, 1, (_row(0),)),
        VoiceBatch(0, 0, 1, (_row(0),)),
        VoiceBatch(0, 0, 0, (_row(0), _row(1, role="assistant"))),
        VoiceBatch(0, 0, 4, (_row(0), _row(4, gap_before=(2, 3)))),
        VoiceBatch(0, 0, 4, (_row(0), _row(4, gap_before=(0, 3)))),
        VoiceBatch(0, 0, 4, (_row(0), _row(4, gap_before=(1, 2)))),
        VoiceBatch(0, 1, 4, (_row(4, gap_before=(2, 3)),)),
    ],
    ids=[
        "hole",
        "descending",
        "repeat",
        "other-generation",
        "starts-after-seq-from",
        "ends-before-seq-through",
        "ends-after-seq-through",
        "gap-leaves-a-hole",
        "gap-overlaps-a-row",
        "row-does-not-follow-its-gap",
        "gap-starts-after-seq-from",
    ],
)
def test_a_batch_that_does_not_partition_its_range_is_refused_whole(batch: VoiceBatch) -> None:
    with pytest.raises(ArchiveRefusal) as refusal:
        check_partition(batch)
    assert _category(refusal) == "partition"
    with pytest.raises(ArchiveRefusal) as refusal:
        split_batch(None, batch)
    assert _category(refusal) == "partition"


@pytest.mark.parametrize(
    "batch",
    [
        VoiceBatch(0, 0, 0, ()),
        VoiceBatch(0, 0, MAX_BATCH_ROWS, tuple(_row(seq) for seq in range(MAX_BATCH_ROWS + 1))),
        VoiceBatch(0, 0, 0, [_row(0)]),  # type: ignore[arg-type]
        VoiceBatch(0, 0, 0, ("row",)),  # type: ignore[arg-type]
        VoiceBatch(True, 0, 0, (_row(0),)),
        VoiceBatch(0, -1, 0, (_row(0),)),
        (_row(0),),
    ],
    ids=["empty", "oversized", "list", "not-a-row", "bool-generation", "negative", "bare-rows"],
)
def test_malformed_batches_are_invalid(batch: object) -> None:
    with pytest.raises(ArchiveRefusal) as refusal:
        check_partition(batch)  # type: ignore[arg-type]
    assert _category(refusal) == "invalid"


# --- gaps on rows --------------------------------------------------------------------------


def test_a_gap_is_hashed_as_part_of_its_row() -> None:
    plain = _row(6)
    gapped = _row(6, gap_before=(4, 5))
    assert voice_metadata(gapped) == {
        "voice": {"gap_before": [4, 5], "gen": 0, "interrupted": False, "seq": 6}
    }
    assert canonical_row(expected_row_values(_CONVERSATION, plain)) != canonical_row(
        expected_row_values(_CONVERSATION, gapped)
    )
    assert expected_after(genesis(EXPECTED_HEADER), _CONVERSATION, (plain,)) != expected_after(
        genesis(EXPECTED_HEADER), _CONVERSATION, (gapped,)
    )


def test_only_a_user_row_may_carry_a_gap() -> None:
    with pytest.raises(ValueError):
        _row(6, role="assistant", gap_before=(4, 5))


@pytest.mark.parametrize(
    ("gap", "error"),
    [
        ([4, 5], TypeError),
        ((4,), TypeError),
        ((4, True), TypeError),
        ((5, 4), ValueError),
        ((-1, 4), ValueError),
    ],
    ids=["list", "short", "bool", "reversed", "negative"],
)
def test_a_gap_is_an_exact_ordered_pair(gap: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        _row(6, gap_before=gap)  # type: ignore[arg-type]


# --- archive planning ----------------------------------------------------------------------


def _fingerprints(
    committed_rows: tuple[VoiceRow, ...], new_rows: tuple[VoiceRow, ...]
) -> tuple[Fingerprint, Fingerprint]:
    committed = expected_after(genesis(EXPECTED_HEADER), _CONVERSATION, committed_rows)
    return committed, expected_after(committed, _CONVERSATION, new_rows)


def test_a_committed_archive_plans_only_the_missing_rows() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    plan = plan_archive(_projection(_batch(0, 4)), _CONVERSATION, _batch(2, 6), committed,
                        pending, MAX_ARCHIVE_ROWS)
    assert plan.inserts == _batch(4, 6)
    assert plan.already_applied is False


def test_an_archive_already_at_pending_is_already_applied() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    plan = plan_archive(_projection(_batch(0, 6)), _CONVERSATION, _batch(4, 6), committed,
                        pending, MAX_ARCHIVE_ROWS)
    assert plan.inserts == () and plan.already_applied is True


def test_an_all_duplicate_retry_is_verified_and_already_applied() -> None:
    committed, pending = _fingerprints(_batch(0, 4), ())
    assert committed == pending
    plan = plan_archive(_projection(_batch(0, 4)), _CONVERSATION, _batch(0, 4), committed,
                        pending, MAX_ARCHIVE_ROWS)
    assert plan.inserts == () and plan.already_applied is True


def test_a_missing_session_is_refused() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(None, _CONVERSATION, _batch(4, 6), committed, pending, MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "missing"


def test_an_archive_matching_neither_fingerprint_is_a_mismatch() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    tampered = _batch(0, 3) + (_row(3, role="assistant", text="row 3", interrupted=False),)
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(tampered), _CONVERSATION, _batch(4, 6), committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "mismatch"


def test_a_changed_payload_under_a_stored_identity_is_a_conflict() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    conflicting = (_row(3, role="assistant", text="other words", interrupted=True),) + _batch(4, 6)
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 4)), _CONVERSATION, conflicting, committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "conflict"


def test_a_stored_row_after_a_missing_one_is_an_identity_error() -> None:
    committed, pending = _fingerprints(_batch(0, 4), ())
    batch = (_row(9), *_batch(0, 1))
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 4)), _CONVERSATION, batch, committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "identity"


def test_an_applied_archive_lacking_a_batch_row_is_an_identity_error() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 6)), _CONVERSATION, _batch(4, 7), committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "identity"


def test_an_insert_past_the_cap_is_refused() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 4)), _CONVERSATION, _batch(4, 6), committed, pending,
                     5)
    assert _category(refusal) == "capacity"
    plan = plan_archive(_projection(_batch(0, 4)), _CONVERSATION, _batch(4, 6), committed,
                        pending, 6)
    assert len(plan.inserts) == 2


def test_refusal_categories_are_closed() -> None:
    with pytest.raises(ValueError):
        ArchiveRefusal("anything")
    assert ArchiveRefusal("conflict").category == "conflict"
    assert ArchiveRefusal("conflict").at_pending is False
    with pytest.raises(TypeError):
        ArchiveRefusal("conflict", at_pending=1)  # type: ignore[arg-type]


# --- only an exact extension is ever inserted ----------------------------------------------


def test_an_old_generation_replay_is_refused_not_inserted() -> None:
    # Generation 0 stopped at seq 3 when generation 1 began; its later seqs were never sent.
    stored = (*_batch(0, 4), *_batch(0, 2, generation=1))
    committed = expected_after(genesis(EXPECTED_HEADER), _CONVERSATION, stored)
    replay = VoiceBatch(0, 4, 5, _batch(4, 6))
    split = split_batch(Identity(1, 1), replay)
    assert split.new == ()  # Every replayed identity is at or before the cursor.
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(stored), _CONVERSATION, replay.rows, committed, committed,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "identity"
    assert refusal.value.at_pending is False


def test_a_row_resent_into_a_recorded_gap_is_refused_not_inserted() -> None:
    stored = (*_batch(0, 4), _row(6, gap_before=(4, 5)), _row(7, role="assistant"))
    committed = expected_after(genesis(EXPECTED_HEADER), _CONVERSATION, stored)
    resent = VoiceBatch(0, 4, 4, (_row(4),))
    assert split_batch(Identity(0, 7), resent).new == ()
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(stored), _CONVERSATION, resent.rows, committed, committed,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "identity"


def test_a_refusal_on_the_already_applied_branch_says_the_archive_holds_pending() -> None:
    committed, pending = _fingerprints(_batch(0, 4), _batch(4, 6))
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 6)), _CONVERSATION, _batch(4, 7), committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "identity" and refusal.value.at_pending is True
    conflicting = (_row(5, role="assistant", text="other"),)
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 6)), _CONVERSATION, conflicting, committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert _category(refusal) == "conflict" and refusal.value.at_pending is True
    with pytest.raises(ArchiveRefusal) as refusal:
        plan_archive(_projection(_batch(0, 4)), _CONVERSATION, conflicting, committed, pending,
                     MAX_ARCHIVE_ROWS)
    assert refusal.value.at_pending is False


# --- text Hermes would not store as text ---------------------------------------------------


@pytest.mark.parametrize(
    "text", ["a\x00b", "\x00json:[1]", "\x00"], ids=["nul", "json-prefix", "bare-nul"]
)
def test_text_hermes_would_decode_as_something_else_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        _row(0, text=text)
