from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_realtime.companion.review import review_snapshot_admitted
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.integration import run_record as run_record_module
from hermes_realtime.integration import voice_tail as voice_tail_module
from hermes_realtime.integration.voice_tail import (
    ArchiveBatch,
    ArchiveOutbox,
    OutboxRow,
    ReviewProgress,
    ReviewRange,
    VoiceTail,
    VoiceTailWriter,
    max_voice_tail_bytes,
    parse_voice_tail,
    voice_tail_bytes,
)
from hermes_realtime.speech import Transcript

_MARKER = "[voice-tail] "
_LOCK_MARKER = "[voice-tail-lock] "


def _rows(*rows: tuple[str, str, bool], prior_work: bool = False) -> DurableConversation:
    return DurableConversation(
        messages=tuple(
            ConversationMessage(role, text, interrupted) for role, text, interrupted in rows
        ),
        prior_work=prior_work,
    )


def _markers(output: str, prefix: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(prefix)]


def _writer(path: Path, **kwargs: float) -> VoiceTailWriter:
    return VoiceTailWriter(path, **kwargs)


def _parse(raw: bytes) -> VoiceTail | None:
    return parse_voice_tail(raw, max_messages=16, max_item_chars=64)


def _conversation(raw: bytes) -> DurableConversation:
    tail = _parse(raw)
    assert tail is not None
    return tail.conversation


def _outbox(raw: bytes) -> ArchiveOutbox:
    tail = _parse(raw)
    assert tail is not None and tail.archive is not None
    return tail.archive


def _legacy(view: DurableConversation) -> bytes:
    """A version-1 tail, as the previous release wrote it."""
    document = {
        "messages": [
            {"interrupted": message.interrupted, "role": message.role, "text": message.text}
            for message in view.messages
        ],
        "prior_work": view.prior_work,
        "version": 1,
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _archive(**overrides: object) -> ArchiveOutbox:
    fields: dict[str, object] = {
        "conversation_id": "conv",
        "generation": 0,
        "next_seq": 0,
        "settled": 0,
        "cursor": None,
        "rows": (),
        "frozen": 0,
        "gap": None,
        "review": ReviewProgress(),
    }
    fields.update(overrides)
    return ArchiveOutbox(**fields)  # type: ignore[arg-type]


def _orphan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(b"orphaned plaintext")
    return Path(temporary)


def test_the_tail_is_sorted_compact_versioned_json() -> None:
    tail = _rows(("user", "Hi é", False), ("assistant", "Cut", True), prior_work=True)
    archive = _archive(
        next_seq=4,
        settled=1,
        cursor=0,
        rows=(OutboxRow(2, "user", "Hi é", False, 1.5, (1, 1)),),
        frozen=1,
        gap=(3, 3),
    )

    assert voice_tail_bytes(tail, archive) == (
        b'{"archive":{"conversation_id":"conv","cursor":0,"frozen":1,"gap":[3,3],'
        b'"generation":0,"next_seq":4,"outbox":[{"gap_before":[1,1],"interrupted":false,'
        b'"role":"user","seq":2,"text":"Hi \\u00e9","ts":1.5}],'
        b'"review":{"close_reviewed":false,"close_targets":[],"cursor":null,'
        b'"overflow":false,"pending":null,"recent":[],"reviewed_users":0,"rows":[],'
        b'"users":0},'
        b'"settled":1},'
        b'"messages":[{"interrupted":false,"role":"user","text":"Hi \\u00e9"},'
        b'{"interrupted":true,"role":"assistant","text":"Cut"}],'
        b'"prior_work":true,"version":3}'
    )
    assert _parse(voice_tail_bytes(tail, archive)) == VoiceTail(tail, archive)
    assert _parse(voice_tail_bytes(_rows(), _archive())) == VoiceTail(_rows(), _archive())


@pytest.mark.parametrize(
    ("pending_end", "targets"),
    [
        pytest.param(0, [1], id="pending-stops-short-of-checkpoint"),
        pytest.param(1, [0], id="pending-runs-past-checkpoint"),
        pytest.param(1, [], id="checkpoint-missing"),
    ],
)
@pytest.mark.asyncio
async def test_pending_closing_review_is_bound_to_first_checkpoint_on_restart(
    tmp_path: Path, pending_end: int, targets: list[int]
) -> None:
    conversation = _rows(("user", "Hi", False), ("assistant", "Hello", False))
    pending = ReviewRange("conv", 0, 0, 1, 1, True)
    archive = _archive(
        next_seq=2,
        settled=2,
        cursor=1,
        review=ReviewProgress(
            users=1,
            pending=pending,
            close_targets=(1,),
            rows=((0, True, 40), (1, False, 40)),
        ),
    )
    valid = voice_tail_bytes(conversation, archive)
    assert _parse(valid) == VoiceTail(conversation, archive)

    path = tmp_path / "voice-tail.json"
    run_record_module.write_run_record(path, valid)
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        assert await asyncio.wait_for(writer.next_review(10), 1) == pending
    finally:
        await writer.close()

    malformed = json.loads(valid)
    malformed["archive"]["review"]["pending"]["seq_through"] = pending_end
    malformed["archive"]["review"]["close_targets"] = targets
    assert _parse(json.dumps(malformed).encode()) is None


@pytest.mark.parametrize("closing", [False, True], ids=["nonempty", "empty-closing"])
def test_pending_review_cannot_omit_the_first_retained_row(closing: bool) -> None:
    if closing:
        conversation = _rows(*(("user", f"Q{index}", False) for index in range(4)))
        pending = ReviewRange("conv", 0, 0, 2, 2, True)
        archive = _archive(
            next_seq=4,
            settled=4,
            cursor=3,
            review=ReviewProgress(
                users=3,
                pending=pending,
                close_targets=(2,),
                rows=((0, True, 40), (1, True, 40), (3, True, 40)),
            ),
        )
        malformed_start, malformed_users = 2, 0
    else:
        conversation = _rows(*(("user", f"Q{index}", False) for index in range(3)))
        pending = ReviewRange("conv", 0, 0, 1, 2, False)
        archive = _archive(
            next_seq=3,
            settled=3,
            cursor=2,
            review=ReviewProgress(
                users=3,
                pending=pending,
                rows=((0, True, 40), (1, True, 40), (2, True, 40)),
            ),
        )
        malformed_start, malformed_users = 1, 1
    valid = voice_tail_bytes(conversation, archive)
    assert _parse(valid) == VoiceTail(conversation, archive)
    malformed = json.loads(valid)
    malformed["archive"]["review"]["pending"]["seq_from"] = malformed_start
    malformed["archive"]["review"]["pending"]["users"] = malformed_users
    assert _parse(json.dumps(malformed).encode()) is None


@pytest.mark.asyncio
async def test_periodic_pending_cannot_cross_the_first_close_checkpoint(tmp_path: Path) -> None:
    conversation = _rows(*(("user", f"Q{seq}", False) for seq in range(58, 74)))
    retained = tuple((seq, True, 40) for seq in range(50, 74))
    archive = _archive(
        next_seq=74,
        settled=16,
        cursor=73,
        review=ReviewProgress(
            users=24,
            pending=ReviewRange("conv", 0, 50, 73, 24, False),
            close_targets=(73,),
            rows=retained,
        ),
    )
    valid = voice_tail_bytes(conversation, archive)
    assert _parse(valid) == VoiceTail(conversation, archive)
    before = json.loads(valid)
    before["archive"]["review"]["pending"]["seq_through"] = 72
    before["archive"]["review"]["pending"]["users"] = 23
    assert _parse(json.dumps(before).encode()) is not None
    path = tmp_path / "tail.json"
    run_record_module.write_run_record(path, valid)
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        periodic = await asyncio.wait_for(writer.next_review(10), 2)
        assert periodic == archive.review.pending
        assert writer.acknowledge_review(periodic)
        closing = await asyncio.wait_for(writer.next_review(10), 2)
        assert (closing.seq_from, closing.seq_through, closing.closing) == (50, 73, True)
        assert writer.acknowledge_review(closing)
    finally:
        await writer.close()

    # A restored periodic ACK crossing target 60 would move the actual review
    # cursor to 73, leaving an older close checkpoint without a forward range.
    malformed = json.loads(valid)
    malformed["archive"]["review"]["close_targets"] = [60]
    assert _parse(json.dumps(malformed).encode()) is None


@pytest.mark.parametrize(
    ("review_cursor", "recent", "target", "canonical"),
    [
        # 24 retained 1,000-byte rows: the longest suffix within 16,384 bytes is 16 rows.
        pytest.param(
            50,
            tuple((seq, seq % 2 == 0, 1000) for seq in range(27, 51)),
            74,
            35,
            id="retained-suffix-within-budget",
        ),
        pytest.param(50, (), 74, 50, id="older-tail-replays-the-last-reviewed-row"),
        pytest.param(None, (), 24, 24, id="no-prior-row"),
    ],
)
@pytest.mark.asyncio
async def test_empty_closing_replay_start_is_exact(
    tmp_path: Path,
    review_cursor: int | None,
    recent: tuple[tuple[int, bool, int], ...],
    target: int,
    canonical: int,
) -> None:
    conversation = _rows(("user", "Later", False))
    pending = ReviewRange("conv", 0, canonical, target, int(review_cursor is not None), True)
    archive = _archive(
        next_seq=target + 2,
        settled=1,
        cursor=target + 1,
        review=ReviewProgress(
            cursor=review_cursor,
            users=1 + int(review_cursor is not None),
            reviewed_users=int(review_cursor is not None),
            pending=pending,
            close_targets=(target,),
            rows=((target + 1, True, 40),),
            recent=recent,
        ),
    )
    valid = voice_tail_bytes(conversation, archive)
    assert _parse(valid) == VoiceTail(conversation, archive)
    for wrong_start in (canonical - 1, canonical + 1):
        malformed = json.loads(valid)
        malformed["archive"]["review"]["pending"]["seq_from"] = wrong_start
        assert _parse(json.dumps(malformed).encode()) is None
    if review_cursor is None:
        path = tmp_path / "tail.json"
        run_record_module.write_run_record(path, valid)
        writer = _writer(path)
        await writer.open(ConversationContextStore(on_change=writer.update))
        try:
            planning = asyncio.create_task(writer.next_review(10))
            try:
                await _until(lambda: writer._review.close_targets == ())
            finally:
                planning.cancel()
                await asyncio.gather(planning, return_exceptions=True)
            assert writer._review.pending is None
            assert not writer._review.close_reviewed
            assert writer._review.rows == ((target + 1, True, 40),)
        finally:
            await writer.close()


def test_a_close_checkpoint_cannot_precede_the_review_cursor() -> None:
    conversation = _rows(("user", "Q100", False))
    archive = _archive(
        next_seq=101,
        settled=1,
        cursor=100,
        review=ReviewProgress(cursor=100, users=1, reviewed_users=1, close_targets=(100,)),
    )
    valid = voice_tail_bytes(conversation, archive)
    assert _parse(valid) == VoiceTail(conversation, archive)
    malformed = json.loads(valid)
    malformed["archive"]["review"]["close_targets"] = [1]
    assert _parse(json.dumps(malformed).encode()) is None


def test_reviewed_user_count_cannot_exceed_review_cursor_span() -> None:
    conversation = _rows()
    archive = _archive(
        next_seq=100,
        cursor=99,
        review=ReviewProgress(cursor=0, users=1, reviewed_users=1),
    )
    valid = voice_tail_bytes(conversation, archive)
    assert _parse(valid) == VoiceTail(conversation, archive)
    at_boundary = _archive(
        next_seq=100,
        cursor=99,
        review=ReviewProgress(cursor=99, users=100, reviewed_users=100),
    )
    assert _parse(voice_tail_bytes(conversation, at_boundary)) == VoiceTail(
        conversation, at_boundary
    )
    malformed = json.loads(valid)
    malformed["archive"]["review"]["users"] = 100
    malformed["archive"]["review"]["reviewed_users"] = 100
    assert _parse(json.dumps(malformed).encode()) is None


@pytest.mark.asyncio
async def test_restored_completed_close_survives_until_new_live_activity(tmp_path: Path) -> None:
    conversation = _rows(("user", "Earlier", False))
    archive = _archive(
        next_seq=1,
        settled=1,
        cursor=0,
        review=ReviewProgress(cursor=0, users=1, reviewed_users=1, close_reviewed=True),
    )
    path = tmp_path / "tail.json"
    run_record_module.write_run_record(path, voice_tail_bytes(conversation, archive))
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        assert writer._review.close_reviewed
        writer.request_review_close()
        assert writer._review.close_targets == ()
        store.record_user_transcript(Transcript(text="Later", final=True))
        assert not writer._review.close_reviewed
        writer.request_review_close()
        assert writer._review.close_targets == (1,)
    finally:
        await writer.close()


def test_a_version_one_tail_parses_without_an_archive_so_it_migrates() -> None:
    tail = _rows(("user", "Hi", False), prior_work=True)

    assert _parse(_legacy(tail)) == VoiceTail(tail, None)


def test_the_byte_bound_admits_a_worst_case_tail_at_the_stores_bounds() -> None:
    # One astral character is one str character but twelve escaped bytes.
    tail = DurableConversation(
        messages=tuple(
            ConversationMessage("assistant", "\U0001f600" * 8, interrupted=True) for _ in range(3)
        ),
        prior_work=False,
    )
    rows = tuple(
        OutboxRow(seq, "user", "\U0001f600" * 8, False, 1.0e18, (0, 0) if seq == 1 else None)
        for seq in range(1, 5)
    )
    archive = _archive(
        conversation_id="c" * 64,
        generation=2**53 - 1,
        next_seq=2**53 - 1,
        settled=3,
        cursor=None,
        rows=rows,
        frozen=4,
        gap=(5, 2**53 - 2),
    )
    raw = voice_tail_bytes(tail, archive)
    bound = max_voice_tail_bytes(3, 8, 4)

    def parse(data: bytes) -> VoiceTail | None:
        return parse_voice_tail(data, max_messages=3, max_item_chars=8, max_outbox_rows=4)

    assert len(raw) <= bound
    assert parse(raw) == VoiceTail(tail, archive)
    assert parse(b" " * (bound + 1)) is None
    assert parse(raw + b" " * (bound + 1 - len(raw))) is None
    assert parse(raw + b" " * (bound - len(raw))) == VoiceTail(tail, archive)


def _document(**overrides: object) -> bytes:
    document: dict[str, object] = {
        "messages": [{"interrupted": False, "role": "user", "text": "Hi"}],
        "prior_work": False,
        "version": 1,
    }
    document.update(overrides)
    return json.dumps(document).encode("utf-8")


def _row_document(**overrides: object) -> bytes:
    row: dict[str, object] = {"interrupted": False, "role": "user", "text": "Hi"}
    row.update(overrides)
    return _document(messages=[row])


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe{}", id="not-utf8"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(b"[" * 5_000 + b"]" * 5_000, id="nested-past-the-recursion-limit"),
        pytest.param(
            b'{"messages":[],"prior_work":false,"version":1,"version":1}',
            id="duplicate-top-level-key",
        ),
        pytest.param(
            b'{"messages":[{"interrupted":false,"role":"user","role":"user","text":"Hi"}],'
            b'"prior_work":false,"version":1}',
            id="duplicate-row-key",
        ),
        pytest.param(_document(extra=1), id="extra-top-level-key"),
        pytest.param(
            json.dumps({"prior_work": False, "version": 1}).encode(), id="missing-messages"
        ),
        pytest.param(
            json.dumps({"messages": [], "prior_work": False}).encode(), id="missing-version"
        ),
        pytest.param(json.dumps({"messages": [], "version": 1}).encode(), id="missing-prior-work"),
        pytest.param(_document(prior_work=0), id="prior-work-not-a-boolean"),
        pytest.param(_document(version=2), id="version-two-without-an-archive"),
        pytest.param(_document(version=3), id="unknown-version"),
        pytest.param(_document(version="1"), id="string-version"),
        pytest.param(_document(version=1.0), id="float-version"),
        pytest.param(_document(version=True), id="boolean-version"),
        pytest.param(_document(messages={}), id="messages-not-a-list"),
        pytest.param(_document(messages=["row"]), id="row-not-an-object"),
        pytest.param(_row_document(extra=1), id="extra-row-key"),
        pytest.param(
            _document(messages=[{"role": "user", "text": "Hi"}]),
            id="missing-row-key",
        ),
        pytest.param(_row_document(role=1), id="role-not-a-string"),
        pytest.param(_row_document(text=["Hi"]), id="text-not-a-string"),
        pytest.param(_row_document(interrupted=0), id="interrupted-not-a-boolean"),
        pytest.param(_row_document(role="system"), id="unknown-role"),
        pytest.param(_row_document(interrupted=True), id="interrupted-user-row"),
        pytest.param(_row_document(text="   "), id="blank-text"),
        pytest.param(_row_document(text="said deleg_private"), id="private-run-token"),
        pytest.param(_row_document(text="lone \ud800 surrogate"), id="not-utf8-encodable"),
        pytest.param(
            _document(messages=[{"interrupted": False, "role": "user", "text": "Hi"}] * 17),
            id="more-rows-than-max-messages",
        ),
    ],
)
def test_every_malformed_class_is_refused(raw: bytes) -> None:
    assert _parse(raw) is None


def _v2(archive: dict[str, object] | None = None, **overrides: object) -> bytes:
    fields: dict[str, object] = {
        "conversation_id": "conv",
        "cursor": 0,
        "frozen": 1,
        "gap": None,
        "generation": 0,
        "next_seq": 3,
        "outbox": [
            {
                "gap_before": None,
                "interrupted": False,
                "role": "user",
                "seq": 1,
                "text": "Hi",
                "ts": 1.0,
            },
            {
                "gap_before": None,
                "interrupted": True,
                "role": "assistant",
                "seq": 2,
                "text": "Cut",
                "ts": 2.0,
            },
        ],
        "settled": 1,
    }
    fields.update(overrides)
    document = {
        "archive": fields if archive is None else archive,
        "messages": [{"interrupted": False, "role": "user", "text": "Hi"}],
        "prior_work": False,
        "version": 2,
    }
    return json.dumps(document).encode("utf-8")


def _outbox_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "gap_before": None,
        "interrupted": False,
        "role": "user",
        "seq": 1,
        "text": "Hi",
        "ts": 1.0,
    }
    row.update(overrides)
    return row


def test_a_well_formed_version_two_archive_parses() -> None:
    archive = _outbox(_v2())
    assert (archive.cursor, archive.next_seq, archive.frozen, len(archive.rows)) == (0, 3, 1, 2)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(_v2(archive=[]), id="archive-not-an-object"),
        pytest.param(_v2(extra=1), id="extra-archive-key"),
        pytest.param(_v2(conversation_id="has space"), id="bad-conversation-id"),
        pytest.param(_v2(conversation_id="c" * 65), id="long-conversation-id"),
        pytest.param(_v2(generation=-1), id="negative-generation"),
        pytest.param(_v2(generation=True), id="boolean-generation"),
        pytest.param(_v2(next_seq=2**53), id="seq-past-the-identity-bound"),
        pytest.param(_v2(settled=2), id="settled-past-the-messages"),
        pytest.param(_v2(cursor=1.0), id="float-cursor"),
        pytest.param(_v2(frozen=3), id="frozen-past-the-outbox"),
        pytest.param(_v2(frozen=-1), id="negative-frozen"),
        pytest.param(_v2(outbox={}), id="outbox-not-a-list"),
        pytest.param(_v2(outbox=[_outbox_row(extra=1)]), id="extra-outbox-row-key"),
        pytest.param(_v2(outbox=[_outbox_row(ts=1)]), id="integer-timestamp"),
        pytest.param(_v2(outbox=[_outbox_row(ts=-1.0)]), id="negative-timestamp"),
        pytest.param(_v2(outbox=[_outbox_row(text="x" * 65)]), id="outbox-text-past-the-bound"),
        pytest.param(_v2(outbox=[_outbox_row(text=" ")]), id="blank-outbox-text"),
        pytest.param(_v2(outbox=[_outbox_row(interrupted=True)]), id="interrupted-user-row"),
        pytest.param(
            _v2(outbox=[_outbox_row(role="assistant", gap_before=[1, 1], seq=2)], cursor=0),
            id="gap-on-an-assistant-row",
        ),
        pytest.param(_v2(outbox=[_outbox_row(gap_before=[1, 0], seq=2)]), id="reversed-gap"),
        pytest.param(_v2(outbox=[_outbox_row(gap_before=[1], seq=2)]), id="short-gap"),
        pytest.param(_v2(outbox=[_outbox_row(seq=2)], frozen=0), id="hole-after-the-cursor"),
        pytest.param(
            _v2(outbox=[_outbox_row(seq=1), _outbox_row(seq=1)], frozen=0, next_seq=2),
            id="repeated-seq",
        ),
        pytest.param(
            _v2(outbox=[_outbox_row(gap_before=[1, 1], seq=3)], frozen=0, next_seq=4),
            id="gap-that-does-not-reach-its-row",
        ),
        pytest.param(_v2(next_seq=4), id="seq-assigned-but-unaccounted"),
        pytest.param(_v2(next_seq=2), id="outbox-row-past-next-seq"),
        pytest.param(_v2(gap=[4, 4], next_seq=5), id="trailing-gap-after-a-hole"),
        pytest.param(_v2(gap=[3, 3], next_seq=5), id="trailing-gap-short-of-next-seq"),
        pytest.param(
            _v2(outbox=[_outbox_row(seq=n) for n in range(1, 131)], frozen=0, next_seq=131),
            id="more-outbox-rows-than-the-bound",
        ),
    ],
)
def test_every_malformed_archive_class_is_refused(raw: bytes) -> None:
    assert _parse(raw) is None


@pytest.mark.asyncio
async def test_open_restores_the_tail_before_anything_else_and_reports_only_a_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "state" / "voice-tail-v1.json"
    tail = _rows(
        ("user", "Earlier question", False),
        ("assistant", "Earlier answer", True),
        prior_work=True,
    )
    run_record_module.write_run_record(path, _legacy(tail))
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)

    await writer.open(store)
    try:
        assert store.snapshot().messages == tail.messages
        assert store.snapshot().revision == 1
        assert store.snapshot().terminal_task_count == 1
        assert store.snapshot().active_tasks == ()
        output = capsys.readouterr().out
        assert _markers(output, _MARKER) == [
            _MARKER + '{"outbox":2,"restored":2,"tail_version":1,"version":1}'
        ]
        assert "Earlier" not in output
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_open_without_a_tail_starts_empty_and_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)

    await writer.open(store)
    await writer.close()

    assert store.snapshot().messages == ()
    assert store.snapshot().revision == 0
    assert _markers(capsys.readouterr().out, _MARKER) == []
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"{not json", id="unparseable"),
        pytest.param(
            _legacy(_rows(("user", "x" * 65, False))),
            id="outside-the-stores-bounds",
        ),
    ],
)
async def test_a_refused_tail_starts_a_fresh_conversation_with_one_marker(
    raw: bytes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    path.write_bytes(raw)
    writer = _writer(path)
    store = ConversationContextStore(max_item_chars=64, on_change=writer.update)

    await writer.open(store)
    try:
        assert store.snapshot().messages == ()
        assert store.snapshot().revision == 0
        assert path.read_bytes() == raw
        output = capsys.readouterr().out
        assert _markers(output, _MARKER) == [_MARKER + '{"refusal":"malformed","version":1}']
        assert "x" * 65 not in output

        store.record_user_transcript(Transcript(text="Fresh start", final=True))
    finally:
        await writer.close()

    assert _conversation(path.read_bytes()) == _rows(("user", "Fresh start", False))
    assert _markers(capsys.readouterr().out, _MARKER) == []


@pytest.mark.asyncio
async def test_open_removes_orphaned_plaintext_temporaries_after_locking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    orphan = _orphan(path)
    writer = _writer(path)

    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        assert not orphan.exists()
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_scanner_held_orphan_never_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    tail = _rows(("user", "Kept", False))
    run_record_module.write_run_record(path, _legacy(tail))
    held = _orphan(path)
    real_unlink = Path.unlink

    def scanner_held(self: Path, missing_ok: bool = False) -> None:
        if self == held:
            raise PermissionError(13, "sharing violation")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", scanner_held)
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)

    with caplog.at_level(logging.WARNING, logger=run_record_module.__name__):
        await writer.open(store)
    await writer.close()

    assert store.snapshot().messages == tail.messages
    assert held.exists()
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings == ["orphaned temporary could not be removed (PermissionError)"]


@pytest.mark.asyncio
async def test_an_unreadable_tail_fails_start_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    tail = _rows(("user", "Kept", False))
    run_record_module.write_run_record(path, _legacy(tail))

    def sharing_violation(_path: Path, _max_bytes: int) -> bytes | None:
        raise PermissionError(13, "sharing violation")

    monkeypatch.setattr(voice_tail_module, "read_run_record", sharing_violation)
    failed = _writer(path)
    failed_store = ConversationContextStore(on_change=failed.update)
    with pytest.raises(PermissionError):
        await failed.open(failed_store)
    await failed.close()
    monkeypatch.undo()

    assert failed_store.snapshot().revision == 0
    assert _markers(capsys.readouterr().out, _MARKER) == []
    retry = _writer(path)
    retry_store = ConversationContextStore(on_change=retry.update)
    await retry.open(retry_store)
    await retry.close()
    assert retry_store.snapshot().messages == tail.messages


@pytest.mark.asyncio
async def test_a_second_host_cannot_open_a_live_hosts_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    tail = _rows(("user", "Owned", False))
    run_record_module.write_run_record(path, _legacy(tail))
    first = _writer(path)
    await first.open(ConversationContextStore(on_change=first.update))
    in_flight = _orphan(path)
    capsys.readouterr()
    second = _writer(path)
    second_store = ConversationContextStore(on_change=second.update)

    try:
        with pytest.raises(RuntimeError, match="another host holds the voice tail"):
            await second.open(second_store)

        assert second_store.snapshot().revision == 0
        assert in_flight.exists()
        output = capsys.readouterr().out
        assert _markers(output, _LOCK_MARKER) == [_LOCK_MARKER + '{"cause":"held","version":1}']
        assert _markers(output, _MARKER) == []
        await second.close()
        assert _conversation(path.read_bytes()) == tail
    finally:
        await first.close()
    in_flight.unlink()

    third = _writer(path)
    third_store = ConversationContextStore(on_change=third.update)
    await third.open(third_store)
    await third.close()
    assert third_store.snapshot().messages == tail.messages


class _GatedWrites:
    """Replace the atomic write with one that records bytes and can block or fail."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.written: list[bytes] = []
        self.failures: list[BaseException] = []
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        monkeypatch.setattr(voice_tail_module, "write_run_record", self._write)

    def _write(self, path: Path, data: bytes) -> None:
        self.started.set()
        self.release.wait(timeout=5)
        if self.failures:
            raise self.failures.pop(0)
        self.written.append(data)
        run_record_module.write_run_record(path, data)


async def _until(condition: object, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():  # type: ignore[operator]
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_updates_coalesce_and_the_latest_snapshot_always_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    first = _rows(("user", "one", False))
    writes.release.clear()
    writer.update(first)
    await asyncio.to_thread(writes.started.wait, 2)

    for text in ("two", "three", "four"):
        writer.update(_rows(("user", text, False)))
    writes.release.set()
    await _until(lambda: len(writes.written) == 2)
    await asyncio.sleep(0.05)

    latest = _rows(("user", "four", False))
    assert [_conversation(data) for data in writes.written] == [first, latest]
    await writer.close()
    assert len(writes.written) == 2
    assert _conversation(path.read_bytes()) == latest


def test_update_only_stores_the_latest_snapshot_synchronously(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "voice-tail-v1.json")

    writer.update(_rows(("user", "one", False)))
    writer.update(_rows(("user", "two", False)))

    assert not (tmp_path / "voice-tail-v1.json").exists()
    with pytest.raises(TypeError):
        writer.update(_rows(("user", "x", False)).messages)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_sharing_violation_backs_off_and_retries_with_the_latest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    writes = _GatedWrites(monkeypatch)
    writes.failures.extend([PermissionError(13, "sharing violation"), PermissionError(13, "x")])
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path, initial_backoff_seconds=0.01, max_backoff_seconds=0.02)
    await writer.open(ConversationContextStore(on_change=writer.update))

    with caplog.at_level(logging.WARNING, logger=voice_tail_module.__name__):
        writer.update(_rows(("user", "private words", False)))
        await _until(lambda: len(writes.failures) == 0)
        writer.update(_rows(("user", "newer words", False)))
        await _until(lambda: len(writes.written) >= 1)
        await asyncio.sleep(0.05)

    assert _conversation(writes.written[-1]) == _rows(("user", "newer words", False))
    stale = _rows(("user", "private words", False))
    assert all(_conversation(data) != stale for data in writes.written[1:])
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("words" not in record.getMessage() for record in warnings)
    assert all(str(tmp_path) not in record.getMessage() for record in warnings)
    await writer.close()
    assert _conversation(path.read_bytes()) == _rows(("user", "newer words", False))


@pytest.mark.asyncio
async def test_a_non_os_write_failure_is_retried_and_never_kills_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    writes = _GatedWrites(monkeypatch)
    writes.failures.append(ValueError("serializer bug"))
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path, initial_backoff_seconds=0.01, max_backoff_seconds=0.02)
    await writer.open(ConversationContextStore(on_change=writer.update))

    with caplog.at_level(logging.WARNING, logger=voice_tail_module.__name__):
        writer.update(_rows(("user", "one", False)))
        await _until(lambda: len(writes.written) == 1)
    writer.update(_rows(("user", "two", False)))
    await _until(lambda: len(writes.written) == 2)
    await writer.close()

    assert _conversation(path.read_bytes()) == _rows(("user", "two", False))
    assert [record.getMessage() for record in caplog.records] == [
        "voice tail write failed (ValueError); retrying in 0.01 s"
    ]


@pytest.mark.asyncio
async def test_the_backoff_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _GatedWrites(monkeypatch)
    writes.failures.extend(PermissionError(13, "sharing violation") for _ in range(8))
    delays: list[float] = []

    async def recording_backoff(delay: float) -> None:
        delays.append(delay)
        await asyncio.sleep(0)

    writer = _writer(tmp_path / "voice-tail-v1.json")
    monkeypatch.setattr(writer, "_wait_backoff", recording_backoff)
    await writer.open(ConversationContextStore(on_change=writer.update))

    writer.update(_rows(("user", "one", False)))
    await _until(lambda: len(writes.written) == 1)
    # A success resets the backoff for the next failure.
    writes.failures.append(PermissionError(13, "sharing violation"))
    writer.update(_rows(("user", "two", False)))
    await _until(lambda: len(writes.written) == 2)
    await writer.close()

    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 0.05]


@pytest.mark.asyncio
async def test_close_flushes_the_final_dirty_snapshot_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writes.release.clear()
    writer.update(_rows(("assistant", "Cut", False)))
    await asyncio.to_thread(writes.started.wait, 2)
    writer.update(_rows(("assistant", "Cut", True)))

    closing = asyncio.create_task(writer.close())
    await asyncio.sleep(0.02)
    writes.release.set()
    await asyncio.wait_for(closing, timeout=2)

    assert _conversation(writes.written[-1]) == _rows(("assistant", "Cut", True))
    assert _conversation(path.read_bytes()) == _rows(("assistant", "Cut", True))
    next_owner = _writer(path)
    await next_owner.open(ConversationContextStore(on_change=next_owner.update))
    await next_owner.close()


@pytest.mark.asyncio
async def test_a_failed_final_write_backs_off_fully_and_retries_before_close_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    delays: list[float] = []
    writer = _writer(path, initial_backoff_seconds=0.05, max_backoff_seconds=0.05)
    real_backoff = writer._wait_backoff

    async def recording_backoff(delay: float) -> None:
        delays.append(delay)
        started = asyncio.get_running_loop().time()
        await real_backoff(delay)
        delays.append(round(asyncio.get_running_loop().time() - started, 2))

    monkeypatch.setattr(writer, "_wait_backoff", recording_backoff)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writes.failures.append(PermissionError(13, "sharing violation"))
    writer.update(_rows(("assistant", "Final", True)))

    await asyncio.wait_for(writer.close(), timeout=2)
    assert [_conversation(data) for data in writes.written] == [_rows(("assistant", "Final", True))]
    assert [_conversation(data) for data in writes.written] == [_rows(("assistant", "Final", True))]
    assert delays[0] == 0.05
    # Close never cuts the backoff short.
    assert delays[1] >= 0.04
    assert _conversation(path.read_bytes()) == _rows(("assistant", "Final", True))


@pytest.mark.asyncio
async def test_a_close_timeout_keeps_the_tail_owned_and_a_retried_close_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path, close_timeout_seconds=0.05)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writes.release.clear()
    writer.update(_rows(("user", "slow", False)))
    await asyncio.to_thread(writes.started.wait, 2)

    with pytest.raises(RuntimeError, match="close timeout"):
        await writer.close()

    contender = _writer(path)
    with pytest.raises(RuntimeError, match="another host holds the voice tail"):
        await contender.open(ConversationContextStore(on_change=contender.update))
    writes.release.set()
    # The retried close keeps the same short timeout, which a loaded machine can outlast
    # while the released write finishes; it is retried until it finishes, as a host would.
    for _ in range(100):
        try:
            await writer.close()
            break
        except RuntimeError:
            await asyncio.sleep(0.05)
    else:
        raise AssertionError("a retried close never finished")

    assert _conversation(path.read_bytes()) == _rows(("user", "slow", False))
    successor = _writer(path)
    await successor.open(ConversationContextStore(on_change=successor.update))
    await successor.close()


@pytest.mark.asyncio
async def test_close_waits_for_an_in_flight_write_so_the_newest_lands_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[bytes] = []
    first_started = threading.Event()
    release_first = threading.Event()
    first_finished = threading.Event()

    def write(path: Path, data: bytes) -> None:
        first = not first_started.is_set()
        if first:
            first_started.set()
            release_first.wait(timeout=5)
        written.append(data)
        run_record_module.write_run_record(path, data)
        if first:
            first_finished.set()

    monkeypatch.setattr(voice_tail_module, "write_run_record", write)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writer.update(_rows(("user", "stale", False)))
    await asyncio.to_thread(first_started.wait, 2)
    writer.update(_rows(("user", "newest", False)))

    closing = asyncio.create_task(writer.close())
    await asyncio.sleep(0.05)
    release_first.set()
    await asyncio.wait_for(closing, timeout=2)
    # Every write has landed before the assertions, whichever order they took.
    assert await asyncio.to_thread(first_finished.wait, 2)

    assert _conversation(written[-1]) == _rows(("user", "newest", False))
    assert _conversation(path.read_bytes()) == _rows(("user", "newest", False))


@pytest.mark.asyncio
async def test_close_without_a_change_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    writer = _writer(tmp_path / "voice-tail-v1.json")
    await writer.open(ConversationContextStore(on_change=writer.update))

    await writer.close()
    await writer.close()

    assert writes.written == []


@pytest.mark.asyncio
async def test_a_writer_that_never_opened_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    writer = _writer(tmp_path / "voice-tail-v1.json")
    writer.update(_rows(("user", "before open", False)))

    await writer.close()

    assert writes.written == []
    assert not (tmp_path / "voice-tail-v1.json").exists()


@pytest.mark.asyncio
async def test_open_is_one_shot_and_requires_an_exact_store(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "voice-tail-v1.json")
    with pytest.raises(TypeError, match="store"):
        await writer.open(object())  # type: ignore[arg-type]
    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        with pytest.raises(RuntimeError, match="once"):
            await writer.open(ConversationContextStore())
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_binding_is_available_only_while_tail_is_open(tmp_path: Path) -> None:
    writer = VoiceTailWriter(
        tmp_path / "voice-tail-v1.json", conversation_ids=lambda: "conversation_fixed"
    )
    with pytest.raises(RuntimeError, match="open"):
        _ = writer.binding

    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        assert writer.binding == ("conversation_fixed", 0)
    finally:
        await writer.close()
    with pytest.raises(RuntimeError, match="open"):
        _ = writer.binding


@pytest.mark.parametrize("path", ["voice-tail-v1.json", b"voice-tail-v1.json", 1])
def test_the_tail_path_must_be_an_exact_path(path: object) -> None:
    with pytest.raises(TypeError, match="path"):
        VoiceTailWriter(path)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"close_timeout_seconds": 0.0},
        {"close_timeout_seconds": -1.0},
        {"close_timeout_seconds": float("inf")},
        {"close_timeout_seconds": float("nan")},
    ],
)
def test_the_close_timeout_is_finite_and_positive(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="close timeout"):
        VoiceTailWriter(Path("voice-tail-v1.json"), **kwargs)


# --- the archive outbox (tail version 2) -----------------------------------------------------

_OUTBOX_MARKER = "[voice-tail-outbox] "


class _Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _archiver(path: Path, **kwargs: object) -> VoiceTailWriter:
    ids = iter(f"conv{index}" for index in range(100))
    options: dict[str, object] = {
        "clock": _Clock(),
        "conversation_ids": lambda: next(ids),
        "max_outbox_rows": 4,
        "max_batch_rows": 2,
    }
    options.update(kwargs)
    return VoiceTailWriter(path, **options)  # type: ignore[arg-type]


async def _opened(path: Path, **kwargs: object) -> tuple[VoiceTailWriter, ConversationContextStore]:
    writer = _archiver(path, **kwargs)
    store = ConversationContextStore(max_messages=16, max_item_chars=64, on_change=writer.update)
    await writer.open(store)
    return writer, store


def _say(store: ConversationContextStore, text: str) -> None:
    store.record_user_transcript(Transcript(text=text, final=True))


def _reply(
    store: ConversationContextStore,
    text: str,
    *,
    interrupted: bool = False,
    unsettled: bool = False,
) -> None:
    from hermes_realtime.conversation import AssistantSegmentKey
    from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk

    segment = AssistantSegmentKey()
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_tail",
        chunk_id="chunk_tail",
        text=text,
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = store.prepare_assistant_text(text, segment=segment, heard_text=text)
    ledger.queue(chunk, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(ledger.mark_started("turn_tail", "chunk_tail"))
    store.record_assistant_delivery(admission=admission, ledger=ledger, confirmation=confirmation)
    if unsettled:
        return
    if interrupted:
        store.mark_assistant_segment_interrupted(segment)
    else:
        store.close_assistant_segment(segment)


def _shape(rows: tuple[OutboxRow, ...]) -> list[tuple[int, str, str, bool, object]]:
    return [(row.seq, row.role, row.text, row.interrupted, row.gap_before) for row in rows]


async def _written(path: Path, writer: VoiceTailWriter) -> ArchiveOutbox:
    """The archive the file holds after this version's write completes."""
    target = writer._version
    await _until(lambda: writer._written_version >= target and path.exists())
    return _outbox(path.read_bytes())


@pytest.mark.asyncio
async def test_written_receipt_waits_for_held_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "tail.json"
    writer, store = await _opened(path)
    try:
        _say(store, "seed")
        assert (await _written(path, writer)).rows[0].text == "seed"
        writes.release.clear()
        _say(store, "one")
        receipt = asyncio.create_task(_written(path, writer))
        await asyncio.to_thread(writes.started.wait, 2)
        assert not receipt.done()
        writes.release.set()
        assert (await asyncio.wait_for(receipt, 2)).rows[-1].text == "one"
    finally:
        writes.release.set()
        await writer.close()


@pytest.mark.asyncio
async def test_closed_rows_get_identities_and_close_times_but_an_open_row_never_does(
    tmp_path: Path,
) -> None:
    from hermes_realtime.conversation import AssistantSegmentKey

    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=8)
    try:
        _say(store, "Question")
        segment = AssistantSegmentKey()
        ledger_text = "Answer in progress"
        from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk

        ledger = DeliveredSpeechLedger()
        chunk = SpeechChunk(
            turn_id="turn_open",
            chunk_id="chunk_open",
            text=ledger_text,
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )
        admission = store.prepare_assistant_text(
            ledger_text, segment=segment, heard_text=ledger_text
        )
        ledger.queue(chunk, admission=admission)
        store.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=ledger.mark_delivered_confirmed(
                ledger.mark_started("turn_open", "chunk_open")
            ),
        )
        archive = await _written(path, writer)
        assert _shape(archive.rows) == [(0, "user", "Question", False, None)]
        assert archive.rows[0].ts == 1_700_000_001.0
        assert (archive.next_seq, archive.settled) == (1, 1)

        store.mark_assistant_segment_interrupted(segment)
        archive = await _written(path, writer)
        assert _shape(archive.rows) == [
            (0, "user", "Question", False, None),
            (1, "assistant", ledger_text, True, None),
        ]
        assert archive.rows[1].ts == 1_700_000_002.0
        assert (archive.conversation_id, archive.generation) == ("conv0", 0)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_row_is_eligible_only_after_a_completed_write_holds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "tail.json"
    writer, store = await _opened(path)
    try:
        writes.release.clear()
        _say(store, "Durable first")
        batch = asyncio.create_task(writer.next_batch())
        await asyncio.sleep(0.05)
        assert not batch.done()
        writes.release.set()
        result = await asyncio.wait_for(batch, timeout=2)
        assert _shape(result.rows) == [(0, "user", "Durable first", False, None)]
        assert any(_outbox(data).rows and _outbox(data).frozen == 0 for data in writes.written)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_batch_is_frozen_in_the_tail_before_it_is_returned_and_resent_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=8, max_batch_rows=2)
    try:
        _say(store, "One")
        _reply(store, "Two")
        _say(store, "Three")
        await _written(path, writer)
        first = await asyncio.wait_for(writer.next_batch(), timeout=2)
        # The write that froze it had completed before it was returned.
        frozen = _outbox(writes.written[-1])
        assert frozen.frozen == 2
        assert frozen.rows[:2] == first.rows
        assert (first.conversation_id, first.generation, first.seq_from, first.seq_through) == (
            "conv0",
            0,
            0,
            1,
        )
        _reply(store, "Four")
        await _written(path, writer)
        assert await asyncio.wait_for(writer.next_batch(), timeout=2) == first
    finally:
        await writer.close()

    # A restart resends exactly the frozen batch, close times included.
    successor, _ = await _opened(path, max_outbox_rows=8, max_batch_rows=2)
    try:
        assert await asyncio.wait_for(successor.next_batch(), timeout=2) == first
        assert successor.conversation_id == "conv0"
    finally:
        await successor.close()


@pytest.mark.asyncio
async def test_the_cursor_advances_only_on_an_exact_acknowledgment(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=8, max_batch_rows=2)
    try:
        for text in ("One", "Two", "Three"):
            _say(store, text)
        batch = await asyncio.wait_for(writer.next_batch(), timeout=2)
        for wrong in (
            ("conv1", 0, 0, 1),
            ("conv0", 1, 0, 1),
            ("conv0", 0, 1, 1),
            ("conv0", 0, 0, 0),
            ("conv0", 0, 0, 2),
            ("conv0", False, 0, 1),
        ):
            assert writer.acknowledge(*wrong) is False  # type: ignore[arg-type]
        assert await asyncio.wait_for(writer.next_batch(), timeout=2) == batch

        assert writer.acknowledge("conv0", 0, 0, 1) is True
        assert writer.acknowledge("conv0", 0, 0, 1) is False
        archive = await _written(path, writer)
        assert (archive.cursor, archive.frozen) == (1, 0)
        assert _shape(archive.rows) == [(2, "user", "Three", False, None)]
        after = await asyncio.wait_for(writer.next_batch(), timeout=2)
        assert (after.seq_from, after.seq_through) == (2, 2)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_overflow_discards_unsent_rows_up_to_the_next_user_row_as_its_gap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=4, max_batch_rows=2)
    try:
        _say(store, "Q0")
        _reply(store, "A1")
        await _written(path, writer)
        frozen = await asyncio.wait_for(writer.next_batch(), timeout=2)
        _say(store, "Q2")
        _reply(store, "A3")
        # Full: the oldest unsent row (Q2) and the reply after it go; Q4 carries the gap.
        _say(store, "Q4")
        archive = await _written(path, writer)
        assert _shape(archive.rows) == [
            (0, "user", "Q0", False, None),
            (1, "assistant", "A1", False, None),
            (4, "user", "Q4", False, (2, 3)),
        ]
        assert archive.frozen == 2
        assert await asyncio.wait_for(writer.next_batch(), timeout=2) == frozen
        output = capsys.readouterr().out
        assert _markers(output, _OUTBOX_MARKER) == [
            _OUTBOX_MARKER + '{"cause":"overflow","version":1}'
        ]
        assert "Q2" not in output and "A3" not in output
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_closing_review_ends_at_checkpoint_even_when_its_outbox_row_was_dropped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=4, max_batch_rows=3)
    try:
        _say(store, "Q0")
        first = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            first.conversation_id, first.generation, first.seq_from, first.seq_through
        )
        _say(store, "Q1")
        writer.request_review_close()
        assert writer._review.close_targets == (1,)
        for index in range(2, 6):
            _say(store, f"Q{index}")
        assert [row.seq for row in writer._outbox] == [2, 3, 4, 5]
        later = await asyncio.wait_for(writer.next_batch(), 2)
        assert later.rows[0].gap_before == (1, 1)
        assert writer.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )

        request = await asyncio.wait_for(writer.next_review(10), 2)
        assert (request.seq_from, request.seq_through, request.closing) == (0, 1, True)
        archive = await _written(path, writer)
        assert archive.review is not None and archive.review.pending == request
        assert writer.acknowledge_review(request)
        assert writer._review.close_targets == ()
    finally:
        await writer.close()


@pytest.mark.parametrize("target", [1, 30], ids=["short-gap", "long-gap"])
@pytest.mark.asyncio
async def test_dropped_close_target_after_periodic_review_restarts_without_claiming_later_users(
    tmp_path: Path, target: int
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=4, max_batch_rows=3)
    try:
        _say(store, "Q0")
        first = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            first.conversation_id, first.generation, first.seq_from, first.seq_through
        )
        periodic = await asyncio.wait_for(writer.next_review(1), 2)
        assert (periodic.seq_from, periodic.seq_through, periodic.users, periodic.closing) == (
            0,
            0,
            1,
            False,
        )
        assert writer.acknowledge_review(periodic)

        for index in range(1, target + 1):
            _say(store, f"Q{index}")
        writer.request_review_close()
        for index in range(target + 1, target + 5):
            _say(store, f"Q{index}")
        later = await asyncio.wait_for(writer.next_batch(), 2)
        assert later.rows[0].gap_before == (1, target)
        assert writer.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )
        closing = await asyncio.wait_for(writer.next_review(1), 2)
        assert (closing.seq_from, closing.seq_through, closing.users, closing.closing) == (
            0,
            target,
            1,
            True,
        )
        archive = await _written(path, writer)
        assert archive.review is not None and archive.review.pending == closing
    finally:
        await writer.close()

    restored = _archiver(path, max_outbox_rows=4, max_batch_rows=3)
    await restored.open(ConversationContextStore(on_change=restored.update))
    try:
        assert await asyncio.wait_for(restored.next_review(1), 2) == closing
        assert restored.acknowledge_review(closing)
        assert restored._review.close_targets == ()
        next_periodic = await asyncio.wait_for(restored.next_review(1), 2)
        assert (next_periodic.seq_from, next_periodic.seq_through, next_periodic.users) == (
            target + 1,
            target + 1,
            2,
        )
        assert not next_periodic.closing
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_dropped_first_close_target_is_skipped_without_blocking_later_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=4, max_batch_rows=3)
    try:
        _say(store, "Q0")
        writer.request_review_close()
        before_archive = asyncio.create_task(writer.next_review(10))
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert writer._review.close_targets == (0,)
            assert writer._review.pending is None
        finally:
            before_archive.cancel()
            await asyncio.gather(before_archive, return_exceptions=True)
        for index in range(1, 5):
            _say(store, f"Q{index}")
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert batch.rows[0].gap_before == (0, 0)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        planning = asyncio.create_task(writer.next_review(10))
        try:
            await _until(lambda: writer._review.close_targets == ())
        finally:
            planning.cancel()
            await asyncio.gather(planning, return_exceptions=True)
        assert writer._review.pending is None
        assert writer._review.cursor is None
        assert writer._review.reviewed_users == 0
        assert not writer._review.close_reviewed
        archive = await _written(path, writer)
        assert archive.review is not None and archive.review.close_targets == ()
        assert archive.review.pending is None
        assert [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[voice-review-close] ")
        ] == ['[voice-review-close] {"refusal":"empty_window","version":1}']

        writer.request_review_close()
        assert writer._review.close_targets == (4,)
        later = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )
        closing = await asyncio.wait_for(writer.next_review(10), 2)
        assert (closing.seq_from, closing.seq_through, closing.users, closing.closing) == (
            1,
            4,
            4,
            True,
        )
        assert writer.acknowledge_review(closing)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_restored_empty_closing_pending_is_reconciled_without_losing_later_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    conversation = _rows(*(("user", f"Q{index}", False) for index in range(5)))
    stale = ReviewRange("conv", 0, 0, 0, 0, True)
    archive = _archive(
        next_seq=5,
        settled=5,
        cursor=3,
        rows=(OutboxRow(4, "user", "Q4", False, 1.0),),
        review=ReviewProgress(
            users=3,
            pending=stale,
            close_targets=(0,),
            rows=((1, True, 40), (2, True, 40), (3, True, 40)),
        ),
    )
    raw = voice_tail_bytes(conversation, archive)
    assert _parse(raw) == VoiceTail(conversation, archive)
    run_record_module.write_run_record(path, raw)
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        assert writer.conversation_id == "conv"
        assert len(store.durable_view().messages) == 5
        planning = asyncio.create_task(writer.next_review(10))
        try:
            await _until(lambda: writer._review.close_targets == ())
        finally:
            planning.cancel()
            await asyncio.gather(planning, return_exceptions=True)
        assert writer._review.pending is None
        assert writer._review.reviewed_users == 0
        assert not writer._review.close_reviewed
        assert (await _written(path, writer)).review is not None

        writer.request_review_close()
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        closing = await asyncio.wait_for(writer.next_review(10), 2)
        assert (closing.seq_from, closing.seq_through, closing.users, closing.closing) == (
            1,
            4,
            4,
            True,
        )
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_repeated_trailing_gap_closes_replay_the_last_actual_archived_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=4, max_batch_rows=3)
    try:
        _say(store, "Q0")
        first = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            first.conversation_id, first.generation, first.seq_from, first.seq_through
        )
        periodic = await asyncio.wait_for(writer.next_review(1), 2)
        assert writer.acknowledge_review(periodic)
        assert writer._review.cursor == 0

        for index in range(1, 101):
            _say(store, f"Q{index}")
        writer.request_review_close()
        for index in range(101, 201):
            _say(store, f"Q{index}")
        writer.request_review_close()
        assert writer._review.close_targets == (100, 200)
        for index in range(201, 205):
            _say(store, f"Q{index}")
        later = await asyncio.wait_for(writer.next_batch(), 2)
        assert later.rows[0].gap_before == (1, 200)
        assert writer.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )

        for target in (100, 200):
            closing = await asyncio.wait_for(writer.next_review(1), 2)
            assert (closing.seq_from, closing.seq_through, closing.users, closing.closing) == (
                0,
                target,
                1,
                True,
            )
            assert (await _written(path, writer)).review is not None
            assert writer.acknowledge_review(closing)
            assert writer._review.cursor == 0
        assert writer._review.close_targets == ()
        later_periodic = await asyncio.wait_for(writer.next_review(1), 2)
        assert (later_periodic.seq_from, later_periodic.seq_through, later_periodic.users) == (
            201,
            201,
            2,
        )
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_trailing_unarchivable_gap_does_not_requeue_a_completed_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    timestamps = iter((5.0, float("inf")))
    writer, store = await _opened(path, clock=lambda: next(timestamps))
    try:
        _say(store, "Q0")
        _reply(store, "A1")
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        assert writer._next_seq == 2 and writer._cursor == 0
        assert writer._gap == (1, 1)
        writer.request_review_close()
        closing = await asyncio.wait_for(writer.next_review(10), 2)
        assert (closing.seq_from, closing.seq_through, closing.closing) == (0, 0, True)
        assert writer.acknowledge_review(closing)
        assert writer._review.close_reviewed
        writer.request_review_close()
        assert writer._review.close_targets == ()
    finally:
        await writer.close()

    restored = _archiver(path)
    restored_store = ConversationContextStore(on_change=restored.update)
    await restored.open(restored_store)
    try:
        assert restored._review.close_reviewed
        restored.request_review_close()
        assert restored._review.close_targets == ()
        _say(restored_store, "Q2")
        assert not restored._review.close_reviewed
        restored.request_review_close()
        assert restored._review.close_targets == (2,)
        later = await asyncio.wait_for(restored.next_batch(), 2)
        assert later.rows[0].gap_before == (1, 1)
        assert restored.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )
        final = await asyncio.wait_for(restored.next_review(10), 2)
        assert (final.seq_from, final.seq_through, final.users, final.closing) == (
            2, 2, 2, True
        )
        assert restored.acknowledge_review(final)
        assert restored._review.close_reviewed
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_new_archivable_row_before_close_ack_requires_a_later_close(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path)
    try:
        _say(store, "Q0")
        first = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            first.conversation_id, first.generation, first.seq_from, first.seq_through
        )
        writer.request_review_close()
        first_close = await asyncio.wait_for(writer.next_review(10), 2)
        assert first_close.seq_through == 0 and first_close.closing

        _say(store, "Q1")
        assert writer.acknowledge_review(first_close)
        assert not writer._review.close_reviewed
        writer.request_review_close()
        assert writer._review.close_targets == (1,)
        later = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )
        second_close = await asyncio.wait_for(writer.next_review(10), 2)
        assert (second_close.seq_from, second_close.seq_through, second_close.users) == (
            1, 1, 2
        )
        assert second_close.closing
        assert writer.acknowledge_review(second_close)
        assert writer._review.close_reviewed
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_unsettled_row_before_close_ack_is_reviewed_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path)
    try:
        _say(store, "Q0")
        first = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            first.conversation_id, first.generation, first.seq_from, first.seq_through
        )
        writer.request_review_close()
        first_close = await asyncio.wait_for(writer.next_review(10), 2)
        _reply(store, "A1", unsettled=True)
        assert writer._latest.unsettled == 1
        assert writer.acknowledge_review(first_close)
        assert not writer._review.close_reviewed
    finally:
        await writer.close()

    restored = _archiver(path)
    restored_store = ConversationContextStore(on_change=restored.update)
    await restored.open(restored_store)
    try:
        assert restored._outbox[-1].seq == 1
        assert not restored._review.close_reviewed
        restored.request_review_close()
        assert restored._review.close_targets == (1,)
        later = await asyncio.wait_for(restored.next_batch(), 2)
        assert restored.acknowledge(
            later.conversation_id, later.generation, later.seq_from, later.seq_through
        )
        second_close = await asyncio.wait_for(restored.next_review(10), 2)
        assert (second_close.seq_from, second_close.seq_through, second_close.closing) == (
            1, 1, True
        )
        assert restored.acknowledge_review(second_close)
        assert restored._review.close_reviewed
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_a_trailing_gap_waits_in_the_tail_until_a_user_row_carries_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=3, max_batch_rows=2)
    try:
        _say(store, "Q0")
        _reply(store, "A1")
        await _written(path, writer)
        await asyncio.wait_for(writer.next_batch(), timeout=2)
        _reply(store, "A2")
        # Full, and no user row follows the unsent A2: it becomes a trailing gap, and
        # replies after it join the gap rather than reach the archive without a question.
        _reply(store, "A3")
        _reply(store, "A4")
        archive = await _written(path, writer)
        assert _shape(archive.rows) == [
            (0, "user", "Q0", False, None),
            (1, "assistant", "A1", False, None),
        ]
        assert (archive.gap, archive.next_seq) == ((2, 4), 5)

        _say(store, "Q5")
        archive = await _written(path, writer)
        assert archive.gap is None
        assert _shape(archive.rows)[-1] == (5, "user", "Q5", False, (2, 4))
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_row_evicted_before_it_was_seen_closed_becomes_a_gap(tmp_path: Path) -> None:
    from hermes_realtime.conversation import AssistantSegmentKey
    from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk

    path = tmp_path / "tail.json"
    writer = _archiver(path, max_outbox_rows=4)
    store = ConversationContextStore(max_messages=1, max_item_chars=64, on_change=writer.update)
    await writer.open(store)
    try:
        ledger = DeliveredSpeechLedger()
        chunk = SpeechChunk(
            turn_id="turn_evict",
            chunk_id="chunk_evict",
            text="Open",
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )
        admission = store.prepare_assistant_text(
            "Open", segment=AssistantSegmentKey(), heard_text="Open"
        )
        ledger.queue(chunk, admission=admission)
        store.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=ledger.mark_delivered_confirmed(
                ledger.mark_started("turn_evict", "chunk_evict")
            ),
        )
        _say(store, "Pushes it out")
        archive = await _written(path, writer)
        assert _shape(archive.rows) == [(1, "user", "Pushes it out", False, (0, 0))]
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_batch_is_bounded_by_its_wire_bytes(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer = _archiver(path, max_outbox_rows=16, max_batch_rows=8)
    store = ConversationContextStore(max_messages=16, max_item_chars=4096, on_change=writer.update)
    await writer.open(store)
    try:
        for index in range(3):
            _say(store, f"{index}" + "x" * 4000)
        batch = await asyncio.wait_for(writer.next_batch(), timeout=2)
        # 6 bytes a character at worst: two 4 KB rows fit 48 KiB, three do not.
        assert len(batch.rows) == 1 + (2 * (160 + 6 * 4001) <= 48 * 1024)
        assert type(batch) is ArchiveBatch
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_version_one_tail_migrates_to_version_two_on_the_first_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    tail = _rows(("user", "Old question", False), ("assistant", "Old answer", True))
    path.write_bytes(_legacy(tail))
    writer, store = await _opened(path, max_outbox_rows=8)
    try:
        archive = await _written(path, writer)
        assert _conversation(path.read_bytes()) == tail
        assert archive.conversation_id == "conv0"
        assert _shape(archive.rows) == [
            (0, "user", "Old question", False, None),
            (1, "assistant", "Old answer", True, None),
        ]
        assert (archive.next_seq, archive.settled) == (2, 2)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_restart_continues_the_identities_of_a_version_two_tail(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=8)
    try:
        _say(store, "Before")
        from hermes_realtime.conversation import AssistantSegmentKey
        from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk

        ledger = DeliveredSpeechLedger()
        chunk = SpeechChunk(
            turn_id="turn_cut",
            chunk_id="chunk_cut",
            text="Cut off",
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )
        admission = store.prepare_assistant_text(
            "Cut off", segment=AssistantSegmentKey(), heard_text="Cut off"
        )
        ledger.queue(chunk, admission=admission)
        store.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=ledger.mark_delivered_confirmed(
                ledger.mark_started("turn_cut", "chunk_cut")
            ),
        )
        await _written(path, writer)
    finally:
        await writer.close()
    # The crash cut the open row off: restored, it is final (and flagged), so it closes now.
    successor, restored = await _opened(path, max_outbox_rows=8)
    try:
        _say(restored, "After")
        archive = await _written(path, successor)
        assert archive.conversation_id == "conv0"
        assert _shape(archive.rows) == [
            (0, "user", "Before", False, None),
            (1, "assistant", "Cut off", True, None),
            (2, "user", "After", False, None),
        ]
    finally:
        await successor.close()


@pytest.mark.asyncio
async def test_a_malformed_tail_starts_a_fresh_conversation_and_outbox(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    path.write_bytes(_v2(conversation_id="bad id"))
    writer, store = await _opened(path)
    try:
        _say(store, "Fresh")
        archive = await _written(path, writer)
        assert archive.conversation_id == "conv0"
        assert _shape(archive.rows) == [(0, "user", "Fresh", False, None)]
    finally:
        await writer.close()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_outbox_rows": 2, "max_batch_rows": 2}, ValueError),
        ({"max_outbox_rows": 4097, "max_batch_rows": 2}, ValueError),
        ({"max_outbox_rows": 512, "max_batch_rows": 257}, ValueError),
        ({"max_outbox_rows": 4, "max_batch_rows": 0}, ValueError),
        ({"max_outbox_rows": 4.0, "max_batch_rows": 2}, TypeError),
        ({"clock": 1}, TypeError),
    ],
)
def test_the_outbox_bounds_are_exact_and_leave_room_for_overflow(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        VoiceTailWriter(Path("tail.json"), **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_open_refuses_a_store_whose_rows_cannot_fit_one_batch(tmp_path: Path) -> None:
    writer = _archiver(tmp_path / "tail.json")
    store = ConversationContextStore(max_item_chars=65_536, on_change=writer.update)

    with pytest.raises(ValueError, match="batch"):
        await writer.open(store)


@pytest.mark.asyncio
async def test_a_row_the_companion_would_refuse_becomes_a_gap_before_it_is_eligible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=8)
    try:
        _say(store, "Q0")
        _say(store, "has a \x00 NUL")  # The store holds it; the archive never would.
        _reply(store, "A2")
        _say(store, "Q3")
        archive = await _written(path, writer)
        # The refused row and the reply that depended on it are one gap on the next user row.
        assert _shape(archive.rows) == [
            (0, "user", "Q0", False, None),
            (3, "user", "Q3", False, (1, 2)),
        ]
        batch = await asyncio.wait_for(writer.next_batch(), timeout=2)
        assert all("\x00" not in row.text for row in batch.rows)
        output = capsys.readouterr().out
        assert _markers(output, _OUTBOX_MARKER) == [
            _OUTBOX_MARKER + '{"cause":"invalid","version":1}'
        ]
        assert "NUL" not in output
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_close_time_the_companion_would_refuse_becomes_a_gap(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    times = iter([float("inf"), 5.0, 6.0])
    writer, store = await _opened(path, max_outbox_rows=8, clock=lambda: next(times))
    try:
        _say(store, "Q0")
        _say(store, "Q1")
        archive = await _written(path, writer)
        assert _shape(archive.rows) == [(1, "user", "Q1", False, (0, 0))]
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_the_discard_marker_is_once_per_episode_and_an_ack_ends_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=3, max_batch_rows=2)
    try:
        _say(store, "Q0")
        _reply(store, "A1")
        await _written(path, writer)
        batch = await asyncio.wait_for(writer.next_batch(), timeout=2)
        for index in range(2, 8):  # Overflows repeatedly while the batch is in flight.
            _say(store, f"Q{index}")
        assert len(_markers(capsys.readouterr().out, _OUTBOX_MARKER)) == 1
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        await _written(path, writer)
        await asyncio.wait_for(writer.next_batch(), timeout=2)
        for index in range(8, 12):
            _say(store, f"Q{index}")
        assert len(_markers(capsys.readouterr().out, _OUTBOX_MARKER)) == 1
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_marker_that_cannot_print_never_raises_into_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=3, max_batch_rows=2)
    try:
        _say(store, "Q0")
        _reply(store, "A1")
        await _written(path, writer)
        await asyncio.wait_for(writer.next_batch(), timeout=2)
        closed = io.StringIO()
        closed.close()
        monkeypatch.setattr(sys, "stdout", closed)
        for index in range(2, 6):
            _say(store, f"Q{index}")  # Overflow: the marker's print raises; speech does not.
        monkeypatch.undo()
        assert len(store.snapshot().messages) == 6
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_restart_restores_the_frozen_batch_exactly_under_other_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=8, max_batch_rows=3)
    try:
        for index in range(4):
            _say(store, f"Q{index}")
        frozen = await asyncio.wait_for(writer.next_batch(), timeout=2)
        assert len(frozen.rows) == 3
    finally:
        await writer.close()

    # A smaller batch bound after the restart: the frozen batch is resent whole, as frozen.
    successor, restored = await _opened(path, max_outbox_rows=4, max_batch_rows=2)
    try:
        assert await asyncio.wait_for(successor.next_batch(), timeout=2) == frozen
        # Overflow after the restart discards only unsent rows, never the frozen batch.
        for index in range(4, 9):
            _say(restored, f"Q{index}")
        archive = await _written(path, successor)
        assert archive.rows[:3] == frozen.rows
        assert archive.frozen == 3
        assert await asyncio.wait_for(successor.next_batch(), timeout=2) == frozen
    finally:
        await successor.close()


# --- review windows within the companion's snapshot budget -------------------------------


def _companion_snapshot_bytes(rows: list[tuple[str, str]]) -> int:
    """What the companion measures before it admits a review window."""
    snapshot = [{"role": role, "content": text} for role, text in rows]
    return len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))


def test_a_review_row_costs_exactly_its_share_of_the_companion_snapshot() -> None:
    rows = [
        ("user", "Plain question?"),
        ("assistant", 'Quotes " and \\ and a control \x01 character.'),
        ("assistant", "Accents é, ideographs 漢字 and a bird 🐦."),
    ]
    costs = [voice_tail_module.review_row_bytes(role, text) for role, text in rows]

    assert 2 + sum(costs) + 2 * (len(rows) - 1) == _companion_snapshot_bytes(rows)
    assert voice_tail_module.REVIEW_SNAPSHOT_BYTES == 16_384
    assert voice_tail_module.review_row_bytes("user", "x" * 20_000) == 16_385


def test_a_review_window_is_the_longest_prefix_within_the_budget_and_never_empty() -> None:
    window = voice_tail_module._review_window
    # Brackets, two rows and one ", " separator: exactly 16,384 bytes fits.
    assert window([(0, True, 8190), (1, False, 8190)]) == [(0, True, 8190), (1, False, 8190)]
    assert window([(0, True, 8190), (1, False, 8191)]) == [(0, True, 8190)]
    assert window([(0, True, 16_385), (1, False, 1)]) == [(0, True, 16_385)]
    assert window([(0, True, 10), (1, False, 10), (2, True, 16_360)]) == [
        (0, True, 10),
        (1, False, 10),
    ]


async def _archive_everything(writer: VoiceTailWriter) -> None:
    while writer._outbox:
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )


async def _review_store(path: Path) -> tuple[VoiceTailWriter, ConversationContextStore]:
    writer = _archiver(path, max_outbox_rows=64, max_batch_rows=8)
    store = ConversationContextStore(max_item_chars=1024, on_change=writer.update)
    await writer.open(store)
    return writer, store


def _texts_by_seq(store: ConversationContextStore) -> dict[int, tuple[str, str]]:
    view = store.durable_view()
    return {
        view.first + index: (message.role, message.text)
        for index, message in enumerate(view.messages)
    }


@pytest.mark.asyncio
async def test_a_periodic_review_window_fits_the_companions_byte_budget(tmp_path: Path) -> None:
    writer, store = await _review_store(tmp_path / "tail.json")
    try:
        for turn in range(10):
            _say(store, f"Question {turn}?")
            # Two UTF-8 bytes a character: a whole reply is about 2 KB in the snapshot.
            _reply(store, "é" * 1000)
        await _archive_everything(writer)
        rows = _texts_by_seq(store)

        first = await asyncio.wait_for(writer.next_review(10), 2)
        window = [rows[seq] for seq in range(first.seq_from, first.seq_through + 1)]
        # Twenty rows is within the 24-row limit; the bytes bound the window first.
        assert (first.seq_from, first.closing) == (0, False)
        assert _companion_snapshot_bytes(window) <= 16_384
        assert _companion_snapshot_bytes([*window, rows[first.seq_through + 1]]) > 16_384
        assert writer.acknowledge_review(first)

        second = await asyncio.wait_for(writer.next_review(10), 2)
        assert second.seq_from == first.seq_through + 1
        window = [rows[seq] for seq in range(second.seq_from, second.seq_through + 1)]
        assert _companion_snapshot_bytes(window) <= 16_384
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_closing_window_over_budget_is_not_final_until_it_covers_every_row(
    tmp_path: Path,
) -> None:
    writer, store = await _review_store(tmp_path / "tail.json")
    try:
        # Eight turns, below the review interval: sixteen rows, over 16 KB in all.
        for turn in range(8):
            _say(store, f"Question {turn}?")
            _reply(store, "é" * 1000)
        await _archive_everything(writer)
        writer.request_review_close()
        rows = _texts_by_seq(store)
        assert _companion_snapshot_bytes(list(rows.values())) > 16_384

        partial = await asyncio.wait_for(writer.next_review(10), 2)
        assert (partial.seq_from, partial.closing) == (0, False)
        assert partial.seq_through < 15
        window = [rows[seq] for seq in range(partial.seq_from, partial.seq_through + 1)]
        assert _companion_snapshot_bytes(window) <= 16_384
        assert writer.acknowledge_review(partial)

        final = await asyncio.wait_for(writer.next_review(10), 2)
        assert (final.seq_from, final.seq_through, final.closing) == (
            partial.seq_through + 1,
            15,
            True,
        )
        assert writer.acknowledge_review(final)
        assert writer._review.close_targets == ()
    finally:
        await writer.close()


def test_review_rows_an_older_build_retained_are_costed_as_the_largest_row() -> None:
    conversation = _rows(("user", "Q0", False))
    archive = _archive(
        next_seq=1,
        settled=1,
        cursor=0,
        review=ReviewProgress(users=1, rows=((0, True, 40),)),
    )
    document = json.loads(voice_tail_bytes(conversation, archive))
    assert document["archive"]["review"]["rows"] == [[0, True, 40]]

    document["archive"]["review"]["rows"] = [[0, True]]
    tail = _parse(json.dumps(document).encode())
    assert tail is not None and tail.archive is not None and tail.archive.review is not None
    # The worst a 64-character row can cost: six bytes a character, plus its fields.
    assert tail.archive.review.rows == ((0, True, 36 + 6 * 64),)

    for cost in (0, 16_386, True, 1.0):
        document["archive"]["review"]["rows"] = [[0, True, cost]]
        assert _parse(json.dumps(document).encode()) is None


# The heaviest text a row can hold under the companion's JSON: a control character is a
# six-byte escape, an astral character four UTF-8 bytes, an ideograph three.
_MAXIMAL_TEXTS = ("\x01", "\U0001f426", "漢")


@pytest.mark.parametrize("character", _MAXIMAL_TEXTS)
def test_one_maximal_row_of_the_default_store_always_fits_one_review(character: str) -> None:
    # The at-least-one-row rule then never builds a window the companion refuses.
    text = character * ConversationContextStore().max_item_chars
    snapshot = [{"role": "assistant", "content": text}]

    assert review_snapshot_admitted(snapshot)
    assert 2 + voice_tail_module.review_row_bytes("assistant", text) <= (
        voice_tail_module.REVIEW_SNAPSHOT_BYTES
    )


def _maximal_turns(store: ConversationContextStore, turns: int) -> dict[int, tuple[str, str]]:
    """Say ``turns`` maximal user and reply rows; their text by seq."""
    rows: dict[int, tuple[str, str]] = {}
    for turn in range(turns):
        user = _MAXIMAL_TEXTS[turn % 3] * store.max_item_chars
        reply = _MAXIMAL_TEXTS[(turn + 1) % 3] * store.max_item_chars
        _say(store, user)
        _reply(store, reply)
        rows[2 * turn], rows[2 * turn + 1] = ("user", user), ("assistant", reply)
    return rows


def _snapshot(rows: dict[int, tuple[str, str]], request: ReviewRange) -> list[dict[str, str]]:
    """What the companion reads for ``request``: every archived row in its range."""
    return [
        {"role": role, "content": text}
        for seq, (role, text) in sorted(rows.items())
        if request.seq_from <= seq <= request.seq_through
    ]


@pytest.mark.asyncio
async def test_every_main_path_window_of_maximal_rows_passes_the_companions_own_check(
    tmp_path: Path,
) -> None:
    writer, store = await _review_store(tmp_path / "tail.json")
    try:
        rows = _maximal_turns(store, 12)
        await _archive_everything(writer)
        requests: list[ReviewRange] = []
        # Periodic windows until ten users are reviewed, then the close covers the rest.
        while writer._review.reviewed_users < 10:
            requests.append(await asyncio.wait_for(writer.next_review(10), 2))
            assert writer.acknowledge_review(requests[-1])
        writer.request_review_close()
        while writer._review.close_targets:
            requests.append(await asyncio.wait_for(writer.next_review(10), 2))
            assert writer.acknowledge_review(requests[-1])

        assert requests[-1].closing and requests[-1].seq_through == 23
        assert any(not request.closing and request.seq_from > 19 for request in requests)
        for request in requests:
            assert review_snapshot_admitted(_snapshot(rows, request))
        # Together the windows cover every row exactly once.
        assert [seq for request in requests for seq in
                range(request.seq_from, request.seq_through + 1)] == list(range(24))
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_an_empty_close_replays_the_longest_fitting_suffix_of_reviewed_maximal_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _review_store(path)
    try:
        rows = _maximal_turns(store, 6)
        await _archive_everything(writer)
        while writer._review.rows:
            periodic = await asyncio.wait_for(writer.next_review(1), 2)
            assert review_snapshot_admitted(_snapshot(rows, periodic))
            assert writer.acknowledge_review(periodic)
        assert [seq for seq, _, _ in writer._review.recent] == list(range(12))

        # Every row is reviewed, so the close has nothing new: it replays.
        writer.request_review_close()
        replay = await asyncio.wait_for(writer.next_review(1), 2)
        assert (replay.seq_through, replay.closing) == (11, True)
        replayed = _snapshot(rows, replay)
        assert review_snapshot_admitted(replayed)
        one_more = replace(replay, seq_from=replay.seq_from - 1)
        assert not review_snapshot_admitted(_snapshot(rows, one_more))
        # The frozen replay survives a restart, checked against the retained costs.
        archive = await _written(path, writer)
        assert archive.review is not None and archive.review.pending == replay
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_acknowledged_reviews_retain_the_last_24_reviewed_rows_with_their_costs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.json"
    writer, store = await _opened(path, max_outbox_rows=64, max_batch_rows=8)
    try:
        for index in range(30):
            _say(store, f"Q{index}")
        await _archive_everything(writer)
        while writer._review.rows:
            assert writer.acknowledge_review(await asyncio.wait_for(writer.next_review(1), 2))

        recent = writer._review.recent
        assert [seq for seq, _, _ in recent] == list(range(6, 30))
        assert recent[-1] == (29, True, voice_tail_module.review_row_bytes("user", "Q29"))
        assert writer._review.cursor == 29
        assert (await _written(path, writer)).review == writer._review
    finally:
        await writer.close()


def _recent_document(recent: object, cursor: int | None = 30) -> bytes:
    archive = _archive(
        next_seq=31,
        settled=1,
        cursor=30,
        review=ReviewProgress(cursor=30, users=31, reviewed_users=31),
    )
    document = json.loads(voice_tail_bytes(_rows(("user", "Q30", False)), archive))
    review = document["archive"]["review"]
    review["recent"] = recent
    review["cursor"] = cursor
    if cursor is None:
        review["users"] = review["reviewed_users"] = 0
    return json.dumps(document).encode()


def test_retained_review_rows_are_strict_and_end_at_the_review_cursor() -> None:
    valid = _parse(_recent_document([[29, False, 40], [30, True, 41]]))
    assert valid is not None and valid.archive is not None and valid.archive.review is not None
    assert valid.archive.review.recent == ((29, False, 40), (30, True, 41))
    assert _parse(_recent_document([[seq, True, 40] for seq in range(7, 31)])) is not None
    assert _parse(_recent_document([], None)) is not None

    for recent, cursor in (
        ([[29, True, 40]], 30),  # ends before the cursor
        ([[30, True, 40]], None),  # nothing reviewed
        ([[30, True, 40], [29, True, 40]], 30),  # out of order
        ([[30, True, 40], [30, True, 40]], 30),  # repeated
        ([[30, True]], 30),  # no cost
        ([[30, True, 0]], 30),
        ([[30, True, 16_386]], 30),
        ([[30, 1, 40]], 30),
        ([[seq, True, 40] for seq in range(6, 31)], 30),  # 25 rows
        ({"30": 40}, 30),
    ):
        assert _parse(_recent_document(recent, cursor)) is None, recent

    # An older version-4 tail without retained rows still parses.
    legacy = json.loads(_recent_document([]))
    del legacy["archive"]["review"]["recent"]
    tail = _parse(json.dumps(legacy).encode())
    assert tail is not None and tail.archive is not None and tail.archive.review is not None
    assert tail.archive.review.recent == ()
