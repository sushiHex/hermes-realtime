"""A voice delete survives anything that resets the tail, and settles one binding at a time."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.integration.run_record import unlock_run_record
from hermes_realtime.integration.voice_deletes import (
    MAX_PENDING_DELETES,
    VoiceDeletes,
    parse_voice_deletes,
    voice_deletes_bytes,
)
from hermes_realtime.integration.voice_tail import VoiceTailWriter
from hermes_realtime.speech import Transcript
from tests.support.live_record import read_live_record


def _ids(*names: str):  # type: ignore[no-untyped-def]
    return iter(names).__next__


async def _open(path: Path, *names: str, **store_options: int) -> tuple[
    VoiceTailWriter, ConversationContextStore
]:
    writer = VoiceTailWriter(path, conversation_ids=_ids(*names))
    store = ConversationContextStore(on_change=writer.update, **store_options)
    await writer.open(store)
    return writer, store


def _deletes_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.deletes{path.suffix}")


async def _offer(writer: VoiceTailWriter, store: ConversationContextStore) -> None:
    """Let one batch leave the host, so the companion may hold this binding."""
    store.record_user_transcript(Transcript(text="said before the delete", final=True))
    await asyncio.wait_for(writer.next_batch(), 5)


async def _forget(
    writer: VoiceTailWriter, store: ConversationContextStore
) -> tuple[str, int]:
    await _offer(writer, store)
    return await writer.request_forget(store)


def _markers(output: str, prefix: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(prefix))
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", ["corrupt", "refused_restore", "rolled_back"])
async def test_a_pending_delete_survives_any_tail_reset(
    reset: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "old", "new")
    store.record_user_transcript(Transcript(text="a" * 40, final=True))
    old = await _forget(writer, store)
    store.record_user_transcript(Transcript(text="b" * 40, final=True))
    await writer.close()
    if reset == "corrupt":
        path.write_bytes(b"{not json")
    elif reset == "rolled_back":
        # A build without the delete record rewrites the tail in its own version.
        document = json.loads(path.read_bytes())
        document["version"] = 3
        path.write_bytes(json.dumps(document).encode())
    options = {"max_item_chars": 20} if reset == "refused_restore" else {}

    reopened, _store = await _open(path, "fresh", **options)

    assert reopened.pending_deletes == (old,)
    if reset != "rolled_back":
        assert _markers(capsys.readouterr().out, "[voice-tail] ")[-1] == {
            "refusal": "malformed", "version": 1
        }
    await reopened.close()


@pytest.mark.asyncio
async def test_a_tail_still_bound_to_a_recorded_delete_is_retired_when_it_opens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "old")
    store.record_user_transcript(Transcript(text="said before the delete", final=True))
    await writer.close()
    # The delete was recorded, then the process died before the cleared tail was written.
    _deletes_path(path).write_bytes(voice_deletes_bytes(VoiceDeletes(pending=(("old", 0),))))

    reopened, reopened_store = await _open(path, "new")

    assert reopened_store.snapshot().messages == ()
    assert reopened.binding == ("new", 1)
    assert reopened.pending_deletes == (("old", 0),)
    assert {"retired": 1, "version": 1} in _markers(capsys.readouterr().out, "[voice-tail] ")
    await reopened.close()


@pytest.mark.asyncio
async def test_an_unreadable_delete_record_reads_as_unknown_never_idle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail.json"
    _deletes_path(path).write_bytes(b"garbage")

    writer, _store = await _open(path, "conv")

    assert writer.pending_deletes == ()
    assert writer.deletes_lost is True and writer.delete_outcome is None
    assert _markers(capsys.readouterr().out, "[voice-deletes] ") == [
        {"refusal": "malformed", "version": 1}
    ]
    await writer.close()
    # The finding is kept durably, so a restart cannot forget it.
    reopened, _store = await _open(path, "conv2")
    assert reopened.deletes_lost is True
    await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settled_by", ["companion", "host"])
async def test_a_lost_delete_record_stays_known_after_later_deletes_complete(
    settled_by: str, tmp_path: Path
) -> None:
    path = tmp_path / "voice-tail.json"
    _deletes_path(path).write_bytes(b"garbage")
    writer, store = await _open(path, "a", "b", "c")

    if settled_by == "companion":
        old = await _forget(writer, store)
        assert writer.acknowledge_forget(old) is True
    else:
        # Never archived, so the host itself completes it.
        old = await writer.request_forget(store)

    # The newer delete is complete, yet the lost intents may still be unfinished in Hermes.
    assert writer.delete_outcome == old and writer.deleted_previous is True
    assert writer.deletes_lost is True
    await writer.close()
    assert _recorded(path).lost is True


@pytest.mark.asyncio
async def test_a_pending_delete_never_blocks_a_later_one(tmp_path: Path) -> None:
    writer, store = await _open(tmp_path / "voice-tail.json", "a", "b", "c")

    first = await _forget(writer, store)
    second = await _forget(writer, store)

    assert writer.pending_deletes == (first, second) == (("a", 0), ("b", 1))
    await writer.close()


@pytest.mark.asyncio
async def test_pending_deletes_are_bounded(tmp_path: Path) -> None:
    names = [f"c{number}" for number in range(MAX_PENDING_DELETES + 2)]
    writer, store = await _open(tmp_path / "voice-tail.json", *names)
    for _ in range(MAX_PENDING_DELETES):
        # A request waits for its durable write; a write that keeps failing fails the test.
        await asyncio.wait_for(_forget(writer, store), 5)
    binding = writer.binding

    with pytest.raises(RuntimeError, match="capacity"):
        await asyncio.wait_for(_forget(writer, store), 5)

    assert writer.binding == binding and len(writer.pending_deletes) == MAX_PENDING_DELETES
    await writer.close()


@pytest.mark.asyncio
async def test_a_completed_delete_is_scoped_to_the_conversation_it_retired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "a", "b", "c")
    first = await _forget(writer, store)
    second = await _forget(writer, store)

    assert writer.acknowledge_forget(first) is True
    assert writer.acknowledge_forget(first) is False
    # The first delete retired "a", which "b" replaced; the current conversation is "c".
    assert writer.pending_deletes == (second,)
    assert writer.delete_outcome == first and writer.deleted_previous is False
    assert writer.acknowledge_forget(second) is True
    assert writer.delete_outcome == second and writer.deleted_previous is True
    await writer.close()
    reopened, _store = await _open(path, "d")
    assert reopened.delete_outcome == second and reopened.deleted_previous is True
    await reopened.close()


@pytest.mark.asyncio
async def test_an_intent_in_a_version_4_tail_moves_into_the_record(tmp_path: Path) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "new")
    store.record_user_transcript(Transcript(text="after the delete", final=True))
    await writer.close()
    document = json.loads(path.read_bytes())
    document["archive"]["generation"] = 1
    document |= {"version": 4, "pending_forget": ["old", 0], "forget_complete": False}
    path.write_bytes(json.dumps(document).encode())

    reopened, _store = await _open(path)

    assert reopened.pending_deletes == (("old", 0),) and reopened.binding == ("new", 1)
    await reopened.close()
    assert parse_voice_deletes(_deletes_path(path).read_bytes()) == VoiceDeletes(
        pending=(("old", 0),), live=("new", 1)
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"outcome":null,"pending":[],"version":1}',
        b'{"lost":false,"live":null,"outcome":null,"pending":[],"version":2}',
        b'{"lost":false,"live":null,"outcome":null,"pending":[],"version":true}',
        b'{"lost":false,"live":null,"outcome":null,"pending":[["a",0],["a",0]],"version":1}',
        b'{"lost":false,"live":null,"outcome":["a",0],"pending":[["a",0]],"version":1}',
        b'{"lost":false,"live":null,"outcome":"done","pending":[],"version":1}',
        b'{"lost":false,"live":null,"outcome":null,"pending":[["a",-1]],"version":1}',
        b'{"lost":false,"live":null,"outcome":null,"pending":[["a b",0]],"version":1}',
        b'{"lost":false,"live":["a",0],"outcome":null,"pending":[["a",0]],"version":1}',
        b'{"lost":false,"live":["a",0],"outcome":["a",0],"pending":[],"version":1}',
        b'{"lost":false,"live":["a"],"outcome":null,"pending":[],"version":1}',
        b'{"lost":"yes","live":null,"outcome":null,"pending":[],"version":1}',
        b'{"live":null,"outcome":null,"pending":[],"version":1}',
    ],
)
def test_a_malformed_delete_record_is_refused(raw: bytes) -> None:
    assert parse_voice_deletes(raw) is None


@pytest.mark.asyncio
async def test_the_live_conversation_after_a_delete_is_restored(tmp_path: Path) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "old", "new")
    await _forget(writer, store)
    store.record_user_transcript(Transcript(text="said after the delete", final=True))
    await writer.close()

    reopened, reopened_store = await _open(path)

    # Failing closed never costs the conversation the record names as live.
    assert [message.text for message in reopened_store.snapshot().messages] == [
        "said after the delete"
    ]
    assert reopened.binding == ("new", 1)
    await reopened.close()


@pytest.mark.asyncio
async def test_no_delete_record_is_written_until_a_delete_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "conv")
    store.record_user_transcript(Transcript(text="kept history", final=True))
    await _until(lambda: b"kept history" in (read_live_record(path) or b""))
    await writer.close()

    assert not _deletes_path(path).exists()
    reopened, reopened_store = await _open(path)
    assert [message.text for message in reopened_store.snapshot().messages] == ["kept history"]
    await reopened.close()


def test_a_delete_record_round_trips() -> None:
    deletes = VoiceDeletes(pending=(("a", 0), ("b", 3)), outcome=("c", 1))
    assert parse_voice_deletes(voice_deletes_bytes(deletes)) == deletes
    unknown = VoiceDeletes(outcome=("c", 1), lost=True)
    assert parse_voice_deletes(voice_deletes_bytes(unknown)) == unknown


class _TailWriteFails(VoiceTailWriter):
    """Fails only the tail write, as a sharing violation or a full disk would."""

    fail = False

    async def _write(self, data: bytes) -> None:
        if self.fail:
            raise PermissionError("the tail write failed")
        await super()._write(data)


async def _until(predicate) -> None:  # type: ignore[no-untyped-def]
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


def _recorded(path: Path) -> VoiceDeletes:
    raw = read_live_record(_deletes_path(path))
    if raw is None:
        return VoiceDeletes()
    recorded = parse_voice_deletes(raw)
    assert recorded is not None
    return recorded


async def _crash(writer: VoiceTailWriter, *tasks: asyncio.Task[object]) -> None:
    """Stop the writer as a killed process would: no final write, the lock released."""
    assert writer._task is not None and writer._owner is not None
    for task in (writer._task, *tasks):
        task.cancel()
    await asyncio.gather(writer._task, *tasks, return_exceptions=True)
    unlock_run_record(writer._owner)


async def _delete_with_the_tail_unwritable(
    path: Path,
) -> tuple[_TailWriteFails, asyncio.Task[tuple[str, int]]]:
    writer = _TailWriteFails(path, conversation_ids=_ids("old", "new"))
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    store.record_user_transcript(Transcript(text="a deleted phrase", final=True))
    await _offer(writer, store)
    await _until(lambda: b"a deleted phrase" in (read_live_record(path) or b""))
    writer.fail = True
    request = asyncio.create_task(writer.request_forget(store))
    # The record lands; the cleared tail does not.
    await _until(lambda: _recorded(path).pending == (("old", 0),))
    return writer, request


@pytest.mark.asyncio
async def test_a_delete_reaches_the_companion_only_after_its_cleared_tail_is_durable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, request = await _delete_with_the_tail_unwritable(path)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(writer.next_deletes(), 0.3)
    assert b"a deleted phrase" in (read_live_record(path) or b"") and not request.done()

    writer.fail = False
    assert await asyncio.wait_for(writer.next_deletes(), 5) == (("old", 0),)
    assert await asyncio.wait_for(request, 5) == ("old", 0)
    assert b"a deleted phrase" not in (read_live_record(path) or b"")
    await writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settled", [False, True])
async def test_a_tail_the_record_knows_as_deleted_is_never_restored(
    settled: bool, tmp_path: Path
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, request = await _delete_with_the_tail_unwritable(path)
    if settled:
        # However the acknowledgment arrived, the record now names it complete.
        assert writer.acknowledge_forget(("old", 0)) is True
        await _until(
            lambda: parse_voice_deletes(read_live_record(_deletes_path(path)) or b"")
            == VoiceDeletes(outcome=("old", 0), live=("new", 1))
        )
    assert b"a deleted phrase" in path.read_bytes()
    await _crash(writer, request)

    reopened, reopened_store = await _open(path, "fresh")

    assert reopened_store.snapshot().messages == ()
    assert reopened.binding == ("fresh", 1)
    if settled:
        assert reopened.pending_deletes == () and reopened.deleted_previous is True
    else:
        assert reopened.pending_deletes == (("old", 0),)
    await reopened.close()
    assert b"a deleted phrase" not in path.read_bytes()


@pytest.mark.asyncio
async def test_a_tail_retired_at_open_is_durable_before_its_delete_is_handed_out(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "old")
    store.record_user_transcript(Transcript(text="a deleted phrase", final=True))
    await writer.close()
    # The record landed and the process died before the cleared tail did.
    _deletes_path(path).write_bytes(voice_deletes_bytes(VoiceDeletes(pending=(("old", 0),))))
    reopened = _TailWriteFails(path, conversation_ids=_ids("new"))
    reopened.fail = True
    await reopened.open(ConversationContextStore(on_change=reopened.update))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(reopened.next_deletes(), 0.3)

    reopened.fail = False
    assert await asyncio.wait_for(reopened.next_deletes(), 5) == (("old", 0),)
    assert b"a deleted phrase" not in path.read_bytes()
    await reopened.close()


@pytest.mark.asyncio
async def test_a_version_4_intent_already_completed_is_not_pending_again(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    _deletes_path(path).write_bytes(voice_deletes_bytes(VoiceDeletes(outcome=("old", 0))))
    writer, store = await _open(path, "new")
    store.record_user_transcript(Transcript(text="after the delete", final=True))
    await writer.close()
    document = json.loads(path.read_bytes())
    document["archive"]["generation"] = 1
    document |= {"version": 4, "pending_forget": ["old", 0], "forget_complete": False}
    path.write_bytes(json.dumps(document).encode())
    # This build named the live binding before the older build wrote its tail.
    _deletes_path(path).write_bytes(
        voice_deletes_bytes(VoiceDeletes(outcome=("old", 0), live=("new", 1)))
    )

    reopened, _store = await _open(path)
    assert reopened.pending_deletes == () and reopened.deleted_previous is True
    await asyncio.wait_for(reopened.close(), 5)


@pytest.mark.asyncio
async def test_an_older_delete_settling_late_keeps_the_newer_outcome(tmp_path: Path) -> None:
    writer, store = await _open(tmp_path / "voice-tail.json", "a", "b", "c")
    first = await _forget(writer, store)
    second = await _forget(writer, store)

    assert writer.acknowledge_forget(second) is True
    assert writer.acknowledge_forget(first) is True

    assert writer.delete_outcome == second and writer.deleted_previous is True
    await writer.close()


@pytest.mark.asyncio
async def test_a_version_4_intent_moves_in_even_when_the_record_is_full(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    full = tuple((f"c{number}", number) for number in range(MAX_PENDING_DELETES))
    _deletes_path(path).write_bytes(voice_deletes_bytes(VoiceDeletes(pending=full)))
    writer, store = await _open(path, "new")
    store.record_user_transcript(Transcript(text="after the delete", final=True))
    await writer.close()
    document = json.loads(path.read_bytes())
    document["archive"]["generation"] = 1
    document |= {"version": 4, "pending_forget": ["old", 0], "forget_complete": False}
    path.write_bytes(json.dumps(document).encode())
    _deletes_path(path).write_bytes(
        voice_deletes_bytes(VoiceDeletes(pending=full, live=("new", 1)))
    )

    reopened, _store = await _open(path)
    await asyncio.wait_for(reopened.close(), 5)

    recorded = parse_voice_deletes(_deletes_path(path).read_bytes())
    assert recorded == VoiceDeletes(pending=(*full, ("old", 0)), live=("new", 1))
    assert json.loads(path.read_bytes())["version"] == 3


def _version_1_tail(path: Path, text: str) -> None:
    """A tail from a version-1 build: rows and no conversation identity."""
    path.write_bytes(json.dumps({
        "version": 1,
        "prior_work": False,
        "messages": [{"role": "user", "text": text, "interrupted": False}],
    }).encode())


@pytest.mark.asyncio
async def test_a_version_1_tail_is_never_restored_once_a_delete_is_recorded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    _version_1_tail(path, "a deleted phrase")
    writer = _TailWriteFails(path, conversation_ids=_ids("old", "new"))
    writer.fail = True
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    assert [message.text for message in store.snapshot().messages] == ["a deleted phrase"]
    request = asyncio.create_task(writer.request_forget(store))
    await _until(lambda: _recorded(path).holds_deletes)
    await _crash(writer, request)
    # The version-1 tail is still on disk; it carries no identity the record could clear.
    assert b"a deleted phrase" in path.read_bytes()

    reopened, reopened_store = await _open(path, "fresh")

    assert reopened_store.snapshot().messages == ()
    assert reopened.binding == ("fresh", 0)
    await reopened.close()
    assert b"a deleted phrase" not in path.read_bytes()


@pytest.mark.asyncio
async def test_a_version_1_tail_is_still_restored_when_no_delete_was_ever_recorded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    _version_1_tail(path, "kept history")

    writer, store = await _open(path, "conv")

    assert [message.text for message in store.snapshot().messages] == ["kept history"]
    await writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("found", ["malformed", "refused", "missing"])
async def test_a_pending_delete_waits_for_a_fresh_tail_whatever_the_open_found(
    found: str, tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "old")
    store.record_user_transcript(Transcript(text="a deleted phrase that is long", final=True))
    await writer.close()
    _deletes_path(path).write_bytes(
        voice_deletes_bytes(VoiceDeletes(pending=(("gone", 3),), live=("old", 0)))
    )
    if found == "malformed":
        path.write_bytes(b"{not json")
    elif found == "missing":
        path.unlink()
    reopened = _TailWriteFails(path, conversation_ids=_ids("new"))
    reopened.fail = True
    options = {"max_item_chars": 20} if found == "refused" else {}
    await reopened.open(ConversationContextStore(on_change=reopened.update, **options))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(reopened.next_deletes(), 0.3)

    reopened.fail = False
    assert await asyncio.wait_for(reopened.next_deletes(), 5) == (("gone", 3),)
    await reopened.close()


@pytest.mark.asyncio
async def test_a_stale_tail_of_any_earlier_deleted_conversation_is_never_restored(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "a", "b", "c")
    store.record_user_transcript(Transcript(text="said in a", final=True))
    await _offer(writer, store)
    stale = path.read_bytes()
    first = await writer.request_forget(store)
    second = await _forget(writer, store)
    assert writer.acknowledge_forget(second) and writer.acknowledge_forget(first)
    await writer.close()
    # The outcome now names only "b"; a backup of "a"'s tail comes back beside it.
    assert _recorded(path).outcome == ("b", 1)
    path.write_bytes(stale)

    reopened, reopened_store = await _open(path, "fresh")

    assert reopened_store.snapshot().messages == ()
    assert reopened.binding == ("fresh", 1)
    await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unreadable",
    [
        b"\x00garbage",
        b'{"live":null,"outcome":null,"pending":[["old",0]],"version":2}',
        b'{"extra":1,"live":null,"outcome":null,"pending":[["old",0]],"version":1}',
    ],
)
async def test_an_unreadable_record_restores_no_tail_and_keeps_the_loss_known(
    unreadable: bytes, tmp_path: Path
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, request = await _delete_with_the_tail_unwritable(path)
    await _crash(writer, request)
    _deletes_path(path).write_bytes(unreadable)

    reopened, reopened_store = await _open(path, "fresh")

    assert reopened_store.snapshot().messages == ()
    assert reopened.deletes_lost is True
    await reopened.close()
    assert b"a deleted phrase" not in path.read_bytes()
    # Rewritten as a known loss, never as an empty record.
    assert _recorded(path) == VoiceDeletes(lost=True, live=("fresh", 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", ["frozen_unacknowledged", "acknowledged_then_drained"])
async def test_a_delete_after_any_batch_left_the_host_is_never_completed_locally(
    sent: str, tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "a", "b")
    store.record_user_transcript(Transcript(text="archived row", final=True))
    batch = await asyncio.wait_for(writer.next_batch(), 5)
    if sent == "acknowledged_then_drained":
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        # Nothing is frozen any more; only the acknowledged cursor remembers the send.
        assert writer._frozen == 0 and writer._cursor is not None

    old = await writer.request_forget(store)

    assert writer.pending_deletes == (old,) and writer.delete_outcome is None
    await writer.close()


@pytest.mark.asyncio
async def test_a_delete_of_a_conversation_no_batch_ever_left_completes_locally(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "a", "b")
    # Rows were queued, but none was frozen into a batch, so none was ever sent.
    store.record_user_transcript(Transcript(text="never archived", final=True))

    old = await writer.request_forget(store)

    assert writer.pending_deletes == () and writer.delete_outcome == old
    assert writer.deleted_previous is True
    await writer.close()
    assert _recorded(path) == VoiceDeletes(outcome=("a", 0), live=("b", 1))
