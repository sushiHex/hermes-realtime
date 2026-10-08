"""A voice delete survives anything that resets the tail, and settles one binding at a time."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.integration.voice_deletes import (
    MAX_PENDING_DELETES,
    VoiceDeletes,
    parse_voice_deletes,
    voice_deletes_bytes,
)
from hermes_realtime.integration.voice_tail import VoiceTailWriter
from hermes_realtime.speech import Transcript


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
    old = await writer.request_forget(store)
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
    assert writer.delete_outcome == "unknown"
    assert _markers(capsys.readouterr().out, "[voice-deletes] ") == [
        {"refusal": "malformed", "version": 1}
    ]
    await writer.close()
    # The finding is kept durably, so a restart cannot forget it.
    reopened, _store = await _open(path, "conv2")
    assert reopened.delete_outcome == "unknown"
    await reopened.close()


@pytest.mark.asyncio
async def test_a_pending_delete_never_blocks_a_later_one(tmp_path: Path) -> None:
    writer, store = await _open(tmp_path / "voice-tail.json", "a", "b", "c")

    first = await writer.request_forget(store)
    second = await writer.request_forget(store)

    assert writer.pending_deletes == (first, second) == (("a", 0), ("b", 1))
    await writer.close()


@pytest.mark.asyncio
async def test_pending_deletes_are_bounded(tmp_path: Path) -> None:
    names = [f"c{number}" for number in range(MAX_PENDING_DELETES + 2)]
    writer, store = await _open(tmp_path / "voice-tail.json", *names)
    for _ in range(MAX_PENDING_DELETES):
        # A request waits for its durable write; a write that keeps failing fails the test.
        await asyncio.wait_for(writer.request_forget(store), 5)
    binding = writer.binding

    with pytest.raises(RuntimeError, match="capacity"):
        await asyncio.wait_for(writer.request_forget(store), 5)

    assert writer.binding == binding and len(writer.pending_deletes) == MAX_PENDING_DELETES
    await writer.close()


@pytest.mark.asyncio
async def test_a_completed_delete_is_scoped_to_the_conversation_it_retired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    writer, store = await _open(path, "a", "b", "c")
    first = await writer.request_forget(store)
    second = await writer.request_forget(store)

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
        pending=(("old", 0),)
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"outcome":null,"pending":[],"version":2}',
        b'{"outcome":null,"pending":[],"version":true}',
        b'{"outcome":null,"pending":[["a",0],["a",0]],"version":1}',
        b'{"outcome":["a",0],"pending":[["a",0]],"version":1}',
        b'{"outcome":"done","pending":[],"version":1}',
        b'{"outcome":null,"pending":[["a",-1]],"version":1}',
        b'{"outcome":null,"pending":[["a b",0]],"version":1}',
    ],
)
def test_a_malformed_delete_record_is_refused(raw: bytes) -> None:
    assert parse_voice_deletes(raw) is None


def test_a_delete_record_round_trips() -> None:
    deletes = VoiceDeletes(pending=(("a", 0), ("b", 3)), outcome=("c", 1))
    assert parse_voice_deletes(voice_deletes_bytes(deletes)) == deletes
    unknown = VoiceDeletes(outcome="unknown")
    assert parse_voice_deletes(voice_deletes_bytes(unknown)) == unknown
