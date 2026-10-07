from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.integration import voice_tail as voice_tail_module
from hermes_realtime.integration.voice_archive import VoiceArchiveSender
from hermes_realtime.integration.voice_forget import VoiceForgetSender
from hermes_realtime.integration.voice_tail import VoiceTailWriter
from hermes_realtime.protocol import (
    VOICE_FORGET_CAPABILITY,
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceForgetAckEvent,
    VoiceForgetEvent,
)
from hermes_realtime.speech import Transcript


@pytest.mark.asyncio
async def test_delete_intent_is_durable_and_stale_archive_ack_is_fenced(tmp_path: Path) -> None:
    path = tmp_path / "voice-tail.json"
    writer = VoiceTailWriter(path, conversation_ids=iter(("old", "new")).__next__)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    context.record_user_transcript(Transcript(text="secret", final=True))
    batch = await asyncio.wait_for(writer.next_batch(), 2)

    old = await writer.request_forget(context)

    assert old == ("old", 0)
    assert writer.binding == ("new", 1)
    assert writer.pending_deletes == (old,)
    assert context.snapshot().messages == ()
    assert not writer.acknowledge(*(
        batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
    ))
    await writer.close()

    restored = VoiceTailWriter(path)
    restored_context = ConversationContextStore(on_change=restored.update)
    await restored.open(restored_context)
    assert restored.binding == ("new", 1)
    assert restored.pending_deletes == (old,)
    assert restored_context.snapshot().messages == ()
    assert await asyncio.wait_for(restored.next_deletes(), 2) == (old,)
    assert restored.acknowledge_forget(old)
    await restored.close()


@pytest.mark.asyncio
async def test_old_archive_refusal_cannot_fence_new_generation(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=iter(("old", "new")).__next__)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    entered = asyncio.Event()
    release = asyncio.Event()
    sent: list[tuple[str, int]] = []

    class Link:
        capabilities = frozenset({"voice_archive"})

        async def archive(
            self, event: VoiceArchiveEvent
        ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent:
            sent.append((event.conversation_id, event.generation))
            if event.conversation_id == "old":
                entered.set()
                await release.wait()
                return VoiceArchiveRefusedEvent(
                    protocol_version="0.3", type="voice_archive_refused",
                    conversation_id=event.conversation_id, generation=event.generation,
                    seq_from=event.seq_from, seq_through=event.seq_through,
                    category="quarantined",
                )
            return VoiceArchiveAckEvent(
                protocol_version="0.3", type="voice_archive_ack",
                conversation_id=event.conversation_id, generation=event.generation,
                seq_from=event.seq_from, seq_through=event.seq_through,
            )

        async def close(self) -> None:
            pass

    async def connect() -> Link:
        return Link()

    sender = VoiceArchiveSender(writer, connect, initial_backoff_seconds=0.01)
    sender.start()
    context.record_user_transcript(Transcript(text="old", final=True))
    await asyncio.wait_for(entered.wait(), 2)
    await writer.request_forget(context)
    context.record_user_transcript(Transcript(text="new", final=True))
    release.set()
    async with asyncio.timeout(2):
        while ("new", 1) not in sent:
            await asyncio.sleep(0.01)
    assert sender.fence is None
    await sender.close()
    await writer.close()

@pytest.mark.asyncio
async def test_a_stuck_delete_never_holds_up_a_later_one(tmp_path: Path) -> None:
    writer = VoiceTailWriter(
        tmp_path / "tail.json", conversation_ids=iter(("a", "b", "c")).__next__
    )
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    stuck = await writer.request_forget(context)
    later = await writer.request_forget(context)

    class Link:
        capabilities = frozenset({VOICE_FORGET_CAPABILITY})

        async def forget(self, event: VoiceForgetEvent) -> VoiceForgetAckEvent:
            return VoiceForgetAckEvent(
                protocol_version="0.3", type="voice_forget_ack",
                conversation_id=event.conversation_id, generation=event.generation,
                state="pending" if event.conversation_id == "a" else "complete",
            )

        async def close(self) -> None:
            pass

    async def connect() -> Link:
        return Link()

    sender = VoiceForgetSender(
        writer, connect, initial_backoff_seconds=0.01, max_backoff_seconds=0.01
    )
    sender.start()
    async with asyncio.timeout(2):
        while later in writer.pending_deletes:
            await asyncio.sleep(0.01)
    assert writer.pending_deletes == (stuck,)
    assert sender.negotiated
    await sender.close()
    await writer.close()


@pytest.mark.asyncio
async def test_the_sender_reports_negotiation_only_from_a_live_capable_link(
    tmp_path: Path,
) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "a")
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    capabilities: frozenset[str] = frozenset({"voice_archive"})
    connected = asyncio.Event()

    class Link:
        def __init__(self) -> None:
            self.capabilities = capabilities

        async def forget(self, event: VoiceForgetEvent) -> VoiceForgetAckEvent:
            raise AssertionError("no delete is pending")

        async def close(self) -> None:
            pass

    async def connect() -> Link:
        connected.set()
        return Link()

    sender = VoiceForgetSender(
        writer, connect, initial_backoff_seconds=0.01, max_backoff_seconds=0.01
    )
    assert not sender.negotiated
    sender.start()
    await asyncio.wait_for(connected.wait(), 2)
    await asyncio.sleep(0.05)
    assert not sender.negotiated
    capabilities = frozenset({VOICE_FORGET_CAPABILITY})
    async with asyncio.timeout(2):
        while not sender.negotiated:
            await asyncio.sleep(0.01)
    await sender.close()
    assert not sender.negotiated
    await writer.close()


@pytest.mark.asyncio
async def test_sender_retains_pending_until_exact_complete(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=iter(("a", "b")).__next__)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    old = await writer.request_forget(context)
    requests: list[VoiceForgetEvent] = []

    class Link:
        capabilities = frozenset({VOICE_FORGET_CAPABILITY})

        async def forget(self, event: VoiceForgetEvent) -> VoiceForgetAckEvent:
            requests.append(event)
            return VoiceForgetAckEvent(
                protocol_version="0.3",
                type="voice_forget_ack",
                conversation_id="other" if len(requests) == 2 else event.conversation_id,
                generation=event.generation,
                state="pending" if len(requests) == 1 else "complete",
            )

        async def close(self) -> None:
            pass

    async def connect() -> Link:
        return Link()

    sender = VoiceForgetSender(
        writer,
        connect,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.01,
    )
    sender.start()
    async with asyncio.timeout(2):
        while writer.pending_deletes:
            await asyncio.sleep(0.01)
    assert len(requests) >= 3
    assert all((r.conversation_id, r.generation) == old for r in requests)
    await sender.close()
    await writer.close()


@pytest.mark.asyncio
async def test_sender_waits_for_durable_delete_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=iter(("a", "b")).__next__)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    entered = threading.Event()
    release = threading.Event()
    original_write = voice_tail_module.write_run_record

    def delayed_write(path: Path, data: bytes) -> None:
        # Only the delete record is held back; the sender must wait for exactly it.
        if path.name == "tail.deletes.json":
            entered.set()
            release.wait(5)
        original_write(path, data)

    monkeypatch.setattr(voice_tail_module, "write_run_record", delayed_write)
    calls: list[VoiceForgetEvent] = []

    class Link:
        capabilities = frozenset({VOICE_FORGET_CAPABILITY})

        async def forget(self, event: VoiceForgetEvent) -> VoiceForgetAckEvent:
            calls.append(event)
            return VoiceForgetAckEvent(
                protocol_version="0.3", type="voice_forget_ack",
                conversation_id=event.conversation_id, generation=event.generation,
                state="complete",
            )

        async def close(self) -> None:
            pass

    async def connect() -> Link:
        return Link()

    sender = VoiceForgetSender(writer, connect, initial_backoff_seconds=0.01)
    sender.start()
    deletion = asyncio.create_task(writer.request_forget(context))
    assert await asyncio.to_thread(entered.wait, 2)
    await asyncio.sleep(0.05)
    assert calls == []
    assert not deletion.done()
    release.set()
    await asyncio.wait_for(deletion, 2)
    async with asyncio.timeout(2):
        while writer.pending_deletes:
            await asyncio.sleep(0.01)
    assert len(calls) == 1
    await sender.close()
    await writer.close()


@pytest.mark.asyncio
async def test_old_review_ack_cannot_advance_new_binding(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=iter(("old", "new")).__next__)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    writer.update(DurableConversation(
        messages=(ConversationMessage("user", "old question"),
                  ConversationMessage("assistant", "old answer")),
        prior_work=False,
    ))
    batch = await asyncio.wait_for(writer.next_batch(), 2)
    assert writer.acknowledge(
        batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through,
    )
    old_review = await asyncio.wait_for(writer.next_review(1), 2)
    await writer.request_forget(context)
    assert not writer.acknowledge_review(old_review)
    assert writer.binding == ("new", 1)
    await writer.close()


@pytest.mark.asyncio
async def test_cancelled_delete_request_still_publishes_durable_clear(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=iter(("old", "new")).__next__)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_write = writer._write
    cleared: list[tuple[str, int]] = []

    async def delayed_write(data: bytes) -> None:
        entered.set()
        await release.wait()
        await original_write(data)

    writer._write = delayed_write  # type: ignore[method-assign]
    deletion = asyncio.create_task(writer.request_forget(
        context, on_durable_clear=lambda: cleared.append(writer.binding),
    ))
    await asyncio.wait_for(entered.wait(), 2)
    deletion.cancel()
    with pytest.raises(asyncio.CancelledError):
        await deletion
    assert cleared == []
    release.set()
    async with asyncio.timeout(2):
        while not cleared:
            await asyncio.sleep(0.01)
    assert cleared == [("new", 1)]
    assert writer.pending_deletes == (("old", 0),)
    await writer.close()


@pytest.mark.asyncio
async def test_failed_tail_write_does_not_publish_clear(tmp_path: Path) -> None:
    writer = VoiceTailWriter(
        tmp_path / "tail.json", conversation_ids=iter(("old", "new")).__next__,
        initial_backoff_seconds=0.01,
    )
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    original_write = writer._write
    failed = asyncio.Event()
    retry_entered = asyncio.Event()
    release = asyncio.Event()
    attempts = 0
    clears: list[str] = []

    async def transient_failure(data: bytes) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            failed.set()
            raise OSError("synthetic write failure")
        retry_entered.set()
        await release.wait()
        await original_write(data)

    writer._write = transient_failure  # type: ignore[method-assign]
    deletion = asyncio.create_task(writer.request_forget(
        context, on_durable_clear=lambda: clears.append("cleared"),
    ))
    await asyncio.wait_for(failed.wait(), 2)
    await asyncio.wait_for(retry_entered.wait(), 2)
    assert clears == []
    assert not deletion.done()
    release.set()
    await asyncio.wait_for(deletion, 2)
    assert clears == ["cleared"]
    await writer.close()
