"""Acknowledged archive coverage schedules native review off the voice path."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.integration.voice_archive import VoiceArchiveSender
from hermes_realtime.integration.voice_review import VoiceReviewSender, review_idle_allowed
from hermes_realtime.integration.voice_tail import (
    ArchiveOutbox,
    ReviewProgress,
    VoiceTailWriter,
    max_voice_tail_bytes,
    parse_voice_tail,
    voice_tail_bytes,
)
from hermes_realtime.protocol import (
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceReviewAckEvent,
    VoiceReviewEvent,
    VoiceReviewRefusedEvent,
)
from hermes_realtime.speech import Transcript


class _Link:
    capabilities = frozenset({"voice_review"})
    review_interval = 2

    def __init__(self) -> None:
        self.requests: list[VoiceReviewEvent] = []
        self.refuse_once = False
        self.permanent_refusal: str | None = None
        self.wrong_range_once = False

    async def review(
        self, event: VoiceReviewEvent
    ) -> VoiceReviewAckEvent | VoiceReviewRefusedEvent:
        await asyncio.sleep(0)
        self.requests.append(event)
        fields = dict(
            protocol_version="0.3",
            conversation_id=event.conversation_id,
            generation=event.generation,
            seq_from=event.seq_from,
            seq_through=event.seq_through,
            closing=event.closing,
        )
        if self.refuse_once:
            self.refuse_once = False
            return VoiceReviewRefusedEvent(type="voice_review_refused", category="busy", **fields)
        if self.permanent_refusal is not None:
            return VoiceReviewRefusedEvent(
                type="voice_review_refused", category=self.permanent_refusal, **fields
            )
        if self.wrong_range_once:
            self.wrong_range_once = False
            fields["seq_through"] = event.seq_through + 1
        return VoiceReviewAckEvent(
            type="voice_review_ack", review_id="review_1", status="accepted", **fields
        )

    async def close(self) -> None:
        pass


async def _until(predicate: object) -> None:
    assert callable(predicate)
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_cadence_and_exact_threshold_close_are_distinct(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    link = _Link()

    async def connect() -> _Link:
        return link

    sender = VoiceReviewSender(
        writer,
        connect,
        idle_allowed=lambda: False,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.002,
    )
    sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        await asyncio.sleep(0.02)
        assert link.requests == []

        store.record_user_transcript(Transcript(text="two", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        await _until(lambda: len(link.requests) == 1)
        assert (
            link.requests[0].seq_from,
            link.requests[0].seq_through,
            link.requests[0].closing,
        ) == (0, 1, False)

        writer.request_review_close()
        await _until(lambda: len(link.requests) == 2)
        assert (
            link.requests[1].seq_from,
            link.requests[1].seq_through,
            link.requests[1].closing,
        ) == (0, 1, True)
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_normal_close_settles_one_final_archive_and_review_without_restart(
    tmp_path: Path,
) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    review_link = _Link()

    class ArchiveLink:
        capabilities = frozenset({"voice_archive"})

        async def archive(self, event: VoiceArchiveEvent) -> VoiceArchiveAckEvent:
            return VoiceArchiveAckEvent(
                protocol_version="0.3",
                type="voice_archive_ack",
                conversation_id=event.conversation_id,
                generation=event.generation,
                seq_from=event.seq_from,
                seq_through=event.seq_through,
            )

        async def close(self) -> None:
            pass

    async def connect_archive() -> ArchiveLink:
        return ArchiveLink()

    async def connect_review() -> _Link:
        return review_link

    archive_sender = VoiceArchiveSender(writer, connect_archive)
    review_sender = VoiceReviewSender(writer, connect_review, idle_allowed=lambda: False)
    archive_sender.start()
    review_sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        assert await writer.wait_review_close(timeout=2)
        assert len(review_link.requests) == 1
        assert review_link.requests[0].closing
        assert writer._review.close_reviewed
    finally:
        await archive_sender.close()
        await review_sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_close_deadline_keeps_target_and_emits_bounded_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tail.json"
    writer = VoiceTailWriter(path, conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        store.record_user_transcript(Transcript(text="private user text", final=True))
        assert not await writer.wait_review_close(timeout=0.01)
        assert writer._review.close_targets == (0,)
    finally:
        await writer.close()
    marker = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[voice-review-close] ")
    ]
    assert len(marker) == 1
    assert '"refusal":"deadline"' in marker[0]
    assert "private user text" not in marker[0]

    restored = VoiceTailWriter(path, conversation_ids=lambda: "unused")
    restored_store = ConversationContextStore(on_change=restored.update)
    await restored.open(restored_store)
    try:
        assert restored._review.close_targets == (0,)
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_busy_close_retains_exact_range_until_accepted(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    link = _Link()
    link.refuse_once = True

    async def connect() -> _Link:
        return link

    sender = VoiceReviewSender(
        writer,
        connect,
        idle_allowed=lambda: False,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.002,
    )
    sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        writer.request_review_close()
        await _until(lambda: len(link.requests) >= 2)
        assert link.requests[0] == link.requests[1]
        await _until(lambda: writer._review.close_reviewed)
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.parametrize("category", ["lease_lost", "lease_held"])
@pytest.mark.asyncio
async def test_review_retries_the_frozen_range_after_a_recoverable_lease_refusal(
    tmp_path: Path, category: str
) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    link = _Link()
    connects = 0

    async def connect() -> _Link:
        nonlocal connects
        connects += 1
        link.permanent_refusal = category if connects == 1 else None
        return link

    sender = VoiceReviewSender(
        writer,
        connect,
        idle_allowed=lambda: False,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.002,
    )
    sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        writer.request_review_close()
        await _until(lambda: writer._review.close_reviewed)
        assert connects >= 2
        assert len(link.requests) == 2
        assert link.requests[0] == link.requests[1]
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.parametrize("invalid_field", ["capability", "interval"])
@pytest.mark.asyncio
async def test_invalid_review_handshake_closes_with_a_bound_before_reconnect(
    tmp_path: Path, invalid_field: str
) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    release = asyncio.Event()
    close_started = asyncio.Event()

    class HungInvalidLink:
        capabilities = frozenset() if invalid_field == "capability" else frozenset({"voice_review"})
        review_interval = None if invalid_field == "interval" else 2

        async def close(self) -> None:
            close_started.set()
            await release.wait()

    valid = _Link()
    connects = 0

    async def connect() -> object:
        nonlocal connects
        connects += 1
        return HungInvalidLink() if connects == 1 else valid

    sender = VoiceReviewSender(
        writer,
        connect,  # type: ignore[arg-type]
        idle_allowed=lambda: False,
        close_timeout=0.01,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.002,
    )
    sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        writer.request_review_close()
        await _until(lambda: writer._review.close_reviewed)
        assert close_started.is_set()
        assert connects >= 2
        assert len(valid.requests) == 1
    finally:
        release.set()
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_wrong_review_identity_retries_the_frozen_range(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    link = _Link()
    link.wrong_range_once = True

    async def connect() -> _Link:
        return link

    sender = VoiceReviewSender(
        writer,
        connect,
        idle_allowed=lambda: False,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.002,
    )
    sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        writer.request_review_close()
        await _until(lambda: len(link.requests) >= 2)
        assert link.requests[0] == link.requests[1]
        await _until(lambda: writer._review.close_reviewed)
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_one_batched_ack_splits_at_ten_user_cadence(tmp_path: Path) -> None:
    writer = VoiceTailWriter(
        tmp_path / "tail.json",
        conversation_ids=lambda: "conv",
        max_outbox_rows=64,
        max_batch_rows=48,
    )
    store = ConversationContextStore(max_messages=64, on_change=writer.update)
    await writer.open(store)
    try:
        writer.update(
            DurableConversation(
                messages=tuple(
                    ConversationMessage("user" if index % 2 == 0 else "assistant", f"row {index}")
                    for index in range(40)
                ),
                prior_work=False,
            )
        )
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert len(batch.rows) == 40
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        first = await asyncio.wait_for(writer.next_review(10), 2)
        assert (first.seq_from, first.seq_through, first.users) == (0, 19, 10)
        assert writer.acknowledge_review(first)
        second = await asyncio.wait_for(writer.next_review(10), 2)
        assert (second.seq_from, second.seq_through, second.users) == (20, 39, 20)
        assert writer.acknowledge_review(second)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_window_bound_splits_long_assistant_run_before_close(tmp_path: Path) -> None:
    writer = VoiceTailWriter(
        tmp_path / "tail.json",
        conversation_ids=lambda: "conv",
        max_outbox_rows=64,
        max_batch_rows=32,
    )
    store = ConversationContextStore(max_messages=64, on_change=writer.update)
    await writer.open(store)
    try:
        writer.update(
            DurableConversation(
                messages=(ConversationMessage("user", "question"),)
                + tuple(ConversationMessage("assistant", f"piece {index}") for index in range(30)),
                prior_work=False,
            )
        )
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        first = await asyncio.wait_for(writer.next_review(1), 2)
        assert (first.seq_from, first.seq_through, first.closing) == (0, 23, False)
        assert writer.acknowledge_review(first)
        writer.request_review_close()
        final = await asyncio.wait_for(writer.next_review(1), 2)
        assert (final.seq_from, final.seq_through, final.closing) == (24, 30, True)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_idle_close_is_retained_while_companion_is_offline(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)

    async def offline() -> _Link:
        raise ConnectionError("offline")

    sender = VoiceReviewSender(
        writer,
        offline,
        idle_allowed=lambda: True,
        idle_seconds=0.02,
        initial_backoff_seconds=0.005,
        max_backoff_seconds=0.01,
    )
    sender.start()
    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        await _until(lambda: writer._review.close_targets == (0,))
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.parametrize("field", ["idle_seconds", "reply_timeout", "initial_backoff_seconds"])
def test_review_sender_rejects_boolean_timing(field: str, tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json")

    async def connect() -> _Link:
        return _Link()

    with pytest.raises(ValueError):
        VoiceReviewSender(writer, connect, idle_allowed=lambda: True, **{field: True})


def test_named_work_does_not_hold_open_a_quiet_voice_turn() -> None:
    store = ConversationContextStore()
    store.record_task_accepted(task_id="task_one", run_id="deleg_one", objective="Background work")
    assert store.snapshot().active_tasks
    assert review_idle_allowed(speech_active=False, foreground_tasks=0)
    assert not review_idle_allowed(speech_active=True, foreground_tasks=0)
    assert not review_idle_allowed(speech_active=False, foreground_tasks=1)


@pytest.mark.asyncio
async def test_later_idle_end_does_not_erase_busy_prior_end(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        for text in ("one", "two"):
            store.record_user_transcript(Transcript(text=text, final=True))
            batch = await asyncio.wait_for(writer.next_batch(), 2)
            assert writer.acknowledge(
                batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
            )
            writer.request_review_close()
        assert writer._review.close_targets == (0, 1)
        first = await asyncio.wait_for(writer.next_review(10), 2)
        assert (first.seq_from, first.seq_through, first.closing) == (0, 0, True)
        assert writer.acknowledge_review(first)
        second = await asyncio.wait_for(writer.next_review(10), 2)
        assert (second.seq_from, second.seq_through, second.closing) == (1, 1, True)
        assert writer.acknowledge_review(second)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_v2_migration_preserves_outbox_and_baselines_review(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer = VoiceTailWriter(path, conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    store.record_user_transcript(Transcript(text="archived", final=True))
    batch = await asyncio.wait_for(writer.next_batch(), 2)
    assert writer.acknowledge(
        batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
    )
    store.record_user_transcript(Transcript(text="pending", final=True))
    await writer.close()

    old = json.loads(path.read_text())
    old["version"] = 2
    del old["archive"]["review"]
    path.write_text(json.dumps(old, separators=(",", ":"), sort_keys=True))

    restored = VoiceTailWriter(path, conversation_ids=lambda: "unused")
    restored_store = ConversationContextStore(on_change=restored.update)
    await restored.open(restored_store)
    try:
        assert restored.conversation_id == "conv"
        assert restored._review.cursor == 0
        assert restored._review.rows == ()
        replay = await asyncio.wait_for(restored.next_batch(), 2)
        assert replay.rows[0].text == "pending"
        assert restored.acknowledge(
            replay.conversation_id, replay.generation, replay.seq_from, replay.seq_through
        )
        review = await asyncio.wait_for(restored.next_review(1), 2)
        assert (review.seq_from, review.seq_through) == (1, 1)
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_restart_replays_same_frozen_closing_range(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer = VoiceTailWriter(path, conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    store.record_user_transcript(Transcript(text="one", final=True))
    batch = await asyncio.wait_for(writer.next_batch(), 2)
    assert writer.acknowledge(
        batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
    )
    writer.request_review_close()
    first = await asyncio.wait_for(writer.next_review(10), 2)
    await writer.close()

    restored = VoiceTailWriter(path, conversation_ids=lambda: "unused")
    restored_store = ConversationContextStore(on_change=restored.update)
    await restored.open(restored_store)
    try:
        assert await asyncio.wait_for(restored.next_review(10), 2) == first
        assert restored.acknowledge_review(first)
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_review_request_is_written_before_sender_can_take_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    release = asyncio.Event()
    original = writer._write

    async def held(data: bytes) -> None:
        await release.wait()
        await original(data)

    try:
        store.record_user_transcript(Transcript(text="one", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        await _until(lambda: writer._written_version >= writer._version)
        monkeypatch.setattr(writer, "_write", held)
        writer.request_review_close()
        pending = asyncio.create_task(writer.next_review(10))
        await _until(lambda: writer._review.pending is not None)
        assert not pending.done()
        resend = asyncio.create_task(writer.next_review(10))
        await asyncio.sleep(0)
        assert not resend.done()
        release.set()
        assert await asyncio.wait_for(pending, 2) == await asyncio.wait_for(resend, 2)
    finally:
        release.set()
        await writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("users", [10, 20])
async def test_close_before_periodic_dispatch_keeps_each_threshold_review(
    tmp_path: Path, users: int
) -> None:
    writer = VoiceTailWriter(
        tmp_path / "tail.json",
        conversation_ids=lambda: "conv",
        max_outbox_rows=64,
        max_batch_rows=32,
    )
    store = ConversationContextStore(max_messages=32, on_change=writer.update)
    await writer.open(store)
    try:
        writer.update(
            DurableConversation(
                messages=tuple(
                    ConversationMessage("user", f"row {index}") for index in range(users)
                ),
                prior_work=False,
            )
        )
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        writer.request_review_close()
        requests = []
        for _ in range(users // 10 + 1):
            request = await asyncio.wait_for(writer.next_review(10), 2)
            requests.append(request)
            assert writer.acknowledge_review(request)
        assert [request.closing for request in requests] == [False] * (users // 10) + [True]
        assert [request.users for request in requests[:-1]] == list(range(10, users + 1, 10))
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_permanent_refusal_still_records_a_later_idle_end(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    link = _Link()
    link.permanent_refusal = "disabled"

    async def connect() -> _Link:
        return link

    sender = VoiceReviewSender(
        writer,
        connect,
        idle_allowed=lambda: True,
        idle_seconds=0.02,
        initial_backoff_seconds=0.005,
        max_backoff_seconds=0.01,
    )
    sender.start()
    try:
        for text in ("one", "two"):
            store.record_user_transcript(Transcript(text=text, final=True))
            batch = await asyncio.wait_for(writer.next_batch(), 2)
            assert writer.acknowledge(
                batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
            )
        await _until(lambda: len(link.requests) == 1)
        store.record_user_transcript(Transcript(text="later", final=True))
        batch = await asyncio.wait_for(writer.next_batch(), 2)
        assert writer.acknowledge(
            batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
        )
        await _until(
            lambda: bool(writer._review.close_targets) and writer._review.close_targets[-1] == 2
        )
        assert len(link.requests) == 1
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_hung_link_close_has_a_bound(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json")

    async def connect() -> _Link:
        return _Link()

    sender = VoiceReviewSender(
        writer,
        connect,
        idle_allowed=lambda: False,
        close_timeout=0.01,
    )

    class HungLink(_Link):
        async def close(self) -> None:
            await asyncio.Event().wait()

    sender._link = HungLink()
    await asyncio.wait_for(sender._drop(), 0.2)


@pytest.mark.asyncio
async def test_cancelled_close_keeps_ownership_of_a_live_sender(tmp_path: Path) -> None:
    writer = VoiceTailWriter(tmp_path / "tail.json")

    async def connect() -> _Link:
        return _Link()

    sender = VoiceReviewSender(writer, connect, idle_allowed=lambda: False)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def stubborn() -> None:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(stubborn())
    await entered.wait()
    sender._task = task
    closing = asyncio.create_task(sender.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert sender._task is task and not task.done()
    release.set()
    await task
    await sender.close()
    assert sender._task is None


@pytest.mark.asyncio
async def test_pending_user_count_and_close_queue_are_validated_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    writer = VoiceTailWriter(path, conversation_ids=lambda: "conv")
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    store.record_user_transcript(Transcript(text="one", final=True))
    batch = await asyncio.wait_for(writer.next_batch(), 2)
    assert writer.acknowledge(
        batch.conversation_id, batch.generation, batch.seq_from, batch.seq_through
    )
    writer.request_review_close()
    await asyncio.wait_for(writer.next_review(10), 2)
    await writer.close()
    document = json.loads(path.read_text())
    document["archive"]["review"]["pending"]["users"] = 0
    malformed = json.dumps(document).encode()
    assert parse_voice_tail(malformed, max_messages=16, max_item_chars=1024) is None
    document["archive"]["review"]["pending"]["users"] = 1
    document["archive"]["review"]["close_reviewed"] = True
    malformed = json.dumps(document).encode()
    assert parse_voice_tail(malformed, max_messages=16, max_item_chars=1024) is None


def test_tail_size_bound_includes_maximum_review_rows_and_close_checkpoints() -> None:
    maximum = 2**53 - 2
    progress = ReviewProgress(
        users=4096,
        rows=tuple((maximum - 4095 + index, True) for index in range(4096)),
        close_targets=tuple(maximum - 4095 + index for index in range(4096)),
    )
    archive = ArchiveOutbox(
        conversation_id="conv",
        generation=0,
        next_seq=maximum + 1,
        settled=0,
        cursor=maximum,
        rows=(),
        frozen=0,
        gap=None,
        review=progress,
    )
    raw = voice_tail_bytes(DurableConversation(messages=(), prior_work=False), archive)
    assert len(raw) <= max_voice_tail_bytes(0, 0, 0)
    assert parse_voice_tail(raw, max_messages=0, max_item_chars=0, max_outbox_rows=0) is not None


def test_review_metadata_caps_are_strict() -> None:
    progress = ReviewProgress(
        users=4096,
        rows=tuple((index, True) for index in range(4096)),
    )
    archive = ArchiveOutbox(
        conversation_id="conv",
        generation=0,
        next_seq=4096,
        settled=0,
        cursor=4095,
        rows=(),
        frozen=0,
        gap=None,
        review=progress,
    )
    document = json.loads(voice_tail_bytes(DurableConversation((), False), archive))
    document["archive"]["review"]["rows"].append([4096, True])
    document["archive"]["next_seq"] = 4097
    document["archive"]["cursor"] = 4096
    document["archive"]["review"]["users"] = 4097
    assert (
        parse_voice_tail(json.dumps(document).encode(), max_messages=16, max_item_chars=1024)
        is None
    )
    document["archive"]["review"]["rows"].pop()
    document["archive"]["next_seq"] = 4097
    document["archive"]["cursor"] = 4095
    document["archive"]["gap"] = [4096, 4096]
    document["archive"]["review"]["users"] = 4096
    document["archive"]["review"]["close_targets"] = list(range(4097))
    assert (
        parse_voice_tail(json.dumps(document).encode(), max_messages=16, max_item_chars=1024)
        is None
    )


def test_review_cursor_and_overflow_cannot_claim_untracked_coverage() -> None:
    archive = ArchiveOutbox(
        conversation_id="conv",
        generation=0,
        next_seq=1,
        settled=0,
        cursor=0,
        rows=(),
        frozen=0,
        gap=None,
        review=ReviewProgress(cursor=None, users=0),
    )
    original = json.loads(voice_tail_bytes(DurableConversation((), False), archive))
    forged = json.loads(json.dumps(original))
    forged["archive"]["review"].update(users=1, reviewed_users=1)
    assert (
        parse_voice_tail(json.dumps(forged).encode(), max_messages=16, max_item_chars=1024) is None
    )
    forged = json.loads(json.dumps(original))
    forged["archive"]["review"]["overflow"] = True
    assert (
        parse_voice_tail(json.dumps(forged).encode(), max_messages=16, max_item_chars=1024) is None
    )
