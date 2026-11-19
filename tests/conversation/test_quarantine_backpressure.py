"""A clean ordinary reply drains its bounded quarantine at consumer pace."""
from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from hermes_realtime.conversation import (
    ForegroundOutputBackpressure,
    ForegroundTurnCoordinator,
    ForegroundTurnLease,
)
from tests.conversation.test_cancel_selection_prose import Inference
from tests.conversation.test_confirmed_multi_cancel import _owner, _say, _settle
from tests.conversation.test_streaming import RecordingPlayback


class SlowPlayback(RecordingPlayback):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def play(self, chunk, *, is_valid):
        self.entered.set()
        await self.release.wait()
        await super().play(chunk, is_valid=is_valid)


@pytest.mark.asyncio
async def test_existing_publish_remains_nonblocking_when_the_queue_is_full():
    coordinator = ForegroundTurnCoordinator(output_capacity=1)
    refused, release = asyncio.Event(), asyncio.Event()

    async def runner(lease):
        assert await lease.publish("first")
        try:
            await lease.publish("overflow")
        except ForegroundOutputBackpressure:
            refused.set()
        await release.wait()

    await coordinator.start("owner", runner)
    try:
        await asyncio.wait_for(refused.wait(), 1)
        assert coordinator.pending_publication_count == 1
    finally:
        release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_more_than_default_queue_capacity_drains_in_order_with_a_slow_consumer():
    texts = tuple(f"Ordinary synthetic response segment number {index:02d}." for index in range(40))
    inference = Inference(texts=texts)
    inference.initial.release.set()
    loop, _, controller, proposals, synth, _ = _owner("alpha", inference=inference)
    playback = SlowPlayback()
    loop._playback = playback
    peaks = []
    publish = loop._foreground._publish

    async def observed_publish(*args, **kwargs):
        accepted = await publish(*args, **kwargs)
        peaks.append(loop._foreground.pending_publication_count)
        return accepted

    loop._foreground._publish = observed_publish
    turn = asyncio.create_task(_say(loop, proposals, "An ordinary question.", "long_reply"))
    entered = asyncio.create_task(playback.entered.wait())
    try:
        done, _ = await asyncio.wait({turn, entered}, timeout=1,
                                    return_when=asyncio.FIRST_COMPLETED)
        assert done
        if turn in done:
            await turn  # The broken implementation raises queue backpressure here.
        await asyncio.wait_for(entered, 1)
        assert not turn.done()
        assert loop._foreground.pending_publication_count <= 32
        playback.release.set()
        await asyncio.wait_for(turn, 1)
        assert synth.texts == list(texts)
        assert [chunk.text for chunk in playback.chunks] == list(texts)
        assert len(peaks) == len(texts) and max(peaks) == 32
        assert loop._foreground.pending_publication_count == 0
        assert controller.cancellations == []
    finally:
        playback.release.set()
        entered.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await entered
            await turn
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["revoke", "close", "replace"])
async def test_capacity_wait_is_cancelled_and_cannot_publish_after_revocation(operation):
    coordinator = ForegroundTurnCoordinator(output_capacity=1)
    waiting, finished = asyncio.Event(), asyncio.Event()
    result = []

    async def runner(lease):
        assert await lease.publish("first")
        waiting.set()
        try:
            await lease.publish_when_ready("blocked")
        except asyncio.CancelledError:
            result.append(await lease.publish_when_ready("after revocation"))
        finally:
            finished.set()

    lease = await coordinator.start("owner", runner)
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        assert not finished.is_set()
        if operation == "close":
            await asyncio.wait_for(coordinator.close(), 1)
        elif operation == "replace":
            async def replacement(_lease):
                await asyncio.Future()
            await asyncio.wait_for(coordinator.start("replacement", replacement), 1)
        else:
            revoked = coordinator.revoke_if_current(lease)
            assert revoked is not None
            task, turn = revoked
            await asyncio.wait_for(coordinator.settle_revoked(task, turn_id=turn), 1)
        assert finished.is_set() and result == [False]
        assert coordinator.pending_publication_count == 0
        assert coordinator.active_task_count == (1 if operation == "replace" else 0)
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_waiting_publication_keeps_source_admission_and_rejects_non_owner():
    coordinator = ForegroundTurnCoordinator(output_capacity=1)
    ready, release = asyncio.Event(), asyncio.Event()
    admitted = []

    async def runner(lease: ForegroundTurnLease):
        assert await lease.publish_when_ready(
            "host text", on_admit=admitted.append, source="host_task_status")
        ready.set()
        await release.wait()

    lease = await coordinator.start("owner", runner)
    try:
        await asyncio.wait_for(ready.wait(), 1)
        assert await asyncio.wait_for(lease.publish_when_ready("intruder"), 1) is False
        publication = await asyncio.wait_for(coordinator.next_publication(), 1)
        assert admitted == [publication]
        assert publication.source == "host_task_status" and publication.text == "host text"
        assert publication.sequence == 1 and publication.generation == lease.generation
        with pytest.raises(ValueError):
            await lease.publish_when_ready("invalid", source="not_a_speech_source")
        assert len(admitted) == 1 and coordinator.pending_publication_count == 0
    finally:
        release.set()
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["lease", "task"])
async def test_capacity_wakeup_rechecks_each_atomic_authority(authority):
    coordinator = ForegroundTurnCoordinator(output_capacity=1)
    waiting, finished = asyncio.Event(), asyncio.Event()
    result = []

    async def runner(lease):
        assert await lease.publish("first")
        waiting.set()
        result.append(await lease.publish_when_ready("must be refused"))
        finished.set()

    await coordinator.start("owner", runner)
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        # Isolate an authority change before cancellation is delivered. Only one
        # ownership field changes, so the other cannot hide a missing recheck.
        async with coordinator._publication_available:
            if authority == "lease":
                coordinator._active_lease = None
            else:
                coordinator._active_task = None
            coordinator._publications.clear()
            coordinator._publication_available.notify_all()
        await asyncio.wait_for(finished.wait(), 1)
        assert result == [False]
        assert coordinator.pending_publication_count == 0
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_tts_failure_settles_the_queue_blocked_quarantine_producer():
    texts = tuple(f"Ordinary synthetic response segment number {index:02d}." for index in range(65))
    inference = Inference(texts=texts)
    inference.initial.release.set()
    loop, _, controller, proposals, _, _ = _owner("alpha", inference=inference)
    entered, fail = asyncio.Event(), asyncio.Event()

    class FailingSynthesizer:
        async def synthesize(self, text, turn_id):
            entered.set()
            await fail.wait()
            raise ValueError("scripted synthesis failure")
            yield  # The synthesizer interface is an asynchronous iterator.

        async def cancel(self, turn_id):
            pass

    loop._synthesizer = FailingSynthesizer()
    turn = asyncio.create_task(_say(loop, proposals, "An ordinary question.", "consumer_failure"))
    ready = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait({turn, ready}, timeout=1,
                                    return_when=asyncio.FIRST_COMPLETED)
        assert done
        if turn in done:
            await turn
        await asyncio.wait_for(ready, 1)
        await _settle()
        assert loop._foreground.pending_publication_count == 32
        assert loop._foreground.active_task_count == 1 and not turn.done()
        fail.set()
        with pytest.raises(ValueError, match="scripted synthesis failure"):
            await asyncio.wait_for(turn, 1)
        assert loop._foreground.active_task_count == 0
        assert loop._foreground.pending_publication_count == 0
        assert controller.cancellations == []
    finally:
        fail.set()
        ready.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await ready
            await turn
        await loop.close()
