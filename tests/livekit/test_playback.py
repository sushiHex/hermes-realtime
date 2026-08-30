import asyncio
import struct
from collections.abc import Callable

import pytest

from hermes_realtime.livekit import (
    LiveKitPCMDeliveryConfirmation,
    LiveKitSpeechPlayback,
)
from hermes_realtime.speech import AudioFrame, SpeechChunk, WordTiming


class RecordingPublisher:
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.prepared: list[AudioFrame] = []
        self.clear_count = 0
        self._stream_counter = 0

    async def prepare_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str:
        del timeout_seconds
        self.prepared.append(chunk.audio)
        self._stream_counter += 1
        return f"stream_{self._stream_counter}"

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del timeout_seconds
        self.frames.append(chunk.audio)

    async def finish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del chunk, timeout_seconds

    async def cancel_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del chunk
        del timeout_seconds
        self.clear_count += 1


class RecordingEchoReference:
    def __init__(self) -> None:
        self.started: list[tuple[AudioFrame, str]] = []
        self.ended: list[object] = []
        self.token = object()

    def begin_playback(self, frame: AudioFrame, text: str) -> object:
        self.started.append((frame, text))
        return self.token

    def end_playback(self, token: object) -> None:
        self.ended.append(token)


class EchoReferenceOrderingPublisher(RecordingPublisher):
    def __init__(self, reference: RecordingEchoReference) -> None:
        super().__init__()
        self.reference = reference
        self.started_during_prepare = False
        self.started_at_publish_entry = False

    async def prepare_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str:
        self.started_during_prepare = bool(self.reference.started)
        return await super().prepare_speech_chunk(chunk, timeout_seconds=timeout_seconds)

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self.started_at_publish_entry = bool(self.reference.started)
        await super().publish_speech_chunk(chunk, timeout_seconds=timeout_seconds)


class BlockingConfirmation:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.confirmed: list[SpeechChunk] = []
        self.cancelled_turns: list[str] = []
        self.prepared_turn_id: str | None = None

    async def prepare(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        del stream_identity
        self.prepared_turn_id = chunk.turn_id
        if not is_valid():
            raise asyncio.CancelledError

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        self.started.set()
        await self.release.wait()
        if not is_valid():
            raise asyncio.CancelledError
        self.confirmed.append(chunk)

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)
        if self.prepared_turn_id == turn_id:
            self.release.set()


class BlockingCancellationUnwindConfirmation(BlockingConfirmation):
    def __init__(self) -> None:
        super().__init__()
        self.unwind_started = asyncio.Event()
        self.release_unwind = asyncio.Event()

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        del chunk
        self.started.set()
        await self.release.wait()
        if not is_valid():
            self.unwind_started.set()
            await self.release_unwind.wait()
            raise asyncio.CancelledError


class FailingConfirmation(BlockingConfirmation):
    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        del chunk, is_valid
        raise RuntimeError("confirmation failed")


class RecoveringConfirmationCleanup(FailingConfirmation):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls = 0
        self.fail_confirm = True

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if self.fail_confirm:
            await super().confirm(chunk, is_valid=is_valid)

    async def cancel(self, turn_id: str) -> None:
        self.cancel_calls += 1
        if self.cancel_calls == 1:
            raise TimeoutError("confirmation cancel failed")
        await super().cancel(turn_id)


class BlockingOwnedCleanupConfirmation(FailingConfirmation):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.cancel_calls = 0
        self.fail_confirm = True

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if self.fail_confirm:
            await super().confirm(chunk, is_valid=is_valid)

    async def cancel(self, turn_id: str) -> None:
        self.cancel_calls += 1
        self.cleanup_started.set()
        await self.release_cleanup.wait()
        await super().cancel(turn_id)


class FailingFinishPublisher(RecordingPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.fail_finish = True

    async def finish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del chunk, timeout_seconds
        if self.fail_finish:
            raise RuntimeError("finish failed")

    async def cancel_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        if self.fail_finish:
            raise RuntimeError("finish failed")
        await super().cancel_speech_chunk(chunk, timeout_seconds=timeout_seconds)


class CancellationRacingPublishFailurePublisher(RecordingPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.fail_publish = True
        self.publish_started = asyncio.Event()
        self.release_publish = asyncio.Event()
        self.cancellation_finished = asyncio.Event()
        self.cancel_calls = 0

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        if self.fail_publish:
            self.publish_started.set()
            await self.release_publish.wait()
            raise RuntimeError("publish failed")
        await super().publish_speech_chunk(chunk, timeout_seconds=timeout_seconds)

    async def cancel_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self.cancel_calls += 1
        if self.cancel_calls > 1:
            raise RuntimeError("speech chunk is not prepared")
        await super().cancel_speech_chunk(chunk, timeout_seconds=timeout_seconds)
        self.cancellation_finished.set()


class QueuedReceiver:
    def __init__(self) -> None:
        self.frames: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self.prepared: list[SpeechChunk] = []

    async def prepare_receive_speech_chunk(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del stream_identity, timeout_seconds
        self.prepared.append(chunk)

    async def receive_audio(self, *, timeout_seconds: float) -> AudioFrame:
        return await asyncio.wait_for(self.frames.get(), timeout_seconds)


class FirstPrepareBlockingReceiver(QueuedReceiver):
    def __init__(self) -> None:
        super().__init__()
        self.first_prepare_started = asyncio.Event()
        self.prepare_calls = 0

    async def prepare_receive_speech_chunk(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self.prepare_calls += 1
        if self.prepare_calls == 1:
            self.first_prepare_started.set()
            await asyncio.Event().wait()
        await super().prepare_receive_speech_chunk(
            chunk,
            stream_identity,
            timeout_seconds=timeout_seconds,
        )


class CancellationResistantPrepareReceiver(QueuedReceiver):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_started = asyncio.Event()
        self.unwind_started = asyncio.Event()
        self.release_unwind = asyncio.Event()

    async def prepare_receive_speech_chunk(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self.prepare_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.unwind_started.set()
            await self.release_unwind.wait()
            raise


class StructurallyTaggedPublisher(RecordingPublisher):
    def __init__(self, receiver: QueuedReceiver) -> None:
        super().__init__()
        self._receiver = receiver
        self.first_published = asyncio.Event()

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        await super().publish_speech_chunk(chunk, timeout_seconds=timeout_seconds)
        if chunk.turn_id == "turn_001":
            self.first_published.set()
            return
        await self._receiver.frames.put(chunk.audio)


class CancellationDuringPublishPublisher(RecordingPublisher):
    def __init__(self, receiver: QueuedReceiver) -> None:
        super().__init__()
        self._receiver = receiver
        self._publish_calls = 0
        self.first_publish_started = asyncio.Event()

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self._publish_calls += 1
        if self._publish_calls == 1:
            self.first_publish_started.set()
            await asyncio.Event().wait()
        await super().publish_speech_chunk(chunk, timeout_seconds=timeout_seconds)
        await self._receiver.frames.put(chunk.audio)


def _chunk() -> SpeechChunk:
    return SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="hello",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )


def _marked_chunk(turn_id: str, chunk_id: str, sample: int) -> SpeechChunk:
    return SpeechChunk(
        turn_id=turn_id,
        chunk_id=chunk_id,
        text=chunk_id,
        audio=AudioFrame(
            pcm=sample.to_bytes(2, "little", signed=True),
            sample_rate_hz=16_000,
            channels=1,
        ),
    )


def test_pcm_confirmation_rejects_unbounded_frame_capacity() -> None:
    with pytest.raises(ValueError, match="supported maximum"):
        LiveKitPCMDeliveryConfirmation(QueuedReceiver(), max_frames=4097)


@pytest.mark.asyncio
async def test_pcm_confirmation_accepts_natural_speech_longer_than_legacy_frame_budget() -> None:
    receiver = QueuedReceiver()
    confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=0.2)
    frame_pcm = struct.pack("<160h", *([1_000] * 160))
    chunk = SpeechChunk(
        turn_id="turn_long",
        chunk_id="chunk_long",
        text="A natural sentence longer than the former confirmation budget.",
        audio=AudioFrame(
            pcm=frame_pcm * 300,
            sample_rate_hz=16_000,
            channels=1,
        ),
    )

    await confirmation.prepare(chunk, "worker_stream", is_valid=lambda: True)
    for _ in range(300):
        await receiver.frames.put(AudioFrame(pcm=frame_pcm, sample_rate_hz=16_000, channels=1))

    await confirmation.confirm(chunk, is_valid=lambda: True)


@pytest.mark.asyncio
async def test_pcm_confirmation_accounts_for_source_quiet_onset_without_waiting() -> None:
    receiver = QueuedReceiver()
    confirmation = LiveKitPCMDeliveryConfirmation(
        receiver,
        timeout_seconds=0.05,
        max_frames=8,
    )
    quiet = AudioFrame(pcm=b"\x00\x00" * 160, sample_rate_hz=16_000, channels=1)
    material = AudioFrame(
        pcm=(1000).to_bytes(2, "little", signed=True) * 160,
        sample_rate_hz=16_000,
        channels=1,
    )
    chunk = SpeechChunk(
        turn_id="turn_onset",
        chunk_id="chunk_onset",
        text="Audible after a quiet onset.",
        audio=AudioFrame(
            pcm=quiet.pcm + material.pcm + quiet.pcm,
            sample_rate_hz=16_000,
            channels=1,
        ),
    )
    await confirmation.prepare(chunk, "stream_onset", is_valid=lambda: True)
    for frame in (quiet, quiet, material, quiet):
        await receiver.frames.put(frame)

    await confirmation.confirm(chunk, is_valid=lambda: True)


@pytest.mark.asyncio
async def test_pcm_confirmation_revalidates_mutated_chunk_before_receiver() -> None:
    receiver = QueuedReceiver()
    confirmation = LiveKitPCMDeliveryConfirmation(receiver)
    chunk = _chunk()
    object.__setattr__(chunk, "audio", object())

    with pytest.raises(TypeError, match="AudioFrame"):
        await confirmation.prepare(chunk, "stream_mutated", is_valid=lambda: True)

    assert receiver.prepared == []


@pytest.mark.asyncio
async def test_livekit_playback_reports_exact_prepared_stream_for_advisory_timing() -> None:
    publisher = RecordingPublisher()
    confirmation = BlockingConfirmation()
    prepared: list[tuple[SpeechChunk, str]] = []
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        on_stream_prepared=lambda chunk, stream_id: prepared.append((chunk, stream_id)),
    )
    chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_timed",
        text="hello",
        audio=AudioFrame(
            pcm=b"\x01\x00" * 160,
            sample_rate_hz=16_000,
            channels=1,
        ),
        word_timings=(WordTiming("hello", 0, 160, 0, 5),),
        timing_source="provider",
    )
    task = asyncio.create_task(playback.play(chunk, is_valid=lambda: True))

    await asyncio.wait_for(confirmation.started.wait(), timeout=1)
    assert prepared == [(chunk, "stream_1")]

    confirmation.release.set()
    await task


@pytest.mark.asyncio
async def test_livekit_playback_requires_confirmation_after_transport_write() -> None:
    publisher = RecordingPublisher()
    confirmation = BlockingConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
    )
    task = asyncio.create_task(playback.play(_chunk(), is_valid=lambda: True))

    await asyncio.wait_for(confirmation.started.wait(), timeout=1)
    assert publisher.frames == [_chunk().audio]
    assert task.done() is False

    confirmation.release.set()
    await task
    assert confirmation.confirmed == [_chunk()]


@pytest.mark.asyncio
async def test_livekit_playback_scopes_exact_echo_reference_through_playout() -> None:
    confirmation = BlockingConfirmation()
    reference = RecordingEchoReference()
    playback = LiveKitSpeechPlayback(
        publisher=RecordingPublisher(),
        confirmation=confirmation,
        echo_reference=reference,
    )
    chunk = _chunk()
    task = asyncio.create_task(playback.play(chunk, is_valid=lambda: True))

    await asyncio.wait_for(confirmation.started.wait(), timeout=1)
    assert reference.started == [(chunk.audio, chunk.text)]
    assert reference.ended == []

    confirmation.release.set()
    await task
    assert reference.ended == [reference.token]


@pytest.mark.asyncio
async def test_livekit_playback_starts_echo_reference_at_transport_publish_boundary() -> None:
    confirmation = BlockingConfirmation()
    reference = RecordingEchoReference()
    publisher = EchoReferenceOrderingPublisher(reference)
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        echo_reference=reference,
    )
    task = asyncio.create_task(playback.play(_chunk(), is_valid=lambda: True))

    await asyncio.wait_for(confirmation.started.wait(), timeout=1)
    assert publisher.started_during_prepare is False
    assert publisher.started_at_publish_entry is True

    confirmation.release.set()
    await task


@pytest.mark.asyncio
async def test_repeated_play_cancellation_waits_for_echo_reference_finalization() -> None:
    confirmation = BlockingOwnedCleanupConfirmation()
    reference = RecordingEchoReference()
    playback = LiveKitSpeechPlayback(
        publisher=RecordingPublisher(),
        confirmation=confirmation,
        echo_reference=reference,
    )
    task = asyncio.create_task(playback.play(_chunk(), is_valid=lambda: True))

    await asyncio.wait_for(confirmation.cleanup_started.wait(), timeout=1)
    task.cancel()
    task.cancel()
    confirmation.release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert reference.ended == [reference.token]


@pytest.mark.asyncio
async def test_livekit_playback_rejects_invalid_generation_before_publish() -> None:
    publisher = RecordingPublisher()
    confirmation = BlockingConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
    )

    with pytest.raises(asyncio.CancelledError):
        await playback.play(_chunk(), is_valid=lambda: False)

    assert publisher.frames == []
    assert confirmation.started.is_set() is False


@pytest.mark.asyncio
async def test_livekit_playback_forwards_turn_cancellation() -> None:
    confirmation = BlockingConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=RecordingPublisher(),
        confirmation=confirmation,
    )

    await playback.cancel("turn_001")

    assert confirmation.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_exact_stream_claim_rejects_stale_authority_before_cancelling() -> None:
    publisher = RecordingPublisher()
    confirmation = BlockingConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        stream_generation=lambda _chunk: 7,
    )
    chunk = _chunk()
    playing = asyncio.create_task(playback.play(chunk, is_valid=lambda: True))
    await confirmation.started.wait()

    assert not await playback.cancel_if_active_stream(
        turn_id=chunk.turn_id,
        turn_generation=6,
        chunk_id=chunk.chunk_id,
        stream_identity="stream_1",
    )
    assert not await playback.cancel_if_active_stream(
        turn_id=chunk.turn_id,
        turn_generation=7,
        chunk_id="stale_chunk",
        stream_identity="stream_1",
    )
    assert not await playback.cancel_if_active_stream(
        turn_id=chunk.turn_id,
        turn_generation=7,
        chunk_id=chunk.chunk_id,
        stream_identity="stale_stream",
    )
    assert await playback.cancel_if_active_stream(
        turn_id=chunk.turn_id,
        turn_generation=7,
        chunk_id=chunk.chunk_id,
        stream_identity="stream_1",
    )
    with pytest.raises(asyncio.CancelledError):
        await playing

    assert publisher.clear_count == 1
    assert confirmation.cancelled_turns == [chunk.turn_id]


@pytest.mark.asyncio
async def test_livekit_playback_bounds_delivery_confirmation() -> None:
    confirmation = BlockingConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=RecordingPublisher(),
        confirmation=confirmation,
        confirmation_timeout_seconds=0.02,
    )

    with pytest.raises(TimeoutError, match="delivery confirmation"):
        await playback.play(_chunk(), is_valid=lambda: True)


@pytest.mark.asyncio
async def test_livekit_playback_preserves_confirmation_and_finish_failures() -> None:
    publisher = FailingFinishPublisher()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=FailingConfirmation(),
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        await playback.play(_chunk(), is_valid=lambda: True)

    messages = {str(error) for error in captured.value.exceptions}
    assert messages == {"confirmation failed", "finish failed"}

    with pytest.raises(RuntimeError, match="already has an active chunk"):
        await playback.play(_chunk(), is_valid=lambda: True)

    publisher.fail_finish = False
    await playback.cancel("turn_001")
    with pytest.raises(RuntimeError, match="confirmation failed"):
        await playback.play(_marked_chunk("turn_002", "chunk_002", 1000), is_valid=lambda: True)


@pytest.mark.asyncio
async def test_pcm_confirmation_cancel_interrupts_blocked_receive() -> None:
    receiver = QueuedReceiver()
    confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=1)
    chunk = _marked_chunk("turn_001", "chunk_a", 1000)
    await confirmation.prepare(chunk, "stream_cancel", is_valid=lambda: True)
    task = asyncio.create_task(confirmation.confirm(chunk, is_valid=lambda: True))
    await asyncio.sleep(0)

    await confirmation.cancel("turn_001")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.1)


@pytest.mark.asyncio
async def test_cancel_interrupts_receiver_prepare_before_releasing_publication_authority() -> None:
    receiver = FirstPrepareBlockingReceiver()
    publisher = RecordingPublisher()
    confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=0.05)
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=0.05,
        confirmation_timeout_seconds=0.05,
    )
    first = _marked_chunk("turn_001", "chunk_a", 1000)
    first_task = asyncio.create_task(playback.play(first, is_valid=lambda: True))
    await asyncio.wait_for(receiver.first_prepare_started.wait(), timeout=1)

    await playback.cancel(first.turn_id)
    with pytest.raises(asyncio.CancelledError):
        await first_task

    replacement = _marked_chunk("turn_002", "chunk_b", 2000)
    await receiver.frames.put(replacement.audio)
    await playback.play(replacement, is_valid=lambda: True)
    assert publisher.clear_count == 1


@pytest.mark.asyncio
async def test_child_cleanup_after_finish_failure_does_not_wait_on_its_parent() -> None:
    publisher = FailingFinishPublisher()
    confirmation = BlockingConfirmation()
    confirmation.release.set()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=0.02,
        confirmation_timeout_seconds=0.02,
    )
    failed = _marked_chunk("turn_001", "chunk_a", 1000)

    async def owning_consumer() -> None:
        with pytest.raises(RuntimeError, match="finish failed"):
            await playback.play(failed, is_valid=lambda: True)
        publisher.fail_finish = False
        cleanup = asyncio.create_task(playback.cancel(failed.turn_id))
        await cleanup

    await asyncio.wait_for(owning_consumer(), timeout=1)
    await playback.play(
        _marked_chunk("turn_002", "chunk_b", 2000),
        is_valid=lambda: True,
    )


@pytest.mark.asyncio
async def test_publish_failure_during_external_cancel_can_be_retried() -> None:
    publisher = CancellationRacingPublishFailurePublisher()
    confirmation = BlockingConfirmation()
    confirmation.release.set()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=0.02,
        confirmation_timeout_seconds=0.02,
    )
    failed = _marked_chunk("turn_001", "chunk_a", 1000)
    play_task = asyncio.create_task(playback.play(failed, is_valid=lambda: True))
    await asyncio.wait_for(publisher.publish_started.wait(), timeout=1)

    cancel_task = asyncio.create_task(playback.cancel(failed.turn_id))
    await asyncio.wait_for(publisher.cancellation_finished.wait(), timeout=1)
    publisher.release_publish.set()

    with pytest.raises(RuntimeError, match="publish failed"):
        await play_task
    with pytest.raises(RuntimeError, match="publish failed"):
        await cancel_task

    await playback.cancel(failed.turn_id)
    publisher.fail_publish = False
    await playback.play(
        _marked_chunk("turn_002", "chunk_b", 2000),
        is_valid=lambda: True,
    )


@pytest.mark.asyncio
async def test_confirmation_cleanup_retry_does_not_repeat_publication_release() -> None:
    publisher = CancellationRacingPublishFailurePublisher()
    publisher.fail_publish = False
    confirmation = RecoveringConfirmationCleanup()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=0.02,
        confirmation_timeout_seconds=0.02,
    )
    failed = _marked_chunk("turn_001", "chunk_a", 1000)

    with pytest.raises(BaseExceptionGroup):
        await playback.play(failed, is_valid=lambda: True)

    await playback.cancel(failed.turn_id)
    assert confirmation.cancel_calls == 2
    assert publisher.cancel_calls == 1

    confirmation.fail_confirm = False
    await playback.play(
        _marked_chunk("turn_002", "chunk_b", 2000),
        is_valid=lambda: True,
    )


@pytest.mark.asyncio
async def test_external_cancel_cannot_race_owned_cleanup_release() -> None:
    publisher = CancellationRacingPublishFailurePublisher()
    publisher.fail_publish = False
    confirmation = BlockingOwnedCleanupConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=0.02,
        confirmation_timeout_seconds=1,
    )
    failed = _marked_chunk("turn_001", "chunk_a", 1000)
    play_task = asyncio.create_task(playback.play(failed, is_valid=lambda: True))
    await asyncio.wait_for(confirmation.cleanup_started.wait(), timeout=1)

    cancel_task = asyncio.create_task(playback.cancel(failed.turn_id))
    await asyncio.sleep(0)
    confirmation.release_cleanup.set()

    with pytest.raises(RuntimeError, match="confirmation failed"):
        await play_task
    await cancel_task
    assert confirmation.cancel_calls == 1
    assert publisher.cancel_calls == 1

    confirmation.fail_confirm = False
    await playback.play(
        _marked_chunk("turn_002", "chunk_b", 2000),
        is_valid=lambda: True,
    )


@pytest.mark.asyncio
async def test_cancel_waits_for_receiver_prepare_unwind_before_publication_release() -> None:
    receiver = CancellationResistantPrepareReceiver()
    publisher = RecordingPublisher()
    confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=1)
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=1,
        confirmation_timeout_seconds=1,
    )
    chunk = _marked_chunk("turn_001", "chunk_a", 1000)
    playing = asyncio.create_task(playback.play(chunk, is_valid=lambda: True))
    await asyncio.wait_for(receiver.prepare_started.wait(), timeout=1)

    cancelling = asyncio.create_task(playback.cancel(chunk.turn_id))
    await asyncio.wait_for(receiver.unwind_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert publisher.clear_count == 0
    assert not cancelling.done()

    receiver.release_unwind.set()
    await asyncio.wait_for(cancelling, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await playing
    assert publisher.clear_count == 1


@pytest.mark.asyncio
async def test_structural_replacement_does_not_wait_for_cancelled_track_silence() -> None:
    receiver = QueuedReceiver()
    publisher = StructurallyTaggedPublisher(receiver)
    confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=0.1)
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        confirmation_timeout_seconds=0.2,
    )
    first = _marked_chunk("turn_001", "chunk_a", 1000)
    replacement = _marked_chunk("turn_002", "chunk_b", 2000)
    first_task = asyncio.create_task(playback.play(first, is_valid=lambda: True))
    await asyncio.wait_for(publisher.first_published.wait(), timeout=1)

    await playback.cancel("turn_001")
    with pytest.raises(asyncio.CancelledError):
        await first_task

    await playback.play(replacement, is_valid=lambda: True)

    assert publisher.frames == [first.audio, replacement.audio]


@pytest.mark.asyncio
async def test_play_task_cancellation_releases_prepared_confirmation() -> None:
    receiver = QueuedReceiver()
    publisher = CancellationDuringPublishPublisher(receiver)
    confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=0.2)
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        confirmation_timeout_seconds=0.5,
    )
    first = _marked_chunk("turn_001", "chunk_a", 1000)
    replacement = _marked_chunk("turn_002", "chunk_b", 2000)
    first_task = asyncio.create_task(playback.play(first, is_valid=lambda: True))
    await publisher.first_publish_started.wait()

    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    assert publisher.clear_count == 1

    await playback.play(replacement, is_valid=lambda: True)


@pytest.mark.asyncio
async def test_cancel_waits_for_obsolete_play_cleanup_before_replacement() -> None:
    publisher = RecordingPublisher()
    confirmation = BlockingCancellationUnwindConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
    )
    obsolete = asyncio.create_task(playback.play(_chunk(), is_valid=lambda: True))
    await confirmation.started.wait()

    cancellation = asyncio.create_task(playback.cancel("turn_001"))
    await confirmation.unwind_started.wait()

    assert not cancellation.done()
    with pytest.raises(RuntimeError, match="already has an active chunk"):
        await playback.play(
            _marked_chunk("turn_001", "chunk_001", 2000),
            is_valid=lambda: True,
        )

    confirmation.release_unwind.set()
    await cancellation
    with pytest.raises(asyncio.CancelledError):
        await obsolete


@pytest.mark.asyncio
async def test_livekit_playback_clears_only_cancelled_active_turn_audio() -> None:
    publisher = RecordingPublisher()
    confirmation = BlockingConfirmation()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
    )
    task = asyncio.create_task(playback.play(_chunk(), is_valid=lambda: True))
    await asyncio.wait_for(confirmation.started.wait(), timeout=1)

    await playback.cancel("turn_stale")
    assert publisher.clear_count == 0

    await playback.cancel("turn_001")
    assert publisher.clear_count == 1

    confirmation.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
