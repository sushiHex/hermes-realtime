"""LiveKit-backed speech playback with explicit remote-delivery confirmation."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from typing import Protocol

from hermes_realtime.speech.types import AudioFrame, SpeechChunk

from .adapter import LiveKitRoomPeer

_MAX_CONFIRMATION_FRAMES = 4096
_MATERIAL_SAMPLE_THRESHOLD = 500
_LOGGER = logging.getLogger(__name__)


def _trusted_chunk(chunk: SpeechChunk) -> SpeechChunk:
    if type(chunk) is not SpeechChunk:
        raise TypeError("chunk must be an exact SpeechChunk")
    if type(chunk.audio) is not AudioFrame:
        raise TypeError("chunk.audio must be an exact AudioFrame")
    return SpeechChunk(
        turn_id=chunk.turn_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        audio=AudioFrame(
            pcm=chunk.audio.pcm,
            sample_rate_hz=chunk.audio.sample_rate_hz,
            channels=chunk.audio.channels,
        ),
        word_timings=chunk.word_timings,
        timing_source=chunk.timing_source,
    )


class PlaybackEchoReference(Protocol):
    """Track exact outbound PCM through its acoustic playback lifetime."""

    def begin_playback(self, frame: AudioFrame, text: str) -> object: ...

    def end_playback(self, token: object) -> None: ...


class LiveKitAudioPublisher(Protocol):
    """Publish provider-neutral PCM into a connected LiveKit room."""

    async def prepare_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str: ...

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None: ...

    async def finish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None: ...

    async def cancel_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None: ...


class ReconnectSafeLiveKitAudioPublisher:
    """Pin each admitted speech chunk to one authoritative worker peer generation."""

    def __init__(self, *, max_active_chunks: int = 16) -> None:
        if type(max_active_chunks) is not int:
            raise TypeError("max_active_chunks must be an exact integer")
        if not 1 <= max_active_chunks <= 256:
            raise ValueError("max_active_chunks must be between 1 and 256")
        self._max_active_chunks = max_active_chunks
        self._peer: LiveKitRoomPeer | None = None
        self._chunks: dict[tuple[str, str], LiveKitRoomPeer] = {}
        self._lock = asyncio.Lock()

    async def bind(self, peer: LiveKitRoomPeer) -> None:
        if type(peer) is not LiveKitRoomPeer:
            raise TypeError("peer must be an exact LiveKitRoomPeer")
        async with self._lock:
            if self._peer is not None:
                raise RuntimeError("LiveKit audio publisher is already bound")
            self._peer = peer

    async def unbind(self, peer: LiveKitRoomPeer) -> None:
        if type(peer) is not LiveKitRoomPeer:
            raise TypeError("peer must be an exact LiveKitRoomPeer")
        async with self._lock:
            if self._peer is not peer:
                raise RuntimeError("LiveKit audio publisher does not own this peer")
            if any(owner is peer for owner in self._chunks.values()):
                raise RuntimeError("cannot unbind a peer with active speech chunks")
            self._peer = None

    async def prepare_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str:
        trusted = _trusted_chunk(chunk)
        key = (trusted.turn_id, trusted.chunk_id)
        async with self._lock:
            peer = self._peer
            if peer is None:
                raise RuntimeError("LiveKit audio publisher is not bound")
            if key in self._chunks:
                raise RuntimeError("LiveKit speech chunk is already prepared")
            if len(self._chunks) >= self._max_active_chunks:
                raise RuntimeError("LiveKit publisher chunk capacity exhausted")
            self._chunks[key] = peer
        # The mapping is cleanup authority, not proof that preparation completed.
        # Retain it across failure/cancellation because the peer may already own a
        # publication or audio source; the playback owner must release it exactly.
        return await peer.prepare_speech_chunk(
            trusted,
            timeout_seconds=timeout_seconds,
        )

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        trusted = _trusted_chunk(chunk)
        peer = await self._chunk_peer(trusted)
        await peer.publish_speech_chunk(trusted, timeout_seconds=timeout_seconds)

    async def finish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        await self._release_chunk(chunk, timeout_seconds=timeout_seconds, cancel=False)

    async def cancel_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        await self._release_chunk(chunk, timeout_seconds=timeout_seconds, cancel=True)

    async def _release_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float,
        cancel: bool,
    ) -> None:
        trusted = _trusted_chunk(chunk)
        key = (trusted.turn_id, trusted.chunk_id)
        peer = await self._chunk_peer(trusted)
        if cancel:
            await peer.cancel_speech_chunk(trusted, timeout_seconds=timeout_seconds)
        else:
            await peer.finish_speech_chunk(trusted, timeout_seconds=timeout_seconds)
        async with self._lock:
            if self._chunks.get(key) is peer:
                del self._chunks[key]

    async def _chunk_peer(self, chunk: SpeechChunk) -> LiveKitRoomPeer:
        key = (chunk.turn_id, chunk.chunk_id)
        async with self._lock:
            peer = self._chunks.get(key)
        if peer is None:
            raise RuntimeError("LiveKit speech chunk is not prepared")
        return peer


class LiveKitDeliveryConfirmation(Protocol):
    """Confirm that one exact published speech chunk reached the delivery boundary."""

    async def prepare(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        is_valid: Callable[[], bool],
    ) -> None: ...

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None: ...

    async def cancel(self, turn_id: str) -> None: ...


class LiveKitAudioReceiver(Protocol):
    """Receive decoded PCM from one dedicated remote LiveKit audio track."""

    async def prepare_receive_speech_chunk(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        timeout_seconds: float = 10,
    ) -> None: ...

    async def receive_audio(self, *, timeout_seconds: float) -> AudioFrame: ...


class LiveKitPCMDeliveryConfirmation:
    """Confirm one serialized exact chunk from complete remote decoded PCM."""

    def __init__(
        self,
        receiver: LiveKitAudioReceiver,
        *,
        timeout_seconds: float = 10.0,
        max_frames: int = _MAX_CONFIRMATION_FRAMES,
    ) -> None:
        if type(timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an exact number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(max_frames) is not int:
            raise TypeError("max_frames must be an exact integer")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if max_frames > _MAX_CONFIRMATION_FRAMES:
            raise ValueError("max_frames exceeds supported maximum")
        self._receiver = receiver
        self._timeout_seconds = float(timeout_seconds)
        self._max_frames = max_frames
        self._active: tuple[SpeechChunk, int, str] | None = None
        self._cancelled = asyncio.Event()
        self._prepare_task: asyncio.Task[None] | None = None
        self._receive_task: asyncio.Task[AudioFrame] | None = None
        self._lock = asyncio.Lock()

    async def prepare(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        trusted = _trusted_chunk(chunk)
        if type(stream_identity) is not str:
            raise TypeError("stream_identity must be an exact built-in string")
        if not stream_identity or len(stream_identity) > 128:
            raise ValueError("stream_identity must contain 1 to 128 characters")
        expected_samples = len(trusted.audio.pcm) // (2 * trusted.audio.channels)
        active = (trusted, expected_samples, stream_identity)
        async with self._lock:
            if self._active is not None:
                raise RuntimeError("PCM delivery confirmation is already active")
            self._active = active
            self._cancelled = asyncio.Event()
        prepare_task: asyncio.Task[None] | None = None
        try:
            if not is_valid():
                raise asyncio.CancelledError
            prepare_task = asyncio.create_task(
                self._receiver.prepare_receive_speech_chunk(
                    trusted,
                    stream_identity,
                    timeout_seconds=self._timeout_seconds,
                )
            )
            async with self._lock:
                if self._active != active:
                    prepare_task.cancel()
                    raise asyncio.CancelledError
                self._prepare_task = prepare_task
            await prepare_task
            if not is_valid():
                raise asyncio.CancelledError
        except BaseException:
            async with self._lock:
                if self._active == active:
                    self._active = None
            raise
        finally:
            async with self._lock:
                if self._prepare_task is prepare_task:
                    self._prepare_task = None

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        trusted = _trusted_chunk(chunk)
        async with self._lock:
            active = self._active
            cancelled = self._cancelled
        if active is None or active[0] != trusted:
            raise RuntimeError("PCM confirmation does not match the active chunk")
        expected_samples = active[1]
        confirmation_target_samples = expected_samples
        received_samples = 0
        received_frames = 0
        materially_non_silent = False
        delivery_started = False
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        try:
            while received_samples < confirmation_target_samples:
                if cancelled.is_set() or not is_valid():
                    raise asyncio.CancelledError
                received_frames += 1
                if received_frames > self._max_frames:
                    raise RuntimeError("PCM confirmation frame capacity exceeded")
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("PCM delivery confirmation timed out")
                receive_task = asyncio.create_task(
                    self._receiver.receive_audio(timeout_seconds=remaining)
                )
                async with self._lock:
                    if self._active is not active:
                        receive_task.cancel()
                        raise asyncio.CancelledError
                    self._receive_task = receive_task
                try:
                    frame = await receive_task
                finally:
                    async with self._lock:
                        if self._receive_task is receive_task:
                            self._receive_task = None
                if (
                    frame.sample_rate_hz != trusted.audio.sample_rate_hz
                    or frame.channels != trusted.audio.channels
                ):
                    raise ValueError("confirmed PCM format does not match the chunk")
                frame_samples = len(frame.pcm) // (2 * frame.channels)
                samples = memoryview(frame.pcm).cast("h")
                frame_is_material = (
                    max((abs(sample) for sample in samples), default=0) > _MATERIAL_SAMPLE_THRESHOLD
                )
                if not delivery_started:
                    if not frame_is_material:
                        continue
                    delivery_started = True
                    confirmation_target_samples -= self._source_quiet_onset_samples(
                        trusted.audio,
                        frame_samples,
                    )
                if received_samples + frame_samples > confirmation_target_samples:
                    raise ValueError("confirmed PCM crosses the exact chunk boundary")
                received_samples += frame_samples
                materially_non_silent = materially_non_silent or frame_is_material
            if cancelled.is_set() or not is_valid():
                raise asyncio.CancelledError
            if not materially_non_silent:
                raise AssertionError("confirmed PCM was silent")
        finally:
            async with self._lock:
                if self._active is active:
                    self._active = None

    @staticmethod
    def _source_quiet_onset_samples(audio: AudioFrame, frame_samples: int) -> int:
        samples = memoryview(audio.pcm).cast("h")
        interleaved_frame_samples = frame_samples * audio.channels
        quiet_samples = 0
        for offset in range(0, len(samples), interleaved_frame_samples):
            frame = samples[offset : offset + interleaved_frame_samples]
            if len(frame) != interleaved_frame_samples:
                break
            if max((abs(sample) for sample in frame), default=0) > _MATERIAL_SAMPLE_THRESHOLD:
                break
            quiet_samples += frame_samples
        return quiet_samples

    async def cancel(self, turn_id: str) -> None:
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        receive_task: asyncio.Task[AudioFrame] | None = None
        prepare_task: asyncio.Task[None] | None = None
        async with self._lock:
            active = self._active
            if active is None or active[0].turn_id != turn_id:
                return
            self._cancelled.set()
            prepare_task = self._prepare_task
            receive_task = self._receive_task
        owned_tasks = tuple(task for task in (prepare_task, receive_task) if task is not None)
        for task in owned_tasks:
            if not task.done():
                task.cancel()
        if owned_tasks:
            done, pending = await asyncio.wait(
                owned_tasks,
                timeout=self._timeout_seconds,
            )
            if pending:
                raise TimeoutError("PCM confirmation cancellation exceeded finite timeout")
            task_errors: list[BaseException] = []
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:
                    task_errors.append(error)
        else:
            task_errors = []
        async with self._lock:
            if self._active is active:
                self._active = None
            if self._prepare_task is prepare_task:
                self._prepare_task = None
            if self._receive_task is receive_task:
                self._receive_task = None
        if len(task_errors) == 1:
            raise task_errors[0]
        if task_errors:
            raise BaseExceptionGroup(
                "PCM confirmation cancellation failed",
                task_errors,
            )


class LiveKitSpeechPlayback:
    """Publish one chunk and await explicit chunk-scoped delivery confirmation."""

    def __init__(
        self,
        *,
        publisher: LiveKitAudioPublisher,
        confirmation: LiveKitDeliveryConfirmation,
        publish_timeout_seconds: float = 10.0,
        confirmation_timeout_seconds: float = 10.0,
        on_stream_prepared: Callable[[SpeechChunk, str], None] | None = None,
        stream_generation: Callable[[SpeechChunk], int | None] | None = None,
        echo_reference: PlaybackEchoReference | None = None,
    ) -> None:
        if type(publish_timeout_seconds) not in (int, float):
            raise TypeError("publish_timeout_seconds must be an exact number")
        if not math.isfinite(publish_timeout_seconds) or publish_timeout_seconds <= 0:
            raise ValueError("publish_timeout_seconds must be finite and positive")
        if type(confirmation_timeout_seconds) not in (int, float):
            raise TypeError("confirmation_timeout_seconds must be an exact number")
        if not math.isfinite(confirmation_timeout_seconds) or confirmation_timeout_seconds <= 0:
            raise ValueError("confirmation_timeout_seconds must be finite and positive")
        if echo_reference is not None:
            for method in ("begin_playback", "end_playback"):
                if not callable(getattr(echo_reference, method, None)):
                    raise TypeError(f"echo_reference must provide {method}()")
        self._publisher = publisher
        self._confirmation = confirmation
        self._publish_timeout_seconds = float(publish_timeout_seconds)
        self._confirmation_timeout_seconds = float(confirmation_timeout_seconds)
        self._on_stream_prepared = on_stream_prepared
        self._stream_generation = stream_generation
        self._echo_reference = echo_reference
        self._lifecycle_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._active: (
            tuple[
                SpeechChunk,
                object,
                asyncio.Task[None] | None,
            ]
            | None
        ) = None
        self._active_stream: tuple[object, str, int | None] | None = None
        self._cancelled: set[object] = set()
        self._confirmation_released: set[object] = set()
        self._publication_released: set[object] = set()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        trusted = _trusted_chunk(chunk)
        token = object()
        play_task = asyncio.current_task()
        if play_task is None:
            raise RuntimeError("LiveKit playback requires an owning task")
        echo_token: object | None = None
        async with self._lifecycle_lock:
            if self._active is not None:
                raise RuntimeError("LiveKit speech playback already has an active chunk")
            reference = self._echo_reference
            self._active = (trusted, token, play_task)

        def publication_is_valid() -> bool:
            return (
                is_valid()
                and self._active == (trusted, token, play_task)
                and token not in self._cancelled
            )

        def begin_echo_reference() -> None:
            nonlocal echo_token
            if reference is not None:
                echo_token = reference.begin_playback(trusted.audio, trusted.text)

        operation_error: BaseException | None = None
        try:
            await self._play_active(
                trusted,
                token,
                play_task,
                publication_is_valid,
                begin_echo_reference,
            )
        except BaseException as error:
            operation_error = error

        async def finalize() -> list[BaseException]:
            async with self._cleanup_lock:
                external_cleanup_owned = token in self._cancelled
                cleanup_errors: list[BaseException] = []
                if operation_error is not None and not external_cleanup_owned:
                    try:
                        async with asyncio.timeout(self._confirmation_timeout_seconds):
                            await self._confirmation.cancel(trusted.turn_id)
                        self._confirmation_released.add(token)
                    except BaseException as error:
                        cleanup_errors.append(error)
                try:
                    if operation_error is None:
                        await self._publisher.finish_speech_chunk(
                            trusted,
                            timeout_seconds=self._publish_timeout_seconds,
                        )
                    elif not external_cleanup_owned:
                        await self._publisher.cancel_speech_chunk(
                            trusted,
                            timeout_seconds=self._publish_timeout_seconds,
                        )
                        self._publication_released.add(token)
                except BaseException as error:
                    cleanup_errors.append(error)

                if cleanup_errors or (external_cleanup_owned and operation_error is not None):
                    async with self._lifecycle_lock:
                        if self._active == (trusted, token, play_task):
                            self._active = (trusted, token, None)

                if not cleanup_errors and not external_cleanup_owned:
                    async with self._lifecycle_lock:
                        if self._active == (trusted, token, play_task):
                            self._active = None
                        if self._active_stream is not None and self._active_stream[0] is token:
                            self._active_stream = None
                        self._cancelled.discard(token)
                        self._confirmation_released.discard(token)
                        self._publication_released.discard(token)

            if reference is not None and echo_token is not None:
                try:
                    reference.end_playback(echo_token)
                except BaseException as error:
                    cleanup_errors.append(error)
            return cleanup_errors

        cleanup_task = asyncio.create_task(
            finalize(),
            name=f"livekit-playback-finalize:{trusted.chunk_id}",
        )
        cleanup_errors, caller_cancelled = await self._await_owned_finalization(cleanup_task)
        if caller_cancelled:
            raise asyncio.CancelledError

        errors = ([operation_error] if operation_error is not None else []) + cleanup_errors
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "LiveKit playback operation and cleanup failed",
                errors,
            )

    async def _play_active(
        self,
        trusted: SpeechChunk,
        token: object,
        play_task: asyncio.Task[None],
        publication_is_valid: Callable[[], bool],
        begin_echo_reference: Callable[[], None],
    ) -> None:
        if not publication_is_valid():
            raise asyncio.CancelledError
        stream_identity = await self._publisher.prepare_speech_chunk(
            trusted,
            timeout_seconds=self._publish_timeout_seconds,
        )
        if type(stream_identity) is not str:
            raise TypeError("publisher stream identity must be an exact built-in string")
        if not stream_identity or len(stream_identity) > 128:
            raise ValueError("publisher stream identity must contain 1 to 128 characters")
        if not publication_is_valid():
            raise asyncio.CancelledError
        generation_resolver = self._stream_generation
        turn_generation = generation_resolver(trusted) if generation_resolver is not None else None
        if turn_generation is not None and (
            type(turn_generation) is not int or not 1 <= turn_generation <= (1 << 53) - 1
        ):
            raise TypeError("stream generation resolver must return a safe exact integer or None")
        async with self._lifecycle_lock:
            if self._active != (trusted, token, play_task) or token in self._cancelled:
                raise asyncio.CancelledError
            self._active_stream = (token, stream_identity, turn_generation)
        deadline = asyncio.get_running_loop().time() + self._confirmation_timeout_seconds
        try:
            async with asyncio.timeout(self._remaining_confirmation_time(deadline)):
                await self._confirmation.prepare(
                    trusted,
                    stream_identity,
                    is_valid=publication_is_valid,
                )
            callback = self._on_stream_prepared
            if callback is not None:
                try:
                    callback(trusted, stream_identity)
                except Exception:
                    _LOGGER.exception(
                        "speech timing projection failed for %s",
                        trusted.chunk_id,
                    )
            if not publication_is_valid():
                raise asyncio.CancelledError
            begin_echo_reference()
            await self._publisher.publish_speech_chunk(
                trusted,
                timeout_seconds=min(
                    self._publish_timeout_seconds,
                    self._remaining_confirmation_time(deadline),
                ),
            )
            if not publication_is_valid():
                raise asyncio.CancelledError
            async with asyncio.timeout(self._remaining_confirmation_time(deadline)):
                await self._confirmation.confirm(
                    trusted,
                    is_valid=publication_is_valid,
                )
        except TimeoutError:
            raise TimeoutError("LiveKit delivery confirmation exceeded finite timeout") from None
        if not publication_is_valid():
            raise asyncio.CancelledError

    @staticmethod
    def _remaining_confirmation_time(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("LiveKit delivery confirmation exceeded finite timeout")
        return remaining

    async def cancel(self, turn_id: str) -> None:
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id:
            raise ValueError("turn_id must not be empty")
        claimed_token: object | None = None
        async with self._lifecycle_lock:
            claimed = self._active
            if claimed is not None and claimed[0].turn_id == turn_id:
                claimed_token = claimed[1]
                self._cancelled.add(claimed_token)
        if claimed_token is None:
            async with self._cleanup_lock:
                async with asyncio.timeout(self._confirmation_timeout_seconds):
                    await self._confirmation.cancel(turn_id)
            return

        await self._cancel_claimed(turn_id, claimed_token)

    async def cancel_if_active_stream(
        self,
        *,
        turn_id: str,
        turn_generation: int,
        chunk_id: str,
        stream_identity: str,
    ) -> bool:
        """Cancel only an exact active chunk/stream authority claim."""

        for name, value in (
            ("turn_id", turn_id),
            ("chunk_id", chunk_id),
            ("stream_identity", stream_identity),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact built-in string")
            if not value or len(value) > 128:
                raise ValueError(f"{name} must contain 1 to 128 characters")
        if type(turn_generation) is not int:
            raise TypeError("turn_generation must be an exact integer")
        if not 1 <= turn_generation <= (1 << 53) - 1:
            raise ValueError("turn_generation must be a positive safe integer")
        async with self._lifecycle_lock:
            active = self._active
            if (
                active is None
                or active[0].turn_id != turn_id
                or active[0].chunk_id != chunk_id
                or self._active_stream != (active[1], stream_identity, turn_generation)
            ):
                return False
            claimed_token = active[1]
            self._cancelled.add(claimed_token)
        await self._cancel_claimed(turn_id, claimed_token)
        return True

    async def _cancel_claimed(self, turn_id: str, claimed_token: object) -> None:

        cleanup_errors: list[BaseException] = []
        active_chunk: SpeechChunk | None = None
        active_play_task: asyncio.Task[None] | None = None
        async with self._cleanup_lock:
            async with self._lifecycle_lock:
                active = self._active
                if active is None or active[0].turn_id != turn_id or active[1] is not claimed_token:
                    return
                active_chunk, _, active_play_task = active
                confirmation_released = claimed_token in self._confirmation_released
                publication_released = claimed_token in self._publication_released

            if not confirmation_released:
                try:
                    async with asyncio.timeout(self._confirmation_timeout_seconds):
                        await self._confirmation.cancel(turn_id)
                    confirmation_released = True
                    self._confirmation_released.add(claimed_token)
                except BaseException as error:
                    cleanup_errors.append(error)
            if confirmation_released and not publication_released:
                try:
                    await self._publisher.cancel_speech_chunk(
                        active_chunk,
                        timeout_seconds=self._publish_timeout_seconds,
                    )
                    self._publication_released.add(claimed_token)
                except BaseException as error:
                    cleanup_errors.append(error)

        # The play task may need the cleanup lock to observe this cancellation
        # claim and detach its completed owner, so never settle it while locked.
        if active_play_task is not None:
            cleanup_errors.extend(await self._settle_play_task(active_play_task))

        if not cleanup_errors:
            async with self._cleanup_lock, self._lifecycle_lock:
                active = self._active
                if active == (active_chunk, claimed_token, active_play_task) or active == (
                    active_chunk,
                    claimed_token,
                    None,
                ):
                    self._active = None
                if self._active_stream is not None and self._active_stream[0] is claimed_token:
                    self._active_stream = None
                self._cancelled.discard(claimed_token)
                self._confirmation_released.discard(claimed_token)
                self._publication_released.discard(claimed_token)
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise BaseExceptionGroup("LiveKit playback cancellation failed", cleanup_errors)

    async def _settle_play_task(
        self,
        play_task: asyncio.Task[None],
    ) -> list[BaseException]:
        if play_task is asyncio.current_task():
            return []
        errors: list[BaseException] = []
        caller_cancellation: asyncio.CancelledError | None = None
        deadline = asyncio.get_running_loop().time() + (
            self._publish_timeout_seconds + self._confirmation_timeout_seconds
        )
        while not play_task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                errors.append(TimeoutError("LiveKit play cleanup exceeded finite timeout"))
                break
            try:
                done, _ = await asyncio.wait({play_task}, timeout=remaining)
                if not done:
                    continue
            except asyncio.CancelledError as error:
                caller_cancellation = error
                continue
        if play_task.done():
            try:
                play_task.result()
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                if self._contains_non_cancellation(error):
                    errors.append(error)
        if caller_cancellation is not None:
            errors.insert(0, caller_cancellation)
        return errors

    @staticmethod
    async def _await_owned_finalization(
        finalization: asyncio.Task[list[BaseException]],
    ) -> tuple[list[BaseException], bool]:
        caller_cancelled = False
        while True:
            try:
                return await asyncio.shield(finalization), caller_cancelled
            except asyncio.CancelledError:
                caller_cancelled = True

    @staticmethod
    def _contains_non_cancellation(error: BaseException) -> bool:
        if isinstance(error, asyncio.CancelledError):
            return False
        if isinstance(error, BaseExceptionGroup):
            return any(
                LiveKitSpeechPlayback._contains_non_cancellation(candidate)
                for candidate in error.exceptions
            )
        return True
