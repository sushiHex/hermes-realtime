"""Deterministic in-memory conversation harness."""

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Protocol, runtime_checkable

from hermes_realtime.conversation import TurnState, TurnStateMachine
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    PlaybackReceipt,
    SpeechChunk,
    StreamingSynthesizer,
    StreamingTranscriber,
    Transcript,
    VoiceActivity,
    VoiceActivityDetector,
)


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


class InMemoryConversationHarness:
    """Exercise media lifecycle and cancellation without network services."""

    def __init__(
        self,
        *,
        vad: VoiceActivityDetector,
        stt: StreamingTranscriber,
        tts: StreamingSynthesizer,
    ) -> None:
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._participants: set[str] = set()
        self._active_background_work: set[str] = set()
        self._events: list[str] = []
        self._ledger = DeliveredSpeechLedger()
        self._turn_state = TurnStateMachine()
        self._speech_cancel_lock = asyncio.Lock()
        self._turn_cancel_lock = asyncio.Lock()
        self._disconnect_lock = asyncio.Lock()
        self._disconnecting = False
        self._disconnect_requests = 0
        self._disconnect_generation = 0
        self._turn_generation = 0
        self._participant_generation = 0
        self._participant_generations: dict[str, int] = {}
        self._turn_id: str | None = None
        self._synthesis: AsyncIterator[SpeechChunk] | None = None
        self._pending_synthesis_task: asyncio.Future[SpeechChunk] | None = None
        self._speech_cancelled_tasks: set[asyncio.Future[SpeechChunk]] = set()
        self._speech_cancel_requested_tasks: set[asyncio.Future[SpeechChunk]] = set()
        self._synthesis_finished = False
        self._speaking = False

    @property
    def participants(self) -> frozenset[str]:
        return frozenset(self._participants)

    @property
    def active_background_work(self) -> frozenset[str]:
        return frozenset(self._active_background_work)

    @property
    def queued_speech(self) -> tuple[SpeechChunk, ...]:
        return self._ledger.pending()

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    @property
    def turn_state(self) -> TurnState:
        return self._turn_state.current

    @property
    def turn_state_history(self) -> tuple[TurnState, ...]:
        return self._turn_state.history

    @property
    def active_turn_id(self) -> str | None:
        return self._turn_id

    def delivered_text(self, turn_id: str) -> str:
        return self._ledger.delivered_text(turn_id)

    def join(self, identity: str) -> None:
        if not identity.strip():
            raise ValueError("participant identity must not be blank")
        if self._disconnecting or self._disconnect_requests:
            raise RuntimeError("disconnect is in progress")
        if identity in self._participants:
            raise ValueError(f"participant already joined: {identity}")
        self._participant_generation += 1
        self._participant_generations[identity] = self._participant_generation
        self._participants.add(identity)
        self._events.append(f"participant.joined:{identity}")

    def start_background_work(self, task_id: str) -> None:
        if not task_id.strip():
            raise ValueError("task_id must not be blank")
        self._active_background_work.add(task_id)

    async def receive_audio(
        self, identity: str, frame: AudioFrame
    ) -> tuple[Transcript, ...]:
        participant_generation = self._participant_generations.get(identity)
        if participant_generation is None:
            raise ValueError(f"participant is not joined: {identity}")
        if self._disconnecting or self._disconnect_requests:
            raise RuntimeError("disconnect is in progress")

        async with self._disconnect_lock:
            if self._disconnecting or self._disconnect_requests:
                raise RuntimeError("disconnect is in progress")
            if self._participant_generations.get(identity) != participant_generation:
                raise ValueError(f"participant is not joined: {identity}")
            return await self._receive_audio_connected(identity, frame)

    async def _receive_audio_connected(
        self, identity: str, frame: AudioFrame
    ) -> tuple[Transcript, ...]:
        transcripts = list(await self._stt.push(frame))
        activity = self._vad.process(frame)
        if activity is VoiceActivity.SPEECH_STARTED:
            try:
                if self._turn_id is not None:
                    await self.cancel_turn()
            finally:
                if self._turn_state.current is not TurnState.LISTENING:
                    self._turn_state.transition(TurnState.LISTENING)
                self._events.append(f"user.speech.started:{identity}")
        elif activity is VoiceActivity.SPEECH_ENDED:
            self._turn_state.transition(TurnState.TRANSCRIBING)
            self._events.append(f"user.speech.ended:{identity}")
            final_transcript = await self._stt.finish_utterance()
            if final_transcript is not None:
                transcripts.append(final_transcript)

        for transcript in transcripts:
            state = "final" if transcript.final else "partial"
            self._events.append(f"user.transcript.{state}:{transcript.text}")
        return tuple(transcripts)

    async def begin_response(self, turn_id: str, text: str) -> None:
        if not turn_id.strip() or not text.strip():
            raise ValueError("turn_id and response text must not be blank")
        disconnect_generation = self._disconnect_generation
        if self._disconnecting or self._disconnect_requests:
            raise RuntimeError("disconnect is in progress")
        async with self._disconnect_lock:
            if (
                self._disconnecting
                or self._disconnect_requests
                or self._disconnect_generation != disconnect_generation
            ):
                raise RuntimeError("disconnect occurred during response admission")
            self._begin_response_locked(turn_id, text)

    def _begin_response_locked(self, turn_id: str, text: str) -> None:
        if self._turn_id is not None:
            raise RuntimeError("a response is already active")
        self._turn_state.transition(TurnState.RESPONDING)
        self._ledger.begin_turn(turn_id)
        self._turn_generation += 1
        self._turn_id = turn_id
        try:
            self._synthesis = self._tts.synthesize(text, turn_id)
        except BaseException:
            self._turn_state.transition(TurnState.RECOVERING)
            self._turn_id = None
            self._synthesis = None
            self._turn_state.transition(TurnState.IDLE)
            raise
        self._synthesis_finished = False
        self._events.append(f"assistant.response.started:{turn_id}")

    async def synthesize_next(self) -> bool:
        if self._disconnecting or self._disconnect_requests:
            return False
        if self._synthesis is None or self._turn_id is None:
            return False
        if self._turn_state.current not in {TurnState.RESPONDING, TurnState.SPEAKING}:
            return False
        if self._pending_synthesis_task is not None:
            raise RuntimeError("synthesis is already in progress")
        synthesis_task: asyncio.Future[SpeechChunk] = asyncio.ensure_future(
            anext(self._synthesis)
        )
        self._pending_synthesis_task = synthesis_task
        chunk: SpeechChunk | None = None
        operation_error: BaseException | None = None
        caller_cancelled = False
        try:
            chunk = await asyncio.shield(synthesis_task)
        except BaseException as error:
            operation_error = error
            current_task = asyncio.current_task()
            caller_cancelled = (
                isinstance(error, asyncio.CancelledError)
                and current_task is not None
                and current_task.cancelling() > 0
            )

        try:
            await self._speech_cancel_lock.acquire()
            try:
                if synthesis_task in self._speech_cancelled_tasks:
                    cancellation_requested = (
                        synthesis_task in self._speech_cancel_requested_tasks
                    )
                    self._speech_cancelled_tasks.discard(synthesis_task)
                    self._speech_cancel_requested_tasks.discard(synthesis_task)
                    if self._pending_synthesis_task is synthesis_task:
                        self._pending_synthesis_task = None
                    if operation_error is not None:
                        if caller_cancelled:
                            raise operation_error
                        if isinstance(operation_error, StopAsyncIteration):
                            return False
                        if not (
                            isinstance(operation_error, asyncio.CancelledError)
                            and cancellation_requested
                        ):
                            raise operation_error
                    return False

                if isinstance(operation_error, StopAsyncIteration):
                    self._pending_synthesis_task = None
                    self._synthesis = None
                    self._synthesis_finished = True
                    self._complete_response_if_drained()
                    return False

                if operation_error is not None:
                    cleanup_errors = await self._recover_synthesis_failure()
                    if cleanup_errors:
                        raise BaseExceptionGroup(
                            "synthesis and cleanup failed",
                            [operation_error, *cleanup_errors],
                        ) from None
                    raise operation_error

                self._pending_synthesis_task = None
                assert chunk is not None
                try:
                    if chunk.turn_id != self._turn_id:
                        raise ValueError(f"unexpected turn_id: {chunk.turn_id}")
                    self._ledger.queue(chunk)
                except BaseException as validation_error:
                    cleanup_errors = await self._recover_synthesis_failure()
                    if cleanup_errors:
                        raise BaseExceptionGroup(
                            "synthesis validation and cleanup failed",
                            [validation_error, *cleanup_errors],
                        ) from None
                    raise
            finally:
                self._speech_cancel_lock.release()
        except asyncio.CancelledError:
            self._speech_cancelled_tasks.discard(synthesis_task)
            self._speech_cancel_requested_tasks.discard(synthesis_task)
            if self._pending_synthesis_task is synthesis_task and synthesis_task.done():
                self._pending_synthesis_task = None
            raise
        return True

    async def _settle_pending_synthesis(
        self, *, speech_cancelled: bool
    ) -> list[BaseException]:
        task = self._pending_synthesis_task
        if task is None:
            return []
        if speech_cancelled:
            self._speech_cancelled_tasks.add(task)
        errors: list[BaseException] = []
        if not task.done():
            if speech_cancelled:
                self._speech_cancel_requested_tasks.add(task)
            task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    if task.done():
                        break
                    errors.append(error)
                except BaseException as error:
                    errors.append(error)
                    break
        self._pending_synthesis_task = None
        return errors

    @staticmethod
    async def _await_cleanup(operation: Awaitable[None]) -> list[BaseException]:
        task: asyncio.Future[None] = asyncio.ensure_future(operation)
        errors: list[BaseException] = []
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError as error:
                errors.append(error)
                if not task.done():
                    continue
                if not task.cancelled():
                    try:
                        task.result()
                    except BaseException as operation_error:
                        errors.append(operation_error)
                break
            except BaseException as error:
                errors.append(error)
                break
        return errors

    async def _recover_synthesis_failure(self) -> list[BaseException]:
        if self._turn_id is None:
            return []
        turn_id = self._turn_id
        synthesis = self._synthesis
        cleanup_errors = await self._settle_pending_synthesis(speech_cancelled=False)
        self._turn_state.transition(TurnState.RECOVERING)
        cleanup_errors.extend(await self._await_cleanup(self._tts.cancel(turn_id)))
        if isinstance(synthesis, _AsyncClosable):
            cleanup_errors.extend(await self._await_cleanup(synthesis.aclose()))
        self._ledger.cancel_pending(turn_id)
        self._ledger.close_turn(turn_id)
        self._synthesis = None
        self._synthesis_finished = False
        if self._speaking:
            self._events.append(f"assistant.playback.interrupted:{turn_id}")
            self._events.append(f"assistant.speaking.stopped:{turn_id}")
        self._speaking = False
        self._turn_state.transition(TurnState.INTERRUPTED)
        return cleanup_errors

    def start_next_playback(self) -> PlaybackReceipt | None:
        if self._disconnecting or self._disconnect_requests:
            return None
        if self._turn_id is None:
            return None
        if self._turn_state.current not in {TurnState.RESPONDING, TurnState.SPEAKING}:
            return None
        queued = self._ledger.queued(self._turn_id)
        if not queued:
            return None
        if not self._speaking:
            self._turn_state.transition(TurnState.SPEAKING)
            self._speaking = True
            self._events.append(f"assistant.speaking.started:{self._turn_id}")
        return self._ledger.mark_started(self._turn_id, queued[0].chunk_id)

    def confirm_playback_delivered(self, receipt: PlaybackReceipt) -> str:
        if self._disconnecting or self._disconnect_requests:
            raise RuntimeError("disconnect is in progress")
        if self._turn_id is None:
            raise RuntimeError("no response is active")
        if self._turn_state.current is not TurnState.SPEAKING:
            raise RuntimeError("playback is not active")
        delivered = self._ledger.mark_delivered(receipt)
        self._complete_response_if_drained()
        return delivered.text

    async def play_next(self) -> str | None:
        started = self.start_next_playback()
        if started is None:
            return None
        return self.confirm_playback_delivered(started)

    def _complete_response_if_drained(self) -> None:
        if (
            self._turn_id is None
            or not self._synthesis_finished
            or self._ledger.pending(self._turn_id)
        ):
            return
        turn_id = self._turn_id
        self._events.append(f"assistant.response.completed:{turn_id}")
        self._ledger.close_turn(turn_id)
        if self._speaking:
            self._events.append(f"assistant.speaking.stopped:{turn_id}")
        self._speaking = False
        self._turn_id = None
        self._synthesis_finished = False
        self._turn_state.transition(TurnState.IDLE)

    async def cancel_speech(self) -> None:
        async with self._speech_cancel_lock:
            await self._cancel_speech_locked()

    async def _cancel_speech_locked(self) -> None:
        if self._turn_id is None:
            return
        turn_id = self._turn_id
        if (
            self._turn_state.current is TurnState.INTERRUPTED
            and self._synthesis is None
            and not self._ledger.pending(turn_id)
        ):
            return
        synthesis = self._synthesis
        self._turn_state.transition(TurnState.INTERRUPTED)
        cleanup_errors = await self._settle_pending_synthesis(speech_cancelled=True)
        cleanup_errors.extend(await self._await_cleanup(self._tts.cancel(turn_id)))
        if isinstance(synthesis, _AsyncClosable):
            cleanup_errors.extend(await self._await_cleanup(synthesis.aclose()))
        if cleanup_errors:
            self._turn_state.transition(TurnState.RECOVERING)
        self._ledger.cancel_pending(turn_id)
        self._ledger.close_turn(turn_id)
        self._synthesis = None
        self._synthesis_finished = False
        if self._speaking:
            self._events.append(f"assistant.playback.interrupted:{turn_id}")
            self._events.append(f"assistant.speaking.stopped:{turn_id}")
        self._speaking = False
        if self._turn_state.current is TurnState.RECOVERING:
            self._turn_state.transition(TurnState.INTERRUPTED)
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise BaseExceptionGroup("speech cancellation failed", cleanup_errors)

    async def cancel_turn(self) -> None:
        if self._turn_id is None:
            return
        target_turn_generation = self._turn_generation
        cleanup_errors = await self._await_cleanup(
            self._cancel_turn_serialized(target_turn_generation)
        )
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise BaseExceptionGroup("turn cancellation failed", cleanup_errors)

    async def _cancel_turn_serialized(self, target_turn_generation: int) -> None:
        async with self._turn_cancel_lock:
            if (
                self._turn_id is None
                or self._turn_generation != target_turn_generation
            ):
                return
            cleanup_errors = await self._await_cleanup(self.cancel_speech())
            self._turn_id = None
            if self._turn_state.current is not TurnState.IDLE:
                self._turn_state.transition(TurnState.IDLE)
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            if cleanup_errors:
                raise BaseExceptionGroup("turn cancellation failed", cleanup_errors)

    async def disconnect(self, identity: str) -> None:
        participant_generation = self._participant_generations.get(identity)
        if participant_generation is None:
            raise ValueError(f"participant is not joined: {identity}")
        cleanup_errors: list[BaseException] = []
        self._disconnect_requests += 1
        try:
            cleanup_errors = await self._await_cleanup(
                self._disconnect_serialized(identity, participant_generation)
            )
        finally:
            self._disconnect_requests -= 1
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise BaseExceptionGroup("disconnect cleanup failed", cleanup_errors)

    async def _disconnect_serialized(
        self, identity: str, participant_generation: int
    ) -> None:
        async with self._disconnect_lock:
            if self._participant_generations.get(identity) != participant_generation:
                return
            self._disconnect_generation += 1
            self._disconnecting = True
            try:
                await self._finish_disconnect(identity)
            finally:
                self._disconnecting = False

    async def _finish_disconnect(self, identity: str) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            await self.cancel_turn()
        except BaseException as error:
            cleanup_errors.append(error)
        cleanup_errors.extend(await self._await_cleanup(self._stt.cancel()))
        if self._turn_state.current is not TurnState.IDLE:
            self._turn_state.transition(TurnState.IDLE)
        self._participants.remove(identity)
        del self._participant_generations[identity]
        self._events.append(f"participant.left:{identity}")
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise BaseExceptionGroup("disconnect cleanup failed", cleanup_errors)
