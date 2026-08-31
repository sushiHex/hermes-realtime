"""Production session owner for microphone endpointing and foreground responses."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol, cast

from hermes_realtime.evidence import (
    CommandRoutingOutcome,
    CommandRoutingResultV1,
    FinalInputAuthorityV1,
    InputSource,
    UserTurnAuthorityV1,
)
from hermes_realtime.evidence.lifecycle import EvidenceConversationAuthorityV1
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    _ProductionObservationRecorderV1,
)
from hermes_realtime.speech import (
    AudioFrame,
    ParticipantAudioFrame,
    SpeechPresence,
    SpeechPresenceVerifier,
    Transcript,
    VoiceActivity,
)

from .actions import ConversationUpdateExecutor
from .ingress import BoundedPcmIngress, IngressRecord
from .knowledge import KnowledgePrefetchCoordinator
from .telemetry import KnowledgeTurnBudget, UtteranceTicket

_MAX_IDENTITY_CHARS = 128
_MAX_SESSION_GENERATION = (1 << 63) - 1
_MAX_RESPONSE_OPERATIONS = 64
_MAX_TYPED_TRANSCRIPT_CHARS = 4096
_DEFAULT_MAX_BUFFERED_AUDIO_BYTES = 256 * 1024
_MAX_BUFFERED_AUDIO_BYTES = 2 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)

_PublicValue = str | int | bool | None
_Observer = Callable[[str, dict[str, _PublicValue]], None]


def _operation_failed(operation: asyncio.Task[None]) -> bool:
    if not operation.done():
        return False
    if operation.cancelled():
        return True
    return operation.exception() is not None


class _VoiceActivityDetector(Protocol):
    def process(self, frame: AudioFrame) -> VoiceActivity: ...


class _StreamingTranscriber(Protocol):
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]: ...

    async def finish_utterance(self) -> Transcript | None: ...

    async def cancel(self) -> None: ...


class _TaskCommandRouter(Protocol):
    async def route(
        self,
        text: str,
        authority: FinalInputAuthorityV1,
    ) -> CommandRoutingResultV1: ...


class _ParticipantAudioSource(Protocol):
    async def receive_participant_audio(
        self,
        *,
        timeout_seconds: float,
    ) -> ParticipantAudioFrame: ...


class _PlaybackEchoGuard(Protocol):
    def is_echo_dominated(self, frames: tuple[AudioFrame, ...]) -> bool: ...

    def is_transcript_echo(self, text: str) -> bool: ...

    def has_recent_playback(self) -> bool: ...


class ConversationSessionWorker:
    """Own one participant/session generation from PCM through response admission."""

    def __init__(
        self,
        *,
        participant_identity: str,
        session_generation: int,
        media_incarnation_provider: Callable[[], int | None] | None = None,
        vad: _VoiceActivityDetector,
        stt: _StreamingTranscriber,
        actions: ConversationUpdateExecutor,
        stt_streams_partials: bool = True,
        command_router: _TaskCommandRouter | None = None,
        evidence_lifecycle: EvidenceConversationAuthorityV1 | None = None,
        evidence_lifecycle_resolver: (
            Callable[[], EvidenceConversationAuthorityV1 | None] | None
        ) = None,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
        observer: _Observer | None = None,
        knowledge_coordinator: KnowledgePrefetchCoordinator | None = None,
        knowledge_budget_seconds: float = 3.5,
        on_audio_ready: Callable[[], None] | None = None,
        audio_ready_repeat_frames: int = 50,
        on_voice_activity: Callable[[bool], None] | None = None,
        max_response_operations: int = 16,
        cleanup_timeout_seconds: float = 5.0,
        pre_roll_frames: int = 1,
        max_buffered_audio_bytes: int = _DEFAULT_MAX_BUFFERED_AUDIO_BYTES,
        echo_guard: _PlaybackEchoGuard | None = None,
        speech_presence_verifier: SpeechPresenceVerifier | None = None,
        echo_recheck_frames: int = 8,
        echo_promotion_checks: int = 2,
    ) -> None:
        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if not participant_identity.strip() or len(participant_identity) > _MAX_IDENTITY_CHARS:
            raise ValueError("participant_identity must contain 1 to 128 characters")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if not 1 <= session_generation <= _MAX_SESSION_GENERATION:
            raise ValueError("session_generation is outside the supported range")
        if media_incarnation_provider is not None and not callable(
            media_incarnation_provider
        ):
            raise TypeError("media_incarnation_provider must be callable")
        if not callable(getattr(vad, "process", None)):
            raise TypeError("vad must provide process()")
        for method in ("push", "finish_utterance", "cancel"):
            if not callable(getattr(stt, method, None)):
                raise TypeError(f"stt must provide {method}()")
        if type(stt_streams_partials) is not bool:
            raise TypeError("stt_streams_partials must be an exact bool")
        if type(actions) is not ConversationUpdateExecutor:
            raise TypeError("actions must be an exact ConversationUpdateExecutor")
        if command_router is not None and not callable(getattr(command_router, "route", None)):
            raise TypeError("command_router must provide route()")
        if (
            evidence_lifecycle is not None
            and type(evidence_lifecycle) is not EvidenceConversationAuthorityV1
        ):
            raise TypeError("evidence_lifecycle must be exact or None")
        if evidence_lifecycle_resolver is not None and not callable(evidence_lifecycle_resolver):
            raise TypeError("evidence_lifecycle_resolver must be callable or None")
        if evidence_lifecycle is not None and evidence_lifecycle_resolver is not None:
            raise ValueError("evidence lifecycle and resolver are mutually exclusive")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        if (
            (evidence_lifecycle is not None or evidence_lifecycle_resolver is not None)
            and command_router is None
        ):
            raise TypeError("evidence lifecycle requires command_router")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable")
        if (
            knowledge_coordinator is not None
            and type(knowledge_coordinator) is not KnowledgePrefetchCoordinator
        ):
            raise TypeError("knowledge_coordinator must be exact")
        if (
            type(knowledge_budget_seconds) not in (int, float)
            or isinstance(knowledge_budget_seconds, bool)
            or not math.isfinite(float(knowledge_budget_seconds))
            or not 0.05 <= float(knowledge_budget_seconds) <= 30.0
        ):
            raise ValueError("knowledge_budget_seconds must be finite and between 0.05 and 30")
        if on_audio_ready is not None and not callable(on_audio_ready):
            raise TypeError("on_audio_ready must be callable")
        if type(audio_ready_repeat_frames) is not int:
            raise TypeError("audio_ready_repeat_frames must be an exact integer")
        if not 1 <= audio_ready_repeat_frames <= 1_000:
            raise ValueError("audio_ready_repeat_frames is outside the supported range")
        if on_voice_activity is not None and not callable(on_voice_activity):
            raise TypeError("on_voice_activity must be callable")
        if type(max_response_operations) is not int:
            raise TypeError("max_response_operations must be an exact integer")
        if not 1 <= max_response_operations <= _MAX_RESPONSE_OPERATIONS:
            raise ValueError("max_response_operations must be between 1 and 64")
        if type(cleanup_timeout_seconds) not in (int, float):
            raise TypeError("cleanup_timeout_seconds must be an exact number")
        if not math.isfinite(cleanup_timeout_seconds) or not 0 < cleanup_timeout_seconds <= 60:
            raise ValueError("cleanup_timeout_seconds must be between 0 and 60")
        if type(pre_roll_frames) is not int:
            raise TypeError("pre_roll_frames must be an exact integer")
        if not 0 <= pre_roll_frames <= 999:
            raise ValueError("pre_roll_frames must be between 0 and 999")
        if (
            type(max_buffered_audio_bytes) is not int
            or not 1 <= max_buffered_audio_bytes <= _MAX_BUFFERED_AUDIO_BYTES
        ):
            raise ValueError("max_buffered_audio_bytes is outside the supported range")
        if echo_guard is not None:
            for method in (
                "is_echo_dominated",
                "is_transcript_echo",
                "has_recent_playback",
            ):
                if not callable(getattr(echo_guard, method, None)):
                    raise TypeError(f"echo_guard must provide {method}()")
        if speech_presence_verifier is not None and not callable(
            getattr(speech_presence_verifier, "classify", None)
        ):
            raise TypeError("speech_presence_verifier must provide classify()")
        for name, value, ceiling in (
            ("echo_recheck_frames", echo_recheck_frames, 100),
            ("echo_promotion_checks", echo_promotion_checks, 10),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= ceiling:
                raise ValueError(f"{name} is outside the supported range")

        self._participant_identity = participant_identity
        self._session_generation = session_generation
        self._media_incarnation_provider = media_incarnation_provider or (lambda: 1)
        self._observed_media_incarnations: set[int] = set()
        self._vad = vad
        self._stt = stt
        self._stt_streams_partials = stt_streams_partials
        self._actions = actions
        self._command_router = command_router
        self._evidence_lifecycle = evidence_lifecycle
        self._evidence_lifecycle_resolver = evidence_lifecycle_resolver
        self._production_observation_recorder = production_observation_recorder
        self._typed_sequence = 0
        self._microphone_input_incarnation = 0
        self._observer = observer
        self._knowledge_coordinator = knowledge_coordinator
        self._knowledge_budget_seconds = float(knowledge_budget_seconds)
        self._utterance_sequence = 0
        self._active_utterance_ticket: UtteranceTicket | None = None
        self._on_audio_ready = on_audio_ready
        self._audio_ready_confirmed = False
        self._audio_ready_repeat_frames = audio_ready_repeat_frames
        self._audio_ready_countdown = 0
        self._on_voice_activity = on_voice_activity
        self._voice_activity_active = False
        self._max_response_operations = max_response_operations
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)
        self._turn_sequence = 0
        self._utterance_active = False
        self._utterance_has_final_transcript = False
        self._preliminary_final_transcript: Transcript | None = None
        self._pre_roll_frame_capacity = pre_roll_frames
        self._max_buffered_audio_bytes = max_buffered_audio_bytes
        self._pre_roll_frames: deque[AudioFrame] = deque()
        self._pre_roll_bytes = 0
        self._echo_guard = echo_guard
        self._speech_presence_verifier = speech_presence_verifier
        self._echo_recheck_frames = echo_recheck_frames
        self._echo_promotion_checks = echo_promotion_checks
        self._echo_suppressed = False
        self._echo_recheck_countdown = echo_recheck_frames
        self._echo_rechecks = 0
        self._echo_non_echo_checks = 0
        self._echo_candidate_frames: deque[AudioFrame] = deque()
        self._echo_candidate_bytes = 0
        self._interruption_confirmation_pending = False
        self._interruption_committed = False
        self._playback_candidate_requires_streaming_evidence = False
        self._playback_candidate_streaming_evidence = False
        self._playback_candidate_authority_revoked = False
        self._interruption_foreground_turn_id: str | None = None
        self._response_operations: set[asyncio.Task[None]] = set()
        self._foreground_cleanup_task: asyncio.Task[None] | None = None
        self._stt_cleanup_task: asyncio.Task[None] | None = None
        self._response_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._binding_cleanup_timed_out = False
        self._binding_cleanup_needs_retry = False
        self._audio_lock = asyncio.Lock()
        self._terminal_error: BaseException | None = None
        self._binding_close_operation: asyncio.Task[None] | None = None
        self._close_operation: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False

    @property
    def participant_identity(self) -> str:
        return self._participant_identity

    @property
    def session_generation(self) -> int:
        return self._session_generation

    @property
    def active_response_count(self) -> int:
        return len(self._response_operations)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("conversation session worker is closed")
        self._raise_terminal_error()
        if self._started:
            return
        self._actions.start()
        self._started = True

    async def receive_audio(
        self,
        participant_identity: str,
        session_generation: int,
        frame: AudioFrame,
    ) -> None:
        """Admit one generation-bound PCM frame and route exact final transcripts."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if type(frame) is not AudioFrame:
            raise TypeError("frame must be an exact AudioFrame")
        frame = AudioFrame(
            pcm=frame.pcm,
            sample_rate_hz=frame.sample_rate_hz,
            channels=frame.channels,
        )
        if participant_identity != self._participant_identity:
            raise PermissionError("audio participant does not own this worker")
        if session_generation != self._session_generation:
            raise RuntimeError("audio belongs to a stale session generation")
        if not self._started:
            raise RuntimeError("conversation session worker is not started")

        async with self._audio_lock:
            if self._closed:
                raise RuntimeError("conversation session worker is closed")
            self._raise_terminal_error()
            try:
                self._audio_ready_countdown -= 1
                if not self._audio_ready_confirmed or self._audio_ready_countdown <= 0:
                    self._audio_ready_confirmed = True
                    self._audio_ready_countdown = self._audio_ready_repeat_frames
                    on_audio_ready = self._on_audio_ready
                    if on_audio_ready is not None:
                        self._notify_advisory(on_audio_ready)
                raw_activity = self._vad.process(frame)
                if type(raw_activity) is not VoiceActivity:
                    raise TypeError("vad must return an exact VoiceActivity")
                if type(raw_activity.value) is not str:
                    raise TypeError("voice activity value must be an exact string")
                activity = VoiceActivity(raw_activity.value)

                frames_to_push: tuple[AudioFrame, ...] = ()
                utterance_ended = False
                if activity is VoiceActivity.SILENCE:
                    if self._utterance_active:
                        raise RuntimeError("vad emitted silence before ending the active utterance")
                    if self._echo_suppressed:
                        self._reset_echo_suppression()
                    self._append_pre_roll(frame)
                elif activity is VoiceActivity.SPEECH_STARTED:
                    if self._utterance_active or self._echo_suppressed:
                        raise RuntimeError("vad started an already active utterance")
                    interrupting = self._actions.foreground_active
                    self._interruption_foreground_turn_id = (
                        self._actions.foreground_turn_id if interrupting else None
                    )
                    self._interruption_committed = False
                    self._utterance_has_final_transcript = False
                    self._preliminary_final_transcript = None
                    self._playback_candidate_authority_revoked = False
                    guard = self._echo_guard
                    start_frames = self._bounded_recent_frames((*self._pre_roll_frames, frame))
                    echo_dominated = guard is not None and guard.is_echo_dominated(start_frames)
                    recent_playback = (
                        guard is not None and guard.has_recent_playback()
                    )
                    observer = self._observer
                    confirmation_required = guard is not None and (
                        interrupting or recent_playback
                    )
                    self._playback_candidate_requires_streaming_evidence = (
                        confirmation_required and self._stt_streams_partials
                    )
                    self._playback_candidate_streaming_evidence = False
                    if echo_dominated or confirmation_required:
                        self._echo_suppressed = True
                        self._echo_recheck_countdown = self._echo_recheck_frames
                        self._echo_rechecks = 0
                        self._echo_non_echo_checks = 0 if echo_dominated else 1
                        self._set_echo_candidate(start_frames)
                        self._clear_pre_roll()
                        if observer is not None:
                            self._notify_advisory(observer, "echo_suppressed", {})
                    else:
                        cancelled = False
                        if interrupting:
                            turn_id = self._interruption_foreground_turn_id
                            assert turn_id is not None
                            cancelled = await self._cancel_foreground_for_echo_promotion(
                                turn_id
                            )
                            if not cancelled:
                                self._playback_candidate_authority_revoked = True
                        if cancelled and observer is not None:
                            observer("interrupt_requested", {"turnId": turn_id})
                            observer("playback_silenced", {})
                        self._utterance_active = True
                        self._set_voice_activity(True)
                        frames_to_push = start_frames
                        self._clear_pre_roll()
                elif activity is VoiceActivity.SPEECH_CONTINUED:
                    if self._echo_suppressed:
                        self._append_echo_candidate(frame)
                        self._echo_recheck_countdown -= 1
                        if self._echo_recheck_countdown == 0:
                            self._echo_recheck_countdown = self._echo_recheck_frames
                            self._echo_rechecks += 1
                            guard = self._echo_guard
                            echo_dominated = guard is not None and guard.is_echo_dominated(
                                tuple(self._echo_candidate_frames)
                            )
                            if echo_dominated:
                                self._echo_non_echo_checks = 0
                            else:
                                self._echo_non_echo_checks += 1
                            if self._echo_non_echo_checks >= self._echo_promotion_checks:
                                retained = tuple(self._echo_candidate_frames)
                                speech_admitted = await self._speech_candidate_admitted(retained)
                                self._raise_if_closed()
                                if speech_admitted:
                                    self._utterance_active = True
                                    self._set_voice_activity(True)
                                    self._echo_suppressed = False
                                    self._interruption_confirmation_pending = (
                                        self._interruption_foreground_turn_id is not None
                                    )
                                    frames_to_push = retained
                                    self._clear_echo_candidate()
                                self._echo_non_echo_checks = 0
                    elif not self._utterance_active:
                        raise RuntimeError("vad continued without an active utterance")
                    else:
                        frames_to_push = (frame,)
                elif activity is VoiceActivity.SPEECH_ENDED:
                    if self._echo_suppressed:
                        self._append_echo_candidate(frame)
                        retained = tuple(self._echo_candidate_frames)
                        guard = self._echo_guard
                        echo_dominated = guard is not None and guard.is_echo_dominated(retained)
                        speech_admitted = (
                            not echo_dominated
                            and await self._speech_candidate_admitted(retained)
                        )
                        self._raise_if_closed()
                        if speech_admitted:
                            self._interruption_confirmation_pending = (
                                self._interruption_foreground_turn_id is not None
                            )
                            self._set_voice_activity(True)
                            self._set_voice_activity(False)
                            frames_to_push = retained
                            utterance_ended = True
                        self._reset_echo_suppression()
                        if not speech_admitted:
                            self._clear_playback_candidate_authority()
                    elif not self._utterance_active:
                        raise RuntimeError("vad ended without an active utterance")
                    else:
                        frames_to_push = (frame,)
                        self._utterance_active = False
                        self._set_voice_activity(False)
                        utterance_ended = True

                transcripts: tuple[Transcript, ...] = ()
                for utterance_frame in frames_to_push:
                    raw_transcripts = self._without_transcript_echoes(
                        self._trusted_transcripts(await self._stt.push(utterance_frame))
                    )
                    self._raise_if_closed()
                    frame_finals = tuple(
                        transcript for transcript in raw_transcripts if transcript.final
                    )
                    if frame_finals:
                        self._preliminary_final_transcript = frame_finals[-1]
                    frame_transcripts = self._with_playback_authority(
                        tuple(transcript for transcript in raw_transcripts if not transcript.final)
                    )
                    for transcript in frame_transcripts:
                        await self._confirm_interruption_from_transcript(transcript)
                    if self._playback_candidate_authority_revoked:
                        frame_transcripts = ()
                        await self._revoke_active_ticket("playback_authority_revoked")
                    transcripts = (*transcripts, *frame_transcripts)
                if utterance_ended:
                    observer = self._observer
                    if observer is not None:
                        observer("speech_ended", {})
                    final_transcript = await self._stt.finish_utterance()
                    if final_transcript is None:
                        final_transcript = self._preliminary_final_transcript
                    self._preliminary_final_transcript = None
                    candidate_final_transcript = (
                        self._trusted_transcript(final_transcript)
                        if final_transcript is not None
                        else None
                    )
                    trusted_final_transcript = (
                        None
                        if candidate_final_transcript is not None
                        and self._is_transcript_echo(candidate_final_transcript)
                        else candidate_final_transcript
                    )
                    if trusted_final_transcript is not None:
                        admitted_final = self._with_playback_authority(
                            (trusted_final_transcript,)
                        )
                        trusted_final_transcript = (
                            admitted_final[0] if admitted_final else None
                        )
                    if trusted_final_transcript is not None:
                        transcripts = (
                            *transcripts,
                            trusted_final_transcript,
                        )
                    self._raise_if_closed()
                    if trusted_final_transcript is not None:
                        await self._confirm_interruption_from_transcript(
                            trusted_final_transcript
                        )
                        self._utterance_has_final_transcript = True
                    if (
                        self._interruption_committed
                        and not self._utterance_has_final_transcript
                    ):
                        await self._actions.resume_foreground()
                        self._raise_if_closed()
                    if not self._utterance_has_final_transcript:
                        await self._revoke_active_ticket("utterance_abandoned")
                    self._utterance_has_final_transcript = False
                    self._interruption_committed = False
                    self._interruption_confirmation_pending = False
                    self._clear_playback_candidate_authority()
                for transcript in transcripts:
                    if transcript.final:
                        self._observe_final_transcript(transcript)
                        router = self._command_router
                        lifecycle = self._resolve_evidence_lifecycle()
                        user_authority: UserTurnAuthorityV1 | None = None
                        if router is not None and lifecycle is not None:
                            self._microphone_input_incarnation += 1
                            media_incarnation = self._media_incarnation_provider()
                            if type(media_incarnation) is not int or media_incarnation < 1:
                                raise RuntimeError(
                                    "microphone final has no active media incarnation"
                                )
                            authority = lifecycle.mint_final_input(
                                source=InputSource.MICROPHONE,
                                input_incarnation=self._microphone_input_incarnation,
                                media_incarnation=media_incarnation,
                                typed_sequence=None,
                            )
                            routing = await router.route(transcript.text, authority)
                            if type(routing) is not CommandRoutingResultV1:
                                raise TypeError("command router returned the wrong result")
                            handled = routing.outcome is not CommandRoutingOutcome.NOT_COMMAND
                            user_authority = routing.user_turn_authority
                        else:
                            legacy_router = cast(Any, router)
                            handled = router is not None and bool(
                                await legacy_router.route(transcript.text)
                            )
                        self._raise_if_closed()
                        if handled:
                            await self._revoke_active_ticket("command_routed")
                        else:
                            self._start_response(transcript, user_authority)
                    else:
                        await self._observe_partial_transcript(transcript)
            except BaseException as error:
                current = asyncio.current_task()
                caller_cancelled = (
                    isinstance(error, asyncio.CancelledError)
                    and current is not None
                    and current.cancelling() > 0
                )
                if not self._closed and not caller_cancelled and self._terminal_error is None:
                    self._terminal_error = error
                raise

    async def _speech_candidate_admitted(self, frames: tuple[AudioFrame, ...]) -> bool:
        verifier = self._speech_presence_verifier
        if verifier is None:
            return True
        try:
            presence = await asyncio.to_thread(verifier.classify, frames)
        except Exception:
            self._raise_if_closed()
            _LOGGER.exception("speech presence verifier failed")
            observer = self._observer
            if observer is not None:
                self._notify_advisory(
                    observer,
                    "barge_in_verifier_unavailable",
                    {"reason": "classification_error"},
                )
            return False
        self._raise_if_closed()
        if type(presence) is not SpeechPresence:
            _LOGGER.error("speech presence verifier returned an invalid decision type")
            observer = self._observer
            if observer is not None:
                self._notify_advisory(
                    observer,
                    "barge_in_verifier_unavailable",
                    {"reason": "invalid_decision"},
                )
            return False
        if presence is not SpeechPresence.CONFIRMED_SPEECH:
            observer = self._observer
            if observer is not None:
                self._notify_advisory(
                    observer,
                    "barge_in_non_speech_suppressed",
                    {"reason": presence.value},
                )
            return False
        return True

    def _with_playback_authority(
        self,
        transcripts: tuple[Transcript, ...],
    ) -> tuple[Transcript, ...]:
        if self._playback_candidate_authority_revoked:
            return ()
        if not self._playback_candidate_requires_streaming_evidence:
            return transcripts
        admitted: list[Transcript] = []
        for transcript in transcripts:
            if not transcript.final:
                if sum(character.isalnum() for character in transcript.text) >= 3:
                    self._playback_candidate_streaming_evidence = True
                admitted.append(transcript)
            elif self._playback_candidate_streaming_evidence:
                admitted.append(transcript)
        return tuple(admitted)

    def _clear_playback_candidate_authority(self) -> None:
        self._playback_candidate_requires_streaming_evidence = False
        self._playback_candidate_streaming_evidence = False
        self._playback_candidate_authority_revoked = False
        self._interruption_foreground_turn_id = None

    async def _cancel_foreground_for_echo_promotion(self, turn_id: str) -> bool:
        try:
            cancelled = await self._actions.cancel_foreground_if(turn_id)
            self._raise_if_closed()
            self._interruption_committed = cancelled
            return cancelled
        except BaseException:
            self._utterance_active = False
            self._interruption_confirmation_pending = False
            self._interruption_committed = False
            self._set_voice_activity(False)
            self._reset_echo_suppression()
            raise

    async def _confirm_interruption_from_transcript(self, transcript: Transcript) -> None:
        if not self._interruption_confirmation_pending:
            return
        if not transcript.final and sum(character.isalnum() for character in transcript.text) < 3:
            return
        self._interruption_confirmation_pending = False
        turn_id = self._interruption_foreground_turn_id
        if turn_id is None:
            return
        cancelled = await self._cancel_foreground_for_echo_promotion(turn_id)
        if not cancelled:
            self._playback_candidate_authority_revoked = True
            return
        observer = self._observer
        if observer is not None:
            observer("interrupt_requested", {"turnId": turn_id})
            observer("playback_silenced", {})
            self._notify_advisory(observer, "echo_barge_in_confirmed", {})

    def _without_transcript_echoes(
        self,
        transcripts: tuple[Transcript, ...],
    ) -> tuple[Transcript, ...]:
        return tuple(
            transcript
            for transcript in transcripts
            if not self._is_transcript_echo(transcript)
        )

    def _is_transcript_echo(self, transcript: Transcript) -> bool:
        guard = self._echo_guard
        if guard is None or not guard.is_transcript_echo(transcript.text):
            return False
        observer = self._observer
        if observer is not None:
            self._notify_advisory(observer, "transcript_echo_suppressed", {})
        return True

    def _set_voice_activity(self, active: bool) -> None:
        if self._voice_activity_active is active:
            return
        self._voice_activity_active = active
        callback = self._on_voice_activity
        if callback is not None:
            self._notify_advisory(callback, active)

    @staticmethod
    def _notify_advisory(callback: Callable[..., None], *args: object) -> None:
        with contextlib.suppress(Exception):
            callback(*args)

    def _append_pre_roll(self, frame: AudioFrame) -> None:
        if self._pre_roll_frame_capacity == 0:
            return
        frame_bytes = len(frame.pcm)
        if frame_bytes > self._max_buffered_audio_bytes:
            raise ValueError("audio frame exceeds buffered audio capacity")
        while self._pre_roll_frames and (
            len(self._pre_roll_frames) >= self._pre_roll_frame_capacity
            or self._pre_roll_bytes + frame_bytes > self._max_buffered_audio_bytes
        ):
            self._pre_roll_bytes -= len(self._pre_roll_frames.popleft().pcm)
        self._pre_roll_frames.append(frame)
        self._pre_roll_bytes += frame_bytes

    def _bounded_recent_frames(
        self,
        frames: tuple[AudioFrame, ...],
    ) -> tuple[AudioFrame, ...]:
        retained: deque[AudioFrame] = deque()
        retained_bytes = 0
        for frame in reversed(frames):
            frame_bytes = len(frame.pcm)
            if frame_bytes > self._max_buffered_audio_bytes:
                raise ValueError("audio frame exceeds buffered audio capacity")
            if retained_bytes + frame_bytes > self._max_buffered_audio_bytes:
                break
            retained.appendleft(frame)
            retained_bytes += frame_bytes
        return tuple(retained)

    def _clear_pre_roll(self) -> None:
        self._pre_roll_frames.clear()
        self._pre_roll_bytes = 0

    def _set_echo_candidate(self, frames: tuple[AudioFrame, ...]) -> None:
        self._clear_echo_candidate()
        for frame in frames:
            self._append_echo_candidate(frame)

    def _append_echo_candidate(self, frame: AudioFrame) -> None:
        frame_bytes = len(frame.pcm)
        if frame_bytes > self._max_buffered_audio_bytes:
            raise ValueError("audio frame exceeds buffered audio capacity")
        while self._echo_candidate_frames and (
            self._echo_candidate_bytes + frame_bytes > self._max_buffered_audio_bytes
        ):
            self._echo_candidate_bytes -= len(self._echo_candidate_frames.popleft().pcm)
        self._echo_candidate_frames.append(frame)
        self._echo_candidate_bytes += frame_bytes

    def _clear_echo_candidate(self) -> None:
        self._echo_candidate_frames.clear()
        self._echo_candidate_bytes = 0

    def _reset_echo_suppression(self) -> None:
        self._echo_suppressed = False
        self._echo_recheck_countdown = self._echo_recheck_frames
        self._echo_rechecks = 0
        self._echo_non_echo_checks = 0
        self._clear_echo_candidate()

    async def submit_final_transcript(
        self,
        *,
        participant_identity: str,
        session_generation: int,
        typed_sequence: int,
        text: str,
    ) -> None:
        """Admit one bounded typed user turn through the foreground response path."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if type(typed_sequence) is not int:
            raise TypeError("typed_sequence must be an exact integer")
        if not 1 <= typed_sequence <= (1 << 63) - 1:
            raise ValueError("typed_sequence is outside the supported range")
        if type(text) is not str:
            raise TypeError("text must be an exact built-in string")
        if not text.strip() or len(text) > _MAX_TYPED_TRANSCRIPT_CHARS:
            raise ValueError("text must contain 1 to 4096 characters")
        transcript = Transcript(text=text, final=True)
        if participant_identity != self._participant_identity:
            raise PermissionError("typed participant does not own this worker")
        if session_generation != self._session_generation:
            raise RuntimeError("typed input belongs to a stale session generation")
        if not self._started:
            raise RuntimeError("conversation session worker is not started")

        async with self._audio_lock:
            if self._closed:
                raise RuntimeError("conversation session worker is closed")
            self._raise_terminal_error()
            try:
                self._observe_final_transcript(transcript)
                await self._actions.cancel_foreground()
                self._raise_if_closed()
                router = self._command_router
                lifecycle = self._resolve_evidence_lifecycle()
                user_authority: UserTurnAuthorityV1 | None = None
                if router is not None and lifecycle is not None:
                    if typed_sequence <= self._typed_sequence:
                        raise RuntimeError("typed input sequence is stale or duplicated")
                    self._typed_sequence = typed_sequence
                    authority = lifecycle.mint_final_input(
                        source=InputSource.TYPED,
                        input_incarnation=typed_sequence,
                        media_incarnation=None,
                        typed_sequence=typed_sequence,
                    )
                    routing = await router.route(transcript.text, authority)
                    if type(routing) is not CommandRoutingResultV1:
                        raise TypeError("command router returned the wrong result")
                    handled = routing.outcome is not CommandRoutingOutcome.NOT_COMMAND
                    user_authority = routing.user_turn_authority
                else:
                    legacy_router = cast(Any, router)
                    handled = router is not None and bool(
                        await legacy_router.route(transcript.text)
                    )
                self._raise_if_closed()
                if handled:
                    await self._revoke_active_ticket("command_routed")
                else:
                    self._start_response(transcript, user_authority)
            except BaseException as error:
                current = asyncio.current_task()
                caller_cancelled = (
                    isinstance(error, asyncio.CancelledError)
                    and current is not None
                    and current.cancelling() > 0
                )
                if not self._closed and not caller_cancelled and self._terminal_error is None:
                    self._terminal_error = error
                raise

    async def wait_for_responses(self) -> None:
        """Wait until every currently accepted foreground response settles."""

        while self._response_operations:
            operations = tuple(self._response_operations)
            await asyncio.gather(*operations, return_exceptions=True)
        self._raise_terminal_error()

    async def close(self) -> None:
        """Stop this binding and close the shared action authority."""

        operation = self._close_operation
        if operation is None or _operation_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name=f"conversation-session-close:{self._session_generation}",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    async def close_binding(self) -> None:
        """Stop only this media/STT generation for an authenticated reconnect."""

        await asyncio.shield(self._ensure_binding_close())

    async def _close_owned(self) -> None:
        errors: list[BaseException] = []
        recorder = self._production_observation_recorder
        try:
            await asyncio.shield(self._ensure_binding_close())
        except BaseException as error:
            errors.append(error)
        try:
            await self._actions.close()
        except BaseException as error:
            errors.append(error)
        if len(errors) == 1:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.SESSION_WORKER,
                    result=CloseResultV1.FAILED,
                )
            raise errors[0]
        if errors:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.SESSION_WORKER,
                    result=CloseResultV1.FAILED,
                )
            raise BaseExceptionGroup("conversation session close failed", errors)
        if recorder is not None:
            recorder.record_close_stage(
                stage=CloseStageV1.SESSION_WORKER,
                result=CloseResultV1.SUCCEEDED,
            )

    def _ensure_binding_close(self) -> asyncio.Task[None]:
        operation = self._binding_close_operation
        if operation is None or (
            operation.done()
            and (self._binding_cleanup_timed_out or self._binding_cleanup_needs_retry)
        ):
            operation = asyncio.create_task(
                self._close_binding_owned(),
                name=f"conversation-session-binding-close:{self._session_generation}",
            )
            self._binding_close_operation = operation
        return operation

    async def _close_binding_owned(self) -> None:
        recorder = self._production_observation_recorder
        coordinator = self._knowledge_coordinator
        if coordinator is not None:
            incarnations = self._observed_media_incarnations or {1}
            for media_incarnation in tuple(incarnations):
                await coordinator.close_binding(
                    self._session_generation,
                    media_incarnation,
                )
        self._active_utterance_ticket = None
        if not self._closed:
            self._closed = True
            for operation in tuple(self._response_operations):
                operation.cancel()
                self._response_cleanup_tasks.add(operation)

        foreground_cleanup = self._foreground_cleanup_task
        if foreground_cleanup is None or _operation_failed(foreground_cleanup):
            foreground_cleanup = asyncio.create_task(
                self._actions._cancel_foreground_for_cleanup(binding_closed=True),
                name=(f"conversation-session-foreground-close:{self._session_generation}"),
            )
            foreground_cleanup.add_done_callback(self._consume_cleanup_result)
            self._foreground_cleanup_task = foreground_cleanup

        stt_cleanup = self._stt_cleanup_task
        if stt_cleanup is None or _operation_failed(stt_cleanup):
            stt_cleanup = asyncio.create_task(
                self._stt.cancel(),
                name=f"conversation-session-stt-close:{self._session_generation}",
            )
            stt_cleanup.add_done_callback(self._consume_cleanup_result)
            self._stt_cleanup_task = stt_cleanup

        active_cleanup = {
            foreground_cleanup,
            stt_cleanup,
            *self._response_cleanup_tasks,
        }
        done, pending = await asyncio.wait(
            active_cleanup,
            timeout=self._cleanup_timeout_seconds,
        )
        if pending:
            self._binding_cleanup_timed_out = True
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.BINDING_CLEANUP,
                    result=CloseResultV1.TIMED_OUT,
                )
            raise TimeoutError(
                f"binding cleanup exceeded {self._cleanup_timeout_seconds:g} seconds"
            )

        self._binding_cleanup_timed_out = False
        errors: list[BaseException] = []
        for operation in done:
            if operation.cancelled():
                continue
            error = operation.exception()
            if error is not None:
                errors.append(error)
        self._binding_cleanup_needs_retry = _operation_failed(
            foreground_cleanup
        ) or _operation_failed(stt_cleanup)
        if self._terminal_error is not None:
            errors.append(self._terminal_error)
        if len(errors) == 1:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.BINDING_CLEANUP,
                    result=CloseResultV1.FAILED,
                )
            raise errors[0]
        if errors:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.BINDING_CLEANUP,
                    result=CloseResultV1.FAILED,
                )
            raise BaseExceptionGroup("conversation session worker close failed", errors)
        if recorder is not None:
            recorder.record_close_stage(
                stage=CloseStageV1.BINDING_CLEANUP,
                result=CloseResultV1.SUCCEEDED,
            )

    @staticmethod
    def _consume_cleanup_result(operation: asyncio.Task[None]) -> None:
        if operation.cancelled():
            return
        with contextlib.suppress(BaseException):
            operation.exception()

    def _observe_final_transcript(self, transcript: Transcript) -> None:
        observer = self._observer
        if observer is not None:
            observer(
                "transcript_final",
                {"role": "user", "text": transcript.text},
            )

    async def _observe_partial_transcript(self, transcript: Transcript) -> None:
        coordinator = self._knowledge_coordinator
        if coordinator is not None:
            await coordinator.observe_partial(self._active_ticket(), transcript.text)
        observer = self._observer
        if observer is not None:
            observer(
                "transcript_partial",
                {"role": "user", "text": transcript.text},
            )

    def _active_ticket(self) -> UtteranceTicket:
        ticket = self._active_utterance_ticket
        if ticket is None:
            media_incarnation = self._media_incarnation_provider()
            if type(media_incarnation) is not int or media_incarnation <= 0:
                raise RuntimeError("authoritative media incarnation is unavailable")
            self._observed_media_incarnations.add(media_incarnation)
            self._utterance_sequence += 1
            ticket = UtteranceTicket(
                session_generation=self._session_generation,
                media_incarnation=media_incarnation,
                utterance_sequence=self._utterance_sequence,
            )
            self._active_utterance_ticket = ticket
        return ticket

    async def _revoke_active_ticket(self, reason: str) -> None:
        ticket = self._active_utterance_ticket
        self._active_utterance_ticket = None
        coordinator = self._knowledge_coordinator
        if ticket is not None and coordinator is not None:
            await coordinator.revoke(ticket, reason)

    def _start_response(
        self,
        transcript: Transcript,
        authority: UserTurnAuthorityV1 | None = None,
    ) -> None:
        if len(self._response_operations) >= self._max_response_operations:
            raise RuntimeError("conversation response operation capacity exhausted")
        reservation = self._actions.reserve_response()
        response: Any | None = None
        try:
            self._turn_sequence += 1
            turn_id = f"session_{self._session_generation}_turn_{self._turn_sequence}"
            coordinator = self._knowledge_coordinator
            if coordinator is not None:
                ticket = self._active_ticket()
                coordinator.admit_final(
                    turn_id,
                    ticket,
                    transcript.text,
                    KnowledgeTurnBudget.start(
                        ticket=ticket,
                        total_seconds=self._knowledge_budget_seconds,
                    ),
                )
            self._active_utterance_ticket = None
            response = self._actions.respond(
                turn_id,
                transcript,
                authority,
                reservation=reservation,
            )
            operation = asyncio.create_task(
                response,
                name=f"conversation-session-response:{turn_id}",
            )
        except BaseException:
            if response is not None:
                response.close()
            self._actions.release_response(reservation)
            raise
        self._response_operations.add(operation)
        operation.add_done_callback(self._response_done)

    def _resolve_evidence_lifecycle(self) -> EvidenceConversationAuthorityV1 | None:
        resolver = self._evidence_lifecycle_resolver
        lifecycle = resolver() if resolver is not None else self._evidence_lifecycle
        if lifecycle is not None and type(lifecycle) is not EvidenceConversationAuthorityV1:
            raise TypeError("evidence lifecycle resolver must return an exact authority or None")
        return lifecycle

    def _response_done(self, operation: asyncio.Task[None]) -> None:
        self._response_operations.discard(operation)
        if operation.cancelled():
            return
        error = operation.exception()
        if error is not None and self._terminal_error is None:
            self._terminal_error = error

    @staticmethod
    def _trusted_transcripts(value: object) -> tuple[Transcript, ...]:
        if type(value) is not tuple:
            raise TypeError("stt push must return an exact tuple")
        return tuple(ConversationSessionWorker._trusted_transcript(item) for item in value)

    @staticmethod
    def _trusted_transcript(value: object) -> Transcript:
        if type(value) is not Transcript:
            raise TypeError("stt must return exact Transcript values")
        return Transcript(text=value.text, final=value.final)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("conversation session worker is closed")

    def _raise_terminal_error(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error


class ReconnectSafeConversationWorker:
    """Replace media/STT bindings while preserving one action authority."""

    def __init__(
        self,
        *,
        actions: ConversationUpdateExecutor,
        binding_factory: Callable[
            [str, int, ConversationUpdateExecutor],
            ConversationSessionWorker,
        ],
        require_media_activation: bool = False,
    ) -> None:
        if type(actions) is not ConversationUpdateExecutor:
            raise TypeError("actions must be an exact ConversationUpdateExecutor")
        if not callable(binding_factory):
            raise TypeError("binding_factory must be callable")
        if type(require_media_activation) is not bool:
            raise TypeError("require_media_activation must be an exact boolean")
        self._actions = actions
        self._binding_factory = binding_factory
        self._require_media_activation = require_media_activation
        self._media_incarnation: int | None = None
        self._media_track_name: str | None = None
        self._binding: ConversationSessionWorker | None = None
        self._generation = 0
        self._audio_input_suppression: tuple[int, str, object] | None = None
        self._binding_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def active_generation(self) -> int | None:
        binding = self._binding
        return None if binding is None else binding.session_generation

    async def bind(self, participant_identity: str) -> int:
        """Close the previous binding before admitting a fresh generation."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        async with self._binding_lock:
            if self._closed:
                raise RuntimeError("reconnect-safe conversation worker is closed")
            previous = self._binding
            if previous is not None:
                await previous.close_binding()
                if self._binding is previous:
                    self._binding = None
            if self._generation >= _MAX_SESSION_GENERATION:
                raise RuntimeError("session generation capacity exhausted")
            self._generation += 1
            binding = self._binding_factory(
                participant_identity,
                self._generation,
                self._actions,
            )
            if type(binding) is not ConversationSessionWorker:
                raise TypeError("binding_factory must return an exact session worker")
            if binding._actions is not self._actions:
                raise PermissionError("binding_factory replaced the action authority")
            if binding.session_generation != self._generation:
                raise ValueError("binding_factory returned a contradictory generation")
            if binding.participant_identity != participant_identity:
                raise PermissionError("binding_factory replaced the participant identity")
            binding.start()
            self._binding = binding
            self._media_incarnation = None
            self._media_track_name = None
            self._audio_input_suppression = None
            return self._generation

    async def close_binding(self) -> None:
        """Settle only the current media/STT generation before transport replacement."""

        async with self._binding_lock:
            if self._closed:
                raise RuntimeError("reconnect-safe conversation worker is closed")
            binding = self._binding
            if binding is None:
                return
            await binding.close_binding()
            if self._binding is binding:
                self._binding = None
                self._media_incarnation = None
                self._media_track_name = None
                self._audio_input_suppression = None

    async def activate_media(
        self,
        *,
        participant_identity: str,
        session_generation: int,
        media_incarnation: int,
    ) -> None:
        """Authorize one exact LiveKit microphone publication for PCM admission."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(session_generation) is not int or type(media_incarnation) is not int:
            raise TypeError("media authority values must be exact integers")
        if not 1 <= media_incarnation <= (1 << 53) - 1:
            raise ValueError("media_incarnation is outside the browser-safe range")
        async with self._binding_lock:
            binding = self._binding
            if (
                binding is None
                or session_generation != self._generation
                or participant_identity != binding.participant_identity
            ):
                raise PermissionError("media activation does not own the active binding")
            if self._media_incarnation is not None:
                if media_incarnation < self._media_incarnation:
                    raise ValueError("media_incarnation must increase monotonically")
                if media_incarnation == self._media_incarnation:
                    return
            self._media_incarnation = media_incarnation
            self._media_track_name = f"microphone-{media_incarnation}"

    async def suppress_audio_input(
        self,
        *,
        participant_identity: str,
        session_generation: int,
    ) -> object:
        """Discard exact-session PCM until the returned authority token resumes it."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        async with self._binding_lock:
            binding = self._binding
            if (
                self._closed
                or binding is None
                or session_generation != self._generation
                or participant_identity != binding.participant_identity
            ):
                raise PermissionError("audio suppression does not own the active binding")
            if self._audio_input_suppression is not None:
                raise RuntimeError("audio input is already suppressed")
            token = object()
            self._audio_input_suppression = (
                session_generation,
                participant_identity,
                token,
            )
            return token

    async def resume_audio_input(self, token: object) -> bool:
        """Resume PCM only for the exact current suppression token."""

        async with self._binding_lock:
            suppression = self._audio_input_suppression
            if suppression is None or suppression[2] is not token:
                return False
            self._audio_input_suppression = None
            return True

    async def receive_audio(
        self,
        participant_identity: str,
        session_generation: int,
        frame: AudioFrame,
    ) -> None:
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        async with self._binding_lock:
            if self._closed:
                raise RuntimeError("reconnect-safe conversation worker is closed")
            binding = self._binding
            if binding is None or session_generation != self._generation:
                raise RuntimeError("audio belongs to a stale session generation")
            suppression = self._audio_input_suppression
            if (
                suppression is not None
                and suppression[0] == session_generation
                and suppression[1] == participant_identity
            ):
                return
        await binding.receive_audio(
            participant_identity,
            session_generation,
            frame,
        )

    async def submit_final_transcript(
        self,
        *,
        participant_identity: str,
        session_generation: int,
        typed_sequence: int,
        text: str,
    ) -> None:
        """Route one identity- and generation-bound typed user turn."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if type(typed_sequence) is not int:
            raise TypeError("typed_sequence must be an exact integer")
        if type(text) is not str:
            raise TypeError("text must be an exact built-in string")
        async with self._binding_lock:
            if self._closed:
                raise RuntimeError("reconnect-safe conversation worker is closed")
            binding = self._binding
            if binding is None or session_generation != self._generation:
                raise RuntimeError("typed input belongs to a stale session generation")
        await binding.submit_final_transcript(
            participant_identity=participant_identity,
            session_generation=session_generation,
            typed_sequence=typed_sequence,
            text=text,
        )

    async def run_source(
        self,
        source: _ParticipantAudioSource,
        session_generation: int,
        *,
        receive_timeout_seconds: float = 1.0,
    ) -> None:
        """Pump identity-bound transport PCM until close or binding replacement."""

        if not callable(getattr(source, "receive_participant_audio", None)):
            raise TypeError("source must provide receive_participant_audio()")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if type(receive_timeout_seconds) not in (int, float):
            raise TypeError("receive_timeout_seconds must be an exact number")
        if not math.isfinite(receive_timeout_seconds) or not (0 < receive_timeout_seconds <= 60):
            raise ValueError("receive_timeout_seconds must be between 0 and 60")
        async with self._binding_lock:
            if self._closed or session_generation != self._generation:
                return
            binding = self._binding
            if binding is None:
                return
        ingress = BoundedPcmIngress()
        consumer = asyncio.create_task(
            self._consume_source_ingress(binding, ingress),
            name=f"conversation-pcm-consumer:{session_generation}",
        )
        receive_operation: asyncio.Task[ParticipantAudioFrame] | None = None
        sequence = 0
        try:
            while True:
                ingress.check_health()
                async with self._binding_lock:
                    if (
                        self._closed
                        or session_generation != self._generation
                        or self._binding is not binding
                    ):
                        return
                receive_operation = asyncio.create_task(
                    source.receive_participant_audio(
                        timeout_seconds=min(
                            receive_timeout_seconds,
                            ingress.max_age_seconds,
                        )
                    ),
                    name=f"conversation-pcm-receive:{session_generation}",
                )
                done, _ = await asyncio.wait(
                    (receive_operation, consumer),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if consumer in done:
                    receive_operation.cancel()
                    await asyncio.gather(receive_operation, return_exceptions=True)
                    consumer_error: RuntimeError | None = None
                    try:
                        await consumer
                    except RuntimeError as error:
                        consumer_error = error
                    async with self._binding_lock:
                        if (
                            self._closed
                            or session_generation != self._generation
                            or self._binding is not binding
                        ):
                            return
                    if consumer_error is not None:
                        raise consumer_error
                    raise RuntimeError("PCM ingress consumer exited unexpectedly")
                try:
                    packet = receive_operation.result()
                except TimeoutError:
                    ingress.check_health()
                    continue
                finally:
                    receive_operation = None
                if type(packet) is not ParticipantAudioFrame:
                    raise TypeError("source must return an exact ParticipantAudioFrame")
                observed_at = time.monotonic()
                try:
                    async with self._binding_lock:
                        if (
                            self._closed
                            or session_generation != self._generation
                            or self._binding is not binding
                        ):
                            return
                        if (
                            self._require_media_activation
                            and packet.track_name != self._media_track_name
                        ):
                            continue
                        suppression = self._audio_input_suppression
                        if (
                            suppression is not None
                            and suppression[0] == session_generation
                            and suppression[1] == packet.participant_identity
                        ):
                            continue
                        sequence += 1
                        ingress.admit(
                            IngressRecord(
                                sequence=sequence,
                                participant_identity=packet.participant_identity,
                                session_generation=session_generation,
                                track_name=packet.track_name,
                                frame=packet.frame,
                                observed_at=observed_at,
                            )
                        )
                except RuntimeError:
                    async with self._binding_lock:
                        if self._closed or session_generation != self._generation:
                            return
                    raise
        finally:
            ingress.close()
            if receive_operation is not None:
                receive_operation.cancel()
                await asyncio.gather(receive_operation, return_exceptions=True)
            if not consumer.done():
                consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)

    async def _consume_source_ingress(
        self,
        binding: ConversationSessionWorker,
        ingress: BoundedPcmIngress,
    ) -> None:
        while True:
            record = await ingress.receive()
            try:
                async with self._binding_lock:
                    if (
                        self._closed
                        or record.session_generation != self._generation
                        or self._binding is not binding
                    ):
                        return
                    if (
                        self._require_media_activation
                        and record.track_name != self._media_track_name
                    ):
                        continue
                    suppression = self._audio_input_suppression
                    if (
                        suppression is not None
                        and suppression[0] == record.session_generation
                        and suppression[1] == record.participant_identity
                    ):
                        continue
                await binding.receive_audio(
                    record.participant_identity,
                    record.session_generation,
                    record.frame,
                )
            finally:
                ingress.complete(record)

    async def close(self) -> None:
        operation = self._close_operation
        if operation is None or _operation_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="reconnect-safe-conversation-worker-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    async def _close_owned(self) -> None:
        async with self._binding_lock:
            self._closed = True
            self._audio_input_suppression = None
            binding = self._binding
            errors: list[BaseException] = []
            if binding is not None:
                try:
                    await binding.close_binding()
                except BaseException as error:
                    errors.append(error)
                else:
                    if self._binding is binding:
                        self._binding = None
            try:
                await self._actions.close()
            except BaseException as error:
                errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("reconnect-safe worker close failed", errors)
