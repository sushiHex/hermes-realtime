"""Provider-neutral incremental inference-to-speech foreground loop."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import uuid4

from hermes_realtime._qualification import (
    _current_qualification_checkpoint_channel,
    _current_qualification_owner_context,
)
from hermes_realtime.evidence import (
    AppendDisposition,
    EvidenceAdmissionControllerV1,
    EvidenceAdmissionViewV1,
    EvidenceTurnLease,
    ProactiveTurnAuthorityV1,
    TerminalReason,
    TurnKind,
)
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    _ProductionObservationRecorderV1,
)
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    DurationBoundedSynthesizer,
    SpeechChunk,
    SpeechPlayback,
    SpeechTurnCounts,
    StreamingSynthesizer,
    Transcript,
)

from .context import (
    AssistantTextAdmission,
    ConversationContextSnapshot,
    ConversationContextStore,
)
from .foreground import (
    ForegroundPublication,
    ForegroundTurnCoordinator,
    ForegroundTurnLease,
)
from .tasks import TaskTerminalOutcome
from .updates import UpdateDecision, UpdateDecisionKind

_MAX_SEGMENT_CHARS_LIMIT = 65_536
_MAX_SEGMENTS_LIMIT = 4096
_MAX_PROMPT_UPDATES = 16
DEFAULT_MAX_RESPONSE_SEGMENTS = 256

_PublicValue = str | int | bool | None
_Observer = Callable[[str, dict[str, _PublicValue]], None]
_LOGGER = logging.getLogger(__name__)


def _is_cancellation_only(error: BaseException) -> bool:
    if isinstance(error, asyncio.CancelledError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return all(_is_cancellation_only(child) for child in error.exceptions)
    return False


@dataclass(frozen=True, slots=True)
class _PrefetchedSpeech:
    publication: ForegroundPublication
    chunk: SpeechChunk
    segment_text_offset_utf16: int | None
    synthesis_attempt_id: str


@dataclass(slots=True)
class _TerminalEvidenceState:
    counts: SpeechTurnCounts | None = None
    ledger_close_failed: bool = False
    generated_segment_count: int = 0
    committed_conversation_context_snapshot: bytes | None = None


def _committed_conversation_context_snapshot_bytes(
    *,
    revision: int,
    messages: tuple[tuple[str, str], ...],
    active_tasks: tuple[tuple[str, str], ...],
    terminal_task_count: int,
    updates: tuple[tuple[int, str, str, str], ...] = (),
) -> bytes:
    """Encode the authoritative adapter boundary for content-free observation."""

    return json.dumps(
        {
            "activeTasks": active_tasks,
            "messages": messages,
            "revision": revision,
            "terminalTaskCount": terminal_task_count,
            "updates": updates,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _PrefetchedPublicationEnd:
    publication: ForegroundPublication


_PrefetchedItem = _PrefetchedSpeech | _PrefetchedPublicationEnd


@dataclass(frozen=True, slots=True)
class _ReplaySegment:
    text: str
    confirmed_chunk_count: int
    presentation_turn_id: str
    presentation_turn_generation: int
    presentation_segment_id: str


@dataclass(frozen=True, slots=True)
class _ResumableSpeech:
    source_turn_id: str
    segments: tuple[_ReplaySegment, ...]
    replay_of_evidence_turn_id: str | None = None
    replay_generation: int | None = None
    source_binding_id: str | None = None
    source_binding_generation: int | None = None
    source_logical_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ReplayIdentity:
    replay_of_evidence_turn_id: str
    replay_generation: int
    source_binding_id: str
    source_binding_generation: int
    source_logical_session_id: str


@dataclass(frozen=True, slots=True)
class _RevokedForegroundOwner:
    consumer: asyncio.Task[None]
    lease: ForegroundTurnLease
    turn_id: str
    generated: list[ForegroundPublication]
    delivered: set[int]
    delivered_chunks: dict[int, int]
    replay_presentations: dict[int, _ReplaySegment]
    revision: int
    preserve_resumable: bool
    replay_identity: _ReplayIdentity | None


@dataclass(frozen=True, slots=True)
class ConversationPromptUpdate:
    """One framed model input claimed for a genuine user turn."""

    sequence: int
    task_id: str
    status: str
    text: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int:
            raise TypeError("sequence must be an exact integer")
        if type(self.task_id) is not str:
            raise TypeError("task_id must be an exact built-in string")
        if type(self.status) is not str:
            raise TypeError("status must be an exact built-in string")
        if type(self.text) is not str:
            raise TypeError("text must be an exact built-in string")
        if self.status == "completed":
            completion = TaskTerminalOutcome(
                task_id=self.task_id,
                status=self.status,
                summary="validated",
            )
        else:
            completion = TaskTerminalOutcome(
                task_id=self.task_id,
                status=self.status,
                reason="validated",
            )
        decision = UpdateDecision(
            sequence=self.sequence,
            completion=completion,
            kind=UpdateDecisionKind.MENTION_NEXT.value,
            text=self.text,
        )
        object.__setattr__(self, "sequence", decision.sequence)
        object.__setattr__(self, "task_id", decision.task_id)
        object.__setattr__(self, "status", decision.status)
        assert decision.text is not None
        object.__setattr__(self, "text", decision.text)


@dataclass(frozen=True, slots=True)
class ConversationInferenceRequest(ConversationContextSnapshot):
    """Immutable context snapshot plus separately framed proactive updates."""

    updates: tuple[ConversationPromptUpdate, ...] = ()

    def __post_init__(self) -> None:
        ConversationContextSnapshot.__post_init__(self)
        if type(self.updates) is not tuple:
            raise TypeError("inference request updates must be an exact tuple")
        if len(self.updates) > _MAX_PROMPT_UPDATES:
            raise ValueError("inference request update capacity exceeded")
        updates = tuple(
            ConversationPromptUpdate(
                sequence=update.sequence,
                task_id=update.task_id,
                status=update.status,
                text=update.text,
            )
            for update in self.updates
            if type(update) is ConversationPromptUpdate
        )
        if len(updates) != len(self.updates):
            raise TypeError("inference request must contain exact ConversationPromptUpdate values")
        object.__setattr__(self, "updates", updates)

    @property
    def context(self) -> ConversationContextSnapshot:
        return ConversationContextSnapshot(
            revision=self.revision,
            messages=self.messages,
            active_tasks=self.active_tasks,
            terminal_task_count=self.terminal_task_count,
        )


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


class StreamingInference(Protocol):
    """Produce bounded foreground text segments from one immutable snapshot."""

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]: ...

    async def cancel(self, turn_id: str) -> None: ...


@dataclass(frozen=True, slots=True, eq=False)
class _UpdateSpeechAuthority:
    """Opaque capability held by the sole proactive update executor."""


@dataclass(frozen=True, slots=True, eq=False)
class _UpdateSpeechOperation:
    """Opaque delivery-proof scope for one proactive speech operation."""


class StreamingSpeechLoop:
    """Stream final user turns through inference, synthesis, and playback."""

    def __init__(
        self,
        *,
        context: ConversationContextStore,
        foreground: ForegroundTurnCoordinator,
        inference: StreamingInference,
        synthesizer: StreamingSynthesizer,
        playback: SpeechPlayback,
        ledger: DeliveredSpeechLedger,
        evidence_admission: EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1 | None = None,
        evidence_resolver: Callable[
            [],
            tuple[
                object | None,
                EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1 | None,
            ],
        ]
        | None = None,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
        observer: _Observer | None = None,
        max_segment_chars: int = 4096,
        max_segments: int = DEFAULT_MAX_RESPONSE_SEGMENTS,
        cleanup_timeout_seconds: float = 1.0,
        max_speech_chunk_duration_seconds: float | None = None,
    ) -> None:
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable")
        if (
            evidence_admission is not None
            and type(evidence_admission)
            not in (EvidenceAdmissionControllerV1, EvidenceAdmissionViewV1)
        ):
            raise TypeError("evidence_admission must be an exact admission surface or None")
        if evidence_resolver is not None and not callable(evidence_resolver):
            raise TypeError("evidence_resolver must be callable or None")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        if type(max_segment_chars) is not int:
            raise TypeError("max_segment_chars must be an exact integer")
        if max_segment_chars <= 0:
            raise ValueError("max_segment_chars must be positive")
        if max_segment_chars > _MAX_SEGMENT_CHARS_LIMIT:
            raise ValueError("max_segment_chars exceeds supported maximum")
        if type(max_segments) is not int:
            raise TypeError("max_segments must be an exact integer")
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        if max_segments > _MAX_SEGMENTS_LIMIT:
            raise ValueError("max_segments exceeds supported maximum")
        if type(cleanup_timeout_seconds) not in (int, float):
            raise TypeError("cleanup_timeout_seconds must be an exact number")
        if not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be finite and positive")
        self._context = context
        self._foreground = foreground
        self._inference = inference
        self._synthesizer = (
            synthesizer
            if max_speech_chunk_duration_seconds is None
            else DurationBoundedSynthesizer(
                synthesizer,
                max_duration_seconds=max_speech_chunk_duration_seconds,
            )
        )
        self._playback = playback
        self._ledger = ledger
        self._evidence_admission = evidence_admission
        self._evidence_resolver = evidence_resolver
        self._production_observation_recorder = production_observation_recorder
        # Raw qualification text may flow only through this internal owner
        # context capability.  Public construction has no trace injection.
        self._qualification_owner_context = _current_qualification_owner_context()
        self._qualification_checkpoint_channel = _current_qualification_checkpoint_channel()
        self._observer = observer
        self._max_segment_chars = max_segment_chars
        self._max_segments = max_segments
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._active_turn_id: str | None = None
        self._active_lease: ForegroundTurnLease | None = None
        self._active_consumer: asyncio.Task[None] | None = None
        self._active_generated_segments: list[ForegroundPublication] | None = None
        self._active_delivered_sequences: set[int] | None = None
        self._active_delivered_chunk_counts: dict[int, int] | None = None
        self._active_replay_presentations: dict[int, _ReplaySegment] | None = None
        self._active_preserve_resumable_on_cancel: bool | None = None
        self._active_replay_identity: _ReplayIdentity | None = None
        self._active_evidence_lease: EvidenceTurnLease | None = None
        self._resumable_cleanup_count = 0
        self._authority_revision = 0
        self._priority_admissions = 0
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_settlement_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        self._resumable_speech: _ResumableSpeech | None = None
        self._resume_sequence = 0
        self._provider_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._update_action_owner: object | None = None
        self._update_speech_authority: _UpdateSpeechAuthority | None = None
        self._update_operations: dict[
            int,
            tuple[_UpdateSpeechOperation, str, bool],
        ] = {}

    @property
    def active_turn_id(self) -> str | None:
        return self._active_turn_id

    def _resolve_evidence_admission(
        self,
    ) -> EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1 | None:
        resolver = self._evidence_resolver
        if resolver is None:
            return self._evidence_admission
        _lifecycle, admission = resolver()
        if admission is not None and type(admission) not in (
            EvidenceAdmissionControllerV1,
            EvidenceAdmissionViewV1,
        ):
            raise TypeError("evidence resolver returned a non-exact admission surface")
        return admission

    def _observe_committed_conversation_context_snapshot(
        self,
        committed_conversation_context_snapshot: bytes | None,
    ) -> None:
        recorder = self._production_observation_recorder
        if recorder is None or committed_conversation_context_snapshot is None:
            return
        recorder.record_context_committed()

    def _observe_terminal_settlement(
        self,
        evidence: EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1,
        lease: EvidenceTurnLease,
        settlement_disposition: AppendDisposition,
    ) -> bool:
        if settlement_disposition is not AppendDisposition.ADMITTED:
            return False
        recorder = self._production_observation_recorder
        if recorder is None:
            return False
        outcome = evidence.settled_terminal_outcome(lease)
        if outcome is None:
            return False
        recorder.record_terminal_settled(
            terminal_disposition=outcome.terminal_disposition,
            terminal_reason=outcome.terminal_reason,
            context_committed=outcome.context_committed,
        )
        return outcome.context_committed

    def _notify_completion_advisory(self, *, turn_id: str, turn_generation: int) -> None:
        """Keep advisory projection failure outside authoritative settlement."""

        observer = self._observer
        if observer is None:
            return
        try:
            observer(
                "assistant_turn_completed",
                {"turnId": turn_id, "turnGeneration": turn_generation},
            )
        except Exception:
            _LOGGER.warning("assistant completion observer failed", exc_info=True)

    def _eligible_replay_identity(self) -> _ReplayIdentity | None:
        resumable = self._resumable_speech
        if (
            resumable is None
            or resumable.replay_of_evidence_turn_id is None
            or resumable.replay_generation is None
            or resumable.source_binding_id is None
            or resumable.source_binding_generation is None
            or resumable.source_logical_session_id is None
        ):
            return None
        return _ReplayIdentity(
            replay_of_evidence_turn_id=resumable.replay_of_evidence_turn_id,
            replay_generation=resumable.replay_generation,
            source_binding_id=resumable.source_binding_id,
            source_binding_generation=resumable.source_binding_generation,
            source_logical_session_id=resumable.source_logical_session_id,
        )

    async def respond(self, turn_id: str, transcript: Transcript) -> None:
        """Run one final transcript until its incremental speech is delivered."""

        self._validate_transcript(transcript)
        self._priority_admissions += 1
        try:
            await self._run_turn(
                turn_id,
                transcript=transcript,
                announcement=None,
                updates=(),
                update_operation=None,
            )
        finally:
            self._priority_admissions -= 1

    async def resume_interrupted(
        self,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> bool:
        """Replay unconfirmed generated segments without another inference call."""

        if evidence_lease is not None and type(evidence_lease) is not EvidenceTurnLease:
            raise TypeError("evidence_lease must be exact or None")
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("streaming speech loop is closed")
            resumable = self._resumable_speech
            if resumable is None:
                return False
            self._resume_sequence += 1
            turn_id = f"resume_{self._resume_sequence:016x}"
        self._priority_admissions += 1
        try:
            return await self._run_turn(
                turn_id,
                transcript=None,
                announcement=None,
                segments=resumable.segments,
                updates=(),
                update_operation=None,
                record_generation=False,
                preserve_resumable=True,
                expected_resumable=resumable,
                replay_identity=(
                    None
                    if resumable.replay_of_evidence_turn_id is None
                    or resumable.replay_generation is None
                    or resumable.source_binding_id is None
                    or resumable.source_binding_generation is None
                    or resumable.source_logical_session_id is None
                    else _ReplayIdentity(
                        replay_of_evidence_turn_id=resumable.replay_of_evidence_turn_id,
                        replay_generation=resumable.replay_generation + 1,
                        source_binding_id=resumable.source_binding_id,
                        source_binding_generation=resumable.source_binding_generation,
                        source_logical_session_id=resumable.source_logical_session_id,
                    )
                ),
                evidence_lease=evidence_lease,
            )
        finally:
            self._priority_admissions -= 1

    def _bind_update_executor(self, owner: object) -> _UpdateSpeechAuthority:
        if owner is None:
            raise TypeError("update action owner must not be None")
        if self._update_action_owner is None:
            authority = _UpdateSpeechAuthority()
            self._update_action_owner = owner
            self._update_speech_authority = authority
            return authority
        if self._update_action_owner is not owner:
            raise RuntimeError("streaming speech loop already has an update executor")
        assert self._update_speech_authority is not None
        return self._update_speech_authority

    async def _respond_with_updates(
        self,
        authority: _UpdateSpeechAuthority,
        operation: _UpdateSpeechOperation,
        turn_id: str,
        transcript: Transcript,
        updates: tuple[UpdateDecision, ...],
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> None:
        self._require_update_authority(authority)
        self._require_update_operation(operation, turn_id)
        if evidence_lease is not None and type(evidence_lease) is not EvidenceTurnLease:
            raise TypeError("evidence_lease must be exact or None")
        self._validate_transcript(transcript)
        prompt_updates = self._prompt_updates(updates)
        self._priority_admissions += 1
        try:
            await self._run_turn(
                turn_id,
                transcript=transcript,
                announcement=None,
                updates=prompt_updates,
                update_operation=operation,
                evidence_lease=evidence_lease,
                replay_identity=(
                    None
                    if evidence_lease is None
                    else _ReplayIdentity(
                        replay_of_evidence_turn_id=evidence_lease.evidence_turn_id,
                        replay_generation=1,
                        source_binding_id=evidence_lease.binding_id,
                        source_binding_generation=evidence_lease.binding_generation,
                        source_logical_session_id=evidence_lease.logical_session_id,
                    )
                ),
            )
        finally:
            self._priority_admissions -= 1

    async def _announce_update(
        self,
        authority: _UpdateSpeechAuthority,
        operation: _UpdateSpeechOperation,
        turn_id: str,
        decision: UpdateDecision,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> None:
        self._require_update_authority(authority)
        self._require_update_operation(operation, turn_id)
        if type(decision) is not UpdateDecision:
            raise TypeError("decision must be an exact UpdateDecision")
        if evidence_lease is not None and type(evidence_lease) is not EvidenceTurnLease:
            raise TypeError("evidence_lease must be exact or None")
        decision = UpdateDecision(
            sequence=decision.sequence,
            completion=decision.completion,
            kind=decision.kind,
            text=decision.text,
        )
        if decision.kind != UpdateDecisionKind.INTERRUPT.value:
            raise ValueError("only interrupt decisions can start an announcement")
        assert decision.text is not None
        self._priority_admissions += 1
        try:
            await self._run_turn(
                turn_id,
                transcript=None,
                announcement=decision.text,
                updates=(),
                update_operation=operation,
                evidence_lease=evidence_lease,
            )
        finally:
            self._priority_admissions -= 1

    async def _announce_idle_update(
        self,
        authority: _UpdateSpeechAuthority,
        operation: _UpdateSpeechOperation,
        turn_id: str,
        decision: UpdateDecision,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> bool:
        self._require_update_authority(authority)
        self._require_update_operation(operation, turn_id)
        if type(decision) is not UpdateDecision:
            raise TypeError("decision must be an exact UpdateDecision")
        if evidence_lease is not None and type(evidence_lease) is not EvidenceTurnLease:
            raise TypeError("evidence_lease must be exact or None")
        decision = UpdateDecision(
            sequence=decision.sequence,
            completion=decision.completion,
            kind=decision.kind,
            text=decision.text,
        )
        if decision.kind != UpdateDecisionKind.MENTION_NEXT.value:
            raise ValueError("only mention-next decisions can start an idle announcement")
        assert decision.text is not None
        return await self._run_turn(
            turn_id,
            transcript=None,
            announcement=decision.text,
            updates=(),
            update_operation=operation,
            idle_only=True,
            preserve_resumable_on_cancel=False,
            evidence_lease=evidence_lease,
        )

    @staticmethod
    def _validate_transcript(transcript: Transcript) -> None:
        if type(transcript) is not Transcript:
            raise TypeError("transcript must be an exact Transcript value")
        Transcript(text=transcript.text, final=transcript.final)
        if not transcript.final:
            raise ValueError("only final transcripts can start a response")

    def _require_update_authority(
        self,
        authority: _UpdateSpeechAuthority,
    ) -> None:
        if (
            type(authority) is not _UpdateSpeechAuthority
            or authority is not self._update_speech_authority
        ):
            raise RuntimeError("caller does not own proactive speech execution")

    def _begin_update_operation(
        self,
        authority: _UpdateSpeechAuthority,
        turn_id: str,
        evidence_authority: ProactiveTurnAuthorityV1 | None = None,
    ) -> _UpdateSpeechOperation:
        self._require_update_authority(authority)
        if (
            evidence_authority is not None
            and type(evidence_authority) is not ProactiveTurnAuthorityV1
        ):
            raise TypeError("evidence_authority must be exact or None")
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id.strip():
            raise ValueError("turn_id must not be blank")
        if len(self._update_operations) >= 64:
            raise RuntimeError("proactive speech operation capacity exhausted")
        operation = _UpdateSpeechOperation()
        self._update_operations[id(operation)] = (operation, turn_id, False)
        return operation

    def _require_update_operation(
        self,
        operation: _UpdateSpeechOperation,
        turn_id: str,
    ) -> tuple[_UpdateSpeechOperation, str, bool]:
        if type(operation) is not _UpdateSpeechOperation:
            raise TypeError("operation must be an exact update speech operation")
        registered = self._update_operations.get(id(operation))
        if registered is None or registered[0] is not operation:
            raise RuntimeError("update speech operation is not active")
        if registered[1] != turn_id:
            raise RuntimeError("update speech operation turn does not match")
        return registered

    def _mark_update_operation_delivered(
        self,
        operation: _UpdateSpeechOperation,
    ) -> None:
        if type(operation) is not _UpdateSpeechOperation:
            raise TypeError("operation must be an exact update speech operation")
        registered = self._update_operations.get(id(operation))
        if registered is None or registered[0] is not operation:
            raise RuntimeError("update speech operation is not active")
        self._update_operations[id(operation)] = (
            registered[0],
            registered[1],
            True,
        )

    def _finish_update_operation(
        self,
        authority: _UpdateSpeechAuthority,
        operation: _UpdateSpeechOperation,
    ) -> bool:
        self._require_update_authority(authority)
        if type(operation) is not _UpdateSpeechOperation:
            raise TypeError("operation must be an exact update speech operation")
        registered = self._update_operations.get(id(operation))
        if registered is None or registered[0] is not operation:
            raise RuntimeError("update speech operation is not active")
        del self._update_operations[id(operation)]
        return registered[2]

    async def _run_turn(
        self,
        turn_id: str,
        *,
        transcript: Transcript | None,
        announcement: str | None,
        updates: tuple[ConversationPromptUpdate, ...],
        update_operation: _UpdateSpeechOperation | None,
        segments: tuple[_ReplaySegment, ...] | None = None,
        record_generation: bool = True,
        preserve_resumable: bool = False,
        expected_resumable: _ResumableSpeech | None = None,
        idle_only: bool = False,
        preserve_resumable_on_cancel: bool = True,
        replay_identity: _ReplayIdentity | None = None,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> bool:
        if sum(value is not None for value in (transcript, announcement, segments)) != 1:
            raise ValueError("exactly one turn input must be present")

        producer_done = asyncio.Event()
        producer_result: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        pending_admissions: list[AssistantTextAdmission] = []
        generated_segments: list[ForegroundPublication] = []
        delivered_sequences: set[int] = set()
        delivered_chunk_counts: dict[int, int] = {}
        replay_skip_counts = (
            {
                sequence: segment.confirmed_chunk_count
                for sequence, segment in enumerate(segments, start=1)
            }
            if segments is not None
            else {}
        )
        replay_presentations = (
            {sequence: segment for sequence, segment in enumerate(segments, start=1)}
            if segments is not None
            else {}
        )
        terminal_evidence = _TerminalEvidenceState()

        async def producer(lease: ForegroundTurnLease) -> None:
            def admit_generated(publication: ForegroundPublication) -> None:
                if record_generation:
                    self._context.validate_assistant_generation(publication.text)
                    self._qualification_owner_context.record_generated_text(
                        publication.text.encode("utf-8")
                    )
                admission = self._resolve_evidence_admission()
                if admission is not None and evidence_lease is not None and record_generation:
                    with suppress(Exception):
                        disposition = admission.try_admit_generated(
                            evidence_lease, publication.text
                        )
                        if disposition is AppendDisposition.ADMITTED:
                            terminal_evidence.generated_segment_count += 1
                observer = self._observer
                if observer is not None and record_generation:
                    observer(
                        "assistant_text_generated",
                        {
                            "role": "assistant",
                            "text": publication.text,
                            "turnId": publication.turn_id,
                            "turnGeneration": publication.generation,
                            "segmentId": publication.segment_id,
                        },
                    )
                if record_generation:
                    self._context.record_assistant_generation(publication.text)
                generated_segments.append(publication)

            try:
                if segments is not None:
                    inference = self._segments_stream(segments)
                elif transcript is None:
                    assert announcement is not None
                    inference = self._announcement_stream(announcement)
                else:
                    self._context.record_user_transcript(transcript)
                    source_snapshot = self._context.snapshot()
                    snapshot = ConversationInferenceRequest(
                        revision=source_snapshot.revision,
                        messages=source_snapshot.messages,
                        active_tasks=source_snapshot.active_tasks,
                        terminal_task_count=source_snapshot.terminal_task_count,
                        updates=updates,
                    )
                    committed_conversation_context_snapshot = (
                        _committed_conversation_context_snapshot_bytes(
                            revision=snapshot.revision,
                            messages=tuple(
                                (message.role, message.text) for message in snapshot.messages
                            ),
                            active_tasks=tuple(
                                (task.task_id, task.objective)
                                for task in snapshot.active_tasks
                            ),
                            terminal_task_count=snapshot.terminal_task_count,
                            updates=tuple(
                                (update.sequence, update.task_id, update.status, update.text)
                                for update in snapshot.updates
                            ),
                        )
                    )
                    terminal_evidence.committed_conversation_context_snapshot = (
                        committed_conversation_context_snapshot
                    )
                    inference = self._inference.stream(snapshot, turn_id=turn_id)
                    self._qualification_owner_context.record_committed_conversation_context_snapshot(
                        committed_conversation_context_snapshot
                    )
                segment_count = 0
                iteration_error: BaseException | None = None
                try:
                    async for segment in inference:
                        segment_count += 1
                        if segment_count > self._max_segments:
                            raise ValueError("inference segment count capacity exceeded")
                        if type(segment) is not str:
                            raise TypeError("inference segment must be an exact built-in string")
                        if len(segment) > self._max_segment_chars:
                            raise ValueError("inference segment character capacity exceeded")
                        if not segment.strip():
                            raise ValueError("inference segment must not be blank")
                        if segment_count == 1 and transcript is not None:
                            observer = self._observer
                            if observer is not None:
                                observer("first_foreground_token", {})
                        if not await lease.publish(segment, on_admit=admit_generated):
                            break
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    lease.invalidate_publications()
                    iteration_error = error
                close_error = await self._close_stream(inference)
                if iteration_error is not None or close_error is not None:
                    admission = self._resolve_evidence_admission()
                    if admission is not None and evidence_lease is not None:
                        with suppress(Exception):
                            admission.record_terminal_cause(
                                evidence_lease.terminal_cause,
                                TerminalReason.PROVIDER_FAILED,
                            )
                self._raise_iteration_or_close_error(
                    "inference iteration and close failed",
                    iteration_error,
                    close_error,
                )
            except asyncio.CancelledError:
                producer_result.cancel()
                raise
            except BaseException as error:
                if not producer_result.done():
                    producer_result.set_exception(error)
                raise
            else:
                if not producer_result.done():
                    producer_result.set_result(None)
            finally:
                producer_done.set()

        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("streaming speech loop is closed")
            if idle_only and (
                self._priority_admissions
                or self._resumable_speech is not None
                or self._resumable_cleanup_count
            ):
                return False
            if expected_resumable is not None and (
                self._resumable_speech is not expected_resumable
                or self._active_consumer is not None
            ):
                return False
            self._authority_revision += 1
            revision = self._authority_revision
            if expected_resumable is not None:
                self._resumable_speech = None
            else:
                self._revoke_active_locked(
                    preserve_resumable=preserve_resumable,
                    terminal_reason=TerminalReason.RESPONSE_REPLACED,
                )
            if not preserve_resumable:
                self._resumable_speech = None

        await self._settle_owned_cleanups()
        self._raise_if_provider_cleanup_pending()
        lease = await self._foreground.start(turn_id, producer)
        consumer: asyncio.Task[None] | None = None
        opening_rejected = False
        startup_error: BaseException | None = None
        async with self._lifecycle_lock:
            if self._closed or revision != self._authority_revision:
                revoked = self._foreground.revoke_if_current(lease)
                if revoked is not None:
                    task, owner_turn_id = revoked
                    self._track_cleanup(
                        asyncio.create_task(
                            self._foreground.settle_revoked(task, turn_id=owner_turn_id)
                        )
                    )
                raise asyncio.CancelledError
            if (
                evidence_lease is not None
                and getattr(evidence_lease, "turn_kind", None)
                in (TurnKind.PROACTIVE_UPDATE, TurnKind.REPLAY)
            ):
                evidence = self._resolve_evidence_admission()
                opening_rejected = (
                    evidence is None
                    or evidence.try_open_non_user_turn(evidence_lease)
                    is not AppendDisposition.ADMITTED
                )
            if opening_rejected:
                revoked = self._foreground.revoke_if_current(lease)
                if revoked is not None:
                    task, owner_turn_id = revoked
                    self._track_cleanup(
                        asyncio.create_task(
                            self._foreground.settle_revoked(task, turn_id=owner_turn_id)
                        )
                    )
            else:
                try:
                    self._ledger.begin_turn(turn_id)
                    consumer = asyncio.create_task(
                        self._consume_turn(
                            turn_id,
                            turn_generation=lease.generation,
                            producer_done=producer_done,
                            producer_result=producer_result,
                            pending_admissions=pending_admissions,
                            delivered_sequences=delivered_sequences,
                            delivered_chunk_counts=delivered_chunk_counts,
                            replay_skip_counts=replay_skip_counts,
                            replay_presentations=replay_presentations,
                            cancel_inference=transcript is not None,
                            update_operation=update_operation,
                            evidence_lease=evidence_lease,
                            terminal_evidence=terminal_evidence,
                        ),
                        name=f"foreground-speech:{turn_id}:{lease.generation}",
                    )
                except BaseException as error:
                    startup_error = error
                    with suppress(BaseException):
                        self._ledger.close_turn(turn_id)
                    evidence = self._resolve_evidence_admission()
                    if evidence is not None and evidence_lease is not None:
                        with suppress(Exception):
                            evidence.settle_spawn_failed(evidence_lease)
                    revoked = self._foreground.revoke_if_current(lease)
                    if revoked is not None:
                        task, owner_turn_id = revoked
                        self._track_cleanup(
                            asyncio.create_task(
                                self._foreground.settle_revoked(task, turn_id=owner_turn_id)
                            )
                        )
                else:
                    self._active_turn_id = turn_id
                    self._active_lease = lease
                    self._active_consumer = consumer
                    self._active_generated_segments = generated_segments
                    self._active_delivered_sequences = delivered_sequences
                    self._active_delivered_chunk_counts = delivered_chunk_counts
                    self._active_replay_presentations = replay_presentations
                    self._active_preserve_resumable_on_cancel = preserve_resumable_on_cancel
                    self._active_replay_identity = replay_identity
                    self._active_evidence_lease = evidence_lease

        if opening_rejected:
            await self._settle_owned_cleanups()
            return False
        if startup_error is not None:
            await self._settle_owned_cleanups()
            raise startup_error
        assert consumer is not None

        async def stop_consumer_on_producer_failure() -> None:
            try:
                await asyncio.shield(producer_result)
            except asyncio.CancelledError:
                return
            except BaseException:
                await self._foreground.cancel_if_current(lease)
                if not consumer.done():
                    consumer.cancel()

        failure_monitor = asyncio.create_task(
            stop_consumer_on_producer_failure(),
            name=f"foreground-speech-monitor:{turn_id}:{lease.generation}",
        )

        try:
            await consumer
            while lease.is_current:
                await asyncio.sleep(0)
        except asyncio.CancelledError as cancellation_error:
            caller = asyncio.current_task()
            caller_cancelled = caller is not None and caller.cancelling() > 0
            errors: list[BaseException] = []
            if caller_cancelled:
                evidence = self._resolve_evidence_admission()
                if evidence is not None and evidence_lease is not None:
                    with suppress(Exception):
                        evidence.record_terminal_cause(
                            evidence_lease.terminal_cause,
                            TerminalReason.CALLER_CANCELLED,
                        )
                self._qualification_owner_context.record_cancellation(
                    TerminalReason.CALLER_CANCELLED
                )
            async with self._lifecycle_lock:
                if self._active_consumer is consumer:
                    self._authority_revision += 1
                    self._revoke_active_locked(preserve_resumable=preserve_resumable_on_cancel)
            try:
                await self._settle_owned_cleanups()
            except BaseException as candidate:
                if not isinstance(candidate, asyncio.CancelledError):
                    errors.append(candidate)

            if not producer_result.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.wait(
                        {producer_result},
                        timeout=self._cleanup_timeout_seconds,
                    )
            if producer_result.done() and not producer_result.cancelled():
                producer_error = producer_result.exception()
                if producer_error is not None and not any(
                    self._exception_contains(candidate, producer_error) for candidate in errors
                ):
                    errors.append(producer_error)

            if not caller_cancelled:
                if len(errors) == 1:
                    raise errors[0] from None
                if errors:
                    raise BaseExceptionGroup("streaming response failed", errors) from None
                raise cancellation_error

            if errors:
                raise BaseExceptionGroup(
                    "caller cancellation and streaming response failure",
                    [cancellation_error, *errors],
                ) from None
            raise cancellation_error
        except BaseException as error:
            async with self._lifecycle_lock:
                if self._active_consumer is consumer:
                    self._authority_revision += 1
                    self._revoke_active_locked(preserve_resumable=preserve_resumable_on_cancel)
            try:
                await self._settle_owned_cleanups()
            except BaseException as cleanup_error:
                if cleanup_error is error:
                    raise
                raise BaseExceptionGroup(
                    "speech consumer and foreground cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise
        finally:
            if not failure_monitor.done():
                failure_monitor.cancel()
            with suppress(BaseException):
                await failure_monitor
            async with self._lifecycle_lock:
                if self._active_consumer is consumer and consumer.done():
                    self._active_consumer = None
                    self._active_lease = None
                    self._active_turn_id = None
                    self._active_generated_segments = None
                    self._active_delivered_sequences = None
                    self._active_delivered_chunk_counts = None
                    self._active_replay_presentations = None
                    self._active_preserve_resumable_on_cancel = None
                    self._active_replay_identity = None
                    self._active_evidence_lease = None
        return True

    @property
    def foreground_active(self) -> bool:
        """Whether one foreground response or proactive speech turn is active."""

        return self._active_consumer is not None and not self._active_consumer.done()

    async def cancel_if_active(self, turn_id: str) -> bool:
        """Cancel only when ``turn_id`` still owns foreground speech authority."""

        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id:
            raise ValueError("turn_id must not be empty")
        async with self._lifecycle_lock:
            if self._active_turn_id != turn_id or not self.foreground_active:
                return False
            self._authority_revision += 1
            self._revoke_active_locked(terminal_reason=TerminalReason.BARGE_IN)
        await self._settle_owned_cleanups()
        return True

    async def cancel(self) -> None:
        """Invalidate and settle the complete active foreground speech turn."""

        async with self._lifecycle_lock:
            self._authority_revision += 1
            self._revoke_active_locked(terminal_reason=TerminalReason.STOP_SPEAKING)
        await self._settle_owned_cleanups()

    async def cancel_for_host_shutdown(self) -> None:
        """Settle active foreground as host-owned shutdown before final close."""

        async with self._lifecycle_lock:
            self._authority_revision += 1
            self._revoke_active_locked(
                preserve_resumable=False,
                terminal_reason=TerminalReason.HOST_SHUTDOWN,
            )
        await self._settle_owned_cleanups()

    async def cancel_for_binding_close(self) -> None:
        """Settle active foreground as an authoritative binding closure."""

        async with self._lifecycle_lock:
            self._authority_revision += 1
            self._revoke_active_locked(
                preserve_resumable=False,
                terminal_reason=TerminalReason.BINDING_CLOSED,
            )
        await self._settle_owned_cleanups()

    async def cancel_for_retention_expiry(self) -> None:
        """Settle active foreground as retention-owned cancellation."""

        async with self._lifecycle_lock:
            self._authority_revision += 1
            self._revoke_active_locked(
                preserve_resumable=False,
                terminal_reason=TerminalReason.RETENTION_EXPIRED,
            )
        await self._settle_owned_cleanups()

    async def close(self) -> None:
        """Permanently stop transcript admission and settle active speech."""

        operation = self._close_operation
        if operation is None or self._close_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="streaming-speech-loop-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    @staticmethod
    def _close_failed(operation: asyncio.Task[None]) -> bool:
        if not operation.done():
            return False
        if operation.cancelled():
            return True
        return operation.exception() is not None

    async def _close_owned(self) -> None:
        recorder = self._production_observation_recorder
        async with self._lifecycle_lock:
            self._closed = True
            self._authority_revision += 1
            self._revoke_active_locked(
                preserve_resumable=False,
                terminal_reason=TerminalReason.HOST_SHUTDOWN,
            )
        errors: list[BaseException] = []
        try:
            await self._settle_owned_cleanups()
        except BaseException as error:
            errors.append(error)
        try:
            await self._foreground.close()
        except BaseException as error:
            errors.append(error)
            foreground_result = CloseResultV1.FAILED
        else:
            foreground_result = CloseResultV1.SUCCEEDED
        if recorder is not None:
            recorder.record_close_stage(
                stage=CloseStageV1.FOREGROUND_CLOSE,
                result=foreground_result,
            )
        try:
            self._raise_if_provider_cleanup_pending()
        except BaseException as error:
            errors.append(error)
        if len(errors) == 1:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.SPEECH_LOOP,
                    result=CloseResultV1.FAILED,
                )
            raise errors[0]
        if errors:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.SPEECH_LOOP,
                    result=CloseResultV1.FAILED,
                )
            raise BaseExceptionGroup("streaming speech loop close failed", errors)
        if recorder is not None:
            recorder.record_close_stage(
                stage=CloseStageV1.SPEECH_LOOP,
                result=CloseResultV1.SUCCEEDED,
            )

    async def _consume_turn(
        self,
        turn_id: str,
        *,
        turn_generation: int,
        producer_done: asyncio.Event,
        producer_result: asyncio.Future[None],
        pending_admissions: list[AssistantTextAdmission],
        delivered_sequences: set[int],
        delivered_chunk_counts: dict[int, int],
        replay_skip_counts: dict[int, int],
        replay_presentations: dict[int, _ReplaySegment],
        cancel_inference: bool,
        update_operation: _UpdateSpeechOperation | None,
        evidence_lease: EvidenceTurnLease | None,
        terminal_evidence: _TerminalEvidenceState,
    ) -> None:
        first_playable_observed = False
        replay_lifecycle = replay_presentations.get(1)
        lifecycle_turn_id = (
            replay_lifecycle.presentation_turn_id if replay_lifecycle is not None else turn_id
        )
        lifecycle_generation = (
            replay_lifecycle.presentation_turn_generation
            if replay_lifecycle is not None
            else turn_generation
        )
        prefetched: asyncio.Queue[_PrefetchedItem] = asyncio.Queue(maxsize=1)
        prefetch_slots = asyncio.Semaphore(1)
        partially_delivered_sequences: set[int] = set()
        last_delivered_chunk_ids: dict[int, str] = {}
        prefetch_task = asyncio.create_task(
            self._prefetch_speech(
                producer_done=producer_done,
                output=prefetched,
                slots=prefetch_slots,
                replay_skip_counts=replay_skip_counts,
                evidence_lease=evidence_lease,
            )
        )
        try:
            while True:
                item = await self._next_prefetched_speech(prefetched, prefetch_task)
                if item is None:
                    break
                publication = item.publication
                replay_presentation = replay_presentations.get(publication.sequence)
                presentation_turn_id = (
                    replay_presentation.presentation_turn_id
                    if replay_presentation is not None
                    else turn_id
                )
                presentation_generation = (
                    replay_presentation.presentation_turn_generation
                    if replay_presentation is not None
                    else turn_generation
                )
                presentation_segment_id = (
                    replay_presentation.presentation_segment_id
                    if replay_presentation is not None
                    else publication.segment_id
                )
                if type(item) is _PrefetchedPublicationEnd:
                    if publication.sequence in partially_delivered_sequences:
                        delivered_sequences.add(publication.sequence)
                        observer = self._observer
                        if observer is not None:
                            observer(
                                "transcript_final",
                                {
                                    "role": "assistant",
                                    "text": publication.text,
                                    "turnId": presentation_turn_id,
                                    "turnGeneration": presentation_generation,
                                    "chunkId": last_delivered_chunk_ids[publication.sequence],
                                    "segmentId": presentation_segment_id,
                                },
                            )
                        if update_operation is not None:
                            self._mark_update_operation_delivered(update_operation)
                    continue
                if not isinstance(item, _PrefetchedSpeech):
                    raise TypeError("prefetch queue contained an unknown item")
                prefetch_slots.release()
                chunk = item.chunk
                if not publication.is_valid:
                    raise asyncio.CancelledError
                admission = self._context.prepare_assistant_text(
                    chunk.text,
                    record_delivery=False,
                )
                try:
                    self._ledger.queue(chunk, admission=admission)
                except BaseException:
                    self._context.discard_assistant_text(admission)
                    raise
                pending_admissions.append(admission)
                receipt = self._ledger.mark_started(chunk.turn_id, chunk.chunk_id)
                transport_attempt_id = str(uuid4())
                if not first_playable_observed:
                    observer = self._observer
                    if observer is not None:
                        observer("first_playable_audio", {})
                    first_playable_observed = True
                observer = self._observer
                if observer is not None:
                    observer(
                        "transcript_partial",
                        {
                            "role": "assistant",
                            "text": chunk.text,
                            "turnId": presentation_turn_id,
                            "turnGeneration": presentation_generation,
                            "chunkId": chunk.chunk_id,
                            "segmentId": presentation_segment_id,
                            "segmentTextOffsetUtf16": item.segment_text_offset_utf16,
                        },
                    )
                try:
                    await self._playback.play(
                        chunk,
                        is_valid=self._validity_probe(publication),
                    )
                except Exception:
                    evidence = self._resolve_evidence_admission()
                    if evidence is not None and evidence_lease is not None:
                        with suppress(Exception):
                            evidence.record_terminal_cause(
                                evidence_lease.terminal_cause,
                                TerminalReason.TRANSPORT_FAILED,
                            )
                    raise
                if not publication.is_valid:
                    raise asyncio.CancelledError
                confirmation = self._ledger.mark_delivered_confirmed(receipt)
                confirmed_text = self._ledger.confirmed_text(confirmation, admission)
                self._qualification_owner_context.record_transport_confirmed_chunk(
                    confirmed_text.encode("utf-8")
                )
                evidence = self._resolve_evidence_admission()
                if evidence is not None and evidence_lease is not None:
                    with suppress(Exception):
                        evidence.try_admit_transport_confirmed_full(
                            evidence_lease,
                            segment_ordinal=(
                                publication.sequence if replay_presentation is None else None
                            ),
                            synthesis_attempt_id=item.synthesis_attempt_id,
                            transport_attempt_id=transport_attempt_id,
                            text=(confirmed_text if replay_presentation is None else None),
                        )
                self._context.record_assistant_delivery(
                    admission=admission,
                    ledger=self._ledger,
                    confirmation=confirmation,
                )
                partially_delivered_sequences.add(publication.sequence)
                delivered_chunk_counts[publication.sequence] = (
                    delivered_chunk_counts.get(publication.sequence, 0) + 1
                )
                last_delivered_chunk_ids[publication.sequence] = chunk.chunk_id
                pending_admissions.remove(admission)
            await asyncio.shield(prefetch_task)
            await asyncio.shield(producer_result)
            turn_counts = self._ledger.snapshot_turn_counts(turn_id)
            terminal_evidence.counts = turn_counts
            try:
                self._ledger.close_turn(turn_id)
            except BaseException:
                terminal_evidence.ledger_close_failed = True
                evidence = self._resolve_evidence_admission()
                if evidence is not None and evidence_lease is not None:
                    with suppress(Exception):
                        evidence.record_terminal_cause(
                            evidence_lease.terminal_cause,
                            TerminalReason.LEDGER_CLOSE_FAILED,
                        )
                raise
            evidence = self._resolve_evidence_admission()
            authoritative_response_settled = False
            if evidence is not None and evidence_lease is not None:
                with suppress(Exception):
                    settlement_disposition = evidence.settle_completed(
                        evidence_lease,
                        queued_chunk_count=turn_counts.queued_chunk_count,
                        started_chunk_count=turn_counts.started_chunk_count,
                        transport_confirmed_full_count=(turn_counts.transport_confirmed_full_count),
                        assistant_delivery_context_recorded=(
                            turn_counts.assistant_delivery_context_recorded
                        ),
                    )
                    if self._observe_terminal_settlement(
                        evidence,
                        evidence_lease,
                        settlement_disposition,
                    ):
                        authoritative_response_settled = True
                        self._observe_committed_conversation_context_snapshot(
                            terminal_evidence.committed_conversation_context_snapshot
                        )
            if (
                authoritative_response_settled
                and self._qualification_checkpoint_channel is not None
            ):
                await self._qualification_checkpoint_channel.emit(
                    "host_response_completed_before_shutdown"
                )
            self._notify_completion_advisory(
                turn_id=lifecycle_turn_id,
                turn_generation=lifecycle_generation,
            )
        except BaseException as error:
            interruption_projection_error: BaseException | None = None
            observer = self._observer
            if observer is not None:
                try:
                    observer(
                        "assistant_turn_interrupted",
                        {
                            "turnId": lifecycle_turn_id,
                            "turnGeneration": lifecycle_generation,
                        },
                    )
                except BaseException as candidate:
                    interruption_projection_error = candidate
            if not prefetch_task.done():
                prefetch_task.cancel()
            cleanup_errors = await self._cleanup_failed_turn(
                turn_id,
                pending_admissions=pending_admissions,
                cancel_inference=cancel_inference,
                evidence_lease=evidence_lease,
                terminal_evidence=terminal_evidence,
            )
            cleanup_errors.extend(
                await self._settle_prefetch_task(
                    prefetch_task,
                    primary_error=error,
                )
            )
            if interruption_projection_error is not None:
                cleanup_errors.append(interruption_projection_error)
            evidence = self._resolve_evidence_admission()
            counts = terminal_evidence.counts
            if evidence is not None and evidence_lease is not None and counts is not None:
                with suppress(Exception):
                    settlement_disposition = evidence.settle_terminal(
                        evidence_lease,
                        queued_chunk_count=counts.queued_chunk_count,
                        started_chunk_count=counts.started_chunk_count,
                        assistant_delivery_context_recorded=(
                            counts.assistant_delivery_context_recorded
                        ),
                    )
                    if self._observe_terminal_settlement(
                        evidence,
                        evidence_lease,
                        settlement_disposition,
                    ):
                        self._observe_committed_conversation_context_snapshot(
                            terminal_evidence.committed_conversation_context_snapshot
                        )
            failures = [error, *cleanup_errors]
            if all(_is_cancellation_only(failure) for failure in failures):
                raise asyncio.CancelledError from None
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "streaming response and cleanup failed",
                    failures,
                ) from None
            raise

    async def _prefetch_speech(
        self,
        *,
        producer_done: asyncio.Event,
        output: asyncio.Queue[_PrefetchedItem],
        slots: asyncio.Semaphore,
        replay_skip_counts: dict[int, int],
        evidence_lease: EvidenceTurnLease | None,
    ) -> None:
        while True:
            publication = await self._next_publication_or_done(producer_done)
            if publication is None:
                return
            if not publication.is_valid:
                raise asyncio.CancelledError
            synthesis_attempt_id = str(uuid4())
            try:
                synthesis = self._synthesizer.synthesize(
                    publication.text,
                    publication.turn_id,
                )
            except BaseException:
                admission = self._resolve_evidence_admission()
                if admission is not None and evidence_lease is not None:
                    with suppress(Exception):
                        admission.record_terminal_cause(
                            evidence_lease.terminal_cause,
                            TerminalReason.PROVIDER_FAILED,
                        )
                raise
            iteration_error: BaseException | None = None
            text_cursor = 0
            try:
                while True:
                    await slots.acquire()
                    slot_owned = True
                    try:
                        try:
                            chunk = await anext(synthesis)
                        except StopAsyncIteration:
                            break
                        if type(chunk) is not SpeechChunk:
                            raise TypeError("synthesizer must yield exact SpeechChunk values")
                        chunk = self._snapshot_speech_chunk(chunk)
                        if chunk.turn_id != publication.turn_id:
                            raise ValueError("speech chunk turn_id does not match publication")
                        if not publication.is_valid:
                            raise asyncio.CancelledError
                        text_offset = publication.text.find(chunk.text, text_cursor)
                        segment_text_offset_utf16 = None
                        if text_offset >= 0:
                            segment_text_offset_utf16 = sum(
                                2 if ord(character) > 0xFFFF else 1
                                for character in publication.text[:text_offset]
                            )
                            text_cursor = text_offset + len(chunk.text)
                        confirmed_prefix = replay_skip_counts.get(publication.sequence, 0)
                        if confirmed_prefix > 0:
                            replay_skip_counts[publication.sequence] = confirmed_prefix - 1
                            continue
                        await output.put(
                            _PrefetchedSpeech(
                                publication=publication,
                                chunk=chunk,
                                segment_text_offset_utf16=segment_text_offset_utf16,
                                synthesis_attempt_id=synthesis_attempt_id,
                            )
                        )
                        slot_owned = False
                    finally:
                        if slot_owned:
                            slots.release()
            except BaseException as error:
                iteration_error = error
            close_error = await self._close_stream(synthesis)
            if iteration_error is None and not publication.is_valid:
                iteration_error = asyncio.CancelledError()
            if (
                iteration_error is not None and not _is_cancellation_only(iteration_error)
            ) or close_error is not None:
                admission = self._resolve_evidence_admission()
                if admission is not None and evidence_lease is not None:
                    with suppress(Exception):
                        admission.record_terminal_cause(
                            evidence_lease.terminal_cause,
                            TerminalReason.PROVIDER_FAILED,
                        )
            self._raise_iteration_or_close_error(
                "synthesis iteration and close failed",
                iteration_error,
                close_error,
            )
            if not publication.is_valid:
                raise asyncio.CancelledError
            await output.put(_PrefetchedPublicationEnd(publication=publication))

    @staticmethod
    async def _next_prefetched_speech(
        output: asyncio.Queue[_PrefetchedItem],
        producer: asyncio.Task[None],
    ) -> _PrefetchedItem | None:
        if not output.empty():
            return output.get_nowait()
        if producer.done():
            producer.result()
            return None
        item_task = asyncio.create_task(output.get())
        try:
            done, _ = await asyncio.wait(
                (item_task, producer),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if item_task in done:
                return item_task.result()
            producer.result()
            return None
        finally:
            if not item_task.done():
                item_task.cancel()
            await asyncio.gather(item_task, return_exceptions=True)

    async def _settle_prefetch_task(
        self,
        task: asyncio.Task[None],
        *,
        primary_error: BaseException,
    ) -> list[BaseException]:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except asyncio.CancelledError:
            return []
        except BaseException as error:
            if error is primary_error:
                return []
            return [error]
        return []

    async def _next_publication_or_done(
        self,
        producer_done: asyncio.Event,
    ) -> ForegroundPublication | None:
        if producer_done.is_set() and self._foreground.pending_publication_count == 0:
            return None
        publication_task = asyncio.create_task(self._foreground.next_publication())
        done_task = asyncio.create_task(producer_done.wait())
        try:
            done, _ = await asyncio.wait(
                (publication_task, done_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if publication_task in done:
                return publication_task.result()
            if self._foreground.pending_publication_count:
                return await publication_task
            return None
        finally:
            for task in (publication_task, done_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                publication_task,
                done_task,
                return_exceptions=True,
            )

    def _revoke_active_locked(
        self,
        *,
        preserve_resumable: bool | None = None,
        terminal_reason: TerminalReason | None = None,
    ) -> None:
        """Synchronously revoke publication and state authority, then own cleanup."""

        consumer = self._active_consumer
        lease = self._active_lease
        if consumer is None or lease is None:
            return
        assert self._active_turn_id is not None
        assert self._active_generated_segments is not None
        assert self._active_delivered_sequences is not None
        assert self._active_delivered_chunk_counts is not None
        assert self._active_replay_presentations is not None
        policy = self._active_preserve_resumable_on_cancel
        if policy is None:
            raise RuntimeError("active speech is missing its resumability policy")
        if preserve_resumable is None:
            preserve_resumable = policy
        evidence_lease = self._active_evidence_lease
        evidence = self._resolve_evidence_admission()
        if terminal_reason is not None and evidence is not None and evidence_lease is not None:
            with suppress(Exception):
                evidence.record_terminal_cause(
                    evidence_lease.terminal_cause,
                    terminal_reason,
                )
        if terminal_reason is not None:
            self._qualification_owner_context.record_cancellation(terminal_reason)
        owner = _RevokedForegroundOwner(
            consumer=consumer,
            lease=lease,
            turn_id=self._active_turn_id,
            generated=self._active_generated_segments,
            delivered=self._active_delivered_sequences,
            delivered_chunks=self._active_delivered_chunk_counts,
            replay_presentations=self._active_replay_presentations,
            revision=self._authority_revision,
            preserve_resumable=preserve_resumable,
            replay_identity=self._active_replay_identity,
        )
        self._active_consumer = None
        self._active_lease = None
        self._active_turn_id = None
        self._active_generated_segments = None
        self._active_delivered_sequences = None
        self._active_delivered_chunk_counts = None
        self._active_replay_presentations = None
        self._active_preserve_resumable_on_cancel = None
        self._active_replay_identity = None
        self._active_evidence_lease = None
        revoked = self._foreground.revoke_if_current(lease)
        if not consumer.done():
            consumer.cancel()
        if preserve_resumable:
            self._resumable_cleanup_count += 1
        self._track_cleanup(asyncio.create_task(self._settle_revoked_owner(owner, revoked)))

    def _track_cleanup(self, cleanup: asyncio.Task[None]) -> None:
        self._cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._cleanup_tasks.discard)

    async def _settle_owned_cleanups(self) -> None:
        async with self._cleanup_settlement_lock:
            caller_cancellation: asyncio.CancelledError | None = None
            errors: list[BaseException] = []
            attempted = False
            while self._cleanup_tasks:
                attempted = True
                cleanups = tuple(self._cleanup_tasks)
                try:
                    results = await asyncio.shield(
                        asyncio.gather(*cleanups, return_exceptions=True)
                    )
                except asyncio.CancelledError as error:
                    caller_cancellation = error
                    continue
                errors.extend(result for result in results if isinstance(result, BaseException))
            if caller_cancellation is not None:
                if attempted:
                    self._qualification_owner_context.record_foreground_cleanup(
                        succeeded=not errors
                    )
                if errors:
                    raise BaseExceptionGroup(
                        "caller cancellation and foreground cleanup failure",
                        [caller_cancellation, *errors],
                    ) from None
                raise caller_cancellation
            if attempted:
                self._qualification_owner_context.record_foreground_cleanup(succeeded=not errors)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("foreground cleanup failed", errors)

    async def _settle_revoked_owner(
        self,
        owner: _RevokedForegroundOwner,
        revoked: tuple[asyncio.Task[None], str] | None,
    ) -> None:
        errors: list[BaseException] = []
        if revoked is not None:
            task, turn_id = revoked
            try:
                await self._foreground.settle_revoked(task, turn_id=turn_id)
            except BaseException as error:
                errors.append(error)
        try:
            await self._settle_consumer(owner.consumer)
        except BaseException as error:
            errors.append(error)
        async with self._lifecycle_lock:
            if owner.preserve_resumable:
                if self._resumable_cleanup_count <= 0:
                    raise RuntimeError("resumable cleanup barrier underflow")
                try:
                    if self._authority_revision == owner.revision:
                        self._capture_resumable_owner(owner)
                finally:
                    self._resumable_cleanup_count -= 1
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("foreground and speech-consumer cleanup failed", errors)

    def _capture_resumable_owner(self, owner: _RevokedForegroundOwner) -> None:
        if self._closed:
            self._resumable_speech = None
            return
        suffix: list[_ReplaySegment] = []
        for publication in owner.generated:
            if publication.sequence in owner.delivered:
                continue
            source = owner.replay_presentations.get(publication.sequence)
            confirmed_chunk_count = owner.delivered_chunks.get(publication.sequence, 0)
            if source is not None:
                confirmed_chunk_count += source.confirmed_chunk_count
            suffix.append(
                _ReplaySegment(
                    text=publication.text,
                    confirmed_chunk_count=confirmed_chunk_count,
                    presentation_turn_id=(
                        publication.turn_id if source is None else source.presentation_turn_id
                    ),
                    presentation_turn_generation=(
                        publication.generation
                        if source is None
                        else source.presentation_turn_generation
                    ),
                    presentation_segment_id=(
                        publication.segment_id if source is None else source.presentation_segment_id
                    ),
                )
            )
        self._resumable_speech = (
            _ResumableSpeech(
                source_turn_id=owner.turn_id,
                segments=tuple(suffix),
                replay_of_evidence_turn_id=(
                    None
                    if owner.replay_identity is None
                    else owner.replay_identity.replay_of_evidence_turn_id
                ),
                replay_generation=(
                    None
                    if owner.replay_identity is None
                    else owner.replay_identity.replay_generation
                ),
                source_binding_id=(
                    None
                    if owner.replay_identity is None
                    else owner.replay_identity.source_binding_id
                ),
                source_binding_generation=(
                    None
                    if owner.replay_identity is None
                    else owner.replay_identity.source_binding_generation
                ),
                source_logical_session_id=(
                    None
                    if owner.replay_identity is None
                    else owner.replay_identity.source_logical_session_id
                ),
            )
            if suffix
            else None
        )

    async def _settle_consumer(
        self,
        consumer: asyncio.Task[None],
        *,
        ignored_cancellation_count: int = 0,
    ) -> None:
        caller_cancellation: asyncio.CancelledError | None = None
        deadline = asyncio.get_running_loop().time() + self._cleanup_timeout_seconds
        while not consumer.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                timeout = TimeoutError("speech consumer cleanup exceeded finite timeout")
                if caller_cancellation is not None:
                    raise BaseExceptionGroup(
                        "caller cancellation and speech consumer cleanup timeout",
                        [caller_cancellation, timeout],
                    ) from None
                raise timeout
            try:
                done, _ = await asyncio.wait({consumer}, timeout=remaining)
                if not done:
                    continue
            except asyncio.CancelledError as error:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling() > ignored_cancellation_count:
                    caller_cancellation = error
                    continue
                if consumer.done():
                    break
            except BaseException:
                break
        if consumer.done() and not consumer.cancelled():
            consumer.result()
        if caller_cancellation is not None:
            raise caller_cancellation
        caller = asyncio.current_task()
        if caller is not None and caller.cancelling() > ignored_cancellation_count:
            raise asyncio.CancelledError

    async def _cleanup_failed_turn(
        self,
        turn_id: str,
        *,
        pending_admissions: list[AssistantTextAdmission],
        cancel_inference: bool,
        evidence_lease: EvidenceTurnLease | None,
        terminal_evidence: _TerminalEvidenceState,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        evidence = self._resolve_evidence_admission()
        if evidence is not None and evidence_lease is not None:
            with suppress(Exception):
                terminal_evidence.counts = self._ledger.snapshot_turn_counts(turn_id)
        # Release model-visible admissions and retained PCM before invoking
        # provider-controlled cleanup, which may resist cancellation.
        try:
            self._ledger.cancel_pending(turn_id)
        except BaseException as error:
            errors.append(error)
        for admission in tuple(pending_admissions):
            try:
                self._context.discard_assistant_text(admission)
            except BaseException as error:
                errors.append(error)
        try:
            self._ledger.close_turn(turn_id)
        except BaseException as error:
            terminal_evidence.ledger_close_failed = True
            if evidence is not None and evidence_lease is not None:
                with suppress(Exception):
                    evidence.record_terminal_cause(
                        evidence_lease.terminal_cause,
                        TerminalReason.LEDGER_CLOSE_FAILED,
                    )
            errors.append(error)

        operations = [
            ("playback", self._playback.cancel),
            ("synthesis", self._synthesizer.cancel),
        ]
        if cancel_inference:
            operations.insert(0, ("inference", self._inference.cancel))
        cleanup_tasks = {
            asyncio.create_task(operation(turn_id)): name for name, operation in operations
        }
        self._provider_cleanup_tasks.update(cleanup_tasks)
        done, pending = await asyncio.wait(
            cleanup_tasks,
            timeout=self._cleanup_timeout_seconds,
        )
        for cleanup in done:
            self._provider_cleanup_tasks.discard(cleanup)
            try:
                cleanup.result()
            except BaseException as error:
                errors.append(error)
        for cleanup in pending:
            cleanup.cancel()
            cleanup.add_done_callback(self._provider_cleanup_done)
            errors.append(
                TimeoutError(
                    f"{cleanup_tasks[cleanup]} provider cancellation exceeded finite timeout"
                )
            )
        # Iterators are closed by the producer/consumer task that owns their
        # active ``anext()`` operation. Closing them concurrently from cleanup
        # can itself fail or race provider code.
        return errors

    async def _close_stream(self, stream: object) -> BaseException | None:
        if not isinstance(stream, _AsyncClosable):
            return None
        cleanup = asyncio.create_task(stream.aclose())
        self._provider_cleanup_tasks.add(cleanup)
        try:
            done, _ = await asyncio.wait(
                {cleanup},
                timeout=self._cleanup_timeout_seconds,
            )
            if not done:
                cleanup.cancel()
                cleanup.add_done_callback(self._provider_cleanup_done)
                return TimeoutError("provider iterator cleanup exceeded finite timeout")
            self._provider_cleanup_tasks.discard(cleanup)
            cleanup.result()
        except BaseException as error:
            self._provider_cleanup_tasks.discard(cleanup)
            return error
        return None

    @staticmethod
    async def _segments_stream(
        segments: tuple[_ReplaySegment, ...],
    ) -> AsyncIterator[str]:
        for segment in segments:
            await asyncio.sleep(0)
            yield segment.text

    @staticmethod
    async def _announcement_stream(text: str) -> AsyncIterator[str]:
        yield text

    @staticmethod
    def _prompt_updates(
        decisions: tuple[UpdateDecision, ...],
    ) -> tuple[ConversationPromptUpdate, ...]:
        if type(decisions) is not tuple:
            raise TypeError("updates must be an exact tuple")
        if len(decisions) > _MAX_PROMPT_UPDATES:
            raise ValueError("update capacity exceeded")
        updates: list[ConversationPromptUpdate] = []
        previous_sequence = 0
        for candidate in decisions:
            if type(candidate) is not UpdateDecision:
                raise TypeError("updates must contain exact UpdateDecision values")
            decision = UpdateDecision(
                sequence=candidate.sequence,
                completion=candidate.completion,
                kind=candidate.kind,
                text=candidate.text,
            )
            if decision.kind != UpdateDecisionKind.MENTION_NEXT.value:
                raise ValueError("only mention_next decisions may enter inference")
            if decision.sequence <= previous_sequence:
                raise ValueError("update sequence must be strictly increasing")
            assert decision.text is not None
            updates.append(
                ConversationPromptUpdate(
                    sequence=decision.sequence,
                    task_id=decision.task_id,
                    status=decision.status,
                    text=decision.text,
                )
            )
            previous_sequence = decision.sequence
        return tuple(updates)

    def _raise_if_provider_cleanup_pending(self) -> None:
        for cleanup in tuple(self._provider_cleanup_tasks):
            if cleanup.done():
                self._provider_cleanup_done(cleanup)
        if self._provider_cleanup_tasks:
            raise RuntimeError("provider cleanup is still pending")

    def _provider_cleanup_done(self, cleanup: asyncio.Task[None]) -> None:
        self._provider_cleanup_tasks.discard(cleanup)
        with suppress(BaseException):
            cleanup.result()

    @staticmethod
    def _exception_contains(container: BaseException, target: BaseException) -> bool:
        if container is target:
            return True
        if isinstance(container, BaseExceptionGroup):
            return any(
                StreamingSpeechLoop._exception_contains(candidate, target)
                for candidate in container.exceptions
            )
        return False

    @staticmethod
    def _validity_probe(
        publication: ForegroundPublication,
    ) -> Callable[[], bool]:
        def is_valid() -> bool:
            return publication.is_valid

        return is_valid

    @staticmethod
    def _raise_iteration_or_close_error(
        message: str,
        iteration_error: BaseException | None,
        close_error: BaseException | None,
    ) -> None:
        if iteration_error is not None and close_error is not None:
            raise BaseExceptionGroup(message, [iteration_error, close_error]) from None
        if iteration_error is not None:
            raise iteration_error
        if close_error is not None:
            raise close_error

    @staticmethod
    def _snapshot_speech_chunk(chunk: SpeechChunk) -> SpeechChunk:
        """Revalidate and detach an exact chunk before invoking any field methods."""

        if type(chunk.turn_id) is not str:
            raise TypeError("speech chunk turn_id must be an exact built-in string")
        if type(chunk.chunk_id) is not str:
            raise TypeError("speech chunk chunk_id must be an exact built-in string")
        if type(chunk.text) is not str:
            raise TypeError("speech chunk text must be an exact built-in string")
        audio = chunk.audio
        if type(audio) is not AudioFrame:
            raise TypeError("speech chunk audio must be an exact AudioFrame")
        if type(audio.pcm) is not bytes:
            raise TypeError("speech chunk PCM must be exact bytes")
        if type(audio.sample_rate_hz) is not int:
            raise TypeError("speech chunk sample_rate_hz must be an exact integer")
        if type(audio.channels) is not int:
            raise TypeError("speech chunk channels must be an exact integer")
        return SpeechChunk(
            turn_id=chunk.turn_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            audio=AudioFrame(
                pcm=audio.pcm,
                sample_rate_hz=audio.sample_rate_hz,
                channels=audio.channels,
            ),
            word_timings=chunk.word_timings,
            timing_source=chunk.timing_source,
        )
