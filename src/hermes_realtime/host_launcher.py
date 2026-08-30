"""Full loopback Hermes host composition with real task and approval authority."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import signal
import ssl
import stat
import sys
import unicodedata
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

from hermes_realtime._qualification import (
    _current_qualification_full_host_dependencies,
    _new_qualification_checkpoint_channel,
    _qualification_checkpoint_channel_scope,
    _QualificationCheckpointChannelV1,
)
from hermes_realtime.client import (
    BrowserBindingSnapshot,
    BrowserClientRuntime,
    BrowserEventProjection,
    BrowserEvidenceConsentOperation,
    BrowserEvidenceControlResponse,
    BrowserEvidenceRevokeOperation,
    BrowserModelCatalog,
    BrowserModelConfiguration,
    BrowserSelectableModel,
    BrowserSpeechRuntime,
    LoopbackPeerAuthorizer,
    TailnetPeerAuthorizer,
    TailscaleCliWhoIsResolver,
)
from hermes_realtime.conversation import (
    DEFAULT_MAX_RESPONSE_SEGMENTS,
    ConversationContextStore,
    ConversationInferenceRequest,
    ConversationMessage,
    ConversationRole,
    ConversationSessionWorker,
    ConversationTaskCommandRouter,
    ConversationTaskController,
    ConversationUpdateDirector,
    ConversationUpdateExecutor,
    ConversationWorkControlSurface,
    ForegroundTurnCoordinator,
    ReconnectSafeConversationWorker,
    StreamingSpeechLoop,
    UpdateDirective,
    UpdatePolicyInput,
)
from hermes_realtime.conversation.knowledge import KnowledgePrefetchCoordinator
from hermes_realtime.evidence import (
    CaptureState,
    ConsentDisposition,
    ControlError,
    EvidenceConsentRequestV1,
    EvidenceRevokeRequestV1,
    FullPurgeV1,
    PurgeDisposition,
    RecoveryDisposition,
    RevokeDisposition,
    SentinelState,
)
from hermes_realtime.evidence.runtime import (
    HostEvidenceRuntimeV1,
    _validate_writer_transport_v1,
    _WriterTransportV1,
)
from hermes_realtime.evidence.sqlite_spool import (
    SQLiteEvidenceSpool,
    SQLiteEvidenceWriterDaemonV1,
)
from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.launcher import (
    ConversationOnlyTaskSession,
    ConversationOnlyUpdatePolicy,
    LocalBrowserLauncher,
)
from hermes_realtime.livekit import (
    MAX_SPEECH_CHUNK_DURATION_SECONDS,
    LiveKitConnection,
    LiveKitConversationWorker,
    LiveKitPCMDeliveryConfirmation,
    LiveKitRoomPeer,
    LiveKitSpeechPlayback,
    ReconnectSafeLiveKitAudioPublisher,
)
from hermes_realtime.network_origin import canonical_remote_hostname
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    _new_observation_channel,
    _ProductionObservationRecorderV1,
)
from hermes_realtime.protocol import (
    ControlCancelAcknowledgedEvent,
    ControlCancelEvent,
    WorkCompletedEvent,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchRequestedEvent,
)
from hermes_realtime.providers import (
    CodexAppServerStreamingInference,
    EdgeTtsSynthesizer,
    FasterWhisperTranscriber,
    KokoroSynthesizer,
    MoonshineStreamingTranscriber,
    OllamaStreamingInference,
    PlaybackEchoGuard,
    SileroSpeechPresenceVerifier,
    WebRtcVoiceActivityDetector,
)
from hermes_realtime.providers.codex_app_server import (
    CodexModelConfiguration,
    CodexTokenUsage,
    HermesRepresentativeContext,
)
from hermes_realtime.providers.current_facts import (
    ConsentBoundCurrentFactLookup,
    CurrentFactLookup,
    PublicRssCurrentFactLookup,
)
from hermes_realtime.search_egress import SearchEgressAuthority
from hermes_realtime.speech import (
    DeliveredSpeechLedger,
    SpeechChunk,
    SpeechPlayback,
)

_PublicValue = str | int | bool | None
_LIVEKIT_CONFIRMATION_MARGIN_SECONDS = 10.0
_LIVEKIT_CONFIRMATION_TIMEOUT_SECONDS = (
    MAX_SPEECH_CHUNK_DURATION_SECONDS + _LIVEKIT_CONFIRMATION_MARGIN_SECONDS
)
_LIVEKIT_PUBLICATION_TIMEOUT_SECONDS = _LIVEKIT_CONFIRMATION_TIMEOUT_SECONDS
_LEGACY_SPEECH_END_FRAMES = 40
_NATURAL_V1_SPEECH_END_FRAMES = 70
_MAX_PUBLIC_TOKEN_COUNT = (1 << 53) - 1
_LOGGER = logging.getLogger(__name__)


class _HostEvidenceConsentDependenciesV1:
    """Exact pre-publication factory for the host consent writer transport."""

    def __init__(
        self,
        *,
        writer_transport_factory: Callable[[HostEvidenceRuntimeV1], _WriterTransportV1],
    ) -> None:
        if not callable(writer_transport_factory):
            raise TypeError("writer transport factory must be callable")
        self._writer_transport_factory = writer_transport_factory

    def create_writer_transport(self, runtime: HostEvidenceRuntimeV1) -> _WriterTransportV1:
        if type(runtime) is not HostEvidenceRuntimeV1:
            raise TypeError("writer transport factory requires an exact host evidence runtime")
        transport = self._writer_transport_factory(runtime)
        _validate_writer_transport_v1(transport)
        return transport


class _HostEvidenceConsentGatewayV1:
    """One constructor-bound host bridge for authenticated browser consent ingress."""

    def __init__(
        self,
        *,
        runtime: HostEvidenceRuntimeV1,
        projection: BrowserEventProjection,
        live_generation: Callable[[], int | None],
        dependencies: _HostEvidenceConsentDependenciesV1,
    ) -> None:
        if type(runtime) is not HostEvidenceRuntimeV1:
            raise TypeError("runtime must be an exact HostEvidenceRuntimeV1")
        if type(projection) is not BrowserEventProjection:
            raise TypeError("projection must be an exact BrowserEventProjection")
        if not callable(live_generation):
            raise TypeError("live_generation must be callable")
        if type(dependencies) is not _HostEvidenceConsentDependenciesV1:
            raise TypeError("dependencies must be exact")
        self._runtime = runtime
        self._projection = projection
        self._live_generation = live_generation
        self._dependencies = dependencies

    def reserve(
        self,
        binding: BrowserBindingSnapshot,
        request: EvidenceConsentRequestV1,
    ) -> BrowserEvidenceConsentOperation:
        if type(binding) is not BrowserBindingSnapshot:
            raise TypeError("binding must be an exact BrowserBindingSnapshot")
        if type(request) is not EvidenceConsentRequestV1:
            raise TypeError("request must be an exact EvidenceConsentRequestV1")
        return _reserve_host_evidence_consent(
            runtime=self._runtime,
            projection=self._projection,
            live_generation=self._live_generation,
            binding=binding,
            request=request,
            dependencies=self._dependencies,
        )


def _evidence_disclosure_digest() -> str:
    disclosure = Path(__file__).with_name("evidence") / "disclosure_v1.txt"
    digest_input = b"realtime-evidence-consent-v1\0" + disclosure.read_bytes()
    return hashlib.sha256(digest_input).hexdigest()


def _reserve_host_evidence_consent(
    *,
    runtime: HostEvidenceRuntimeV1,
    projection: BrowserEventProjection,
    live_generation: Callable[[], int | None],
    binding: BrowserBindingSnapshot,
    request: EvidenceConsentRequestV1,
    dependencies: _HostEvidenceConsentDependenciesV1,
) -> BrowserEvidenceConsentOperation:
    """Reserve one host-owned consent operation while browser authority is locked."""

    if type(dependencies) is not _HostEvidenceConsentDependenciesV1:
        raise TypeError("host evidence consent dependencies must be exact")
    # Validate the fault/block injection before it can obtain browser, create,
    # scheduler, or projection authority.  A hostile factory result is therefore
    # indistinguishable from a clean pre-consent retry.
    transport = dependencies.create_writer_transport(runtime)
    (
        status_reservation,
        revoke_status_reservation,
        terminal_status_reservation,
        fault_status_reservation,
    ) = projection.reserve_capture_status_slots(4)
    reservations = (
        status_reservation,
        revoke_status_reservation,
        terminal_status_reservation,
        fault_status_reservation,
    )
    try:
        authority = runtime.reserve_browser_consent(
            binding_generation=binding.binding_generation,
            request=request,
            projection_reservation=status_reservation,
            validate_projection_reservation=projection.validate_capture_status_reservation,
            microphone_available=True,
            typed_available=True,
        )
    except Exception:
        projection.release_capture_status_reservations(reservations)
        raise
    lifecycle_reservations = (
        revoke_status_reservation,
        terminal_status_reservation,
        fault_status_reservation,
    )
    runtime.retain_pending_consent_projection_reservation(
        authority,
        status_reservation,
        projection.release_capture_status_reservations,
    )
    runtime.retain_lifecycle_status_reservations(
        lifecycle_reservations,
        projection.release_capture_status_reservations,
    )
    lifecycle_released = False
    completion_operation: asyncio.Task[BrowserEvidenceControlResponse] | None = None
    settlement_response: asyncio.Future[BrowserEvidenceControlResponse] | None = None
    final_response: BrowserEvidenceControlResponse | None = None
    timeout_response = BrowserEvidenceControlResponse(
        status=503,
        payload={
            "captureState": CaptureState.IDLE.value,
            "error": ControlError.CONTROL_TIMEOUT.value,
            "sequence": request.sequence,
        },
    )

    def release_failed_activation_lifecycle() -> None:
        """Return every non-published lifecycle slot exactly once after rollback."""

        nonlocal lifecycle_released
        if lifecycle_released:
            return
        lifecycle_released = True
        runtime.release_lifecycle_status_reservations(lifecycle_reservations)

    def publish_final(disposition: ConsentDisposition) -> BrowserEvidenceControlResponse:
        """Publish exactly one settled activation result after its owner is definitive."""

        nonlocal final_response
        if final_response is not None:
            return final_response
        sequence = request.sequence
        if disposition is not ConsentDisposition.CONSENT_ACTIVATED:
            # The runtime has already retired the exact pending authority before
            # this failure status is public.  Its retained projection owner
            # releases every unconsumed lifecycle slot first.
            release_failed_activation_lifecycle()
            state = CaptureState.FAULTED.value
            outcome = {"captureState": state, "error": "writer_unavailable", "sequence": sequence}
        else:
            state = CaptureState.ACTIVE.value
            outcome = {"captureState": state, "result": "consent_activated", "sequence": sequence}
        projection.publish_capture_status(
            status_reservation,
            {
                "available": True,
                "captureState": state,
                "consentVersion": request.consent_version,
                "disclosureDigest": request.disclosure_digest,
                "retentionHours": request.retention_hours,
            },
        )
        runtime.mark_pending_consent_projection_published(authority, status_reservation)
        final_response = BrowserEvidenceControlResponse(
            status=200 if disposition is ConsentDisposition.CONSENT_ACTIVATED else 503,
            payload=outcome,
        )
        return final_response

    async def complete_owned() -> BrowserEvidenceControlResponse:
        nonlocal settlement_response
        disposition = await runtime.activate_consent(
            authority,
            transport=transport,
            binding_is_current=lambda command: command.binding_generation == live_generation(),
            timeout_seconds=2.0,
        )
        if disposition is ConsentDisposition.CONTROL_TIMED_OUT:
            # Claim the exact retained runtime operation before returning the
            # public timeout.  This linearizes timeout/retry handling without
            # inventing a public "pending" result or redispatching consent.
            settlement = runtime.claim_consent_settlement_task(authority)
            if settlement_response is None:
                response = asyncio.get_running_loop().create_future()
                settlement_response = response

                def observe_settlement(
                    completed: asyncio.Task[ConsentDisposition],
                ) -> None:
                    if response.done():
                        return
                    try:
                        settled_disposition = completed.result()
                        response.set_result(publish_final(settled_disposition))
                    except BaseException as error:
                        response.set_exception(error)

                settlement.add_done_callback(observe_settlement)
            return timeout_response
        return publish_final(disposition)

    async def complete() -> BrowserEvidenceControlResponse:
        """Coalesce duplicate callers so rollback and status publication are one-shot."""

        nonlocal completion_operation
        settlement = settlement_response
        if settlement is not None:
            if settlement.done():
                return await asyncio.shield(settlement)
            return timeout_response
        operation = completion_operation
        if operation is None:
            operation = asyncio.create_task(
                complete_owned(),
                name="host-evidence-consent-complete",
            )
            completion_operation = operation
        return await asyncio.shield(operation)

    async def settlement() -> BrowserEvidenceControlResponse:
        response = settlement_response
        if response is None:
            raise RuntimeError("consent timeout settlement is unavailable")
        return await asyncio.shield(response)

    return BrowserEvidenceConsentOperation(complete=complete, settlement=settlement)


def _reserve_host_evidence_revoke(
    *,
    runtime: HostEvidenceRuntimeV1,
    projection: BrowserEventProjection,
    binding: BrowserBindingSnapshot,
    request: EvidenceRevokeRequestV1,
) -> BrowserEvidenceRevokeOperation:
    status_reservation = runtime.take_lifecycle_status_reservation()
    authority = runtime.reserve_browser_revoke(
        binding_generation=binding.binding_generation,
        request=request,
        projection_reservation=status_reservation,
        validate_projection_reservation=projection.validate_capture_status_reservation,
    )
    status = runtime.capture_status(disclosure_digest=_evidence_disclosure_digest())
    projection.publish_capture_status(
        status_reservation,
        {
            "available": status.available,
            "captureState": status.capture_state.value,
            "retentionHours": status.retention_hours,
            "consentVersion": status.consent_version,
            "disclosureDigest": status.disclosure_digest,
        },
    )

    async def complete() -> BrowserEvidenceControlResponse:
        disposition = await runtime.activate_revoke(authority, timeout_seconds=2.0)
        if disposition is RevokeDisposition.REVOKE_DURABLY_SCHEDULED:
            terminal_status_reservation = runtime.take_lifecycle_status_reservation()
            terminal_published = False

            def publish_terminal(terminal: RevokeDisposition) -> bool:
                nonlocal terminal_published
                if terminal not in {
                    RevokeDisposition.PURGE_COMPLETED,
                    RevokeDisposition.PURGE_FAILED,
                    RevokeDisposition.WRITER_FAULT,
                }:
                    return False
                if terminal_published:
                    return True
                terminal_status = runtime.capture_status(
                    disclosure_digest=_evidence_disclosure_digest()
                )
                projection.publish_capture_status(
                    terminal_status_reservation,
                    {
                        "available": terminal_status.available,
                        "captureState": terminal_status.capture_state.value,
                        "retentionHours": terminal_status.retention_hours,
                        "consentVersion": terminal_status.consent_version,
                        "disclosureDigest": terminal_status.disclosure_digest,
                    },
                )
                terminal_published = True
                return True

            def release_unpublished_terminal_projection() -> None:
                if not terminal_published:
                    projection.release_capture_status_reservations(
                        (terminal_status_reservation,)
                    )

            try:
                runtime.observe_revoke_terminal(
                    publish_terminal,
                    release_unpublished_terminal_projection,
                )
            except BaseException:
                release_unpublished_terminal_projection()
                raise
            return BrowserEvidenceControlResponse(
                status=202,
                payload={
                    "captureState": CaptureState.REVOKED_PURGING.value,
                    "result": "revoke_durably_scheduled",
                    "sequence": request.sequence,
                },
            )
        if disposition is RevokeDisposition.PURGE_COMPLETED:
            return BrowserEvidenceControlResponse(
                status=200,
                payload={
                    "captureState": CaptureState.IDLE.value,
                    "result": "purge_completed",
                    "sequence": request.sequence,
                },
            )
        if disposition is RevokeDisposition.PURGE_FAILED:
            return BrowserEvidenceControlResponse(
                status=503,
                payload={
                    "captureState": CaptureState.PURGE_FAILED.value,
                    "error": "purge_failed",
                    "sequence": request.sequence,
                },
            )
        if disposition is RevokeDisposition.WRITER_FAULT:
            return BrowserEvidenceControlResponse(
                status=503,
                payload={
                    "captureState": CaptureState.FAULTED.value,
                    "error": "revoke_not_durable",
                    "sequence": request.sequence,
                },
            )
        return BrowserEvidenceControlResponse(
            status=503,
            payload={
                "captureState": CaptureState.REVOKED_PURGING.value,
                "error": "control_timeout",
                "sequence": request.sequence,
            },
        )

    return BrowserEvidenceRevokeOperation(complete=complete)


_ADVISORY_CONVERSATION_EVENTS = frozenset(
    {
        "echo_barge_in_confirmed",
        "echo_suppressed",
        "barge_in_non_speech_suppressed",
        "barge_in_verifier_unavailable",
        "knowledge_timing",
        "transcript_echo_suppressed",
        "transcript_partial",
    }
)


class _SerializedSpeechPlayback:
    def __init__(self, delegate: SpeechPlayback, gate: asyncio.Lock) -> None:
        if not isinstance(gate, asyncio.Lock):
            raise TypeError("playback gate must be an asyncio Lock")
        self._delegate = delegate
        self._gate = gate

    async def play(self, chunk: SpeechChunk, *, is_valid: Callable[[], bool]) -> None:
        async with self._gate:
            await self._delegate.play(chunk, is_valid=is_valid)

    async def cancel(self, turn_id: str) -> None:
        await self._delegate.cancel(turn_id)


class _ReadinessCueTasks:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def create(self, operation: Coroutine[object, object, None], *, name: str) -> None:
        if self._closed:
            operation.close()
            return
        task = asyncio.create_task(operation, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._settle)

    def _settle(self, completed: asyncio.Task[None]) -> None:
        self._tasks.discard(completed)
        if completed.cancelled():
            return
        error = completed.exception()
        if error is not None:
            _LOGGER.error(
                "readiness cue playback failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class _AsyncClose(Protocol):
    async def close(self) -> None: ...


class _TaskSessionStarter(Protocol):
    async def start(self) -> None: ...


class _PreflightSynthesizer(Protocol):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]: ...


class _NonOwningHermesApiTaskSession:
    """Delegate task traffic while reserving API-session close for the host owner."""

    def __init__(self, session: HermesApiTaskSession) -> None:
        if type(session) is not HermesApiTaskSession:
            raise TypeError("host task session must be an exact Hermes API session")
        self._session = session

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        try:
            return await self._session.dispatch(request)
        except BaseException:
            _LOGGER.exception("Hermes background dispatch failed before acknowledgement")
            raise

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        return await self._session.cancel(request)

    async def next_update(self) -> WorkCompletedEvent:
        return await self._session.next_update()

    async def close(self) -> None:
        """Let the controller quiesce without closing its host-owned API dependency."""


class _FullHostWorkCloseOwner:
    """Close foreground routing and work authority in strict dependency order."""

    def __init__(
        self,
        *,
        inference: _AsyncClose,
        surface: _AsyncClose,
        task_controller: _AsyncClose,
        task_session: _AsyncClose,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
    ) -> None:
        owners = (inference, surface, task_controller, task_session)
        if any(not callable(getattr(owner, "close", None)) for owner in owners):
            raise TypeError("full-host work close owners must provide close()")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        self._owners = owners
        self._closed = [False] * len(owners)
        self._close_operation: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._production_observation_recorder = production_observation_recorder

    async def close(self) -> None:
        async with self._lock:
            operation = self._close_operation
            if operation is None or (
                operation.done() and (operation.cancelled() or operation.exception() is not None)
            ):
                operation = asyncio.create_task(
                    self._close_owned(),
                    name="full-host-work-routing-close",
                )
                self._close_operation = operation
        await asyncio.shield(operation)

    async def _close_owned(self) -> None:
        errors: list[BaseException] = []
        for index, owner in enumerate(self._owners):
            if self._closed[index]:
                continue
            try:
                await owner.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._closed[index] = True
        if len(errors) == 1:
            self._record_close_result(CloseResultV1.FAILED)
            raise errors[0]
        if errors:
            self._record_close_result(CloseResultV1.FAILED)
            raise BaseExceptionGroup("full-host work close failed", errors)
        self._record_close_result(CloseResultV1.SUCCEEDED)

    def _record_close_result(self, result: CloseResultV1) -> None:
        recorder = self._production_observation_recorder
        if recorder is not None:
            recorder.record_close_stage(stage=CloseStageV1.HOST_WORK, result=result)


class _QualificationNoTaskCloseOwner:
    """Close exact no-task dependencies while preserving real command routing."""

    def __init__(self, *owners: _AsyncClose) -> None:
        if not owners or any(not callable(getattr(owner, "close", None)) for owner in owners):
            raise TypeError("qualification no-task close owners must provide close()")
        self._owners = owners
        self._closed = [False] * len(owners)

    async def close(self) -> None:
        errors: list[BaseException] = []
        for index, owner in enumerate(self._owners):
            if self._closed[index]:
                continue
            try:
                await owner.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._closed[index] = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("qualification work close failed", errors)


def _readiness_cue_chunk(
    template: SpeechChunk,
    *,
    generation: int,
    media_incarnation: int,
) -> SpeechChunk:
    if type(template) is not SpeechChunk:
        raise TypeError("readiness cue template must be an exact SpeechChunk")
    if type(generation) is not int or type(media_incarnation) is not int:
        raise TypeError("readiness cue authority values must be exact integers")
    if generation <= 0 or media_incarnation <= 0:
        raise ValueError("readiness cue authority values must be positive")
    identity = f"readiness_{generation}_{media_incarnation}"
    return SpeechChunk(
        turn_id=identity,
        chunk_id=identity,
        text=template.text,
        audio=template.audio,
        word_timings=template.word_timings,
        timing_source=template.timing_source,
    )


class _ReadinessCueAuthority:
    def __init__(self) -> None:
        self._owner: tuple[int, int] | None = None

    def admit(self, generation: int, media_incarnation: int) -> bool:
        if type(generation) is not int or type(media_incarnation) is not int:
            raise TypeError("readiness cue authority values must be exact integers")
        if generation <= 0 or media_incarnation <= 0:
            raise ValueError("readiness cue authority values must be positive")
        owner = self._owner
        if owner is not None:
            if generation < owner[0]:
                return False
            if generation == owner[0] and media_incarnation <= owner[1]:
                return False
        self._owner = (generation, media_incarnation)
        return True


class _SessionTokenUsageAccumulator:
    def __init__(self) -> None:
        self._turns: dict[str, CodexTokenUsage] = {}

    def observe(self, turn_id: str, usage: CodexTokenUsage) -> dict[str, _PublicValue] | None:
        if turn_id == "turn_preflight":
            return None
        self._turns[turn_id] = usage
        return {
            "cachedInputTokens": self._total(lambda item: item.cached_input_tokens),
            "contextWindowTokens": usage.context_window_tokens,
            "inputTokens": self._total(lambda item: item.input_tokens),
            "outputTokens": self._total(lambda item: item.output_tokens),
            "reasoningOutputTokens": self._total(lambda item: item.reasoning_output_tokens),
            "totalTokens": self._total(lambda item: item.total_tokens),
        }

    def reset(self, participant_identity: str, generation: int) -> None:
        if type(participant_identity) is not str or not participant_identity:
            raise ValueError("session identity must be a non-empty exact string")
        if type(generation) is not int or generation <= 0:
            raise ValueError("session generation must be a positive exact integer")
        self._turns.clear()

    def _total(self, value: Callable[[CodexTokenUsage], int]) -> int:
        return min(_MAX_PUBLIC_TOKEN_COUNT, sum(value(item) for item in self._turns.values()))


def _voice_input_ready_data(
    generation: int,
    media_incarnation: int,
) -> dict[str, _PublicValue]:
    if type(generation) is not int or generation <= 0:
        raise ValueError("worker generation must be a positive exact integer")
    if type(media_incarnation) is not int or not 1 <= media_incarnation <= (1 << 53) - 1:
        raise ValueError("media incarnation must be a positive browser-safe exact integer")
    return {"generation": generation, "mediaIncarnation": media_incarnation}


def _publish_voice_input_ready(
    projection: BrowserEventProjection,
    generation: int,
    media_incarnation: int | None,
) -> None:
    if media_incarnation is not None:
        projection.publish_advisory(
            "voice_input_ready",
            _voice_input_ready_data(generation, media_incarnation),
        )


def _project_conversation_observation(
    projection: BrowserEventProjection,
    kind: str,
    data: dict[str, _PublicValue],
) -> None:
    if kind in _ADVISORY_CONVERSATION_EVENTS:
        projection.publish_advisory(kind, data)
    else:
        projection.publish(kind, data)


_SpeechRendererAuthority = tuple[str, int, str, str]
_RendererPresentationAuthority = tuple[str, int]


def _renderer_authority_ends(
    kind: str,
    data: dict[str, _PublicValue],
    presentation: _RendererPresentationAuthority | None,
) -> bool:
    if kind in {"interrupt_requested", "session_stopped"}:
        return True
    if kind != "assistant_turn_completed":
        return False
    turn_id = data.get("turnId")
    turn_generation = data.get("turnGeneration")
    return (
        presentation is not None
        and isinstance(turn_id, str)
        and isinstance(turn_generation, int)
        and not isinstance(turn_generation, bool)
        and presentation == (turn_id, turn_generation)
    )


def _renderer_yield_event_data(
    kind: str,
    data: dict[str, _PublicValue],
    authority: _SpeechRendererAuthority | None,
) -> dict[str, _PublicValue]:
    if kind not in {"interrupt_requested", "barge_in_non_speech_suppressed"}:
        return data
    if authority is None:
        return {}
    turn_id, turn_generation, chunk_id, stream_id = authority
    if kind == "interrupt_requested" and data.get("turnId") != turn_id:
        return {}
    return {
        "turnId": turn_id,
        "turnGeneration": turn_generation,
        "chunkId": chunk_id,
        "streamId": stream_id,
    }


def _default_tts_provider() -> str:
    return "kokoro" if sys.platform == "win32" else "edge"


def _build_browser_speech_runtime(
    *,
    stt_provider: str,
    whisper_model: str,
    moonshine_model_tier: str,
    tts_provider: str,
    edge_voice: str,
) -> BrowserSpeechRuntime:
    if stt_provider == "moonshine":
        stt_model = f"moonshine-v2-{moonshine_model_tier}"
    elif stt_provider == "faster-whisper":
        stt_model = whisper_model
    else:
        raise ValueError("stt_provider must be exactly 'moonshine' or 'faster-whisper'")
    if tts_provider == "kokoro":
        tts_model = "kokoro-v1.0.onnx"
    elif tts_provider == "edge":
        tts_model = edge_voice
    else:
        raise ValueError("tts_provider must be exactly 'kokoro' or 'edge'")
    return BrowserSpeechRuntime(
        stt_provider=stt_provider,
        stt_model=stt_model,
        tts_provider=tts_provider,
        tts_model=tts_model,
    )


def _validate_livekit_origin(value: str, *, remote_mode: bool) -> str:
    if type(value) is not str:
        raise TypeError("LiveKit URL must be an exact built-in string")
    if type(remote_mode) is not bool:
        raise TypeError("remote_mode must be an exact boolean")
    error = (
        "remote LiveKit URL must be an exact secure WebSocket origin"
        if remote_mode
        else "local LiveKit URL must be an exact loopback WebSocket origin"
    )
    try:
        if any(ord(character) < 33 or ord(character) > 126 for character in value) or "\\" in value:
            raise ValueError
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError(error) from None
    shared_invalid = (
        not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or bool(parsed.query)
        or bool(parsed.fragment)
    )
    if shared_invalid:
        raise ValueError(error)
    assert hostname is not None
    if remote_mode:
        if parsed.scheme != "wss":
            raise ValueError(error)
        try:
            hostname = canonical_remote_hostname(hostname, boundary="remote LiveKit URL")
        except ValueError:
            raise ValueError(error) from None
        bracketed = f"[{hostname}]" if ":" in hostname else hostname
        port_suffix = "" if port is None or port == 443 else f":{port}"
        return f"wss://{bracketed}{port_suffix}"
    if (
        shared_invalid
        or parsed.scheme != "ws"
        or hostname not in {"127.0.0.1", "::1", "localhost"}
        or port is None
    ):
        raise ValueError(error)
    bracketed = f"[{hostname}]" if ":" in hostname else hostname
    return f"ws://{bracketed}:{port}"


def _certificate_san_covers_hostname(
    subject_alt_names: object,
    hostname: str,
) -> bool:
    if not isinstance(subject_alt_names, (list, tuple)):
        return False
    try:
        expected_address = ipaddress.ip_address(hostname)
    except ValueError:
        expected_address = None
    for entry in subject_alt_names:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
        ):
            continue
        kind, value = entry
        if expected_address is None:
            if kind == "DNS" and value.lower() == hostname.lower():
                return True
            continue
        if kind != "IP Address":
            continue
        try:
            if ipaddress.ip_address(value) == expected_address:
                return True
        except ValueError:
            continue
    return False


def _build_server_ssl_context(
    *,
    remote_mode: bool,
    certificate_file: Path | None,
    private_key_file: Path | None,
    canonical_origin: str | None,
) -> ssl.SSLContext | None:
    if type(remote_mode) is not bool:
        raise TypeError("remote_mode must be an exact boolean")
    if not remote_mode:
        if certificate_file is not None or private_key_file is not None:
            raise ValueError("TLS files are accepted only with --remote")
        if canonical_origin is not None:
            raise ValueError("canonical origin is accepted only with --remote")
        return None
    if certificate_file is None or private_key_file is None:
        raise ValueError("remote mode requires both --tls-cert and --tls-key")
    if not isinstance(certificate_file, Path) or not isinstance(private_key_file, Path):
        raise TypeError("TLS files must be pathlib Paths")
    if type(canonical_origin) is not str:
        raise ValueError("remote mode requires an explicit canonical HTTPS origin")
    try:
        if (
            any(ord(character) < 33 or ord(character) > 126 for character in canonical_origin)
            or "\\" in canonical_origin
        ):
            raise ValueError
        parsed_origin = urlsplit(canonical_origin)
        origin_hostname = parsed_origin.hostname
        _ = parsed_origin.port
        if (
            parsed_origin.scheme != "https"
            or origin_hostname is None
            or parsed_origin.path not in {"", "/"}
            or parsed_origin.query
            or parsed_origin.fragment
            or parsed_origin.username is not None
            or parsed_origin.password is not None
        ):
            raise ValueError
        origin_hostname = canonical_remote_hostname(
            origin_hostname,
            boundary="canonical browser origin",
        )
    except ValueError:
        raise ValueError("remote mode requires an exact canonical HTTPS origin") from None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=certificate_file, keyfile=private_key_file)
    decoded_certificate = ssl._ssl._test_decode_cert(  # type: ignore[attr-defined]
        str(certificate_file)
    )
    subject_alt_names = decoded_certificate.get("subjectAltName", ())
    if not _certificate_san_covers_hostname(subject_alt_names, origin_hostname):
        raise ValueError("TLS certificate does not cover canonical origin hostname")
    return context


def _load_livekit_credentials(*, remote_mode: bool) -> tuple[str, str]:
    if type(remote_mode) is not bool:
        raise TypeError("remote_mode must be an exact boolean")
    if not remote_mode:
        return "devkey", "local" + "-" + ("x" * 32)
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("remote mode requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET")
    if api_key == "devkey" or api_secret == "local" + "-" + ("x" * 32):
        raise RuntimeError("remote mode rejects loopback development credentials")
    if len(api_key) > 256 or not 32 <= len(api_secret) <= 512:
        raise RuntimeError("remote LiveKit credentials are outside supported bounds")
    return api_key, api_secret


def _load_tailnet_authorizer(*, enabled: bool) -> TailnetPeerAuthorizer | None:
    if type(enabled) is not bool:
        raise TypeError("enabled must be an exact boolean")
    if not enabled:
        return None
    stable_id = os.environ.get("HERMES_REALTIME_TAILNET_NODE_STABLE_ID")
    if stable_id is None:
        raise RuntimeError("Tailnet launch requires configured node identity")
    try:
        TailnetPeerAuthorizer.validate_allowed_stable_id(stable_id)
    except (TypeError, ValueError):
        raise RuntimeError("Tailnet launch node identity is invalid") from None
    try:
        resolver = TailscaleCliWhoIsResolver()
    except RuntimeError:
        raise RuntimeError("Tailnet launch authorization is unavailable") from None
    return TailnetPeerAuthorizer(allowed_stable_id=stable_id, resolver=resolver)


def _parse_kokoro_pronunciation(value: str) -> tuple[str, str]:
    term, separator, spoken = value.partition("=")
    if (
        not separator
        or term != term.strip()
        or spoken != spoken.strip()
        or not 1 <= len(term) <= 64
        or not 1 <= len(spoken) <= 128
        or any(unicodedata.category(character).startswith("C") for character in term + spoken)
    ):
        raise argparse.ArgumentTypeError("pronunciation must be TERM=SPOKEN")
    return term, spoken


def _parse_kokoro_pronunciations(
    values: Sequence[tuple[str, str]] | None,
) -> dict[str, str]:
    if values is not None and len(values) > 64:
        raise ValueError("Kokoro pronunciations must contain at most 64 entries")
    pronunciations: dict[str, str] = {}
    normalized: set[str] = set()
    for term, spoken in values or ():
        folded = term.casefold()
        if folded in normalized:
            raise ValueError("Kokoro pronunciation terms must be case-insensitively unique")
        normalized.add(folded)
        pronunciations[term] = spoken
    return pronunciations


def _parse_kokoro_stream_chunk_chars(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stream chunk characters must be an integer") from exc
    if not 32 <= parsed <= 1024:
        raise argparse.ArgumentTypeError("stream chunk characters must be between 32 and 1024")
    return parsed


def _parse_kokoro_worker_python(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise argparse.ArgumentTypeError(
            "--kokoro-worker-python must be an existing absolute native Python executable"
        )
    return path.resolve()


def _encode_speech_timing_event(
    chunk: SpeechChunk,
    stream_identity: str,
    *,
    turn_generation: int,
    presentation_turn_id: str | None = None,
    segment_id: str | None = None,
    segment_text_offset_utf16: int = 0,
) -> dict[str, _PublicValue] | None:
    """Encode bounded advisory timings without exposing private controller handles."""

    if type(chunk) is not SpeechChunk:
        raise TypeError("chunk must be an exact SpeechChunk")
    if type(stream_identity) is not str or not stream_identity:
        raise ValueError("stream_identity must be a non-empty exact string")
    if type(turn_generation) is not int or turn_generation <= 0:
        raise ValueError("turn_generation must be a positive exact integer")
    if presentation_turn_id is not None and (
        type(presentation_turn_id) is not str or not presentation_turn_id
    ):
        raise ValueError("presentation_turn_id must be a non-empty exact string")
    if segment_id is not None and (type(segment_id) is not str or not segment_id):
        raise ValueError("segment_id must be a non-empty exact string")
    if type(segment_text_offset_utf16) is not int or segment_text_offset_utf16 < 0:
        raise ValueError("segment_text_offset_utf16 must be a non-negative exact integer")
    if not chunk.word_timings or chunk.timing_source is None:
        return None
    utf16_offsets = [0]
    for character in chunk.text:
        utf16_offsets.append(utf16_offsets[-1] + (2 if ord(character) > 0xFFFF else 1))
    encoded = ";".join(
        f"{segment_text_offset_utf16 + utf16_offsets[item.text_start]},"
        f"{segment_text_offset_utf16 + utf16_offsets[item.text_end]},"
        f"{item.start_sample},{item.end_sample}"
        for item in chunk.word_timings
    )
    if len(encoded) > 4096:
        return None
    return {
        "turnId": chunk.turn_id,
        "presentationTurnId": (
            chunk.turn_id if presentation_turn_id is None else presentation_turn_id
        ),
        "turnGeneration": turn_generation,
        "chunkId": chunk.chunk_id,
        "segmentId": chunk.chunk_id if segment_id is None else segment_id,
        "streamId": stream_identity,
        "sampleRate": chunk.audio.sample_rate_hz,
        "timingSource": chunk.timing_source,
        "timings": encoded,
    }


def _compose_public_search(
    *,
    operator_enabled: bool,
    knowledge_budget_seconds: float,
    knowledge_speculation: bool,
    knowledge_recovery: bool,
) -> tuple[
    SearchEgressAuthority,
    CurrentFactLookup | None,
    KnowledgePrefetchCoordinator | None,
]:
    if type(operator_enabled) is not bool:
        raise TypeError("operator_enabled must be an exact boolean")
    if type(knowledge_speculation) is not bool or type(knowledge_recovery) is not bool:
        raise TypeError("knowledge feature controls must be exact booleans")
    authority = SearchEgressAuthority(operator_enabled=operator_enabled)
    if not operator_enabled:
        return authority, None, None
    raw_lookup = PublicRssCurrentFactLookup(
        timeout_seconds=float(knowledge_budget_seconds),
        max_results=8,
        enrich_results=0,
    )
    lookup = ConsentBoundCurrentFactLookup(
        delegate=raw_lookup,
        authority=authority,
    )
    coordinator = (
        KnowledgePrefetchCoordinator(
            lookup=lookup,
            enabled=knowledge_speculation,
            recovery_enabled=knowledge_recovery,
        )
        if knowledge_speculation or knowledge_recovery
        else None
    )
    return authority, lookup, coordinator


def _build_streaming_inference(
    *,
    inference_provider: str,
    ollama_base_url: str,
    ollama_model: str,
    codex_model: str,
    codex_effort: str,
    codex_executable: str | None,
    current_fact_lookup: CurrentFactLookup | None = None,
    token_usage_observer: Callable[[str, CodexTokenUsage], None] | None = None,
    knowledge_timing_observer: Callable[[dict[str, _PublicValue]], None] | None = None,
    knowledge_coordinator: KnowledgePrefetchCoordinator | None = None,
    hermes_context: HermesRepresentativeContext | None = None,
) -> OllamaStreamingInference | CodexAppServerStreamingInference:
    if type(inference_provider) is not str:
        raise TypeError("inference_provider must be an exact built-in string")
    if inference_provider == "ollama":
        if hermes_context is not None:
            raise ValueError("Hermes representative context requires --inference-provider codex")
        return OllamaStreamingInference(
            base_url=ollama_base_url,
            model=ollama_model,
            request_timeout_seconds=60.0,
        )
    if inference_provider == "codex":
        knowledge_lookup: CurrentFactLookup | None
        if knowledge_coordinator is not None:
            knowledge_lookup = knowledge_coordinator.lookup
        else:
            knowledge_lookup = current_fact_lookup
        codex_options: dict[str, object] = {
            "model": codex_model,
            "effort": codex_effort,
            "codex_executable": codex_executable,
            "token_usage_observer": token_usage_observer,
            "current_fact_lookup": knowledge_lookup,
            "knowledge_coordinator": knowledge_coordinator,
            "knowledge_timing_observer": knowledge_timing_observer,
            "request_timeout_seconds": 60.0,
        }
        if hermes_context is not None:
            codex_options["hermes_context"] = hermes_context
        return CodexAppServerStreamingInference(**codex_options)  # type: ignore[arg-type]
    raise ValueError("inference_provider must be exactly 'ollama' or 'codex'")


def _bind_natural_work_tools(
    *,
    inference: OllamaStreamingInference | CodexAppServerStreamingInference,
    surface: ConversationWorkControlSurface,
    enabled: bool,
) -> None:
    if type(enabled) is not bool:
        raise TypeError("natural_work_tools must be an exact boolean")
    if not enabled:
        return
    if type(inference) is not CodexAppServerStreamingInference:
        raise TypeError("natural work tools require exact Codex app-server inference")
    if type(surface) is not ConversationWorkControlSurface:
        raise TypeError("natural work tools require an exact work-control surface")
    inference.bind_work_tools(surface)


async def _run_full_host_preflight(
    *,
    task_session: _TaskSessionStarter,
    inference: OllamaStreamingInference | CodexAppServerStreamingInference,
    synthesizer: _PreflightSynthesizer,
    surface: ConversationWorkControlSurface,
    natural_work_tools: bool,
) -> SpeechChunk:
    """Warm tool-less dependencies, then grant the exact Codex instance work authority."""

    await task_session.start()
    snapshot = ConversationInferenceRequest(
        revision=0,
        messages=(
            ConversationMessage(
                role=ConversationRole.USER.value,
                text="Reply with one short sentence confirming readiness.",
            ),
        ),
        active_tasks=(),
        updates=(),
    )
    segments = [segment async for segment in inference.stream(snapshot, turn_id="turn_preflight")]
    if not segments:
        raise RuntimeError("local inference preflight produced no speakable output")
    if type(synthesizer) is KokoroSynthesizer:
        await synthesizer.warm()
    tts_probe = [
        chunk
        async for chunk in synthesizer.synthesize(
            "Ready.",
            "turn_tts_preflight",
        )
    ]
    if len(tts_probe) != 1 or type(tts_probe[0]) is not SpeechChunk:
        raise RuntimeError("TTS preflight did not produce one playable chunk")
    _bind_natural_work_tools(
        inference=inference,
        surface=surface,
        enabled=natural_work_tools,
    )
    return tts_probe[0]


def _build_browser_model_configuration(
    *,
    inference_provider: str,
    ollama_model: str,
    codex_model: str,
    codex_effort: str,
) -> BrowserModelConfiguration:
    if inference_provider == "codex":
        return BrowserModelConfiguration(
            authentication="subscription",
            provider="openai-codex",
            transport="subscription-app-server",
            model=codex_model,
            effort=codex_effort,
            context_window_tokens=None,
            reports_token_usage=True,
        )
    if inference_provider == "ollama":
        return BrowserModelConfiguration(
            authentication="local",
            provider="ollama",
            transport="local-http",
            model=ollama_model,
            effort=None,
            context_window_tokens=None,
            reports_token_usage=False,
        )
    raise ValueError("inference_provider must be exactly 'ollama' or 'codex'")


def _build_browser_selectable_model_catalog(
    configuration: CodexModelConfiguration,
) -> BrowserModelCatalog:
    if type(configuration) is not CodexModelConfiguration:
        raise TypeError("configuration must be an exact CodexModelConfiguration")
    return BrowserModelCatalog(
        models=tuple(
            BrowserSelectableModel(
                model=model.model,
                display_name=model.display_name,
                description=model.description,
                supported_efforts=model.supported_efforts,
                default_effort=model.default_effort,
            )
            for model in configuration.models
        ),
        selected_model=configuration.selected_model,
        selected_effort=configuration.selected_effort,
    )


def _build_streaming_transcriber(
    *,
    stt_provider: str,
    whisper_model: str,
    moonshine_model_tier: str,
    moonshine_update_interval_seconds: float,
) -> FasterWhisperTranscriber | MoonshineStreamingTranscriber:
    if type(stt_provider) is not str:
        raise TypeError("stt_provider must be an exact built-in string")
    if stt_provider == "moonshine":
        return MoonshineStreamingTranscriber(
            language="en",
            model_tier=moonshine_model_tier,
            update_interval_seconds=moonshine_update_interval_seconds,
            sample_rate_hz=48_000,
            channels=1,
        )
    if stt_provider == "faster-whisper":
        return FasterWhisperTranscriber(
            model_size_or_path=whisper_model,
            device="cpu",
            compute_type="int8",
        )
    raise ValueError("stt_provider must be exactly 'moonshine' or 'faster-whisper'")


def _build_voice_activity_detector(
    *,
    conversation_profile: str,
    classify: Callable[[bytes, int], bool] | None = None,
) -> WebRtcVoiceActivityDetector:
    if type(conversation_profile) is not str:
        raise TypeError("conversation_profile must be an exact built-in string")
    if conversation_profile == "legacy":
        speech_end_frames = _LEGACY_SPEECH_END_FRAMES
    elif conversation_profile == "natural_v1":
        speech_end_frames = _NATURAL_V1_SPEECH_END_FRAMES
    else:
        raise ValueError("conversation_profile must be legacy or natural_v1")
    return WebRtcVoiceActivityDetector(
        speech_end_frames=speech_end_frames,
        classify=classify,
    )


def _build_synthesizer(
    *,
    tts_provider: str,
    edge_voice: str,
    kokoro_voice: str,
    kokoro_speed: float,
    kokoro_intra_op_threads: int,
    kokoro_pronunciations: Mapping[str, str] | None = None,
    kokoro_stream_chunk_chars: int = 400,
    kokoro_worker_python: Path | None = None,
) -> EdgeTtsSynthesizer | KokoroSynthesizer:
    if type(tts_provider) is not str:
        raise TypeError("tts_provider must be an exact built-in string")
    if tts_provider == "kokoro":
        return KokoroSynthesizer(
            voice=kokoro_voice,
            speed=kokoro_speed,
            intra_op_threads=kokoro_intra_op_threads,
            pronunciations=kokoro_pronunciations,
            stream_chunk_chars=kokoro_stream_chunk_chars,
            worker_python=kokoro_worker_python,
        )
    if tts_provider == "edge":
        return EdgeTtsSynthesizer(voice=edge_voice)
    raise ValueError("tts_provider must be exactly 'kokoro' or 'edge'")


def _load_api_bearer(env_file: Path | None) -> str:
    if env_file is None:
        bearer = os.environ.get("API_SERVER_KEY")
        if bearer is None:
            raise RuntimeError("API_SERVER_KEY or --hermes-env-file is required for the full host")
        return bearer
    if not isinstance(env_file, Path):
        raise TypeError("hermes env file must be a Path")
    matches: list[str] = []
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("API_SERVER_KEY="):
            matches.append(line.split("=", 1)[1].strip().strip('"').strip("'"))
    if len(matches) != 1 or len(matches[0]) < 32:
        raise RuntimeError("Hermes env file must contain one explicit strong API_SERVER_KEY")
    return matches[0]


def _load_hermes_context(context_file: Path | None) -> HermesRepresentativeContext | None:
    if context_file is None:
        return None
    if not isinstance(context_file, Path):
        raise TypeError("Hermes context file must be a Path")
    try:
        with context_file.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise RuntimeError("Hermes context file must be a regular file")
            raw = stream.read(4097)
    except OSError as error:
        raise RuntimeError("Hermes context file could not be opened") from error
    if len(raw) > 4096:
        raise RuntimeError("Hermes context file exceeds the 4096-byte input bound")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("Hermes context file must be UTF-8") from error

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("Hermes context file must contain unique-field JSON") from error
    if type(decoded) is not dict:
        raise RuntimeError("Hermes context must be one JSON object")
    allowed = {"version", "identity", "persona", "user_preferences", "location"}
    version = decoded.get("version")
    if set(decoded) - allowed or type(version) is not int or version != 1:
        raise RuntimeError("Hermes context requires version 1 and only documented fields")
    if "identity" not in decoded:
        raise RuntimeError("Hermes context requires identity")
    try:
        return HermesRepresentativeContext(
            identity=decoded["identity"],
            persona=decoded.get("persona"),
            user_preferences=decoded.get("user_preferences"),
            location=decoded.get("location"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Hermes context contains an invalid descriptive field") from error


_COMPLETION_URL = re.compile(r"https?://[^\s<>()]+")
_COMPLETION_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(https?://[^\s)]+\)")
_COMPLETION_MACHINE_TOKEN = re.compile(r"\b(?:task_|run_|deleg_)[A-Za-z0-9_.:-]+")
_PRIVATE_RESULT_HANDLE = re.compile(r"(?:deleg_[A-Za-z0-9_-]*|run_[A-Za-z0-9_-]{8,})")
_TAGGED_COMPLETION = re.compile(
    r"\A<voice>(?P<voice>[\s\S]*?)</voice>\s*"
    r"<detail>(?P<detail>[\s\S]*?)</detail>\Z"
)
_COMPLETION_PROTOCOL_TAG = re.compile(r"</?(?:voice|detail)>", re.IGNORECASE)
_MAX_SPOKEN_COMPLETION_CHARS = 240
_MAX_CONVERSATION_COMPLETION_DETAIL_CHARS = 3200
_NO_DISPLAYABLE_RESULT = "No displayable result was returned."


def _completion_projections(detail: str) -> tuple[str, str]:
    tagged = _TAGGED_COMPLETION.fullmatch(detail)
    if tagged is not None:
        voice = tagged.group("voice").strip()
        visual = tagged.group("detail").strip()
        visual = _COMPLETION_PROTOCOL_TAG.sub("", visual).strip()
        valid_voice = (
            1 <= len(voice) <= _MAX_SPOKEN_COMPLETION_CHARS
            and len(voice.splitlines()) == 1
            and "<" not in voice
            and ">" not in voice
        )
        if valid_voice:
            return voice, visual or voice
        protocol_free = _COMPLETION_PROTOCOL_TAG.sub("", detail).strip()
        return "", visual or protocol_free or _NO_DISPLAYABLE_RESULT
    if _COMPLETION_PROTOCOL_TAG.search(detail) is not None:
        visual = _COMPLETION_PROTOCOL_TAG.sub("", detail).strip()
        return "", visual or _NO_DISPLAYABLE_RESULT
    return detail, detail


def _voice_friendly_completion_text(status: str, detail: str) -> str:
    detail = _COMPLETION_MARKDOWN_LINK.sub(r"\1", detail)
    detail = _COMPLETION_URL.sub("the source link", detail)
    detail = _COMPLETION_MACHINE_TOKEN.sub("the reference", detail)
    detail = " ".join(detail.replace("•", " ").split()).strip()
    if detail:
        detail = detail[0].upper() + detail[1:]
        if detail[-1] not in ".!?":
            detail += "."
    shortened = len(detail) > _MAX_SPOKEN_COMPLETION_CHARS
    if shortened:
        boundary = detail.rfind(". ", 0, _MAX_SPOKEN_COMPLETION_CHARS)
        if boundary < 160:
            boundary = detail.rfind(" ", 0, _MAX_SPOKEN_COMPLETION_CHARS)
        if boundary < 160:
            boundary = _MAX_SPOKEN_COMPLETION_CHARS
        detail = detail[:boundary].rstrip(" ,;:") + "."
    if status == "completed":
        parts = [detail] if detail else []
        if shortened or not detail:
            parts.append("The full result is on screen.")
        return " ".join(parts)
    lead = {
        "failed": "I ran into a problem with that work.",
        "interrupted": "That work stopped.",
    }[status]
    parts = [lead]
    if detail:
        parts.append(detail)
    if shortened or not detail:
        parts.append("The full result is on screen.")
    return " ".join(parts)


def _conversation_completion_text(status: str, voice_detail: str, visual_detail: str) -> str:
    """Give inference a natural handoff plus the result substance it must explain."""

    handoff = _voice_friendly_completion_text(status, voice_detail)
    full_result = _COMPLETION_MARKDOWN_LINK.sub(r"\1", visual_detail)
    full_result = _COMPLETION_URL.sub("the source link", full_result)
    full_result = _COMPLETION_MACHINE_TOKEN.sub("the reference", full_result)
    full_result = " ".join(full_result.replace("•", " ").split()).strip()
    if len(full_result) > _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS:
        boundary = full_result.rfind(" ", 0, _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS - 3)
        if boundary < _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS // 2:
            boundary = _MAX_CONVERSATION_COMPLETION_DETAIL_CHARS - 3
        full_result = full_result[:boundary].rstrip(" ,;:") + "..."
    normalized_voice = " ".join(voice_detail.split()).strip()
    if full_result == _NO_DISPLAYABLE_RESULT:
        return handoff
    if not full_result or (
        len(full_result) <= _MAX_SPOKEN_COMPLETION_CHARS
        and full_result.casefold() == normalized_voice.casefold()
    ):
        return handoff
    # The detail projection is the complete worker result. Supplying its short voice
    # projection as well makes conversational inference repeat the same finding and
    # sometimes speak transport scaffolding such as "Full background result".
    return full_result


def _browser_safe_task_result(detail: str) -> str:
    projected = _PRIVATE_RESULT_HANDLE.sub("private reference", detail)
    if len(projected) <= 1024:
        return projected
    boundary = projected.rfind(" ", 0, 1021)
    if boundary < 512:
        boundary = 1021
    return projected[:boundary].rstrip() + "..."


class ProactiveCompletionUpdatePolicy:
    """Project terminal task state now and speak only into an idle floor."""

    def __init__(
        self,
        *,
        projection: BrowserEventProjection,
        floor_available: Callable[[], bool] | None = None,
    ) -> None:
        if type(projection) is not BrowserEventProjection:
            raise TypeError("projection must be an exact BrowserEventProjection")
        if floor_available is not None and not callable(floor_available):
            raise TypeError("floor_available must be callable")
        self._projection = projection
        self._floor_available = floor_available

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        if type(policy_input) is not UpdatePolicyInput:
            raise TypeError("policy_input must be an exact UpdatePolicyInput")
        completion = policy_input.completion
        self._projection.ensure_capacity(3)
        self._projection.publish(
            "completion_received",
            {"status": completion.status, "taskId": completion.task_id},
        )
        self._projection.publish(
            "task_state",
            {"status": completion.status, "taskId": completion.task_id},
        )
        if completion.status == "completed":
            assert completion.summary is not None
            detail = completion.summary
        elif completion.status == "failed":
            assert completion.reason is not None
            detail = completion.reason
        else:
            assert completion.reason is not None
            detail = completion.reason
        voice_detail, visual_detail = _completion_projections(detail)
        browser_detail = _browser_safe_task_result(visual_detail)
        self._projection.publish(
            "task_result",
            {
                "status": completion.status,
                "taskId": completion.task_id,
                "text": browser_detail,
            },
        )
        floor_available = False
        if self._floor_available is not None:
            floor_available = self._floor_available()
            if type(floor_available) is not bool:
                raise TypeError("floor_available must return an exact boolean")
        return UpdateDirective(
            kind="interrupt" if floor_available else "mention_next",
            text=_conversation_completion_text(
                completion.status,
                voice_detail,
                visual_detail,
            ),
        )


def _conversation_profile_duration_limit(profile: str) -> float | None:
    if type(profile) is not str:
        raise TypeError("conversation_profile must be an exact built-in string")
    if profile == "legacy":
        return None
    if profile == "natural_v1":
        return None
    raise ValueError("conversation_profile must be legacy or natural_v1")


_QUALIFICATION_NO_TASK_COMPOSITION: ContextVar[bool] = ContextVar(
    "hermes_realtime_qualification_no_task_composition",
    default=False,
)


@contextmanager
def _qualification_no_task_composition() -> Iterator[None]:
    """Enable no-task host construction without extending the public builder API."""

    activation = _QUALIFICATION_NO_TASK_COMPOSITION.set(True)
    try:
        yield
    finally:
        _QUALIFICATION_NO_TASK_COMPOSITION.reset(activation)


def _compose_cli_host(
    *,
    qualification_no_tasks: bool,
    qualification_checkpoint_channel: _QualificationCheckpointChannelV1 | None = None,
    factory: Callable[[], LocalBrowserLauncher],
) -> LocalBrowserLauncher:
    scope = _qualification_no_task_composition() if qualification_no_tasks else nullcontext()
    checkpoint_scope = (
        _qualification_checkpoint_channel_scope(qualification_checkpoint_channel)
        if qualification_checkpoint_channel is not None
        else nullcontext()
    )
    with scope, checkpoint_scope:
        return factory()


def build_local_host_launcher(
    *,
    hermes_api_bearer: str | None,
    hermes_api_url: str = "http://127.0.0.1:8642",
    livekit_url: str = "ws://127.0.0.1:7880",
    livekit_api_key: str = "devkey",
    livekit_api_secret: str = "local" + "-" + ("x" * 32),
    room_name: str = "hermes-realtime-host",
    worker_identity: str = "worker_local_realtime_host",
    browser_host: str = "127.0.0.1",
    browser_port: int = 8765,
    browser_canonical_origin: str | None = None,
    browser_certificate_file: Path | None = None,
    browser_private_key_file: Path | None = None,
    bootstrap_ttl_seconds: int = 120,
    inference_provider: str = "ollama",
    ollama_base_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "hermes-4.3-36b-iq4xs-16k:latest",
    codex_model: str = "gpt-5.6-terra",
    codex_effort: str = "low",
    codex_executable: str | None = None,
    hermes_context: HermesRepresentativeContext | None = None,
    stt_provider: str = "moonshine",
    whisper_model: str = "base.en",
    moonshine_model_tier: str = "tiny",
    moonshine_update_interval_seconds: float = 0.2,
    tts_provider: str | None = None,
    edge_voice: str = "en-US-AriaNeural",
    kokoro_voice: str = "bf_isabella",
    kokoro_speed: float = 1.0,
    kokoro_intra_op_threads: int = 8,
    kokoro_pronunciations: Mapping[str, str] | None = None,
    kokoro_stream_chunk_chars: int = 400,
    kokoro_worker_python: Path | None = None,
    allow_unsandboxed_tasks: bool = False,
    natural_work_tools: bool = False,
    public_search: bool = False,
    knowledge_speculation: bool = False,
    knowledge_recovery: bool = False,
    knowledge_budget_seconds: float = 3.5,
    conversation_profile: str = "legacy",
    remote_mode: bool = False,
    tailnet_authorizer: TailnetPeerAuthorizer | None = None,
    loopback_authorizer: LoopbackPeerAuthorizer | None = None,
    evidence_capture: bool = False,
    evidence_retention_hours: int = 24,
    evidence_database: Path | None = None,
) -> LocalBrowserLauncher:
    """Compose the explicit full host without provider fallback."""

    qualification_dependencies = _current_qualification_full_host_dependencies()
    qualification_no_hermes_tasks = (
        _QUALIFICATION_NO_TASK_COMPOSITION.get() or qualification_dependencies is not None
    )
    if qualification_dependencies is not None:
        qualification_dependencies = qualification_dependencies.take()
    if type(allow_unsandboxed_tasks) is not bool:
        raise TypeError("allow_unsandboxed_tasks must be an exact boolean")
    if qualification_no_hermes_tasks:
        if hermes_api_bearer is not None:
            raise ValueError("qualification no-task composition rejects Hermes credentials")
        if allow_unsandboxed_tasks or natural_work_tools:
            raise ValueError("qualification no-task composition rejects task flags")
    elif type(hermes_api_bearer) is not str:
        raise TypeError("hermes_api_bearer must be an exact string outside qualification mode")
    if type(evidence_capture) is not bool:
        raise TypeError("evidence_capture must be an exact boolean")
    if type(evidence_retention_hours) is not int or not 1 <= evidence_retention_hours <= 168:
        raise ValueError("evidence_retention_hours must be an integer from 1 through 168")
    if evidence_database is not None and not isinstance(evidence_database, Path):
        raise TypeError("evidence_database must be a pathlib Path or None")
    if evidence_capture and evidence_database is None:
        raise ValueError("evidence capture requires an explicit database path")
    if evidence_capture and sys.platform != "win32":
        raise RuntimeError("unsupported_platform")
    if not qualification_no_hermes_tasks and not allow_unsandboxed_tasks:
        raise PermissionError(
            "full Hermes tasks are unsandboxed; explicit operator opt-in is required"
        )
    if type(natural_work_tools) is not bool:
        raise TypeError("natural_work_tools must be an exact boolean")
    if natural_work_tools and inference_provider != "codex":
        raise ValueError("--natural-work-tools requires --inference-provider codex")
    if type(public_search) is not bool:
        raise TypeError("public_search must be an exact boolean")
    if public_search and inference_provider != "codex":
        raise ValueError("public search requires --inference-provider codex")
    if type(knowledge_speculation) is not bool or type(knowledge_recovery) is not bool:
        raise TypeError("knowledge feature controls must be exact booleans")
    if (
        type(knowledge_budget_seconds) not in (int, float)
        or isinstance(knowledge_budget_seconds, bool)
        or not math.isfinite(float(knowledge_budget_seconds))
        or not 0.05 <= float(knowledge_budget_seconds) <= 3.5
    ):
        raise ValueError("knowledge_budget_seconds must be between 0.05 and 3.5")
    if (knowledge_speculation or knowledge_recovery) and inference_provider != "codex":
        raise ValueError("knowledge overlap and recovery require --inference-provider codex")
    if (knowledge_speculation or knowledge_recovery) and not public_search:
        raise ValueError("knowledge overlap and recovery require public search")
    if knowledge_speculation and stt_provider != "moonshine":
        raise ValueError("knowledge speculation requires --stt-provider moonshine")
    speech_chunk_duration_limit = _conversation_profile_duration_limit(conversation_profile)
    if tailnet_authorizer is not None and not remote_mode:
        raise ValueError("Tailnet launch authorization is accepted only in remote mode")
    if loopback_authorizer is not None and remote_mode:
        raise ValueError("loopback launch authorization is accepted only in local mode")
    if tailnet_authorizer is not None and loopback_authorizer is not None:
        raise ValueError("browser launch authorizers are mutually exclusive")
    if tts_provider is None:
        tts_provider = _default_tts_provider()
    api_config = (
        None
        if qualification_no_hermes_tasks
        else HermesApiConfig(
            base_url=hermes_api_url,
            bearer=cast(str, hermes_api_bearer),
            run_provider="openai-codex" if inference_provider == "codex" else None,
            run_model=codex_model if inference_provider == "codex" else None,
            run_reasoning_effort=codex_effort if inference_provider == "codex" else None,
            run_service_tier="priority" if inference_provider == "codex" else None,
        )
    )
    livekit_url = _validate_livekit_origin(livekit_url, remote_mode=remote_mode)
    if remote_mode:
        try:
            canonical_remote_hostname(browser_host, boundary="remote browser listener")
        except ValueError:
            raise ValueError("remote browser listener must be explicitly non-loopback") from None
        if livekit_api_key == "devkey" or livekit_api_secret == "local" + "-" + ("x" * 32):
            raise ValueError("remote mode rejects loopback development credentials")
    browser_ssl_context = _build_server_ssl_context(
        remote_mode=remote_mode,
        certificate_file=browser_certificate_file,
        private_key_file=browser_private_key_file,
        canonical_origin=browser_canonical_origin,
    )
    connection = LiveKitConnection(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
    )
    projection = BrowserEventProjection()
    usage_accumulator = _SessionTokenUsageAccumulator()
    production_observations, production_observation_recorder = _new_observation_channel()
    consent_dependencies = _HostEvidenceConsentDependenciesV1(
        writer_transport_factory=(
            cast(
                Callable[[HostEvidenceRuntimeV1], _WriterTransportV1],
                qualification_dependencies.writer_transport_factory,
            )
            if qualification_dependencies is not None
            and qualification_dependencies.writer_transport_factory is not None
            else lambda runtime: runtime.create_sqlite_transport()
        )
    )

    evidence_runtime = (
        HostEvidenceRuntimeV1(
            database=cast(Path, evidence_database),
            owner_generation=(uuid.uuid4().int & (2**63 - 1)) or 1,
            retention_hours=evidence_retention_hours,
            production_observations=production_observations,
            production_observation_recorder=production_observation_recorder,
        )
        if evidence_capture
        else None
    )

    def observe_token_usage(turn_id: str, usage: CodexTokenUsage) -> None:
        data = usage_accumulator.observe(turn_id, usage)
        if data is not None:
            projection.publish_advisory("session_usage", data)

    speech_generations: dict[str, tuple[str, int, str, int]] = {}
    active_renderer_authority: _SpeechRendererAuthority | None = None
    active_renderer_presentation: _RendererPresentationAuthority | None = None

    def observe(kind: str, data: dict[str, _PublicValue]) -> None:
        nonlocal active_renderer_authority, active_renderer_presentation
        if kind == "transcript_partial" and data.get("role") == "assistant":
            turn_id = data.get("turnId")
            generation = data.get("turnGeneration")
            chunk_id = data.get("chunkId")
            segment_id = data.get("segmentId")
            segment_text_offset_utf16 = data.get("segmentTextOffsetUtf16")
            if (
                type(turn_id) is str
                and type(generation) is int
                and type(chunk_id) is str
                and type(segment_id) is str
                and type(segment_text_offset_utf16) is int
                and segment_text_offset_utf16 >= 0
            ):
                for stale_chunk, ownership in tuple(speech_generations.items()):
                    if ownership[0] != turn_id:
                        speech_generations.pop(stale_chunk, None)
                speech_generations.pop(chunk_id, None)
                speech_generations[chunk_id] = (
                    turn_id,
                    generation,
                    segment_id,
                    segment_text_offset_utf16,
                )
                while len(speech_generations) > 128:
                    speech_generations.pop(next(iter(speech_generations)))
        elif kind in {"assistant_turn_completed", "assistant_turn_interrupted"}:
            terminal_turn = data.get("turnId")
            terminal_generation = data.get("turnGeneration")
            for chunk_id, ownership in tuple(speech_generations.items()):
                if (ownership[0], ownership[1]) == (
                    terminal_turn,
                    terminal_generation,
                ):
                    speech_generations.pop(chunk_id, None)
        projected_data = _renderer_yield_event_data(kind, data, active_renderer_authority)
        if kind in {"interrupt_requested", "session_stopped"}:
            speech_generations.clear()
        if _renderer_authority_ends(kind, data, active_renderer_presentation):
            active_renderer_authority = None
            active_renderer_presentation = None
        _project_conversation_observation(projection, kind, projected_data)

    def observe_prepared_stream(chunk: SpeechChunk, stream_identity: str) -> None:
        nonlocal active_renderer_authority, active_renderer_presentation
        active_renderer_authority = None
        active_renderer_presentation = None
        ownership = speech_generations.get(chunk.chunk_id)
        if ownership is None:
            return
        data = _encode_speech_timing_event(
            chunk,
            stream_identity,
            turn_generation=ownership[1],
            presentation_turn_id=ownership[0],
            segment_id=ownership[2],
            segment_text_offset_utf16=ownership[3],
        )
        if data is None:
            active_renderer_authority = None
            active_renderer_presentation = None
            return
        active_renderer_authority = (
            chunk.turn_id,
            ownership[1],
            chunk.chunk_id,
            stream_identity,
        )
        active_renderer_presentation = (ownership[0], ownership[1])
        projection.publish_advisory("speech_timing", data)

    def resolve_stream_generation(chunk: SpeechChunk) -> int | None:
        ownership = speech_generations.get(chunk.chunk_id)
        if ownership is None:
            return None
        return ownership[1]

    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator(
        drain_timeout=10.0,
        output_capacity=DEFAULT_MAX_RESPONSE_SEGMENTS,
    )

    def observe_approval(data: dict[str, _PublicValue]) -> None:
        projection.publish("approval_state", data)

    task_session = (
        ConversationOnlyTaskSession()
        if qualification_no_hermes_tasks
        else HermesApiTaskSession(
            config=cast(HermesApiConfig, api_config),
            session_id="session_local_host",
            approval_observer=observe_approval,
        )
    )
    task_controller = ConversationTaskController(
        context=context,
        session=(
            task_session
            if qualification_no_hermes_tasks
            else _NonOwningHermesApiTaskSession(cast(HermesApiTaskSession, task_session))
        ),
        session_id="session_local_host",
        id_factory=lambda: uuid.uuid4().hex,
    )
    work_surface = ConversationWorkControlSurface(
        controller=task_controller,
        context=context,
        observer=observe,
        reserve_observer_capacity=projection.ensure_capacity,
        utterance_id_factory=lambda: uuid.uuid4().hex,
    )
    command_router = ConversationTaskCommandRouter(
        surface=work_surface,
        observer=observe,
        reserve_observer_capacity=projection.ensure_capacity,
        utterance_id_factory=lambda: uuid.uuid4().hex,
        lifecycle_owner=(
            evidence_runtime.conversation_authority
            if evidence_runtime is not None
            else None
        ),
    )
    search_egress_authority, current_fact_lookup, knowledge_coordinator = (
        _compose_public_search(
            operator_enabled=public_search,
            knowledge_budget_seconds=float(knowledge_budget_seconds),
            knowledge_speculation=knowledge_speculation,
            knowledge_recovery=knowledge_recovery,
        )
    )

    inference = (
        cast(
            OllamaStreamingInference | CodexAppServerStreamingInference,
            qualification_dependencies.inference_factory(),
        )
        if qualification_dependencies is not None
        else _build_streaming_inference(
            inference_provider=inference_provider,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            codex_model=codex_model,
            codex_effort=codex_effort,
            codex_executable=codex_executable,
            current_fact_lookup=current_fact_lookup,
            token_usage_observer=observe_token_usage,
            knowledge_timing_observer=lambda data: observe("knowledge_timing", data),
            knowledge_coordinator=knowledge_coordinator,
            hermes_context=hermes_context,
        )
    )
    synthesizer = (
        cast(
            EdgeTtsSynthesizer | KokoroSynthesizer,
            qualification_dependencies.synthesizer_factory(),
        )
        if qualification_dependencies is not None
        else _build_synthesizer(
            tts_provider=tts_provider,
            edge_voice=edge_voice,
            kokoro_voice=kokoro_voice,
            kokoro_speed=kokoro_speed,
            kokoro_intra_op_threads=kokoro_intra_op_threads,
            kokoro_pronunciations=kokoro_pronunciations,
            kokoro_stream_chunk_chars=kokoro_stream_chunk_chars,
            kokoro_worker_python=kokoro_worker_python,
        )
    )
    transcriber = (
        cast(
            FasterWhisperTranscriber | MoonshineStreamingTranscriber,
            qualification_dependencies.transcriber_factory(),
        )
        if qualification_dependencies is not None
        else _build_streaming_transcriber(
            stt_provider=stt_provider,
            whisper_model=whisper_model,
            moonshine_model_tier=moonshine_model_tier,
            moonshine_update_interval_seconds=moonshine_update_interval_seconds,
        )
    )
    media_incarnations: dict[int, int] = {}
    readiness_cue_authority = _ReadinessCueAuthority()
    readiness_cue_template: SpeechChunk | None = None
    readiness_cue_tasks = _ReadinessCueTasks()
    publisher = ReconnectSafeLiveKitAudioPublisher()
    verifier_identity = (
        qualification_dependencies.identity_factory()
        if qualification_dependencies is not None
        else f"verifier_{uuid.uuid4().hex}"
    )
    if type(verifier_identity) is not str or not verifier_identity:
        raise TypeError("qualification identity factory must return a non-empty exact string")
    verifier = LiveKitRoomPeer(
        connection=connection,
        identity=verifier_identity,
    )
    verifier.bind_remote_identity(worker_identity)
    confirmation = LiveKitPCMDeliveryConfirmation(
        verifier,
        timeout_seconds=_LIVEKIT_CONFIRMATION_TIMEOUT_SECONDS,
        max_frames=4096,
    )
    echo_guard = PlaybackEchoGuard()
    speech_presence_verifier = (
        cast(
            SileroSpeechPresenceVerifier,
            qualification_dependencies.speech_presence_factory(),
        )
        if qualification_dependencies is not None
        else SileroSpeechPresenceVerifier()
    )
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=_LIVEKIT_PUBLICATION_TIMEOUT_SECONDS,
        confirmation_timeout_seconds=_LIVEKIT_CONFIRMATION_TIMEOUT_SECONDS,
        on_stream_prepared=observe_prepared_stream,
        stream_generation=resolve_stream_generation,
        echo_reference=echo_guard,
    )
    playback_gate = asyncio.Lock()
    serialized_playback = _SerializedSpeechPlayback(playback, playback_gate)

    def schedule_readiness_cue(
        participant_identity: str,
        generation: int,
        media_incarnation: int,
    ) -> None:
        template = readiness_cue_template
        if template is None:
            return
        chunk = _readiness_cue_chunk(
            template,
            generation=generation,
            media_incarnation=media_incarnation,
        )

        async def play_with_suppressed_input() -> None:
            if speech.active_turn_id is not None or playback_gate.locked():
                return
            async with playback_gate:
                if speech.active_turn_id is not None or not readiness_cue_authority.admit(
                    generation,
                    media_incarnation,
                ):
                    return
                token = await conversation.suppress_audio_input(
                    participant_identity=participant_identity,
                    session_generation=generation,
                )
                try:
                    await playback.play(
                        chunk,
                        is_valid=lambda: media_incarnations.get(generation) == media_incarnation,
                    )
                finally:
                    resume_task = asyncio.create_task(
                        conversation.resume_audio_input(token),
                        name=f"readiness-resume:{generation}:{media_incarnation}",
                    )
                    try:
                        await asyncio.shield(resume_task)
                    except asyncio.CancelledError:
                        await asyncio.gather(resume_task, return_exceptions=True)
                        raise

        readiness_cue_tasks.create(
            play_with_suppressed_input(),
            name=f"readiness-cue:{generation}:{media_incarnation}",
        )

    speech = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=synthesizer,
        playback=serialized_playback,
        ledger=DeliveredSpeechLedger(),
        observer=observe,
        max_segments=DEFAULT_MAX_RESPONSE_SEGMENTS,
        cleanup_timeout_seconds=10.0,
        max_speech_chunk_duration_seconds=speech_chunk_duration_limit,
        evidence_admission=(
            evidence_runtime.evidence_admission if evidence_runtime is not None else None
        ),
        evidence_resolver=(
            evidence_runtime.resolve_evidence_pair if evidence_runtime is not None else None
        ),
        production_observation_recorder=production_observation_recorder,
    )
    if evidence_runtime is not None:
        evidence_runtime.configure_retention_owner(
            cancel=speech.cancel_for_retention_expiry,
        )

    user_speaking = False

    def completion_floor_available() -> bool:
        return not user_speaking and not speech.foreground_active

    director = ConversationUpdateDirector(
        context=context,
        completions=task_controller,
        policy=(
            ConversationOnlyUpdatePolicy()
            if qualification_no_hermes_tasks
            else ProactiveCompletionUpdatePolicy(
                projection=projection,
                floor_available=completion_floor_available,
            )
        ),
    )
    actions = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        cleanup_timeout_seconds=10.0,
        operation_scheduler=(
            evidence_runtime.operation_scheduler if evidence_runtime is not None else None
        ),
        evidence_resolver=(
            evidence_runtime.resolve_evidence_pair if evidence_runtime is not None else None
        ),
        production_observation_recorder=production_observation_recorder,
    )

    def observe_voice_activity(active: bool) -> None:
        nonlocal user_speaking
        user_speaking = active
        projection.publish_advisory(
            "voice_activity_started" if active else "voice_activity_ended",
            {},
        )

    def binding_factory(
        participant_identity: str,
        generation: int,
        shared_actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        vad = (
            cast(WebRtcVoiceActivityDetector, qualification_dependencies.vad_factory())
            if qualification_dependencies is not None
            else _build_voice_activity_detector(conversation_profile=conversation_profile)
        )

        def voice_input_ready() -> None:
            media_incarnation = media_incarnations.get(generation)
            _publish_voice_input_ready(projection, generation, media_incarnation)
            if media_incarnation is not None:
                schedule_readiness_cue(
                    participant_identity,
                    generation,
                    media_incarnation,
                )

        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            media_incarnation_provider=lambda: media_incarnations.get(generation),
            vad=vad,
            stt=transcriber,
            actions=shared_actions,
            stt_streams_partials=stt_provider == "moonshine",
            command_router=command_router,
            evidence_lifecycle_resolver=(
                (lambda: evidence_runtime.evidence_lifecycle)
                if evidence_runtime is not None
                else None
            ),
            production_observation_recorder=production_observation_recorder,
            observer=observe,
            knowledge_coordinator=knowledge_coordinator,
            knowledge_budget_seconds=float(knowledge_budget_seconds),
            on_audio_ready=voice_input_ready,
            on_voice_activity=observe_voice_activity,
            cleanup_timeout_seconds=10.0,
            pre_roll_frames=vad.required_pre_roll_frames,
            echo_guard=echo_guard,
            speech_presence_verifier=speech_presence_verifier,
        )

    conversation = ReconnectSafeConversationWorker(
        actions=actions,
        binding_factory=binding_factory,
        require_media_activation=True,
    )
    livekit_worker = LiveKitConversationWorker(
        runtime=conversation,
        publisher=publisher,
        production_observation_recorder=production_observation_recorder,
    )

    async def activate_media(
        participant_identity: str,
        generation: int,
        media_incarnation: int,
    ) -> None:
        previous_incarnation = media_incarnations.get(generation)
        await conversation.activate_media(
            participant_identity=participant_identity,
            session_generation=generation,
            media_incarnation=media_incarnation,
        )
        if (
            knowledge_coordinator is not None
            and previous_incarnation is not None
            and previous_incarnation != media_incarnation
        ):
            await knowledge_coordinator.close_binding(generation, previous_incarnation)
        media_incarnations[generation] = media_incarnation

    def session_started(participant_identity: str, generation: int) -> None:
        media_incarnations.clear()
        usage_accumulator.reset(participant_identity, generation)

    async def decide_approval(
        participant_identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        del participant_identity, generation
        if qualification_no_hermes_tasks:
            raise PermissionError("task approvals are disabled in the qualification child")
        if not isinstance(task_session, HermesApiTaskSession):
            raise RuntimeError("ordinary task session composition is unavailable")
        await task_session.decide_approval(
            approval_id=approval_id,
            sequence=sequence,
            decision=decision,
        )

    model_catalog: Callable[[], Awaitable[BrowserModelCatalog]] | None = None
    select_model: Callable[[str, str], Awaitable[BrowserModelCatalog]] | None = None
    if isinstance(inference, CodexAppServerStreamingInference):
        codex_inference = inference

        async def current_model_catalog() -> BrowserModelCatalog:
            return _build_browser_selectable_model_catalog(
                await codex_inference.model_configuration()
            )

        async def select_model_configuration(
            model: str,
            effort: str,
        ) -> BrowserModelCatalog:
            return _build_browser_selectable_model_catalog(
                await codex_inference.select_model_configuration(
                    model=model,
                    effort=effort,
                )
            )

        model_catalog = current_model_catalog
        select_model = select_model_configuration

    voice_configuration: Callable[[], tuple[tuple[str, ...], str | None]] | None = None
    select_voice: Callable[[str], Awaitable[None]] | None = None
    if isinstance(synthesizer, KokoroSynthesizer):

        def current_voice_configuration() -> tuple[tuple[str, ...], str | None]:
            return synthesizer.available_voices, synthesizer.selected_voice

        voice_configuration = current_voice_configuration
        select_voice = synthesizer.select_voice

    async def yield_active_speech(
        participant_identity: str,
        generation: int,
        turn_id: str,
        turn_generation: int,
        chunk_id: str,
        stream_id: str,
    ) -> bool:
        del participant_identity
        if livekit_worker.active_generation != generation:
            raise PermissionError("yield claim does not own the active worker generation")
        matched = await playback.cancel_if_active_stream(
            turn_id=turn_id,
            turn_generation=turn_generation,
            chunk_id=chunk_id,
            stream_identity=stream_id,
        )
        if not matched:
            return False
        projection.publish_advisory("playback_silenced", {})
        return True

    runtime = BrowserClientRuntime(
        connection=connection,
        room_name=room_name,
        worker_identity=worker_identity,
        worker=livekit_worker,
        approval=decide_approval,
        static_root=Path(__file__).resolve().parent / "client" / "static",
        host=browser_host,
        port=browser_port,
        lan_mode=remote_mode,
        ssl_context=browser_ssl_context,
        canonical_origin=browser_canonical_origin,
        bootstrap_ttl_seconds=bootstrap_ttl_seconds,
        speech_runtime=_build_browser_speech_runtime(
            stt_provider=stt_provider,
            whisper_model=whisper_model,
            moonshine_model_tier=moonshine_model_tier,
            tts_provider=tts_provider,
            edge_voice=edge_voice,
        ),
        model_configuration=_build_browser_model_configuration(
            inference_provider=inference_provider,
            ollama_model=ollama_model,
            codex_model=codex_model,
            codex_effort=codex_effort,
        ),
        model_catalog=model_catalog,
        select_model=select_model,
        activate_media=activate_media,
        yield_speech=yield_active_speech if conversation_profile == "natural_v1" else None,
        conversation_profile=conversation_profile,
        on_session_started=session_started,
        projection=projection,
        voice_configuration=voice_configuration,
        select_voice=select_voice,
        evidence_consent=(
            (
                lambda binding, request: _reserve_host_evidence_consent(
                    runtime=evidence_runtime,
                    projection=projection,
                    live_generation=lambda: livekit_worker.active_generation,
                    binding=binding,
                    request=request,
                    dependencies=consent_dependencies,
                )
            )
            if evidence_runtime is not None
            else None
        ),
        evidence_status=(
            (
                lambda: evidence_runtime.capture_status(
                    disclosure_digest=_evidence_disclosure_digest()
                )
            )
            if evidence_runtime is not None
            else None
        ),
        evidence_revoke=(
            (
                lambda binding, request: _reserve_host_evidence_revoke(
                    runtime=evidence_runtime,
                    projection=projection,
                    binding=binding,
                    request=request,
                )
            )
            if evidence_runtime is not None
            else None
        ),
        evidence_invalidate=(
            evidence_runtime.invalidate_active_binding if evidence_runtime is not None else None
        ),
        search_egress_authority=search_egress_authority,
        tailnet_authorizer=tailnet_authorizer,
        persistent_tailnet_mode=tailnet_authorizer is not None,
        loopback_authorizer=loopback_authorizer,
        persistent_loopback_mode=loopback_authorizer is not None,
        production_observation_recorder=production_observation_recorder,
    )

    async def preflight() -> None:
        nonlocal readiness_cue_template
        if qualification_no_hermes_tasks:
            snapshot = ConversationInferenceRequest(
                revision=0,
                messages=(
                    ConversationMessage(
                        role=ConversationRole.USER.value,
                        text="Reply with one short sentence confirming readiness.",
                    ),
                ),
                active_tasks=(),
                updates=(),
            )
            segments = [
                segment async for segment in inference.stream(snapshot, turn_id="turn_preflight")
            ]
            if not segments:
                raise RuntimeError("local inference preflight produced no speakable output")
            if type(synthesizer) is KokoroSynthesizer:
                await synthesizer.warm()
            probes = [
                chunk async for chunk in synthesizer.synthesize("Ready.", "turn_tts_preflight")
            ]
            if len(probes) != 1 or type(probes[0]) is not SpeechChunk:
                raise RuntimeError("TTS preflight did not produce one playable chunk")
            readiness_cue_template = probes[0]
            return
        readiness_cue_template = await _run_full_host_preflight(
            task_session=cast(_TaskSessionStarter, task_session),
            inference=inference,
            synthesizer=synthesizer,
            surface=work_surface,
            natural_work_tools=natural_work_tools,
        )

    stt_providers = (
        (cast(MoonshineStreamingTranscriber, transcriber),) if stt_provider == "moonshine" else ()
    )
    work_close_owner = (
        _QualificationNoTaskCloseOwner(
            inference,
            work_surface,
            task_controller,
            task_session,
        )
        if qualification_no_hermes_tasks
        else _FullHostWorkCloseOwner(
            inference=inference,
            surface=work_surface,
            task_controller=task_controller,
            task_session=task_session,
            production_observation_recorder=production_observation_recorder,
        )
    )

    class _HostShutdownRuntime:
        """Close browser ingress before foreground, work, speech, and evidence teardown."""

        async def start(self) -> str:
            return await runtime.start()

        async def close(self) -> None:
            errors: list[BaseException] = []
            try:
                await runtime.close_ingress()
            except BaseException as error:
                errors.append(error)
            try:
                await actions.close()
            except BaseException as error:
                errors.append(error)
            try:
                await work_close_owner.close()
            except BaseException as error:
                errors.append(error)
            try:
                await readiness_cue_tasks.close()
            except BaseException as error:
                errors.append(error)
            try:
                await runtime.close()
            except BaseException as error:
                errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("host shutdown runtime close failed", errors)

    return LocalBrowserLauncher(
        runtime=_HostShutdownRuntime(),
        verifier=verifier,
        room_name=room_name,
        preflight=preflight,
        providers=(
            readiness_cue_tasks,
            work_close_owner,
            *((knowledge_coordinator,) if knowledge_coordinator is not None else ()),
            synthesizer,
            *stt_providers,
            *((evidence_runtime,) if evidence_runtime is not None else ()),
        ),
        production_observations=production_observations,
        production_observation_recorder=production_observation_recorder,
    )

async def _run_host_cli(args: argparse.Namespace) -> None:
    qualification_checkpoint = _qualification_checkpoint_contract(args, os.environ)
    qualification_no_hermes_tasks = qualification_checkpoint is not None
    if args.tailnet_launch and not args.remote:
        raise ValueError("--tailnet-launch requires --remote")
    if args.persistent_loopback_launch and args.remote:
        raise ValueError("--persistent-loopback-launch rejects --remote")
    tailnet_authorizer = _load_tailnet_authorizer(enabled=args.tailnet_launch)
    loopback_authorizer = LoopbackPeerAuthorizer() if args.persistent_loopback_launch else None
    env_file = Path(args.hermes_env_file) if args.hermes_env_file is not None else None
    bearer = None if qualification_no_hermes_tasks else _load_api_bearer(env_file)
    hermes_context_file = (
        Path(args.hermes_context_file) if args.hermes_context_file is not None else None
    )
    hermes_context = (
        None
        if qualification_no_hermes_tasks
        else _load_hermes_context(hermes_context_file)
    )
    certificate_file = Path(args.tls_cert) if args.tls_cert is not None else None
    private_key_file = Path(args.tls_key) if args.tls_key is not None else None
    livekit_api_key, livekit_api_secret = _load_livekit_credentials(remote_mode=args.remote)
    evidence_capture = _operator_evidence_capture_enabled(args.evidence_capture, os.environ)
    evidence_database = (
        _resolve_evidence_database_path(args.evidence_db, os.environ)
        if evidence_capture
        else None
    )
    shutdown_requested = asyncio.Event()
    checkpoint_failures: list[BaseException] = []

    def fail_qualification_child(error: BaseException) -> None:
        if not checkpoint_failures:
            checkpoint_failures.append(error)
            shutdown_requested.set()

    checkpoint_channel = (
        _new_qualification_checkpoint_channel(
            write_handle=qualification_checkpoint[0],
            resume_handle=qualification_checkpoint[1],
            nonce=qualification_checkpoint[2],
            failure_callback=fail_qualification_child,
        )
        if qualification_checkpoint is not None
        else None
    )
    try:
        launcher = _compose_cli_host(
            qualification_no_tasks=qualification_no_hermes_tasks,
            qualification_checkpoint_channel=checkpoint_channel,
            factory=lambda: build_local_host_launcher(
            hermes_api_bearer=bearer,
            hermes_api_url=args.hermes_api_url,
            livekit_url=args.livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            room_name=args.room,
            worker_identity=args.worker_identity,
            browser_host=args.browser_host,
            browser_port=args.port,
            browser_canonical_origin=args.browser_origin,
            browser_certificate_file=certificate_file,
            browser_private_key_file=private_key_file,
            bootstrap_ttl_seconds=args.bootstrap_ttl_seconds,
            inference_provider=args.inference_provider,
            ollama_model=args.ollama_model,
            codex_model=args.codex_model,
            codex_effort=args.codex_effort,
            codex_executable=args.codex_executable,
            hermes_context=hermes_context,
            stt_provider=args.stt_provider,
            whisper_model=args.whisper_model,
            moonshine_model_tier=args.moonshine_model_tier,
            moonshine_update_interval_seconds=args.moonshine_update_interval,
            tts_provider=args.tts_provider,
            edge_voice=args.edge_voice,
            kokoro_voice=args.kokoro_voice,
            kokoro_speed=args.kokoro_speed,
            kokoro_intra_op_threads=args.kokoro_intra_op_threads,
            kokoro_pronunciations=_parse_kokoro_pronunciations(args.kokoro_pronunciation),
            kokoro_stream_chunk_chars=args.kokoro_stream_chunk_chars,
            kokoro_worker_python=args.kokoro_worker_python,
            allow_unsandboxed_tasks=args.allow_unsandboxed_hermes_tasks,
            natural_work_tools=args.natural_work_tools,
            public_search=args.enable_public_search,
            knowledge_speculation=args.knowledge_speculation,
            knowledge_recovery=args.knowledge_recovery,
            knowledge_budget_seconds=args.knowledge_budget_seconds,
            conversation_profile=args.conversation_profile,
            remote_mode=args.remote,
            tailnet_authorizer=tailnet_authorizer,
            loopback_authorizer=loopback_authorizer,
            evidence_capture=evidence_capture,
            evidence_retention_hours=args.evidence_retention_hours,
            evidence_database=evidence_database,
            ),
        )
    except BaseException:
        if checkpoint_channel is not None:
            checkpoint_channel.close()
        raise
    previous_sigbreak_handler: object | None = None
    try:
        if qualification_no_hermes_tasks:
            loop = asyncio.get_running_loop()

            def request_owned_shutdown(signum: int, _frame: object) -> None:
                if signum == signal.SIGBREAK:
                    loop.call_soon_threadsafe(shutdown_requested.set)

            previous_sigbreak_handler = signal.signal(
                signal.SIGBREAK,
                request_owned_shutdown,
            )
        launch_url = await launcher.start()
        if qualification_no_hermes_tasks:
            print('{"hermesTaskMode":"qualification_disabled","version":1}', flush=True)
        else:
            print(
                "WARNING: Hermes tasks are unsandboxed; command approvals are not a "
                "comprehensive mutation guard.",
                flush=True,
            )
        print(
            "Knowledge path: backend=public-rss "
            f"speculation={'enabled' if args.knowledge_speculation else 'disabled'} "
            f"recovery={'enabled' if args.knowledge_recovery else 'disabled'} "
            f"budget={args.knowledge_budget_seconds:.3f}s",
            flush=True,
        )
        print(
            "Open this stable private URL in the configured remote browser:"
            if args.tailnet_launch
            else "Open this one-use URL in the configured remote browser:"
            if args.remote
            else "Open this stable loopback URL in a local browser:"
            if args.persistent_loopback_launch
            else "Open this one-use URL in one local browser:",
            flush=True,
        )
        print(launch_url, flush=True)
        if qualification_no_hermes_tasks:
            await shutdown_requested.wait()
            if checkpoint_failures:
                raise RuntimeError(
                    "qualification checkpoint channel failed"
                ) from checkpoint_failures[0]
        else:
            await asyncio.Event().wait()
    finally:
        try:
            await launcher.close()
        finally:
            try:
                if previous_sigbreak_handler is not None:
                    signal.signal(
                        signal.SIGBREAK,
                        cast(signal.Handlers, previous_sigbreak_handler),
                    )
            finally:
                if checkpoint_channel is not None:
                    checkpoint_channel.close()


def _compact_json(document: Mapping[str, object]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _run_evidence_no_service(args: argparse.Namespace) -> tuple[int, str]:
    database = _resolve_evidence_database_path(args.evidence_db, os.environ)
    writer = SQLiteEvidenceWriterDaemonV1(
        lambda: SQLiteEvidenceSpool(
            database,
            clock=lambda: datetime.now(UTC),
        )
    )
    if args.evidence_status:
        recovered = writer.recover_existing_and_close()
        if recovered is RecoveryDisposition.ABSENT:
            return 0, _compact_json(
                {"ownerState": "absent", "result": "absent", "version": 1}
            )
        if recovered in (
            RecoveryDisposition.RECOVERED,
            RecoveryDisposition.PURGE_COMPLETED,
        ):
            result = (
                "recovered"
                if recovered is RecoveryDisposition.RECOVERED
                else "purge_completed"
            )
            return 0, _compact_json(
                {"ownerState": "recovery_only", "result": result, "version": 1}
            )
        if recovered is RecoveryDisposition.FAULTED:
            return 0, _compact_json(
                {"ownerState": "faulted", "result": "faulted", "version": 1}
            )
        if recovered is RecoveryDisposition.OWNERSHIP_UNAVAILABLE:
            return 3, _compact_json({"error": "ownership_unavailable", "version": 1})
        return 4, _compact_json({"error": "purge_failed", "version": 1})

    purge = writer.purge_full_store_and_close(
        FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=str(uuid.uuid4()),
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
    )
    if purge is PurgeDisposition.PURGE_COMPLETED:
        return 0, _compact_json({"result": "purge_completed", "version": 1})
    if purge is PurgeDisposition.ALREADY_ABSENT:
        return 0, _compact_json({"result": "already_absent", "version": 1})
    if purge is PurgeDisposition.OWNERSHIP_UNAVAILABLE:
        return 3, _compact_json({"error": "ownership_unavailable", "version": 1})
    return 4, _compact_json({"error": "purge_failed", "version": 1})


def _resolve_evidence_database_path(
    configured: str | None,
    environment: Mapping[str, str],
) -> Path:
    if configured is not None:
        return Path(configured)
    local_app_data = environment.get("LOCALAPPDATA")
    if not local_app_data:
        raise ValueError("LOCALAPPDATA is required for the default evidence database")
    return Path(local_app_data) / "HermesRealtime" / "evidence" / "capture-v1.sqlite3"


def _operator_evidence_capture_enabled(
    requested: bool,
    environment: Mapping[str, str],
) -> bool:
    if type(requested) is not bool:
        raise TypeError("evidence capture request must be an exact boolean")
    return requested and environment.get("HERMES_REALTIME_DISABLE_EVIDENCE") != "1"


def _parse_knowledge_budget_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("knowledge budget must be numeric") from error
    if not math.isfinite(parsed) or not 0.05 <= parsed <= 3.5:
        raise argparse.ArgumentTypeError("knowledge budget must be between 0.05 and 3.5")
    return parsed


def _qualification_checkpoint_contract(
    args: argparse.Namespace,
    environment: Mapping[str, str],
) -> tuple[int, int, str] | None:
    """Accept the hidden no-task mode only for one runner-owned child process."""

    no_tasks = args.qualification_no_hermes_tasks
    write_handle = args.qualification_checkpoint_write_handle
    resume_handle = args.qualification_checkpoint_resume_handle
    nonce = args.qualification_checkpoint_nonce
    present = (no_tasks, write_handle is not None, resume_handle is not None, nonce is not None)
    marker = environment.get("HERMES_REALTIME_QUALIFICATION_CHILD")
    if not any(present) and marker is None:
        return None
    if not all(present) or marker != "1":
        raise ValueError("qualification child requires the complete owned-child contract")
    if args.allow_unsandboxed_hermes_tasks or args.natural_work_tools:
        raise ValueError("qualification child rejects public Hermes task flags")
    if type(write_handle) is not str or type(resume_handle) is not str or type(nonce) is not str:
        raise ValueError("qualification child checkpoint arguments are invalid")
    if not write_handle.isascii() or not write_handle.isdecimal():
        raise ValueError("qualification child write handle is invalid")
    if not resume_handle.isascii() or not resume_handle.isdecimal():
        raise ValueError("qualification child resume handle is invalid")
    parsed_write = int(write_handle)
    parsed_resume = int(resume_handle)
    if not 1 <= parsed_write <= (1 << 64) - 1 or not 1 <= parsed_resume <= (1 << 64) - 1:
        raise ValueError("qualification child checkpoint handle is outside uint64")
    if parsed_write == parsed_resume:
        raise ValueError("qualification child checkpoint handles must be distinct")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ValueError("qualification child checkpoint nonce is invalid")
    return parsed_write, parsed_resume, nonce


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the explicit Hermes Realtime full host profile.",
    )
    parser.add_argument("--hermes-env-file")
    parser.add_argument(
        "--hermes-context-file",
        help="UTF-8 version-1 JSON file containing bounded descriptive Hermes profile data",
    )
    parser.add_argument("--hermes-api-url", default="http://127.0.0.1:8642")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="require secure remote LiveKit and an HTTPS browser listener",
    )
    launch_authority = parser.add_mutually_exclusive_group()
    launch_authority.add_argument(
        "--tailnet-launch",
        action="store_true",
        help="enable exact Tailnet-node launch authorization in remote mode",
    )
    launch_authority.add_argument(
        "--persistent-loopback-launch",
        action="store_true",
        help="serve one stable loopback URL across sequential local sessions",
    )
    parser.add_argument("--livekit-url", default="ws://127.0.0.1:7880")
    parser.add_argument("--browser-host", default="127.0.0.1")
    parser.add_argument("--browser-origin")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--room", default="hermes-realtime-host")
    parser.add_argument("--worker-identity", default="worker_local_realtime_host")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--evidence-capture",
        action="store_true",
        help="make consent-bound evidence controls available; grants no consent",
    )
    parser.add_argument(
        "--evidence-retention-hours",
        type=int,
        choices=range(1, 169),
        default=24,
    )
    parser.add_argument("--evidence-db")
    evidence_command = parser.add_mutually_exclusive_group()
    evidence_command.add_argument("--evidence-status", action="store_true")
    evidence_command.add_argument("--purge-evidence", action="store_true")
    parser.add_argument(
        "--bootstrap-ttl-seconds",
        type=int,
        choices=range(30, 3_601),
        default=120,
    )
    parser.add_argument(
        "--inference-provider",
        choices=("ollama", "codex"),
        default="ollama",
    )
    parser.add_argument(
        "--ollama-model",
        default="hermes-4.3-36b-iq4xs-16k:latest",
    )
    parser.add_argument("--codex-model", default="gpt-5.6-terra")
    parser.add_argument(
        "--codex-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max", "ultra"),
        default="low",
    )
    parser.add_argument("--codex-executable")
    parser.add_argument(
        "--conversation-profile",
        choices=("legacy", "natural_v1"),
        default="legacy",
        help="explicitly enable the disabled-by-default natural duplex profile",
    )
    natural_work_group = parser.add_mutually_exclusive_group()
    natural_work_group.add_argument(
        "--natural-work-tools",
        action="store_true",
        dest="natural_work_tools",
        help="enable experimental natural Hermes work routing for Codex app-server",
    )
    natural_work_group.add_argument(
        "--no-natural-work-tools",
        action="store_false",
        dest="natural_work_tools",
        help="keep explicit task-prefix routing without foreground work tools",
    )
    parser.set_defaults(natural_work_tools=False)
    parser.add_argument(
        "--enable-public-search",
        action="store_true",
        help=(
            "allow Bing Search RSS and outcome-shaped Google News RSS lookup only after the "
            "active browser accepts the versioned egress disclosure"
        ),
    )
    parser.add_argument(
        "--knowledge-speculation",
        action="store_true",
        help="enable partial-transcript public-search prefetch; requires --enable-public-search",
    )
    parser.add_argument(
        "--knowledge-recovery",
        action="store_true",
        help="enable one bounded public-search recovery attempt; requires --enable-public-search",
    )
    parser.add_argument(
        "--knowledge-budget-seconds",
        type=_parse_knowledge_budget_seconds,
        default=3.5,
    )
    parser.add_argument(
        "--stt-provider",
        choices=("moonshine", "faster-whisper"),
        default="moonshine",
    )
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument(
        "--moonshine-model-tier",
        choices=("tiny", "small", "medium"),
        default="tiny",
        help="Moonshine v2 streaming model tier",
    )
    parser.add_argument("--moonshine-update-interval", type=float, default=0.2)
    parser.add_argument(
        "--tts-provider",
        choices=("kokoro", "edge"),
        default=_default_tts_provider(),
    )
    parser.add_argument("--edge-voice", default="en-US-AriaNeural")
    parser.add_argument("--kokoro-voice", default="bf_isabella")
    parser.add_argument("--kokoro-speed", type=float, default=1.0)
    parser.add_argument("--kokoro-intra-op-threads", type=int, default=8)
    parser.add_argument(
        "--kokoro-pronunciation",
        action="append",
        type=_parse_kokoro_pronunciation,
        default=[],
        metavar="TERM=SPOKEN",
    )
    parser.add_argument(
        "--kokoro-stream-chunk-chars",
        type=_parse_kokoro_stream_chunk_chars,
        default=400,
    )
    parser.add_argument(
        "--kokoro-worker-python",
        type=_parse_kokoro_worker_python,
        help="absolute Python executable in an isolated CUDA Kokoro environment",
    )
    parser.add_argument(
        "--allow-unsandboxed-hermes-tasks",
        action="store_true",
        help="acknowledge that Hermes tasks may mutate state without an approval prompt",
    )
    parser.add_argument(
        "--qualification-no-hermes-tasks",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--qualification-checkpoint-write-handle", help=argparse.SUPPRESS)
    parser.add_argument("--qualification-checkpoint-resume-handle", help=argparse.SUPPRESS)
    parser.add_argument("--qualification-checkpoint-nonce", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        _qualification_checkpoint_contract(args, os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    args.evidence_capture = _operator_evidence_capture_enabled(
        args.evidence_capture,
        os.environ,
    )
    evidence_requested = (
        args.evidence_capture or args.evidence_status or args.purge_evidence
    )
    if evidence_requested and sys.platform != "win32":
        print('{"error":"unsupported_platform","version":1}', flush=True)
        return 2
    if args.evidence_status or args.purge_evidence:
        exit_code, document = _run_evidence_no_service(args)
        print(document, flush=True)
        return exit_code
    try:
        _parse_kokoro_pronunciations(args.kokoro_pronunciation)
    except ValueError as exc:
        parser.error(str(exc))
    if args.tailnet_launch and not args.remote:
        parser.error("--tailnet-launch requires --remote")
    if args.persistent_loopback_launch and args.remote:
        parser.error("--persistent-loopback-launch rejects --remote")
    try:
        asyncio.run(_run_host_cli(args))
    except KeyboardInterrupt:
        return 130
    return 0


__all__ = [
    "ProactiveCompletionUpdatePolicy",
    "build_local_host_launcher",
    "main",
]
