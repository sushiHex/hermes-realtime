"""Identity-bound browser session provisioning authority."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import cast

from hermes_realtime.evidence.models import (
    BindingCloseReason,
    CaptureState,
    CaptureStatusV1,
    EvidenceConsentRequestV1,
    EvidenceRevokeRequestV1,
    capture_status_to_primitive,
)
from hermes_realtime.search_egress import (
    SearchEgressAuthority,
    SearchEgressBindingV1,
    SearchEgressConsentRequestV1,
    SearchEgressRevokeRequestV1,
    SearchEgressStatusV1,
    search_egress_status_to_primitive,
)

from .bootstrap import BrowserJoinCredential, BrowserTokenIssuer
from .projection import BrowserEventProjection, BrowserPublicEvent, PublicValue

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BrowserAudioDiagnostic:
    stream_id: str
    supported_echo_cancellation: bool
    supported_auto_gain_control: bool
    supported_noise_suppression: bool
    supported_voice_isolation: bool
    applied_echo_cancellation: str
    applied_auto_gain_control: str
    applied_noise_suppression: str
    applied_voice_isolation: str
    subscribed_to_attach_ms: int
    attach_to_playing_ms: int
    playing_to_advance_ms: int

    def __post_init__(self) -> None:
        if type(self.stream_id) is not str or re.fullmatch(
            r"[A-Za-z0-9_-]{8,128}", self.stream_id
        ) is None:
            raise ValueError("audio diagnostic stream identity is invalid")
        for supported in (
            self.supported_echo_cancellation,
            self.supported_auto_gain_control,
            self.supported_noise_suppression,
            self.supported_voice_isolation,
        ):
            if type(supported) is not bool:
                raise TypeError("audio diagnostic support fields must be exact booleans")
        for applied in (
            self.applied_echo_cancellation,
            self.applied_auto_gain_control,
            self.applied_noise_suppression,
            self.applied_voice_isolation,
        ):
            if type(applied) is not str or applied not in {"enabled", "disabled", "unknown"}:
                raise ValueError("audio diagnostic processor state is invalid")
        for delta_ms in (
            self.subscribed_to_attach_ms,
            self.attach_to_playing_ms,
            self.playing_to_advance_ms,
        ):
            if type(delta_ms) is not int or not 0 <= delta_ms <= 120_000:
                raise ValueError("audio diagnostic renderer delta is invalid")


@dataclass(frozen=True, slots=True)
class BrowserBindingSnapshot:
    """One identity-checked snapshot of the current browser binding."""

    participant_identity: str
    binding_generation: int

    def __post_init__(self) -> None:
        if type(self.participant_identity) is not str or not self.participant_identity:
            raise ValueError("participant_identity must be a nonempty exact string")
        if (
            type(self.binding_generation) is not int
            or not 1 <= self.binding_generation <= (1 << 53) - 1
        ):
            raise ValueError("binding_generation is outside the browser-safe range")


@dataclass(slots=True)
class _EvidenceIngressGate:
    """One binding-scoped, synchronously closed evidence-ingress gate."""

    binding: BrowserBindingSnapshot
    closed: bool = False


@dataclass(frozen=True, slots=True)
class BrowserEvidenceControlResponse:
    status: int
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.status) is not int or self.status not in {200, 202, 409, 503}:
            raise ValueError("evidence control response status is invalid")
        if type(self.payload) is not dict:
            raise TypeError("evidence control payload must be an exact dictionary")


@dataclass(frozen=True, slots=True)
class BrowserEvidenceConsentOperation:
    """One consent reservation completed outside the browser authority lock."""

    complete: Callable[[], Awaitable[BrowserEvidenceControlResponse]]
    settlement: Callable[[], Awaitable[BrowserEvidenceControlResponse]] | None = None

    def __post_init__(self) -> None:
        if not callable(self.complete):
            raise TypeError("complete must be callable")
        if self.settlement is not None and not callable(self.settlement):
            raise TypeError("settlement must be callable or None")


@dataclass(frozen=True, slots=True)
class BrowserEvidenceRevokeOperation:
    """One independently gated revoke completed outside the gate lock."""

    complete: Callable[[], Awaitable[BrowserEvidenceControlResponse]]

    def __post_init__(self) -> None:
        if not callable(self.complete):
            raise TypeError("complete must be callable")


@dataclass(frozen=True, slots=True)
class BrowserSelectableModel:
    model: str
    display_name: str
    description: str
    supported_efforts: tuple[str, ...]
    default_effort: str

    def __post_init__(self) -> None:
        if (
            type(self.model) is not str
            or not self.model
            or len(self.model) > 256
            or type(self.display_name) is not str
            or not self.display_name
            or len(self.display_name) > 256
            or type(self.description) is not str
            or len(self.description) > 1024
            or type(self.supported_efforts) is not tuple
            or not self.supported_efforts
            or any(
                type(effort) is not str
                or effort not in {"none", "low", "medium", "high", "xhigh", "max", "ultra"}
                for effort in self.supported_efforts
            )
            or len(set(self.supported_efforts)) != len(self.supported_efforts)
            or self.default_effort not in self.supported_efforts
        ):
            raise ValueError("selectable browser model is invalid")

    def public_data(self) -> dict[str, object]:
        return {
            "defaultEffort": self.default_effort,
            "description": self.description,
            "displayName": self.display_name,
            "model": self.model,
            "supportedEfforts": list(self.supported_efforts),
        }


@dataclass(frozen=True, slots=True)
class BrowserModelCatalog:
    models: tuple[BrowserSelectableModel, ...]
    selected_model: str
    selected_effort: str

    def __post_init__(self) -> None:
        if type(self.models) is not tuple or not self.models or any(
            type(model) is not BrowserSelectableModel for model in self.models
        ):
            raise TypeError("browser model catalog must contain selectable models")
        if len({model.model for model in self.models}) != len(self.models):
            raise ValueError("browser model catalog contains duplicate models")
        selected = next(
            (model for model in self.models if model.model == self.selected_model), None
        )
        if selected is None or self.selected_effort not in selected.supported_efforts:
            raise ValueError("browser model catalog selection is invalid")

    def public_data(self) -> dict[str, object]:
        return {
            "models": [model.public_data() for model in self.models],
            "selectedEffort": self.selected_effort,
            "selectedModel": self.selected_model,
            "version": 1,
        }


@dataclass(frozen=True, slots=True)
class BrowserModelConfiguration:
    """Bounded browser-safe identity and reporting capabilities of the session LLM."""

    authentication: str
    provider: str
    transport: str
    model: str
    effort: str | None
    context_window_tokens: int | None
    reports_token_usage: bool

    def __post_init__(self) -> None:
        if self.authentication not in {"subscription", "api-token", "local"}:
            raise ValueError("authentication must identify subscription, API token, or local")
        for name, value, maximum in (
            ("provider", self.provider, 64),
            ("transport", self.transport, 64),
            ("model", self.model, 256),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact built-in string")
            if not value.strip() or len(value) > maximum:
                raise ValueError(f"{name} must be nonblank and bounded")
        if self.effort is not None and (
            type(self.effort) is not str
            or self.effort not in {"none", "low", "medium", "high", "xhigh", "max", "ultra"}
        ):
            raise ValueError("effort must be a supported reasoning level or None")
        if self.context_window_tokens is not None and (
            type(self.context_window_tokens) is not int
            or not 1 <= self.context_window_tokens <= 100_000_000
        ):
            raise ValueError("context window tokens must be a bounded exact integer or None")
        if type(self.reports_token_usage) is not bool:
            raise TypeError("reports_token_usage must be an exact boolean")

    def public_data(self) -> dict[str, str | int | bool | None]:
        return {
            "authentication": self.authentication,
            "contextWindowTokens": self.context_window_tokens,
            "effort": self.effort,
            "model": self.model,
            "provider": self.provider,
            "reportsTokenUsage": self.reports_token_usage,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class BrowserSpeechRuntime:
    """Bounded browser-safe identities of the admitted speech providers and models."""

    stt_provider: str
    stt_model: str
    tts_provider: str
    tts_model: str

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("stt_provider", self.stt_provider, 64),
            ("stt_model", self.stt_model, 256),
            ("tts_provider", self.tts_provider, 64),
            ("tts_model", self.tts_model, 256),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact built-in string")
            if not value.strip() or len(value) > maximum:
                raise ValueError(f"{name} must be nonblank and bounded")

    def public_data(self) -> dict[str, str]:
        return {
            "sttModel": self.stt_model,
            "sttProvider": self.stt_provider,
            "ttsModel": self.tts_model,
            "ttsProvider": self.tts_provider,
        }


class BrowserSessionDirector:
    """Provision one matching server worker before exposing its browser token."""

    def __init__(
        self,
        *,
        issuer: BrowserTokenIssuer,
        provision: Callable[[str], Awaitable[int]],
        reprovision: Callable[[str], Awaitable[int]] | None = None,
        submit: Callable[[str, int, int, str], Awaitable[None]],
        stop: Callable[[str, int], Awaitable[None]],
        approval: Callable[[str, int, int, str, str], Awaitable[None]],
        projection: BrowserEventProjection,
        speech_runtime: BrowserSpeechRuntime | None = None,
        model_configuration: BrowserModelConfiguration | None = None,
        model_catalog: Callable[[], Awaitable[BrowserModelCatalog]] | None = None,
        select_model: Callable[[str, str], Awaitable[BrowserModelCatalog]] | None = None,
        activate_media: Callable[[str, int, int], Awaitable[None]] | None = None,
        observe_audio_diagnostic: Callable[
            [str, int, BrowserAudioDiagnostic], Awaitable[None]
        ]
        | None = None,
        yield_speech: Callable[[str, int, str, int, str, str], Awaitable[bool]] | None = None,
        conversation_profile: str = "legacy",
        on_session_started: Callable[[str, int], None] | None = None,
        voice_configuration: Callable[[], tuple[tuple[str, ...], str | None]] | None = None,
        select_voice: Callable[[str], Awaitable[None]] | None = None,
        evidence_consent: Callable[
            [BrowserBindingSnapshot, EvidenceConsentRequestV1],
            BrowserEvidenceConsentOperation,
        ]
        | None = None,
        evidence_status: Callable[[], CaptureStatusV1] | None = None,
        evidence_revoke: Callable[
            [BrowserBindingSnapshot, EvidenceRevokeRequestV1],
            BrowserEvidenceRevokeOperation,
        ]
        | None = None,
        evidence_invalidate: Callable[[BindingCloseReason], Awaitable[None]] | None = None,
        search_egress_authority: SearchEgressAuthority | None = None,
        inactivity_timeout_seconds: float = 300.0,
        rollback_timeout_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(issuer) is not BrowserTokenIssuer:
            raise TypeError("issuer must be an exact BrowserTokenIssuer")
        if not callable(provision):
            raise TypeError("provision must be callable")
        if reprovision is not None and not callable(reprovision):
            raise TypeError("reprovision must be callable")
        if not callable(submit):
            raise TypeError("submit must be callable")
        if not callable(stop):
            raise TypeError("stop must be callable")
        if not callable(approval):
            raise TypeError("approval must be callable")
        if type(projection) is not BrowserEventProjection:
            raise TypeError("projection must be an exact BrowserEventProjection")
        if speech_runtime is not None and type(speech_runtime) is not BrowserSpeechRuntime:
            raise TypeError("speech_runtime must be an exact BrowserSpeechRuntime")
        if (
            model_configuration is not None
            and type(model_configuration) is not BrowserModelConfiguration
        ):
            raise TypeError("model_configuration must be an exact BrowserModelConfiguration")
        if (model_catalog is None) != (select_model is None):
            raise ValueError("model catalog and selector must be configured together")
        if model_catalog is not None and not callable(model_catalog):
            raise TypeError("model_catalog must be callable")
        if select_model is not None and not callable(select_model):
            raise TypeError("select_model must be callable")
        if activate_media is not None and not callable(activate_media):
            raise TypeError("activate_media must be callable")
        if observe_audio_diagnostic is not None and not callable(observe_audio_diagnostic):
            raise TypeError("observe_audio_diagnostic must be callable")
        if yield_speech is not None and not callable(yield_speech):
            raise TypeError("yield_speech must be callable")
        if type(conversation_profile) is not str:
            raise TypeError("conversation_profile must be an exact built-in string")
        if conversation_profile not in {"legacy", "natural_v1"}:
            raise ValueError("conversation_profile must be legacy or natural_v1")
        if conversation_profile == "legacy" and yield_speech is not None:
            raise ValueError("legacy conversation profile cannot enable speech yielding")
        if conversation_profile == "natural_v1" and yield_speech is None:
            raise ValueError("natural_v1 conversation profile requires speech yielding")
        if on_session_started is not None and not callable(on_session_started):
            raise TypeError("on_session_started must be callable")
        if voice_configuration is not None and not callable(voice_configuration):
            raise TypeError("voice_configuration must be callable")
        if select_voice is not None and not callable(select_voice):
            raise TypeError("select_voice must be callable")
        if evidence_consent is not None and not callable(evidence_consent):
            raise TypeError("evidence_consent must be callable or None")
        if evidence_status is not None and not callable(evidence_status):
            raise TypeError("evidence_status must be callable or None")
        if evidence_revoke is not None and not callable(evidence_revoke):
            raise TypeError("evidence_revoke must be callable or None")
        if evidence_invalidate is not None and not callable(evidence_invalidate):
            raise TypeError("evidence_invalidate must be callable or None")
        if (
            search_egress_authority is not None
            and type(search_egress_authority) is not SearchEgressAuthority
        ):
            raise TypeError("search_egress_authority must be exact or None")
        if type(inactivity_timeout_seconds) not in (int, float):
            raise TypeError("inactivity timeout must be an exact number")
        if (
            not math.isfinite(inactivity_timeout_seconds)
            or not 30 <= inactivity_timeout_seconds <= 3600
        ):
            raise ValueError("inactivity timeout must be from 30 through 3600 seconds")
        if type(rollback_timeout_seconds) not in (int, float):
            raise TypeError("rollback timeout must be an exact number")
        if (
            not math.isfinite(rollback_timeout_seconds)
            or not 0.01 <= rollback_timeout_seconds <= 30
        ):
            raise ValueError("rollback timeout must be from 0.01 through 30 seconds")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._issuer = issuer
        self._provision = provision
        self._reprovision = reprovision
        self._submit = submit
        self._stop = stop
        self._approval = approval
        self._projection = projection
        self._speech_runtime = speech_runtime or BrowserSpeechRuntime(
            stt_provider="unavailable",
            stt_model="unavailable",
            tts_provider="unavailable",
            tts_model="unavailable",
        )
        self._model_configuration = model_configuration
        self._model_catalog = model_catalog
        self._select_model = select_model
        self._activate_media = activate_media
        self._observe_audio_diagnostic = observe_audio_diagnostic
        self._yield_speech = yield_speech
        self._conversation_profile = conversation_profile
        self._on_session_started = on_session_started
        self._voice_configuration = voice_configuration
        self._select_voice = select_voice
        self._evidence_consent = evidence_consent
        self._evidence_status = evidence_status
        self._evidence_revoke = evidence_revoke
        self._evidence_invalidate = evidence_invalidate
        self._search_egress_authority = search_egress_authority
        self._evidence_gate_lock = Lock()
        self._evidence_binding: BrowserBindingSnapshot | None = None
        self._evidence_ingress_gate: _EvidenceIngressGate | None = None
        self._last_evidence_control_sequence = 0
        self._pending_evidence_consent_key: (
            tuple[BrowserBindingSnapshot, EvidenceConsentRequestV1] | None
        ) = None
        self._pending_evidence_consent_operation: BrowserEvidenceConsentOperation | None = None
        self._pending_evidence_consent: asyncio.Future[BrowserEvidenceControlResponse] | None = None
        self._pending_evidence_consent_settlement: (
            asyncio.Task[None] | None
        ) = None
        self._last_evidence_consent_key: (
            tuple[BrowserBindingSnapshot, EvidenceConsentRequestV1] | None
        ) = None
        self._last_evidence_consent_result: BrowserEvidenceControlResponse | None = None
        self._pending_evidence_revoke_key: (
            tuple[BrowserBindingSnapshot, EvidenceRevokeRequestV1] | None
        ) = None
        self._pending_evidence_revoke_operation: BrowserEvidenceRevokeOperation | None = None
        self._pending_evidence_revoke: asyncio.Future[BrowserEvidenceControlResponse] | None = None
        self._last_evidence_revoke_key: (
            tuple[BrowserBindingSnapshot, EvidenceRevokeRequestV1] | None
        ) = None
        self._last_evidence_revoke_result: BrowserEvidenceControlResponse | None = None
        self._inactivity_timeout = float(inactivity_timeout_seconds)
        self._rollback_timeout = float(rollback_timeout_seconds)
        self._clock = clock
        self._start_lock = asyncio.Lock()
        self._model_operation_lock = asyncio.Lock()
        self._model_operation_epoch = 0
        self._pending_model_selection: asyncio.Future[BrowserModelCatalog] | None = None
        self._active_identity: str | None = None
        self._active_generation: int | None = None
        self._previous_rebind_request: tuple[str, str] | None = None
        self._previous_rebind_credential: BrowserJoinCredential | None = None
        self._rebind_finalizer: asyncio.Task[None] | None = None
        self._rebind_finalization_failed = False
        self._projection_resync_consumed = False
        self._active_media_incarnation: int | None = None
        self._last_input_sequence = 0
        self._last_input_request: tuple[int, str] | None = None
        self._last_approval_sequence = 0
        self._last_approval_request: tuple[int, str, str] | None = None
        self._audio_diagnostics: dict[str, BrowserAudioDiagnostic] = {}
        self._last_activity: float | None = None

    @property
    def active_generation(self) -> int | None:
        return self._active_generation

    @property
    def active_identity(self) -> str | None:
        return self._active_identity

    async def current_binding_snapshot(
        self,
        *,
        participant_identity: str,
    ) -> BrowserBindingSnapshot:
        """Return one current binding snapshot under the session authority lock."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("participant does not own the active session")
            return BrowserBindingSnapshot(
                participant_identity=identity,
                binding_generation=generation,
            )

    async def consent_to_search_egress(
        self,
        *,
        participant_identity: str,
        request: SearchEgressConsentRequestV1,
    ) -> SearchEgressStatusV1:
        """Apply exact search-egress consent to the active browser binding."""

        if type(request) is not SearchEgressConsentRequestV1:
            raise TypeError("request must be an exact SearchEgressConsentRequestV1")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("participant does not own the active session")
            authority = self._search_egress_authority
            if authority is None:
                raise RuntimeError("public search egress is unavailable")
            binding = SearchEgressBindingV1(
                participant_identity=identity,
                binding_generation=generation,
            )
            self._projection.ensure_capacity()
            status = authority.consent(binding, request)
            self._projection.publish(
                "search_egress_status",
                cast(dict[str, PublicValue], search_egress_status_to_primitive(status)),
            )
            self._touch_activity()
            return status

    async def revoke_search_egress(
        self,
        *,
        participant_identity: str,
        request: SearchEgressRevokeRequestV1,
    ) -> SearchEgressStatusV1:
        """Revoke search egress for the active browser binding."""

        if type(request) is not SearchEgressRevokeRequestV1:
            raise TypeError("request must be an exact SearchEgressRevokeRequestV1")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("participant does not own the active session")
            authority = self._search_egress_authority
            if authority is None:
                raise RuntimeError("public search egress is unavailable")
            binding = SearchEgressBindingV1(
                participant_identity=identity,
                binding_generation=generation,
            )
            self._projection.ensure_capacity()
            status = authority.revoke(binding, request)
            self._projection.publish(
                "search_egress_status",
                cast(dict[str, PublicValue], search_egress_status_to_primitive(status)),
            )
            self._touch_activity()
            return status

    async def consent_to_evidence(
        self,
        *,
        participant_identity: str,
        request: EvidenceConsentRequestV1,
    ) -> BrowserEvidenceControlResponse:
        """Bind one authenticated consent request to one current session generation."""

        if type(request) is not EvidenceConsentRequestV1:
            raise TypeError("request must be an exact EvidenceConsentRequestV1")
        callback = self._evidence_consent
        if callback is None:
            raise RuntimeError("evidence capture is unavailable")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("participant does not own the active session")
            binding = BrowserBindingSnapshot(
                participant_identity=identity,
                binding_generation=generation,
            )
            key = (binding, request)
            if key == self._last_evidence_consent_key:
                result = self._last_evidence_consent_result
                if result is None:
                    raise RuntimeError("completed evidence consent result is unavailable")
                return result
            pending_key = self._pending_evidence_consent_key
            if pending_key is not None:
                if key != pending_key:
                    raise RuntimeError("a different evidence consent is pending")
                task = self._pending_evidence_consent
                if task is None:
                    operation = self._pending_evidence_consent_operation
                    if operation is None:
                        raise RuntimeError("pending evidence consent lost its operation")
                    task = asyncio.ensure_future(operation.complete())
                    self._pending_evidence_consent = task
            else:
                task = None
            if task is None:
                status_provider = self._evidence_status
                if status_provider is None:
                    raise RuntimeError("evidence capture status is unavailable")
                status = status_provider()
                if type(status) is not CaptureStatusV1:
                    raise TypeError("evidence_status must return an exact CaptureStatusV1")
                if (
                    not status.available
                    or status.capture_state is not CaptureState.IDLE
                    or request.consent_version != status.consent_version
                    or request.disclosure_digest != status.disclosure_digest
                    or request.retention_hours != status.retention_hours
                ):
                    raise RuntimeError("consent does not match the published capture status")
                if request.sequence != self._last_evidence_control_sequence + 1:
                    raise RuntimeError("evidence control sequence conflicts with current binding")
                operation = callback(binding, request)
                if type(operation) is not BrowserEvidenceConsentOperation:
                    raise TypeError("evidence consent must return an exact operation")
                task = asyncio.ensure_future(operation.complete())
                self._pending_evidence_consent_key = key
                self._pending_evidence_consent_operation = operation
                self._pending_evidence_consent = task
        result = await asyncio.shield(task)
        if type(result) is not BrowserEvidenceControlResponse:
            raise TypeError("evidence consent result must be an exact control response")
        if result.status == 200 or result.payload.get("error") == "writer_unavailable":
            async with self._start_lock:
                if self._evidence_binding != binding:
                    raise RuntimeError("evidence consent binding became stale")
                if self._pending_evidence_consent is task:
                    self._last_evidence_control_sequence = request.sequence
                    self._last_evidence_consent_key = key
                    self._last_evidence_consent_result = result
                    self._pending_evidence_consent_key = None
                    self._pending_evidence_consent_operation = None
                    self._pending_evidence_consent = None
        elif result.payload.get("error") == "control_timeout":
            async with self._start_lock:
                if self._pending_evidence_consent is task:
                    operation = self._pending_evidence_consent_operation
                    if operation is None or operation.settlement is None:
                        raise RuntimeError("timed-out evidence consent has no settlement observer")
                    if self._pending_evidence_consent_settlement is None:
                        self._pending_evidence_consent_settlement = asyncio.create_task(
                            self._settle_pending_evidence_consent(
                                binding=binding,
                                key=key,
                                task=task,
                                operation=operation,
                            ),
                            name="browser-evidence-consent-settlement-observer",
                        )
        else:
            # A failed activation retires its pending authority and returns its
            # lifecycle capacity.  Errors other than writer_unavailable are not
            # completed control results, so the same authenticated request may
            # construct fresh authority on retry.
            async with self._start_lock:
                if self._pending_evidence_consent is task:
                    self._pending_evidence_consent_key = None
                    self._pending_evidence_consent_operation = None
                    self._pending_evidence_consent = None
        return result

    async def _settle_pending_evidence_consent(
        self,
        *,
        binding: BrowserBindingSnapshot,
        key: tuple[BrowserBindingSnapshot, EvidenceConsentRequestV1],
        task: asyncio.Future[BrowserEvidenceControlResponse],
        operation: BrowserEvidenceConsentOperation,
    ) -> None:
        """Observe-and-detach from host settlement without owning its cancellation."""

        settle = operation.settlement
        if settle is None:
            raise RuntimeError("pending evidence consent has no settlement callback")
        try:
            result = await asyncio.shield(settle())
        except asyncio.CancelledError:
            # A browser reset cancels this passive observer itself.  In contrast,
            # runtime close can settle the host-owned response with cancellation;
            # detach local state in that case without treating the browser as the
            # owner that cancelled the retained runtime operation.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            async with self._start_lock:
                if (
                    self._pending_evidence_consent is task
                    and self._pending_evidence_consent_operation is operation
                ):
                    self._pending_evidence_consent_key = None
                    self._pending_evidence_consent_operation = None
                    self._pending_evidence_consent = None
                    self._pending_evidence_consent_settlement = None
            return
        except BaseException:
            async with self._start_lock:
                if (
                    self._pending_evidence_consent is task
                    and self._pending_evidence_consent_operation is operation
                ):
                    self._pending_evidence_consent_key = None
                    self._pending_evidence_consent_operation = None
                    self._pending_evidence_consent = None
                    self._pending_evidence_consent_settlement = None
            return
        if type(result) is not BrowserEvidenceControlResponse:
            raise TypeError("evidence consent settlement returned an invalid response")
        async with self._start_lock:
            if (
                self._pending_evidence_consent is not task
                or self._pending_evidence_consent_operation is not operation
                or self._evidence_binding != binding
            ):
                return
            if result.status == 200 or result.payload.get("error") == "writer_unavailable":
                self._last_evidence_control_sequence = key[1].sequence
                self._last_evidence_consent_key = key
                self._last_evidence_consent_result = result
            self._pending_evidence_consent_key = None
            self._pending_evidence_consent_operation = None
            self._pending_evidence_consent = None
            self._pending_evidence_consent_settlement = None

    def _touch_activity(self) -> None:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise RuntimeError("session clock must return a finite exact number")
        now = float(value)
        if self._last_activity is not None and now < self._last_activity:
            raise RuntimeError("session clock regressed")
        self._last_activity = now

    def _bind_search_egress(self, identity: str, generation: int) -> None:
        authority = self._search_egress_authority
        if authority is None:
            return
        authority.bind(
            SearchEgressBindingV1(
                participant_identity=identity,
                binding_generation=generation,
            )
        )
        self._projection.publish(
            "search_egress_status",
            cast(dict[str, PublicValue], search_egress_status_to_primitive(authority.status())),
        )

    def _invalidate_search_egress(self, identity: str, generation: int) -> None:
        authority = self._search_egress_authority
        if authority is None or not authority.status().available:
            return
        authority.invalidate(
            SearchEgressBindingV1(
                participant_identity=identity,
                binding_generation=generation,
            )
        )
        self._projection.publish(
            "search_egress_status",
            cast(dict[str, PublicValue], search_egress_status_to_primitive(authority.status())),
        )

    async def start(self) -> BrowserJoinCredential:
        """Create one credential and provision its exact server-side identity first."""

        async with self._start_lock:
            if self._rebind_finalization_failed:
                raise RuntimeError("browser session recovery failed closed")
            if self._rebind_finalizer is not None:
                raise RuntimeError("browser session recovery is finalizing")
            if self._active_generation is not None:
                raise RuntimeError("a browser session is already active")
            self._projection.reset()
            status_reservation = (
                self._projection.reserve_capture_status()
                if self._evidence_status is not None
                else None
            )
            self._projection.ensure_capacity(
                1
                + (1 if self._model_configuration is not None else 0)
                + (1 if self._search_egress_authority is not None else 0)
            )
            self._touch_activity()
            credential = self._issuer.issue()
            generation = await self._provision(credential.participant_identity)
            if type(generation) is not int:
                raise TypeError("provision must return an exact integer generation")
            if generation <= 0:
                raise ValueError("provision generation must be positive")
            if self._on_session_started is not None:
                try:
                    self._on_session_started(credential.participant_identity, generation)
                except BaseException as callback_error:
                    cancellations, rollback_error = await self._settle_start_rollback(
                        credential.participant_identity,
                        generation,
                    )
                    self._clear_active_session()
                    failures: list[BaseException] = [callback_error, *cancellations]
                    if rollback_error is not None:
                        failures.append(rollback_error)
                    if len(failures) > 1:
                        raise BaseExceptionGroup(
                            "browser session start callback and rollback failed",
                            failures,
                        ) from None
                    raise
            self._projection.publish(
                "session_ready",
                {
                    "conversationProfile": self._conversation_profile,
                    "mode": "microphone_or_typed",
                    **self._speech_runtime.public_data(),
                },
            )
            if self._model_configuration is not None:
                self._projection.publish(
                    "session_model",
                    self._model_configuration.public_data(),
                )
            if status_reservation is not None:
                status_provider = self._evidence_status
                assert status_provider is not None
                evidence_status = status_provider()
                if type(evidence_status) is not CaptureStatusV1:
                    raise TypeError("evidence_status must return an exact CaptureStatusV1")
                self._projection.publish_capture_status(
                    status_reservation,
                    cast(
                        dict[str, PublicValue],
                        capture_status_to_primitive(evidence_status),
                    ),
                )
            self._active_identity = credential.participant_identity
            self._active_generation = generation
            with self._evidence_gate_lock:
                self._evidence_binding = BrowserBindingSnapshot(
                    participant_identity=credential.participant_identity,
                    binding_generation=generation,
                )
                self._evidence_ingress_gate = _EvidenceIngressGate(
                    self._evidence_binding
                )
            self._bind_search_egress(credential.participant_identity, generation)
            self._previous_rebind_request = None
            self._previous_rebind_credential = None
            self._last_input_sequence = 0
            self._last_approval_sequence = 0
            self._last_evidence_control_sequence = 0
            self._projection_resync_consumed = False
            self._active_media_incarnation = None
            self._audio_diagnostics.clear()
            return credential

    async def revoke_evidence(
        self,
        *,
        participant_identity: str,
        request: EvidenceRevokeRequestV1,
    ) -> BrowserEvidenceControlResponse:
        """Close capture through a nonawaiting gate independent of the start lock."""

        if type(request) is not EvidenceRevokeRequestV1:
            raise TypeError("request must be an exact EvidenceRevokeRequestV1")
        callback = self._evidence_revoke
        if callback is None:
            raise RuntimeError("evidence revoke is unavailable")
        with self._evidence_gate_lock:
            binding = self._evidence_binding
            if binding is None:
                raise RuntimeError("no browser evidence binding is active")
            gate = self._evidence_ingress_gate
            if gate is None or gate.binding is not binding:
                raise RuntimeError("browser evidence ingress gate is unavailable")
            if participant_identity != binding.participant_identity:
                raise PermissionError("participant does not own the evidence binding")
            key = (binding, request)
            if key == self._last_evidence_revoke_key:
                result = self._last_evidence_revoke_result
                if result is None:
                    raise RuntimeError("completed evidence revoke result is unavailable")
                return result
            pending_key = self._pending_evidence_revoke_key
            if pending_key is not None:
                if key != pending_key:
                    raise RuntimeError("a different evidence revoke is pending")
                task = self._pending_evidence_revoke
                if task is None:
                    operation = self._pending_evidence_revoke_operation
                    if operation is None:
                        raise RuntimeError("pending evidence revoke lost its operation")
                    task = asyncio.ensure_future(operation.complete())
                    self._pending_evidence_revoke = task
            else:
                if request.sequence != self._last_evidence_control_sequence + 1:
                    raise RuntimeError("evidence control sequence conflicts with current binding")
                gate.closed = True
                operation = callback(binding, request)
                if type(operation) is not BrowserEvidenceRevokeOperation:
                    raise TypeError("evidence revoke must return an exact operation")
                task = asyncio.ensure_future(operation.complete())
                self._pending_evidence_revoke_key = key
                self._pending_evidence_revoke_operation = operation
                self._pending_evidence_revoke = task
        result = await asyncio.shield(task)
        if type(result) is not BrowserEvidenceControlResponse:
            raise TypeError("evidence revoke result must be an exact control response")
        if result.status in {200, 202} or result.payload.get("error") in {
            "revoke_not_durable",
            "purge_failed",
        }:
            with self._evidence_gate_lock:
                if binding is not self._evidence_binding:
                    raise RuntimeError("evidence revoke binding became stale")
                if self._pending_evidence_revoke is task:
                    self._last_evidence_control_sequence = request.sequence
                    self._last_evidence_revoke_key = key
                    self._last_evidence_revoke_result = result
                    self._pending_evidence_revoke_key = None
                    self._pending_evidence_revoke_operation = None
                    self._pending_evidence_revoke = None
        elif result.payload.get("error") == "control_timeout":
            with self._evidence_gate_lock:
                if self._pending_evidence_revoke is task:
                    self._pending_evidence_revoke = None
        return result

    async def rebind(
        self,
        *,
        participant_identity: str,
        request_id: str | None = None,
    ) -> BrowserJoinCredential:
        async with self._model_operation_lock:
            return await self._rebind_after_model_operations(
                participant_identity=participant_identity,
                request_id=request_id,
            )

    async def projection_resync(
        self,
        *,
        participant_identity: str,
    ) -> BrowserJoinCredential:
        """Rotate a poisoned event cursor exactly once through current session authority."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        async with self._model_operation_lock, self._start_lock:
                if self._projection_resync_consumed:
                    raise RuntimeError("projection resync was already attempted")
                identity = self._active_identity
                generation = self._active_generation
                if identity is None or generation is None:
                    raise RuntimeError("no browser session is active")
                if participant_identity != identity:
                    raise PermissionError(
                        "projection resync participant does not own the active session"
                    )
                reprovision = self._reprovision
                if reprovision is None:
                    raise RuntimeError("browser session rebind is unavailable")
                self._projection_resync_consumed = True
                if self._search_egress_authority is not None:
                    self._projection.ensure_capacity()
                    self._invalidate_search_egress(identity, generation)
                await self._invalidate_evidence(BindingCloseReason.PROJECTION_RESYNC)
                credential = self._issuer.issue()
                if credential.participant_identity == identity:
                    raise RuntimeError("projection resync did not rotate browser identity")
                try:
                    replacement_generation = await reprovision(credential.participant_identity)
                    if type(replacement_generation) is not int:
                        raise TypeError("reprovision must return an exact integer generation")
                    if replacement_generation <= 0 or replacement_generation == generation:
                        raise ValueError("reprovision generation must be a new positive generation")
                except BaseException as replacement_error:
                    cancellations, rollback_generation, rollback_error = (
                        await self._settle_rebind_rollback(reprovision, identity)
                    )
                    if rollback_generation is not None:
                        self._active_generation = rollback_generation
                        if self._search_egress_authority is not None:
                            self._projection.ensure_capacity()
                            self._bind_search_egress(identity, rollback_generation)
                        self._touch_activity()
                    failures: list[BaseException] = [replacement_error, *cancellations]
                    if rollback_error is not None:
                        failures.append(rollback_error)
                    if len(failures) > 1:
                        raise BaseExceptionGroup(
                            "projection resync and rollback failed",
                            failures,
                        ) from None
                    raise
                self._projection.reset()
                self._projection.ensure_capacity(
                    1
                    + (1 if self._model_configuration is not None else 0)
                    + (1 if self._search_egress_authority is not None else 0)
                )
                self._projection.publish(
                    "session_ready",
                    {
                        "conversationProfile": self._conversation_profile,
                        "mode": "microphone_or_typed",
                        **self._speech_runtime.public_data(),
                    },
                )
                if self._model_configuration is not None:
                    self._projection.publish(
                        "session_model",
                        self._model_configuration.public_data(),
                    )
                status_reservation = (
                    self._projection.reserve_capture_status()
                    if self._evidence_status is not None
                    else None
                )
                if status_reservation is not None:
                    status_provider = self._evidence_status
                    assert status_provider is not None
                    status = status_provider()
                    if type(status) is not CaptureStatusV1:
                        raise TypeError("evidence_status must return an exact CaptureStatusV1")
                    self._projection.publish_capture_status(
                        status_reservation,
                        cast(dict[str, PublicValue], capture_status_to_primitive(status)),
                    )
                self._previous_rebind_request = None
                self._previous_rebind_credential = None
                self._active_identity = credential.participant_identity
                self._active_generation = replacement_generation
                self._active_media_incarnation = None
                with self._evidence_gate_lock:
                    self._evidence_binding = BrowserBindingSnapshot(
                        participant_identity=credential.participant_identity,
                        binding_generation=replacement_generation,
                    )
                    self._evidence_ingress_gate = _EvidenceIngressGate(
                        self._evidence_binding
                    )
                    self._last_evidence_control_sequence = 0
                self._bind_search_egress(
                    credential.participant_identity,
                    replacement_generation,
                )
                self._last_input_sequence = 0
                self._last_input_request = None
                self._last_approval_sequence = 0
                self._last_approval_request = None
                self._audio_diagnostics.clear()
                self._touch_activity()
                return credential

    async def _rebind_after_model_operations(
        self,
        *,
        participant_identity: str,
        request_id: str | None,
    ) -> BrowserJoinCredential:
        """Rotate a stale browser identity while preserving session-owned conversation state."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if request_id is not None:
            if type(request_id) is not str:
                raise TypeError("request_id must be an exact built-in string or None")
            suffix = request_id.removeprefix("rebind_")
            if (
                not 16 <= len(request_id) <= 128
                or suffix == request_id
                or not suffix.isascii()
                or not suffix.replace("-", "").isalnum()
            ):
                raise ValueError("request_id must be a canonical rebind request identifier")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                if (participant_identity, request_id) == self._previous_rebind_request:
                    replay = self._previous_rebind_credential
                    if replay is None:
                        raise RuntimeError("rebind replay authority is incomplete")
                    self._touch_activity()
                    return replay
                raise PermissionError("rebind participant does not own the active session")
            reprovision = self._reprovision
            if reprovision is None:
                raise RuntimeError("browser session rebind is unavailable")
            if self._search_egress_authority is not None:
                self._projection.ensure_capacity()
                self._invalidate_search_egress(identity, generation)
            await self._invalidate_evidence(BindingCloseReason.BINDING_REPLACED)
            credential = self._issuer.issue()
            if credential.participant_identity == identity:
                raise RuntimeError("browser session rebind did not rotate identity")
            try:
                replacement_generation = await reprovision(credential.participant_identity)
                if type(replacement_generation) is not int:
                    raise TypeError("reprovision must return an exact integer generation")
                if replacement_generation <= 0 or replacement_generation == generation:
                    raise ValueError("reprovision generation must be a new positive generation")
                self._touch_activity()
            except BaseException as replacement_error:
                cancellations, rollback_generation, rollback_error = (
                    await self._settle_rebind_rollback(reprovision, identity)
                )
                if rollback_generation is not None:
                    self._active_generation = rollback_generation
                    if self._search_egress_authority is not None:
                        self._projection.ensure_capacity()
                        self._bind_search_egress(identity, rollback_generation)
                    self._touch_activity()
                failures: list[BaseException] = [replacement_error, *cancellations]
                if rollback_error is not None:
                    failures.append(rollback_error)
                if len(failures) > 1:
                    raise BaseExceptionGroup(
                        "browser session rebind and rollback failed",
                        failures,
                    ) from None
                raise
            self._previous_rebind_request = (
                (identity, request_id) if request_id is not None else None
            )
            self._previous_rebind_credential = credential if request_id is not None else None
            self._active_identity = credential.participant_identity
            self._active_generation = replacement_generation
            self._active_media_incarnation = None
            with self._evidence_gate_lock:
                self._evidence_binding = BrowserBindingSnapshot(
                    participant_identity=credential.participant_identity,
                    binding_generation=replacement_generation,
                )
                self._evidence_ingress_gate = _EvidenceIngressGate(
                    self._evidence_binding
                )
                self._last_evidence_control_sequence = 0
            if self._search_egress_authority is not None:
                self._projection.ensure_capacity()
                self._bind_search_egress(
                    credential.participant_identity,
                    replacement_generation,
                )
            return credential

    async def activate_media(
        self,
        *,
        participant_identity: str,
        media_incarnation: int,
    ) -> None:
        """Bind subsequent PCM readiness to one authenticated browser track incarnation."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(media_incarnation) is not int:
            raise TypeError("media_incarnation must be an exact integer")
        if not 1 <= media_incarnation <= (1 << 53) - 1:
            raise ValueError("media_incarnation is outside the browser-safe range")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("media participant does not own the active session")
            activate = self._activate_media
            if activate is None:
                raise RuntimeError("media activation authority is unavailable")
            if (
                self._active_media_incarnation is not None
                and media_incarnation != self._active_media_incarnation
            ):
                await self._invalidate_evidence(BindingCloseReason.MEDIA_INCARNATION_REPLACED)
            await activate(identity, generation, media_incarnation)
            self._active_media_incarnation = media_incarnation
            self._touch_activity()

    async def submit_audio_diagnostic(
        self,
        *,
        participant_identity: str,
        diagnostic: BrowserAudioDiagnostic,
    ) -> None:
        """Accept bounded advisory browser audio evidence without conversation authority."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(diagnostic) is not BrowserAudioDiagnostic:
            raise TypeError("diagnostic must be an exact BrowserAudioDiagnostic")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError(
                    "audio diagnostic participant does not own the active session"
                )
            previous = self._audio_diagnostics.get(diagnostic.stream_id)
            if previous is not None:
                if diagnostic == previous:
                    return
                raise RuntimeError("audio diagnostic stream was replayed with different data")
            if len(self._audio_diagnostics) >= 1024:
                raise RuntimeError("audio diagnostic session capacity is exhausted")
            self._audio_diagnostics[diagnostic.stream_id] = diagnostic
            observer = self._observe_audio_diagnostic
        if observer is None:
            _LOGGER.info(
                "browser_audio_diagnostic generation=%d stream_id=%s "
                "supported_aec=%s supported_agc=%s supported_ns=%s supported_vi=%s "
                "applied_aec=%s applied_agc=%s applied_ns=%s applied_vi=%s "
                "subscribe_attach_ms=%d attach_playing_ms=%d playing_advance_ms=%d",
                generation,
                diagnostic.stream_id,
                diagnostic.supported_echo_cancellation,
                diagnostic.supported_auto_gain_control,
                diagnostic.supported_noise_suppression,
                diagnostic.supported_voice_isolation,
                diagnostic.applied_echo_cancellation,
                diagnostic.applied_auto_gain_control,
                diagnostic.applied_noise_suppression,
                diagnostic.applied_voice_isolation,
                diagnostic.subscribed_to_attach_ms,
                diagnostic.attach_to_playing_ms,
                diagnostic.playing_to_advance_ms,
            )
            return
        try:
            async with asyncio.timeout(0.1):
                await observer(identity, generation, diagnostic)
        except Exception:
            _LOGGER.warning("browser audio diagnostic observer failed", exc_info=True)

    async def submit_text(
        self,
        *,
        participant_identity: str,
        sequence: int,
        text: str,
    ) -> None:
        """Submit one sequenced typed turn using retained server-side authority."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(sequence) is not int:
            raise TypeError("sequence must be an exact integer")
        if not 1 <= sequence <= (1 << 63) - 1:
            raise ValueError("sequence is outside the supported range")
        if type(text) is not str:
            raise TypeError("text must be an exact built-in string")
        if not text.strip() or len(text) > 4096:
            raise ValueError("text must contain 1 to 4096 characters")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("typed participant does not own the active session")
            request = (sequence, text)
            if sequence == self._last_input_sequence and request == self._last_input_request:
                self._touch_activity()
                return
            if sequence != self._last_input_sequence + 1:
                raise RuntimeError("typed input sequence is not the next expected value")
            self._projection.ensure_capacity()
            await self._submit(identity, generation, sequence, text)
            # The submission already reached production authority, so this
            # sequence is spent whatever the evidence gate now says. Returning
            # without committing it desynchronises the browser permanently: the
            # caller advances to sequence + 1 while this session still expects
            # sequence, and every later typed turn is refused. The gate governs
            # only what may be projected, never input sequencing.
            with self._evidence_gate_lock:
                gate = self._evidence_ingress_gate
                admitted = (
                    gate is not None
                    and gate.binding.participant_identity == identity
                    and gate.binding.binding_generation == generation
                    and not gate.closed
                )
            if admitted:
                self._projection.publish(
                    "typed_input_admitted",
                    {"inputSequence": sequence},
                )
            self._last_input_sequence = sequence
            self._last_input_request = request
            self._touch_activity()

    async def yield_speech(
        self,
        *,
        participant_identity: str,
        turn_id: str,
        turn_generation: int,
        chunk_id: str,
        stream_id: str,
    ) -> bool:
        """Cancel only a browser-claimed exact active speech stream."""

        for name, value in (
            ("participant_identity", participant_identity),
            ("turn_id", turn_id),
            ("chunk_id", chunk_id),
            ("stream_id", stream_id),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact built-in string")
            if not value or len(value) > 128:
                raise ValueError(f"{name} must contain 1 to 128 characters")
        if type(turn_generation) is not int:
            raise TypeError("turn_generation must be an exact integer")
        if not 1 <= turn_generation <= (1 << 53) - 1:
            raise ValueError("turn_generation must be a positive safe integer")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("yield participant does not own the active session")
            callback = self._yield_speech
            if callback is None:
                raise RuntimeError("exact speech yield is unavailable")
        matched = await callback(
            identity, generation, turn_id, turn_generation, chunk_id, stream_id
        )
        if type(matched) is not bool:
            raise TypeError("yield_speech must return an exact boolean")
        async with self._start_lock:
            if self._active_identity == identity and self._active_generation == generation:
                self._touch_activity()
        return matched

    async def decide_approval(
        self,
        *,
        participant_identity: str,
        approval_id: str,
        sequence: int,
        decision: str,
    ) -> None:
        """Route one sequenced browser approval decision to injected authority."""

        if (
            type(participant_identity) is not str
            or type(approval_id) is not str
            or type(decision) is not str
        ):
            raise TypeError("approval values must be exact built-in strings")
        if not approval_id.startswith("approval_") or not 17 <= len(approval_id) <= 137:
            raise ValueError("approval_id is not a canonical public request identifier")
        if type(sequence) is not int:
            raise TypeError("sequence must be an exact integer")
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        if not 1 <= sequence <= (1 << 53) - 1:
            raise ValueError("sequence is outside the supported range")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("approval participant does not own the active session")
            request = (sequence, approval_id, decision)
            if (
                sequence == self._last_approval_sequence
                and request == self._last_approval_request
            ):
                self._touch_activity()
                return
            if sequence != self._last_approval_sequence + 1:
                raise RuntimeError("approval sequence is not the next expected value")
            self._projection.ensure_capacity()
            await self._approval(identity, generation, sequence, approval_id, decision)
            self._projection.publish(
                "approval_state",
                {
                    "actionable": False,
                    "approvalId": approval_id,
                    "state": decision,
                },
            )
            self._last_approval_sequence = sequence
            self._last_approval_request = request
            self._touch_activity()

    async def public_events_after(
        self,
        *,
        participant_identity: str,
        sequence: int,
    ) -> tuple[BrowserPublicEvent, ...]:
        """Return browser-safe events after acknowledging an observed prefix."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(sequence) is not int:
            raise TypeError("sequence must be an exact integer")
        async with self._start_lock:
            identity = self._active_identity
            if identity is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("event participant does not own the active session")
            self._projection.acknowledge_through(sequence)
            events = self._projection.events_after(sequence)
            self._touch_activity()
            return events

    async def refresh_credential(
        self,
        *,
        participant_identity: str,
    ) -> BrowserJoinCredential:
        """Issue another short-lived token only for the active browser identity."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        async with self._start_lock:
            identity = self._active_identity
            if identity is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != identity:
                raise PermissionError("refresh participant does not own the active session")
            credential = self._issuer.issue_for_identity(identity)
            self._touch_activity()
            return credential

    async def voice_configuration(
        self,
        *,
        participant_identity: str,
    ) -> tuple[tuple[str, ...], str | None]:
        async with self._start_lock:
            if self._active_identity is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != self._active_identity:
                raise PermissionError("voice participant does not own the active session")
            configuration = self._voice_configuration
            if configuration is None:
                return (), None
            voices, selected = configuration()
            if type(voices) is not tuple or any(type(voice) is not str for voice in voices):
                raise TypeError("voice configuration returned invalid voices")
            if selected is not None and (type(selected) is not str or selected not in voices):
                raise ValueError("voice configuration returned an invalid selection")
            self._touch_activity()
            return voices, selected

    async def selectable_model_configuration(
        self,
        *,
        participant_identity: str,
    ) -> BrowserModelCatalog:
        async with self._start_lock:
            identity, generation, operation_epoch = self._model_snapshot(participant_identity)
            catalog = self._model_catalog
            if catalog is None:
                raise RuntimeError("live model selection is unavailable")
        async with self._model_operation_lock:
            result = await catalog()
            if type(result) is not BrowserModelCatalog:
                raise TypeError("model catalog returned an invalid configuration")
            async with self._start_lock:
                self._require_current_model_snapshot(identity, generation, operation_epoch)
                self._touch_activity()
                return result

    async def change_model(
        self,
        *,
        participant_identity: str,
        model: str,
        effort: str,
    ) -> BrowserModelCatalog:
        if type(model) is not str or type(effort) is not str:
            raise TypeError("model and effort must be exact built-in strings")
        async with self._start_lock:
            identity, generation, operation_epoch = self._model_snapshot(participant_identity)
            selector = self._select_model
            if selector is None or self._model_configuration is None:
                raise RuntimeError("live model selection is unavailable")
        async with self._model_operation_lock:
            async with self._start_lock:
                self._require_current_model_snapshot(identity, generation, operation_epoch)
                selection = asyncio.ensure_future(selector(model, effort))
                self._pending_model_selection = selection
            try:
                result = await selection
            except asyncio.CancelledError as error:
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
                raise RuntimeError("model selection was cancelled") from error
            finally:
                async with self._start_lock:
                    if self._pending_model_selection is selection:
                        self._pending_model_selection = None
            if type(result) is not BrowserModelCatalog:
                raise TypeError("model selector returned an invalid configuration")
            if result.selected_model != model or result.selected_effort != effort:
                raise RuntimeError("model selector did not apply the requested configuration")
            async with self._start_lock:
                self._require_current_model_snapshot(identity, generation, operation_epoch)
                assert self._model_configuration is not None
                self._projection.ensure_capacity()
                self._model_configuration = replace(
                    self._model_configuration,
                    model=model,
                    effort=effort,
                    context_window_tokens=None,
                )
                self._projection.publish(
                    "session_model",
                    self._model_configuration.public_data(),
                )
                self._touch_activity()
                return result

    async def change_voice(self, *, participant_identity: str, voice: str) -> None:
        if type(voice) is not str:
            raise TypeError("voice must be an exact built-in string")
        async with self._start_lock:
            if self._active_identity is None:
                raise RuntimeError("no browser session is active")
            if participant_identity != self._active_identity:
                raise PermissionError("voice participant does not own the active session")
            configuration = self._voice_configuration
            selector = self._select_voice
            if configuration is None or selector is None:
                raise RuntimeError("live voice selection is unavailable")
            voices, _selected = configuration()
            if voice not in voices:
                raise ValueError("voice is not in the active provider catalog")
            await selector(voice)
            self._touch_activity()

    async def stop(
        self,
        *,
        participant_identity: str,
        request_id: str | None = None,
    ) -> None:
        """Stop the authenticated session, retaining state when cleanup fails."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if request_id is not None:
            if type(request_id) is not str:
                raise TypeError("request_id must be an exact built-in string or None")
            suffix = request_id.removeprefix("rebind_")
            if (
                not 16 <= len(request_id) <= 128
                or suffix == request_id
                or not suffix.isascii()
                or not suffix.replace("-", "").isalnum()
            ):
                raise ValueError("request_id must be a canonical rebind request identifier")
        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            if identity is None or generation is None:
                raise RuntimeError("no browser session is active")
            if (
                participant_identity != identity
                and (participant_identity, request_id) != self._previous_rebind_request
            ):
                raise PermissionError("stop participant does not own the active session")
            self._invalidate_model_operations()
            self._projection.ensure_capacity(
                2 if self._search_egress_authority is not None else 1
            )
            self._invalidate_search_egress(identity, generation)
            await self._invalidate_evidence(BindingCloseReason.CLIENT_CLOSED)
            await self._stop(identity, generation)
            self._projection.publish("session_stopped", {})
            self._clear_active_session()

    async def expire_if_inactive(self) -> bool:
        """Stop an abandoned browser lease after its bounded inactivity window."""

        async with self._start_lock:
            identity = self._active_identity
            generation = self._active_generation
            last_activity = self._last_activity
            if identity is None or generation is None or last_activity is None:
                return False
            value = self._clock()
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RuntimeError("session clock must return a finite exact number")
            now = float(value)
            if now < last_activity:
                raise RuntimeError("session clock regressed")
            if now - last_activity < self._inactivity_timeout:
                return False
            self._invalidate_model_operations()
            if self._search_egress_authority is not None:
                self._projection.ensure_capacity()
                self._invalidate_search_egress(identity, generation)
            await self._invalidate_evidence(BindingCloseReason.INACTIVITY_EXPIRED)
            await self._stop(identity, generation)
            with contextlib.suppress(RuntimeError):
                self._projection.publish("session_stopped", {"reason": "inactive"})
            self._clear_active_session()
            return True

    async def _settle_start_rollback(
        self,
        identity: str,
        generation: int,
    ) -> tuple[tuple[asyncio.CancelledError, ...], BaseException | None]:
        async def run_rollback() -> None:
            await self._stop(identity, generation)

        rollback = asyncio.create_task(
            run_rollback(),
            name=f"browser-session-start-rollback:{generation}",
        )
        current = asyncio.current_task()
        cancellations: list[asyncio.CancelledError] = []
        while not rollback.done():
            try:
                await asyncio.shield(rollback)
            except asyncio.CancelledError as error:
                if current is not None and current.cancelling() > 0:
                    current.uncancel()
                    cancellations.append(error)
                    continue
                break
            except BaseException:
                break
        try:
            rollback.result()
        except BaseException as error:
            return tuple(cancellations), error
        return tuple(cancellations), None

    async def _settle_rebind_rollback(
        self,
        reprovision: Callable[[str], Awaitable[int]],
        identity: str,
    ) -> tuple[tuple[asyncio.CancelledError, ...], int | None, BaseException | None]:
        async def run_rollback() -> int:
            return await reprovision(identity)

        rollback = asyncio.create_task(
            run_rollback(),
            name="browser-session-rebind-rollback",
        )
        current = asyncio.current_task()
        cancellations: list[asyncio.CancelledError] = []
        deadline = asyncio.get_running_loop().time() + self._rollback_timeout
        while not rollback.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self._begin_rebind_finalization(rollback, identity)
                return tuple(cancellations), None, TimeoutError("rebind rollback timed out")
            try:
                await asyncio.wait_for(asyncio.shield(rollback), timeout=remaining)
            except TimeoutError:
                self._begin_rebind_finalization(rollback, identity)
                return tuple(cancellations), None, TimeoutError("rebind rollback timed out")
            except asyncio.CancelledError as error:
                if current is not None and current.cancelling() > 0:
                    current.uncancel()
                    cancellations.append(error)
                    continue
                break
            except BaseException:
                break
        try:
            generation = rollback.result()
            if type(generation) is not int or generation <= 0:
                raise ValueError("rebind rollback must return a positive exact generation")
        except BaseException as error:
            return tuple(cancellations), None, error
        return tuple(cancellations), generation, None

    def _begin_rebind_finalization(
        self,
        rollback: asyncio.Task[int],
        identity: str,
    ) -> None:
        if self._rebind_finalizer is not None or self._rebind_finalization_failed:
            raise RuntimeError("browser session recovery finalization already exists")
        self._clear_active_session()
        self._rebind_finalizer = asyncio.create_task(
            self._finalize_rebind_rollback(rollback, identity),
            name="browser-session-rebind-finalizer",
        )

    async def _finalize_rebind_rollback(
        self,
        rollback: asyncio.Task[int],
        identity: str,
    ) -> None:
        failed = True
        try:
            generation = await rollback
            if type(generation) is not int or generation <= 0:
                raise ValueError("rebind rollback must return a positive exact generation")
            await self._stop(identity, generation)
            failed = False
        except BaseException:
            pass
        async with self._start_lock:
            self._rebind_finalization_failed = failed
            self._rebind_finalizer = None

    def _clear_active_session(self) -> None:
        with self._evidence_gate_lock:
            self._evidence_binding = None
            self._evidence_ingress_gate = None
            self._last_evidence_control_sequence = 0
        pending_consent = self._pending_evidence_consent
        if pending_consent is not None:
            pending_consent.cancel()
        pending_consent_settlement = self._pending_evidence_consent_settlement
        if pending_consent_settlement is not None:
            pending_consent_settlement.cancel()
        pending_revoke = self._pending_evidence_revoke
        if pending_revoke is not None:
            pending_revoke.cancel()
        self._pending_evidence_consent_key = None
        self._pending_evidence_consent_operation = None
        self._pending_evidence_consent = None
        self._pending_evidence_consent_settlement = None
        self._last_evidence_consent_key = None
        self._last_evidence_consent_result = None
        self._pending_evidence_revoke_key = None
        self._pending_evidence_revoke_operation = None
        self._pending_evidence_revoke = None
        self._last_evidence_revoke_key = None
        self._last_evidence_revoke_result = None
        self._active_identity = None
        self._active_generation = None
        self._active_media_incarnation = None
        self._previous_rebind_request = None
        self._previous_rebind_credential = None
        self._last_input_sequence = 0
        self._last_input_request = None
        self._last_approval_sequence = 0
        self._last_approval_request = None
        self._audio_diagnostics.clear()
        self._last_activity = None

    async def _invalidate_evidence(self, reason: BindingCloseReason) -> None:
        """Retire optional evidence before a binding can be replaced or stopped."""

        callback = self._evidence_invalidate
        if callback is not None:
            await callback(reason)

    def _invalidate_model_operations(self) -> None:
        self._model_operation_epoch += 1
        pending = self._pending_model_selection
        if pending is not None and not pending.done():
            pending.cancel()

    def _model_snapshot(self, participant_identity: str) -> tuple[str, int, int]:
        identity = self._active_identity
        generation = self._active_generation
        if identity is None or generation is None:
            raise RuntimeError("no browser session is active")
        if participant_identity != identity:
            raise PermissionError("model participant does not own the active session")
        return identity, generation, self._model_operation_epoch

    def _require_current_model_snapshot(
        self,
        identity: str,
        generation: int,
        operation_epoch: int,
    ) -> None:
        if (
            self._active_identity != identity
            or self._active_generation != generation
            or self._model_operation_epoch != operation_epoch
        ):
            raise RuntimeError("model operation became stale")
