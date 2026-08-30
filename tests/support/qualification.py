"""Test-only raw qualification trace owner and host registration boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from threading import Event, Lock
from typing import Protocol, TypeVar, cast

from hermes_realtime._qualification import (
    _new_qualification_full_host_dependencies,
    _new_qualification_owner_context_capability,
)
from hermes_realtime.evidence.models import TerminalReason
from hermes_realtime.production_observation import (
    ProductionObservationStatusV1,
    ProductionObservationViewV1,
)

_MAX_RECORDS = 256
_HOST_TOKEN = object()
_T = TypeVar("_T")


class _QualificationKindV1(StrEnum):
    COMMITTED_CONVERSATION_CONTEXT_SNAPSHOT = "committed_conversation_context_snapshot"
    GENERATED_TEXT = "generated_text"
    TRANSPORT_CONFIRMED_CHUNK = "transport_confirmed_chunk"
    CANCELLATION = "cancellation"
    FOREGROUND_CLEANUP = "foreground_cleanup"
    HOST_RETURN = "host_return"


class _HostReturnOutcomeV1(StrEnum):
    CANCELLED = "cancelled"
    FAILED = "failed"
    RETURNED = "returned"


@dataclass(frozen=True, slots=True)
class _CommittedConversationContextSnapshotQualificationObservationV1:
    committed_conversation_context_snapshot: bytes

    @property
    def kind(self) -> _QualificationKindV1:
        return _QualificationKindV1.COMMITTED_CONVERSATION_CONTEXT_SNAPSHOT


@dataclass(frozen=True, slots=True)
class _GeneratedTextQualificationObservationV1:
    generated_text: bytes

    @property
    def kind(self) -> _QualificationKindV1:
        return _QualificationKindV1.GENERATED_TEXT


@dataclass(frozen=True, slots=True)
class _TransportConfirmedChunkQualificationObservationV1:
    confirmed_text: bytes

    @property
    def kind(self) -> _QualificationKindV1:
        return _QualificationKindV1.TRANSPORT_CONFIRMED_CHUNK


@dataclass(frozen=True, slots=True)
class _CancellationQualificationObservationV1:
    reason: TerminalReason

    @property
    def kind(self) -> _QualificationKindV1:
        return _QualificationKindV1.CANCELLATION


@dataclass(frozen=True, slots=True)
class _ForegroundCleanupQualificationObservationV1:
    succeeded: bool

    @property
    def kind(self) -> _QualificationKindV1:
        return _QualificationKindV1.FOREGROUND_CLEANUP


@dataclass(frozen=True, slots=True)
class _HostReturnQualificationObservationV1:
    outcome: _HostReturnOutcomeV1

    @property
    def kind(self) -> _QualificationKindV1:
        return _QualificationKindV1.HOST_RETURN


_QualificationObservationV1 = (
    _CommittedConversationContextSnapshotQualificationObservationV1
    | _GeneratedTextQualificationObservationV1
    | _TransportConfirmedChunkQualificationObservationV1
    | _CancellationQualificationObservationV1
    | _ForegroundCleanupQualificationObservationV1
    | _HostReturnQualificationObservationV1
)


@dataclass(slots=True)
class _QualificationTraceState:
    records: tuple[_QualificationObservationV1, ...]
    lock: Lock
    incomplete: Event
    host_return: _HostReturnQualificationObservationV1 | None
    host_return_lock: Lock


class _QualificationTraceV1:
    __slots__ = ("_state",)

    def __init__(self, state: _QualificationTraceState) -> None:
        self._state = state

    def records(self) -> tuple[_QualificationObservationV1, ...]:
        with self._state.lock:
            ordinary = self._state.records
        with self._state.host_return_lock:
            terminal = self._state.host_return
        return ordinary if terminal is None else (*ordinary, terminal)

    def status(self) -> ProductionObservationStatusV1:
        return ProductionObservationStatusV1(trace_complete=not self._state.incomplete.is_set())

    def _record_host_return_once(self, outcome: _HostReturnOutcomeV1) -> None:
        with self._state.host_return_lock:
            if self._state.host_return is None:
                self._state.host_return = _HostReturnQualificationObservationV1(outcome)


def committed_conversation_context_snapshot_bytes(
    *,
    revision: int,
    messages: tuple[tuple[str, str], ...],
    active_tasks: tuple[tuple[str, str], ...],
    terminal_task_count: int,
    updates: tuple[tuple[int, str, str, str], ...] = (),
) -> bytes:
    """Test-owner serialization of the exact conversation-adapter snapshot."""

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


class _QualificationTraceCollectorV1:
    """The sole raw-content owner installed through the source capability."""

    __slots__ = ("_identity", "_state")

    def __init__(self, *, identity: object, state: _QualificationTraceState) -> None:
        self._identity = identity
        self._state = state

    def _qualification_owner_identity_matches(self, identity: object) -> bool:
        return identity is self._identity

    def _append(self, record: _QualificationObservationV1) -> None:
        if not self._state.lock.acquire(blocking=False):
            self._state.incomplete.set()
            return
        try:
            if len(self._state.records) >= _MAX_RECORDS:
                self._state.incomplete.set()
                return
            self._state.records = (*self._state.records, record)
        finally:
            self._state.lock.release()

    def record_committed_conversation_context_snapshot(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("committed conversation context snapshot must be exact bytes")
        self._append(_CommittedConversationContextSnapshotQualificationObservationV1(value))

    def record_generated_text(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("generated text must be exact bytes")
        self._append(_GeneratedTextQualificationObservationV1(value))

    def record_transport_confirmed_chunk(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("transport-confirmed chunk must be exact bytes")
        self._append(_TransportConfirmedChunkQualificationObservationV1(value))

    def record_cancellation(self, reason: TerminalReason) -> None:
        if type(reason) is not TerminalReason:
            raise TypeError("cancellation reason must be exact")
        self._append(_CancellationQualificationObservationV1(reason))

    def record_foreground_cleanup(self, *, succeeded: bool) -> None:
        if type(succeeded) is not bool:
            raise TypeError("foreground cleanup result must be exact")
        self._append(_ForegroundCleanupQualificationObservationV1(succeeded))


def _new_trace_pair() -> tuple[_QualificationTraceV1, _QualificationTraceCollectorV1]:
    identity = object()
    state = _QualificationTraceState(
        records=(),
        lock=Lock(),
        incomplete=Event(),
        host_return=None,
        host_return_lock=Lock(),
    )
    return _QualificationTraceV1(state), _QualificationTraceCollectorV1(
        identity=identity,
        state=state,
    )


class _QualificationHostLifecycle(Protocol):
    async def start(self) -> str: ...

    async def close(self) -> None: ...


class _QualificationHostConstructionRegistryV1:
    """One-shot, test-owned proof that a launcher was constructed for this bundle."""

    __slots__ = ("_active_attempt", "_identity", "_lock", "_records")

    def __init__(self, identity: object) -> None:
        self._identity = identity
        self._lock = Lock()
        self._active_attempt: object | None = None
        self._records: dict[object, object] = {}

    def _qualification_host_construction_identity_matches(self, identity: object) -> bool:
        return identity is self._identity

    @contextmanager
    def constructing(self) -> Iterator[object]:
        attempt = object()
        with self._lock:
            if self._active_attempt is not None:
                raise RuntimeError("a qualification host construction is already active")
            self._active_attempt = attempt
        try:
            yield attempt
        finally:
            with self._lock:
                stale = tuple(
                    host
                    for host, recorded_attempt in self._records.items()
                    if recorded_attempt is attempt
                )
                for host in stale:
                    del self._records[host]
                self._active_attempt = None

    def record_local_browser_launcher_construction(self, host: object) -> None:
        with self._lock:
            attempt = self._active_attempt
            if attempt is None:
                raise RuntimeError("qualification launcher was constructed outside registration")
            self._records[host] = attempt

    def consume(self, host: object, attempt: object) -> bool:
        with self._lock:
            if self._records.get(host) is not attempt:
                return False
            del self._records[host]
            return True


class _QualificationRejectingHostConstructionRegistryV1:
    """Owner-matched context registry that forbids unregistered host creation."""

    __slots__ = ("_identity",)

    def __init__(self, identity: object) -> None:
        self._identity = identity

    def _qualification_host_construction_identity_matches(self, identity: object) -> bool:
        return identity is self._identity

    def record_local_browser_launcher_construction(self, host: object) -> None:
        del host
        raise RuntimeError("qualification host construction requires compose_host()")


class _QualificationHostRegistrationV1:
    """Test-owned registration of one host made during its matching context."""

    __slots__ = ("_host", "_identity", "_started", "_capacity_probe")

    def __init__(
        self,
        *,
        identity: object,
        host: _QualificationHostLifecycle,
        token: object,
    ) -> None:
        if token is not _HOST_TOKEN:
            raise TypeError("qualification host registrations are test-owner minted")
        self._identity = identity
        self._host = host
        self._started = False
        self._capacity_probe: object | None = None


class _QualificationRunningHostV1:
    """One owner-bound interval around an already started registered host."""

    __slots__ = ("_closed", "_host", "_identity", "_url", "_capacity_probe")

    def __init__(
        self,
        *,
        identity: object,
        host: _QualificationHostLifecycle,
        url: str,
        token: object,
    ) -> None:
        if token is not _HOST_TOKEN:
            raise TypeError("qualification running hosts are test-owner minted")
        self._identity = identity
        self._host = host
        self._url = url
        self._capacity_probe: object | None = None
        self._closed = False

    @property
    def url(self) -> str:
        return self._url


class InProcessQualificationComposition:
    """Own one raw trace and only compose/register real production hosts."""

    __slots__ = (
        "_capability",
        "_collector",
        "_host_construction_registry",
        "_identity",
        "_rejecting_host_construction_registry",
        "_trace",
    )

    def __init__(self) -> None:
        self._identity = object()
        state = _QualificationTraceState(
            records=(),
            lock=Lock(),
            incomplete=Event(),
            host_return=None,
            host_return_lock=Lock(),
        )
        self._trace = _QualificationTraceV1(state)
        self._collector = _QualificationTraceCollectorV1(
            identity=self._identity,
            state=state,
        )
        self._capability = _new_qualification_owner_context_capability(self._identity)
        self._host_construction_registry = _QualificationHostConstructionRegistryV1(
            self._identity
        )
        self._rejecting_host_construction_registry = (
            _QualificationRejectingHostConstructionRegistryV1(self._identity)
        )

    @property
    def trace(self) -> _QualificationTraceV1:
        return self._trace

    @contextmanager
    def wire(self) -> Iterator[None]:
        with self._capability.wire(self._collector):
            yield

    def compose(self, factory: Callable[[], _T]) -> _T:
        if not callable(factory):
            raise TypeError("qualification composition factory must be callable")
        with self._capability.wire(
            self._collector,
            self._rejecting_host_construction_registry,
        ):
            return factory()

    def compose_host(self, factory: Callable[[], object]) -> _QualificationHostRegistrationV1:
        """Register only an exact production launcher constructed in this context."""

        if not callable(factory):
            raise TypeError("qualification host factory must be callable")
        from hermes_realtime.launcher import LocalBrowserLauncher

        with self._host_construction_registry.constructing() as attempt:
            with self._capability.wire(
                self._collector,
                self._host_construction_registry,
            ):
                host = factory()
            if (
                type(host) is not LocalBrowserLauncher
                or not self._host_construction_registry.consume(host, attempt)
            ):
                raise TypeError(
                    "qualification host must be an exact composed LocalBrowserLauncher"
                )
        return _QualificationHostRegistrationV1(
            identity=self._identity,
            host=cast(_QualificationHostLifecycle, host),
            token=_HOST_TOKEN,
        )

    def compose_full_host(
        self,
        factory: Callable[[], object],
        *,
        inference_factory: Callable[[], object],
        speech_presence_factory: Callable[[], object],
        synthesizer_factory: Callable[[], object],
        transcriber_factory: Callable[[], object],
        vad_factory: Callable[[], object],
        identity_factory: Callable[[], str],
        writer_transport_factory: Callable[[object], object] | None = None,
    ) -> _QualificationHostRegistrationV1:
        """Compose one registered host with its exact owner-bound provider bundle."""

        if not callable(factory):
            raise TypeError("qualification host factory must be callable")
        dependencies = _new_qualification_full_host_dependencies(
            identity=self._identity,
            inference_factory=inference_factory,
            speech_presence_factory=speech_presence_factory,
            synthesizer_factory=synthesizer_factory,
            transcriber_factory=transcriber_factory,
            vad_factory=vad_factory,
            identity_factory=identity_factory,
            writer_transport_factory=writer_transport_factory,
        )
        from hermes_realtime.launcher import LocalBrowserLauncher

        with self._host_construction_registry.constructing() as attempt:
            with self._capability.wire(
                self._collector,
                self._host_construction_registry,
                full_host_dependencies=dependencies,
            ):
                host = factory()
            if (
                type(host) is not LocalBrowserLauncher
                or not self._host_construction_registry.consume(host, attempt)
            ):
                raise TypeError(
                    "qualification host must be an exact composed LocalBrowserLauncher"
                )
        registration = _QualificationHostRegistrationV1(
            identity=self._identity,
            host=cast(_QualificationHostLifecycle, host),
            token=_HOST_TOKEN,
        )
        registration._capacity_probe = dependencies._capacity_probe
        return registration

    def capacity_observations(self, running: _QualificationRunningHostV1) -> tuple[object, ...]:
        """Return only immutable capacity facts for this exact running host lease."""

        if type(running) is not _QualificationRunningHostV1:
            raise TypeError("qualification capacity observations require a test-owner running host")
        if running._identity is not self._identity:
            raise ValueError("running host must belong to the same qualification bundle")
        probe = running._capacity_probe
        observations = getattr(probe, "observations", None)
        if probe is None or not callable(observations):
            raise RuntimeError("qualification host has no capacity observation probe")
        return cast(tuple[object, ...], observations())

    def production_observations(
        self,
        running: _QualificationRunningHostV1,
    ) -> ProductionObservationViewV1:
        """Return only this lease's existing immutable production observation view."""

        if type(running) is not _QualificationRunningHostV1:
            raise TypeError("qualification observations require a test-owner running host")
        if running._identity is not self._identity:
            raise ValueError("running host must belong to the same qualification bundle")
        from hermes_realtime.launcher import LocalBrowserLauncher

        host = running._host
        if type(host) is not LocalBrowserLauncher:
            raise TypeError("qualification observations require an exact composed launcher")
        return host.production_observations

    async def start_host(
        self,
        registration: _QualificationHostRegistrationV1,
    ) -> _QualificationRunningHostV1:
        if type(registration) is not _QualificationHostRegistrationV1:
            raise TypeError("qualification start requires a test-owner host registration")
        if registration._identity is not self._identity:
            raise ValueError("host registration must belong to the same qualification bundle")
        if registration._started:
            raise RuntimeError("qualification host registration is one-shot")
        registration._started = True
        try:
            url = await registration._host.start()
        except asyncio.CancelledError:
            self._trace._record_host_return_once(_HostReturnOutcomeV1.CANCELLED)
            raise
        except BaseException:
            self._trace._record_host_return_once(_HostReturnOutcomeV1.FAILED)
            raise
        running = _QualificationRunningHostV1(
            identity=self._identity,
            host=registration._host,
            url=url,
            token=_HOST_TOKEN,
        )
        running._capacity_probe = registration._capacity_probe
        return running

    async def close_host(self, running: _QualificationRunningHostV1) -> None:
        if type(running) is not _QualificationRunningHostV1:
            raise TypeError("qualification close requires a test-owner running host")
        if running._identity is not self._identity:
            raise ValueError("running host must belong to the same qualification bundle")
        if running._closed:
            raise RuntimeError("qualification running host is already closed")
        running._closed = True
        outcome = _HostReturnOutcomeV1.FAILED
        try:
            await running._host.close()
        except asyncio.CancelledError:
            outcome = _HostReturnOutcomeV1.CANCELLED
            raise
        except BaseException:
            raise
        else:
            outcome = _HostReturnOutcomeV1.RETURNED
        finally:
            self._trace._record_host_return_once(outcome)

    async def retry_owned_close(self, running: _QualificationRunningHostV1) -> None:
        """Join only this owner's already-started real host close operation."""

        if type(running) is not _QualificationRunningHostV1:
            raise TypeError("qualification close requires a test-owner running host")
        if running._identity is not self._identity:
            raise ValueError("running host must belong to the same qualification bundle")
        if not running._closed:
            raise RuntimeError("qualification running host close was not started")
        await running._host.close()

    async def run_host(self, registration: _QualificationHostRegistrationV1) -> str:
        running = await self.start_host(registration)
        await self.close_host(running)
        return running.url
