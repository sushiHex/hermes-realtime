"""Internal capability for test-owned in-process qualification observation.

Raw qualification traces, their views, host registration, and all raw-content
serialization belong to ``tests.support.qualification``. Production objects can
only retain this opaque context and make the fixed record calls below.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, cast

from hermes_realtime.evidence.models import TerminalReason

_TOKEN = object()

_CHECKPOINTS = (
    "host_consent_active",
    "host_response_completed_before_shutdown",
    "host_drain_started",
)
_CHECKPOINT_TIMEOUT_SECONDS = 10.0
_CHECKPOINT_MAX_FRAME_BYTES = 192


def _canonical_checkpoint_frame(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


async def _read_checkpoint_line(*, fd: int, handle: int) -> bytes:
    import _winapi

    value = bytearray()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _CHECKPOINT_TIMEOUT_SECONDS
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("qualification checkpoint acknowledgment timed out")
        try:
            peek = cast(tuple[bytes, int, int], _winapi.PeekNamedPipe(handle, 1))
            available = peek[1]
        except OSError as error:
            raise RuntimeError("qualification checkpoint acknowledgment pipe failed") from error
        if not available:
            await asyncio.sleep(min(0.01, remaining))
            continue
        chunk = os.read(
            fd,
            min(available, _CHECKPOINT_MAX_FRAME_BYTES + 1 - len(value)),
        )
        if not chunk:
            raise RuntimeError("qualification checkpoint acknowledgment reached EOF")
        if b"\r" in chunk:
            raise RuntimeError("qualification checkpoint acknowledgment is noncanonical")
        value.extend(chunk)
        if len(value) > _CHECKPOINT_MAX_FRAME_BYTES:
            raise RuntimeError("qualification checkpoint acknowledgment is oversized")
        newline = value.find(b"\n")
        if newline >= 0:
            if newline != len(value) - 1:
                raise RuntimeError("qualification checkpoint acknowledgment has trailing bytes")
            return bytes(value)


def _load_checkpoint_document(value: bytes) -> dict[str, object]:
    try:
        text = value[:-1].decode("utf-8", errors="strict")

        def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = item
            return result

        document = json.loads(text, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("qualification checkpoint acknowledgment is malformed") from error
    if type(document) is not dict or _canonical_checkpoint_frame(document) != value:
        raise RuntimeError("qualification checkpoint acknowledgment is noncanonical")
    return document


class _QualificationCheckpointChannelV1:
    """One owned, ordered anonymous-pipe checkpoint channel for a CLI child."""

    __slots__ = (
        "_closed",
        "_failure_callback",
        "_failed",
        "_lock",
        "_next_ordinal",
        "_nonce",
        "_resume_fd",
        "_resume_handle",
        "_write_fd",
        "_write_handle",
    )

    def __init__(
        self,
        *,
        write_handle: int,
        resume_handle: int,
        nonce: str,
        failure_callback: Callable[[BaseException], None] | None,
        token: object,
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("qualification checkpoint channels are internally minted")
        if sys.platform != "win32":
            raise RuntimeError("qualification checkpoint channels require Windows")
        if (
            type(write_handle) is not int
            or type(resume_handle) is not int
            or write_handle <= 0
            or resume_handle <= 0
            or write_handle == resume_handle
            or type(nonce) is not str
            or len(nonce) != 64
            or any(character not in "0123456789abcdef" for character in nonce)
        ):
            raise ValueError("qualification checkpoint channel contract is invalid")
        import _winapi
        import msvcrt

        write_fd = -1
        resume_fd = -1
        owned_write_handle = write_handle
        owned_resume_handle = resume_handle
        try:
            os.set_handle_inheritable(owned_write_handle, False)
            os.set_handle_inheritable(owned_resume_handle, False)
            write_fd = msvcrt.open_osfhandle(owned_write_handle, os.O_WRONLY)
            owned_write_handle = -1
            resume_fd = msvcrt.open_osfhandle(owned_resume_handle, os.O_RDONLY)
            owned_resume_handle = -1
        except BaseException:
            for fd in (write_fd, resume_fd):
                if fd >= 0:
                    os.close(fd)
            for handle in (owned_write_handle, owned_resume_handle):
                if handle >= 0:
                    with suppress(OSError):
                        _winapi.CloseHandle(handle)
            raise
        self._write_handle = write_handle
        self._resume_handle = resume_handle
        self._write_fd = write_fd
        self._resume_fd = resume_fd
        self._nonce = nonce
        self._failure_callback = failure_callback
        self._next_ordinal = 1
        self._lock = asyncio.Lock()
        self._failed = False
        self._closed = False

    @property
    def write_handle(self) -> int:
        return self._write_handle

    @property
    def resume_handle(self) -> int:
        return self._resume_handle

    async def emit(self, checkpoint: str) -> None:
        async with self._lock:
            if self._closed or self._failed:
                raise RuntimeError("qualification checkpoint channel is unavailable")
            ordinal = self._next_ordinal
            if ordinal > len(_CHECKPOINTS) or checkpoint != _CHECKPOINTS[ordinal - 1]:
                error = RuntimeError("qualification checkpoint is duplicate or out of order")
                self._fail(error)
                raise error
            frame = _canonical_checkpoint_frame(
                {
                    "checkpoint": checkpoint,
                    "nonce": self._nonce,
                    "ordinal": ordinal,
                    "protocolVersion": 1,
                }
            )
            try:
                self._write_all(frame)
                acknowledgment = await _read_checkpoint_line(
                    fd=self._resume_fd,
                    handle=self._resume_handle,
                )
                document = _load_checkpoint_document(acknowledgment)
                if (
                    set(document) != {"nonce", "protocolVersion", "resumeOrdinal"}
                    or type(document["nonce"]) is not str
                    or document["nonce"] != self._nonce
                    or type(document["protocolVersion"]) is not int
                    or document["protocolVersion"] != 1
                    or type(document["resumeOrdinal"]) is not int
                    or document["resumeOrdinal"] != ordinal
                ):
                    raise RuntimeError("qualification checkpoint acknowledgment does not match")
            except BaseException as error:
                self._fail(error)
                raise
            self._next_ordinal += 1

    def _fail(self, error: BaseException) -> None:
        if self._failed:
            return
        self._failed = True
        terminal_errors = [error]
        try:
            self.close()
        except BaseException as close_error:
            terminal_errors.append(close_error)
        callback = self._failure_callback
        if callback is not None:
            try:
                callback(error)
            except BaseException as callback_error:
                terminal_errors.append(callback_error)
        if len(terminal_errors) > 1:
            raise BaseExceptionGroup(
                "qualification checkpoint terminal failure cleanup failed",
                terminal_errors,
            ) from None

    def _write_all(self, value: bytes) -> None:
        offset = 0
        while offset < len(value):
            written = os.write(self._write_fd, value[offset:])
            if written <= 0:
                raise RuntimeError("qualification checkpoint write did not progress")
            offset += written

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[OSError] = []
        for fd in (self._write_fd, self._resume_fd):
            try:
                os.close(fd)
            except OSError as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("qualification checkpoint channel close failed", errors)


def _new_qualification_checkpoint_channel(
    *,
    write_handle: int,
    resume_handle: int,
    nonce: str,
    failure_callback: Callable[[BaseException], None] | None = None,
) -> _QualificationCheckpointChannelV1:
    return _QualificationCheckpointChannelV1(
        write_handle=write_handle,
        resume_handle=resume_handle,
        nonce=nonce,
        failure_callback=failure_callback,
        token=_TOKEN,
    )


_ACTIVE_CHECKPOINT_CHANNEL: ContextVar[_QualificationCheckpointChannelV1 | None] = ContextVar(
    "hermes_realtime_qualification_checkpoint_channel",
    default=None,
)


def _current_qualification_checkpoint_channel() -> _QualificationCheckpointChannelV1 | None:
    return _ACTIVE_CHECKPOINT_CHANNEL.get()


@contextmanager
def _qualification_checkpoint_channel_scope(
    channel: _QualificationCheckpointChannelV1,
) -> Iterator[None]:
    if type(channel) is not _QualificationCheckpointChannelV1:
        raise TypeError("qualification checkpoint channel must be exact")
    if _ACTIVE_CHECKPOINT_CHANNEL.get() is not None:
        raise RuntimeError("a qualification checkpoint channel is already active")
    activation = _ACTIVE_CHECKPOINT_CHANNEL.set(channel)
    try:
        yield
    finally:
        _ACTIVE_CHECKPOINT_CHANNEL.reset(activation)


@dataclass(frozen=True, slots=True)
class _QualificationCapacityObservationV1:
    """Content-free controller-authored capacity fact for one owned host."""

    kind: str
    rejection_source: str
    queue_record_count: int
    queue_canonical_bytes: int
    queue_physical_count: int
    all_capacity_released: bool


class _QualificationCheckpointGateV1(Protocol):
    def set(self) -> None: ...

    def is_set(self) -> bool: ...


class _QualificationCapacityProbeV1:
    """Private append-only observer; it has neither production state nor authority."""

    __slots__ = (
        "_identity",
        "_incomplete",
        "_lock",
        "_records",
        "_rollover_publication_fault_armed",
        "_rollover_publication_fault_consumed",
        "_rollover_failure_checkpoint_armed",
        "_rollover_failure_checkpoint_consumed",
        "_rollover_failure_checkpoint_entered",
        "_rollover_failure_checkpoint_resume",
    )

    def __init__(self, *, identity: object, token: object) -> None:
        if token is not _TOKEN:
            raise TypeError("qualification capacity probes are capability-created")
        self._identity = identity
        self._records: tuple[_QualificationCapacityObservationV1, ...] = ()
        self._lock = Lock()
        self._incomplete = False
        self._rollover_publication_fault_armed = False
        self._rollover_publication_fault_consumed = False
        self._rollover_failure_checkpoint_armed = False
        self._rollover_failure_checkpoint_consumed = False
        self._rollover_failure_checkpoint_entered: _QualificationCheckpointGateV1 | None = None
        self._rollover_failure_checkpoint_resume: _QualificationCheckpointGateV1 | None = None

    def arm_rollover_publication_fault(self, identity: object) -> None:
        """Arm the one-shot pre-publication fault for this exact owner context."""

        dependencies = _current_qualification_full_host_dependencies()
        if (
            identity is not self._identity
            or dependencies is None
            or dependencies._capacity_probe is not self
        ):
            raise RuntimeError(
                "rollover publication fault arm requires its active owner context"
            )
        with self._lock:
            if (
                self._rollover_publication_fault_armed
                or self._rollover_publication_fault_consumed
            ):
                raise RuntimeError("rollover publication fault arm is single-use")
            self._rollover_publication_fault_armed = True

    def consume_rollover_publication_fault(self) -> bool:
        """Consume the arm once, recording no controller predicate or outcome."""

        with self._lock:
            if not self._rollover_publication_fault_armed:
                return False
            self._rollover_publication_fault_armed = False
            self._rollover_publication_fault_consumed = True
            return True

    def rollover_publication_fault_consumed(self) -> bool:
        with self._lock:
            return self._rollover_publication_fault_consumed

    def arm_rollover_failure_checkpoint(
        self,
        identity: object,
        *,
        entered: _QualificationCheckpointGateV1,
        resume: _QualificationCheckpointGateV1,
    ) -> None:
        """Arm one owner-bound pause after a non-durable rollover rejection."""

        dependencies = _current_qualification_full_host_dependencies()
        if (
            identity is not self._identity
            or dependencies is None
            or dependencies._capacity_probe is not self
        ):
            raise RuntimeError("rollover failure checkpoint requires its active owner context")
        if not callable(getattr(entered, "set", None)) or not callable(
            getattr(resume, "is_set", None)
        ):
            raise TypeError("rollover failure checkpoint gates are invalid")
        with self._lock:
            if (
                self._rollover_failure_checkpoint_armed
                or self._rollover_failure_checkpoint_consumed
            ):
                raise RuntimeError("rollover failure checkpoint is single-use")
            self._rollover_failure_checkpoint_entered = entered
            self._rollover_failure_checkpoint_resume = resume
            self._rollover_failure_checkpoint_armed = True

    async def pause_after_rollover_rejection(self) -> None:
        """Consume the private checkpoint, waiting no longer than two seconds."""

        with self._lock:
            if not self._rollover_failure_checkpoint_armed:
                return
            self._rollover_failure_checkpoint_armed = False
            self._rollover_failure_checkpoint_consumed = True
            entered = self._rollover_failure_checkpoint_entered
            resume = self._rollover_failure_checkpoint_resume
            self._rollover_failure_checkpoint_entered = None
            self._rollover_failure_checkpoint_resume = None
        assert entered is not None and resume is not None
        entered.set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while not resume.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.005)

    def rollover_failure_checkpoint_consumed(self) -> bool:
        with self._lock:
            return self._rollover_failure_checkpoint_consumed

    def record(
        self, *, kind: str, rejection_source: str, records: int, canonical_bytes: int,
        physical_items: int,
    ) -> None:
        kinds = {
            "ordinary_admitted", "ordinary_rejected", "ordinary_completed", "revoke_accepted",
            "revoke_coalesced", "revoke_terminal", "drain_accepted", "drain_coalesced",
            "drain_terminal",
        }
        if kind not in kinds:
            raise TypeError("qualification capacity observation kind is invalid")
        sources = {"record_capacity", "canonical_byte_capacity", "physical_capacity", "none"}
        if rejection_source not in sources:
            raise TypeError("qualification capacity rejection source is invalid")
        values = (records, canonical_bytes, physical_items)
        if any(type(value) is not int or value < 0 for value in values):
            raise TypeError("qualification capacity counts must be nonnegative exact integers")
        observation = _QualificationCapacityObservationV1(
            kind=kind, rejection_source=rejection_source, queue_record_count=records,
            queue_canonical_bytes=canonical_bytes, queue_physical_count=physical_items,
            all_capacity_released=records == canonical_bytes == physical_items == 0,
        )
        if not self._lock.acquire(blocking=False):
            self._incomplete = True
            return
        try:
            if len(self._records) >= 256:
                self._incomplete = True
                return
            self._records = (*self._records, observation)
        finally:
            self._lock.release()

    def observations(self) -> tuple[_QualificationCapacityObservationV1, ...]:
        with self._lock:
            if self._incomplete:
                raise RuntimeError("qualification capacity observations are incomplete")
            return self._records


_ACTIVE_CAPACITY_PROBE: ContextVar[_QualificationCapacityProbeV1 | None] = ContextVar(
    "hermes_realtime_qualification_capacity_probe",
    default=None,
)


class _QualificationOwnerCollectorV1(Protocol):
    """Test-owned sink protocol accepted by one matching opaque capability."""

    def _qualification_owner_identity_matches(self, identity: object) -> bool: ...

    def record_committed_conversation_context_snapshot(self, value: bytes) -> None: ...

    def record_generated_text(self, value: bytes) -> None: ...

    def record_transport_confirmed_chunk(self, value: bytes) -> None: ...

    def record_cancellation(self, reason: TerminalReason) -> None: ...

    def record_foreground_cleanup(self, *, succeeded: bool) -> None: ...


class _QualificationHostConstructionRegistryV1(Protocol):
    """Test-owned provenance registry usable only during host construction."""

    def _qualification_host_construction_identity_matches(self, identity: object) -> bool: ...

    def record_local_browser_launcher_construction(self, host: object) -> None: ...


class _QualificationFullHostDependenciesV1:
    """One exact-owner, construction-only set of deterministic provider factories."""

    __slots__ = (
        "_consumed",
        "_identity",
        "_capacity_probe",
        "identity_factory",
        "inference_factory",
        "speech_presence_factory",
        "synthesizer_factory",
        "transcriber_factory",
        "vad_factory",
        "writer_transport_factory",
    )

    def __init__(
        self,
        *,
        identity: object,
        inference_factory: Callable[[], object],
        speech_presence_factory: Callable[[], object],
        synthesizer_factory: Callable[[], object],
        transcriber_factory: Callable[[], object],
        vad_factory: Callable[[], object],
        identity_factory: Callable[[], str],
        writer_transport_factory: Callable[[object], object] | None,
        token: object,
    ) -> None:
        if token is not _TOKEN or type(identity) is not object:
            raise TypeError("qualification full-host dependencies are capability-created")
        if any(
            not callable(factory)
            for factory in (
                inference_factory,
                speech_presence_factory,
                synthesizer_factory,
                transcriber_factory,
                vad_factory,
                identity_factory,
            )
        ):
            raise TypeError("qualification full-host dependency factories must be callable")
        if writer_transport_factory is not None and not callable(writer_transport_factory):
            raise TypeError("qualification writer transport factory must be callable or None")
        self._identity = identity
        self._capacity_probe = _QualificationCapacityProbeV1(identity=identity, token=_TOKEN)
        self._consumed = False
        self.inference_factory = inference_factory
        self.speech_presence_factory = speech_presence_factory
        self.synthesizer_factory = synthesizer_factory
        self.transcriber_factory = transcriber_factory
        self.vad_factory = vad_factory
        self.identity_factory = identity_factory
        self.writer_transport_factory = writer_transport_factory

    def _qualification_full_host_dependencies_identity_matches(self, identity: object) -> bool:
        return identity is self._identity

    def take(self) -> _QualificationFullHostDependenciesV1:
        context = _ACTIVE_OWNER_CONTEXT.get()
        if context is None or context._full_host_dependencies is not self:
            raise RuntimeError(
                "qualification full-host dependencies require their active owner context"
            )
        if self._consumed:
            raise RuntimeError("qualification full-host dependencies are one-shot")
        self._consumed = True
        return self


_COLLECTOR_CALLS = (
    "record_committed_conversation_context_snapshot",
    "record_generated_text",
    "record_transport_confirmed_chunk",
    "record_cancellation",
    "record_foreground_cleanup",
)


class _QualificationOwnerContextV1:
    """Opaque production capability with no trace/view or host authority."""

    __slots__ = ("_collector", "_full_host_dependencies", "_host_construction_registry")

    def __init__(
        self,
        collector: _QualificationOwnerCollectorV1 | None,
        *,
        host_construction_registry: _QualificationHostConstructionRegistryV1 | None = None,
        full_host_dependencies: _QualificationFullHostDependenciesV1 | None = None,
        token: object,
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("qualification owner contexts are capability-created")
        self._collector = collector
        self._host_construction_registry = host_construction_registry
        self._full_host_dependencies = full_host_dependencies

    def record_committed_conversation_context_snapshot(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("committed conversation context snapshot must be exact bytes")
        collector = self._collector
        if collector is not None:
            collector.record_committed_conversation_context_snapshot(value)

    def record_generated_text(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("generated text must be exact bytes")
        collector = self._collector
        if collector is not None:
            collector.record_generated_text(value)

    def record_transport_confirmed_chunk(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("transport-confirmed chunk must be exact bytes")
        collector = self._collector
        if collector is not None:
            collector.record_transport_confirmed_chunk(value)

    def record_cancellation(self, reason: TerminalReason) -> None:
        if type(reason) is not TerminalReason:
            raise TypeError("cancellation reason must be an exact TerminalReason")
        collector = self._collector
        if collector is not None:
            collector.record_cancellation(reason)

    def record_foreground_cleanup(self, *, succeeded: bool) -> None:
        if type(succeeded) is not bool:
            raise TypeError("foreground cleanup result must be an exact bool")
        collector = self._collector
        if collector is not None:
            collector.record_foreground_cleanup(succeeded=succeeded)

    def record_local_browser_launcher_construction(self, host: object) -> None:
        registry = self._host_construction_registry
        if registry is not None:
            registry.record_local_browser_launcher_construction(host)


_EMPTY_OWNER_CONTEXT = _QualificationOwnerContextV1(None, token=_TOKEN)
_ACTIVE_OWNER_CONTEXT: ContextVar[_QualificationOwnerContextV1 | None] = ContextVar(
    "hermes_realtime_qualification_owner_context",
    default=None,
)


class _QualificationOwnerContextCapabilityV1:
    """Internal test-only installer for one exact raw-trace owner."""

    __slots__ = ("_identity",)

    def __init__(self, identity: object, *, token: object) -> None:
        if token is not _TOKEN:
            raise TypeError("qualification owner capabilities are internally minted")
        if type(identity) is not object:
            raise TypeError("qualification owner identity must be an opaque exact object")
        self._identity = identity

    @contextmanager
    def wire(
        self,
        collector: _QualificationOwnerCollectorV1,
        host_construction_registry: _QualificationHostConstructionRegistryV1 | None = None,
        full_host_dependencies: _QualificationFullHostDependenciesV1 | None = None,
    ) -> Iterator[None]:
        if not callable(getattr(collector, "_qualification_owner_identity_matches", None)):
            raise TypeError("qualification collector lacks its exact owner check")
        if collector._qualification_owner_identity_matches(self._identity) is not True:
            raise ValueError("qualification collector belongs to a different owner")
        if any(not callable(getattr(collector, method, None)) for method in _COLLECTOR_CALLS):
            raise TypeError("qualification collector lacks a required record call")
        if host_construction_registry is not None:
            if not callable(
                getattr(
                    host_construction_registry,
                    "_qualification_host_construction_identity_matches",
                    None,
                )
            ) or not callable(
                getattr(
                    host_construction_registry,
                    "record_local_browser_launcher_construction",
                    None,
                )
            ):
                raise TypeError("qualification host registry lacks its exact owner check")
            if (
                host_construction_registry._qualification_host_construction_identity_matches(
                    self._identity
                )
                is not True
            ):
                raise ValueError("qualification host registry belongs to a different owner")
        if _ACTIVE_OWNER_CONTEXT.get() is not None:
            raise RuntimeError("a qualification owner context is already active")
        if full_host_dependencies is not None:
            if type(full_host_dependencies) is not _QualificationFullHostDependenciesV1:
                raise TypeError("qualification full-host dependencies must be exact")
            if not full_host_dependencies._qualification_full_host_dependencies_identity_matches(
                self._identity
            ):
                raise ValueError("qualification full-host dependencies belong to a different owner")
        activation = _ACTIVE_OWNER_CONTEXT.set(
            _QualificationOwnerContextV1(
                collector,
                host_construction_registry=host_construction_registry,
                full_host_dependencies=full_host_dependencies,
                token=_TOKEN,
            )
        )
        try:
            yield
        finally:
            _ACTIVE_OWNER_CONTEXT.reset(activation)


def _new_qualification_owner_context_capability(
    identity: object,
) -> _QualificationOwnerContextCapabilityV1:
    """Mint the only context installer accepted by production record calls."""

    return _QualificationOwnerContextCapabilityV1(identity, token=_TOKEN)


def _new_qualification_full_host_dependencies(
    *,
    identity: object,
    inference_factory: Callable[[], object],
    speech_presence_factory: Callable[[], object],
    synthesizer_factory: Callable[[], object],
    transcriber_factory: Callable[[], object],
    vad_factory: Callable[[], object],
    identity_factory: Callable[[], str],
    writer_transport_factory: Callable[[object], object] | None = None,
) -> _QualificationFullHostDependenciesV1:
    """Mint closed deterministic slots for full-host construction only."""

    return _QualificationFullHostDependenciesV1(
        identity=identity,
        inference_factory=inference_factory,
        speech_presence_factory=speech_presence_factory,
        synthesizer_factory=synthesizer_factory,
        transcriber_factory=transcriber_factory,
        vad_factory=vad_factory,
        identity_factory=identity_factory,
        writer_transport_factory=writer_transport_factory,
        token=_TOKEN,
    )


def _current_qualification_owner_context() -> _QualificationOwnerContextV1:
    """Return the optional construction-time context without exposing raw state."""

    return _ACTIVE_OWNER_CONTEXT.get() or _EMPTY_OWNER_CONTEXT


def _current_qualification_full_host_dependencies() -> _QualificationFullHostDependenciesV1 | None:
    """Return the optional construction bundle; only ``take`` can consume it."""

    context = _ACTIVE_OWNER_CONTEXT.get()
    return None if context is None else context._full_host_dependencies


def _current_qualification_capacity_probe() -> _QualificationCapacityProbeV1 | None:
    """Return only the construction-bound opaque recorder conduit."""

    active = _ACTIVE_CAPACITY_PROBE.get()
    if active is not None:
        return active
    dependencies = _current_qualification_full_host_dependencies()
    return None if dependencies is None else dependencies._capacity_probe


@contextmanager
def _qualification_capacity_probe_scope(
    probe: _QualificationCapacityProbeV1 | None,
) -> Iterator[None]:
    """Transfer one capability-created probe across deferred controller construction."""

    if probe is not None and type(probe) is not _QualificationCapacityProbeV1:
        raise TypeError("qualification capacity probe must be exact or None")
    if _ACTIVE_CAPACITY_PROBE.get() is not None:
        raise RuntimeError("a qualification capacity probe is already active")
    activation = _ACTIVE_CAPACITY_PROBE.set(probe)
    try:
        yield
    finally:
        _ACTIVE_CAPACITY_PROBE.reset(activation)


def _record_qualification_local_browser_launcher_construction(host: object) -> None:
    """Offer an exact launcher to the optional active test-owned registry."""

    _current_qualification_owner_context().record_local_browser_launcher_construction(host)
