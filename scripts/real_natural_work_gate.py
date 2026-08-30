"""Run the installed Codex-to-Hermes natural-work and latency release gates."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from scripts.real_gate_support import available_port, load_api_key
elif __package__:
    from .real_gate_support import available_port, load_api_key
else:
    from real_gate_support import available_port, load_api_key

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationTaskController,
    ConversationWorkControlSurface,
    WorkCancelResult,
    WorkStartResult,
)
from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    SubprocessCodexJsonLineTransport,
    _resolve_codex_executable,
)
from hermes_realtime.speech import Transcript

_ARM_ABSENT = "absent"
_ARM_PRESENT = "present"
_MIN_SAMPLE_PAIRS = 30
_MAX_SAMPLE_PAIRS = 1000
_MIN_WARMUPS_PER_ARM = 5
_MAX_WARMUPS_PER_ARM = 100
_MAX_TASK_ACK_BUDGET_MS = 60_000.0
_TIMED_TURN_TIMEOUT_SECONDS = 180.0
_TURN_CANCEL_DRAIN_SECONDS = 5.0
_CLOSE_TIMEOUT_SECONDS = 30.0
_CLOSE_WATCHDOG_GRACE_SECONDS = 1.0
_MAX_VERSION_OUTPUT_BYTES = 64 * 1024
_IDENTIFIER_MAX_CHARS = 256
_WORK_ACTIVITY_METHOD = "item/tool/call"
_WORK_ACTIVITY_ITEM_TYPE = "dynamicToolCall"
_VERSION_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)
_PRIVATE_AUTHORITY = re.compile(
    r"(?:deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*|"
    r"(?<![A-Za-z0-9_-])run_[A-Za-z0-9][A-Za-z0-9_-]{15,127}(?![A-Za-z0-9_-]))"
)
_CODEX_PROTOCOL_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "codex_app_server_dynamic_tools.json"
)
_MAX_PUBLIC_REPORT_BYTES = 256 * 1024
_MAX_PUBLIC_STRING_CHARS = 4096
_VERSION = re.compile(r"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?)")
_CONVERSATIONAL_PROMPTS = (
    "Reply with exactly the word ready.",
    "Answer in one word: what color is a clear daytime sky?",
    "Reply with the single word acknowledged.",
    "In one word, is water wet?",
)
_ResultT = TypeVar("_ResultT")


class _CleanupFailures(BaseExceptionGroup):
    """Identify failures raised solely by bounded cleanup."""


class _PrimaryFailureGroup(BaseExceptionGroup):
    """Keep the primary failure first when secondary failures also occur."""


class _GatePhaseFailure(RuntimeError):
    """Attach a bounded public phase label without exposing exception text."""

    def __init__(self, stage: str, error: BaseException) -> None:
        super().__init__(stage)
        self.stage = stage
        self.error = error


class _BoundedDiscardSink(io.TextIOBase):
    """Discard runtime output while retaining only a saturated byte count."""

    def __init__(self, *, maximum_bytes: int = _MAX_VERSION_OUTPUT_BYTES) -> None:
        super().__init__()
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("discard sink byte budget must be positive")
        self._maximum_bytes = maximum_bytes
        self._discarded_bytes = 0

    @property
    def discarded_bytes(self) -> int:
        return self._discarded_bytes

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if type(value) is not str:
            raise TypeError("runtime output must be text")
        encoded_size = len(value.encode("utf-8", errors="replace"))
        self._discarded_bytes = min(
            self._maximum_bytes,
            self._discarded_bytes + encoded_size,
        )
        return len(value)


@contextmanager
def _discard_runtime_output(sink: _BoundedDiscardSink) -> Iterator[None]:
    """Discard Python and native runtime output while preserving the CLI channels."""

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        with redirect_stdout(sink), redirect_stderr(sink):
            yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


@dataclass(frozen=True, slots=True)
class LatencySample:
    """One sanitized conversational timing sample in milliseconds."""

    transcript_to_thread_ms: float
    thread_to_first_delta_ms: float
    first_delta_to_speakable_ms: float

    def __post_init__(self) -> None:
        for value in (
            self.transcript_to_thread_ms,
            self.thread_to_first_delta_ms,
            self.first_delta_to_speakable_ms,
        ):
            _validate_latency(value)

    @property
    def foreground_total_ms(self) -> float:
        return (
            float(self.transcript_to_thread_ms)
            + float(self.thread_to_first_delta_ms)
            + float(self.first_delta_to_speakable_ms)
        )


@dataclass(frozen=True, slots=True)
class TaskAcknowledgementSample:
    """Separate request-to-accept time from installed-handler acceptance time."""

    request_to_accept_ms: float
    handler_to_accept_ms: float

    def __post_init__(self) -> None:
        _validate_latency(self.request_to_accept_ms)
        _validate_latency(self.handler_to_accept_ms)
        if self.handler_to_accept_ms > self.request_to_accept_ms:
            raise ValueError("handler acknowledgement cannot exceed request acknowledgement")

    @property
    def decision_and_tool_call_ms(self) -> float:
        return float(self.request_to_accept_ms) - float(self.handler_to_accept_ms)


class TurnTimingProbe:
    """Capture one foreground turn at exact observable protocol boundaries."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._transcript_at: float | None = None
        self._thread_started_at: float | None = None
        self._first_delta_at: float | None = None
        self._first_speakable_at: float | None = None
        self._thread_request_id: int | str | None = None
        self._turn_request_id: int | str | None = None
        self._thread_id: str | None = None
        self._turn_id: str | None = None

    def begin(self, *, transcript_at: float | None = None) -> None:
        if transcript_at is None:
            transcript_at = self._clock()
        if type(transcript_at) not in (int, float) or not math.isfinite(transcript_at):
            raise ValueError("transcript timestamp must be finite")
        self._transcript_at = float(transcript_at)
        self._thread_started_at = None
        self._first_delta_at = None
        self._first_speakable_at = None
        self._thread_request_id = None
        self._turn_request_id = None
        self._thread_id = None
        self._turn_id = None

    def observe_send(self, message: Mapping[str, object]) -> None:
        method = message.get("method")
        if method not in ("thread/start", "turn/start"):
            return
        if self._transcript_at is None:
            raise RuntimeError("timing request arrived outside the current turn")
        request_id = _request_identity(message.get("id"), f"{method} request")
        if method == "thread/start":
            if (
                self._thread_request_id is not None
                or self._thread_id is not None
                or self._turn_request_id is not None
            ):
                raise RuntimeError("turn emitted duplicate thread/start requests")
            self._thread_request_id = request_id
            return
        if self._thread_id is None or self._thread_request_id is not None:
            raise RuntimeError("turn/start preceded the current thread/start response")
        if self._turn_request_id is not None or self._turn_id is not None:
            raise RuntimeError("turn emitted duplicate turn/start requests")
        params = _exact_mapping(message.get("params"), "turn/start params")
        thread_id = _bounded_identifier(params.get("threadId"), "turn/start thread id")
        if thread_id != self._thread_id:
            raise RuntimeError("turn/start targeted a foreign thread")
        self._turn_request_id = request_id

    def observe_receive(self, message: Mapping[str, object]) -> None:
        request_id = message.get("id")
        if self._thread_request_id is not None and request_id == self._thread_request_id:
            if self._thread_started_at is not None:
                raise RuntimeError("turn emitted duplicate thread/start responses")
            result = _exact_response_result(message, "thread/start")
            thread = _exact_mapping(result.get("thread"), "thread/start thread")
            self._thread_id = _bounded_identifier(thread.get("id"), "thread/start thread id")
            self._thread_started_at = self._clock()
            self._thread_request_id = None
            return
        if self._turn_request_id is not None and request_id == self._turn_request_id:
            result = _exact_response_result(message, "turn/start")
            turn = _exact_mapping(result.get("turn"), "turn/start turn")
            self._turn_id = _bounded_identifier(turn.get("id"), "turn/start turn id")
            self._turn_request_id = None
            return
        if message.get("method") != "item/agentMessage/delta":
            return
        params = _exact_mapping(message.get("params"), "agent delta params")
        thread_id = _bounded_identifier(params.get("threadId"), "agent delta thread id")
        turn_id = _bounded_identifier(params.get("turnId"), "agent delta turn id")
        if self._thread_id is None or self._turn_id is None:
            raise RuntimeError("agent delta preceded the current turn/start response")
        if thread_id != self._thread_id or turn_id != self._turn_id:
            raise RuntimeError("agent delta belongs to a stale or foreign turn")
        if self._first_delta_at is None:
            self._first_delta_at = self._clock()

    def mark_first_speakable(self) -> None:
        if self._first_speakable_at is None:
            self._first_speakable_at = self._clock()

    def sample(self) -> LatencySample:
        boundaries = (
            self._transcript_at,
            self._thread_started_at,
            self._first_delta_at,
            self._first_speakable_at,
        )
        if any(value is None for value in boundaries):
            raise RuntimeError("foreground timing sample is incomplete")
        transcript, thread, delta, speakable = cast(tuple[float, float, float, float], boundaries)
        if not transcript <= thread <= delta <= speakable:
            raise RuntimeError("foreground timing boundaries are out of order")
        return LatencySample(
            transcript_to_thread_ms=(thread - transcript) * 1000.0,
            thread_to_first_delta_ms=(delta - thread) * 1000.0,
            first_delta_to_speakable_ms=(speakable - delta) * 1000.0,
        )


class WorkActivityObserver:
    """Record sanitized dynamic-tool transport evidence without retaining payloads."""

    __slots__ = (
        "_cancel_requests",
        "_lifecycle_boundaries",
        "_private_authority_messages",
        "_request_boundaries",
        "_start_requests",
        "_thread_start_requests",
        "_thread_start_schema_hashes",
        "_thread_start_schemas",
        "_turn_start_requests",
    )

    def __init__(self) -> None:
        self._lifecycle_boundaries = 0
        self._request_boundaries = 0
        self._start_requests = 0
        self._cancel_requests = 0
        self._private_authority_messages = 0
        self._thread_start_requests = 0
        self._turn_start_requests = 0
        self._thread_start_schemas: list[tuple[str, ...]] = []
        self._thread_start_schema_hashes: list[str] = []

    @property
    def lifecycle_boundaries(self) -> int:
        return self._lifecycle_boundaries

    @property
    def request_boundaries(self) -> int:
        return self._request_boundaries

    @property
    def start_requests(self) -> int:
        return self._start_requests

    @property
    def cancel_requests(self) -> int:
        return self._cancel_requests

    @property
    def private_authority_messages(self) -> int:
        return self._private_authority_messages

    @property
    def thread_start_requests(self) -> int:
        return self._thread_start_requests

    @property
    def turn_start_requests(self) -> int:
        return self._turn_start_requests

    @property
    def thread_start_schemas(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._thread_start_schemas)

    @property
    def thread_start_schema_hashes(self) -> tuple[str, ...]:
        return tuple(self._thread_start_schema_hashes)

    @property
    def total_boundaries(self) -> int:
        return self._lifecycle_boundaries + self._request_boundaries

    def reset(self) -> None:
        self._lifecycle_boundaries = 0
        self._request_boundaries = 0
        self._start_requests = 0
        self._cancel_requests = 0
        self._private_authority_messages = 0
        self._thread_start_requests = 0
        self._turn_start_requests = 0
        self._thread_start_schemas.clear()
        self._thread_start_schema_hashes.clear()

    def observe_send(self, message: Mapping[str, object]) -> None:
        try:
            serialized = json.dumps(
                dict(message),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("Codex protocol message is not JSON serializable") from error
        if _PRIVATE_AUTHORITY.search(serialized) is not None:
            self._private_authority_messages += 1
            raise RuntimeError("Codex protocol output disclosed private authority")
        method = message.get("method")
        if method == "turn/start":
            self._turn_start_requests += 1
            return
        if method != "thread/start":
            return
        self._thread_start_requests += 1
        params = message.get("params")
        if type(params) is not dict:
            self._thread_start_schemas.append(("<malformed>",))
            self._thread_start_schema_hashes.append("<malformed>")
            return
        tools = params.get("dynamicTools")
        if type(tools) is not list:
            self._thread_start_schemas.append(("<malformed>",))
            self._thread_start_schema_hashes.append("<malformed>")
            return
        self._thread_start_schema_hashes.append(
            hashlib.sha256(
                json.dumps(
                    tools,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        names: list[str] = []
        for tool in tools:
            if type(tool) is not dict or type(tool.get("name")) is not str:
                names.append("<malformed>")
            else:
                names.append(cast(str, tool["name"]))
        self._thread_start_schemas.append(tuple(names))

    def observe_receive(self, message: Mapping[str, object]) -> None:
        try:
            serialized = json.dumps(
                dict(message),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("Codex protocol message is not JSON serializable") from error
        if _PRIVATE_AUTHORITY.search(serialized) is not None:
            self._private_authority_messages += 1
            raise RuntimeError("Codex protocol input contained private authority")
        if message.get("method") == _WORK_ACTIVITY_METHOD:
            self._request_boundaries += 1
            params = message.get("params")
            tool = params.get("tool") if type(params) is dict else None
            if tool == "start_work":
                self._start_requests += 1
            elif tool == "cancel_active_work":
                self._cancel_requests += 1
            return
        if message.get("method") not in ("item/started", "item/completed"):
            return
        params = message.get("params")
        if type(params) is not dict:
            return
        item = params.get("item")
        if type(item) is dict and item.get("type") == _WORK_ACTIVITY_ITEM_TYPE:
            self._lifecycle_boundaries += 1


class ObservedTransport:
    """Delegate Codex JSONL transport while retaining sanitized timing only."""

    def __init__(
        self,
        transport: Any,
        probe: TurnTimingProbe,
        activity: WorkActivityObserver | None = None,
    ) -> None:
        for method in ("send", "receive", "close"):
            if not callable(getattr(transport, method, None)):
                raise TypeError(f"transport must provide {method}()")
        if type(probe) is not TurnTimingProbe:
            raise TypeError("probe must be an exact TurnTimingProbe")
        if activity is not None and type(activity) is not WorkActivityObserver:
            raise TypeError("activity must be an exact WorkActivityObserver")
        self._transport = transport
        self._probe = probe
        self._activity = activity

    async def send(self, message: Mapping[str, object]) -> None:
        if self._activity is not None:
            self._activity.observe_send(message)
        self._probe.observe_send(message)
        await self._transport.send(message)

    async def receive(self) -> Mapping[str, object]:
        message = await self._transport.receive()
        if self._activity is not None:
            self._activity.observe_receive(message)
        self._probe.observe_receive(message)
        return cast(Mapping[str, object], message)

    async def close(self) -> None:
        await self._transport.close()


class RecordingWorkHandler:
    """Retain sanitized call counts and acknowledgement timing around a real surface."""

    __slots__ = (
        "_cancel_acknowledgement_ms",
        "_clock",
        "_delegate",
        "_last_accepted_cancel_task_id",
        "_last_accepted_start_task_id",
        "_last_cancel_acknowledged_at",
        "_last_cancel_was_exactly_accepted",
        "_last_start_acknowledged_at",
        "_start_acknowledgement_ms",
        "accepted_cancel_calls",
        "accepted_start_calls",
        "cancel_attempts",
        "start_attempts",
    )

    def __init__(self, delegate: Any, *, clock: Callable[[], float] = time.perf_counter) -> None:
        if not callable(getattr(delegate, "start_work", None)):
            raise TypeError("delegate must provide start_work()")
        if not callable(getattr(delegate, "cancel_active_work", None)):
            raise TypeError("delegate must provide cancel_active_work()")
        maximum = getattr(delegate, "max_objective_chars", None)
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError("delegate objective limit is incompatible")
        if type(getattr(delegate, "can_cancel_work", None)) is not bool:
            raise TypeError("delegate cancellation availability must be an exact boolean")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._delegate = delegate
        self._clock = clock
        self._start_acknowledgement_ms: list[float] = []
        self._cancel_acknowledgement_ms: list[float] = []
        self.start_attempts = 0
        self.cancel_attempts = 0
        self.accepted_start_calls = 0
        self.accepted_cancel_calls = 0
        self._last_accepted_start_task_id: str | None = None
        self._last_accepted_cancel_task_id: str | None = None
        self._last_start_acknowledged_at: float | None = None
        self._last_cancel_acknowledged_at: float | None = None
        self._last_cancel_was_exactly_accepted = False

    @property
    def max_objective_chars(self) -> int:
        return cast(int, self._delegate.max_objective_chars)

    @property
    def can_cancel_work(self) -> bool:
        return cast(bool, self._delegate.can_cancel_work)

    @property
    def start_acknowledgement_ms(self) -> tuple[float, ...]:
        return tuple(self._start_acknowledgement_ms)

    @property
    def cancel_acknowledgement_ms(self) -> tuple[float, ...]:
        return tuple(self._cancel_acknowledgement_ms)

    @property
    def last_cancel_was_exactly_accepted(self) -> bool:
        return self._last_cancel_was_exactly_accepted

    @property
    def last_accepted_start_task_id(self) -> str | None:
        return self._last_accepted_start_task_id

    @property
    def last_accepted_cancel_task_id(self) -> str | None:
        return self._last_accepted_cancel_task_id

    @property
    def last_start_acknowledged_at(self) -> float | None:
        return self._last_start_acknowledged_at

    @property
    def last_cancel_acknowledged_at(self) -> float | None:
        return self._last_cancel_acknowledged_at

    async def start_work(self, *, objective: str, invocation_id: str) -> Any:
        self.start_attempts += 1
        self._last_start_acknowledged_at = None
        started_at = self._clock()
        result = await self._delegate.start_work(
            objective=objective,
            invocation_id=invocation_id,
        )
        if (
            type(result) is WorkStartResult
            and result.accepted is True
            and result.state in ("active", "cancelling")
            and result.task_id is not None
        ):
            self.accepted_start_calls += 1
            self._last_accepted_start_task_id = result.task_id
            acknowledged_at = self._clock()
            self._last_start_acknowledged_at = acknowledged_at
            self._start_acknowledgement_ms.append((acknowledged_at - started_at) * 1000.0)
        return result

    async def cancel_active_work(self, *, invocation_id: str) -> Any:
        self.cancel_attempts += 1
        self._last_cancel_was_exactly_accepted = False
        self._last_accepted_cancel_task_id = None
        self._last_cancel_acknowledged_at = None
        started_at = self._clock()
        result = await self._delegate.cancel_active_work(invocation_id=invocation_id)
        if (
            type(result) is WorkCancelResult
            and result.accepted is True
            and result.state == "cancelling"
            and result.task_id is not None
        ):
            self.accepted_cancel_calls += 1
            self._last_cancel_was_exactly_accepted = True
            self._last_accepted_cancel_task_id = result.task_id
            acknowledged_at = self._clock()
            self._last_cancel_acknowledged_at = acknowledged_at
            self._cancel_acknowledgement_ms.append((acknowledged_at - started_at) * 1000.0)
        return result

    async def cancel_work(self, *, task_id: str, invocation_id: str) -> Any:
        self.cancel_attempts += 1
        self._last_cancel_was_exactly_accepted = False
        self._last_accepted_cancel_task_id = None
        self._last_cancel_acknowledged_at = None
        started_at = self._clock()
        result = await self._delegate.cancel_work(
            task_id=task_id,
            invocation_id=invocation_id,
        )
        if (
            type(result) is WorkCancelResult
            and result.accepted is True
            and result.state == "cancelling"
            and result.task_id is not None
        ):
            self.accepted_cancel_calls += 1
            self._last_cancel_was_exactly_accepted = True
            self._last_accepted_cancel_task_id = result.task_id
            acknowledged_at = self._clock()
            self._last_cancel_acknowledged_at = acknowledged_at
            self._cancel_acknowledgement_ms.append((acknowledged_at - started_at) * 1000.0)
        return result


class _ObservedTaskSession:
    """Expose session-close failures that the controller intentionally consumes."""

    __slots__ = ("_delegate", "close_completed", "close_failures", "close_started")

    def __init__(self, delegate: HermesApiTaskSession) -> None:
        self._delegate = delegate
        self.close_started = False
        self.close_completed = False
        self.close_failures: list[BaseException] = []

    async def dispatch(self, request: Any) -> Any:
        return await self._delegate.dispatch(request)

    async def cancel(self, request: Any) -> Any:
        return await self._delegate.cancel(request)

    async def next_update(self) -> Any:
        return await self._delegate.next_update()

    async def close(self) -> None:
        self.close_started = True
        self.close_completed = False
        try:
            await self._delegate.close()
        except BaseException as error:
            self.close_failures.append(error)
            raise
        finally:
            self.close_completed = True


def _exact_mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError(f"{label} is malformed")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise RuntimeError(f"{label} is malformed")
    return cast(dict[str, object], mapping)


def _bounded_identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > _IDENTIFIER_MAX_CHARS:
        raise RuntimeError(f"{label} is malformed")
    return value


def _request_identity(value: object, label: str) -> int | str:
    if type(value) is int:
        if not 0 <= value <= 2**63 - 1:
            raise RuntimeError(f"{label} lacks a bounded request identity")
        return value
    if type(value) is str and 0 < len(value) <= _IDENTIFIER_MAX_CHARS:
        return value
    raise RuntimeError(f"{label} lacks a bounded request identity")


def _exact_response_result(
    message: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    if set(message) != {"id", "result"}:
        raise RuntimeError(f"{label} response is malformed")
    return _exact_mapping(message.get("result"), f"{label} result")


def _validate_latency(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("latency must be a finite non-negative number")


def paired_orders(pairs: int) -> tuple[tuple[str, str], ...]:
    """Return deterministic alternating AB/BA orders for a release gate."""

    if type(pairs) is not int:
        raise TypeError("sample pairs must be an exact built-in integer")
    if not _MIN_SAMPLE_PAIRS <= pairs <= _MAX_SAMPLE_PAIRS:
        raise ValueError("sample pairs must be between 30 and 1000")
    return tuple(
        ((_ARM_ABSENT, _ARM_PRESENT) if index % 2 == 0 else (_ARM_PRESENT, _ARM_ABSENT))
        for index in range(pairs)
    )


def validate_configuration(
    *,
    pairs: int,
    warmups_per_arm: int,
    task_ack_budget_ms: float,
) -> None:
    """Reject invalid release ranges before credentials or real runs are touched."""

    paired_orders(pairs)
    if (
        type(warmups_per_arm) is not int
        or not _MIN_WARMUPS_PER_ARM <= warmups_per_arm <= _MAX_WARMUPS_PER_ARM
    ):
        raise ValueError("warmups per arm must be between 5 and 100")
    _validate_latency(task_ack_budget_ms)
    if not 0 < task_ack_budget_ms <= _MAX_TASK_ACK_BUDGET_MS:
        raise ValueError("task acknowledgement budget must be between 0 and 60000 ms")


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _distribution(values: tuple[float, ...]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency distribution must not be empty")
    for value in values:
        _validate_latency(value)
    return {
        "count": len(values),
        "p50": _rounded(_percentile(values, 0.50)),
        "p95": _rounded(_percentile(values, 0.95)),
        "max": _rounded(max(values)),
        "mean": _rounded(statistics.fmean(values)),
    }


def _sample_distributions(samples: tuple[LatencySample, ...]) -> dict[str, object]:
    return {
        "transcript_to_thread_ms": _distribution(
            tuple(float(sample.transcript_to_thread_ms) for sample in samples)
        ),
        "thread_to_first_delta_ms": _distribution(
            tuple(float(sample.thread_to_first_delta_ms) for sample in samples)
        ),
        "first_delta_to_speakable_ms": _distribution(
            tuple(float(sample.first_delta_to_speakable_ms) for sample in samples)
        ),
        "foreground_total_ms": _distribution(
            tuple(sample.foreground_total_ms for sample in samples)
        ),
    }


def build_latency_report(
    *,
    absent: tuple[LatencySample, ...],
    present: tuple[LatencySample, ...],
    warmups_per_arm: int,
    task_acknowledgement_ms: tuple[float, ...],
    task_handler_acknowledgement_ms: tuple[float, ...],
    task_ack_budget_ms: float,
) -> dict[str, object]:
    """Build the bounded paired release report and evaluate its exact budgets."""

    if type(absent) is not tuple or type(present) is not tuple:
        raise TypeError("latency arms must be exact tuples")
    if len(absent) != len(present) or not _MIN_SAMPLE_PAIRS <= len(absent) <= _MAX_SAMPLE_PAIRS:
        raise ValueError("latency arms must contain the same 30 to 1000 samples")
    if any(type(sample) is not LatencySample for sample in (*absent, *present)):
        raise TypeError("latency arms must contain exact LatencySample values")
    if (
        type(warmups_per_arm) is not int
        or not _MIN_WARMUPS_PER_ARM <= warmups_per_arm <= _MAX_WARMUPS_PER_ARM
    ):
        raise ValueError("warmups per arm must be between 5 and 100")
    if type(task_acknowledgement_ms) is not tuple or not task_acknowledgement_ms:
        raise ValueError("at least one task acknowledgement sample is required")
    if type(task_handler_acknowledgement_ms) is not tuple or len(
        task_handler_acknowledgement_ms
    ) != len(task_acknowledgement_ms):
        raise ValueError("handler acknowledgement samples must match request samples")
    acknowledgement_samples = tuple(
        TaskAcknowledgementSample(
            request_to_accept_ms=request_ms,
            handler_to_accept_ms=handler_ms,
        )
        for request_ms, handler_ms in zip(
            task_acknowledgement_ms,
            task_handler_acknowledgement_ms,
            strict=True,
        )
    )
    _validate_latency(task_ack_budget_ms)
    if not 0 < task_ack_budget_ms <= _MAX_TASK_ACK_BUDGET_MS:
        raise ValueError("task acknowledgement budget must be between 0 and 60000 ms")

    absent_totals = tuple(sample.foreground_total_ms for sample in absent)
    present_totals = tuple(sample.foreground_total_ms for sample in present)
    paired_deltas = tuple(
        present_total - absent_total
        for absent_total, present_total in zip(absent_totals, present_totals, strict=True)
    )
    absent_p95 = _percentile(absent_totals, 0.95)
    present_p95 = _percentile(present_totals, 0.95)
    regression_budget = max(100.0, absent_p95 * 0.10)
    p95_delta = present_p95 - absent_p95
    foreground_passed = p95_delta <= regression_budget
    acknowledgement_passed = max(task_acknowledgement_ms) <= task_ack_budget_ms

    report: dict[str, object] = {
        "sample_pairs": len(absent),
        "warmups_per_arm": warmups_per_arm,
        "arms": {
            _ARM_ABSENT: _sample_distributions(absent),
            _ARM_PRESENT: _sample_distributions(present),
        },
        "foreground_p95_comparison_ms": {
            "absent_p95_ms": _rounded(absent_p95),
            "present_p95_ms": _rounded(present_p95),
            "p95_delta_ms": _rounded(p95_delta),
            "regression_budget_ms": _rounded(regression_budget),
        },
        "paired_foreground_delta_ms": {
            "signed_mean_delta_ms": _rounded(statistics.fmean(paired_deltas)),
            "absolute_distribution": _distribution(tuple(abs(value) for value in paired_deltas)),
        },
        "task_acknowledgement": {
            "samples_ms": [_rounded(value) for value in task_acknowledgement_ms],
            "count": len(task_acknowledgement_ms),
            "max_ms": _rounded(max(task_acknowledgement_ms)),
            "budget_ms": _rounded(task_ack_budget_ms),
        },
        "task_acknowledgement_breakdown": {
            "decision_and_tool_call_ms": {
                "samples_ms": [
                    _rounded(sample.decision_and_tool_call_ms) for sample in acknowledgement_samples
                ],
                "max_ms": _rounded(
                    max(sample.decision_and_tool_call_ms for sample in acknowledgement_samples)
                ),
            },
            "handler_acceptance_ms": {
                "samples_ms": [_rounded(value) for value in task_handler_acknowledgement_ms],
                "max_ms": _rounded(max(task_handler_acknowledgement_ms)),
            },
        },
        "acceptance": {
            "present_unused_p95_within_budget": foreground_passed,
            "task_acknowledgement_within_budget": acknowledgement_passed,
            "passed": foreground_passed and acknowledgement_passed,
        },
    }
    validate_public_report(report)
    return report


def validate_public_report(value: object) -> None:
    """Reject private authority tokens, unbounded text, and non-JSON values."""

    pending = [value]
    while pending:
        item = pending.pop()
        if item is None or type(item) in (bool, int):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("public report contains a non-finite number")
            continue
        if type(item) is str:
            text = item
            if len(text) > _MAX_PUBLIC_STRING_CHARS:
                raise ValueError("public report contains oversized text")
            if _PRIVATE_AUTHORITY.search(text) is not None:
                raise ValueError("public report contains a private authority token")
            continue
        if type(item) is list or type(item) is tuple:
            pending.extend(item)
            continue
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise TypeError("public report object keys must be exact strings")
            pending.extend(mapping.keys())
            pending.extend(mapping.values())
            continue
        raise TypeError("public report contains a non-JSON value")
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_PUBLIC_REPORT_BYTES:
        raise ValueError("public report exceeds its byte budget")


class _NoDispatchHandler:
    __slots__ = ("invocations",)

    max_objective_chars = 1024
    can_cancel_work = False

    def __init__(self) -> None:
        self.invocations = 0

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        del objective, invocation_id
        self.invocations += 1
        raise RuntimeError("conversational latency prompt unexpectedly requested work")

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        del invocation_id
        self.invocations += 1
        raise RuntimeError("conversational latency prompt unexpectedly requested cancellation")

    async def cancel_work(self, *, task_id: str, invocation_id: str) -> WorkCancelResult:
        del task_id, invocation_id
        self.invocations += 1
        raise RuntimeError("conversational latency prompt unexpectedly requested cancellation")


@dataclass(slots=True)
class _InferenceArm:
    inference: CodexAppServerStreamingInference
    probe: TurnTimingProbe
    activity: WorkActivityObserver
    workspace: Path
    last_activity_delta: int = 0
    last_private_authority_delta: int = 0
    last_transcript_at: float | None = None
    observed_activity_total: int = 0

    @classmethod
    def create(
        cls,
        *,
        model: str,
        effort: str,
        executable: str,
        environment: Mapping[str, str],
    ) -> _InferenceArm:
        workspace = Path(tempfile.mkdtemp(prefix="hermes-natural-gate-"))
        probe = TurnTimingProbe()
        activity = WorkActivityObserver()

        async def transport_factory() -> ObservedTransport:
            raw = await SubprocessCodexJsonLineTransport.create(
                executable=executable,
                cwd=str(workspace),
                environment=environment,
            )
            return ObservedTransport(raw, probe, activity)

        try:
            inference = CodexAppServerStreamingInference(
                model=model,
                effort=effort,
                transport_factory=transport_factory,
                request_timeout_seconds=120,
                work_tool_timeout_seconds=120,
            )
        except BaseException:
            shutil.rmtree(workspace, ignore_errors=True)
            raise
        return cls(
            inference=inference,
            probe=probe,
            activity=activity,
            workspace=workspace,
        )

    async def close(self) -> None:
        await self.inference.close()
        shutil.rmtree(self.workspace)

    async def abort(self) -> None:
        """Close the owned transport so a timed-out stream cannot survive the CLI."""

        transport = self.inference._transport
        if transport is not None:
            await transport.close()

    async def close_for_cleanup(self) -> None:
        """Close normally, forcing transport shutdown before dependants close."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CLOSE_TIMEOUT_SECONDS
        try:
            await asyncio.wait_for(
                self.close(),
                timeout=_CLOSE_TIMEOUT_SECONDS / 2.0,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as primary:
            failures: list[BaseException] = []
            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("inference cleanup deadline expired before abort")
                await asyncio.wait_for(self.abort(), timeout=remaining)
            except BaseException as error:
                failures.append(error)
            try:
                shutil.rmtree(self.workspace)
            except BaseException as error:
                failures.append(error)
            if failures:
                raise _PrimaryFailureGroup(
                    "inference close and forced cleanup failed",
                    [primary, *failures],
                ) from None
            raise


async def _timed_turn(
    *,
    arm: _InferenceArm,
    context: ConversationContextStore,
    text: str,
    turn_id: str,
    reject_work_activity: bool = False,
    timeout_seconds: float = _TIMED_TURN_TIMEOUT_SECONDS,
) -> tuple[tuple[str, ...], LatencySample]:
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 600
    ):
        raise ValueError("turn timeout must be between 0 and 600 seconds")

    async def run_turn() -> tuple[tuple[str, ...], LatencySample]:
        activity_before = arm.activity.total_boundaries
        if reject_work_activity and activity_before != arm.observed_activity_total:
            raise RuntimeError("work-tool activity arrived between conversational turns")
        private_before = arm.activity.private_authority_messages
        transcript_at = time.perf_counter()
        arm.last_transcript_at = transcript_at
        context.record_user_transcript(Transcript(text=text, final=True))
        arm.probe.begin(transcript_at=transcript_at)
        segments: list[str] = []
        async for segment in arm.inference.stream(context.snapshot(), turn_id=turn_id):
            if not segments:
                arm.probe.mark_first_speakable()
            segments.append(segment)
        arm.last_activity_delta = arm.activity.total_boundaries - activity_before
        arm.observed_activity_total = arm.activity.total_boundaries
        arm.last_private_authority_delta = arm.activity.private_authority_messages - private_before
        if reject_work_activity and arm.last_activity_delta:
            raise RuntimeError("conversational turn emitted work-tool activity")
        if arm.last_private_authority_delta:
            raise RuntimeError("Codex protocol output disclosed private authority")
        if not segments:
            raise RuntimeError("Codex turn produced no speakable output")
        validate_public_report(segments)
        return tuple(segments), arm.probe.sample()

    operation = asyncio.create_task(run_turn(), name=f"natural-gate-turn:{turn_id}")
    done, _pending = await asyncio.wait((operation,), timeout=float(timeout_seconds))
    if operation in done:
        return operation.result()
    abort_error: BaseException | None = None
    try:
        await arm.abort()
    except BaseException as error:
        abort_error = error
    done, _pending = await asyncio.wait(
        (operation,),
        timeout=_TURN_CANCEL_DRAIN_SECONDS,
    )
    if operation not in done:
        operation.cancel()
        done, _pending = await asyncio.wait(
            (operation,),
            timeout=_TURN_CANCEL_DRAIN_SECONDS,
        )
    if operation in done and not operation.cancelled():
        operation.exception()
    elif operation not in done:
        operation.add_done_callback(lambda task: None if task.cancelled() else task.exception())
    deadline = TimeoutError("Codex turn exceeded its hard deadline")
    if abort_error is not None:
        raise _PrimaryFailureGroup(
            "Codex turn deadline and transport abort failed",
            (deadline, abort_error),
        )
    raise deadline


async def _untimed_truthful_turn(
    *,
    arm: _InferenceArm,
    context: ConversationContextStore,
    text: str,
    turn_id: str,
) -> tuple[str, ...]:
    activity_before = arm.activity.total_boundaries
    if activity_before != arm.observed_activity_total:
        raise RuntimeError("work-tool activity arrived between conversational turns")
    private_before = arm.activity.private_authority_messages
    wire_before = (
        arm.activity.thread_start_requests,
        arm.activity.turn_start_requests,
    )
    context.record_user_transcript(Transcript(text=text, final=True))
    segments: list[str] = []
    async with asyncio.timeout(_TIMED_TURN_TIMEOUT_SECONDS):
        async for segment in arm.inference.stream(context.snapshot(), turn_id=turn_id):
            segments.append(segment)
    arm.last_activity_delta = arm.activity.total_boundaries - activity_before
    arm.observed_activity_total = arm.activity.total_boundaries
    arm.last_private_authority_delta = arm.activity.private_authority_messages - private_before
    if arm.last_activity_delta:
        raise RuntimeError("truthful continuation turn emitted work-tool activity")
    if arm.last_private_authority_delta:
        raise RuntimeError("Codex protocol output disclosed private authority")
    if (
        arm.activity.thread_start_requests,
        arm.activity.turn_start_requests,
    ) != wire_before:
        raise RuntimeError("truthful continuation turn reached Codex wire transport")
    if not segments:
        raise RuntimeError("truthful continuation turn produced no speakable output")
    validate_public_report(segments)
    return tuple(segments)


def _public_terminal(outcome: Any) -> dict[str, object]:
    value = {
        "status": outcome.status,
        "has_summary": outcome.summary is not None,
        "has_reason": outcome.reason is not None,
    }
    validate_public_report(
        {
            **value,
            "task_id": outcome.task_id,
            "summary": outcome.summary,
            "reason": outcome.reason,
        }
    )
    return value


def _validate_completion_task_evidence(
    *,
    accepted_task_id: str | None,
    active_task_ids: tuple[str, ...],
    terminal_task_id: str,
) -> None:
    if accepted_task_id is None:
        raise RuntimeError("natural completion lacked accepted task identity")
    if len(active_task_ids) > 1:
        raise RuntimeError("natural completion observed multiple active tasks")
    if active_task_ids and active_task_ids != (accepted_task_id,):
        raise RuntimeError("natural completion accepted task did not match active task")
    if terminal_task_id != accepted_task_id:
        raise RuntimeError("natural completion terminal task did not match accepted task")


async def _finish_with_bounded_cleanup(
    primary: BaseException | None,
    closes: tuple[Callable[[], Awaitable[object]], ...],
    *,
    label: str,
    stop_after_failure: bool,
) -> None:
    failures: list[BaseException] = []
    for close in closes:
        try:
            await asyncio.wait_for(
                close(),
                timeout=_CLOSE_TIMEOUT_SECONDS + _CLOSE_WATCHDOG_GRACE_SECONDS,
            )
        except BaseException as error:
            failures.append(error)
            if stop_after_failure:
                break
    if primary is not None:
        if failures:
            raise _PrimaryFailureGroup(
                f"{label} failed after a primary failure",
                [primary, *failures],
            )
        raise primary
    if failures:
        raise _CleanupFailures(f"{label} failed", failures)


async def _run_boundary_gate(
    *,
    key: str,
    model: str,
    effort: str,
    executable: str,
    environment: Mapping[str, str],
) -> tuple[dict[str, object], tuple[float, ...], tuple[float, ...]]:
    from gateway.config import PlatformConfig  # type: ignore[import-not-found]
    from gateway.platforms.api_server import APIServerAdapter  # type: ignore[import-not-found]

    api_port = available_port()
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": api_port,
                "key": key,
                "cors_origins": [],
            },
        )
    )
    session: HermesApiTaskSession | None = None
    observed_session: _ObservedTaskSession | None = None
    controller: ConversationTaskController | None = None
    surface: ConversationWorkControlSurface | None = None
    arm: _InferenceArm | None = None
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    identifiers = count(1)
    acknowledgement_ms: list[float] = []
    primary: BaseException | None = None
    report: dict[str, object] | None = None
    try:
        if not await asyncio.wait_for(adapter.connect(), timeout=_CLOSE_TIMEOUT_SECONDS):
            raise RuntimeError("installed Hermes API adapter did not start")
        context = ConversationContextStore(max_item_chars=1024)
        session = HermesApiTaskSession(
            config=HermesApiConfig(
                base_url=f"http://127.0.0.1:{api_port}",
                bearer=key,
                request_timeout_seconds=120,
            ),
            session_id="session_natural_work_gate",
            private_id_factory=lambda: f"natural_gate_{secrets.token_hex(12)}",
        )
        await session.start()
        observed_session = _ObservedTaskSession(session)
        controller = ConversationTaskController(
            context=context,
            session=observed_session,
            session_id="session_natural_work_gate",
            id_factory=lambda: f"gate{next(identifiers)}",
        )
        surface = ConversationWorkControlSurface(
            controller=controller,
            context=context,
            observer=lambda kind, data: events.append((kind, dict(data))),
            reserve_observer_capacity=lambda: None,
            utterance_id_factory=lambda: f"utterance_gate_{next(identifiers)}",
        )
        handler = RecordingWorkHandler(surface)
        arm = _InferenceArm.create(
            model=model,
            effort=effort,
            executable=executable,
            environment=environment,
        )
        preflight_context = ConversationContextStore(max_item_chars=1024)
        await _timed_turn(
            arm=arm,
            context=preflight_context,
            text="Reply with exactly the word ready.",
            turn_id="turn_preflight",
            reject_work_activity=True,
        )
        arm.inference.bind_work_tools(handler)

        await _timed_turn(
            arm=arm,
            context=context,
            text=(
                "Start background work to return one concise sentence confirming "
                "the installed natural-work completion gate."
            ),
            turn_id="turn_natural_gate_completion",
        )
        completion_active_tasks = context.snapshot().active_tasks
        if arm.last_transcript_at is None or handler.last_start_acknowledged_at is None:
            raise RuntimeError("natural completion acknowledgement was not observed")
        acknowledgement_ms.append(
            (handler.last_start_acknowledged_at - arm.last_transcript_at) * 1000.0
        )
        if handler.start_attempts != 1:
            raise RuntimeError("natural completion start attempt count was not one")
        if handler.accepted_start_calls != 1:
            raise RuntimeError("natural completion accepted start count was not one")
        if arm.activity.start_requests != 1:
            raise RuntimeError("natural completion wire start count was not one")
        completion = await asyncio.wait_for(controller.next_completion(), timeout=300)
        _validate_completion_task_evidence(
            accepted_task_id=handler.last_accepted_start_task_id,
            active_task_ids=tuple(task.task_id for task in completion_active_tasks),
            terminal_task_id=completion.task_id,
        )
        if completion.status != "completed":
            raise RuntimeError("natural completion terminal status was not completed")
        if context.snapshot().active_tasks:
            raise RuntimeError("natural completion terminal evidence left an active task")
        completion_public = _public_terminal(completion)

        await _timed_turn(
            arm=arm,
            context=context,
            text=(
                "Start background work that uses the terminal to wait for 120 seconds "
                "before returning one concise sentence."
            ),
            turn_id="turn_natural_gate_long_start",
        )
        cancellation_active_tasks = context.snapshot().active_tasks
        if arm.last_transcript_at is None or handler.last_start_acknowledged_at is None:
            raise RuntimeError("natural cancellation setup acknowledgement was not observed")
        acknowledgement_ms.append(
            (handler.last_start_acknowledged_at - arm.last_transcript_at) * 1000.0
        )
        if (
            handler.start_attempts != 2
            or handler.accepted_start_calls != 2
            or arm.activity.start_requests != 2
            or len(cancellation_active_tasks) != 1
            or handler.last_accepted_start_task_id != cancellation_active_tasks[0].task_id
        ):
            raise RuntimeError("natural cancellation setup did not start exactly one task")
        await _timed_turn(
            arm=arm,
            context=context,
            text="Stop that background task.",
            turn_id="turn_natural_gate_cancel",
        )
        if arm.last_transcript_at is None or handler.last_cancel_acknowledged_at is None:
            raise RuntimeError("natural cancellation acknowledgement was not observed")
        cancel_request_acknowledgement_ms = (
            handler.last_cancel_acknowledged_at - arm.last_transcript_at
        ) * 1000.0
        acknowledgement_ms.append(cancel_request_acknowledgement_ms)
        if (
            handler.cancel_attempts != 1
            or handler.accepted_cancel_calls != 1
            or arm.activity.cancel_requests != 1
            or not handler.last_cancel_was_exactly_accepted
            or handler.last_accepted_cancel_task_id != cancellation_active_tasks[0].task_id
        ):
            raise RuntimeError("natural cancellation was not exactly accepted")
        interrupted = await asyncio.wait_for(controller.next_completion(), timeout=90)
        if interrupted.status != "interrupted" or context.snapshot().active_tasks:
            raise RuntimeError("natural cancellation lacked interrupted terminal evidence")
        interrupted_public = _public_terminal(interrupted)

        attempts_before_continuation = (handler.start_attempts, handler.cancel_attempts)
        continuation_segments = await _untimed_truthful_turn(
            arm=arm,
            context=context,
            text="Continue the task you just canceled.",
            turn_id="boundary-cancel-continuation",
        )
        if not continuation_segments:
            raise RuntimeError("inactive continuation produced no conversational response")
        if (
            handler.start_attempts,
            handler.cancel_attempts,
        ) != attempts_before_continuation:
            raise RuntimeError("inactive continuation invoked a work-management tool")

        attempts_before_conversation = (handler.start_attempts, handler.cancel_attempts)
        conversational_segments, _ = await _timed_turn(
            arm=arm,
            context=context,
            text="Reply with exactly the word ready.",
            turn_id="turn_natural_gate_conversation",
            reject_work_activity=True,
        )
        if (
            handler.start_attempts,
            handler.cancel_attempts,
        ) != attempts_before_conversation:
            raise RuntimeError("ordinary conversation invoked a work-management tool")

        public_tasks = tuple(
            {"task_id": task.task_id, "objective": task.objective}
            for task in context.snapshot().active_tasks
        )
        validate_public_report(
            {
                "events": events,
                "conversation": conversational_segments,
                "active_tasks": public_tasks,
            }
        )
        boundary_passed = (
            handler.start_attempts == 2
            and handler.accepted_start_calls == 2
            and handler.cancel_attempts == 1
            and handler.accepted_cancel_calls == 1
            and handler.last_cancel_was_exactly_accepted
            and completion.status == "completed"
            and interrupted.status == "interrupted"
            and arm.last_activity_delta == 0
            and arm.activity.start_requests == 2
            and arm.activity.cancel_requests == 1
            and arm.activity.private_authority_messages == 0
        )
        report = {
            "start_attempts": handler.start_attempts,
            "accepted_start_calls": handler.accepted_start_calls,
            "cancel_attempts": handler.cancel_attempts,
            "accepted_cancel_calls": handler.accepted_cancel_calls,
            "observed_start_tool_requests": arm.activity.start_requests,
            "observed_cancel_tool_requests": arm.activity.cancel_requests,
            "private_authority_messages": arm.activity.private_authority_messages,
            "completion": completion_public,
            "cancellation": interrupted_public,
            "ordinary_conversation_work_activity_boundaries": (arm.last_activity_delta),
            "cancel_acknowledgement": {
                "samples_ms": [_rounded(cancel_request_acknowledgement_ms)],
                "count": 1,
                "max_ms": _rounded(cancel_request_acknowledgement_ms),
            },
            "passed": boundary_passed,
        }
        handler_acknowledgement_ms = (
            *handler.start_acknowledgement_ms,
            *handler.cancel_acknowledgement_ms,
        )
        if len(handler_acknowledgement_ms) != len(acknowledgement_ms):
            raise RuntimeError("handler acknowledgement evidence did not match request evidence")
        validate_public_report(report)
        return report, tuple(acknowledgement_ms), handler_acknowledgement_ms
    except BaseException as error:
        primary = error
    finally:
        closes: list[Callable[[], Awaitable[object]]] = []
        if arm is not None:
            closes.append(arm.close_for_cleanup)
        if surface is not None:
            closes.append(surface.close)
        if controller is not None:

            async def close_controller_with_session_evidence() -> None:
                await controller.close()
                assert observed_session is not None
                if observed_session.close_failures:
                    raise BaseExceptionGroup(
                        "controller consumed a Hermes session close failure",
                        list(observed_session.close_failures),
                    )
                if observed_session.close_started and not observed_session.close_completed:
                    raise RuntimeError("Hermes session close did not settle")

            closes.append(close_controller_with_session_evidence)
        elif session is not None:
            closes.append(session.close)
        closes.append(adapter.disconnect)
        await _finish_with_bounded_cleanup(
            primary,
            tuple(closes),
            label="installed-boundary cleanup",
            stop_after_failure=False,
        )
        if (
            arm is not None
            and report is not None
            and (
                arm.activity.start_requests != 2
                or arm.activity.cancel_requests != 1
                or arm.activity.private_authority_messages != 0
            )
        ):
            raise RuntimeError("unexpected work-tool traffic arrived during boundary shutdown")
    raise AssertionError("unreachable boundary gate state")


async def _run_latency_gate(
    *,
    model: str,
    effort: str,
    executable: str,
    environment: Mapping[str, str],
    pairs: int,
    warmups_per_arm: int,
    task_acknowledgement_ms: tuple[float, ...],
    task_handler_acknowledgement_ms: tuple[float, ...],
    task_ack_budget_ms: float,
) -> dict[str, object]:
    orders = paired_orders(pairs)
    if (
        type(warmups_per_arm) is not int
        or not _MIN_WARMUPS_PER_ARM <= warmups_per_arm <= _MAX_WARMUPS_PER_ARM
    ):
        raise ValueError("warmups per arm must be between 5 and 100")
    absent: _InferenceArm | None = None
    present: _InferenceArm | None = None
    unexpected = _NoDispatchHandler()
    warmups: dict[str, list[LatencySample]] = {_ARM_ABSENT: [], _ARM_PRESENT: []}
    samples: dict[str, list[LatencySample]] = {_ARM_ABSENT: [], _ARM_PRESENT: []}
    present_work_activity_boundaries = 0
    present_active_task_count = 0
    primary: BaseException | None = None
    try:
        absent = _InferenceArm.create(
            model=model,
            effort=effort,
            executable=executable,
            environment=environment,
        )
        present = _InferenceArm.create(
            model=model,
            effort=effort,
            executable=executable,
            environment=environment,
        )
        present.inference.bind_work_tools(unexpected)
        contexts = {
            _ARM_ABSENT: ConversationContextStore(max_item_chars=1024),
            _ARM_PRESENT: ConversationContextStore(max_item_chars=1024),
        }
        arms = {_ARM_ABSENT: absent, _ARM_PRESENT: present}
        for index in range(warmups_per_arm):
            warmup_order = (
                (_ARM_ABSENT, _ARM_PRESENT) if index % 2 == 0 else (_ARM_PRESENT, _ARM_ABSENT)
            )
            for arm_name in warmup_order:
                _, sample = await _timed_turn(
                    arm=arms[arm_name],
                    context=contexts[arm_name],
                    text=_CONVERSATIONAL_PROMPTS[index % len(_CONVERSATIONAL_PROMPTS)],
                    turn_id=f"turn_warmup_{arm_name}_{index + 1}",
                    reject_work_activity=arm_name == _ARM_PRESENT,
                )
                warmups[arm_name].append(sample)
                if arm_name == _ARM_PRESENT:
                    present_work_activity_boundaries += arms[arm_name].last_activity_delta
                    if unexpected.invocations != 0:
                        raise RuntimeError(
                            "tools-present latency arm invoked a work-management tool"
                        )
        for pair_index, order in enumerate(orders, start=1):
            prompt = _CONVERSATIONAL_PROMPTS[((pair_index - 1) // 2) % len(_CONVERSATIONAL_PROMPTS)]
            for arm_name in order:
                _, sample = await _timed_turn(
                    arm=arms[arm_name],
                    context=contexts[arm_name],
                    text=prompt,
                    turn_id=f"turn_pair_{pair_index}_{arm_name}",
                    reject_work_activity=arm_name == _ARM_PRESENT,
                )
                samples[arm_name].append(sample)
                if arm_name == _ARM_PRESENT:
                    present_work_activity_boundaries += arms[arm_name].last_activity_delta
                    if unexpected.invocations != 0:
                        raise RuntimeError(
                            "tools-present latency arm invoked a work-management tool"
                        )
        present_active_task_count = len(contexts[_ARM_PRESENT].snapshot().active_tasks)
    except BaseException as error:
        primary = error
    finally:
        closes: list[Callable[[], Awaitable[object]]] = []
        for arm in (absent, present):
            if arm is not None:
                closes.append(arm.close_for_cleanup)
        await _finish_with_bounded_cleanup(
            primary,
            tuple(closes),
            label="latency-arm cleanup",
            stop_after_failure=False,
        )

    if present is not None and present.activity.total_boundaries:
        raise RuntimeError("work-tool activity arrived during latency-arm shutdown")
    if present is not None and present.activity.private_authority_messages:
        raise RuntimeError("private authority arrived during latency-arm shutdown")
    if unexpected.invocations:
        raise RuntimeError("work-tool dispatch arrived during latency-arm shutdown")
    if present is None:
        raise AssertionError("tools-present latency arm was not constructed")
    observed_schemas = present.activity.thread_start_schemas
    observed_schema_hashes = present.activity.thread_start_schema_hashes
    expected_schema_count = warmups_per_arm + pairs
    expected_schema_hash = _idle_start_schema_sha256()
    if (
        len(observed_schemas) != expected_schema_count
        or set(observed_schemas) != {("start_work",)}
        or len(observed_schema_hashes) != expected_schema_count
        or set(observed_schema_hashes) != {expected_schema_hash}
        or present_active_task_count != 0
    ):
        raise RuntimeError("tools-present wire configuration was not the expected idle schema")

    report = build_latency_report(
        absent=tuple(samples[_ARM_ABSENT]),
        present=tuple(samples[_ARM_PRESENT]),
        warmups_per_arm=warmups_per_arm,
        task_acknowledgement_ms=task_acknowledgement_ms,
        task_handler_acknowledgement_ms=task_handler_acknowledgement_ms,
        task_ack_budget_ms=task_ack_budget_ms,
    )
    report["warmup_arms"] = {
        arm_name: _sample_distributions(tuple(values)) for arm_name, values in warmups.items()
    }
    report["present_arm_configuration"] = {
        "active_task_count": present_active_task_count,
        "observed_thread_starts": len(observed_schemas),
        "dynamic_tool_count": len(observed_schemas[0]),
        "dynamic_tool_names": list(observed_schemas[0]),
        "dynamic_tools_sha256": observed_schema_hashes[0],
    }
    report["present_arm_work_activity_boundaries"] = present_work_activity_boundaries
    report["present_arm_handler_attempts"] = unexpected.invocations
    validate_public_report(report)
    return report


@dataclass(frozen=True, slots=True)
class _CodexFingerprint:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    sha256: str


def _idle_start_schema_sha256() -> str:
    tools = CodexAppServerStreamingInference._dynamic_tools(1024, include_cancel=False)
    encoded = json.dumps(
        tools,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _lock_codex_executable(path: Path) -> Iterator[None]:
    """Prevent write/delete replacement of the verified executable on Windows."""

    if os.name != "nt":
        yield
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    handle = create_file(
        str(path),
        0x80000000,  # GENERIC_READ
        0x00000001,  # FILE_SHARE_READ only: deny writes and replacement
        None,
        3,  # OPEN_EXISTING
        0x00000080,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        raise OSError(ctypes.get_last_error(), "could not lock Codex executable")
    try:
        yield
    except BaseException as primary:
        if not close_handle(handle):
            unlock = OSError(ctypes.get_last_error(), "could not unlock Codex executable")
            raise _PrimaryFailureGroup(
                "Codex operation and executable unlock failed",
                [primary, unlock],
            ) from None
        raise
    else:
        if not close_handle(handle):
            raise OSError(ctypes.get_last_error(), "could not unlock Codex executable")


def _codex_fingerprint(executable: str) -> _CodexFingerprint:
    if type(executable) is not str or not executable:
        raise TypeError("Codex executable must be a non-empty exact string")
    path = Path(executable).resolve(strict=True)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("resolved Codex executable is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or not stat.S_ISREG(after.st_mode):
        raise RuntimeError("Codex executable changed while it was hashed")
    return _CodexFingerprint(
        path=path,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        modified_ns=int(after.st_mtime_ns),
        changed_ns=int(after.st_ctime_ns),
        sha256=digest.hexdigest(),
    )


def _assert_pinned_codex(metadata: Mapping[str, str]) -> None:
    try:
        fixture = json.loads(_CODEX_PROTOCOL_FIXTURE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("pinned Codex protocol fixture is unavailable") from error
    if type(fixture) is not dict:
        raise RuntimeError("pinned Codex protocol fixture is invalid")
    expected_version = fixture.get("codexVersion")
    expected_sha256 = fixture.get("codexBinarySha256")
    if (
        type(expected_version) is not str
        or type(expected_sha256) is not str
        or expected_version != f"codex-cli {metadata.get('version')}"
        or expected_sha256 != metadata.get("binary_sha256")
    ):
        raise RuntimeError("Codex executable does not match the pinned protocol fixture")


def _assert_codex_unchanged(
    executable: str,
    expected: _CodexFingerprint,
) -> None:
    if _codex_fingerprint(executable) != expected:
        raise RuntimeError("Codex executable identity or hash changed during the gate")


def _drain_bounded_pipe(
    stream: Any,
    captured: bytearray,
    overflow: list[bool],
) -> None:
    while True:
        chunk = stream.read(8192)
        if not chunk:
            return
        remaining = _MAX_VERSION_OUTPUT_BYTES - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            overflow[0] = True


def _run_bounded_version(
    executable: str,
    *,
    cwd: str,
    environment: Mapping[str, str],
) -> bytes:
    process = subprocess.Popen(
        [executable, "--version"],
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Codex version process lacks bounded output pipes")
    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = [False]
    stderr_overflow = [False]
    readers = (
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(process.stdout, stdout, stdout_overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(process.stderr, stderr, stderr_overflow),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    try:
        return_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as kill_error:
            raise TimeoutError("Codex version process did not terminate") from kill_error
        raise TimeoutError("Codex version command timed out") from error
    finally:
        for reader in readers:
            reader.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
    if any(reader.is_alive() for reader in readers):
        raise RuntimeError("Codex version output drain did not terminate")
    if stdout_overflow[0] or stderr_overflow[0]:
        raise RuntimeError("Codex version output exceeded its byte budget")
    if return_code != 0:
        raise RuntimeError("Codex version command failed")
    return bytes(stdout)


def _version_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in source.items():
        canonical = key.upper()
        if canonical not in _VERSION_ENVIRONMENT_ALLOWLIST:
            continue
        existing = environment.get(canonical)
        if existing is not None and existing != value:
            raise RuntimeError("version environment has case-conflicting values")
        environment[canonical] = value
    return environment


def _codex_metadata(executable: str) -> dict[str, str]:
    before = _codex_fingerprint(executable)
    environment = _version_environment(os.environ)
    with tempfile.TemporaryDirectory(prefix="hermes-codex-version-") as workspace:
        stdout = _run_bounded_version(
            str(before.path),
            cwd=workspace,
            environment=environment,
        )
    _assert_codex_unchanged(executable, before)
    match = _VERSION.search(stdout.decode("utf-8", errors="replace"))
    if match is None:
        raise RuntimeError("Codex version output is unsupported")
    metadata = {
        "version": match.group(1),
        "binary_sha256": before.sha256,
    }
    validate_public_report(metadata)
    return metadata


async def _run_verified_phase(
    *,
    executable: str,
    expected: _CodexFingerprint,
    operation: Callable[[], Awaitable[_ResultT]],
) -> _ResultT:
    _assert_codex_unchanged(executable, expected)
    primary: BaseException | None = None
    try:
        return await operation()
    except BaseException as error:
        primary = error
    finally:
        try:
            _assert_codex_unchanged(executable, expected)
        except BaseException as verification_error:
            if primary is not None:
                raise _PrimaryFailureGroup(
                    "gate phase failed and Codex executable changed",
                    [primary, verification_error],
                ) from None
            raise
    assert primary is not None
    raise primary


async def _run_named_phase(
    stage: str,
    operation: Callable[[], Awaitable[_ResultT]],
) -> _ResultT:
    try:
        return await operation()
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        raise
    except BaseException as error:
        raise _GatePhaseFailure(stage, error) from None


def _run_named_setup(
    stage: str,
    operation: Callable[[], _ResultT],
) -> _ResultT:
    try:
        return operation()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        raise _GatePhaseFailure(stage, error) from None


async def run_real_gate(
    *,
    env_file: Path,
    model: str,
    effort: str,
    codex_executable: str | None,
    pairs: int,
    warmups_per_arm: int,
    task_ack_budget_ms: float,
) -> dict[str, object]:
    validate_configuration(
        pairs=pairs,
        warmups_per_arm=warmups_per_arm,
        task_ack_budget_ms=task_ack_budget_ms,
    )
    executable = _run_named_setup(
        "executable",
        lambda: _resolve_codex_executable(codex_executable),
    )
    with _lock_codex_executable(Path(executable).resolve()):
        metadata = _run_named_setup("codex_metadata", lambda: _codex_metadata(executable))
        _run_named_setup("codex_pin", lambda: _assert_pinned_codex(metadata))
        expected_binary = _run_named_setup(
            "codex_fingerprint",
            lambda: _codex_fingerprint(executable),
        )
        if expected_binary.sha256 != metadata["binary_sha256"]:
            raise _GatePhaseFailure(
                "codex_fingerprint",
                RuntimeError("Codex executable changed after metadata collection"),
            )
        verified_executable = str(expected_binary.path)
        frozen_environment = dict(os.environ)
        key = _run_named_setup("credential_load", lambda: load_api_key(env_file))
        boundary, acknowledgement_ms, handler_acknowledgement_ms = await _run_named_phase(
            "boundary",
            lambda: _run_verified_phase(
                executable=verified_executable,
                expected=expected_binary,
                operation=lambda: _run_boundary_gate(
                    key=key,
                    model=model,
                    effort=effort,
                    executable=verified_executable,
                    environment=frozen_environment,
                ),
            ),
        )
        latency = await _run_named_phase(
            "latency",
            lambda: _run_verified_phase(
                executable=verified_executable,
                expected=expected_binary,
                operation=lambda: _run_latency_gate(
                    model=model,
                    effort=effort,
                    executable=verified_executable,
                    environment=frozen_environment,
                    pairs=pairs,
                    warmups_per_arm=warmups_per_arm,
                    task_acknowledgement_ms=acknowledgement_ms,
                    task_handler_acknowledgement_ms=handler_acknowledgement_ms,
                    task_ack_budget_ms=task_ack_budget_ms,
                ),
            ),
        )
    acceptance = cast(dict[str, object], latency["acceptance"])
    boundary_passed = boundary.get("passed") is True
    gate_passed = boundary_passed and acceptance["passed"] is True
    report: dict[str, object] = {
        "gate": "passed" if gate_passed else "failed",
        "model": model,
        "effort": effort,
        "codex": metadata,
        "boundary": boundary,
        "latency": latency,
    }
    validate_public_report(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--codex-executable")
    parser.add_argument("--pairs", type=int, default=125)
    parser.add_argument("--warmups-per-arm", type=int, default=5)
    parser.add_argument("--task-ack-budget-ms", type=float, default=8000.0)
    return parser


def _failure_code(error: BaseException) -> str:
    if isinstance(error, _GatePhaseFailure):
        return _failure_code(error.error)
    if isinstance(error, _PrimaryFailureGroup):
        return _failure_code(error.exceptions[0])
    if isinstance(error, _CleanupFailures):
        return "cleanup_failed"
    if isinstance(error, BaseExceptionGroup):
        return "cleanup_failed"
    if isinstance(error, (TypeError, ValueError)):
        return "invalid_configuration"
    if isinstance(error, FileNotFoundError):
        return "dependency_unavailable"
    if isinstance(error, TimeoutError):
        return "timed_out"
    if isinstance(error, RuntimeError):
        return "execution_failed"
    return "internal_failure"


def _failure_stage(error: BaseException) -> str | None:
    if isinstance(error, _GatePhaseFailure):
        return error.stage
    if isinstance(error, _PrimaryFailureGroup):
        return _failure_stage(error.exceptions[0])
    return None


def main() -> int:
    args = _parser().parse_args()
    sink = _BoundedDiscardSink()
    try:
        with _discard_runtime_output(sink):
            report = asyncio.run(
                run_real_gate(
                    env_file=args.env_file,
                    model=args.model,
                    effort=args.effort,
                    codex_executable=args.codex_executable,
                    pairs=args.pairs,
                    warmups_per_arm=args.warmups_per_arm,
                    task_ack_budget_ms=args.task_ack_budget_ms,
                )
            )
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        code = _failure_code(error)
        print(code, file=sys.stderr)
        error_report: dict[str, object] = {"code": code}
        stage = _failure_stage(error)
        if stage is not None:
            error_report["stage"] = stage
        report = {
            "gate": "failed",
            "error": error_report,
            "discarded_python_output_bytes": sink.discarded_bytes,
        }
        validate_public_report(report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2
    report["discarded_python_output_bytes"] = sink.discarded_bytes
    validate_public_report(report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["gate"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
