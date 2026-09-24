"""Authenticated loopback Hermes API authority for background tasks and approvals."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp

from hermes_realtime.integration import run_record
from hermes_realtime.integration.run_record import PendingRun
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
)

logger = logging.getLogger(__name__)

_RUN_ID_EXACT = re.compile(r"run_[a-fA-F0-9]{16,64}\Z")
_RUN_ID_DISCLOSURE = re.compile(r"run_[A-Za-z0-9][A-Za-z0-9_-]{15,127}")
_PRIVATE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}\Z")
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_REQUIRED_CAPABILITIES = frozenset(
    {
        "approval_events",
        "run_approval_response",
        "run_events_sse",
        "run_status",
        "run_stop",
        "run_submission",
    }
)
_REQUIRED_ENDPOINTS = {
    "runs": ("POST", "/v1/runs"),
    "run_status": ("GET", "/v1/runs/{run_id}"),
    "run_events": ("GET", "/v1/runs/{run_id}/events"),
    "run_approval": ("POST", "/v1/runs/{run_id}/approval"),
    "run_stop": ("POST", "/v1/runs/{run_id}/stop"),
}
_MAX_HTTP_BODY_BYTES = 64 * 1024
_MAX_SSE_LINE_BYTES = 32 * 1024
_MAX_TERMINAL_TEXT_CHARS = 4096
_MAX_ACTIVE_RUNS = 8
_MAX_PENDING_APPROVALS = 32
_MAX_SEQUENCE = (1 << 63) - 1
# Every status Hermes can report for a run it holds, and so every status a replay may carry.
_LIVE_RUN_STATUSES = frozenset({"queued", "running", "waiting_for_approval", "stopping"})
_RUN_STATUSES = _LIVE_RUN_STATUSES | {"completed", "failed", "cancelled", "interrupted"}
_DISPATCH_RECOVERY_PREFIX = "[hermes-dispatch-recovery] "
_RESTART_SETTLEMENT_PREFIX = "[hermes-restart-settlement] "
_RUN_RECORD_LOCK_PREFIX = "[hermes-run-record-lock] "
_MAX_RUN_RECORD_BYTES = run_record.max_run_record_bytes(_MAX_ACTIVE_RUNS, _MAX_HTTP_BODY_BYTES)
# Path() constructs the platform's concrete class, which is the only accepted record path type.
_CONCRETE_PATH = type(Path())
_BACKGROUND_INSTRUCTIONS = (
    "Complete this bounded background objective with a hard latency target. "
    "Stay exactly within the requested scope; do not add a topic, domain, category, "
    "constraint, source, or entity the user did not request. Ask for approval normally "
    "when Hermes policy requires it. For a simple external current-information lookup, make "
    "exactly one parallel batch of no more than five web_search calls, then answer immediately "
    "from those results. For that lookup path, do not call skill_view, web_extract, browser "
    "tools, terminal, or perform follow-up searches unless the objective explicitly requires "
    "page-level validation that search results cannot provide. Local, runtime, file, process, "
    "build, and verification objectives are operational work, not external current-information "
    "lookups; use the tools needed to inspect them directly. Never interpret an unavailable "
    "tool, failed command, or empty output as a negative finding; state the failure or "
    "uncertainty unless positive evidence verifies the result. Return the final answer exactly "
    "<voice>one or two natural conversational sentences, at most 220 characters, covering "
    "the direct answer and all material findings from detail, including source attribution "
    "when source-backed</voice><detail>friendly, naturally written, display-ready supporting "
    "detail, at most 1800 characters. Preserve every material finding, its useful context, "
    "uncertainty, and source attribution; do not collapse a multi-item result into a terse "
    "headline list</detail>. Voice and detail must be semantically consistent; detail may "
    "expand voice but must not replace its important findings. Do not put line breaks or "
    "markup inside voice, or text outside these elements."
)


@dataclass(frozen=True, slots=True, repr=False)
class HermesApiConfig:
    """Explicit secret-bearing configuration for one loopback Hermes API."""

    base_url: str
    bearer: str = field(repr=False)
    request_timeout_seconds: float = 30.0
    settlement_timeout_seconds: float = 10.0
    settlement_poll_seconds: float = 0.1
    run_provider: str | None = None
    run_model: str | None = None
    run_reasoning_effort: str | None = None
    run_service_tier: str | None = None

    def __post_init__(self) -> None:
        if type(self.base_url) is not str:
            raise TypeError("base_url must be an exact built-in string")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Hermes API base_url must be an exact loopback HTTP origin")
        if type(self.bearer) is not str:
            raise TypeError("bearer must be an exact built-in string")
        if len(self.bearer) < 32 or len(self.bearer) > 4096 or not self.bearer.strip():
            raise ValueError("Hermes API requires an explicit strong bearer")
        if type(self.request_timeout_seconds) not in (int, float):
            raise TypeError("request_timeout_seconds must be an exact number")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or not 0 < self.request_timeout_seconds <= 300
        ):
            raise ValueError("request_timeout_seconds must be between 0 and 300")
        hostname = parsed.hostname
        assert hostname is not None
        bracketed = f"[{hostname}]" if ":" in hostname else hostname
        object.__setattr__(self, "base_url", f"http://{bracketed}:{parsed.port}")
        object.__setattr__(self, "request_timeout_seconds", float(self.request_timeout_seconds))
        for field_name, upper_bound in (
            ("settlement_timeout_seconds", 300.0),
            ("settlement_poll_seconds", 10.0),
        ):
            value = getattr(self, field_name)
            if type(value) not in (int, float):
                raise TypeError(f"{field_name} must be an exact number")
            if not math.isfinite(value) or not 0 < value <= upper_bound:
                raise ValueError(f"{field_name} is outside the supported bound")
            object.__setattr__(self, field_name, float(value))
        for field_name, maximum in (("run_provider", 128), ("run_model", 256)):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not str
                or not value.strip()
                or len(value) > maximum
                or not value.isprintable()
            ):
                raise ValueError(f"{field_name} must be a bounded printable string or None")
        if self.run_reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "ultra",
        }:
            raise ValueError("run_reasoning_effort is unsupported")
        if self.run_service_tier not in {None, "auto", "default", "priority"}:
            raise ValueError("run_service_tier is unsupported")

    def __repr__(self) -> str:
        return (
            "HermesApiConfig("
            f"base_url={self.base_url!r}, bearer=<redacted>, "
            f"request_timeout_seconds={self.request_timeout_seconds!r}, "
            f"settlement_timeout_seconds={self.settlement_timeout_seconds!r}, "
            f"settlement_poll_seconds={self.settlement_poll_seconds!r}, "
            f"run_provider={self.run_provider!r}, run_model={self.run_model!r}, "
            f"run_reasoning_effort={self.run_reasoning_effort!r}, "
            f"run_service_tier={self.run_service_tier!r})"
        )

    __str__ = __repr__


class _NoResponse(RuntimeError):
    """A request may have reached Hermes, but no complete response came back."""


class _RunNotFound(RuntimeError):
    """Hermes answered that it does not know the run."""


@dataclass(frozen=True, slots=True)
class HermesRestartSettlement:
    """What a start settled from the run record a previous process left behind.

    ``stopped`` counts runs confirmed no longer running on Hermes: stopped now, already
    ended, or no longer known to Hermes. ``unknown`` counts dispatches whose outcome could not
    be learned without risking starting the work again; nothing was resent for them.
    """

    stopped: int
    unknown: int


def _is_run_not_found(status: int, payload: dict[str, Any]) -> bool:
    error = payload.get("error")
    return status == 404 and type(error) is dict and error.get("code") == "run_not_found"


@dataclass(slots=True)
class _RunAuthority:
    task_id: str
    api_run_id: str
    protocol_run_id: str
    event_task: asyncio.Task[None] | None = None
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class _PendingApproval:
    approval_id: str
    task_id: str
    api_run_id: str


class HermesApiTaskSession:
    """Map Hermes API runs into the existing private task-control protocol."""

    def __init__(
        self,
        *,
        config: HermesApiConfig,
        session_id: str,
        private_id_factory: Callable[[], str] | None = None,
        approval_observer: Callable[[dict[str, str | int | bool | None]], None] | None = None,
        event_observer: Callable[[str, str | None], None] | None = None,
        run_record_path: Path | None = None,
    ) -> None:
        if type(config) is not HermesApiConfig:
            raise TypeError("config must be an exact HermesApiConfig")
        if type(session_id) is not str or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("session_id must be a canonical bounded identifier")
        if private_id_factory is not None and not callable(private_id_factory):
            raise TypeError("private_id_factory must be callable")
        if approval_observer is not None and not callable(approval_observer):
            raise TypeError("approval_observer must be callable")
        if event_observer is not None and not callable(event_observer):
            raise TypeError("event_observer must be callable")
        if run_record_path is not None and type(run_record_path) is not _CONCRETE_PATH:
            raise TypeError("run_record_path must be an exact pathlib Path or None")
        self._config = config
        self._session_id = session_id
        self._private_id_factory = private_id_factory or (lambda: secrets.token_hex(16))
        self._approval_observer = approval_observer
        self._event_observer = event_observer
        self._client: aiohttp.ClientSession | None = None
        self._runs_by_task: dict[str, _RunAuthority] = {}
        self._runs_by_api: dict[str, _RunAuthority] = {}
        self._pending_dispatches: set[str] = set()
        self._inflight_dispatches: set[asyncio.Task[Any]] = set()
        self._unpublished_api_run_ids: set[str] = set()
        self._reserved_api_run_ids: set[str] = set()
        self._reserved_protocol_run_ids: set[str] = set()
        self._reserved_approval_ids: set[str] = set()
        self._pending_approvals: deque[_PendingApproval] = deque()
        self._updates: asyncio.Queue[WorkCompletedEvent | BaseException] = asyncio.Queue(maxsize=32)
        self._sequence = 0
        self._approval_sequence = 0
        self._state_lock = asyncio.Lock()
        self._approval_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        # Seconds Hermes retains a run's idempotency record; None when a resend is unsafe.
        self._resend_window: float | None = None
        # The durable record of runs this process may have running on Hermes, so a restart
        # after a crash can find and stop them. It stays empty when no path is configured.
        self._run_record_path = run_record_path
        self._record_pending: list[PendingRun] = []
        self._record_admitted: list[str] = []
        self._record_lock = asyncio.Lock()
        # The OS lock that makes this session the record's only owner while it runs.
        self._record_owner: int | None = None
        self._restart_settlement: HermesRestartSettlement | None = None
        self._started = False
        self._closed = False

    async def start(self) -> None:
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("Hermes API task session is closed")
            if self._started:
                raise RuntimeError("Hermes API task session is already started")
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_seconds)
            self._client = aiohttp.ClientSession(
                timeout=timeout,
                trust_env=False,
                raise_for_status=False,
            )
        try:
            if self._run_record_path is not None:
                # One host owns a record at a time; any other would stop runs it does not own.
                self._record_owner = run_record.lock_run_record(self._run_record_path)
                if self._record_owner is None:
                    print(
                        _RUN_RECORD_LOCK_PREFIX
                        + json.dumps(
                            {"cause": "held", "version": 1}, separators=(",", ":"), sort_keys=True
                        ),
                        flush=True,
                    )
                    raise RuntimeError("another host holds the Hermes run record")
            status, payload = await self._request_json("GET", "/v1/capabilities")
            if status != 200:
                raise RuntimeError("Hermes API capability discovery failed")
            if (
                payload.get("object") != "hermes.api_server.capabilities"
                or payload.get("platform") != "hermes-agent"
                or type(payload.get("model")) is not str
                or not payload["model"].strip()
            ):
                raise RuntimeError("Hermes API capability identity is malformed")
            auth = payload.get("auth")
            runtime = payload.get("runtime")
            if (
                type(auth) is not dict
                or auth.get("type") != "bearer"
                or auth.get("required") is not True
                or type(runtime) is not dict
                or runtime.get("mode") != "server_agent"
                or runtime.get("tool_execution") != "server"
                or runtime.get("split_runtime") is not False
            ):
                raise RuntimeError("Hermes API requires an authenticated server-side runtime")
            features = payload.get("features")
            if type(features) is not dict:
                raise RuntimeError("Hermes API capabilities are malformed")
            missing = sorted(
                name for name in _REQUIRED_CAPABILITIES if features.get(name) is not True
            )
            if missing:
                raise RuntimeError("Hermes API lacks required capability: " + ", ".join(missing))
            resend_window: float | None = None
            if "runs_idempotency" in features:
                idempotency = features["runs_idempotency"]
                if (
                    type(idempotency) is not dict
                    or type(idempotency.get("supported")) is not bool
                    or type(idempotency.get("durable")) is not bool
                    or type(idempotency.get("retention_seconds")) is not int
                    or idempotency["retention_seconds"] <= 0
                ):
                    raise RuntimeError("Hermes API run idempotency capability is malformed")
                # Only a durable record can answer a resend truthfully across a Hermes
                # restart, and only while Hermes still retains it.
                if idempotency["supported"] is True and idempotency["durable"] is True:
                    resend_window = float(idempotency["retention_seconds"])
            endpoints = payload.get("endpoints")
            if type(endpoints) is not dict:
                raise RuntimeError("Hermes API endpoint capabilities are malformed")
            for name, (method, path) in _REQUIRED_ENDPOINTS.items():
                endpoint = endpoints.get(name)
                if (
                    type(endpoint) is not dict
                    or endpoint.get("method") != method
                    or endpoint.get("path") != path
                ):
                    raise RuntimeError(f"Hermes API endpoint capability is invalid: {name}")
            settlement: HermesRestartSettlement | None = None
            if self._run_record_path is not None:
                settlement = await self._settle_run_record(self._run_record_path, resend_window)
        except BaseException:
            await self._close_client()
            self._release_run_record()
            raise
        async with self._state_lock:
            if self._closed:
                await self._close_client()
                self._release_run_record()
                raise RuntimeError("Hermes API task session closed during startup")
            self._resend_window = resend_window
            self._restart_settlement = settlement
            self._started = True

    def _release_run_record(self) -> None:
        owner = self._record_owner
        self._record_owner = None
        if owner is not None:
            run_record.unlock_run_record(owner)

    @property
    def restart_settlement(self) -> HermesRestartSettlement | None:
        """What start settled from the run record; None without a record or before start."""
        return self._restart_settlement

    async def _settle_run_record(
        self,
        path: Path,
        window: float | None,
    ) -> HermesRestartSettlement:
        """Stop every run a previous process may have left running; tasks are not resumed."""
        # Counts after a settlement, or a bounded refusal category; never record content.
        evidence: dict[str, str | int] | None = None
        try:
            raw = await asyncio.to_thread(run_record.read_run_record, path, _MAX_RUN_RECORD_BYTES)
            parsed = (
                ([], [])
                if raw is None
                else run_record.parse_run_record(
                    raw,
                    max_entries=_MAX_ACTIVE_RUNS,
                    max_request_bytes=_MAX_HTTP_BODY_BYTES,
                    run_id=_RUN_ID_EXACT,
                )
            )
            if parsed is None:
                # Oversized, unknown-version, and malformed records are all refused unread.
                evidence = {"refusal": "malformed", "version": 1}
                raise RuntimeError("Hermes API run record is malformed")
            pending, admitted = parsed
            if not pending and not admitted:
                return HermesRestartSettlement(stopped=0, unknown=0)
            outcomes = await asyncio.gather(
                *(self._settle_recorded_run(api_run_id) for api_run_id in admitted),
                *(self._settle_recorded_dispatch(entry, window) for entry in pending),
                return_exceptions=True,
            )
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            # The record keeps exactly the entries still unsettled, so a later start retries them.
            unresolved_admitted = [
                api_run_id
                for api_run_id, outcome in zip(admitted, outcomes[: len(admitted)], strict=True)
                if isinstance(outcome, BaseException)
            ]
            unresolved_pending = [
                entry
                for entry, outcome in zip(pending, outcomes[len(admitted) :], strict=True)
                if isinstance(outcome, BaseException)
            ]
            data = run_record.run_record_bytes(unresolved_pending, unresolved_admitted)
            await asyncio.to_thread(run_record.write_run_record, path, data)
            stopped = outcomes.count("stopped")
            unknown = outcomes.count("unknown")
            evidence = {
                "stopped": stopped,
                "unknown": unknown,
                "unresolved": len(failures),
                "version": 1,
            }
            if failures:
                raise BaseExceptionGroup(
                    "Hermes API restart settlement left runs unresolved", failures
                )
            return HermesRestartSettlement(stopped=stopped, unknown=unknown)
        finally:
            if evidence is not None:
                print(
                    _RESTART_SETTLEMENT_PREFIX
                    + json.dumps(evidence, separators=(",", ":"), sort_keys=True),
                    flush=True,
                )

    async def _settle_recorded_run(self, api_run_id: str) -> str:
        # A run Hermes no longer knows has nothing left running.
        with contextlib.suppress(_RunNotFound):
            await self._stop_and_wait(api_run_id)
        return "stopped"

    async def _settle_recorded_dispatch(self, entry: PendingRun, window: float | None) -> str:
        """Learn the run a recorded dispatch started, by the resend rules, and stop it."""
        age = time.time() - entry.minted_at
        if entry.key is None or window is None or not 0 <= age < window:
            # Without a retained key a replay could start the work again.
            return "unknown"
        try:
            status, payload = await self._request_json(
                "POST", "/v1/runs", body=entry.body, headers={"Idempotency-Key": entry.key}
            )
        except RuntimeError:
            return "unknown"
        api_run_id = payload.get("run_id")
        # As after a lost attempt, only an admission naming an exact run settles it.
        if status != 202 or type(api_run_id) is not str:
            return "unknown"
        if _RUN_ID_EXACT.fullmatch(api_run_id) is None:
            return "unknown"
        return await self._settle_recorded_run(api_run_id)

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        trusted = self._trusted_dispatch(request)
        operation = asyncio.current_task()
        if operation is None:
            raise RuntimeError("Hermes API dispatch requires an asyncio task")
        async with self._state_lock:
            self._require_started_locked()
            if trusted.task_id in self._runs_by_task or trusted.task_id in self._pending_dispatches:
                return self._dispatch_ack(
                    trusted,
                    accepted=False,
                    reason="task is already active",
                )
            # A recorded run may still be running on Hermes until a restart settles it, so
            # recorded entries hold capacity too; without a record there are none.
            active = len(self._runs_by_task) + len(self._pending_dispatches)
            recorded = len(self._record_pending) + len(self._record_admitted)
            if max(active, recorded) >= _MAX_ACTIVE_RUNS:
                return self._dispatch_ack(
                    trusted,
                    accepted=False,
                    reason="background task capacity exhausted",
                )
            pending = self._mint_pending_run(trusted)
            if self._run_record_path is not None:
                self._record_pending.append(pending)
            self._pending_dispatches.add(trusted.task_id)
            self._inflight_dispatches.add(operation)
        abandoned = asyncio.Event()
        dispatch_operation = asyncio.create_task(
            self._dispatch_admitted(trusted, pending, abandoned=abandoned),
            name=f"hermes-api-dispatch:{trusted.task_id}",
        )
        # close() must wait for the work itself: a caller cancelled twice stops waiting early.
        self._inflight_dispatches.add(dispatch_operation)
        dispatch_operation.add_done_callback(self._inflight_dispatches.discard)
        try:
            return await asyncio.shield(dispatch_operation)
        except asyncio.CancelledError:
            abandoned.set()
            with contextlib.suppress(BaseException):
                await asyncio.shield(dispatch_operation)
            await self._settle_abandoned_task(trusted.task_id)
            raise
        finally:
            async with self._state_lock:
                self._pending_dispatches.discard(trusted.task_id)
                self._inflight_dispatches.discard(operation)

    def _mint_pending_run(self, trusted: WorkDispatchRequestedEvent) -> PendingRun:
        body: dict[str, object] = {
            "input": trusted.payload.objective,
            "instructions": _BACKGROUND_INSTRUCTIONS,
        }
        if self._config.run_provider is not None:
            body["provider"] = self._config.run_provider
        if self._config.run_model is not None:
            body["model"] = self._config.run_model
        model_options: dict[str, str] = {}
        if self._config.run_reasoning_effort is not None:
            model_options["reasoning_effort"] = self._config.run_reasoning_effort
        if self._config.run_service_tier is not None:
            model_options["service_tier"] = self._config.run_service_tier
        if model_options:
            body["model_options"] = model_options
        # Only durable idempotency makes a key answerable after a lost attempt or a restart.
        key = secrets.token_hex(16) if self._resend_window is not None else None
        # Wall-clock time also counts a suspended host, which the retention window does.
        return PendingRun(key=key, minted_at=time.time(), body=body)

    async def _dispatch_admitted(
        self,
        trusted: WorkDispatchRequestedEvent,
        pending: PendingRun,
        *,
        abandoned: asyncio.Event,
    ) -> WorkDispatchAcknowledgedEvent:
        try:
            # Write-ahead: a crash after the POST must find the dispatch in the record.
            await self._persist_record()
        except BaseException:
            async with self._state_lock:
                if pending in self._record_pending:
                    self._record_pending.remove(pending)
            raise
        status, payload = await self._submit_run(pending)
        if status != 202:
            await self._forget_pending(pending)
            logger.warning("Hermes API rejected background dispatch with HTTP %d", status)
            return self._dispatch_ack(
                trusted,
                accepted=False,
                reason="Hermes API rejected dispatch",
            )
        api_run_id = payload.get("run_id") if type(payload) is dict else None
        usable_run_id = (
            type(api_run_id) is str and 1 <= len(api_run_id) <= 128 and api_run_id.isprintable()
        )
        claimed = False
        if usable_run_id:
            assert isinstance(api_run_id, str)
            claimed = await self._claim_unpublished(api_run_id)
            if not claimed:
                # Hermes named a run this session already accounts for.
                await self._forget_pending(pending)
            elif _RUN_ID_EXACT.fullmatch(api_run_id) is not None:
                try:
                    await self._promote_pending(pending, api_run_id)
                except BaseException:
                    # An unrecorded run could outlive a crash unfound, so it stops at once.
                    await self._settle_unpublished(api_run_id)
                    raise
        response_keys = set(payload) if type(payload) is dict else set()
        replayed = payload.get("replayed")
        if self._resend_window is not None:
            well_formed = response_keys == {"run_id", "status", "replayed"} and (
                type(replayed) is bool
            )
        else:
            well_formed = response_keys == {"run_id", "status"} or (
                response_keys == {"run_id", "status", "replayed"} and replayed is False
            )
        if not well_formed:
            if claimed:
                assert isinstance(api_run_id, str)
                await self._settle_unpublished(api_run_id)
            raise RuntimeError("Hermes API dispatch response is malformed")
        # A first admission reports "started"; a replay reports the run's current status.
        admitted_statuses = _RUN_STATUSES if replayed is True else {"started"}
        exact_authority = (
            type(api_run_id) is str
            and _RUN_ID_EXACT.fullmatch(api_run_id) is not None
            and type(payload.get("status")) is str
            and payload["status"] in admitted_statuses
        )
        if not exact_authority:
            if claimed:
                assert isinstance(api_run_id, str)
                await self._settle_unpublished(api_run_id)
            raise RuntimeError("Hermes API dispatch response lacks exact authority")
        assert isinstance(api_run_id, str)
        if not claimed:
            raise RuntimeError("Hermes API reused a run authority")
        private_token = self._private_id_factory()
        if type(private_token) is not str or _PRIVATE_TOKEN.fullmatch(private_token) is None:
            await self._settle_unpublished(api_run_id)
            raise RuntimeError("private run identifier factory returned an invalid value")
        protocol_run_id = f"deleg_{private_token}"
        async with self._state_lock:
            conflicting_authority = protocol_run_id in self._reserved_protocol_run_ids
            if not conflicting_authority:
                self._reserved_protocol_run_ids.add(protocol_run_id)
        if conflicting_authority:
            await self._settle_unpublished(api_run_id)
            raise RuntimeError("private run identifier factory returned a duplicate value")
        authority = _RunAuthority(
            task_id=trusted.task_id,
            api_run_id=api_run_id,
            protocol_run_id=protocol_run_id,
        )
        publication_error: RuntimeError | None = None
        async with self._state_lock:
            if abandoned.is_set():
                publication_error = RuntimeError("Hermes API dispatch was abandoned")
            elif self._closed or not self._started:
                publication_error = RuntimeError("Hermes API task session closed during dispatch")
            elif trusted.task_id in self._runs_by_task or api_run_id in self._runs_by_api:
                publication_error = RuntimeError("Hermes API returned conflicting run authority")
            else:
                self._unpublished_api_run_ids.discard(api_run_id)
                self._runs_by_task[trusted.task_id] = authority
                self._runs_by_api[api_run_id] = authority
                authority.event_task = asyncio.create_task(
                    self._consume_events(authority),
                    name=f"hermes-api-events:{trusted.task_id}",
                )
        if publication_error is not None:
            await self._settle_unpublished(api_run_id)
            raise publication_error
        return self._dispatch_ack(
            trusted,
            accepted=True,
            run_id=protocol_run_id,
        )

    async def _submit_run(self, pending: PendingRun) -> tuple[int, dict[str, Any]]:
        """POST one run, resending it once unchanged if the first attempt got no complete response.

        Both attempts carry the dispatch's recorded idempotency key, so Hermes admits the run at
        most once while it retains the key; a resend older than that retention, counted from
        the key's mint time, is never sent. The resend is the
        only way to learn a run the lost attempt may have started, so it happens
        even for an abandoned dispatch, and publication then stops what it recovers. After a
        lost attempt only an admission settles the dispatch: a refused resend proves nothing
        about the first attempt, so its outcome is unknown rather than rejected.
        """
        body = pending.body
        window = self._resend_window
        if window is None or pending.key is None:
            return await self._request_json("POST", "/v1/runs", body=body)
        headers = {"Idempotency-Key": pending.key}
        minted_at = pending.minted_at
        with contextlib.suppress(_NoResponse):
            return await self._request_json("POST", "/v1/runs", body=body, headers=headers)
        cause: str | None = "error"
        status: int | None = None
        try:
            if time.time() - minted_at >= window:
                # Hermes may have pruned the key, and would then start the work again.
                cause = "expired"
                raise RuntimeError("Hermes API dispatch outcome is unknown")
            try:
                status, payload = await self._request_json(
                    "POST", "/v1/runs", body=body, headers=headers
                )
            except _NoResponse as error:
                cause = "no_response"
                raise RuntimeError("Hermes API dispatch outcome is unknown") from error
            except RuntimeError as error:
                raise RuntimeError("Hermes API dispatch outcome is unknown") from error
            run_id = payload.get("run_id")
            if status != 202:
                cause = "refused"
                raise RuntimeError("Hermes API dispatch outcome is unknown")
            if type(run_id) is not str or _RUN_ID_EXACT.fullmatch(run_id) is None:
                # Only a run it can name lets recovery stop or track what was started.
                cause = "unidentified"
                raise RuntimeError("Hermes API dispatch outcome is unknown")
            cause = None
            return status, payload
        finally:
            if cause is not None:
                # The lost attempt may have started a run nothing now tracks; count it.
                evidence = {"cause": cause, "status": status, "version": 1}
                print(
                    _DISPATCH_RECOVERY_PREFIX
                    + json.dumps(evidence, separators=(",", ":"), sort_keys=True),
                    flush=True,
                )

    async def _settle_unpublished(self, api_run_id: str) -> None:
        await self._stop_and_wait(api_run_id)
        async with self._state_lock:
            self._unpublished_api_run_ids.discard(api_run_id)
            recorded = self._discard_admitted_locked(api_run_id)
        if recorded:
            await self._persist_record_best_effort()

    async def _forget_pending(self, pending: PendingRun) -> None:
        async with self._state_lock:
            recorded = pending in self._record_pending
            if recorded:
                self._record_pending.remove(pending)
        if recorded:
            await self._persist_record_best_effort()

    async def _promote_pending(self, pending: PendingRun, api_run_id: str) -> None:
        if self._run_record_path is None:
            return
        async with self._state_lock:
            if pending in self._record_pending:
                self._record_pending.remove(pending)
            if api_run_id not in self._record_admitted:
                self._record_admitted.append(api_run_id)
        await self._persist_record()

    def _discard_admitted_locked(self, api_run_id: str) -> bool:
        if api_run_id not in self._record_admitted:
            return False
        self._record_admitted.remove(api_run_id)
        return True

    async def _persist_record(self) -> None:
        path = self._run_record_path
        if path is None:
            return
        operation = asyncio.create_task(
            self._persist_record_owned(path),
            name="hermes-api-run-record",
        )
        # A cancelled caller must not let a later snapshot be overtaken by this one.
        await asyncio.shield(operation)

    async def _persist_record_owned(self, path: Path) -> None:
        async with self._record_lock:
            async with self._state_lock:
                data = run_record.run_record_bytes(
                    list(self._record_pending), list(self._record_admitted)
                )
            await asyncio.to_thread(run_record.write_run_record, path, data)

    async def _persist_record_best_effort(self) -> None:
        try:
            await self._persist_record()
        except Exception:
            # Only a removal is lost, so the record over-approximates: a restart settles the
            # stale entry again, stopping anything its replay reveals.
            logger.warning("Hermes run record update failed; a restart re-settles its entries")

    async def _claim_unpublished(self, api_run_id: str) -> bool:
        operation = asyncio.create_task(
            self._claim_unpublished_owned(api_run_id),
            name=f"hermes-api-claim:{api_run_id}",
        )
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            claimed = await asyncio.shield(operation)
            if claimed:
                settlement = asyncio.create_task(
                    self._settle_unpublished(api_run_id),
                    name=f"hermes-api-cancelled-dispatch-settle:{api_run_id}",
                )
                with contextlib.suppress(BaseException):
                    await asyncio.shield(settlement)
            raise

    async def _claim_unpublished_owned(self, api_run_id: str) -> bool:
        async with self._state_lock:
            if api_run_id in self._reserved_api_run_ids:
                return False
            self._reserved_api_run_ids.add(api_run_id)
            self._unpublished_api_run_ids.add(api_run_id)
            return True

    async def _settle_abandoned_task(self, task_id: str) -> None:
        async with self._state_lock:
            authority = self._runs_by_task.get(task_id)
        if authority is None or authority.terminal:
            return
        task = authority.event_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._settle_authority(authority)

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        trusted = self._trusted_cancel(request)
        assert trusted.task_id is not None
        async with self._state_lock:
            self._require_started_locked()
            authority = self._runs_by_task.get(trusted.task_id)
            if authority is None or authority.terminal:
                return self._cancel_ack(
                    trusted,
                    accepted=False,
                    reason="task is not active",
                )
            api_run_id = authority.api_run_id
            protocol_run_id = authority.protocol_run_id
        status, payload = await self._request_json(
            "POST",
            f"/v1/runs/{quote(api_run_id, safe='')}/stop",
            body={},
        )
        if (
            status != 200
            or type(payload) is not dict
            or payload.get("run_id") != api_run_id
            or payload.get("status") != "stopping"
            or set(payload) != {"run_id", "status"}
        ):
            return self._cancel_ack(
                trusted,
                accepted=False,
                reason="Hermes API rejected cancellation",
            )
        return self._cancel_ack(
            trusted,
            accepted=True,
            run_ids=[protocol_run_id],
        )

    async def decide_approval(self, *, approval_id: str, sequence: int, decision: str) -> None:
        if type(approval_id) is not str or _PRIVATE_TOKEN.fullmatch(approval_id) is None:
            raise ValueError("approval_id must be a canonical public identifier")
        if type(sequence) is not int:
            raise TypeError("approval sequence must be an exact integer")
        if type(decision) is not str:
            raise TypeError("approval decision must be an exact built-in string")
        if decision not in {"approve", "reject"}:
            raise ValueError("approval decision must be approve or reject")
        operation = asyncio.create_task(
            self._decide_approval_serialized(
                approval_id=approval_id,
                sequence=sequence,
                decision=decision,
            ),
            name=f"hermes-api-approval-decision:{approval_id}",
        )
        await asyncio.shield(operation)

    async def _decide_approval_serialized(
        self,
        *,
        approval_id: str,
        sequence: int,
        decision: str,
    ) -> None:
        async with self._approval_lock:
            async with self._state_lock:
                self._require_started_locked()
                if sequence != self._approval_sequence + 1:
                    raise RuntimeError("approval sequence is not the next expected value")
                if not self._pending_approvals:
                    raise RuntimeError("no Hermes approval is pending")
                pending = next(
                    (item for item in self._pending_approvals if item.approval_id == approval_id),
                    None,
                )
                if pending is None:
                    raise RuntimeError("approval request is not the active authority")
                authority = self._runs_by_api.get(pending.api_run_id)
                if authority is None or authority.terminal:
                    raise RuntimeError("Hermes approval belongs to an inactive run")
            choice = "once" if decision == "approve" else "deny"
            status, payload = await self._request_json(
                "POST",
                f"/v1/runs/{quote(pending.api_run_id, safe='')}/approval",
                body={"choice": choice},
            )
            if (
                status != 200
                or type(payload) is not dict
                or set(payload) != {"choice", "object", "resolved", "run_id"}
                or payload.get("object") != "hermes.run.approval_response"
                or payload.get("run_id") != pending.api_run_id
                or payload.get("choice") != choice
                or type(payload.get("resolved")) is not int
                or payload.get("resolved") != 1
            ):
                raise RuntimeError("Hermes API approval response is not authoritative")
            async with self._state_lock:
                still_pending = pending in self._pending_approvals
                if still_pending:
                    self._pending_approvals.remove(pending)
                self._approval_sequence = sequence
            if still_pending:
                self._publish_approval_closed(pending, state="resolved")

    async def next_update(self) -> WorkCompletedEvent:
        item = await self._updates.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        async with self._close_lock:
            operation = self._close_operation
            if operation is None or (
                operation.done() and (operation.cancelled() or operation.exception() is not None)
            ):
                operation = asyncio.create_task(
                    self._close_owned(),
                    name="hermes-api-session-close",
                )
                self._close_operation = operation
        await asyncio.shield(operation)

    async def _close_owned(self) -> None:
        async with self._state_lock:
            self._closed = True
            self._started = False
            inflight = tuple(self._inflight_dispatches)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        async with self._state_lock:
            authorities = tuple(self._runs_by_task.values())
            unpublished = tuple(self._unpublished_api_run_ids)
        tasks = tuple(
            authority.event_task
            for authority in authorities
            if authority.event_task is not None and not authority.event_task.done()
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        active = tuple(authority for authority in authorities if not authority.terminal)
        settlements = [self._settle_authority(authority) for authority in active]
        settlements.extend(self._settle_unpublished(api_run_id) for api_run_id in unpublished)
        results = await asyncio.gather(*settlements, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup(
                "Hermes API session has unresolved run authority",
                failures,
            )
        async with self._state_lock:
            self._runs_by_task.clear()
            self._runs_by_api.clear()
            self._pending_dispatches.clear()
            self._inflight_dispatches.clear()
            self._unpublished_api_run_ids.clear()
            self._pending_approvals.clear()
        # Wait for every earlier record write; dispatches whose outcome is unknown stay
        # recorded, so the next start can still find and stop what they started.
        await self._persist_record_best_effort()
        # Only after the final write may another host take the record.
        self._release_run_record()
        await self._close_client()

    async def _consume_events(self, authority: _RunAuthority) -> None:
        client = self._client
        if client is None:
            await self._publish_failure(RuntimeError("Hermes API client is unavailable"))
            return
        path = f"/v1/runs/{quote(authority.api_run_id, safe='')}/events"
        stream_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self._config.request_timeout_seconds,
            sock_connect=self._config.request_timeout_seconds,
            sock_read=None,
        )
        try:
            async with asyncio.timeout(self._config.request_timeout_seconds):
                response = await client.get(
                    self._config.base_url + path,
                    headers=self._headers(),
                    allow_redirects=False,
                    timeout=stream_timeout,
                )
            try:
                if response.status != 200:
                    raise RuntimeError("Hermes API event stream was rejected")
                if response.content_type != "text/event-stream":
                    raise RuntimeError("Hermes API event stream has the wrong content type")
                while True:
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break
                    if len(raw_line) > _MAX_SSE_LINE_BYTES:
                        raise RuntimeError("Hermes API event line exceeds the bound")
                    if raw_line in {b"\n", b"\r\n"} or raw_line.startswith(b":"):
                        continue
                    if not raw_line.startswith(b"data: "):
                        continue
                    payload = self._strict_json(raw_line[6:].strip())
                    if type(payload) is not dict:
                        raise RuntimeError("Hermes API event must be an object")
                    await self._consume_event(authority, payload)
                    if authority.terminal:
                        return
                if not authority.terminal:
                    raise RuntimeError("Hermes API event stream ended before terminal evidence")
            finally:
                response.close()
        except asyncio.CancelledError:
            raise
        except BaseException as stream_error:
            if self._closed:
                return
            settlement = asyncio.create_task(
                self._settle_authority(authority),
                name=f"hermes-api-settle-{authority.task_id}",
            )
            try:
                await asyncio.shield(settlement)
            except asyncio.CancelledError:
                await asyncio.shield(settlement)
                raise
            except BaseException as settlement_error:
                await self._publish_failure(
                    BaseExceptionGroup(
                        "Hermes API event stream failed with unresolved authority",
                        [stream_error, settlement_error],
                    )
                )
                return
            if not authority.terminal:
                await self._publish_failure(
                    RuntimeError("Hermes API event stream settlement lacked terminal evidence")
                )

    async def _consume_event(self, authority: _RunAuthority, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        run_id = payload.get("run_id")
        if type(event) is not str or type(run_id) is not str:
            raise RuntimeError("Hermes API event lacks exact identity")
        if run_id != authority.api_run_id:
            raise RuntimeError("Hermes API event belongs to another run")
        observer = self._event_observer
        if observer is not None:
            raw_tool = payload.get("tool")
            tool = raw_tool if type(raw_tool) is str else None
            observer(event, tool)
        if event == "approval.request":
            await self._consume_approval(authority, payload)
            return
        if event not in {"run.completed", "run.failed", "run.cancelled"}:
            return
        if event == "run.completed":
            output = self._bounded_terminal_text(payload.get("output"), "summary")
            status = WorkTerminalStatus.COMPLETED
            terminal_payload = WorkCompletedPayload(status=status, summary=output)
        elif event == "run.failed":
            reason = self._bounded_terminal_text(payload.get("error"), "reason")
            status = WorkTerminalStatus.FAILED
            terminal_payload = WorkCompletedPayload(status=status, reason=reason)
        else:
            terminal_payload = WorkCompletedPayload(
                status=WorkTerminalStatus.INTERRUPTED,
                reason="Hermes run was interrupted",
            )
        update = WorkCompletedEvent(
            event_id=self._event_id("terminal"),
            session_id=self._session_id,
            sequence=self._next_sequence(),
            timestamp=datetime.now(UTC),
            type="work.completed",
            task_id=authority.task_id,
            run_id=authority.protocol_run_id,
            payload=terminal_payload,
        )
        await self._commit_terminal(authority, update)

    async def _consume_approval(
        self,
        authority: _RunAuthority,
        payload: dict[str, Any],
    ) -> None:
        command = payload.get("command")
        description = payload.get("description")
        choices = payload.get("choices")
        if type(command) is not str or type(description) is not str:
            raise RuntimeError("Hermes approval payload is malformed")
        if (
            not command.strip()
            or not description.strip()
            or len(command) > 4096
            or len(description) > 1024
        ):
            raise RuntimeError("Hermes approval text is outside the bound")
        if type(choices) is not list or "deny" not in choices or "once" not in choices:
            raise RuntimeError("Hermes approval choices are not actionable")
        async with self._state_lock:
            if authority.terminal:
                raise RuntimeError("terminal Hermes run requested approval")
            if len(self._pending_approvals) >= _MAX_PENDING_APPROVALS:
                raise RuntimeError("Hermes approval capacity exhausted")
            if any(item.api_run_id == authority.api_run_id for item in self._pending_approvals):
                raise RuntimeError("this Hermes run already has a pending approval")
            approval_id = f"approval_{secrets.token_hex(16)}"
            if approval_id in self._reserved_approval_ids:
                raise RuntimeError("Hermes approval authority is duplicated")
            self._reserved_approval_ids.add(approval_id)
            pending = _PendingApproval(
                approval_id=approval_id,
                task_id=authority.task_id,
                api_run_id=authority.api_run_id,
            )
            self._pending_approvals.append(pending)
        observer = self._approval_observer
        if observer is not None:
            try:
                observer(
                    {
                        "actionable": True,
                        "approvalId": pending.approval_id,
                        "command": command,
                        "description": description,
                        "state": "pending",
                        "taskId": authority.task_id,
                    }
                )
            except BaseException:
                async with self._state_lock:
                    with_context = deque(
                        item for item in self._pending_approvals if item != pending
                    )
                    self._pending_approvals = with_context
                raise

    async def _commit_terminal(self, authority: _RunAuthority, update: WorkCompletedEvent) -> None:
        async with self._state_lock:
            if authority.terminal:
                raise RuntimeError("Hermes API replayed terminal evidence")
            if self._updates.full():
                raise RuntimeError("Hermes API terminal update capacity exhausted")
            authority.terminal = True
            recorded = self._discard_admitted_locked(authority.api_run_id)
            if self._runs_by_task.get(authority.task_id) is authority:
                del self._runs_by_task[authority.task_id]
            if self._runs_by_api.get(authority.api_run_id) is authority:
                del self._runs_by_api[authority.api_run_id]
            removed = tuple(
                item for item in self._pending_approvals if item.api_run_id == authority.api_run_id
            )
            self._pending_approvals = deque(
                item for item in self._pending_approvals if item.api_run_id != authority.api_run_id
            )
            self._updates.put_nowait(update)
        for pending in removed:
            self._publish_approval_closed(pending, state="withdrawn")
        if recorded:
            await self._persist_record_best_effort()

    def _publish_approval_closed(self, pending: _PendingApproval, *, state: str) -> None:
        observer = self._approval_observer
        if observer is not None:
            with contextlib.suppress(BaseException):
                observer(
                    {
                        "actionable": False,
                        "approvalId": pending.approval_id,
                        "state": state,
                        "taskId": pending.task_id,
                    }
                )

    async def _publish_failure(self, error: BaseException) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._updates.put_nowait(error)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        client = self._client
        if client is None:
            raise RuntimeError("Hermes API client is not started")
        try:
            async with client.request(
                method,
                self._config.base_url + path,
                headers=self._headers(json_body=body is not None) | (headers or {}),
                json=body,
                allow_redirects=False,
            ) as response:
                raw = await response.content.read(_MAX_HTTP_BODY_BYTES + 1)
                if len(raw) > _MAX_HTTP_BODY_BYTES:
                    raise RuntimeError("Hermes API response exceeds the body bound")
                if response.status in {301, 302, 303, 307, 308}:
                    raise RuntimeError("Hermes API redirects are forbidden")
                payload = self._strict_json(raw)
                if type(payload) is not dict:
                    raise RuntimeError("Hermes API response must be an object")
                return response.status, payload
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError) as error:
            raise _NoResponse("Hermes API request failed") from error

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._config.bearer}",
            "Accept": "application/json",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _strict_json(raw: bytes) -> object:
        try:
            return json.loads(raw.decode("utf-8"), object_pairs_hook=run_record.strict_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("Hermes API returned malformed JSON") from error

    async def _best_effort_stop(self, api_run_id: str) -> None:
        client = self._client
        if client is None or client.closed:
            return
        try:
            await self._request_json(
                "POST",
                f"/v1/runs/{quote(api_run_id, safe='')}/stop",
                body={},
            )
        except BaseException:
            return

    async def _settle_authority(self, authority: _RunAuthority) -> None:
        if authority.terminal:
            return
        payload = await self._stop_and_wait(authority.api_run_id)
        if not authority.terminal:
            await self._consume_event(authority, payload)

    async def _stop_and_wait(self, api_run_id: str) -> dict[str, Any]:
        terminal = await self._poll_run_once(api_run_id)
        if terminal is not None:
            return terminal
        status, payload = await self._request_json(
            "POST",
            f"/v1/runs/{quote(api_run_id, safe='')}/stop",
            body={},
        )
        if status == 200:
            if payload.get("run_id") != api_run_id:
                raise RuntimeError("Hermes API stop response is not authoritative")
            if payload.get("status") != "stopping":
                # A stop 200 is either the bare stopping acknowledgment or, for a run that
                # already finished, its full status object; only the latter is evidence.
                if payload.get("object") != "hermes.run":
                    raise RuntimeError("Hermes API stop response is not authoritative")
                terminal = self._terminal_from_status(api_run_id, payload)
                if terminal is None:
                    raise RuntimeError("Hermes API stop response is not authoritative")
                return terminal
        elif _is_run_not_found(status, payload):
            raise _RunNotFound("Hermes API stop request failed")
        elif status != 409:
            raise RuntimeError("Hermes API stop request failed")
        try:
            async with asyncio.timeout(self._config.settlement_timeout_seconds):
                while True:
                    terminal = await self._poll_run_once(api_run_id)
                    if terminal is not None:
                        return terminal
                    await asyncio.sleep(self._config.settlement_poll_seconds)
        except TimeoutError as error:
            raise RuntimeError(
                "Hermes API run did not reach authoritative terminal status"
            ) from error

    async def _poll_run_once(self, api_run_id: str) -> dict[str, Any] | None:
        status, payload = await self._request_json("GET", f"/v1/runs/{quote(api_run_id, safe='')}")
        if _is_run_not_found(status, payload):
            raise _RunNotFound("Hermes API run status is not authoritative")
        if status != 200 or payload.get("run_id") != api_run_id:
            raise RuntimeError("Hermes API run status is not authoritative")
        return self._terminal_from_status(api_run_id, payload)

    @staticmethod
    def _terminal_from_status(api_run_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        run_status = payload.get("status")
        if run_status == "completed":
            return {
                "event": "run.completed",
                "run_id": api_run_id,
                "output": payload.get("output"),
            }
        if run_status == "failed":
            return {
                "event": "run.failed",
                "run_id": api_run_id,
                "error": payload.get("error"),
            }
        # Hermes reports a run whose gateway restarted before it settled as interrupted.
        if run_status in {"cancelled", "interrupted"}:
            return {"event": "run.cancelled", "run_id": api_run_id}
        if run_status not in _LIVE_RUN_STATUSES:
            raise RuntimeError("Hermes API run status is malformed")
        return None

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.closed:
            await client.close()

    def _dispatch_ack(
        self,
        request: WorkDispatchRequestedEvent,
        *,
        accepted: bool,
        run_id: str | None = None,
        reason: str | None = None,
    ) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            event_id=self._event_id("dispatch_ack"),
            session_id=self._session_id,
            sequence=self._next_sequence(request.sequence + 1),
            timestamp=datetime.now(UTC),
            type="work.dispatch.acknowledged",
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=accepted,
                run_id=run_id,
                reason=reason,
            ),
        )

    def _cancel_ack(
        self,
        request: ControlCancelEvent,
        *,
        accepted: bool,
        run_ids: list[str] | None = None,
        reason: str | None = None,
    ) -> ControlCancelAcknowledgedEvent:
        return ControlCancelAcknowledgedEvent(
            event_id=self._event_id("cancel_ack"),
            request_event_id=request.event_id,
            session_id=self._session_id,
            sequence=self._next_sequence(request.sequence + 1),
            timestamp=datetime.now(UTC),
            type="control.cancel.acknowledged",
            scope=CancelScope.TASK,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=accepted,
                signaled_run_ids=run_ids or [],
                reason=reason,
            ),
        )

    def _next_sequence(self, minimum: int = 0) -> int:
        candidate = max(self._sequence + 1, minimum)
        if candidate > _MAX_SEQUENCE:
            raise RuntimeError("Hermes API protocol sequence exhausted")
        self._sequence = candidate
        return candidate

    def _event_id(self, prefix: str) -> str:
        return f"{prefix}_{self._next_sequence()}"

    def _require_started_locked(self) -> None:
        if self._closed:
            raise RuntimeError("Hermes API task session is closed")
        if not self._started:
            raise RuntimeError("Hermes API task session is not started")

    @staticmethod
    def _trusted_dispatch(request: WorkDispatchRequestedEvent) -> WorkDispatchRequestedEvent:
        if type(request) is not WorkDispatchRequestedEvent:
            raise TypeError("dispatch request must be an exact protocol event")
        return WorkDispatchRequestedEvent.model_validate(request.model_dump(mode="python"))

    @staticmethod
    def _trusted_cancel(request: ControlCancelEvent) -> ControlCancelEvent:
        if type(request) is not ControlCancelEvent:
            raise TypeError("cancel request must be an exact protocol event")
        trusted = ControlCancelEvent.model_validate(request.model_dump(mode="python"))
        if trusted.payload.scope is not CancelScope.TASK:
            raise ValueError("Hermes API session only accepts task cancellation")
        return trusted

    @staticmethod
    def _bounded_terminal_text(value: object, field_name: str) -> str:
        if type(value) is not str or not value.strip():
            raise RuntimeError(f"Hermes API terminal {field_name} is missing")
        text = value.strip()
        if len(text) > _MAX_TERMINAL_TEXT_CHARS:
            raise RuntimeError(f"Hermes API terminal {field_name} exceeds the bound")
        if "deleg_" in text or _RUN_ID_DISCLOSURE.search(text) is not None:
            raise RuntimeError(f"Hermes API terminal {field_name} contains private authority")
        return text
