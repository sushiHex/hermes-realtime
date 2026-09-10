"""Bounded child-only observations of real evidence owners and durable storage."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import current_thread
from typing import Any, cast

from scripts.equivalence_process import _require


def _commit(key: bytes, domain: str, value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hmac.new(key, domain.encode("ascii") + b"\0" + raw, hashlib.sha256).hexdigest()


def _user_commitment(key: bytes, source: str, text: str) -> str:
    _require(source in {"typed", "microphone"}, "input source is invalid")
    return _commit(
        key,
        "user",
        json.dumps([source, text], ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


_CONSENT_FIELDS = (
    "consent_version",
    "disclosure_digest",
    "retention_hours",
    "microphone_accepted",
    "typed_accepted",
)


def _consent_commitment(key: bytes, fields: dict[str, Any]) -> str:
    raw = json.dumps(
        {name: fields[name] for name in _CONSENT_FIELDS},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _commit(key, "accepted_consent", raw)


def _request_commitment(key: bytes, sequence: int, fingerprint: str) -> str:
    return _commit(
        key, "accepted_request", json.dumps([sequence, fingerprint], separators=(",", ":"))
    )


def _command_commitment(key: bytes, command: Any) -> str:
    from hermes_realtime.evidence.models import CreateEpochV1, RolloverSessionV1

    _require(type(command) in (CreateEpochV1, RolloverSessionV1), "command type differs")
    return _payload_commitment(key, command)


def _payload_commitment(key: bytes, command: Any) -> str:
    from hermes_realtime.evidence.models import (
        CreateEpochV1,
        DrainAndStopV1,
        QueuedEvidenceRecordV1,
        RolloverSessionV1,
    )

    _require(
        type(command) in (CreateEpochV1, DrainAndStopV1, QueuedEvidenceRecordV1, RolloverSessionV1),
        "queue payload type differs",
    )
    return _commit(
        key,
        "dispatched_command",
        json.dumps(
            asdict(command),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


async def _accept_consent(*, port: int, origin: str, token: str, key: bytes) -> dict[str, str]:
    from tests.integration import test_qualification_full_host_ingress as ingress

    capture = await ingress._wait_event(
        port=port, origin=origin, token=token, kind="capture_status"
    )
    data = capture["data"]
    _require(type(data) is dict, "capture status is malformed")
    assert type(data) is dict
    request = {
        "sequence": 1,
        "accepted": True,
        "consentVersion": "realtime-evidence-consent-v1",
        "disclosureDigest": data["disclosureDigest"],
        "retentionHours": 24,
        "sources": {"microphone": True, "typed": True},
    }
    raw = json.dumps(request, separators=(",", ":")).encode("utf-8")
    status, result = await ingress._request(
        port=port,
        origin=origin,
        path="/api/v1/evidence-consent",
        bearer=token,
        body=raw,
    )
    _require(
        status == 200
        and type(result.get("sequence")) is int
        and result
        == {
            "captureState": "active",
            "result": "consent_activated",
            "sequence": 1,
        },
        "consent request was not accepted",
    )
    # Anchor the receipt to the bytes actually dispatched through the host API,
    # independently of every subsequently observed SQLite row.
    sent = json.loads(raw)
    consent = _consent_commitment(
        key,
        {
            "consent_version": sent["consentVersion"],
            "disclosure_digest": sent["disclosureDigest"],
            "retention_hours": sent["retentionHours"],
            "microphone_accepted": sent["sources"]["microphone"],
            "typed_accepted": sent["sources"]["typed"],
        },
    )

    fingerprint = hashlib.sha256(
        json.dumps(sent, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "consent": consent,
        "request": _request_commitment(key, sent["sequence"], fingerprint),
    }


def _consent_callback_observation(runtime: Any, key: bytes) -> dict[str, str]:
    """Read the actual consent call already invoking the owned writer factory."""
    from inspect import currentframe

    from hermes_realtime.client.session import BrowserBindingSnapshot
    from hermes_realtime.evidence.models import EvidenceConsentRequestV1
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import _reserve_host_evidence_consent

    frame = currentframe()
    try:
        for _ in range(4):
            frame = None if frame is None else frame.f_back
            if frame is not None and frame.f_code is _reserve_host_evidence_consent.__code__:
                break
        else:
            raise ValueError("writer factory is outside the actual consent callback")
        assert frame is not None
        _require(
            type(runtime) is HostEvidenceRuntimeV1 and frame.f_locals["runtime"] is runtime,
            "consent callback differs from the actual writer runtime",
        )
        binding, request = frame.f_locals["binding"], frame.f_locals["request"]
        _require(
            type(binding) is BrowserBindingSnapshot
            and type(binding.binding_generation) is int
            and 1 <= binding.binding_generation < 2**53
            and type(binding.participant_identity) is str
            and bool(binding.participant_identity)
            and type(request) is EvidenceConsentRequestV1,
            "actual consent callback authority is invalid",
        )
        fields = {
            "accepted": request.accepted,
            "consentVersion": request.consent_version,
            "disclosureDigest": request.disclosure_digest,
            "retentionHours": request.retention_hours,
            "sequence": request.sequence,
            "sources": {"microphone": request.sources.microphone, "typed": request.sources.typed},
        }
        fingerprint = hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "generation": _commit(key, "browser_generation", str(binding.binding_generation)),
            "participant": _commit(key, "browser_participant", binding.participant_identity),
            "request": _request_commitment(key, request.sequence, fingerprint),
            "consent": _consent_commitment(
                key,
                {
                    "consent_version": request.consent_version,
                    "disclosure_digest": request.disclosure_digest,
                    "retention_hours": request.retention_hours,
                    "microphone_accepted": request.sources.microphone,
                    "typed_accepted": request.sources.typed,
                },
            ),
        }
    finally:
        # Never retain frames, raw request values, or participant identifiers.
        del frame


async def _live_browser_binding(launcher: Any, key: bytes) -> dict[str, str]:
    from inspect import getclosurevars

    from hermes_realtime.client.runtime import BrowserClientRuntime
    from hermes_realtime.client.session import BrowserBindingSnapshot, BrowserSessionDirector
    from hermes_realtime.launcher import LocalBrowserLauncher
    from hermes_realtime.livekit.worker import LiveKitConversationWorker

    _require(type(launcher) is LocalBrowserLauncher, "live generation launcher is not exact")
    # The composed shutdown facade retains this same browser runtime in start
    # and close. Read its actual owners without replacing either authority.
    runtime = getclosurevars(launcher._runtime.start).nonlocals["runtime"]
    _require(type(runtime) is BrowserClientRuntime, "live browser runtime is not exact")
    sessions, worker = runtime._sessions, runtime._worker
    _require(
        type(sessions) is BrowserSessionDirector and type(worker) is LiveKitConversationWorker,
        "live browser and media owners are not exact",
    )
    binding = await sessions.current_binding_snapshot(participant_identity=sessions.active_identity)
    generation = worker.active_generation
    _require(
        type(binding) is BrowserBindingSnapshot
        and type(generation) is int
        and 1 <= generation < 2**53
        and type(binding.binding_generation) is int
        and binding.binding_generation == generation
        and type(worker._participant_identity) is str
        and binding.participant_identity == worker._participant_identity,
        "live browser and media generations differ",
    )
    return {
        "browser_generation": _commit(key, "browser_generation", str(binding.binding_generation)),
        "worker_generation": _commit(key, "browser_generation", str(generation)),
        "browser_participant": _commit(key, "browser_participant", binding.participant_identity),
        "worker_participant": _commit(key, "browser_participant", worker._participant_identity),
    }


def _control_commitments(key: bytes, snapshots: Any) -> list[str]:
    from hermes_realtime.evidence.models import evidence_snapshot_to_primitive
    from hermes_realtime.evidence.sqlite_spool import canonical_json_bytes

    return [
        _commit(
            key,
            "dispatched_control",
            canonical_json_bytes(evidence_snapshot_to_primitive(snapshot)),
        )
        for snapshot in snapshots
    ]


def _snapshot(database: Path, key: bytes, *, max_events: int = 32) -> dict[str, Any]:
    _require(type(max_events) is int and max_events in {32, 64}, "store read bound differs")
    from hermes_realtime.evidence.models import (
        parse_evidence_snapshot_json,
        validate_canonical_utc,
        validate_event_sequence,
    )
    from hermes_realtime.evidence.sqlite_spool import canonical_json_bytes, hre1_record_hash

    sessions: list[dict[str, Any]] = []
    lineage = []
    recorded_times: list[str] = []
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        _require(
            connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
            "rollover store integrity failed",
        )
        _require(
            not connection.execute("PRAGMA foreign_key_check").fetchall(),
            "rollover store foreign keys failed",
        )
        installations = connection.execute(
            "SELECT installation_id,created_at_utc,clock_high_water_utc,"
            "purge_required,purge_reason,purge_scope FROM producer_installation"
        ).fetchall()
        _require(len(installations) == 1, "rollover installation authority is not unique")
        installation, created, high_water, purge, reason, scope = installations[0]
        validate_canonical_utc(created, field_name="created_at_utc")
        validate_canonical_utc(high_water, field_name="clock_high_water_utc")
        _require((purge, reason, scope) == (0, None, None), "rollover installation requires purge")
        epochs = connection.execute(
            "SELECT consent_epoch_id,producer_instance_id,state,opened_at_utc,closed_at_utc "
            "FROM consent_epochs"
        ).fetchall()
        _require(
            len(epochs) == 1
            and epochs[0][2] == "active"
            and epochs[0][3] == created
            and epochs[0][4] is None,
            "rollover has no unique active consent epoch",
        )
        for table in ("evidence_conflicts", "erasure_requests", "erasure_tombstones"):
            _require(
                connection.execute("SELECT COUNT(*) FROM " + table).fetchone() == (0,),
                "fresh rollover carries conflicts or erasure authority",
            )
        rows = connection.execute(
            "SELECT logical_session_id,consent_epoch_id,producer_instance_id,state,event_count,"
            "canonical_bytes,final_event_sequence,head_hash,taint_code,consent_version, "
            "opened_at_utc,expires_at_utc "
            "FROM evidence_sessions ORDER BY rowid"
        ).fetchall()
        _require(1 <= len(rows) <= 2, "rollover session count differs")
        for row in rows:
            (
                session_id,
                epoch,
                producer,
                state,
                count,
                size,
                final,
                head,
                taint,
                version,
                opened,
                expires,
            ) = row
            validate_canonical_utc(opened, field_name="opened_at_utc")
            validate_canonical_utc(expires, field_name="expires_at_utc")
            _require(opened < expires, "rollover session expiry is not after opening")
            _require((epoch, producer) == epochs[0][:2], "session and active epoch lineage differ")
            events = connection.execute(
                "SELECT event_id,event_sequence,event_kind,canonical_payload,payload_hash,"
                "previous_hash,record_hash,recorded_at_utc,canonical_bytes "
                "FROM evidence_events WHERE logical_session_id=? ORDER BY event_sequence",
                (session_id,),
            ).fetchall()
            _require(1 <= len(events) <= max_events, "rollover event count differs")
            history = []
            previous = None
            previous_time = created
            payloads = []
            for ordinal, event in enumerate(events, 1):
                event_id, sequence, kind, payload, payload_hash, stored_previous, digest, at, n = (
                    event
                )
                validate_canonical_utc(at, field_name="recorded_at_utc")
                _require(at >= previous_time, "stored event time moves backward")
                previous_time = at
                recorded_times.append(at)
                parsed = json.loads(payload)
                raw = canonical_json_bytes(parsed)
                _require(
                    raw == payload.encode("utf-8") and len(raw) == n,
                    "stored payload is noncanonical",
                )
                _require(
                    sequence == ordinal
                    and previous == stored_previous
                    and hashlib.sha256(raw).hexdigest() == payload_hash
                    and hre1_record_hash(
                        installation_id=installation,
                        producer_instance_id=producer,
                        event_id=event_id,
                        logical_session_id=session_id,
                        event_sequence=sequence,
                        event_kind=kind,
                        recorded_at_utc=at,
                        payload_hash=payload_hash,
                        previous_hash=previous,
                    )
                    == digest,
                    "stored evidence chain differs",
                )
                history.append(
                    parse_evidence_snapshot_json(
                        canonical_json_bytes(
                            {
                                "schema_version": 1,
                                "installation_id": installation,
                                "producer_instance_id": producer,
                                "logical_session_id": session_id,
                                "event_id": event_id,
                                "event_sequence": sequence,
                                "event_kind": kind,
                                "payload": parsed,
                            }
                        )
                    )
                )
                payloads.append(parsed)
                previous = digest
            _require(
                count == len(events) and size == sum(e[8] for e in events),
                "stored session aggregate differs",
            )
            _require(
                (state == "open" and final is head is taint is None)
                or (state == "sealed" and final == count and head == previous and taint is None),
                "rollover session terminal authority differs",
            )
            if state == "sealed":
                _require(
                    [e[2] for e in events[-2:]] == ["binding_closed", "session_seal_requested"]
                    and events[-2][7] == events[-1][7]
                    and payloads[-2]["close_reason"] == "capacity_rollover"
                    and payloads[-2]["binding_id"] == payloads[0]["binding_id"]
                    and payloads[-1]["final_event_sequence"] == count
                    and all(
                        payloads[-1][field] == payloads[0][field]
                        for field in ("consent_epoch_id", "consent_version", "disclosure_digest")
                    ),
                    "rollover seal payloads differ",
                )
                validate_event_sequence(tuple(history[:-2]))
            else:
                validate_event_sequence(tuple(history))
            _require(
                payloads[0]["consent_epoch_id"] == epoch
                and payloads[0]["consent_version"] == version,
                "session row and opening epoch differ",
            )
            _require(
                opened == events[0][7] == events[1][7]
                and (
                    opened == created
                    if not sessions
                    else _commit(key, "session_time", opened) == sessions[-1]["last_event_at"]
                ),
                "session opening timestamps differ",
            )
            lag = (
                datetime.fromisoformat(opened)
                + timedelta(hours=payloads[0]["retention_hours"])
                - datetime.fromisoformat(expires)
            )
            lag_us = (lag.days * 86_400 + lag.seconds) * 1_000_000 + lag.microseconds
            # Rollover expiry is sampled by the runtime before the writer's
            # opening clock. Bound that skew by the existing five-second
            # observation authority; epoch creation uses a single clock sample.
            _require(
                (lag_us == 0 if not sessions else 0 <= lag_us <= 5_000_000),
                "session retention interval differs from accepted consent",
            )
            consent = {
                "opening": {
                    name: value
                    for name, value in payloads[0].items()
                    if name != "predecessor_session_id"
                },
                "binding": payloads[1],
            }
            lineage.append((producer, consent))
            _require(
                all(value == lineage[0] for value in lineage),
                "rollover changed the complete consent or production binding envelope",
            )
            predecessor = payloads[0]["predecessor_session_id"]
            sessions.append(
                {
                    "session": _commit(key, "session", session_id),
                    "epoch": _commit(key, "epoch", epoch),
                    "consent": _commit(key, "consent", canonical_json_bytes(consent)),
                    "consent_request": _consent_commitment(key, payloads[0]),
                    "controls": _control_commitments(
                        key, history[:2] + (history[-2:] if state == "sealed" else [])
                    ),
                    "records": _control_commitments(
                        key, history[2:-2] if state == "sealed" else history[2:]
                    ),
                    "opened": _commit(key, "session_time", opened),
                    "expires": _commit(key, "session_time", expires),
                    "retention_lag_us": lag_us,
                    "last_event_at": _commit(key, "session_time", events[-1][7]),
                    "predecessor": ""
                    if predecessor is None
                    else _commit(key, "session", predecessor),
                    "state": state,
                    "events": count,
                    "chain": [_commit(key, "chain", e[6]) for e in events],
                    "kinds": [e[2] for e in events],
                    "user": [
                        _user_commitment(key, p["source"], p["text"])
                        for e, p in zip(events, payloads, strict=True)
                        if e[2] == "user_final_accepted"
                    ],
                    "generated": [
                        _commit(key, "generated", p["text"])
                        for e, p in zip(events, payloads, strict=True)
                        if e[2] == "assistant_segment_generated"
                    ],
                    "transport": [
                        _commit(key, "transport", p["text"])
                        for e, p in zip(events, payloads, strict=True)
                        if e[2] == "assistant_chunk_transport_confirmed_full"
                    ],
                }
            )
        _require(
            high_water == max(recorded_times),
            "installation high-water differs from durable event time",
        )
        authority = _commit(
            key,
            "store_authority",
            canonical_json_bytes(
                [[installation, created, purge, reason, scope], [list(epoch) for epoch in epochs]]
            ),
        )
    return {
        "sessions": sessions,
        "clock": _commit(key, "session_time", high_water),
        "authority": authority,
    }


class _ObserveThreadOwners:
    """Retain actual owner objects; export only invocation-local commitments."""

    def __init__(self, transport: Any, key: bytes, *, scenario: str = "capacity_rollover") -> None:
        from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceWriterDaemonV1

        _require(type(transport) is SQLiteEvidenceWriterDaemonV1, "SQLite transport is not exact")
        _require(scenario in {"capacity_rollover", "over_budget_turn"}, "owner scenario differs")
        self.dequeue_limit = 21 if scenario == "capacity_rollover" else 61
        self.call_limit = 32 if scenario == "capacity_rollover" else 64
        self.transport, self.key = transport, key
        self.event_loop = current_thread()
        self.sqlite = transport._thread
        self.dispatcher: Any = None
        self.bound_dispatcher: Any = None
        self.dispatcher_binding: tuple[Any, Any, Any] | None = None
        self.calls: list[dict[str, str]] = []
        self.dequeues = 0
        self.failed = False

    def _check(self, condition: bool, message: str) -> None:
        if not condition:
            self.failed = True
        _require(condition, message)

    def _identity(self, thread: Any) -> str:
        self._check(thread is not None and type(thread.ident) is int, "owner thread never started")
        return _commit(self.key, "thread", str(thread.ident))

    def dequeued(self) -> None:
        observed = current_thread()
        self._check(
            observed is not self.event_loop
            and observed is not self.sqlite
            and (self.dispatcher is None or self.dispatcher is observed),
            "dequeue did not run on its independent dispatcher thread",
        )
        self.dispatcher = observed
        self.dequeues += 1
        self._check(self.dequeues <= self.dequeue_limit, "dispatcher observation overflow")

    def _check_dispatcher_binding(
        self, owner: Any, queue: Any, admission: Any, transport: Any
    ) -> None:
        from types import MethodType

        from hermes_realtime.evidence.admission import (
            BoundedEvidenceWriterQueueV1,
            EvidenceAdmissionControllerV1,
        )
        from hermes_realtime.evidence.runtime import (
            EvidenceWriterDispatcherV1,
            EvidenceWriterRuntimeOwnerV1,
        )

        self._check(type(owner) is EvidenceWriterRuntimeOwnerV1, "dispatcher owner is not exact")
        method = owner._dispatch
        self._check(
            type(method) is MethodType
            and method.__func__ is EvidenceWriterDispatcherV1.dispatch_item
            and type(method.__self__) is EvidenceWriterDispatcherV1,
            "owner does not call the exact production dispatcher",
        )
        dispatcher = method.__self__
        self._check(
            type(queue) is BoundedEvidenceWriterQueueV1
            and owner._source is dispatcher._source is queue
            and type(admission) is EvidenceAdmissionControllerV1
            and dispatcher._admission is admission
            and admission._writer_sink is queue
            and dispatcher._transport is transport,
            "production dispatcher does not bind the observed queue, admission, and transport",
        )

    def bind_dispatcher(self, owner: Any, *, queue: Any, admission: Any, transport: Any) -> None:
        self._check_dispatcher_binding(owner, queue, admission, transport)
        self._check(
            owner._thread is self.dispatcher and owner.is_running,
            "observed dispatcher is not the retained production owner",
        )
        self.bound_dispatcher = owner
        self.dispatcher_binding = (queue, admission, transport)

    def spool_call(self, stage: str) -> None:
        self._check(
            current_thread() is self.sqlite
            and self.sqlite is not self.event_loop
            and self.sqlite is not self.dispatcher
            and self.transport.owner_thread_id == self.sqlite.ident
            and self.sqlite.is_alive(),
            "SQLite call did not run on its dedicated owner thread",
        )
        self._check(
            stage
            in {
                "factory",
                "create_epoch",
                "append_record",
                "rollover_session",
                "active_session_expiry",
                "drain_and_close",
                "close",
                "close_owner_marker",
            }
            and len(self.calls) < self.call_limit,
            "SQLite owner observation is unknown or oversized",
        )
        self.calls.append({"stage": stage, "thread": self._identity(current_thread())})

    def finished(self) -> dict[str, Any]:
        self._check(self.dispatcher_binding is not None, "dispatcher was never bound")
        assert self.dispatcher_binding is not None
        self._check_dispatcher_binding(self.bound_dispatcher, *self.dispatcher_binding)
        self._check(
            not self.failed
            and self.bound_dispatcher is not None
            and self.bound_dispatcher._thread is self.dispatcher
            and self.bound_dispatcher.failure is None
            and self.transport._sticky_fault is None
            and self.dispatcher is not None
            and not self.dispatcher.is_alive()
            and not self.sqlite.is_alive()
            and not self.transport.is_running,
            "retained writer owners did not both stop cleanly",
        )
        return {
            "event_loop": self._identity(self.event_loop),
            "dispatcher": self._identity(self.dispatcher),
            "sqlite": self._identity(self.sqlite),
            "dequeues": self.dequeues,
            "calls": self.calls,
            "dispatcher_stopped": not self.dispatcher.is_alive(),
            "sqlite_stopped": not self.sqlite.is_alive(),
            "dispatcher_clean": self.bound_dispatcher.failure is None,
            "dispatcher_bound": True,
            "sqlite_clean": not self.failed and self.transport._sticky_fault is None,
        }


def _data_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA data_version").fetchone()
    _require(
        row is not None and len(row) == 1 and type(row[0]) is int and row[0] >= 1,
        "rollover connection data version is invalid",
    )
    return cast(int, row[0])


def _statement_kind(statement: str) -> str:
    remaining = statement.lstrip()
    while remaining.startswith(("--", "/*")):
        if remaining.startswith("--"):
            end = remaining.find("\n")
            _require(end >= 0, "unterminated SQL comment")
            remaining = remaining[end + 1 :].lstrip()
        else:
            end = remaining.find("*/", 2)
            _require(end >= 0, "unterminated SQL comment")
            remaining = remaining[end + 2 :].lstrip()
    _require(bool(remaining), "empty traced SQL statement")
    return remaining.split(None, 1)[0].upper()


_ACTIVE_CONNECTION_AUDIT: _ObserveConnectionAudit | None = None
_CONNECTION_AUDIT_INSTALLED = False


def _observe_connection_open(event: str, arguments: tuple[Any, ...]) -> None:
    audit = _ACTIVE_CONNECTION_AUDIT
    if audit is None or event != "sqlite3.connect":
        return
    # Python audits aliases and opens on other threads too. Never retain the
    # event's database argument, handles, stack, or connection identifiers.
    if (
        audit.reader_thread is current_thread()
        and len(arguments) == 1
        and arguments[0] == audit.readonly_uri
    ):
        audit.reads = min(audit.reads + 1, 128)
    else:
        audit.unexpected = min(audit.unexpected + 1, 128)


class _ObserveConnectionAudit:
    """Latch unexpected connection opens without changing SQLite execution."""

    def __init__(self, database: Path) -> None:
        self.readonly_uri = database.as_uri() + "?mode=ro"
        self.reader_thread: Any = None
        self.reads = self.unexpected = 0

    def __enter__(self) -> _ObserveConnectionAudit:
        global _ACTIVE_CONNECTION_AUDIT, _CONNECTION_AUDIT_INSTALLED
        _require(_ACTIVE_CONNECTION_AUDIT is None, "rollover connection audit overlapped")
        if not _CONNECTION_AUDIT_INSTALLED:
            sys.addaudithook(_observe_connection_open)
            _CONNECTION_AUDIT_INSTALLED = True
        _ACTIVE_CONNECTION_AUDIT = self
        return self

    def __exit__(self, *ignored: Any) -> None:
        global _ACTIVE_CONNECTION_AUDIT
        _ACTIVE_CONNECTION_AUDIT = None

    @contextmanager
    def readonly_snapshot(self) -> Iterator[None]:
        _require(self.reader_thread is None, "rollover reader scope overlapped")
        self.reader_thread = current_thread()
        try:
            yield
        finally:
            self.reader_thread = None

    def observation(self) -> dict[str, int]:
        _require(self.reads == 1 and self.unexpected == 0, "rollover connection observation failed")
        return {"observer_reads": self.reads, "unexpected": self.unexpected}


class _ObserveQueue:
    """Delegate the real blocking dequeue and inspect its unstripped envelope."""

    def __init__(
        self,
        queue: Any,
        key: bytes,
        *,
        owner_generation: int,
        owners: _ObserveThreadOwners | None = None,
        scenario: str = "capacity_rollover",
    ) -> None:
        from hermes_realtime.evidence.admission import BoundedEvidenceWriterQueueV1

        _require(
            type(queue) is BoundedEvidenceWriterQueueV1, "observed queue is not production-owned"
        )
        _require(scenario in {"capacity_rollover", "over_budget_turn"}, "queue scenario differs")
        self.rollover = scenario == "capacity_rollover"
        self.drain_index = 20 if self.rollover else 60
        self.queue, self.key, self.owner_generation = queue, key, owner_generation
        self.owners = owners
        self.original = queue.get_blocking
        self.records: list[dict[str, Any]] = []
        queue.get_blocking = self.get_blocking

    def restore(self) -> None:
        self.queue.get_blocking = self.original

    def get_blocking(self) -> Any:
        from hermes_realtime.evidence.admission import EvidenceWriterQueueItemV1, WriterQueueLane
        from hermes_realtime.evidence.models import (
            CreateEpochV1,
            DrainAndStopV1,
            QueuedEvidenceRecordV1,
            RolloverSessionV1,
        )

        item = self.original()
        if item is None:
            return None
        if self.owners is not None:
            self.owners.dequeued()
        index = len(self.records)
        _require(
            type(item) is EvidenceWriterQueueItemV1
            and type(item.protocol_version) is int
            and item.protocol_version == 1
            and index <= self.drain_index
            and type(item.admission_ordinal) is int
            and item.admission_ordinal == index + 2,
            "dequeued envelope protocol or admission ordinal differs",
        )
        payload = item.payload
        expected = (
            CreateEpochV1
            if index == 0
            else RolloverSessionV1
            if self.rollover and index == 13
            else DrainAndStopV1
            if index == self.drain_index
            else QueuedEvidenceRecordV1
        )
        _require(
            type(payload) is expected
            and item.lane
            is (WriterQueueLane.DRAIN if index == self.drain_index else WriterQueueLane.ORDERED),
            "dequeued envelope lane or payload differs",
        )
        if type(payload) in (QueuedEvidenceRecordV1, RolloverSessionV1):
            _require(
                payload.admission_ordinal == item.admission_ordinal,
                "queue envelope and payload ordinals differ",
            )
        elif type(payload) is DrainAndStopV1:
            _require(
                payload.final_admission_ordinal == item.admission_ordinal - 1
                and payload.owner_generation == self.owner_generation,
                "dequeued drain owner or watermark differs",
            )
        self.records.append(
            {
                "version": item.protocol_version,
                "ordinal": item.admission_ordinal,
                "lane": item.lane.value,
                "kind": {
                    CreateEpochV1: "create",
                    QueuedEvidenceRecordV1: "record",
                    RolloverSessionV1: "rollover",
                    DrainAndStopV1: "drain",
                }[type(payload)],
                "payload": _payload_commitment(self.key, payload),
            }
        )
        return item
