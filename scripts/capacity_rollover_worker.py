"""Observe real session rollover; raw SQLite and conversation bytes stay in the child."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, current_thread
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


def _snapshot(database: Path, key: bytes) -> dict[str, Any]:
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
            _require(1 <= len(events) <= 32, "rollover event count differs")
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

    def __init__(self, transport: Any, key: bytes) -> None:
        from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceWriterDaemonV1

        _require(type(transport) is SQLiteEvidenceWriterDaemonV1, "SQLite transport is not exact")
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
        self._check(self.dequeues <= 21, "dispatcher observation overflow")

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
            and len(self.calls) < 32,
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


class _ObserveRolloverSpool:
    """Observe on the real SQLite owner thread and always delegate the operation."""

    def __init__(
        self, delegate: Any, database: Path, key: bytes, owners: _ObserveThreadOwners | None = None
    ) -> None:
        self.delegate, self.database, self.key = delegate, database, key
        self.owners = owners
        self.snapshots: list[dict[str, Any]] = []
        self.transactions: list[str] = []
        self.transaction_writes: list[str] = []
        self.connections: dict[str, Any] = {}
        self.failed = False
        self.durable_terminals: list[str] = []
        self.third_settled = Event()
        self.commands: dict[str, Any] = {}
        self.records: list[dict[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.delegate, name)
        if self.owners is None or not callable(value):
            return value

        def observed(*arguments: Any, **keywords: Any) -> Any:
            assert self.owners is not None
            self.owners.spool_call(name)
            try:
                return value(*arguments, **keywords)
            except BaseException:
                # The real daemon suppresses close exceptions; retain refusal.
                self.owners.failed = True
                raise

        return observed

    def create_epoch(self, command: Any) -> Any:
        if self.owners is not None:
            self.owners.spool_call("create_epoch")
        _require(not self.commands, "create command was dispatched more than once")
        self.commands["create_dto"] = _command_commitment(self.key, command)
        self.commands["binding"] = _commit(
            self.key, "browser_generation", str(command.binding_generation)
        )
        self.commands["request"] = _request_commitment(
            self.key, command.control_sequence, command.control_fingerprint_hash
        )
        self.commands["create"] = _control_commitments(
            self.key, (command.session_opened, command.binding_opened)
        )
        result = self.delegate.create_epoch(command)
        _require(
            _command_commitment(self.key, self.delegate._accepted_create)
            == self.commands["create_dto"],
            "accepted create differs from the complete dispatched command",
        )
        snapshot = _snapshot(self.database, self.key)
        _require(
            len(snapshot["sessions"]) == 1
            and snapshot["sessions"][0]["controls"] == self.commands["create"],
            "stored opening differs from the dispatched create command",
        )
        return result

    def append_record(self, item: Any) -> Any:
        if self.owners is not None:
            self.owners.spool_call("append_record")
        _require(len(self.records) < 18, "extra ordinary spool record")
        dto = _payload_commitment(self.key, item)
        snapshot = _control_commitments(self.key, (item.snapshot,))[0]
        result = self.delegate.append_record(item)
        self.records.append({"dto": dto, "snapshot": snapshot, "result": result.value})
        if item.snapshot.event_kind.value == "turn_settled":
            _require(len(self.durable_terminals) < 3, "extra durable terminal")
            self.durable_terminals.append(result.value)
            if len(self.durable_terminals) == 3:
                self.third_settled.set()
        return result

    def rollover_session(self, command: Any) -> Any:
        if self.owners is not None:
            self.owners.spool_call("rollover_session")
        _require(not self.snapshots, "rollover was invoked more than once")
        _require(
            set(self.commands) == {"create", "create_dto", "request", "binding"},
            "dispatched create command is absent",
        )
        connection = self.delegate.connection
        _require(type(connection) is sqlite3.Connection, "rollover connection is not exact")
        version_before = _data_version(connection)
        self.commands["rollover_dto"] = _command_commitment(self.key, command)
        self.commands["rollover"] = _control_commitments(self.key, command.snapshots)
        self.commands["successor_expiry"] = _commit(
            self.key, "session_time", command.successor_expires_at_utc
        )
        self.snapshots.append(_snapshot(self.database, self.key))
        connection_audit = _ObserveConnectionAudit(self.database)

        def trace(statement: str) -> None:
            # SQLite suppresses trace callback exceptions. Latch any refusal;
            # retain only control names and write verbs, never SQL or values.
            try:
                kind = _statement_kind(statement)
                active = connection.in_transaction
                if kind == "SELECT" or statement in {"PRAGMA page_count", "PRAGMA max_page_count"}:
                    return
                if kind in {"INSERT", "UPDATE", "DELETE"}:
                    _require(
                        active
                        and self.transactions == ["BEGIN IMMEDIATE"]
                        and len(self.transaction_writes) < 128,
                        "rollover write escaped its observed transaction",
                    )
                    self.transaction_writes.append(kind)
                    return
                _require(
                    statement in {"BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"}
                    and len(self.transactions) < 3,
                    "rollover statement or transaction trace differs",
                )
                if kind == "BEGIN":
                    _require(not active and not self.transactions, "rollover transaction repeated")
                elif kind == "COMMIT":
                    _require(
                        active
                        and self.transactions == ["BEGIN IMMEDIATE"]
                        and len(self.snapshots) == 1
                        and bool(self.transaction_writes),
                        "rollover commit lacks its active write transaction",
                    )
                    with connection_audit.readonly_snapshot():
                        self.snapshots.append(_snapshot(self.database, self.key))
                self.transactions.append(statement)
            except Exception:
                self.failed = True

        with connection_audit:
            connection.set_trace_callback(trace)
            try:
                result = self.delegate.rollover_session(command)
            finally:
                connection.set_trace_callback(None)
        _require(
            not self.failed and not connection.in_transaction,
            "rollover transaction observation failed",
        )
        _require(
            _command_commitment(self.key, self.delegate._accepted_rollover)
            == self.commands["rollover_dto"],
            "accepted rollover differs from the complete dispatched command",
        )
        committed = _snapshot(self.database, self.key)
        self.snapshots.append(committed)
        version_after = _data_version(connection)
        _require(
            self.delegate.connection is connection and version_before == version_after,
            "rollover connection observation found an external commit or replacement",
        )
        self.connections = {
            **connection_audit.observation(),
            "data_version": [version_before, version_after],
        }
        predecessor, successor = committed["sessions"]
        _require(
            predecessor["controls"] == self.commands["create"] + self.commands["rollover"][:2]
            and successor["controls"] == self.commands["rollover"][2:]
            and successor["expires"] == self.commands["successor_expiry"],
            "stored rollover differs from the dispatched command",
        )
        return result


class _ObserveQueue:
    """Delegate the real blocking dequeue and inspect its unstripped envelope."""

    def __init__(
        self,
        queue: Any,
        key: bytes,
        *,
        owner_generation: int,
        owners: _ObserveThreadOwners | None = None,
    ) -> None:
        from hermes_realtime.evidence.admission import BoundedEvidenceWriterQueueV1

        _require(
            type(queue) is BoundedEvidenceWriterQueueV1, "observed queue is not production-owned"
        )
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
            and index < 21
            and type(item.admission_ordinal) is int
            and item.admission_ordinal == index + 2,
            "dequeued envelope protocol or admission ordinal differs",
        )
        payload = item.payload
        expected = (
            CreateEpochV1
            if index == 0
            else RolloverSessionV1
            if index == 13
            else DrainAndStopV1
            if index == 20
            else QueuedEvidenceRecordV1
        )
        _require(
            type(payload) is expected
            and item.lane is (WriterQueueLane.DRAIN if index == 20 else WriterQueueLane.ORDERED),
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


class _ObserveDispatch:
    """Commit commands before transport delegation and observe the real FIFO."""

    def __init__(self, delegate: Any, key: bytes) -> None:
        self.delegate, self.key = delegate, key
        self.commands: dict[str, Any] = {"ordinals": [], "records": []}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def _ordinal(self, ordinal: int) -> None:
        # Epoch creation consumes two record ordinals in one physical item.
        ordinals = self.commands["ordinals"]
        _require(
            type(ordinal) is int and len(ordinals) < 19 and ordinal == len(ordinals) + 3,
            "dispatched admission ordinal is not the next FIFO item",
        )
        ordinals.append(ordinal)

    def create_epoch(self, command: Any) -> Any:
        _require(set(self.commands) == {"ordinals", "records"}, "create dispatch repeated")
        self.commands["create_dto"] = _command_commitment(self.key, command)
        self.commands["request"] = _request_commitment(
            self.key, command.control_sequence, command.control_fingerprint_hash
        )
        return self.delegate.create_epoch(command)

    def append_record(self, item: Any) -> Any:
        self._ordinal(item.admission_ordinal)
        self.commands["records"].append(_payload_commitment(self.key, item))
        return self.delegate.append_record(item)

    def rollover_session(self, command: Any) -> Any:
        _require("rollover_dto" not in self.commands, "rollover dispatch repeated")
        self._ordinal(command.admission_ordinal)
        self.commands["rollover_ordinal"] = command.admission_ordinal
        self.commands["rollover_dto"] = _command_commitment(self.key, command)
        return self.delegate.rollover_session(command)


async def observe_capacity_rollover(workspace: Path, livekit_url: str) -> dict[str, Any]:
    from livekit import rtc

    from hermes_realtime._qualification import _QualificationCapacityObservationV1
    from hermes_realtime.host_launcher import build_local_host_launcher
    from hermes_realtime.production_observation import (
        CloseStageObservationV1,
        RolloverObservationV1,
        TerminalSettledObservationV1,
    )
    from tests.integration import test_qualification_full_host_ingress as ingress
    from tests.support import qualification as trace

    suffix, port = uuid.uuid4().hex[:10], ingress._available_port()
    database = workspace / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    key = os.urandom(32)
    composition = trace.InProcessQualificationComposition()
    transcriber = ingress._Transcriber()
    spools: list[_ObserveRolloverSpool] = []
    dispatches: list[_ObserveDispatch] = []
    queues: list[_ObserveQueue] = []
    reservations: list[tuple[Any, Any]] = []
    thread_owners: list[_ObserveThreadOwners] = []

    def writer_factory(runtime: Any) -> Any:
        reserve = runtime.reserve_browser_consent

        def observe_reservation(**arguments: Any) -> Any:
            authority = reserve(**arguments)
            _require(not queues, "queue reservation observation repeated")
            queues.append(
                _ObserveQueue(
                    runtime._queue, key, owner_generation=runtime._owner_generation, owners=owners
                )
            )
            return authority

        # The factory runs before the real consent reservation creates its
        # queue. Attach after that exact call returns, before consumer startup.
        runtime.reserve_browser_consent = observe_reservation
        reservations.append((runtime, reserve))
        transport = runtime.create_sqlite_transport()
        owners = _ObserveThreadOwners(transport, key)
        thread_owners.append(owners)
        factory = transport._spool_factory

        def observed_factory() -> _ObserveRolloverSpool:
            from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

            owners.spool_call("factory")
            delegate = factory()
            _require(type(delegate) is SQLiteEvidenceSpool, "SQLite spool is not exact")
            spool = _ObserveRolloverSpool(delegate, database, key, owners)
            spools.append(spool)
            return spool

        transport._spool_factory = observed_factory
        dispatch = _ObserveDispatch(transport, key)
        dispatches.append(dispatch)
        return dispatch

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            livekit_url=livekit_url,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=True,
            evidence_database=database,
        ),
        inference_factory=ingress._Inference,
        speech_presence_factory=ingress._Presence,
        synthesizer_factory=ingress._Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: ingress._Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory,
    )
    room, running = rtc.Room(), None
    inputs = []
    try:
        running = await composition.start_host(registration)
        origin, token, _source = await ingress._connect_and_activate(
            port=port,
            launch_url=running.url,
            room=room,
        )
        live_binding = await _live_browser_binding(running._host, key)
        accepted_consent = await _accept_consent(port=port, origin=origin, token=token, key=key)
        live_bindings = [live_binding, await _live_browser_binding(running._host, key)]
        _require(live_bindings[0] == live_bindings[1], "live binding changed during consent")
        _require(len(thread_owners) == len(reservations) == 1, "writer ownership is ambiguous")
        runtime = reservations[0][0]
        thread_owners[0].bind_dispatcher(
            runtime._writer,
            queue=queues[0].queue,
            admission=runtime._admission,
            transport=dispatches[0],
        )
        after = 0
        for sequence in range(1, 4):
            text = f"Synthetic capacity turn {sequence}."
            status, result = await ingress._request(
                port=port,
                origin=origin,
                path="/api/v1/input",
                bearer=token,
                body=json.dumps({"sequence": sequence, "text": text}).encode(),
            )
            _require(
                status == 202 and result == {"sequence": sequence, "version": 1},
                "rollover conversation input was not accepted",
            )
            inputs.append(_user_commitment(key, "typed", text))
            event = await ingress._wait_event(
                port=port,
                origin=origin,
                token=token,
                kind="assistant_turn_completed",
                after=after,
            )
            _require(type(event["sequence"]) is int, "completion sequence is invalid")
            after = cast(int, event["sequence"])
            async with asyncio.timeout(5):
                while True:
                    observations = composition.production_observations(running).records()
                    settled = sum(
                        type(item) is TerminalSettledObservationV1 for item in observations
                    )
                    rolled = any(
                        type(item) is RolloverObservationV1 and item.stage.value == "terminal"
                        for item in observations
                    )
                    if settled == sequence and (sequence < 2 or rolled):
                        break
                    await asyncio.sleep(0.01)
        _require(len(spools) == 1, "rollover writer ownership is ambiguous")
        spool = spools[0]
        async with asyncio.timeout(5):
            while not spool.third_settled.is_set():
                await asyncio.sleep(0.01)
        continued = _snapshot(database, key)
    finally:
        try:
            await room.disconnect()
        finally:
            try:
                if running is not None:
                    await composition.close_host(running)
            finally:
                for queue in queues:
                    queue.restore()
                for runtime, reserve in reservations:
                    runtime.reserve_browser_consent = reserve
    _require(running is not None, "rollover host was not started")
    observations = composition.production_observations(running)
    records = composition.trace.records()
    raw_capacity = composition.capacity_observations(running)
    _require(
        all(type(item) is _QualificationCapacityObservationV1 for item in raw_capacity),
        "capacity observations are not production-owned",
    )
    capacity = cast(tuple[_QualificationCapacityObservationV1, ...], raw_capacity)
    contexts = [
        json.loads(item.committed_conversation_context_snapshot)
        for item in records
        if type(item) is trace._CommittedConversationContextSnapshotQualificationObservationV1
    ]
    _require(bool(contexts), "committed conversation context is absent")
    users = [
        _user_commitment(key, "typed", message[1])
        for message in contexts[-1]["messages"]
        if message[0] == "user"
    ]
    _require(users == inputs, "committed conversation differs from accepted inputs")
    return {
        "arm": "capacity_rollover",
        "commands": spool.commands,
        "dispatch": dispatches[0].commands,
        "queue": queues[0].records,
        "spool_records": spool.records,
        "threads": thread_owners[0].finished(),
        "snapshots": [*spool.snapshots, continued],
        "transactions": spool.transactions,
        "transaction_writes": spool.transaction_writes,
        "connections": spool.connections,
        "durable_terminals": spool.durable_terminals,
        "rollover": [
            {"stage": item.stage.value, "result": item.result.value}
            for item in observations.records()
            if type(item) is RolloverObservationV1
        ],
        "source": {
            "consent": accepted_consent["consent"],
            "request": accepted_consent["request"],
            "binding": live_bindings,
            "user": users,
            "generated": [
                _commit(key, "generated", item.generated_text)
                for item in records
                if type(item) is trace._GeneratedTextQualificationObservationV1
            ],
            "transport": [
                _commit(key, "transport", item.confirmed_text)
                for item in records
                if type(item) is trace._TransportConfirmedChunkQualificationObservationV1
            ],
        },
        "terminals": [
            {
                "disposition": item.terminal_disposition.value,
                "reason": item.terminal_reason.value,
                "contextCommitted": item.context_committed,
            }
            for item in observations.records()
            if type(item) is TerminalSettledObservationV1
        ],
        "all_capacity_released": len([item for item in capacity if item.kind == "drain_terminal"])
        == 1
        and all(item.all_capacity_released for item in capacity if item.kind == "drain_terminal"),
        "ordinary_rejections": sum(item.kind == "ordinary_rejected" for item in capacity),
        "trace_complete": composition.trace.status().trace_complete
        and observations.status().trace_complete,
        "host_return": [
            item.outcome.value
            for item in records
            if type(item) is trace._HostReturnQualificationObservationV1
        ],
        "close": [
            {"stage": item.stage.value, "result": item.result.value}
            for item in observations.records()
            if type(item) is CloseStageObservationV1
        ],
    }
