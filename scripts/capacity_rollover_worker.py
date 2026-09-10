"""Observe real session rollover; raw SQLite and conversation bytes stay in the child."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from pathlib import Path
from threading import Event
from typing import Any, cast

from scripts.equivalence_process import _require
from scripts.evidence_observation import (
    _accept_consent as _accept_consent,
)
from scripts.evidence_observation import (
    _command_commitment as _command_commitment,
)
from scripts.evidence_observation import (
    _commit as _commit,
)
from scripts.evidence_observation import (
    _consent_callback_observation as _consent_callback_observation,
)
from scripts.evidence_observation import (
    _consent_commitment as _consent_commitment,
)
from scripts.evidence_observation import (
    _control_commitments as _control_commitments,
)
from scripts.evidence_observation import (
    _data_version as _data_version,
)
from scripts.evidence_observation import (
    _live_browser_binding as _live_browser_binding,
)
from scripts.evidence_observation import (
    _ObserveConnectionAudit as _ObserveConnectionAudit,
)
from scripts.evidence_observation import (
    _ObserveQueue as _ObserveQueue,
)
from scripts.evidence_observation import (
    _ObserveThreadOwners as _ObserveThreadOwners,
)
from scripts.evidence_observation import (
    _payload_commitment as _payload_commitment,
)
from scripts.evidence_observation import (
    _request_commitment as _request_commitment,
)
from scripts.evidence_observation import (
    _snapshot as _snapshot,
)
from scripts.evidence_observation import (
    _statement_kind as _statement_kind,
)
from scripts.evidence_observation import (
    _user_commitment as _user_commitment,
)


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

    consent_callbacks: list[dict[str, str]] = []

    def writer_factory(runtime: Any) -> Any:
        _require(not consent_callbacks, "consent callback was invoked more than once")
        consent_callbacks.append(_consent_callback_observation(runtime, key))
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
            "consent_callback": consent_callbacks[0],
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
