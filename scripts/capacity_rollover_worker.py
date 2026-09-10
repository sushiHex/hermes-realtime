"""Observe real session rollover; raw SQLite and conversation bytes stay in the child."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any, cast

from scripts.equivalence_process import _require


def _commit(key: bytes, domain: str, value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hmac.new(key, domain.encode("ascii") + b"\0" + raw, hashlib.sha256).hexdigest()


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
    return {"consent": consent, "request": _request_commitment(key, sent["sequence"], fingerprint)}


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
        installation = connection.execute(
            "SELECT installation_id FROM producer_installation WHERE singleton=1"
        ).fetchone()[0]
        epochs = connection.execute(
            "SELECT consent_epoch_id,producer_instance_id,state FROM consent_epochs"
        ).fetchall()
        _require(
            len(epochs) == 1 and epochs[0][2] == "active",
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
            payloads = []
            for ordinal, event in enumerate(events, 1):
                event_id, sequence, kind, payload, payload_hash, stored_previous, digest, at, n = (
                    event
                )
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
                opened == events[0][7] == events[1][7],
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
                    "opened": _commit(key, "session_time", opened),
                    "expires": _commit(key, "session_time", expires),
                    "retention_lag_us": lag_us,
                    "predecessor": ""
                    if predecessor is None
                    else _commit(key, "session", predecessor),
                    "state": state,
                    "events": count,
                    "chain": [_commit(key, "chain", e[6]) for e in events],
                    "kinds": [e[2] for e in events],
                    "user": [
                        _commit(key, "user", p["text"])
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
    return {"sessions": sessions}


class _ObserveRolloverSpool:
    """Observe on the real SQLite owner thread and always delegate the operation."""

    def __init__(self, delegate: Any, database: Path, key: bytes) -> None:
        self.delegate, self.database, self.key = delegate, database, key
        self.snapshots: list[dict[str, Any]] = []
        self.transactions: list[str] = []
        self.failed = False
        self.durable_terminals: list[str] = []
        self.third_settled = Event()
        self.commands: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def create_epoch(self, command: Any) -> Any:
        _require(not self.commands, "create command was dispatched more than once")
        self.commands["create_dto"] = _command_commitment(self.key, command)
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
        result = self.delegate.append_record(item)
        if item.snapshot.event_kind.value == "turn_settled":
            _require(len(self.durable_terminals) < 3, "extra durable terminal")
            self.durable_terminals.append(result.value)
            if len(self.durable_terminals) == 3:
                self.third_settled.set()
        return result

    def rollover_session(self, command: Any) -> Any:
        _require(not self.snapshots, "rollover was invoked more than once")
        _require(
            set(self.commands) == {"create", "create_dto", "request"},
            "dispatched create command is absent",
        )
        self.commands["rollover_dto"] = _command_commitment(self.key, command)
        self.commands["rollover"] = _control_commitments(self.key, command.snapshots)
        self.commands["successor_expiry"] = _commit(
            self.key, "session_time", command.successor_expires_at_utc
        )
        self.snapshots.append(_snapshot(self.database, self.key))

        def trace(statement: str) -> None:
            # SQLite invokes this before executing COMMIT. A separate reader must
            # still see the entire old state. No statement text is retained.
            if statement not in {"BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"}:
                return
            try:
                _require(len(self.transactions) < 3, "rollover transaction trace overflow")
                self.transactions.append(statement)
                if statement == "COMMIT":
                    _require(len(self.snapshots) == 1, "rollover commit repeated")
                    self.snapshots.append(_snapshot(self.database, self.key))
            except Exception:
                # SQLite swallows callback errors; retain a content-free failure.
                self.failed = True

        self.delegate.connection.set_trace_callback(trace)
        try:
            result = self.delegate.rollover_session(command)
        finally:
            self.delegate.connection.set_trace_callback(None)
        _require(not self.failed, "rollover transaction observation failed")
        _require(
            _command_commitment(self.key, self.delegate._accepted_rollover)
            == self.commands["rollover_dto"],
            "accepted rollover differs from the complete dispatched command",
        )
        committed = _snapshot(self.database, self.key)
        self.snapshots.append(committed)
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
        self.commands: dict[str, Any] = {"ordinals": []}

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
        _require(set(self.commands) == {"ordinals"}, "create dispatch repeated")
        self.commands["create_dto"] = _command_commitment(self.key, command)
        self.commands["request"] = _request_commitment(
            self.key, command.control_sequence, command.control_fingerprint_hash
        )
        return self.delegate.create_epoch(command)

    def append_record(self, item: Any) -> Any:
        self._ordinal(item.admission_ordinal)
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

    def writer_factory(runtime: Any) -> Any:
        transport = runtime.create_sqlite_transport()
        factory = transport._spool_factory

        def observed_factory() -> _ObserveRolloverSpool:
            spool = _ObserveRolloverSpool(factory(), database, key)
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
        accepted_consent = await _accept_consent(port=port, origin=origin, token=token, key=key)
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
            inputs.append(_commit(key, "user", text))
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
            if running is not None:
                await composition.close_host(running)
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
        _commit(key, "user", message[1])
        for message in contexts[-1]["messages"]
        if message[0] == "user"
    ]
    _require(users == inputs, "committed conversation differs from accepted inputs")
    return {
        "arm": "capacity_rollover",
        "commands": spool.commands,
        "dispatch": dispatches[0].commands,
        "snapshots": [*spool.snapshots, continued],
        "transactions": spool.transactions,
        "durable_terminals": spool.durable_terminals,
        "rollover": [
            {"stage": item.stage.value, "result": item.result.value}
            for item in observations.records()
            if type(item) is RolloverObservationV1
        ],
        "source": {
            "consent": accepted_consent["consent"],
            "request": accepted_consent["request"],
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
