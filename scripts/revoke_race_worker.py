"""Drive the packaged host's revocation race; keep all raw store data in the child."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from threading import Event
from typing import Any, cast

from scripts.equivalence_process import _require


class _HoldFinalization:
    """Pause one writer operation before delegating to the real SQLite transport."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.entered = Event()
        self.release = Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def finalize_revoke(self, command: Any) -> Any:
        self.entered.set()
        _require(self.release.wait(10), "revocation finalization barrier was not released")
        return self.delegate.finalize_revoke(command)


def _snapshot(database: Path, key: bytes) -> dict[str, Any]:
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")

        def scalar(query: str) -> int:
            return int(connection.execute(query).fetchone()[0])

        authority = connection.execute(
            "SELECT erasure_request_id, scope_kind, scope_key, reason_code, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal FROM erasure_requests"
            " UNION ALL SELECT erasure_request_id, scope_kind, scope_key, reason_code,"
            " control_sequence, control_fingerprint_hash, last_admission_ordinal"
            " FROM erasure_tombstones ORDER BY erasure_request_id"
        ).fetchall()
        result = {
            "events": scalar("SELECT COUNT(*) FROM evidence_events"),
            "sessions": scalar("SELECT COUNT(*) FROM evidence_sessions"),
            "epochs": scalar("SELECT COUNT(*) FROM consent_epochs"),
            "requests": scalar("SELECT COUNT(*) FROM erasure_requests"),
            "pending_revocations": scalar(
                "SELECT COUNT(*) FROM erasure_requests r JOIN consent_epochs e"
                " ON r.scope_key=e.consent_epoch_id WHERE r.state='pending'"
                " AND r.scope_kind='consent_epoch' AND r.reason_code='revoked'"
                " AND r.control_sequence=2 AND e.state='revoked'"
            ),
            "revoked_epochs": scalar("SELECT COUNT(*) FROM consent_epochs WHERE state='revoked'"),
            "tombstones": scalar("SELECT COUNT(*) FROM erasure_tombstones"),
            "erased_events": scalar(
                "SELECT COALESCE(SUM(erased_event_count),0) FROM erasure_tombstones"
            ),
            "erased_sessions": scalar(
                "SELECT COALESCE(SUM(erased_session_count),0) FROM erasure_tombstones"
            ),
            "last_ordinal": sum(row[6] for row in authority),
            "final_ordinal": scalar(
                "SELECT COALESCE(SUM(final_admission_ordinal),0) FROM"
                " (SELECT final_admission_ordinal FROM erasure_requests UNION ALL"
                " SELECT final_admission_ordinal FROM erasure_tombstones)"
            ),
            "integrity_ok": connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
            "scope_matches": len(authority) == 1
            and all(
                (row[1], row[3], row[4]) == ("consent_epoch", "revoked", 2) for row in authority
            ),
            # No request, epoch, session, or binding identifier leaves this child.
            "authority": hmac.new(
                key, json.dumps(authority, separators=(",", ":")).encode(), hashlib.sha256
            ).hexdigest(),
        }
    return result


async def observe_revoke_race(workspace: Path, livekit_url: str) -> dict[str, Any]:
    from livekit import rtc

    from hermes_realtime._qualification import _QualificationCapacityObservationV1
    from hermes_realtime.host_launcher import build_local_host_launcher
    from hermes_realtime.production_observation import (
        CloseStageObservationV1,
        TerminalSettledObservationV1,
    )
    from tests.integration.test_qualification_full_host_ingress import (
        _available_port,
        _connect_and_activate,
        _consent_capture,
        _Inference,
        _Presence,
        _request,
        _Synthesizer,
        _Transcriber,
        _Vad,
        _wait_event,
    )
    from tests.support.qualification import (
        InProcessQualificationComposition,
        _HostReturnQualificationObservationV1,
    )

    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database = workspace / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition = InProcessQualificationComposition()
    transcriber = _Transcriber()
    transports: list[_HoldFinalization] = []
    key = os.urandom(32)

    def writer_factory(runtime: Any) -> _HoldFinalization:
        transport = _HoldFinalization(runtime.create_sqlite_transport())
        transports.append(transport)
        return transport

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
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory,
    )
    room, running = rtc.Room(), None
    snapshots: list[dict[str, Any]] = []

    def capacity_view() -> tuple[_QualificationCapacityObservationV1, ...]:
        assert running is not None
        records = composition.capacity_observations(running)
        _require(
            all(type(item) is _QualificationCapacityObservationV1 for item in records),
            "capacity observations are not production-owned",
        )
        return cast(tuple[_QualificationCapacityObservationV1, ...], records)

    try:
        running = await composition.start_host(registration)
        origin, token, _source = await _connect_and_activate(
            port=port,
            launch_url=running.url,
            room=room,
        )
        await _consent_capture(port=port, origin=origin, token=token)
        status, result = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"synthetic consented turn"}',
        )
        _require(
            status == 202 and result == {"sequence": 1, "version": 1},
            "consented input was not accepted",
        )
        first = await _wait_event(
            port=port, origin=origin, token=token, kind="assistant_turn_completed"
        )
        status, result = await _request(
            port=port,
            origin=origin,
            path="/api/v1/evidence-revoke",
            bearer=token,
            body=b'{"sequence":2}',
        )
        _require(
            status == 202
            and result
            == {
                "sequence": 2,
                "result": "revoke_durably_scheduled",
                "captureState": "revoked_purging",
            },
            "revocation was not durably acknowledged",
        )
        _require(len(transports) == 1, "revocation writer ownership is ambiguous")
        transport = transports[0]
        async with asyncio.timeout(5):
            while not transport.entered.is_set():
                await asyncio.sleep(0.01)
        snapshots.append(_snapshot(database, key))
        admitted_before = sum(item.kind == "ordinary_admitted" for item in capacity_view())
        status, result = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":2,"text":"synthetic post-revoke turn"}',
        )
        _require(
            status == 202 and result == {"sequence": 2, "version": 1},
            "ordinary conversation stopped after revocation",
        )
        first_sequence = first["sequence"]
        _require(type(first_sequence) is int, "completion event sequence is invalid")
        assert type(first_sequence) is int
        await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="assistant_turn_completed",
            after=first_sequence,
        )
        snapshots.append(_snapshot(database, key))
        admitted_after = sum(item.kind == "ordinary_admitted" for item in capacity_view())
        transport.release.set()
        async with asyncio.timeout(5):
            while not any(
                item.kind == "revoke_terminal" and item.all_capacity_released
                for item in capacity_view()
            ):
                await asyncio.sleep(0.01)
        snapshots.append(_snapshot(database, key))
        capacity = capacity_view()
    finally:
        for transport in transports:
            transport.release.set()
        try:
            await room.disconnect()
        finally:
            if running is not None:
                await composition.close_host(running)
    _require(running is not None, "revocation host was not started")
    observations = composition.production_observations(running)
    records = composition.trace.records()
    outcomes = [
        item.outcome.value
        for item in records
        if type(item) is _HostReturnQualificationObservationV1
    ]
    _require(outcomes == ["returned"], "revocation host did not return cleanly")
    return {
        "arm": "revoke_race",
        "snapshots": snapshots,
        "acknowledgment": "durable",
        "completed_turns": sum(item.kind.value == "generated_text" for item in records),
        "capture_terminals": sum(
            type(item) is TerminalSettledObservationV1 for item in observations.records()
        ),
        "admitted_before": admitted_before,
        "admitted_after": admitted_after,
        "revoke_accepted": sum(item.kind == "revoke_accepted" for item in capacity),
        "revoke_terminal": sum(item.kind == "revoke_terminal" for item in capacity),
        "all_capacity_released": all(
            item.all_capacity_released for item in capacity if item.kind == "revoke_terminal"
        ),
        "trace_complete": composition.trace.status().trace_complete
        and observations.status().trace_complete,
        "host_return": outcomes[0],
        "close": [
            {"stage": item.stage.value, "result": item.result.value}
            for item in observations.records()
            if type(item) is CloseStageObservationV1
        ],
    }
