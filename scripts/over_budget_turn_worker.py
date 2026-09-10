"""Exercise real ordinary-queue overflow; raw source and SQLite remain in the child."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from scripts.equivalence_process import _require
from scripts.equivalence_worker import _observation
from scripts.evidence_observation import (
    _accept_consent,
    _command_commitment,
    _commit,
    _consent_callback_observation,
    _control_commitments,
    _live_browser_binding,
    _ObserveQueue,
    _ObserveThreadOwners,
    _payload_commitment,
    _request_commitment,
    _snapshot,
    _user_commitment,
)


class _ObserveCreateSpool:
    def __init__(
        self, delegate: Any, database: Path, key: bytes, owners: _ObserveThreadOwners
    ) -> None:
        self.delegate, self.database, self.key = delegate, database, key
        self.owners = owners
        self.commands: dict[str, Any] = {}
        self.records: list[dict[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.delegate, name)
        if not callable(value):
            return value

        def observed(*arguments: Any, **keywords: Any) -> Any:
            self.owners.spool_call(name)
            try:
                return value(*arguments, **keywords)
            except BaseException:
                self.owners.failed = True
                raise

        return observed

    def create_epoch(self, command: Any) -> Any:
        self.owners.spool_call("create_epoch")
        _require(not self.commands, "overflow create command repeated")
        self.commands = {
            "create_dto": _command_commitment(self.key, command),
            "binding": _commit(self.key, "browser_generation", str(command.binding_generation)),
            "request": _request_commitment(
                self.key, command.control_sequence, command.control_fingerprint_hash
            ),
            "create": _control_commitments(
                self.key, (command.session_opened, command.binding_opened)
            ),
        }
        result = self.delegate.create_epoch(command)
        _require(
            _command_commitment(self.key, self.delegate._accepted_create)
            == self.commands["create_dto"],
            "accepted create differs from the dispatched command",
        )
        opening = _snapshot(self.database, self.key)
        _require(
            len(opening["sessions"]) == 1
            and opening["sessions"][0]["controls"] == self.commands["create"],
            "stored opening differs from the dispatched command",
        )
        return result

    def append_record(self, item: Any) -> Any:
        self.owners.spool_call("append_record")
        _require(len(self.records) < 59, "overflow durable prefix is oversized")
        dto = _payload_commitment(self.key, item)
        snapshot = _control_commitments(self.key, (item.snapshot,))[0]
        result = self.delegate.append_record(item)
        self.records.append(dict(dto=dto, snapshot=snapshot, result=result.value))
        return result


class _ObserveDispatch:
    def __init__(self, delegate: Any, key: bytes) -> None:
        self.delegate, self.key = delegate, key
        self.commands: dict[str, Any] = {"ordinals": [], "records": []}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def create_epoch(self, command: Any) -> Any:
        _require(set(self.commands) == {"ordinals", "records"}, "overflow create dispatch repeated")
        self.commands["create_dto"] = _command_commitment(self.key, command)
        self.commands["request"] = _request_commitment(
            self.key, command.control_sequence, command.control_fingerprint_hash
        )
        return self.delegate.create_epoch(command)

    def append_record(self, item: Any) -> Any:
        ordinals = self.commands["ordinals"]
        _require(
            type(item.admission_ordinal) is int
            and len(ordinals) < 59
            and item.admission_ordinal == len(ordinals) + 3,
            "overflow dispatch lost the real admission order",
        )
        ordinals.append(item.admission_ordinal)
        self.commands["records"].append(_payload_commitment(self.key, item))
        return self.delegate.append_record(item)


def _scan_rejected_source(
    database: Path, generated: list[bytes], users: list[str], key: bytes
) -> dict[str, Any]:
    from hermes_realtime.evidence.storage_security import MANIFEST_V1
    from scripts.qualify_evidence_slice_zero import _is_reparse_or_link

    values = [("generated", raw) for raw in generated[57:]] + [
        ("user", raw.encode("utf-8")) for raw in users[1:]
    ]
    _require(
        len(values) == 25
        and len({raw for _, raw in values}) == 25
        and all(raw and raw.isascii() for _, raw in values),
        "rejected synthetic source set differs",
    )
    count, size = 0, 0
    matches: list[str] = []

    def commitment(domain: str, value: bytes) -> str:
        return (
            _user_commitment(key, "typed", value.decode("utf-8"))
            if domain == "user"
            else _commit(key, domain, value)
        )

    _require(database.is_file(), "closed evidence database is absent")
    paths = sorted(database.parent.iterdir())
    _require(1 <= len(paths) <= len(MANIFEST_V1.all_names), "closed evidence file count differs")
    for path in paths:
        _require(path.name in MANIFEST_V1.all_names, "unlisted evidence file is present")
        _require(not _is_reparse_or_link(path) and path.is_file(), "evidence file is indirect")
        length = path.stat().st_size
        _require(size + length <= 64 * 1024 * 1024, "evidence scan exceeds its byte bound")
        raw = path.read_bytes()
        _require(len(raw) == length, "closed evidence file changed during scan")
        size += length
        count += 1
        matches.extend(commitment(domain, value) for domain, value in values if value in raw)
    return {
        "generated": [_commit(key, "generated", raw) for raw in generated[57:]],
        "user": [_user_commitment(key, "typed", raw) for raw in users[1:]],
        "matches": matches,
        "files": count,
        "bytes": size,
    }


async def _arm(workspace: Path, url: str, capture: bool, key: bytes) -> dict[str, Any]:
    from livekit import rtc

    from hermes_realtime._qualification import _QualificationCapacityObservationV1
    from hermes_realtime.host_launcher import build_local_host_launcher
    from hermes_realtime.speech.types import AudioFrame, SpeechChunk
    from tests.integration import test_qualification_full_host_ingress as ingress
    from tests.support import qualification as trace

    class Inference:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, snapshot: Any, *, turn_id: str) -> AsyncIterator[str]:
            if turn_id == "turn_preflight":
                yield "Ready."
                return
            self.turns += 1
            if self.turns == 1:
                for ordinal in range(80):
                    yield f"Synthetic bounded admission segment {ordinal:02d}."
            else:
                yield "Synthetic continuation response."

        async def cancel(self, turn_id: str) -> None:
            del turn_id

        async def close(self) -> None:
            return None

    class Synthesizer:
        def __init__(self) -> None:
            self.ordinal = 0

        async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
            self.ordinal += 1
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id=f"synthetic-budget-{self.ordinal}",
                text=text,
                audio=AudioFrame(pcm=ingress._pcm(), sample_rate_hz=48_000, channels=1),
            )

        async def cancel(self, turn_id: str) -> None:
            del turn_id

        async def close(self) -> None:
            return None

    suffix, port = uuid.uuid4().hex[:10], ingress._available_port()
    database = workspace / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition = trace.InProcessQualificationComposition()
    transcriber = ingress._Transcriber()
    spools: list[_ObserveCreateSpool] = []
    thread_owners: list[_ObserveThreadOwners] = []
    dispatches: list[_ObserveDispatch] = []
    barriers: list[Any] = []
    queues: list[Any] = []
    reservations: list[tuple[Any, Any]] = []

    consent_callbacks: list[dict[str, str]] = []

    def writer_factory(runtime: Any) -> Any:
        _require(not consent_callbacks, "overflow consent callback repeated")
        consent_callbacks.append(_consent_callback_observation(runtime, key))
        reserve = runtime.reserve_browser_consent

        def observe_reservation(**arguments: Any) -> Any:
            authority = reserve(**arguments)
            _require(not queues, "overflow queue reservation repeated")
            queues.append(
                _ObserveQueue(
                    runtime._queue,
                    key,
                    owner_generation=runtime._owner_generation,
                    scenario="over_budget_turn",
                    owners=owners,
                )
            )
            return authority

        runtime.reserve_browser_consent = observe_reservation
        reservations.append((runtime, reserve))
        transport = runtime.create_sqlite_transport()
        owners = _ObserveThreadOwners(transport, key, scenario="over_budget_turn")
        thread_owners.append(owners)
        factory = transport._spool_factory

        def observed_factory() -> _ObserveCreateSpool:
            from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

            owners.spool_call("factory")
            original = factory()
            _require(type(original) is SQLiteEvidenceSpool, "overflow spool is not exact")
            spool = _ObserveCreateSpool(original, database, key, owners)
            spools.append(spool)
            return spool

        transport._spool_factory = observed_factory
        barrier = ingress._BlockFirstWriterRecordTransport(transport)
        barriers.append(barrier)
        dispatch = _ObserveDispatch(barrier, key)
        dispatches.append(dispatch)
        return dispatch

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            livekit_url=url,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=capture,
            evidence_database=database if capture else None,
        ),
        inference_factory=Inference,
        speech_presence_factory=ingress._Presence,
        synthesizer_factory=Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: ingress._Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory if capture else None,
    )
    room, running = rtc.Room(), None
    users: list[str] = []
    completions: list[int] = []
    snapshots: list[dict[str, Any]] = []
    consent: dict[str, Any] = {}
    try:
        running = await composition.start_host(registration)
        origin, token, _ = await ingress._connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        if capture:
            live_binding = await _live_browser_binding(running._host, key)
            accepted = await _accept_consent(port=port, origin=origin, token=token, key=key)
            live_bindings = [live_binding, await _live_browser_binding(running._host, key)]
            _require(
                live_bindings[0] == live_bindings[1], "overflow live binding changed during consent"
            )
            consent = {
                **accepted,
                "binding": live_bindings,
                "consent_callback": consent_callbacks[0],
            }
            _require(len(thread_owners) == len(reservations) == 1, "overflow owner is ambiguous")
            runtime = reservations[0][0]
            thread_owners[0].bind_dispatcher(
                runtime._writer,
                queue=queues[0].queue,
                admission=runtime._admission,
                transport=dispatches[0],
            )
        after = 0
        for sequence in (1, 2):
            text = f"Synthetic budget turn {sequence}."
            status, result = await ingress._request(
                port=port,
                origin=origin,
                path="/api/v1/input",
                bearer=token,
                body=json.dumps({"sequence": sequence, "text": text}).encode("utf-8"),
            )
            _require(
                status == 202 and result == {"sequence": sequence, "version": 1},
                "overflow conversation input was not accepted",
            )
            users.append(text)
            event = await ingress._wait_event(
                port=port, origin=origin, token=token, kind="assistant_turn_completed", after=after
            )
            _require(type(event["sequence"]) is int, "overflow completion sequence differs")
            after = cast(int, event["sequence"])
            completions.append(sequence)
        if capture:
            _require(
                len(spools) == len(dispatches) == len(barriers) == 1
                and barriers[0].entered.is_set()
                and not barriers[0].release.is_set()
                and not spools[0].records,
                "ordinary writer completions were not held during the conversation",
            )
            barriers[0].release.set()
            async with asyncio.timeout(20):
                while (
                    sum(
                        item.kind == "ordinary_completed"
                        for item in composition.capacity_observations(running)
                    )
                    < 59
                ):
                    await asyncio.sleep(0.01)
            snapshots.append(_snapshot(database, key, max_events=64))
    finally:
        for barrier in barriers:
            barrier.release.set()
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
    _require(running is not None, "overflow host did not start")
    records = composition.trace.records()
    metadata = composition.production_observations(running)
    result = _observation(
        "consented" if capture else "disabled",
        records,
        metadata.records(),
        key,
        composition.trace.status().trace_complete and metadata.status().trace_complete,
    )
    generated = [
        item.generated_text
        for item in records
        if type(item) is trace._GeneratedTextQualificationObservationV1
    ]
    # Use the store's content domains so parent comparisons are independently linked.
    for observed, original in zip(result["records"], records, strict=True):
        if type(original) is trace._GeneratedTextQualificationObservationV1:
            observed["value"] = _commit(key, "generated", original.generated_text)
        elif type(original) is trace._TransportConfirmedChunkQualificationObservationV1:
            observed["value"] = _commit(key, "transport", original.confirmed_text)
    contexts = [
        json.loads(item.committed_conversation_context_snapshot)
        for item in records
        if type(item) is trace._CommittedConversationContextSnapshotQualificationObservationV1
    ]
    # The production context intentionally retains only a bounded window. Bind
    # each accepted input to its own inference snapshot before later segments
    # evict earlier messages from that window.
    context_users = [
        [message[1] for message in context["messages"] if message[0] == "user"]
        for context in contexts
    ]
    _require(
        len(context_users) == len(users) == 2
        and all(messages for messages in context_users)
        and [messages[-1] for messages in context_users] == users,
        "overflow committed context differs from accepted source input",
    )
    result["user"] = [_user_commitment(key, "typed", text) for text in users]
    result["completed_inputs"] = completions
    if capture:
        raw_capacity = composition.capacity_observations(running)
        _require(
            all(type(item) is _QualificationCapacityObservationV1 for item in raw_capacity),
            "overflow capacity observations are not production-owned",
        )
        capacity = cast(tuple[_QualificationCapacityObservationV1, ...], raw_capacity)
        snapshots.append(_snapshot(database, key, max_events=64))
        result.update(
            consent=consent,
            threads=thread_owners[0].finished(),
            queue=queues[0].records,
            spool_records=spools[0].records,
            snapshots=snapshots,
            commands=spools[0].commands,
            dispatch=dispatches[0].commands,
            scan=_scan_rejected_source(database, generated, users, key),
            capacity=[
                {
                    "kind": item.kind,
                    "source": item.rejection_source,
                    "records": item.queue_record_count,
                    "bytes": item.queue_canonical_bytes,
                    "physical": item.queue_physical_count,
                    "released": item.all_capacity_released,
                }
                for item in capacity
            ],
        )
    return result


async def observe_over_budget_turn(workspace: Path, livekit_url: str) -> dict[str, Any]:
    key = os.urandom(32)
    baseline = await _arm(workspace, livekit_url, False, key)
    captured = await _arm(workspace, livekit_url, True, key)
    return {"arm": "over_budget_turn", "baseline": baseline, "captured": captured}
