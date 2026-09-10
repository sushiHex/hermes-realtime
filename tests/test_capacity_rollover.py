"""Rollover acceptance requires durable lineage, source equality, and owned close."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest


def _observations() -> dict:
    from scripts.deterministic_equivalence import _expected_close

    turn = [
        "turn_opened",
        "user_final_accepted",
        "assistant_segment_generated",
        "assistant_chunk_transport_confirmed_full",
        "turn_snapshot",
        "turn_settled",
    ]
    opening = ["session_opened", "binding_opened"]
    source = {
        name: [f"{n:064x}" for n in range(30, 33)] for name in ("user", "generated", "transport")
    }

    def session(identity: str, turns: int, sealed: bool, predecessor: str = "") -> dict:
        kinds = (
            opening
            + turn * turns
            + (["binding_closed", "session_seal_requested"] if sealed else [])
        )
        return dict(
            session=identity,
            epoch="e" * 64,
            consent="c" * 64,
            consent_request="f" * 64,
            controls=(
                ["3" * 64, "4" * 64]
                if predecessor
                else ["0" * 64, "1" * 64] + (["2" * 64, "2" * 64] if sealed else [])
            ),
            records=[
                f"{n:064x}"
                for n in range(
                    212 if predecessor else 200, (212 if predecessor else 200) + 6 * turns
                )
            ],
            opened="8" * 64 if predecessor else "7" * 64,
            expires="9" * 64 if predecessor else "6" * 64,
            retention_lag_us=1000 if predecessor else 0,
            last_event_at=("3" if turns else "2") * 64
            if predecessor
            else ("2" if sealed else "1") * 64,
            predecessor=predecessor,
            state="sealed" if sealed else "open",
            events=len(kinds),
            kinds=kinds,
            chain=[f"{n:064x}" for n in range(len(kinds))],
            **{k: v[2:3] if predecessor and turns else v[:turns] for k, v in source.items()},
        )

    before = {"sessions": [session("a" * 64, 2, False)]}
    committed = {"sessions": [session("a" * 64, 2, True), session("b" * 64, 0, False, "a" * 64)]}
    continued = {"sessions": [session("a" * 64, 2, True), session("b" * 64, 1, False, "a" * 64)]}
    for snapshot in (before, committed, continued):
        snapshot.update(clock=snapshot["sessions"][-1]["last_event_at"], authority="4" * 64)
    return dict(
        arm="capacity_rollover",
        commands={
            "create": ["0" * 64, "1" * 64],
            "rollover": ["2" * 64, "2" * 64, "3" * 64, "4" * 64],
            "successor_expiry": "9" * 64,
            "create_dto": "1" * 64,
            "rollover_dto": "2" * 64,
            "request": "3" * 64,
        },
        snapshots=[before, copy.deepcopy(before), committed, continued],
        transactions=["BEGIN IMMEDIATE", "COMMIT"],
        durable_terminals=["committed"] * 3,
        rollover=[
            {
                "stage": stage,
                "result": "accepted" if stage in {"claimed", "queued"} else "committed",
            }
            for stage in ("claimed", "queued", "durable", "published", "terminal")
        ],
        source=source | {"consent": "f" * 64, "request": "3" * 64},
        dispatch={
            "create_dto": "1" * 64,
            "rollover_dto": "2" * 64,
            "request": "3" * 64,
            "ordinals": list(range(3, 22)),
            "rollover_ordinal": 15,
            "records": [f"{n:064x}" for n in range(100, 118)],
        },
        spool_records=[
            dict(dto=f"{n:064x}", snapshot=f"{n + 100:064x}", result="committed")
            for n in range(100, 118)
        ],
        queue=[
            dict(
                version=1,
                ordinal=n + 2,
                lane="drain" if n == 20 else "ordered",
                kind="create"
                if n == 0
                else "rollover"
                if n == 13
                else "drain"
                if n == 20
                else "record",
                payload=payload,
            )
            for n, payload in enumerate(
                [
                    "1" * 64,
                    *[f"{n:064x}" for n in range(100, 112)],
                    "2" * 64,
                    *[f"{n:064x}" for n in range(112, 118)],
                    "4" * 64,
                ]
            )
        ],
        terminals=[
            dict(
                disposition="completed",
                reason="authoritative_close_completed",
                contextCommitted=True,
            )
            for _ in range(3)
        ],
        all_capacity_released=True,
        ordinary_rejections=0,
        trace_complete=True,
        host_return=["returned"],
        close=_expected_close("consented"),
    )


def test_rollover_validator_accepts_a_durable_transition_and_equal_source() -> None:
    from scripts.capacity_rollover import _validate_observations

    _validate_observations(_observations())


@pytest.mark.parametrize("stored_source", ["typed", "microphone"])
def test_persisted_source_provenance_must_match_the_accepted_typed_input(
    tmp_path: Path, stored_source: str
) -> None:
    from dataclasses import replace

    from hermes_realtime.evidence.models import InputSource, StoreDisposition
    from scripts.capacity_rollover import _validate_observations
    from scripts.capacity_rollover_worker import _snapshot
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_spool,
        ordinary_record,
        turn_opened_snapshot,
        user_final_snapshot,
    )

    commitments = {}
    for source in ("typed", "microphone"):
        owned = make_spool(tmp_path / source)
        try:
            command = make_create_epoch()
            command = replace(
                command,
                microphone_accepted=True,
                session_opened=replace(
                    command.session_opened,
                    payload=replace(command.session_opened.payload, microphone_accepted=True),
                ),
                binding_opened=replace(
                    command.binding_opened,
                    payload=replace(command.binding_opened.payload, microphone_available=True),
                ),
            )
            assert owned.create_epoch(command) is StoreDisposition.COMMITTED
            opened = turn_opened_snapshot(3, event_index=3)
            user = user_final_snapshot(4, event_index=4, text="Synthetic accepted typed input.")
            user = replace(user, payload=replace(user.payload, source=InputSource(source)))
            for snapshot in (opened, user):
                assert (
                    owned.append_record(ordinary_record(snapshot, snapshot.event_sequence))
                    is StoreDisposition.COMMITTED
                )
            # Both histories are valid, consented and hashed by the real spool.
            # Their text is identical; only the persisted provenance differs.
            observed = _snapshot(owned.database, b"synthetic key")
            commitments[source] = observed["sessions"][0]["user"][0]
        finally:
            owned.close()
    row = _observations()
    row["source"]["user"] = [commitments["typed"]] * 3
    for snapshot in row["snapshots"]:
        for session in snapshot["sessions"]:
            session["user"] = [commitments[stored_source]] * len(session["user"])
    if stored_source == "typed":
        _validate_observations(row)
    else:
        with pytest.raises(ValueError, match="persisted content"):
            _validate_observations(row)


def test_coordinated_store_consent_changes_cannot_replace_the_accepted_request() -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    for snapshot in row["snapshots"]:
        for session in snapshot["sessions"]:
            session["consent"] = "d" * 64
            session["consent_request"] = "d" * 64
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledgment", ["accepted", "pending", "bool_sequence"])
async def test_consent_source_commitment_requires_the_exact_dispatched_request_ack(
    monkeypatch: pytest.MonkeyPatch,
    acknowledgment: str,
) -> None:
    import hashlib
    import hmac
    import json

    from scripts.capacity_rollover_worker import _accept_consent
    from tests.integration import test_qualification_full_host_ingress as ingress

    requests = []

    async def event(**kwargs):
        assert kwargs["kind"] == "capture_status"
        return {"data": {"disclosureDigest": "b" * 64}}

    async def request(**kwargs):
        assert kwargs["path"] == "/api/v1/evidence-consent"
        requests.append(json.loads(kwargs["body"]))
        return (202 if acknowledgment == "pending" else 200), {
            "captureState": "active",
            "result": "consent_activated",
            "sequence": True if acknowledgment == "bool_sequence" else 1,
        }

    monkeypatch.setattr(ingress, "_wait_event", event)
    monkeypatch.setattr(ingress, "_request", request)
    arguments = dict(
        port=1, origin="https://example.invalid", token="synthetic", key=b"synthetic key"
    )
    if acknowledgment != "accepted":
        with pytest.raises(ValueError, match="not accepted"):
            await _accept_consent(**arguments)
        return
    commitment = await _accept_consent(**arguments)
    assert len(requests) == 1
    sent = requests[0]
    assert sent == {
        "accepted": True,
        "sequence": 1,
        "consentVersion": "realtime-evidence-consent-v1",
        "disclosureDigest": "b" * 64,
        "retentionHours": 24,
        "sources": {"microphone": True, "typed": True},
    }
    normalized = {
        "consent_version": sent["consentVersion"],
        "disclosure_digest": sent["disclosureDigest"],
        "retention_hours": sent["retentionHours"],
        "microphone_accepted": sent["sources"]["microphone"],
        "typed_accepted": sent["sources"]["typed"],
    }
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    assert (
        commitment["consent"]
        == hmac.new(b"synthetic key", b"accepted_consent\0" + raw, hashlib.sha256).hexdigest()
    )
    fingerprint = hashlib.sha256(
        json.dumps(sent, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert (
        commitment["request"]
        == hmac.new(
            b"synthetic key",
            b"accepted_request\0"
            + json.dumps([sent["sequence"], fingerprint], separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_snapshot",
        "partial_visible",
        "separate_commits",
        "wrong_predecessor",
        "wrong_epoch",
        "wrong_consent",
        "same_session",
        "old_session_changed",
        "old_chain_changed",
        "successor_chain_changed",
        "missing_successor_turn",
        "missing_source",
        "wrong_source",
        "truncated_chain",
        "extra_content",
        "bool_count",
        "missing_rollover",
        "failed_rollover",
        "not_durable",
        "false_terminal",
        "queue_saturation",
        "leaked_capacity",
        "incomplete_trace",
        "failed_return",
        "missing_close",
    ],
)
def test_rollover_validator_rejects_incomplete_or_contradictory_facts(mutation: str) -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    snapshots = row["snapshots"]
    if mutation == "missing_snapshot":
        snapshots.pop()
    elif mutation == "partial_visible":
        snapshots[1] = copy.deepcopy(snapshots[2])
    elif mutation == "separate_commits":
        row["transactions"] *= 2
    elif mutation == "wrong_predecessor":
        snapshots[2]["sessions"][1]["predecessor"] = "c" * 64
    elif mutation == "wrong_epoch":
        snapshots[2]["sessions"][1]["epoch"] = "c" * 64
    elif mutation == "wrong_consent":
        snapshots[2]["sessions"][1]["consent"] = "d" * 64
    elif mutation == "same_session":
        snapshots[2]["sessions"][1]["session"] = "a" * 64
    elif mutation == "old_session_changed":
        snapshots[3]["sessions"][0]["generated"][0] = "c" * 64
    elif mutation == "old_chain_changed":
        snapshots[2]["sessions"][0]["chain"][0] = "c" * 64
    elif mutation == "successor_chain_changed":
        snapshots[3]["sessions"][1]["chain"][0] = "c" * 64
    elif mutation == "missing_successor_turn":
        snapshots[3] = copy.deepcopy(snapshots[2])
    elif mutation == "missing_source":
        row["source"]["generated"].pop()
    elif mutation == "wrong_source":
        row["source"]["transport"][2] = "c" * 64
    elif mutation == "truncated_chain":
        snapshots[0]["sessions"][0]["chain"].pop()
    elif mutation == "extra_content":
        snapshots[3]["sessions"][1]["text"] = "synthetic private marker"
    elif mutation == "bool_count":
        snapshots[0]["sessions"][0]["events"] = True
    elif mutation == "missing_rollover":
        row["rollover"].pop()
    elif mutation == "failed_rollover":
        row["rollover"][2]["result"] = "rejected"
    elif mutation == "not_durable":
        row["durable_terminals"][2] = "faulted"
    elif mutation == "false_terminal":
        row["terminals"][2]["contextCommitted"] = 1
    elif mutation == "queue_saturation":
        row["ordinary_rejections"] = 1
    elif mutation == "leaked_capacity":
        row["all_capacity_released"] = False
    elif mutation == "incomplete_trace":
        row["trace_complete"] = False
    elif mutation == "failed_return":
        row["host_return"] = ["failed"]
    elif mutation == "missing_close":
        row["close"].pop()
    with pytest.raises(ValueError):
        _validate_observations(row)


def test_rollover_receipts_and_registration_reject_supplied_success() -> None:
    from scripts import qualify_evidence_slice_zero as core
    from scripts.capacity_rollover import ObservedCapacityRolloverV1, validate_capacity_rollover_v1

    with pytest.raises(TypeError):
        ObservedCapacityRolloverV1()
    with pytest.raises((TypeError, ValueError)):
        validate_capacity_rollover_v1({"passed": True})
    with pytest.raises(ValueError):
        validate_capacity_rollover_v1(object.__new__(ObservedCapacityRolloverV1))
    assert tuple(item.scenario_id for item in core.SCENARIO_REGISTRY_V1) == tuple(core.ScenarioIdV1)
    assert core.SCENARIO_REGISTRY_V1[11] is core.CAPACITY_ROLLOVER_REGISTRATION_V1
    with pytest.raises(ValueError, match="not canonical"):
        core.CapacityRolloverRegistrationV1().produce(
            object(),
            object(),
            object(),
            livekit_executable=Path("unusable.exe"),
            livekit_sha256="a" * 64,
        )


def test_rollover_observer_reads_the_real_transaction_without_exposing_partial_state(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence.models import StoreDisposition
    from scripts.capacity_rollover_worker import _ObserveRolloverSpool
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)
    try:
        observer = _ObserveRolloverSpool(owned, owned.database, b"synthetic key")
        assert observer.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert observer.rollover_session(make_rollover_command()) is StoreDisposition.COMMITTED
        assert observer.transactions == ["BEGIN IMMEDIATE", "COMMIT"]
        before, committing, committed = observer.snapshots
        assert before == committing
        assert [(s["state"], s["events"]) for s in before["sessions"]] == [("open", 2)]
        assert [(s["state"], s["events"]) for s in committed["sessions"]] == [
            ("sealed", 4),
            ("open", 2),
        ]
        assert committed["sessions"][1]["predecessor"] == committed["sessions"][0]["session"]
    finally:
        owned.close()


@pytest.mark.parametrize("substitution", ["none", "transport", "spool"])
def test_complete_ordinary_record_is_bound_from_dispatch_through_sqlite(
    tmp_path: Path, substitution: str
) -> None:
    from dataclasses import replace

    from hermes_realtime.evidence.models import StoreDisposition
    from scripts.capacity_rollover import _validate_observations
    from scripts.capacity_rollover_worker import (
        _ObserveRolloverSpool,
        _payload_commitment,
        _snapshot,
    )
    from tests.evidence.test_sqlite_spool import (
        event_uuid,
        make_create_epoch,
        make_spool,
        ordinary_record,
        turn_opened_snapshot,
    )

    key = b"synthetic key"
    owned = make_spool(tmp_path)
    sent = ordinary_record(turn_opened_snapshot(3, event_index=3), 3)
    changed = replace(sent, snapshot=replace(sent.snapshot, event_id=event_uuid(99)))

    class SubstitutingSpool:
        def __getattr__(self, name):
            return getattr(owned, name)

        def append_record(self, item):
            return owned.append_record(changed if substitution == "spool" else item)

    observer = _ObserveRolloverSpool(SubstitutingSpool(), owned.database, key)
    try:
        assert observer.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            observer.append_record(changed if substitution == "transport" else sent)
            is StoreDisposition.COMMITTED
        )
        stored = _snapshot(owned.database, key)["sessions"][0]
        row = _observations()
        row["dispatch"]["records"][0] = _payload_commitment(key, sent)
        row["queue"][1]["payload"] = row["dispatch"]["records"][0]
        row["spool_records"][0] = observer.records[0]
        for snapshot in row["snapshots"]:
            snapshot["sessions"][0]["records"][0] = stored["records"][0]
        if substitution == "none":
            _validate_observations(row)
        else:
            with pytest.raises(ValueError):
                _validate_observations(row)
    finally:
        owned.close()


@pytest.mark.parametrize("mutation", ["missing", "dto", "snapshot", "result", "persisted", "extra"])
def test_parent_rejects_a_gap_between_dispatch_spool_and_durable_records(mutation: str) -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    if mutation == "missing":
        row["spool_records"].pop()
    elif mutation == "extra":
        row["spool_records"][0]["accepted"] = True
    elif mutation == "persisted":
        for snapshot in row["snapshots"]:
            snapshot["sessions"][0]["records"][0] = "f" * 64
    else:
        row["spool_records"][0][mutation] = "writer_fault" if mutation == "result" else "f" * 64
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.parametrize("mutation", ["payload", "record_hash", "aggregate", "seal_head"])
def test_rollover_store_reader_rejects_tampered_durable_facts(
    tmp_path: Path, mutation: str
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition
    from scripts.capacity_rollover_worker import _snapshot
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.rollover_session(make_rollover_command()) is StoreDisposition.COMMITTED
        connection = owned.connection
        for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall():
            connection.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        if mutation == "payload":
            connection.execute(
                "UPDATE evidence_events SET canonical_payload='{}',canonical_bytes=2"
            )
        elif mutation == "record_hash":
            connection.execute("UPDATE evidence_events SET record_hash=?", ("0" * 64,))
        elif mutation == "aggregate":
            connection.execute("UPDATE evidence_sessions SET canonical_bytes=canonical_bytes+1")
        else:
            connection.execute(
                "UPDATE evidence_sessions SET head_hash=? WHERE state='sealed'", ("0" * 64,)
            )
        connection.commit()
        with pytest.raises(ValueError):
            _snapshot(owned.database, b"synthetic key")
    finally:
        owned.close()


@pytest.mark.parametrize(
    "failure",
    ["abnormal_exit", "bool_exit", "active", "failed_close", "unwaited", "empty", "wrong_arm"],
)
def test_shared_packaged_boundary_rejects_failed_ownership_and_scenario_identity(
    failure: str,
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from scripts.packaged_scenario import _ObservedRun, _validate_packaged_run
    from scripts.qualify_evidence_slice_zero import canonical_json_bytes

    cleanup = SimpleNamespace(
        closed=True, zero_active_observed=True, failures=(), failed_handles=(), waited_handles=(1,)
    )
    record = _ObservedRun(
        "a" * 40,
        "b" * 40,
        "c" * 64,
        "d" * 64,
        canonical_json_bytes([{"arm": "capacity_rollover"}]),
        (SimpleNamespace(process_handle=1),),
        cleanup,
        0,
    )
    if failure == "abnormal_exit":
        record = replace(record, exit_code=0xC0000005)
    elif failure == "bool_exit":
        record = replace(record, exit_code=False)
    elif failure == "active":
        cleanup.zero_active_observed = False
    elif failure == "failed_close":
        cleanup.failures = ("failed",)
    elif failure == "unwaited":
        cleanup.waited_handles = ()
    elif failure == "empty":
        record = replace(record, processes=())
    else:
        record = replace(record, observations=canonical_json_bytes([{"arm": "revoke_race"}]))
    with pytest.raises(ValueError):
        _validate_packaged_run(record, scenario="capacity_rollover")


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("seal", "disclosure_digest", "0" * 64),
        ("successor", "disclosure_digest", "0" * 64),
        ("successor", "retention_hours", 25),
        ("successor", "microphone_accepted", False),
        ("successor", "typed_accepted", False),
    ],
)
def test_rollover_reader_rejects_rehashed_foreign_consent(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
) -> None:
    import hashlib
    import json
    from dataclasses import replace

    from hermes_realtime.evidence.models import StoreDisposition
    from hermes_realtime.evidence.sqlite_spool import canonical_json_bytes, hre1_record_hash
    from scripts.capacity_rollover_worker import _snapshot
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)
    try:
        create = make_create_epoch()
        create = replace(
            create,
            microphone_accepted=True,
            session_opened=replace(
                create.session_opened,
                payload=replace(create.session_opened.payload, microphone_accepted=True),
            ),
            binding_opened=replace(
                create.binding_opened,
                payload=replace(create.binding_opened.payload, microphone_available=True),
            ),
        )
        command = make_rollover_command()
        command = replace(
            command,
            snapshots=(
                *command.snapshots[:2],
                replace(
                    command.snapshots[2],
                    payload=replace(command.snapshots[2].payload, microphone_accepted=True),
                ),
                replace(
                    command.snapshots[3],
                    payload=replace(command.snapshots[3].payload, microphone_available=True),
                ),
            ),
        )
        assert owned.create_epoch(create) is StoreDisposition.COMMITTED
        assert owned.rollover_session(command) is StoreDisposition.COMMITTED
        connection = owned.connection
        for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall():
            connection.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        installation = connection.execute(
            "SELECT installation_id FROM producer_installation"
        ).fetchone()[0]
        sessions = connection.execute(
            "SELECT logical_session_id,producer_instance_id,state "
            "FROM evidence_sessions ORDER BY rowid"
        ).fetchall()
        changes = 0
        for index, (session, producer, state) in enumerate(sessions):
            previous = None
            size = 0
            for event, sequence, kind, at, payload in connection.execute(
                "SELECT event_id,event_sequence,event_kind,recorded_at_utc,canonical_payload "
                "FROM evidence_events WHERE logical_session_id=? ORDER BY event_sequence",
                (session,),
            ).fetchall():
                document = json.loads(payload)
                if (target == "seal" and index == 0 and kind == "session_seal_requested") or (
                    target == "successor" and index == 1 and kind == "session_opened"
                ):
                    assert document[field] != value
                    document[field] = value
                    changes += 1
                raw = canonical_json_bytes(document)
                payload_hash = hashlib.sha256(raw).hexdigest()
                digest = hre1_record_hash(
                    installation_id=installation,
                    producer_instance_id=producer,
                    event_id=event,
                    logical_session_id=session,
                    event_sequence=sequence,
                    event_kind=kind,
                    recorded_at_utc=at,
                    payload_hash=payload_hash,
                    previous_hash=previous,
                )
                connection.execute(
                    "UPDATE evidence_events SET canonical_payload=?,canonical_bytes=?,"
                    "payload_hash=?,"
                    "previous_hash=?,record_hash=? WHERE event_id=?",
                    (raw.decode(), len(raw), payload_hash, previous, digest, event),
                )
                previous = digest
                size += len(raw)
            connection.execute(
                "UPDATE evidence_sessions SET canonical_bytes=?,head_hash=? "
                "WHERE logical_session_id=?",
                (size, previous if state == "sealed" else None, session),
            )
        connection.commit()
        assert changes == 1
        # Every edited record has a valid recomputed chain. Consent, not a stale
        # checksum, must make the independent reader refuse this transition.
        with pytest.raises(ValueError):
            _snapshot(owned.database, b"synthetic key")
    finally:
        owned.close()


@pytest.mark.parametrize(
    "mutation", ["binding_id", "binding_generation", "microphone_available", "successor_expiry"]
)
def test_rollover_observer_rejects_coordinated_command_substitution(
    tmp_path: Path, mutation: str
) -> None:
    from dataclasses import replace
    from datetime import datetime, timedelta

    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc
    from scripts.capacity_rollover_worker import _ObserveRolloverSpool
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)

    def altered_snapshot(snapshot):
        payload = snapshot.payload
        field = mutation
        if field == "successor_expiry" or not hasattr(payload, field):
            return snapshot
        value = {
            "binding_id": "12345678-1234-4234-8234-123456789abc",
            "binding_generation": 91,
            "microphone_available": True,
        }[field]
        assert getattr(payload, field) != value
        return replace(snapshot, payload=replace(payload, **{field: value}))

    class SubstitutingSpool:
        def __getattr__(self, name):
            return getattr(owned, name)

        def create_epoch(self, command):
            opened = altered_snapshot(command.session_opened)
            binding = altered_snapshot(command.binding_opened)
            return owned.create_epoch(
                replace(
                    command,
                    binding_id=opened.payload.binding_id,
                    binding_generation=binding.payload.binding_generation,
                    session_opened=opened,
                    binding_opened=binding,
                )
            )

        def rollover_session(self, command):
            snapshots = tuple(altered_snapshot(s) for s in command.snapshots)
            return owned.rollover_session(
                replace(
                    command,
                    binding_id=snapshots[2].payload.binding_id,
                    binding_generation=snapshots[3].payload.binding_generation,
                    successor_expires_at_utc=(
                        format_canonical_utc(
                            datetime.fromisoformat(command.successor_expires_at_utc)
                            + timedelta(hours=1)
                        )
                        if mutation == "successor_expiry"
                        else command.successor_expires_at_utc
                    ),
                    snapshots=snapshots,
                )
            )

    observer = _ObserveRolloverSpool(SubstitutingSpool(), owned.database, b"synthetic key")
    try:
        # Both substitutions use real spool writes, valid payloads and HRE1
        # chains. Only the independently captured dispatched command differs.
        with pytest.raises(ValueError, match="dispatched|retention"):
            observer.create_epoch(make_create_epoch())
            observer.rollover_session(make_rollover_command())
    finally:
        owned.close()


@pytest.mark.parametrize("mutation", ["controls", "expiry", "missing_commands"])
def test_rollover_parent_rejects_coordinated_store_changes_against_dispatched_commands(
    mutation: str,
) -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    if mutation == "missing_commands":
        del row["commands"]
    else:
        for snapshot in row["snapshots"]:
            for session in snapshot["sessions"]:
                if mutation == "controls":
                    session["controls"] = ["d" * 64] * len(session["controls"])
                elif session["predecessor"]:
                    session["expires"] = "d" * 64
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.parametrize(
    "mutation", ["create_sequence", "create_fingerprint", "rollover_ordinal", "runtime_retention"]
)
def test_rollover_rejects_foreign_command_identity_and_consent_deadline(
    tmp_path: Path, mutation: str
) -> None:
    from dataclasses import replace
    from datetime import datetime, timedelta

    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc
    from scripts.capacity_rollover_worker import _ObserveRolloverSpool
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)

    class SubstitutingSpool:
        def __getattr__(self, name):
            return getattr(owned, name)

        def create_epoch(self, command):
            if mutation == "create_sequence":
                command = replace(command, control_sequence=command.control_sequence + 1)
            elif mutation == "create_fingerprint":
                command = replace(command, control_fingerprint_hash="f" * 64)
            return owned.create_epoch(command)

        def rollover_session(self, command):
            if mutation == "rollover_ordinal":
                command = replace(command, admission_ordinal=command.admission_ordinal + 1)
            return owned.rollover_session(command)

    observer = _ObserveRolloverSpool(SubstitutingSpool(), owned.database, b"synthetic key")
    try:
        command = make_rollover_command()
        if mutation == "runtime_retention":
            # The wrong deadline originates before observation and is persisted
            # unchanged. Command/store equality alone cannot establish policy.
            command = replace(
                command,
                successor_expires_at_utc=format_canonical_utc(
                    datetime.fromisoformat(command.successor_expires_at_utc) + timedelta(hours=144)
                ),
            )
        with pytest.raises(ValueError):
            observer.create_epoch(make_create_epoch())
            observer.rollover_session(command)
    finally:
        owned.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "dto",
        "transport_dto",
        "request",
        "gap",
        "rollover_order",
        "bool_ordinal",
        "negative_lag",
        "large_lag",
        "bool_lag",
    ],
)
def test_rollover_parent_rejects_foreign_command_or_retention_authority(mutation: str) -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    if mutation == "dto":
        row["commands"]["create_dto"] = "e" * 64
    elif mutation == "transport_dto":
        row["dispatch"]["rollover_dto"] = "e" * 64
    elif mutation == "request":
        row["commands"]["request"] = row["dispatch"]["request"] = "e" * 64
    elif mutation == "gap":
        row["dispatch"]["ordinals"][2] += 1
    elif mutation == "rollover_order":
        row["dispatch"]["rollover_ordinal"] += 1
    elif mutation == "bool_ordinal":
        row["dispatch"]["ordinals"][0] = True
    else:
        value = {"negative_lag": -1, "large_lag": 5_000_001, "bool_lag": False}[mutation]
        for snapshot in row["snapshots"][2:]:
            snapshot["sessions"][1]["retention_lag_us"] = value
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.parametrize(
    "offset_microseconds", [-6_000_000, -5_000_000, 0, 1, 144 * 3600 * 1_000_000]
)
def test_rollover_reader_checks_successor_retention_with_bounded_clock_skew(
    tmp_path: Path, offset_microseconds: int
) -> None:
    from datetime import datetime, timedelta

    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc
    from scripts.capacity_rollover_worker import _snapshot
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)
    try:
        owned.create_epoch(make_create_epoch())
        command = make_rollover_command()
        owned.rollover_session(command)
        opened = owned.connection.execute(
            "SELECT opened_at_utc FROM evidence_sessions WHERE logical_session_id=?",
            (command.successor_logical_session_id,),
        ).fetchone()[0]
        expires = format_canonical_utc(
            datetime.fromisoformat(opened) + timedelta(hours=24, microseconds=offset_microseconds)
        )
        owned.connection.execute(
            "UPDATE evidence_sessions SET expires_at_utc=? WHERE logical_session_id=?",
            (expires, command.successor_logical_session_id),
        )
        owned.connection.commit()
        if offset_microseconds in (-5_000_000, 0):
            observed = _snapshot(owned.database, b"synthetic key")
            assert observed["sessions"][1]["retention_lag_us"] == -offset_microseconds
        else:
            with pytest.raises(ValueError, match="retention"):
                _snapshot(owned.database, b"synthetic key")
    finally:
        owned.close()


@pytest.mark.parametrize("ordinals", [(2,), (4,), (True,), (3, 5), (3, 4, 5)])
def test_dispatch_observer_preserves_fifo_and_rejects_gaps_before_delegation(
    ordinals: tuple,
) -> None:
    from types import SimpleNamespace

    from scripts.capacity_rollover_worker import _ObserveDispatch
    from tests.evidence.test_sqlite_spool import ordinary_record, turn_opened_snapshot

    calls = []
    observer = _ObserveDispatch(SimpleNamespace(append_record=calls.append), b"synthetic key")

    def item(ordinal):
        record = ordinary_record(turn_opened_snapshot(3, event_index=3), 3)
        object.__setattr__(record, "admission_ordinal", ordinal)
        return record

    if ordinals == (3, 4, 5):
        for ordinal in ordinals:
            observer.append_record(item(ordinal))
        assert len(calls) == 3
    else:
        with pytest.raises(ValueError, match="ordinal"):
            for ordinal in ordinals:
                observer.append_record(item(ordinal))
        assert len(calls) == (1 if ordinals == (3, 5) else 0)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "create_ordinal",
        "record_ordinal",
        "rollover_ordinal",
        "protocol",
        "drain_owner",
        "drain_watermark",
    ],
)
def test_dequeue_observer_binds_queue_envelopes_before_payload_stripping(mutation: str) -> None:
    from dataclasses import replace

    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence import models as m
    from scripts.capacity_rollover_worker import _ObserveQueue
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        ordinary_record,
        turn_opened_snapshot,
    )

    queue = a.BoundedEvidenceWriterQueueV1()
    observer = _ObserveQueue(queue, b"synthetic key", owner_generation=41)
    items = [
        a.EvidenceWriterQueueItemV1(
            protocol_version=1,
            lane=a.WriterQueueLane.ORDERED,
            payload=make_create_epoch(),
            admission_ordinal=1 if mutation == "create_ordinal" else 2,
        )
    ]
    for ordinal in range(3, 22):
        if ordinal == 15:
            payload = replace(
                make_rollover_command(),
                admission_ordinal=16 if mutation == "rollover_ordinal" else 15,
            )
        else:
            payload = ordinary_record(
                turn_opened_snapshot(3, event_index=3),
                ordinal + (1 if mutation == "record_ordinal" and ordinal == 3 else 0),
            )
        items.append(
            a.EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=a.WriterQueueLane.ORDERED,
                payload=payload,
                admission_ordinal=ordinal,
            )
        )
    items.append(
        a.EvidenceWriterQueueItemV1(
            protocol_version=1,
            lane=a.WriterQueueLane.DRAIN,
            admission_ordinal=22,
            payload=m.DrainAndStopV1(
                protocol_version=1,
                owner_generation=42 if mutation == "drain_owner" else 41,
                final_admission_ordinal=20 if mutation == "drain_watermark" else 21,
            ),
        )
    )
    if mutation == "protocol":
        object.__setattr__(items[0], "protocol_version", 2)

    def consume():
        for item in items:
            queue.put_nowait(item)
            assert queue.get_blocking() is item

    try:
        if mutation == "none":
            consume()
            assert [r["ordinal"] for r in observer.records] == list(range(2, 23))
            assert observer.records[0]["kind"] == "create"
            assert observer.records[13]["kind"] == "rollover"
            assert observer.records[-1]["kind"] == "drain"
        else:
            with pytest.raises(ValueError):
                consume()
    finally:
        observer.restore()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "create_ordinal",
        "rollover_ordinal",
        "lane",
        "payload",
        "bool_version",
        "bool_ordinal",
    ],
)
def test_parent_rejects_queue_envelopes_that_disagree_with_transport(mutation: str) -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    if mutation == "missing":
        row["queue"].pop()
    elif mutation == "create_ordinal":
        row["queue"][0]["ordinal"] = 1
    elif mutation == "rollover_ordinal":
        row["queue"][13]["ordinal"] = 14
    elif mutation == "lane":
        row["queue"][13]["lane"] = "revoke"
    elif mutation == "payload":
        row["queue"][13]["payload"] = "f" * 64
    elif mutation == "bool_version":
        row["queue"][0]["version"] = True
    else:
        row["queue"][0]["ordinal"] = True
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.parametrize("mutation", ["omitted", "ahead"])
def test_rollover_reader_observes_clock_high_water_before_a_later_turn_can_repair_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from datetime import datetime, timedelta

    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc
    from scripts.capacity_rollover_worker import _ObserveRolloverSpool
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        make_rollover_command,
        make_spool,
    )

    owned = make_spool(tmp_path)
    observer = _ObserveRolloverSpool(owned, owned.database, b"synthetic key")
    try:
        observer.create_epoch(make_create_epoch())
        previous = owned.connection.execute(
            "SELECT clock_high_water_utc FROM producer_installation"
        ).fetchone()[0]
        moment = datetime.fromisoformat(previous) + timedelta(seconds=1)
        monkeypatch.setattr(owned, "_clock", lambda: moment)
        advance = owned._advance_high_water

        def defective_advance(recorded_at):
            if mutation == "ahead":
                advance(format_canonical_utc(moment + timedelta(seconds=1)))

        monkeypatch.setattr(owned, "_advance_high_water", defective_advance)
        with pytest.raises(ValueError, match="high.water"):
            observer.rollover_session(make_rollover_command())
    finally:
        owned.close()


@pytest.mark.parametrize("mutation", ["installation_created", "epoch_opened", "purge_required"])
def test_rollover_reader_rejects_foreign_installation_or_epoch_authority(
    tmp_path: Path, mutation: str
) -> None:
    from datetime import datetime, timedelta

    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc
    from scripts.capacity_rollover_worker import _snapshot
    from tests.evidence.test_sqlite_spool import make_create_epoch, make_spool

    owned = make_spool(tmp_path)
    try:
        owned.create_epoch(make_create_epoch())
        if mutation == "purge_required":
            owned.connection.execute(
                "UPDATE producer_installation SET purge_required=1,"
                "purge_reason='clock_rollback',purge_scope='store'"
            )
        else:
            previous = owned.connection.execute(
                "SELECT created_at_utc FROM producer_installation"
            ).fetchone()[0]
            changed = format_canonical_utc(
                datetime.fromisoformat(previous) + timedelta(microseconds=1)
            )
            if mutation == "installation_created":
                owned.connection.execute(
                    "UPDATE producer_installation SET created_at_utc=?", (changed,)
                )
            else:
                owned.connection.execute("UPDATE consent_epochs SET opened_at_utc=?", (changed,))
        owned.connection.commit()
        with pytest.raises(ValueError):
            _snapshot(owned.database, b"synthetic key")
    finally:
        owned.close()


@pytest.mark.parametrize(
    "mutation", ["missing_clock", "stale_clock", "foreign_authority", "rollover_time"]
)
def test_rollover_parent_rejects_clock_and_store_authority_gaps(mutation: str) -> None:
    from scripts.capacity_rollover import _validate_observations

    row = _observations()
    committed = row["snapshots"][2]
    if mutation == "missing_clock":
        del committed["clock"]
    elif mutation == "stale_clock":
        committed["clock"] = row["snapshots"][0]["clock"]
    elif mutation == "foreign_authority":
        committed["authority"] = "e" * 64
    else:
        committed["sessions"][0]["last_event_at"] = "e" * 64
    with pytest.raises(ValueError):
        _validate_observations(row)
