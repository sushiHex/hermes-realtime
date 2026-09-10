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
    return dict(
        arm="capacity_rollover",
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
        source=source | {"consent": "f" * 64},
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
        commitment
        == hmac.new(b"synthetic key", b"accepted_consent\0" + raw, hashlib.sha256).hexdigest()
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
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        observer = _ObserveRolloverSpool(owned, owned.database, b"synthetic key")
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
