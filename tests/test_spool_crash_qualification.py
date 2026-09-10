"""Acceptance must reject a successful recovery with the wrong durable history."""

from __future__ import annotations

import os

import pytest

from scripts.spool_crash_matrix import ObservedSpoolCrashMatrixV1, _validate_checkpoint


def test_governed_cases_and_existing_driver_have_the_same_closed_matrix() -> None:
    import json
    from pathlib import Path

    from scripts.spool_crash_matrix import FAILPOINTS_V1
    from tests.evidence.spool_crash_worker import FAILPOINTS, case_ids

    schema = json.loads(
        (
            Path(__file__).parents[1] / "scripts/schemas/qualification-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert FAILPOINTS_V1 == FAILPOINTS
    assert list(case_ids()) == schema["$defs"]["SpoolCrashCaseV1"]["properties"]["caseId"]["enum"]


def test_registration_and_receipts_reject_reconstruction() -> None:
    from scripts.qualify_evidence_slice_zero import (
        SCENARIO_REGISTRY_V1,
        SPOOL_CRASH_MATRIX_REGISTRATION_V1,
        SpoolCrashMatrixRegistrationV1,
    )
    from scripts.spool_crash_matrix import validate_spool_crash_matrix_v1

    assert SCENARIO_REGISTRY_V1[13] is SPOOL_CRASH_MATRIX_REGISTRATION_V1
    with pytest.raises(ValueError, match="canonical"):
        SpoolCrashMatrixRegistrationV1().produce(None, None, None)
    with pytest.raises(TypeError):
        validate_spool_crash_matrix_v1({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        validate_spool_crash_matrix_v1(object.__new__(ObservedSpoolCrashMatrixV1))


@pytest.mark.parametrize("event_index", range(7))
def test_fixture_oracle_requires_every_source_field_and_exact_type(event_index) -> None:
    import copy

    from hermes_realtime.evidence.models import evidence_snapshot_to_primitive
    from scripts.spool_crash_oracle import validate_spool_snapshot_v1
    from tests.evidence import spool_crash_worker as driver

    normal = driver._make_create_epoch()
    rollback = driver._make_create_epoch(
        epoch_id="20000000-0000-4000-8000-000000000003",
        session_id="20000000-0000-4000-8000-000000000004",
        binding_id="20000000-0000-4000-8000-000000000006",
        event_offset=100,
    )
    snapshot = evidence_snapshot_to_primitive(
        (
            normal.session_opened,
            normal.binding_opened,
            driver._ordinary_record().snapshot,
            driver._binding_close().snapshot,
            driver._seal_command().snapshot,
            rollback.session_opened,
            rollback.binding_opened,
        )[event_index]
    )
    validate_spool_snapshot_v1(snapshot)
    for layer in (None, "payload"):
        original = snapshot if layer is None else snapshot[layer]
        for field in original:
            for mutation in ("remove", "type"):
                changed = copy.deepcopy(snapshot)
                target = changed if layer is None else changed[layer]
                if mutation == "remove":
                    del target[field]
                else:
                    value = target[field]
                    target[field] = int(value) if type(value) is bool else True
                with pytest.raises(ValueError, match="pinned synthetic source"):
                    validate_spool_snapshot_v1(changed)
        changed = copy.deepcopy(snapshot)
        target = changed if layer is None else changed[layer]
        target["unknown"] = None
        with pytest.raises(ValueError, match="pinned synthetic source"):
            validate_spool_snapshot_v1(changed)


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "record",
        "payload",
        "seal",
        "epoch",
        "delete",
        "candidate_oracle",
        "version",
        "application",
    ],
)
def test_independent_reader_revalidates_real_committed_seals(
    tmp_path, mutation, monkeypatch
) -> None:
    import sqlite3

    from scripts.storage_observation import _database_state
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(tmp_path)
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
        assert spool.append_binding_close(driver._binding_close()).value == "committed"
        assert spool.seal_epoch(driver._seal_command()).value == "committed"
    finally:
        spool.close()
    database = tmp_path / "evidence/capture-v1.sqlite3"
    if mutation == "candidate_oracle":
        from hermes_realtime.evidence import models, sqlite_spool

        def refused(*args, **kwargs):
            raise AssertionError("the candidate cannot supply its own hash oracle")

        monkeypatch.setattr(sqlite_spool, "canonical_json_bytes", refused)
        monkeypatch.setattr(sqlite_spool, "hre1_record_hash", refused)
        monkeypatch.setattr(models, "parse_evidence_snapshot_json", refused)
    if mutation in {None, "candidate_oracle"}:
        observed = _database_state(database)
        assert observed["events"] == 4
        assert observed["sessions"] == ["sealed"]
        assert len(observed["seals"]) == 1
        return
    statements = {
        "record": "UPDATE evidence_events SET record_hash=? WHERE event_sequence=4",
        "payload": "UPDATE evidence_events SET payload_hash=? WHERE event_sequence=4",
        "seal": "UPDATE evidence_sessions SET head_hash=?",
        "epoch": "UPDATE consent_epochs SET state='active',closed_at_utc=NULL",
        "delete": "DELETE FROM evidence_events WHERE event_sequence=4",
        "version": "PRAGMA user_version=2",
        "application": "PRAGMA application_id=0",
    }
    with sqlite3.connect(database) as connection:
        # Model damaged storage while preserving the original schema, so the
        # observer must detect the broken chain rather than an absent trigger.
        trigger = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='evidence_events_are_append_only'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER evidence_events_are_append_only")
        connection.execute(
            statements[mutation], ("f" * 64,) if mutation in {"record", "payload", "seal"} else ()
        )
        connection.execute(trigger)
    with pytest.raises(ValueError):
        _database_state(database)


def _purge_observation():
    import hashlib

    return {
        "before": {
            "files": {
                ".hermes-realtime-evidence-root-v1": "a" * 64,
                "capture-v1.owner": "b" * 64,
                "purge-decoy.bin": hashlib.sha256(b"not-owned").hexdigest(),
            },
            "sentinel": "full_purge_pending",
            "database": {"schema": False},
        },
        "after": {
            "files": {
                ".hermes-realtime-evidence-root-v1": "a" * 64,
                "capture-v1.owner": "c" * 64,
                "purge-decoy.bin": hashlib.sha256(b"not-owned").hexdigest(),
            },
            "sentinel": "clear",
            "database": {"schema": False},
        },
        "disposition": "purge_completed",
        "baseline": {},
    }


@pytest.mark.parametrize(
    "mutation",
    [None, "sidecar", "marker", "sentinel", "decoy", "entry", "prefix", "extra", "activation_temp"],
)
def test_purge_acceptance_requires_exact_absence_and_retained_authority(mutation) -> None:
    from scripts.spool_crash_matrix import _validate_recovery

    row = _purge_observation()
    if mutation == "sidecar":
        row["after"]["files"]["capture-v1.sqlite3-wal"] = "d" * 64
    elif mutation == "marker":
        row["after"]["files"][".hermes-realtime-evidence-root-v1"] = "d" * 64
    elif mutation == "sentinel":
        row["after"]["sentinel"] = "full_purge_pending"
    elif mutation == "decoy":
        del row["after"]["files"]["purge-decoy.bin"]
    elif mutation == "entry":
        row["before"]["sentinel"] = "clear"
    elif mutation == "prefix":
        row["before"]["files"]["capture-v1.sqlite3"] = "d" * 64
    elif mutation == "extra":
        row["after"]["files"]["unexpected"] = "d" * 64
    elif mutation == "activation_temp":
        row["after"]["files"]["capture-v1.owner.init"] = "d" * 64
    if mutation is None:
        _validate_recovery("after_full_purge_absence_verify", "caught-up", row)
    else:
        with pytest.raises(ValueError):
            _validate_recovery("after_full_purge_absence_verify", "caught-up", row)


@pytest.mark.parametrize("mutation", [None, "exit", "unwaited", "active", "open", "failure"])
def test_expected_crash_exit_never_substitutes_for_owned_cleanup(mutation) -> None:
    from types import SimpleNamespace

    from scripts.storage_process import _StorageInvocation, _validate_invocation

    cleanup = SimpleNamespace(
        closed=True,
        zero_active_observed=True,
        failures=(),
        failed_handles=(),
        waited_handles=(71,),
    )
    if mutation == "unwaited":
        cleanup.waited_handles = ()
    elif mutation == "active":
        cleanup.zero_active_observed = False
    elif mutation == "open":
        cleanup.closed = False
    elif mutation == "failure":
        cleanup.failures = ("wait",)
    record = _StorageInvocation(
        b'{"checkpoint":"before_begin"}\n',
        SimpleNamespace(process_handle=71),
        cleanup,
        0 if mutation == "exit" else 197,
        197,
    )
    if mutation is None:
        assert _validate_invocation(record) == {"checkpoint": "before_begin"}
    else:
        with pytest.raises(ValueError):
            _validate_invocation(record)


def test_receipts_cannot_be_constructed_from_supplied_claims() -> None:
    with pytest.raises(TypeError, match="producer-minted"):
        ObservedSpoolCrashMatrixV1()


@pytest.mark.parametrize("point", ["before_begin", "after_event_insert_before_commit"])
def test_uncommitted_event_cannot_be_accepted_as_durable(point: str) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_checkpoint(point, {"events": 3, "sessions": ["open"], "epochs": ["active"]})


def test_seal_commit_requires_the_complete_closed_epoch() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_checkpoint(
            "after_seal_commit_before_ack",
            {"events": 4, "sessions": ["sealed"], "epochs": ["active"]},
        )


@pytest.mark.parametrize("point", ["after_first_schema_commit", "after_recreate_schema_commit"])
def test_schema_commit_checkpoint_cannot_report_an_empty_database(point) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_checkpoint(point, {"schema": False})


@pytest.mark.parametrize("purge_required", [1, -1])
def test_recovered_seal_requires_a_cleared_purge_latch(purge_required) -> None:
    from copy import deepcopy

    from scripts.spool_crash_matrix import _validate_recovery

    state = {
        "files": {
            ".hermes-realtime-evidence-root-v1": "a" * 64,
            "capture-v1.owner": "b" * 64,
            "capture-v1.sqlite3": "c" * 64,
        },
        "sentinel": "clear",
        "database": {
            "schema": True,
            "events": 4,
            "sessions": ["sealed"],
            "epochs": ["closed"],
            "seals": ["d" * 64],
            "purge_required": 0,
            "tombstones": [],
            "erasures": [],
            "logical_digest": "e" * 64,
        },
    }
    row = {"before": state, "after": deepcopy(state), "disposition": "recovered", "baseline": {}}
    _validate_recovery("after_seal_commit_before_ack", "caught-up", row)
    row["after"]["database"]["purge_required"] = purge_required
    with pytest.raises(ValueError):
        _validate_recovery("after_seal_commit_before_ack", "caught-up", row)


@pytest.mark.parametrize("point", ["after_first_db_create", "after_recreate_db_create"])
def test_database_create_checkpoint_requires_the_empty_file_digest(point) -> None:
    import hashlib

    from scripts.spool_crash_matrix import _validate_recovery

    row = _purge_observation()
    for state in (row["before"], row["after"]):
        decoy = state["files"].pop("purge-decoy.bin")
        if "recreate" in point:
            state["files"]["recreation-decoy.bin"] = decoy
    row["before"]["sentinel"] = "first_create_pending"
    row["before"]["files"]["capture-v1.sqlite3"] = hashlib.sha256(b"").hexdigest()
    _validate_recovery(point, "caught-up", row)
    row["before"]["files"]["capture-v1.sqlite3"] = "d" * 64
    with pytest.raises(ValueError):
        _validate_recovery(point, "caught-up", row)


@pytest.mark.parametrize("point", ["after_rollback_sentinel_fsync", "before_rollback_db_latch"])
def test_pre_latch_rollback_requires_the_original_logical_database(point) -> None:
    from scripts.spool_crash_matrix import _validate_recovery

    row = _purge_observation()
    for state in (row["before"], row["after"]):
        state["files"]["rollback-decoy.bin"] = state["files"].pop("purge-decoy.bin")
    row["before"]["files"]["capture-v1.sqlite3"] = "c" * 64
    row["before"]["sentinel"] = "clock_rollback_purge_pending"
    row["before"]["database"] = {
        "schema": True,
        "events": 6,
        "sessions": ["open", "sealed"],
        "epochs": ["active", "closed"],
        "seals": ["d" * 64],
        "purge_required": 0,
        "tombstones": [],
        "erasures": [],
        "logical_digest": "e" * 64,
    }
    row["baseline"] = {"databaseSha256": "e" * 64, "sentinelSha256": "f" * 64}
    _validate_recovery(point, "caught-up", row)
    row["before"]["database"]["logical_digest"] = "a" * 64
    with pytest.raises(ValueError):
        _validate_recovery(point, "caught-up", row)


@pytest.mark.parametrize("owner", ["root_marker", "sentinel"])
@pytest.mark.parametrize("stage", ["full_write", "flush"])
def test_completed_initialization_removes_its_temporary(owner, stage) -> None:
    from copy import deepcopy

    from scripts.spool_crash_matrix import _validate_recovery
    from scripts.spool_crash_oracle import initialization_digest_v1

    name = ".hermes-realtime-evidence-root-v1" if owner == "root_marker" else "capture-v1.owner"
    before = {
        "files": {name + ".init": initialization_digest_v1(owner, stage)},
        "sentinel": "no_final_sentinel",
        "database": {"schema": False},
    }
    if owner == "sentinel":
        before["files"][".hermes-realtime-evidence-root-v1"] = "b" * 64
    after = deepcopy(before)
    del after["files"][name + ".init"]
    row = {"before": before, "after": after, "disposition": "absent", "baseline": {}}
    point = f"after_{owner}_init_{stage}"
    _validate_recovery(point, "caught-up", row)
    after["files"][name + ".init"] = "a" * 64
    with pytest.raises(ValueError):
        _validate_recovery(point, "caught-up", row)


@pytest.mark.parametrize("owner", ["root_marker", "sentinel"])
@pytest.mark.parametrize("stage", ["partial_write", "full_write", "flush"])
def test_initialization_requires_the_exact_image_at_each_boundary(owner, stage) -> None:
    import copy
    import hashlib

    from hermes_realtime.evidence.storage_security import encode_root_marker, initial_sentinel_image
    from scripts.spool_crash_matrix import _validate_recovery

    marker = ".hermes-realtime-evidence-root-v1"
    temporary = marker + ".init" if owner == "root_marker" else "capture-v1.owner.init"
    image = (
        encode_root_marker("50000000-0000-4000-8000-000000000001")
        if owner == "root_marker"
        else initial_sentinel_image("50000000-0000-4000-8000-000000000002")
    )
    if stage == "partial_write":
        image = image[: len(image) // 2]
    retained = {marker: "a" * 64} if owner == "sentinel" else {}
    before = {
        "files": {**retained, temporary: hashlib.sha256(image).hexdigest()},
        "sentinel": "no_final_sentinel",
        "database": {"schema": False},
    }
    after = copy.deepcopy(before)
    if stage != "partial_write":
        after["files"] = retained
    row = {
        "before": before,
        "after": after,
        "disposition": "faulted" if stage == "partial_write" else "absent",
        "baseline": {},
    }
    point = f"after_{owner}_init_{stage}"
    _validate_recovery(point, "caught-up", row)
    before["files"][temporary] = hashlib.sha256(b"malformed-image").hexdigest()
    if stage == "partial_write":
        after["files"][temporary] = before["files"][temporary]
    with pytest.raises(ValueError):
        _validate_recovery(point, "caught-up", row)


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize("field", ["scope_key", "erasure_request_id", "last_admission_ordinal"])
def test_recovery_receipt_cannot_change_its_erased_epoch_or_authority(tmp_path, field) -> None:
    import sqlite3
    from datetime import timedelta

    from scripts.spool_crash_matrix import _validate_recovery
    from scripts.storage_observation import observe_storage
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(tmp_path)
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
    finally:
        spool.close()
    root = tmp_path / "evidence"
    before = observe_storage(root)
    recovered = driver._make_spool(tmp_path, clock=driver._Clock(driver.START + timedelta(hours=2)))
    try:
        assert recovered.recover_existing().value == "recovered"
    finally:
        recovered.close()
    row = {
        "before": before,
        "after": observe_storage(root),
        "disposition": "recovered",
        "baseline": {},
    }
    _validate_recovery("before_begin", "caught-up", row)
    with sqlite3.connect(root / "capture-v1.sqlite3") as connection:
        trigger = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='erasure_tombstones_are_immutable'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER erasure_tombstones_are_immutable")
        value = 3 if field == "last_admission_ordinal" else "90000000-0000-4000-8000-000000000003"
        connection.execute(f"UPDATE erasure_tombstones SET {field}=?", (value,))
        connection.execute(trigger)
    row["after"] = observe_storage(root)
    with pytest.raises(ValueError):
        _validate_recovery("before_begin", "caught-up", row)
