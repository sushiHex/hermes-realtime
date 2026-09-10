"""Acceptance must reject a successful recovery with the wrong durable history."""

from __future__ import annotations

import os

import pytest

from scripts.spool_crash_matrix import ObservedSpoolCrashMatrixV1, _validate_checkpoint


def test_governed_cases_and_existing_driver_have_the_same_closed_matrix() -> None:
    import json
    from pathlib import Path

    from scripts.spool_crash_matrix import FAILPOINTS_V1, _entry_sentinel, _validate_case_contract
    from tests.evidence.spool_crash_worker import FAILPOINTS, case_ids

    schema = json.loads(
        (
            Path(__file__).parents[1] / "scripts/schemas/qualification-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert FAILPOINTS_V1 == FAILPOINTS
    assert list(case_ids()) == schema["$defs"]["SpoolCrashCaseV1"]["properties"]["caseId"]["enum"]
    _validate_case_contract(schema)
    cases = schema["properties"]["scenarios"]["prefixItems"][13]["allOf"][1]["properties"][
        "caseResults"
    ]["prefixItems"]
    for index, case in enumerate(cases):
        fields = case["allOf"][1]["properties"]
        assert fields["sentinelStateAtRecoveryEntry"]["const"] == _entry_sentinel(index // 2)


@pytest.mark.parametrize("field", ["exitMode", "sentinelStateAtRecoveryEntry", "assertions"])
def test_producer_rejects_changed_per_case_contract(field) -> None:
    import json
    from pathlib import Path

    from scripts.spool_crash_matrix import _validate_case_contract

    schema = json.loads(
        (
            Path(__file__).parents[1] / "scripts/schemas/qualification-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    cases = schema["properties"]["scenarios"]["prefixItems"][13]["allOf"][1]["properties"][
        "caseResults"
    ]["prefixItems"]
    cases[18]["allOf"][1]["properties"][field] = {}
    with pytest.raises(ValueError, match="governed spool case contract"):
        _validate_case_contract(schema)


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
        "lineage",
        "conflict",
        "expiry",
        "opened",
        "epoch_opened",
        "epoch_closed",
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
        "expiry": "UPDATE evidence_sessions SET expires_at_utc='2026-08-10T00:00:00.000000Z'",
        "opened": "UPDATE evidence_sessions SET opened_at_utc='2026-08-08T00:00:01.000000Z'",
        "epoch_opened": "UPDATE consent_epochs SET opened_at_utc='2026-08-08T00:00:01.000000Z'",
        "epoch_closed": "UPDATE consent_epochs SET closed_at_utc='2026-08-08T00:00:03.000000Z'",
    }
    with sqlite3.connect(database) as connection:
        # Model damaged storage while preserving the original schema, so the
        # observer must detect the broken chain rather than an absent trigger.
        trigger = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='evidence_events_are_append_only'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER evidence_events_are_append_only")
        if mutation == "lineage":
            for table in ("consent_epochs", "evidence_sessions"):
                connection.execute(
                    f"UPDATE {table} SET consent_epoch_id=?",
                    ("90000000-0000-4000-8000-000000000003",),
                )
        elif mutation == "conflict":
            connection.execute(
                "INSERT INTO evidence_conflicts VALUES (?,?,?,?,?)",
                (
                    "90000000-0000-4000-8000-000000000001",
                    driver.SESSION_ID,
                    "40000000-0000-4000-8000-000000000001",
                    "sealed_session_reuse",
                    "2026-08-08T00:00:03.000000Z",
                ),
            )
        else:
            connection.execute(
                statements[mutation],
                ("f" * 64,) if mutation in {"record", "payload", "seal"} else (),
            )
        connection.execute(trigger)
    with pytest.raises(ValueError):
        _database_state(database)


def _pin_authority_images(row, point):
    from scripts.spool_crash_matrix import FAILPOINTS_V1
    from scripts.spool_crash_oracle import (
        checkpoint_installation_digest_v1,
        checkpoint_sentinel_digest_v1,
        checkpoint_source_digest_v1,
        initialization_digest_v1,
    )

    index = FAILPOINTS_V1.index(point)
    for key in ("before", "after"):
        files = row[key]["files"]
        marker = ".hermes-realtime-evidence-root-v1"
        if marker in files:
            files[marker] = initialization_digest_v1("root_marker", "full_write")
        if "capture-v1.owner" in files:
            files["capture-v1.owner"] = checkpoint_sentinel_digest_v1(
                index, "caught-up", after=key == "after"
            )
        db = row[key]["database"]
        if db["schema"]:
            source_index = index
            if key == "after" and row["disposition"] == "recovered":
                source_index = 4 if index in {4, 8, 28} else 6
            db["source_digest"] = checkpoint_source_digest_v1(source_index)
            db["installation_digest"] = checkpoint_installation_digest_v1(
                index, after=key == "after"
            )
    return row


def _purge_observation():
    import hashlib

    row = {
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
    return _pin_authority_images(row, "after_full_purge_absence_verify")


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
    _pin_authority_images(row, "after_seal_commit_before_ack")
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
    _pin_authority_images(row, point)
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
    _pin_authority_images(row, point)
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
    _pin_authority_images(row, point)
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
    _pin_authority_images(row, point)
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


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize(
    "field",
    ["scope_key", "erasure_request_id", "control_fingerprint_hash", "last_admission_ordinal"],
)
def test_pending_revocation_preserves_the_complete_accepted_authority(tmp_path, field) -> None:
    import sqlite3

    from scripts.spool_crash_matrix import _validate_recovery
    from scripts.storage_observation import observe_storage
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(tmp_path)
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
        request, _ = driver._revoke_commands()
        assert spool.commit_revoke_request(request).value == "revoke_durably_scheduled"
    finally:
        spool.close()
    root = tmp_path / "evidence"
    state = observe_storage(root)
    row = {"before": state, "after": state, "disposition": "faulted", "baseline": {}}
    _validate_recovery("after_revoke_request_commit", "caught-up", row)
    values = {
        "scope_key": "90000000-0000-4000-8000-000000000003",
        "erasure_request_id": "90000000-0000-4000-8000-000000000003",
        "control_fingerprint_hash": "ef" * 32,
        "last_admission_ordinal": 3,
    }
    with sqlite3.connect(root / "capture-v1.sqlite3") as connection:
        trigger = connection.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND sql LIKE ?",
            ("%erasure request authority is immutable%",),
        ).fetchone()
        connection.execute(f"DROP TRIGGER {trigger[0]}")
        connection.execute(f"UPDATE erasure_requests SET {field}=?", (values[field],))
        connection.execute(trigger[1])
    changed = observe_storage(root)
    row.update(before=changed, after=changed)
    with pytest.raises(ValueError):
        _validate_recovery("after_revoke_request_commit", "caught-up", row)


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
def test_final_authority_reader_does_not_call_candidate_parsers(tmp_path, monkeypatch) -> None:
    from hermes_realtime.evidence import storage_security
    from scripts.storage_observation import observe_storage
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(tmp_path)
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
    finally:
        spool.close()

    def refused(*args, **kwargs):
        raise AssertionError("the candidate cannot supply its own authority decoder")

    monkeypatch.setattr(storage_security, "decode_sentinel_image", refused)
    monkeypatch.setattr(storage_security, "parse_root_marker", refused)
    assert observe_storage(tmp_path / "evidence")["sentinel"] == "clear"


@pytest.mark.parametrize("offset", [8, 12, 25, 48, 232, 273, 296, 511])
def test_independent_sentinel_reader_rejects_corrupt_or_noncanonical_slots(offset) -> None:
    from hermes_realtime.evidence.storage_security import initial_sentinel_image
    from scripts.spool_crash_oracle import sentinel_state_v1

    image = initial_sentinel_image("50000000-0000-4000-8000-000000000002")
    assert sentinel_state_v1(image) == "first_create_pending"
    changed = bytearray(image)
    changed[offset] ^= 1
    with pytest.raises(ValueError):
        sentinel_state_v1(bytes(changed))


@pytest.mark.parametrize("key", ["before", "after"])
def test_final_sentinel_must_preserve_the_exact_transition_authority(key) -> None:
    import hashlib

    from hermes_realtime.evidence.models import SentinelState
    from hermes_realtime.evidence.storage_security import (
        initial_sentinel_image,
        next_sentinel_image,
    )
    from scripts.spool_crash_matrix import _validate_recovery

    row = _purge_observation()
    _validate_recovery("after_full_purge_absence_verify", "caught-up", row)
    image = initial_sentinel_image("50000000-0000-4000-8000-000000000002")
    image = next_sentinel_image(image, SentinelState.CLEAR)
    image = next_sentinel_image(
        image,
        SentinelState.FULL_PURGE_PENDING,
        state_generation_id="90000000-0000-4000-8000-000000000096",
    )
    if key == "after":
        image = next_sentinel_image(image, SentinelState.CLEAR)
    row[key]["files"]["capture-v1.owner"] = hashlib.sha256(image).hexdigest()
    with pytest.raises(ValueError):
        _validate_recovery("after_full_purge_absence_verify", "caught-up", row)


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize("binding_close", [False, True])
def test_checkpoint_cannot_substitute_another_valid_fixture_event(tmp_path, binding_close) -> None:
    from scripts.storage_observation import _database_state
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(tmp_path)
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
        result = (
            spool.append_binding_close(driver._binding_close())
            if binding_close
            else spool.append_record(driver._ordinary_record())
        )
        assert result.value == "committed"
    finally:
        spool.close()
    database = _database_state(tmp_path / "evidence/capture-v1.sqlite3")
    expected = "before_seal_commit" if binding_close else "after_event_commit"
    wrong = "after_event_commit" if binding_close else "before_seal_commit"
    _validate_checkpoint(expected, database)
    with pytest.raises(ValueError):
        _validate_checkpoint(wrong, database)


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize(
    "seconds,point,sealed",
    [
        (60, "after_seal_commit_before_ack", True),
        (1, "after_first_epoch_commit", False),
        (0, "after_recreate_epoch_commit", False),
    ],
)
def test_checkpoint_pins_the_absolute_source_clock(tmp_path, seconds, point, sealed) -> None:
    from datetime import timedelta

    from scripts.storage_observation import _database_state
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(
        tmp_path, clock=driver._Clock(driver.START + timedelta(seconds=seconds))
    )
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
        if sealed:
            assert spool.append_binding_close(driver._binding_close()).value == "committed"
            assert spool.seal_epoch(driver._seal_command()).value == "committed"
    finally:
        spool.close()
    observed = _database_state(tmp_path / "evidence/capture-v1.sqlite3")
    with pytest.raises(ValueError):
        _validate_checkpoint(point, observed)


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize("field", ["created_at_utc", "clock_high_water_utc"])
def test_checkpoint_pins_installation_clock_authority(tmp_path, field) -> None:
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
    _validate_checkpoint("after_seal_commit_before_ack", _database_state(database))
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"UPDATE producer_installation SET {field}=?", ("2026-08-08T00:00:03.000000Z",)
        )
    with pytest.raises(ValueError):
        _validate_checkpoint("after_seal_commit_before_ack", _database_state(database))


def test_empty_committed_schema_has_a_closed_wire_observation(tmp_path) -> None:
    import sqlite3

    from hermes_realtime.evidence.sqlite_spool import SCHEMA_DDL_V1
    from scripts.qualify_evidence_slice_zero import canonical_json_bytes
    from scripts.storage_observation import _database_state

    database = tmp_path / "capture-v1.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(SCHEMA_DDL_V1)
        connection.execute("PRAGMA application_id=0x48524531")
        connection.execute("PRAGMA user_version=1")
    observed = _database_state(database)
    assert observed["schema"] is True and observed["purge_required"] == -1
    assert canonical_json_bytes({"database": observed}).endswith(b"\n")


@pytest.mark.skipif(os.name != "nt", reason="creates a real Windows evidence store")
@pytest.mark.parametrize(
    "artifact", ["capture-v1.sqlite3-tmp", "capture-v1.sqlite3-vacuum", "capture-v1.owner.init"]
)
def test_recovered_store_rejects_remaining_owned_temporaries(tmp_path, artifact) -> None:
    from scripts.spool_crash_matrix import _validate_recovery
    from scripts.storage_observation import observe_storage
    from tests.evidence import spool_crash_worker as driver

    spool = driver._make_spool(tmp_path)
    try:
        assert spool.create_epoch(driver._make_create_epoch()).value == "committed"
        assert spool.append_binding_close(driver._binding_close()).value == "committed"
        assert spool.seal_epoch(driver._seal_command()).value == "committed"
    finally:
        spool.close()
    root = tmp_path / "evidence"
    before = observe_storage(root)
    row = {"before": before, "after": before, "disposition": "recovered", "baseline": {}}
    _validate_recovery("after_seal_commit_before_ack", "caught-up", row)
    (root / artifact).write_bytes(b"fixture residue")
    row["after"] = observe_storage(root)
    with pytest.raises(ValueError):
        _validate_recovery("after_seal_commit_before_ack", "caught-up", row)


@pytest.mark.skipif(os.name != "nt", reason="uses the real Windows crash driver")
@pytest.mark.parametrize("point", ["after_event_insert_before_commit", "before_seal_commit"])
@pytest.mark.parametrize("hot", [False, True])
def test_observer_preserves_the_original_crash_image_and_journal(tmp_path, point, hot) -> None:
    from scripts.storage_observation import _database_state
    from tests.evidence.test_sqlite_spool import run_task_5d_worker

    case = tmp_path / "crash"
    run_task_5d_worker(case, point, 197)
    root = case / "evidence"
    if hot:
        # These small transactions can leave an unflushed journal header. Mark
        # its complete, checksummed original-page records as a hot journal so
        # the observer must preserve a crash image SQLite would otherwise undo.
        journal = root / "capture-v1.sqlite3-journal"
        raw = journal.read_bytes()
        sector = int.from_bytes(raw[20:24], "big")
        page_size = int.from_bytes(raw[24:28], "big")
        assert sector > 0 and page_size == 4096
        count, remainder = divmod(len(raw) - sector, page_size + 8)
        assert count > 0 and remainder == 0
        journal.write_bytes(
            b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7" + count.to_bytes(4, "big") + raw[12:]
        )
    original = {p.name: p.read_bytes() for p in root.iterdir()}
    assert "capture-v1.sqlite3-journal" in original
    _validate_checkpoint(point, _database_state(root / "capture-v1.sqlite3"))
    assert {p.name: p.read_bytes() for p in root.iterdir()} == original


@pytest.mark.parametrize("mutation", ["trigger", "constraint", "unique", "page_size"])
def test_schema_commit_requires_the_fixed_complete_v1_schema(tmp_path, mutation) -> None:
    import sqlite3

    from hermes_realtime.evidence.sqlite_spool import SCHEMA_DDL_V1
    from scripts.storage_observation import _database_state

    database = tmp_path / "capture-v1.sqlite3"
    schema = SCHEMA_DDL_V1
    if mutation == "constraint":
        schema = schema.replace("CHECK (singleton = 1)", "CHECK (singleton >= 1)")
    elif mutation == "unique":
        schema = schema.replace(",\n    UNIQUE (logical_session_id, event_sequence)", "")
    with sqlite3.connect(database) as connection:
        if mutation == "page_size":
            connection.execute("PRAGMA page_size=8192")
        connection.executescript(schema)
        connection.execute("PRAGMA application_id=0x48524531")
        connection.execute("PRAGMA user_version=1")
        if mutation == "trigger":
            connection.execute("DROP TRIGGER evidence_events_are_append_only")
    with pytest.raises(ValueError):
        _database_state(database)


@pytest.mark.skipif(os.name != "nt", reason="uses the real Windows crash driver")
def test_full_purge_latch_observes_the_intact_pre_deletion_database(tmp_path) -> None:
    from scripts.storage_observation import observe_storage
    from tests.evidence.test_sqlite_spool import run_task_5d_worker

    case = tmp_path / "crash"
    run_task_5d_worker(case, "after_full_purge_marker_fsync", 197)
    state = observe_storage(case / "evidence")
    assert state["database"]["schema"] is True
    assert state["database"]["events"] == 2


@pytest.mark.skipif(os.name != "nt", reason="uses the real Windows crash driver")
@pytest.mark.parametrize("changed", [False, True])
def test_full_purge_latch_preserves_the_pre_operation_database_image(tmp_path, changed) -> None:
    import json

    from scripts.spool_crash_matrix import _validate_recovery
    from scripts.storage_observation import observe_storage
    from tests.evidence import spool_crash_worker as driver
    from tests.evidence.test_sqlite_spool import run_task_5d_worker

    case = tmp_path / "crash"
    point = "after_full_purge_marker_fsync"
    run_task_5d_worker(case, point, 197)
    root = case / "evidence"
    if changed:
        database = root / "capture-v1.sqlite3"
        image = bytearray(database.read_bytes())
        # Change only the SQLite file-change counter, retaining a valid image
        # with exactly the same schema and logical source history.
        counter = (int.from_bytes(image[24:28], "big") + 1).to_bytes(4, "big")
        image[24:28] = image[92:96] = counter
        database.write_bytes(image)
    before = observe_storage(root)
    _validate_checkpoint(point, before["database"])
    spool = driver._make_spool(case)
    try:
        disposition = spool.recover_existing().value
    finally:
        spool.close()
    row = {
        "before": before,
        "after": observe_storage(root),
        "disposition": disposition,
        "baseline": json.loads((case / "full-purge-baseline.json").read_text(encoding="utf-8")),
    }
    if changed:
        with pytest.raises(ValueError, match="full-purge latch changed"):
            _validate_recovery(point, "caught-up", row)
    else:
        _validate_recovery(point, "caught-up", row)
