"""Independent acceptance of all governed packaged-spool crash boundaries."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from scripts import qualify_evidence_slice_zero as core
from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.candidate_wheel import VerifiedCandidateWheelV1
from scripts.deterministic_equivalence import _keys, _require
from scripts.packaged_scenario import PackagedScenarioEvidenceV1
from scripts.spool_crash_oracle import (
    checkpoint_installation_digest_v1,
    checkpoint_sentinel_digest_v1,
    checkpoint_source_digest_v1,
    erasure_receipt_digest_v1,
    erasure_request_digest_v1,
    initialization_digest_v1,
)
from scripts.storage_process import (
    _run_storage_worker,
    _storage_archive,
    _StorageInvocation,
    _validate_invocation,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

# These names are independently pinned by SpoolCrashCaseV1 in the governed
# report schema. The archived driver must match them before any process runs.
FAILPOINTS_V1 = (
    "before_begin",
    "after_event_insert_before_commit",
    "after_event_commit",
    "before_seal_commit",
    "after_seal_commit_before_ack",
    "after_revoke_request_commit",
    "after_logical_purge_before_vacuum",
    "after_vacuum_before_ack",
    "after_drain_ack_before_exit",
    "after_root_marker_init_create",
    "after_root_marker_init_partial_write",
    "after_root_marker_init_full_write",
    "after_root_marker_init_flush",
    "after_root_marker_activation",
    "after_sentinel_init_create",
    "after_sentinel_init_partial_write",
    "after_sentinel_init_full_write",
    "after_sentinel_init_flush",
    "after_sentinel_activation",
    "after_first_db_create",
    "after_first_schema_commit",
    "after_first_epoch_commit",
    "after_first_marker_clear",
    "after_recreate_pending_fsync",
    "after_recreate_db_create",
    "after_recreate_schema_commit",
    "after_recreate_epoch_commit",
    "after_recreate_marker_clear",
    "before_rollback_sentinel_write",
    "after_rollback_sentinel_fsync",
    "before_rollback_db_latch",
    "after_rollback_db_latch",
    "after_full_purge_marker_fsync",
    "after_full_purge_db_delete",
    "after_full_purge_journal_delete",
    "after_full_purge_wal_delete",
    "after_full_purge_shm_delete",
    "after_full_purge_vacuum_delete",
    "after_full_purge_tmp_delete",
    "after_full_purge_absence_verify",
    "after_full_purge_marker_clear",
)

_ORDINARY: tuple[tuple[int, list[str], list[str]], ...] = (
    (2, ["open"], ["active"]),
    (2, ["open"], ["active"]),
    (3, ["open"], ["active"]),
    (3, ["open"], ["active"]),
    (4, ["sealed"], ["closed"]),
    (2, ["open"], ["revoked"]),
    (0, [], []),
    (0, [], []),
    (4, ["sealed"], ["closed"]),
)
_FIRST = (
    "faulted",
    "faulted",
    "absent",
    "absent",
    "absent",
    "faulted",
    "faulted",
    "absent",
    "absent",
    "purge_completed",
    "purge_completed",
    "purge_completed",
    "purge_completed",
    "recovered",
)
_DATABASE_NAMES = tuple(
    "capture-v1.sqlite3" + suffix for suffix in ("", "-journal", "-wal", "-shm", "-vacuum", "-tmp")
)
_MARKER = ".hermes-realtime-evidence-root-v1"
_SENTINEL = "capture-v1.owner"


def _entry_sentinel(index: int) -> str:
    if 9 <= index <= 17:
        return "no_final_sentinel"
    if 18 <= index <= 21 or 23 <= index <= 26:
        return "first_create_pending"
    if 29 <= index <= 31:
        return "clock_rollback_purge_pending"
    if 32 <= index <= 39:
        return "full_purge_pending"
    return "clear"


def _validate_case_contract(schema: bytes) -> None:
    # Pin the complete reviewed V1 contract, including every per-case field and
    # transitive definition. A schema change requires an explicit producer review.
    _require(
        type(schema) is bytes
        and hashlib.sha256(schema).hexdigest()
        == "e3c8c509612bba844d9329202beb83bee7cbdf6b7a3d98501e2511be9da3e399",
        "governed spool case contract differs",
    )


def _validate_checkpoint(point: str, db: Any) -> None:
    index = FAILPOINTS_V1.index(point)
    if index < 9:
        expected = _ORDINARY[index]
    elif 28 <= index < 32:
        expected = (6, ["open", "sealed"], ["active", "closed"])
    elif index in {21, 22, 26, 27, 32}:
        expected = (2, ["open"], ["active"])
    elif index in {20, 25}:
        expected = (0, [], [])
        _require(
            type(db) is dict and db.get("schema") is True and db.get("purge_required") == -1,
            "schema checkpoint has no committed empty schema",
        )
    else:
        _require(db == {"schema": False}, "checkpoint unexpectedly opened a database")
        return
    _require(
        type(db) is dict
        and type(db.get("events")) is int
        and (db.get("events"), db.get("sessions"), db.get("epochs")) == expected,
        "durable checkpoint history differs",
    )
    _require(
        db.get("source_digest") == checkpoint_source_digest_v1(index),
        "checkpoint source history differs",
    )
    _require(
        db.get("installation_digest") == checkpoint_installation_digest_v1(index),
        "checkpoint installation authority differs",
    )


def _validate_recovery(point: str, clock: str, row: Any) -> None:
    _keys(row, {"before", "disposition", "after", "baseline"})
    before, after = row["before"], row["after"]
    for state in (before, after):
        _keys(state, {"files", "sentinel", "database"})
        _require(type(state["files"]) is dict, "storage inventory is absent")
        allowed = {
            _MARKER,
            _SENTINEL,
            _MARKER + ".init",
            _SENTINEL + ".init",
            *_DATABASE_NAMES,
            "recreation-decoy.bin",
            "rollback-decoy.bin",
            "purge-decoy.bin",
        }
        _require(set(state["files"]) <= allowed, "storage inventory names differ")
        for digest in state["files"].values():
            _require(
                type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None,
                "storage inventory digest differs",
            )
        db = state["database"]
        if db == {"schema": False}:
            _require(db["schema"] is False, "storage schema flag is not boolean")
        else:
            _keys(
                db,
                {
                    "schema",
                    "events",
                    "sessions",
                    "epochs",
                    "seals",
                    "source_digest",
                    "installation_digest",
                    "purge_required",
                    "tombstones",
                    "erasures",
                    "logical_digest",
                },
            )
            _require(
                db["schema"] is True and type(db["events"]) is int and 0 <= db["events"] <= 8,
                "storage event bound differs",
            )
            _require(
                type(db["purge_required"]) is int and db["purge_required"] in {-1, 0, 1},
                "storage purge latch differs",
            )
            for name, options in (
                ("sessions", {"open", "sealed"}),
                ("epochs", {"active", "closed", "revoked"}),
            ):
                _require(
                    type(db[name]) is list
                    and len(db[name]) <= 2
                    and all(type(v) is str and v in options for v in db[name]),
                    "storage state values differ",
                )
            _require(
                type(db["erasures"]) is list and len(db["erasures"]) <= 1,
                "storage erasure request count differs",
            )
            for request in db["erasures"]:
                _require(
                    type(request) is list
                    and len(request) == 2
                    and request[0] in {"pending", "logical_deleted"}
                    and type(request[1]) is str
                    and re.fullmatch("[0-9a-f]{64}", request[1]) is not None,
                    "storage erasure request authority differs",
                )
            _require(
                type(db["seals"]) is list and len(db["seals"]) == db["sessions"].count("sealed"),
                "storage seal count differs",
            )
            for digest in [db["logical_digest"], db["source_digest"], *db["seals"]]:
                _require(
                    type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None,
                    "storage history digest differs",
                )
            _require(
                type(db["installation_digest"]) is str
                and re.fullmatch("[0-9a-f]{64}", db["installation_digest"]) is not None,
                "storage installation digest differs",
            )
            _require(
                type(db["tombstones"]) is list and len(db["tombstones"]) <= 2,
                "storage tombstone count differs",
            )
            for tombstone in db["tombstones"]:
                _require(
                    type(tombstone) is list
                    and len(tombstone) == 4
                    and tombstone[0] in {"revoked", "unclean_epoch", "clock_rollback", "ttl"}
                    and all(type(n) is int and 0 <= n <= 8 for n in tombstone[1:3])
                    and type(tombstone[3]) is str
                    and re.fullmatch("[0-9a-f]{64}", tombstone[3]) is not None,
                    "storage tombstone values differ",
                )
    index = FAILPOINTS_V1.index(point)
    _require(before["sentinel"] == _entry_sentinel(index), "recovery entry sentinel differs")
    _validate_checkpoint(point, before["database"])
    if before["database"]["schema"]:
        request_state = "pending" if index == 5 else "logical_deleted" if index in {6, 7} else None
        expected_requests = (
            []
            if request_state is None
            else [[request_state, erasure_request_digest_v1(request_state)]]
        )
        _require(
            before["database"]["erasures"] == expected_requests
            and before["database"]["tombstones"] == [],
            "checkpoint erasure authority differs",
        )
    if index in {19, 24}:
        _require(
            before["files"].get(_DATABASE_NAMES[0]) == hashlib.sha256(b"").hexdigest(),
            "database-create checkpoint did not retain an empty file",
        )
    if 28 <= index < 32:
        _keys(row["baseline"], {"databaseSha256", "sentinelSha256"})
        for digest in row["baseline"].values():
            _require(
                type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None,
                "rollback baseline digest differs",
            )
        _require(
            before["database"]["purge_required"] == int(index == 31),
            "rollback durable latch differs",
        )
        if index <= 30:
            _require(
                before["database"]["logical_digest"] == row["baseline"]["databaseSha256"],
                "pre-latch rollback changed the original database",
            )
        if index == 28:
            _require(
                before["files"][_SENTINEL] == row["baseline"]["sentinelSha256"],
                "pre-sentinel rollback changed the original durable authority",
            )
    elif index == 32:
        _keys(row["baseline"], {"databaseImageSha256"})
        _require(
            before["files"].get(_DATABASE_NAMES[0]) == row["baseline"]["databaseImageSha256"],
            "full-purge latch changed the original database image",
        )
    else:
        _require(row["baseline"] == {}, "unexpected storage baseline")
    if index >= 32:
        deleted = min(index - 32, 6)
        _require(
            set(before["files"])
            == {_MARKER, _SENTINEL, "purge-decoy.bin", *_DATABASE_NAMES[deleted:]},
            "full-purge checkpoint deletion prefix differs",
        )
    elif index < 9:
        expected_files = {_MARKER, _SENTINEL, _DATABASE_NAMES[0]}
        if index in {1, 3}:
            expected_files.add(_DATABASE_NAMES[1])
        _require(set(before["files"]) == expected_files, "ordinary checkpoint inventory differs")
    elif index == 13:
        _require(set(before["files"]) == {_MARKER}, "marker checkpoint inventory differs")
    elif index >= 18:
        expected_files = {_MARKER, _SENTINEL}
        if index not in {18, 23}:
            expected_files.add(_DATABASE_NAMES[0])
        if 23 <= index <= 27:
            expected_files.add("recreation-decoy.bin")
        if 28 <= index <= 31:
            expected_files.add("rollback-decoy.bin")
        _require(set(before["files"]) == expected_files, "creation checkpoint inventory differs")
    if 9 <= index <= 12 or 14 <= index <= 17:
        temporary = _MARKER + ".init" if index <= 12 else _SENTINEL + ".init"
        expected_files = {temporary} | ({_MARKER} if index >= 14 else set())
        _require(
            set(before["files"]) == expected_files, "initialization checkpoint inventory differs"
        )
        _require(
            before["files"][temporary]
            == initialization_digest_v1(
                "root_marker" if index <= 12 else "sentinel",
                point.partition("_init_")[2],
            ),
            "initialization checkpoint write boundary differs",
        )
    disposition = (
        "faulted"
        if index == 5
        else "recovered"
        if index < 9
        else _FIRST[index - 9]
        if index < 23
        else ("recovered" if index == 27 else "purge_completed")
        if index < 28
        else ("recovered" if index == 28 and clock == "caught-up" else "purge_completed")
        if index < 32
        else "absent"
        if index == 40
        else "purge_completed"
    )
    _require(row["disposition"] == disposition, "storage recovery disposition differs")
    for name in ("recreation-decoy.bin", "rollback-decoy.bin", "purge-decoy.bin"):
        if name in before["files"]:
            _require(
                after["files"].get(name)
                == before["files"][name]
                == hashlib.sha256(b"not-owned").hexdigest(),
                "storage recovery changed an adjacent decoy",
            )
    if _MARKER in before["files"]:
        _require(
            after["files"].get(_MARKER) == before["files"][_MARKER],
            "storage recovery changed root identity",
        )
    if disposition == "purge_completed" or index == 40:
        retained = {_MARKER, _SENTINEL} | {
            name for name in before["files"] if name.endswith("-decoy.bin")
        }
        _require(
            set(after["files"]) == retained
            and after["database"] == {"schema": False}
            and after["sentinel"] == "clear"
            and _MARKER in after["files"]
            and _SENTINEL in after["files"],
            "storage purge absence was not independently verified",
        )
    elif disposition == "recovered":
        retained = {_MARKER, _SENTINEL, _DATABASE_NAMES[0]} | {
            name for name in before["files"] if name.endswith("-decoy.bin")
        }
        _require(set(after["files"]) == retained, "recovered store retained temporary artifacts")
        db = after["database"]
        _require(type(db) is dict, "recovered database is absent")
        _require(
            db["purge_required"] == 0 and db["erasures"] == [] and after["sentinel"] == "clear",
            "recovered database still requires purge",
        )
        sealed = index in {4, 8, 28}
        _require(
            db["installation_digest"] == checkpoint_installation_digest_v1(index, after=True),
            "recovered installation authority differs",
        )
        _require(
            db["source_digest"] == checkpoint_source_digest_v1(4 if sealed else 6),
            "recovered source history differs",
        )
        _require(
            (db["events"], db["sessions"], db["epochs"])
            == ((4, ["sealed"], ["closed"]) if sealed else (0, [], [])),
            "storage recovery retained an unclean session or lost a sealed epoch",
        )
        _require(
            db["seals"] == (before["database"]["seals"] if sealed else []),
            "recovery changed a revalidated seal",
        )
        tombstones: list[tuple[str, int, int]] = (
            []
            if index in {4, 8}
            else [("revoked", 1, 2)]
            if index in {6, 7}
            else [("unclean_epoch", 1, 2 if index in {22, 27, 28} else _ORDINARY[index][0])]
        )
        expected_receipts = [
            [
                reason,
                sessions,
                events,
                erasure_receipt_digest_v1(reason, events, rollback=index == 28),
            ]
            for reason, sessions, events in tombstones
        ]
        _require(db["tombstones"] == expected_receipts, "storage erasure receipt differs")
    elif disposition == "faulted":
        _require(before == after, "refused recovery mutated durable state")
    else:
        _require(
            set(after["files"]) == ({_MARKER} if index >= 13 else set())
            and after["database"] == {"schema": False}
            and after["sentinel"] == "no_final_sentinel",
            "absent recovery retained initialization artifacts",
        )
    for state, recovered in ((before, False), (after, True)):
        if _MARKER in state["files"]:
            _require(
                state["files"][_MARKER] == initialization_digest_v1("root_marker", "full_write"),
                "checkpoint root marker bytes differ",
            )
        _require(
            state["files"].get(_SENTINEL)
            == checkpoint_sentinel_digest_v1(index, clock, after=recovered),
            "checkpoint sentinel authority bytes differ",
        )


@dataclass(frozen=True, slots=True)
class _MatrixRun:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256: str
    invocations: tuple[tuple[str, int, str, _StorageInvocation, _StorageInvocation], ...]


class ObservedSpoolCrashMatrixV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("spool crash receipts are producer-minted only")


_RUNS: WeakKeyDictionary[ObservedSpoolCrashMatrixV1, _MatrixRun] = WeakKeyDictionary()


def produce_spool_crash_matrix_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
) -> ObservedSpoolCrashMatrixV1:
    invocations = []
    with _storage_archive(archive, identity, wheel) as owned:
        _validate_case_contract(
            (owned.source / "scripts/schemas/qualification-report-v1.schema.json").read_bytes()
        )
        for point in FAILPOINTS_V1:
            for mode in (197, 198):
                clocks = (
                    ("caught-up", "regressed") if point in FAILPOINTS_V1[28:32] else ("caught-up",)
                )
                for clock in clocks:
                    crash = _run_storage_worker(
                        owned, point=point, mode=mode, action="crash", clock=clock
                    )
                    recovery = _run_storage_worker(
                        owned, point=point, mode=mode, action="recover", clock=clock
                    )
                    invocations.append((point, mode, clock, crash, recovery))
        record = _MatrixRun(
            owned.metadata.candidate_head_oid,
            owned.metadata.candidate_tree_oid,
            owned.metadata.archive_sha256,
            owned.wheel_sha256,
            tuple(invocations),
        )
        _validate_matrix(record)
    receipt = object.__new__(ObservedSpoolCrashMatrixV1)
    _RUNS[receipt] = record
    return receipt


def validate_spool_crash_matrix_v1(
    receipt: ObservedSpoolCrashMatrixV1,
) -> PackagedScenarioEvidenceV1:
    if type(receipt) is not ObservedSpoolCrashMatrixV1:
        raise TypeError("spool crash receipt type differs")
    _require(receipt in _RUNS, "spool crash receipt is unregistered")
    return _validate_matrix(_RUNS[receipt])


def _validate_matrix(record: _MatrixRun) -> PackagedScenarioEvidenceV1:
    expected = [
        (point, mode, clock)
        for point in FAILPOINTS_V1
        for mode in (197, 198)
        for clock in (
            ("caught-up", "regressed") if point in FAILPOINTS_V1[28:32] else ("caught-up",)
        )
    ]
    _require(
        [(p, m, c) for p, m, c, _, _ in record.invocations] == expected,
        "spool crash matrix is incomplete or reordered",
    )
    observations = []
    identities = set()
    for point, mode, clock, crash, recovery in record.invocations:
        for invocation in (crash, recovery):
            identity = (
                invocation.process.identity.pid,
                invocation.process.identity.creation_filetime,
            )
            _require(identity not in identities, "spool crash matrix reused a process")
            identities.add(identity)
        _require(
            crash.expected_exit == mode and recovery.expected_exit == 0,
            "spool crash exit authority differs",
        )
        checkpoint = _validate_invocation(crash)
        _keys(checkpoint, {"checkpoint", "vacuum_returned"})
        _require(
            checkpoint
            == {"checkpoint": point, "vacuum_returned": point == "after_vacuum_before_ack"},
            "spool crash checkpoint differs",
        )
        row = _validate_invocation(recovery)
        _validate_recovery(point, clock, row)
        observations.append(
            {
                "point": point,
                "mode": mode,
                "clock": clock,
                "recovery": row,
                "drain_released": crash.storage_released,
            }
        )
    return PackagedScenarioEvidenceV1(
        record.source_commit,
        record.source_tree,
        record.source_archive_sha256,
        record.wheel_sha256,
        hashlib.sha256(core.canonical_json_bytes(observations)).hexdigest(),
        len(identities),
        (
            "partial_seal_absent",
            "postcommit_seal_revalidated",
            "precommit_session_unsealed",
            "purge_verified",
        ),
    )
