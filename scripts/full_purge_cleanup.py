"""Independent acceptance of exact Windows full purge and fresh-process idempotence."""

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
from scripts.spool_crash_matrix import _validate_case_contract, _validate_checkpoint
from scripts.spool_crash_oracle import checkpoint_sentinel_digest_v1, initialization_digest_v1
from scripts.storage_process import (
    _run_storage_worker,
    _storage_archive,
    _StorageInvocation,
    _validate_invocation,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_MARKER = ".hermes-realtime-evidence-root-v1"
_SENTINEL = "capture-v1.owner"
_DATABASE = "capture-v1.sqlite3"
_DECOYS = {
    "purge-decoy.bin": b"not-owned",
    "capture-v1.sqlite3.backup": b"database-prefix-decoy",
    "capture-v1.sqlite3-wal.backup": b"sidecar-prefix-decoy",
}
_ADJACENT = {"capture-v1.sqlite3": b"adjacent-database-decoy", "keep.bin": b"adjacent-decoy"}


def _digests(values: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in values.items()}


def _validate_observations(rows: Any) -> None:
    _require(type(rows) is list and len(rows) == 2, "full purge requires both fresh-process phases")
    for phase, row in zip(("purge", "repeat_purge"), rows, strict=True):
        _keys(row, {"phase", "before", "result", "after"})
        _require(row["phase"] == phase, "full-purge phase order differs")
        _require(
            row["result"] == ("purge_completed" if phase == "purge" else "already_absent"),
            "full-purge disposition differs",
        )
        for state in (row["before"], row["after"]):
            _keys(state, {"files", "sentinel", "database", "adjacent"})
            _require(state["sentinel"] == "clear", "full purge left pending authority")
            _require(state["adjacent"] == _digests(_ADJACENT), "adjacent decoys changed")
            _require(type(state["files"]) is dict, "full-purge inventory is missing")
            _require(
                all(
                    type(v) is str and re.fullmatch("[0-9a-f]{64}", v)
                    for v in state["files"].values()
                ),
                "full-purge file commitment differs",
            )
    before, after = rows[0]["before"], rows[0]["after"]
    _validate_checkpoint("before_begin", before["database"])
    db = before["database"]
    _keys(
        db,
        {
            "schema",
            "events",
            "sessions",
            "epochs",
            "source_digest",
            "installation_digest",
            "purge_required",
            "seals",
            "tombstones",
            "erasures",
            "logical_digest",
        },
    )
    _require(
        db["schema"] is True
        and type(db["purge_required"]) is int
        and db["purge_required"] == 0
        and db["seals"] == db["tombstones"] == db["erasures"] == [],
        "full-purge initial database differs",
    )
    _require(
        type(db["logical_digest"]) is str
        and re.fullmatch("[0-9a-f]{64}", db["logical_digest"]) is not None,
        "full-purge logical commitment differs",
    )
    retained = {_MARKER: initialization_digest_v1("root_marker", "full_write"), **_digests(_DECOYS)}
    initial = {
        **retained,
        _SENTINEL: checkpoint_sentinel_digest_v1(0, "caught-up", after=False),
        _DATABASE: before["files"].get(_DATABASE),
        **{
            _DATABASE + suffix: hashlib.sha256(b"owned-sidecar").hexdigest()
            for suffix in ("-journal", "-wal", "-shm", "-vacuum", "-tmp")
        },
    }
    _require(
        before["files"] == initial and initial[_DATABASE] is not None,
        "full-purge initial artifact inventory differs",
    )
    final = {**retained, _SENTINEL: checkpoint_sentinel_digest_v1(40, "caught-up", after=True)}
    _require(
        after["files"] == final and after["database"] == {"schema": False},
        "full purge did not remove exactly its database artifacts",
    )
    _require(rows[1]["before"] == after == rows[1]["after"], "repeated full purge mutated storage")


class ObservedFullPurgeV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("full-purge receipts are producer-minted only")


@dataclass(frozen=True, slots=True)
class _PurgeRun:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256: str
    invocations: tuple[_StorageInvocation, ...]


_RUNS: WeakKeyDictionary[ObservedFullPurgeV1, _PurgeRun] = WeakKeyDictionary()


def _validate_run(run: _PurgeRun) -> PackagedScenarioEvidenceV1:
    _require(len(run.invocations) == 2, "full-purge process count differs")
    identities = {
        (r.process.identity.pid, r.process.identity.creation_filetime) for r in run.invocations
    }
    _require(len(identities) == 2, "full purge reused a process")
    _require(
        all(r.expected_exit == 0 and r.storage_released is True for r in run.invocations),
        "full purge lacks normal exit or live release evidence",
    )
    rows = [_validate_invocation(r) for r in run.invocations]
    _validate_observations(rows)
    return PackagedScenarioEvidenceV1(
        run.source_commit,
        run.source_tree,
        run.source_archive_sha256,
        run.wheel_sha256,
        hashlib.sha256(core.canonical_json_bytes(rows)).hexdigest(),
        2,
        ("adjacent_decoys_preserved", "purge_verified", "purged"),
    )


def produce_full_purge_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
) -> ObservedFullPurgeV1:
    with _storage_archive(archive, identity, wheel) as owned:
        # The complete reviewed report image also pins the full-purge assertion
        # set, proof class, and absence of case/measurement fields.
        _validate_case_contract(
            (owned.source / "scripts/schemas/qualification-report-v1.schema.json").read_bytes()
        )
        invocations = tuple(
            _run_storage_worker(
                owned,
                point="full_purge_cleanup",
                mode=0,
                action=phase,
            )
            for phase in ("purge", "repeat_purge")
        )
        run = _PurgeRun(
            identity.candidate_head_oid,
            identity.candidate_tree_oid,
            owned.metadata.archive_sha256,
            owned.wheel_sha256,
            invocations,
        )
        _validate_run(run)
    receipt = object.__new__(ObservedFullPurgeV1)
    _RUNS[receipt] = run
    return receipt


def validate_full_purge_v1(receipt: ObservedFullPurgeV1) -> PackagedScenarioEvidenceV1:
    if type(receipt) is not ObservedFullPurgeV1:
        raise TypeError("full-purge receipt type differs")
    _require(receipt in _RUNS, "full-purge receipt is unregistered")
    return _validate_run(_RUNS[receipt])
