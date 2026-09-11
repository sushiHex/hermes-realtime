"""Independent acceptance of five real, bounded synthetic fault observations."""

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
from scripts.evidence_protocol_oracle import canonical_json_bytes
from scripts.packaged_scenario import PackagedScenarioEvidenceV1
from scripts.spool_crash_matrix import _validate_case_contract
from scripts.spool_crash_oracle import initialization_digest_v1
from scripts.storage_process import (
    _run_storage_worker,
    _storage_archive,
    _StorageInvocation,
    _validate_invocation,
)
from scripts.synthetic_fault_oracle import (
    ADJACENT,
    CASES_V1,
    DATABASE_NAMES,
    DECOYS,
    MARKER,
    SENTINEL,
    digest,
    expected_history,
    rejected_snapshot,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

ASSERTIONS_V1 = (
    "aggregate_byte_capacity_unreachable",
    "configured_capacity_algebra_equal",
    "physical_capacity_unreachable_before_record_capacity",
    "purge_verified",
    "purged",
    "record_capacity_rejected",
    "rejected_source_absent",
)
LIMITS_V1 = {
    "maxQueueRecords": 64,
    "maxQueueCanonicalBytes": 2097152,
    "maxQueuePhysicalItems": 64,
    "maxCanonicalRecordBytes": 32768,
}


def _same(actual: Any, expected: Any, label: str) -> None:
    # Canonical byte comparison keeps nested booleans distinct from integers.
    _require(core.canonical_json_bytes(actual) == core.canonical_json_bytes(expected), label)


def _state(count: int, case: str = "") -> dict[str, Any]:
    denied = case == "deny_filter"
    return {
        "events": count,
        "source_sha256": digest(canonical_json_bytes(expected_history(count))),
        "state": "tainted" if denied else "open",
        "taint": "deny_filter" if denied else "none",
        "purge_required": 1 if case == "clock_rollback" else 0,
    }


def _validate_cleanup(value: Any, case: str) -> None:
    _keys(value, {"before", "result", "after"})
    _same(value["result"], "purge_completed", "synthetic purge did not complete")
    for phase in ("before", "after"):
        state = value[phase]
        _keys(state, {"files", "sentinel", "adjacent"})
        _require(type(state["files"]) is dict, "synthetic inventory is missing")
        _require(
            all(
                type(v) is str and re.fullmatch("[0-9a-f]{64}", v) for v in state["files"].values()
            ),
            "synthetic file commitment differs",
        )
        _same(
            state["adjacent"],
            {name: digest(raw) for name, raw in ADJACENT.items()},
            "synthetic adjacent files changed",
        )
        expected = {
            MARKER: initialization_digest_v1("root_marker", "full_write"),
            SENTINEL: state["files"].get(SENTINEL),
            **{name: digest(raw) for name, raw in DECOYS.items()},
        }
        if phase == "before":
            expected.update(
                {
                    DATABASE_NAMES[0]: state["files"].get(DATABASE_NAMES[0]),
                    **{name: digest(b"synthetic-sidecar") for name in DATABASE_NAMES[1:]},
                }
            )
        _require(
            all(v is not None for v in expected.values()), "synthetic required artifact is absent"
        )
        _same(state["files"], expected, "synthetic purge inventory differs")
        sentinel = (
            "clock_rollback_purge_pending"
            if phase == "before" and case == "clock_rollback"
            else "clear"
        )
        _same(state["sentinel"], sentinel, "synthetic purge authority differs")


def _validate_observations(rows: Any) -> dict[str, int]:
    _require(type(rows) is list and len(rows) == 5, "synthetic matrix requires five cases")
    queue_bytes = sum(len(canonical_json_bytes(row)) for row in expected_history(68)[4:])
    _require(
        LIMITS_V1["maxQueueCanonicalBytes"]
        == LIMITS_V1["maxQueueRecords"] * LIMITS_V1["maxCanonicalRecordBytes"],
        "synthetic capacity algebra differs",
    )
    _require(
        0 < queue_bytes < LIMITS_V1["maxQueueCanonicalBytes"],
        "synthetic byte capacity is not unreachable",
    )
    for row, case in zip(rows, CASES_V1, strict=True):
        _keys(row, {"caseId", "before", "fault", "after", "scan", "cleanup"})
        _same(row["caseId"], case, "synthetic case order differs")
        _same(
            row["before"],
            _state(4 if case == "queue_capacity_coupled" else 3),
            "synthetic initial source differs",
        )
        _same(
            row["after"],
            _state(68 if case == "queue_capacity_coupled" else 3, case),
            "synthetic fault persistence differs",
        )
        expected_fault: dict[str, Any]
        if case == "queue_capacity_coupled":
            expected_fault = {
                "source_sha256": digest(canonical_json_bytes(rejected_snapshot(case))),
                "reason": "record_capacity",
                "before": [64, queue_bytes, 64],
                "after": [64, queue_bytes, 64],
                "drained": [0, 0, 0],
                "decisions": ["committed"] * 64,
                "limits": LIMITS_V1,
            }
        elif case == "writer_drain_blocked":
            expected_fault = {
                "stages": [
                    "drain_entered",
                    "pending_observed",
                    "drain_released",
                    "drain_stopped",
                    "joined",
                ],
                "outcomes": ["stopped"],
            }
        else:
            expected_fault = {
                "result": "rejected_state" if case == "deny_filter" else "faulted",
                "source_sha256": digest(canonical_json_bytes(rejected_snapshot(case))),
                "sticky_fault": "none"
                if case == "deny_filter"
                else "purge_required"
                if case == "clock_rollback"
                else "sqlite_fault",
                "injections": 1 if case == "sqlite_injected_fault" else 0,
                "in_transaction": False,
            }
        _same(row["fault"], expected_fault, "synthetic injection or capacity observation differs")
        _same(
            row["scan"],
            {} if case == "writer_drain_blocked" else {"files": 6, "matches": 0},
            "synthetic rejected source is not absent",
        )
        _validate_cleanup(row["cleanup"], case)
    return {
        **LIMITS_V1,
        "queueRecordCount": 64,
        "queueCanonicalBytes": queue_bytes,
        "queuePhysicalCount": 64,
    }


class ObservedSyntheticFaultV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("synthetic fault receipts are producer-minted only")


@dataclass(frozen=True, slots=True)
class _FaultRun:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256: str
    invocations: tuple[_StorageInvocation, ...]


_RUNS: WeakKeyDictionary[ObservedSyntheticFaultV1, _FaultRun] = WeakKeyDictionary()


def _validate_run(run: _FaultRun) -> PackagedScenarioEvidenceV1:
    _require(len(run.invocations) == 5, "synthetic process count differs")
    _require(
        len(
            {
                (r.process.identity.pid, r.process.identity.creation_filetime)
                for r in run.invocations
            }
        )
        == 5,
        "synthetic cases reused a process",
    )
    _require(
        all(r.expected_exit == 0 and r.storage_released is True for r in run.invocations),
        "synthetic exit or live release is incomplete",
    )
    rows = [_validate_invocation(r) for r in run.invocations]
    _validate_observations(rows)
    return PackagedScenarioEvidenceV1(
        run.source_commit,
        run.source_tree,
        run.source_archive_sha256,
        run.wheel_sha256,
        hashlib.sha256(core.canonical_json_bytes(rows)).hexdigest(),
        5,
        ASSERTIONS_V1,
    )


def produce_synthetic_fault_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
) -> ObservedSyntheticFaultV1:
    with _storage_archive(archive, identity, wheel) as owned:
        _validate_case_contract(
            (owned.source / "scripts/schemas/qualification-report-v1.schema.json").read_bytes()
        )
        invocations = tuple(
            _run_storage_worker(owned, point=case, mode=0, action="synthetic") for case in CASES_V1
        )
        run = _FaultRun(
            identity.candidate_head_oid,
            identity.candidate_tree_oid,
            owned.metadata.archive_sha256,
            owned.wheel_sha256,
            invocations,
        )
        _validate_run(run)
    receipt = object.__new__(ObservedSyntheticFaultV1)
    _RUNS[receipt] = run
    return receipt


def validate_synthetic_fault_v1(receipt: ObservedSyntheticFaultV1) -> PackagedScenarioEvidenceV1:
    if type(receipt) is not ObservedSyntheticFaultV1:
        raise TypeError("synthetic fault receipt type differs")
    _require(receipt in _RUNS, "synthetic fault receipt is unregistered")
    return _validate_run(_RUNS[receipt])
