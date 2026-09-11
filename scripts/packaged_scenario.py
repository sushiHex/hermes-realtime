"""Shared source/wheel attestation and owned-exit checks for packaged producers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import qualify_evidence_slice_zero as core
from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.candidate_wheel import VerifiedCandidateWheelV1, _wheel_for_consumer
from scripts.task13_artifact_orchestrator import CandidateIdentityV1


@dataclass(frozen=True, slots=True)
class PackagedScenarioEvidenceV1:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256: str
    observation_sha256: str
    process_count: int
    machine_assertions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ObservedRun:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256: str
    observations: bytes
    processes: tuple[core._WindowsBoundProcessV1, ...]
    cleanup: core._WindowsFinalizationResultV1
    exit_code: int


def _observe_packaged_run(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
    *,
    scenario: str,
    livekit_executable: Path,
    livekit_sha256: str,
) -> _ObservedRun:
    from scripts.equivalence_process import _run_archived_scenario

    if scenario not in {
        "revoke_race", "capacity_rollover", "over_budget_turn", "owned_close_faults",
    }:
        raise ValueError("packaged scenario is unavailable")
    package = _wheel_for_consumer(wheel, archive, identity)
    metadata, observations, processes, cleanup, exit_code = _run_archived_scenario(
        archive,
        identity,
        scenario=scenario,
        wheel=wheel,
        livekit_executable=livekit_executable,
        livekit_sha256=livekit_sha256,
    )
    return _ObservedRun(
        metadata.candidate_head_oid,
        metadata.candidate_tree_oid,
        metadata.archive_sha256,
        package.wheel_sha256,
        observations,
        processes,
        cleanup,
        exit_code,
    )


def _validate_packaged_run(record: _ObservedRun, *, scenario: str) -> Any:
    if type(record.exit_code) is not int or record.exit_code != 0:
        raise ValueError("packaged worker did not exit successfully")
    cleanup = record.cleanup
    if not (
        cleanup.closed
        and cleanup.zero_active_observed
        and not cleanup.failures
        and not cleanup.failed_handles
    ):
        raise ValueError("owned packaged cleanup is incomplete")
    if not record.processes or not all(
        process.process_handle in cleanup.waited_handles for process in record.processes
    ):
        raise ValueError("retained packaged processes were not all waited")
    rows = core.load_strict_canonical_json(record.observations, source="packaged observations")
    if type(rows) is not list or len(rows) != 1:
        raise ValueError("packaged scenario is incomplete")
    row = rows[0]
    if type(row) is not dict or row.get("arm") != scenario:
        raise ValueError("packaged scenario identity differs")
    return row


def _evidence(record: _ObservedRun, assertions: tuple[str, ...]) -> PackagedScenarioEvidenceV1:
    return PackagedScenarioEvidenceV1(
        record.source_commit,
        record.source_tree,
        record.source_archive_sha256,
        record.wheel_sha256,
        hashlib.sha256(record.observations).hexdigest(),
        len(record.processes),
        assertions,
    )
