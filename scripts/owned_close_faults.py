"""Derive conversation equality while preserving failed and cancelled close callers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.candidate_wheel import VerifiedCandidateWheelV1
from scripts.capacity_rollover import _digest
from scripts.deterministic_equivalence import _CONVERSATION_KINDS, _expected_close, _keys, _require
from scripts.packaged_scenario import (
    PackagedScenarioEvidenceV1,
    _evidence,
    _observe_packaged_run,
    _ObservedRun,
    _validate_packaged_run,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

ARMS_V1 = (
    "ordinary_disabled", "ordinary_consented",
    "retry_disabled", "retry_consented",
    "cancelled_disabled", "cancelled_consented", "perturbed",
)


def _validate_observations(row: Any) -> None:
    _keys(row, {"arm", "cases"})
    _require(row["arm"] == "owned_close_faults", "owned-close scenario differs")
    cases = row["cases"]
    _require(type(cases) is list and len(cases) == len(ARMS_V1), "close matrix is incomplete")
    for name, case in zip(ARMS_V1, cases, strict=True):
        _keys(case, {"arm", "complete", "records", "terminals", "close", "owner_events"})
        _require(case["arm"] == name and case["complete"] is True, "close arm is incomplete")
        consented = name.endswith("consented")
        retry, cancelled = name.startswith("retry"), name.startswith("cancelled")
        outcome = "failed" if retry else "cancelled" if cancelled else "returned"
        records = case["records"]
        _require(type(records) is list and len(records) == 7, "conversation trace size differs")
        for record, kind in zip(records, _CONVERSATION_KINDS, strict=True):
            _keys(record, {"kind", "value"})
            _require(record["kind"] == kind, "conversation trace order differs")
            if kind == "host_return":
                _require(record["value"] == outcome, "first close caller outcome differs")
            else:
                _digest(record["value"])
        terminals = case["terminals"]
        _require(
            type(terminals) is list and len(terminals) == (2 if consented else 0),
            "close terminal settlement count differs",
        )
        for terminal in terminals:
            _keys(terminal, {"disposition", "reason", "contextCommitted"})
            _require(
                terminal["disposition"] == "completed"
                and terminal["reason"] == "authoritative_close_completed"
                and terminal["contextCommitted"] is True,
                "close changed a completed conversation terminal",
            )
        close = _expected_close("consented" if consented else "disabled")
        if retry:
            close.insert(-1, {"stage": "launcher", "result": "failed"})
            events = [
                "provider_entered", "provider_failed", "caller_failed",
                "provider_entered", "provider_returned", "retry_returned",
            ]
        elif cancelled:
            events = [
                "provider_entered", "caller_cancelled", "provider_released",
                "provider_returned", "join_returned",
            ]
        else:
            events = ["provider_entered", "provider_returned", "caller_returned"]
        _require(case["close"] == close, "close stages were omitted, duplicated, or normalized")
        _require(case["owner_events"] == events, "close owner attempts or outcomes differ")
    baseline = cases[0]["records"]
    _require(
        all(case["records"][:-1] == baseline[:-1] for case in cases[1:6]),
        "close fault changed conversation facts",
    )
    _require(
        all(cases[index]["records"] == cases[index + 1]["records"] for index in (0, 2, 4)),
        "capture changed the close outcome",
    )
    context = "committed_conversation_context_snapshot"
    _require(
        [item for item in cases[-1]["records"] if item["kind"] == context]
        != [item for item in baseline if item["kind"] == context],
        "perturbed ingress failed to change committed context",
    )


class ObservedOwnedCloseFaultsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("owned-close receipts are producer-minted only")


_RUNS: WeakKeyDictionary[ObservedOwnedCloseFaultsV1, _ObservedRun] = WeakKeyDictionary()


def produce_owned_close_faults_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
    *,
    livekit_executable: Path,
    livekit_sha256: str,
) -> ObservedOwnedCloseFaultsV1:
    record = _observe_packaged_run(
        archive, identity, wheel, scenario="owned_close_faults",
        livekit_executable=livekit_executable, livekit_sha256=livekit_sha256,
    )
    receipt = object.__new__(ObservedOwnedCloseFaultsV1)
    _RUNS[receipt] = record
    return receipt


def validate_owned_close_faults_v1(
    receipt: ObservedOwnedCloseFaultsV1,
) -> PackagedScenarioEvidenceV1:
    if type(receipt) is not ObservedOwnedCloseFaultsV1:
        raise TypeError("owned-close receipt type is invalid")
    _require(receipt in _RUNS, "owned-close receipt is unregistered")
    record = _RUNS[receipt]
    _validate_observations(_validate_packaged_run(record, scenario="owned_close_faults"))
    return _evidence(record, (
        "conversation_trace_equal_to_revised_close_baseline", "owned_process_cleanup",
    ))
