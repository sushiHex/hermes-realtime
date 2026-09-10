"""Packaged, candidate-bound revocation race observations and independent acceptance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.candidate_wheel import VerifiedCandidateWheelV1
from scripts.deterministic_equivalence import _expected_close
from scripts.packaged_scenario import (
    PackagedScenarioEvidenceV1 as RevokeRaceEvidenceV1,
)
from scripts.packaged_scenario import (
    _evidence,
    _observe_packaged_run,
    _ObservedRun,
    _validate_packaged_run,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_COUNTS = frozenset(
    {
        "events",
        "sessions",
        "epochs",
        "requests",
        "pending_revocations",
        "revoked_epochs",
        "tombstones",
        "erased_events",
        "erased_sessions",
        "last_ordinal",
        "final_ordinal",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_observations(row: Any) -> None:
    _require(
        type(row) is dict
        and set(row)
        == {
            "arm",
            "snapshots",
            "acknowledgment",
            "completed_turns",
            "capture_terminals",
            "admitted_before",
            "admitted_after",
            "revoke_accepted",
            "revoke_terminal",
            "all_capacity_released",
            "trace_complete",
            "host_return",
            "close",
        },
        "revocation observation fields are not closed",
    )
    _require(
        row["arm"] == "revoke_race" and row["acknowledgment"] == "durable",
        "revocation acknowledgment differs",
    )
    snapshots = row["snapshots"]
    _require(type(snapshots) is list and len(snapshots) == 3, "revocation snapshots are incomplete")
    for snapshot in snapshots:
        _require(
            type(snapshot) is dict
            and set(snapshot) == _COUNTS | {"integrity_ok", "scope_matches", "authority"},
            "revocation snapshot fields are not closed",
        )
        _require(
            snapshot["integrity_ok"] is True and snapshot["scope_matches"] is True,
            "store integrity or revocation scope is unverified",
        )
        _require(
            type(snapshot["authority"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", snapshot["authority"]) is not None,
            "revocation authority commitment is invalid",
        )
        _require(
            all(
                type(snapshot[name]) is int and 0 <= snapshot[name] <= 2**63 - 1 for name in _COUNTS
            ),
            "store counts are not exact bounded integers",
        )
    before, after, purged = snapshots
    _require(before == after, "post-revoke conversation changed pending evidence")
    _require(
        before["events"] > 0
        and before["last_ordinal"] > 0
        and all(
            before[name] == 1
            for name in ("sessions", "epochs", "requests", "pending_revocations", "revoked_epochs")
        )
        and all(
            before[name] == 0
            for name in ("tombstones", "erased_events", "erased_sessions", "final_ordinal")
        ),
        "durable revocation is disconnected from retained evidence",
    )
    _require(
        all(
            purged[name] == 0
            for name in (
                "events",
                "sessions",
                "epochs",
                "requests",
                "pending_revocations",
                "revoked_epochs",
            )
        )
        and purged["tombstones"] == 1
        and purged["erased_events"] == before["events"]
        and purged["erased_sessions"] == before["sessions"]
        and purged["authority"] == before["authority"]
        and purged["last_ordinal"] == before["last_ordinal"]
        and purged["final_ordinal"] >= purged["last_ordinal"],
        "revocation purge is incomplete or disconnected",
    )
    for name, expected in (
        ("completed_turns", 2),
        ("capture_terminals", 1),
        ("revoke_accepted", 1),
        ("revoke_terminal", 1),
    ):
        _require(
            type(row[name]) is int and row[name] == expected,
            "revocation conversation or lifecycle observations differ",
        )
    _require(
        type(row["admitted_before"]) is int
        and 0 < row["admitted_before"] <= 256
        and type(row["admitted_after"]) is int
        and row["admitted_after"] == row["admitted_before"],
        "new evidence was admitted after revocation",
    )
    _require(
        row["all_capacity_released"] is True
        and row["trace_complete"] is True
        and row["host_return"] == "returned",
        "revocation host cleanup is incomplete",
    )
    _require(row["close"] == _expected_close("consented"), "revocation close stages differ")


class ObservedRevokeRaceV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("revocation receipts are producer-minted only")


_RUNS: WeakKeyDictionary[ObservedRevokeRaceV1, _ObservedRun] = WeakKeyDictionary()


def produce_revoke_race_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
    *,
    livekit_executable: Path,
    livekit_sha256: str,
) -> ObservedRevokeRaceV1:
    record = _observe_packaged_run(
        archive,
        identity,
        wheel,
        scenario="revoke_race",
        livekit_executable=livekit_executable,
        livekit_sha256=livekit_sha256,
    )
    receipt = object.__new__(ObservedRevokeRaceV1)
    _RUNS[receipt] = record
    return receipt


def validate_revoke_race_v1(receipt: ObservedRevokeRaceV1) -> RevokeRaceEvidenceV1:
    if type(receipt) is not ObservedRevokeRaceV1:
        raise TypeError("revocation receipt type is invalid")
    _require(receipt in _RUNS, "revocation receipt is unregistered")
    record = _RUNS[receipt]
    _validate_observations(_validate_packaged_run(record, scenario="revoke_race"))
    return _evidence(record, ("purge_verified", "purged", "revocation_request_durable"))
