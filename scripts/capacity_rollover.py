"""Independent acceptance of the packaged host's real session capacity rollover."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.candidate_wheel import VerifiedCandidateWheelV1
from scripts.deterministic_equivalence import _expected_close, _keys, _require
from scripts.packaged_scenario import (
    PackagedScenarioEvidenceV1,
    _evidence,
    _observe_packaged_run,
    _ObservedRun,
    _validate_packaged_run,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_TURN = [
    "turn_opened",
    "user_final_accepted",
    "assistant_segment_generated",
    "assistant_chunk_transport_confirmed_full",
    "turn_snapshot",
    "turn_settled",
]
_CONTENT = {"user", "generated", "transport"}


def _digest(value: Any) -> None:
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "rollover commitment is invalid",
    )


def _digests(values: Any, count: int) -> None:
    _require(type(values) is list and len(values) == count, "rollover commitment count differs")
    for value in values:
        _digest(value)


def _validate_observations(row: Any) -> None:
    _keys(
        row,
        {
            "arm",
            "commands",
            "snapshots",
            "transactions",
            "durable_terminals",
            "rollover",
            "source",
            "terminals",
            "all_capacity_released",
            "ordinary_rejections",
            "trace_complete",
            "host_return",
            "close",
        },
    )
    _require(row["arm"] == "capacity_rollover", "rollover scenario differs")
    _require(
        row["transactions"] == ["BEGIN IMMEDIATE", "COMMIT"],
        "rollover did not use one committed transaction",
    )
    _require(row["durable_terminals"] == ["committed"] * 3, "durable turn completion differs")
    _require(
        row["rollover"]
        == [
            {
                "stage": stage,
                "result": "accepted" if stage in {"claimed", "queued"} else "committed",
            }
            for stage in ("claimed", "queued", "durable", "published", "terminal")
        ],
        "rollover lifecycle is incomplete or reordered",
    )
    commands = row["commands"]
    _keys(commands, {"create", "rollover", "successor_expiry"})
    _digests(commands["create"], 2)
    _digests(commands["rollover"], 4)
    _digest(commands["successor_expiry"])
    snapshots = row["snapshots"]
    _require(type(snapshots) is list and len(snapshots) == 4, "rollover snapshots are incomplete")
    shapes = [[(2, False)], [(2, False)], [(2, True), (0, False)], [(2, True), (1, False)]]
    for snapshot, shape in zip(snapshots, shapes, strict=True):
        _keys(snapshot, {"sessions"})
        sessions = snapshot["sessions"]
        _require(type(sessions) is list and len(sessions) == len(shape), "session count differs")
        for session, (turns, sealed) in zip(sessions, shape, strict=True):
            _keys(
                session,
                {
                    "session",
                    "epoch",
                    "predecessor",
                    "state",
                    "events",
                    "chain",
                    "kinds",
                    "consent",
                    "consent_request",
                    "controls",
                    "opened",
                    "expires",
                }
                | _CONTENT,
            )
            _digest(session["session"])
            _digest(session["epoch"])
            _digest(session["consent"])
            _digest(session["consent_request"])
            _digest(session["opened"])
            _digest(session["expires"])
            _digests(session["controls"], 4 if sealed else 2)
            expected_controls = (
                commands["create"] + commands["rollover"][:2]
                if sealed
                else commands["rollover"][2:]
                if session["predecessor"]
                else commands["create"]
            )
            _require(
                session["controls"] == expected_controls
                and (
                    not session["predecessor"] or session["expires"] == commands["successor_expiry"]
                ),
                "stored control or expiry differs from the dispatched command",
            )
            if session["predecessor"] != "":
                _digest(session["predecessor"])
            kinds = ["session_opened", "binding_opened"] + _TURN * turns
            if sealed:
                kinds += ["binding_closed", "session_seal_requested"]
            _require(
                session["kinds"] == kinds
                and type(session["events"]) is int
                and session["events"] == len(kinds)
                and session["state"] == ("sealed" if sealed else "open"),
                "session history differs from the completed conversation",
            )
            _digests(session["chain"], len(kinds))
            for name in _CONTENT:
                _digests(session[name], turns)
    before, committing, committed, continued = [s["sessions"] for s in snapshots]
    _require(before == committing, "partial rollover became visible before commit")
    old, successor = committed
    _require(old == continued[0], "sealed predecessor changed during successor conversation")
    for name in {"session", "epoch", "predecessor", "consent", "opened", "expires"} | _CONTENT:
        _require(old[name] == before[0][name], "rollover changed predecessor identity or content")
    _require(
        before[0]["predecessor"] == "" and old["chain"][:14] == before[0]["chain"],
        "rollover changed predecessor history",
    )
    _require(
        successor["session"] != old["session"]
        and successor["epoch"] == old["epoch"]
        and successor["consent"] == old["consent"]
        and successor["predecessor"] == old["session"],
        "rollover successor lineage differs",
    )
    for name in ("session", "epoch", "predecessor", "consent", "opened", "expires"):
        _require(continued[1][name] == successor[name], "conversation changed successor lineage")
    _require(continued[1]["chain"][:2] == successor["chain"], "successor opening changed")
    _keys(row["source"], _CONTENT | {"consent"})
    _digest(row["source"]["consent"])
    _require(
        all(
            session["consent_request"] == row["source"]["consent"]
            for snapshot in snapshots
            for session in snapshot["sessions"]
        ),
        "stored consent differs from the accepted source request",
    )
    for name in _CONTENT:
        _digests(row["source"][name], 3)
        _require(
            old[name] + continued[1][name] == row["source"][name],
            "persisted content differs from production source observations",
        )
    terminals = row["terminals"]
    _require(
        type(terminals) is list and len(terminals) == 3, "terminal observations are incomplete"
    )
    for terminal in terminals:
        _keys(terminal, {"disposition", "reason", "contextCommitted"})
        _require(
            terminal["disposition"] == "completed"
            and terminal["reason"] == "authoritative_close_completed"
            and terminal["contextCommitted"] is True,
            "ordinary conversation did not complete",
        )
    _require(
        type(row["ordinary_rejections"]) is int and row["ordinary_rejections"] == 0,
        "queue saturation occurred during session rollover",
    )
    _require(
        row["all_capacity_released"] is True
        and row["trace_complete"] is True
        and row["host_return"] == ["returned"]
        and row["close"] == _expected_close("consented"),
        "rollover observation or owned cleanup is incomplete",
    )


class ObservedCapacityRolloverV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("rollover receipts are producer-minted only")


_RUNS: WeakKeyDictionary[ObservedCapacityRolloverV1, _ObservedRun] = WeakKeyDictionary()


def produce_capacity_rollover_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
    *,
    livekit_executable: Path,
    livekit_sha256: str,
) -> ObservedCapacityRolloverV1:
    record = _observe_packaged_run(
        archive,
        identity,
        wheel,
        scenario="capacity_rollover",
        livekit_executable=livekit_executable,
        livekit_sha256=livekit_sha256,
    )
    receipt = object.__new__(ObservedCapacityRolloverV1)
    _RUNS[receipt] = record
    return receipt


def validate_capacity_rollover_v1(
    receipt: ObservedCapacityRolloverV1,
) -> PackagedScenarioEvidenceV1:
    if type(receipt) is not ObservedCapacityRolloverV1:
        raise TypeError("rollover receipt type is invalid")
    _require(receipt in _RUNS, "rollover receipt is unregistered")
    record = _RUNS[receipt]
    _validate_observations(_validate_packaged_run(record, scenario="capacity_rollover"))
    return _evidence(record, ("persisted_source_equal", "rollover_atomic"))
