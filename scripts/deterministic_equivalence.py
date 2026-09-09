"""Candidate-bound, source-only deterministic conversation qualification.

Raw conversation bytes stay in the archived qualification owner. Only ephemeral
keyed commitments and closed lifecycle facts cross the inherited pipe. This is
not an installed-wheel, physical-device, or complete Slice 0 qualification.
Python-private capabilities are not a boundary against a hostile same-user process.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from scripts import qualify_evidence_slice_zero as core
from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

ARMS_V1 = (
    "disabled",
    "unconsented",
    "consented",
    "blocked",
    "faulted",
    "shutdown_disabled",
    "shutdown_unconsented",
    "shutdown_consented",
    "perturbed",
)
_CONTENT_KINDS = frozenset(
    {
        "committed_conversation_context_snapshot",
        "generated_text",
        "transport_confirmed_chunk",
    }
)
_EVIDENCE_CLOSE = frozenset(
    {
        "writer_drain",
        "writer_stop",
        "transport_close",
        "evidence_runtime",
        "retention_cancellation",
    }
)
_CLOSE_STAGES = frozenset(
    {
        "binding_cleanup",
        "browser_client",
        "consent_settlement",
        "epoch_retirement",
        "evidence_runtime",
        "foreground_close",
        "host_work",
        "launcher",
        "livekit_worker",
        "retention_cancellation",
        "revoke_observer_cancellation",
        "session_worker",
        "speech_loop",
        "transport_close",
        "update_executor",
        "writer_drain",
        "writer_stop",
    }
)
_FAULT_PROPAGATION = frozenset(
    {"writer_drain", "writer_stop", "evidence_runtime", "browser_client", "launcher"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: Any, keys: set[str]) -> None:
    _require(type(value) is dict and set(value) == keys, "observation fields are not closed")


def _validate_trace_set(rows: Any) -> None:
    """Independently derive equality; plain observations never mint run authority."""
    _require(type(rows) is list and len(rows) == len(ARMS_V1), "incomplete comparison matrix")
    for name, row in zip(ARMS_V1, rows, strict=True):
        _keys(row, {"arm", "complete", "records", "close", "terminals"})
        _require(row["arm"] == name and row["complete"] is True, "wrong or incomplete arm")
        terminals = row["terminals"]
        expected_terminals = (
            [("cancelled", "host_shutdown", False)]
            if name == "shutdown_consented"
            else [("completed", "authoritative_close_completed", True)] * 2
            if name in {"consented", "blocked", "faulted"}
            else []
        )
        _require(
            type(terminals) is list and len(terminals) == len(expected_terminals),
            "terminal settlement observations are missing or duplicated",
        )
        for item, expected in zip(terminals, expected_terminals, strict=True):
            _keys(item, {"disposition", "reason", "contextCommitted"})
            _require(
                type(item["contextCommitted"]) is bool
                and (item["disposition"], item["reason"], item["contextCommitted"]) == expected,
                "terminal settlement differs from the production contract",
            )
        records = row["records"]
        _require(type(records) is list and 1 <= len(records) <= 257, "trace is empty or oversized")
        for record in records:
            _keys(record, {"kind", "value"})
            kind, value = record["kind"], record["value"]
            _require(type(kind) is str, "invalid trace kind")
            if kind in _CONTENT_KINDS:
                _require(
                    type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                    "invalid content commitment",
                )
            elif kind == "foreground_cleanup":
                _require(value is True, "foreground cleanup failed")
            elif kind == "cancellation":
                _require(value == "host_shutdown", "unexpected cancellation")
            elif kind == "host_return":
                _require(
                    value == ("failed" if name == "faulted" else "returned"),
                    "unexpected host outcome",
                )
            else:
                raise ValueError("unknown trace kind")
        kinds = [record["kind"] for record in records]
        _require(
            kinds[-1] == "host_return" and kinds.count("host_return") == 1,
            "host return is not exactly terminal",
        )
        if name.startswith("shutdown_"):
            _require(
                kinds.count("cancellation") == 1
                and kinds.count("foreground_cleanup") == 1
                and kinds.count("committed_conversation_context_snapshot") == 1
                and not {"generated_text", "transport_confirmed_chunk"}.intersection(kinds),
                "active-response cancellation is unproven",
            )
        else:
            _require(
                all(kinds.count(kind) == 2 for kind in _CONTENT_KINDS)
                and "foreground_cleanup" not in kinds,
                "typed and PCM conversation observations are absent",
            )
            _require("cancellation" not in kinds, "unexpected conversation cancellation")
        close = row["close"]
        _require(type(close) is list and 1 <= len(close) <= 256, "close observations are absent")
        for item in close:
            _keys(item, {"stage", "result"})
            _require(
                type(item["stage"]) is str and item["stage"] in _CLOSE_STAGES, "unknown close stage"
            )
            allowed = (
                {"succeeded", "failed"}
                if name == "faulted" and item["stage"] in _FAULT_PROPAGATION
                else {"succeeded"}
            )
            _require(
                item["result"] in allowed, "close failed outside the characterized writer fault"
            )
        _require(
            sum(item["stage"] == "launcher" for item in close) == 1, "launcher close is unproven"
        )
        stages = [item["stage"] for item in close]
        _require(
            {"browser_client", "foreground_close", "speech_loop"} <= set(stages),
            "ordinary close stages are absent",
        )
        _require(
            stages.index("browser_client") < stages.index("foreground_close")
            and stages.index("browser_client") < stages.index("speech_loop")
            and stages[-1] == "launcher",
            "ordinary close stage order is invalid",
        )
    baseline = rows[0]["records"]
    for row in rows[1:4]:
        _require(row["records"] == baseline, "capture changed the conversation trace")
    _require(rows[4]["records"][:-1] == baseline[:-1], "writer fault changed conversation facts")
    _require(
        rows[5]["records"] == rows[6]["records"] == rows[7]["records"],
        "capture changed cancellation",
    )

    # The negative control changes real typed ingress. Check the authoritative
    # adapter snapshot specifically: changing an unrelated outcome is insufficient.
    def contexts(row: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            record
            for record in row["records"]
            if record["kind"] == "committed_conversation_context_snapshot"
        ]

    _require(
        contexts(rows[8]) != contexts(rows[0]),
        "perturbed ingress failed to change committed context",
    )
    for group in (rows[:4], rows[5:8]):
        expected_close = [
            item for item in group[0]["close"] if item["stage"] not in _EVIDENCE_CLOSE
        ]
        for row in group[1:]:
            actual = [item for item in row["close"] if item["stage"] not in _EVIDENCE_CLOSE]
            _require(actual == expected_close, "capture changed ordinary close ordering")


class ObservedEquivalenceV1:
    """Opaque receipt minted only after the owned archived process has closed."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("equivalence receipts are producer-minted only")


@dataclass(frozen=True, slots=True)
class EquivalenceEvidenceV1:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    observation_sha256: str
    arm_count: int
    process_count: int


@dataclass(frozen=True, slots=True)
class _ObservedRun:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    observations: bytes
    processes: tuple[core._WindowsBoundProcessV1, ...]
    cleanup: core._WindowsFinalizationResultV1
    exit_code: int


_RUNS: WeakKeyDictionary[ObservedEquivalenceV1, _ObservedRun] = WeakKeyDictionary()


def produce_deterministic_equivalence_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    *,
    livekit_executable: Path,
    livekit_sha256: str,
) -> ObservedEquivalenceV1:
    """Observe the fixed source-only matrix in an owned archived Windows child."""
    from scripts.equivalence_process import run_archived_equivalence

    metadata, observations, processes, cleanup, exit_code = run_archived_equivalence(
        archive,
        identity,
        livekit_executable=livekit_executable,
        livekit_sha256=livekit_sha256,
    )
    receipt = object.__new__(ObservedEquivalenceV1)
    _RUNS[receipt] = _ObservedRun(
        metadata.candidate_head_oid,
        metadata.candidate_tree_oid,
        metadata.archive_sha256,
        observations,
        processes,
        cleanup,
        exit_code,
    )
    return receipt


def validate_deterministic_equivalence_v1(receipt: ObservedEquivalenceV1) -> EquivalenceEvidenceV1:
    """Accept only owned process evidence and derive the comparison independently."""
    if type(receipt) is not ObservedEquivalenceV1:
        raise TypeError("equivalence receipt type is invalid")
    if receipt not in _RUNS:
        raise ValueError("equivalence receipt is unregistered")
    record = _RUNS[receipt]
    _require(record.exit_code == 0, "archived host worker did not exit successfully")
    cleanup = record.cleanup
    _require(
        cleanup.closed
        and cleanup.zero_active_observed
        and not cleanup.failures
        and not cleanup.failed_handles,
        "owned cleanup is incomplete",
    )
    _require(bool(record.processes), "owned process evidence is absent")
    _require(
        all(process.process_handle in cleanup.waited_handles for process in record.processes),
        "retained processes were not all waited",
    )
    rows = core.load_strict_canonical_json(record.observations, source="equivalence observations")
    _validate_trace_set(rows)
    return EquivalenceEvidenceV1(
        record.source_commit,
        record.source_tree,
        record.source_archive_sha256,
        hashlib.sha256(record.observations).hexdigest(),
        len(ARMS_V1),
        len(record.processes),
    )
