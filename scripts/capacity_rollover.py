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


def _validate_thread_owners(threads: Any) -> None:
    _keys(
        threads,
        {
            "event_loop",
            "dispatcher",
            "sqlite",
            "dequeues",
            "calls",
            "dispatcher_stopped",
            "sqlite_stopped",
            "dispatcher_clean",
            "dispatcher_bound",
            "sqlite_clean",
        },
    )
    for name in ("event_loop", "dispatcher", "sqlite"):
        _digest(threads[name])
    _require(
        len({threads[name] for name in ("event_loop", "dispatcher", "sqlite")}) == 3
        and type(threads["dequeues"]) is int
        and threads["dequeues"] == 21
        and threads["dispatcher_stopped"] is True
        and threads["sqlite_stopped"] is True
        and threads["dispatcher_clean"] is True
        and threads["dispatcher_bound"] is True
        and threads["sqlite_clean"] is True,
        "SQLite, dispatcher, and event-loop ownership is not separate or stopped",
    )
    stages = (
        ["factory", "create_epoch", "active_session_expiry"]
        + ["append_record"] * 12
        + ["rollover_session"]
        + ["append_record"] * 6
        + ["drain_and_close", "close"]
    )
    _require(
        threads["calls"] == [{"stage": stage, "thread": threads["sqlite"]} for stage in stages],
        "SQLite factory or spool calls escaped their dedicated owner",
    )


def _validate_live_bindings(bindings: Any, generation: str) -> None:
    _digest(generation)
    _require(
        type(bindings) is list and len(bindings) == 2, "live binding observations are incomplete"
    )
    fields = {
        "browser_generation",
        "worker_generation",
        "browser_participant",
        "worker_participant",
    }
    for binding in bindings:
        _keys(binding, fields)
        for value in binding.values():
            _digest(value)
        _require(
            binding["browser_generation"] == binding["worker_generation"] == generation
            and binding["browser_participant"] == binding["worker_participant"],
            "live browser and media binding observations differ",
        )
    _require(bindings[0] == bindings[1], "live binding changed across consent")


def _validate_observations(row: Any) -> None:
    _keys(
        row,
        {
            "arm",
            "commands",
            "dispatch",
            "queue",
            "spool_records",
            "threads",
            "snapshots",
            "transactions",
            "transaction_writes",
            "connections",
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
    _validate_thread_owners(row["threads"])
    _keys(row["connections"], {"observer_reads", "unexpected", "data_version"})
    connections = row["connections"]
    versions = connections["data_version"]
    _require(
        type(connections["observer_reads"]) is int
        and connections["observer_reads"] == 1
        and type(connections["unexpected"]) is int
        and connections["unexpected"] == 0
        and type(versions) is list
        and len(versions) == 2
        and all(type(value) is int and value >= 1 for value in versions)
        and versions[0] == versions[1],
        "rollover opened an unobserved connection, saw an external commit, or missed its reader",
    )
    _require(
        row["transactions"] == ["BEGIN IMMEDIATE", "COMMIT"],
        "rollover did not use one committed transaction",
    )
    _require(
        type(row["transaction_writes"]) is list
        and 1 <= len(row["transaction_writes"]) <= 128
        and all(
            type(kind) is str and kind in {"INSERT", "UPDATE", "DELETE"}
            for kind in row["transaction_writes"]
        ),
        "rollover write observations are missing or invalid",
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
    _keys(
        commands,
        {
            "create",
            "rollover",
            "successor_expiry",
            "create_dto",
            "rollover_dto",
            "request",
            "binding",
        },
    )
    dispatch = row["dispatch"]
    _keys(
        dispatch,
        {"create_dto", "rollover_dto", "request", "ordinals", "rollover_ordinal", "records"},
    )
    _digests(dispatch["records"], 18)
    spool_records = row["spool_records"]
    _require(
        type(spool_records) is list and len(spool_records) == 18,
        "spool record observations are incomplete",
    )
    for index, record in enumerate(spool_records):
        _keys(record, {"dto", "snapshot", "result"})
        _digest(record["snapshot"])
        _require(
            record["dto"] == dispatch["records"][index] and record["result"] == "committed",
            "spool received a different or uncommitted ordinary record",
        )
    queue = row["queue"]
    _require(type(queue) is list and len(queue) == 21, "dequeued queue envelopes are incomplete")
    expected_payloads = [
        commands["create_dto"],
        *dispatch["records"][:12],
        commands["rollover_dto"],
        *dispatch["records"][12:],
    ]
    for index, item in enumerate(queue):
        _keys(item, {"version", "kind", "lane", "ordinal", "payload"})
        _digest(item["payload"])
        kind = (
            "create"
            if index == 0
            else "rollover"
            if index == 13
            else "drain"
            if index == 20
            else "record"
        )
        _require(
            type(item["version"]) is int
            and item["version"] == 1
            and type(item["ordinal"]) is int
            and item["ordinal"] == index + 2
            and item["kind"] == kind
            and item["lane"] == ("drain" if index == 20 else "ordered")
            and (index == 20 or item["payload"] == expected_payloads[index]),
            "dequeued envelope differs from FIFO transport dispatch",
        )
    for field in ("create_dto", "rollover_dto", "request"):
        _digest(commands[field])
        _require(dispatch[field] == commands[field], "transport changed the complete command")
    _require(
        type(dispatch["ordinals"]) is list
        and all(type(n) is int for n in dispatch["ordinals"])
        and dispatch["ordinals"] == list(range(3, 22))
        and type(dispatch["rollover_ordinal"]) is int
        and dispatch["rollover_ordinal"] == 15,
        "dispatched rollover lost admission order",
    )
    _digests(commands["create"], 2)
    _digests(commands["rollover"], 4)
    _digest(commands["successor_expiry"])
    snapshots = row["snapshots"]
    _require(type(snapshots) is list and len(snapshots) == 4, "rollover snapshots are incomplete")
    shapes = [[(2, False)], [(2, False)], [(2, True), (0, False)], [(2, True), (1, False)]]
    for snapshot, shape in zip(snapshots, shapes, strict=True):
        _keys(snapshot, {"sessions", "clock", "authority"})
        _digest(snapshot["clock"])
        _digest(snapshot["authority"])
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
                    "records",
                    "opened",
                    "expires",
                    "retention_lag_us",
                    "last_event_at",
                }
                | _CONTENT,
            )
            _digest(session["session"])
            _digest(session["epoch"])
            _digest(session["consent"])
            _digest(session["consent_request"])
            _digest(session["opened"])
            _digest(session["expires"])
            _digest(session["last_event_at"])
            lag = session["retention_lag_us"]
            _require(
                type(lag) is int
                and (0 <= lag <= 5_000_000 if session["predecessor"] else lag == 0),
                "stored retention interval differs from consent",
            )
            _digests(session["controls"], 4 if sealed else 2)
            _digests(session["records"], 6 * turns)
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
        _require(
            snapshot["clock"] == sessions[-1]["last_event_at"]
            and snapshot["authority"] == snapshots[0]["authority"],
            "store clock or installation authority differs",
        )
    before, committing, committed, continued = [s["sessions"] for s in snapshots]
    _require(before == committing, "partial rollover became visible before commit")
    old, successor = committed
    _require(
        old["records"] + continued[1]["records"] == [r["snapshot"] for r in spool_records],
        "persisted snapshots differ from the complete dispatched records",
    )
    _require(
        old["last_event_at"] == successor["last_event_at"], "rollover control timestamps differ"
    )
    _require(old == continued[0], "sealed predecessor changed during successor conversation")
    for name in {
        "session",
        "epoch",
        "predecessor",
        "consent",
        "opened",
        "expires",
        "retention_lag_us",
        "records",
    } | _CONTENT:
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
    for name in (
        "session",
        "epoch",
        "predecessor",
        "consent",
        "opened",
        "expires",
        "retention_lag_us",
    ):
        _require(continued[1][name] == successor[name], "conversation changed successor lineage")
    _require(continued[1]["chain"][:2] == successor["chain"], "successor opening changed")
    _keys(row["source"], _CONTENT | {"consent", "request", "binding"})
    _validate_live_bindings(row["source"]["binding"], commands["binding"])
    _digest(row["source"]["request"])
    _require(
        commands["request"] == row["source"]["request"],
        "create differs from accepted request identity",
    )
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
