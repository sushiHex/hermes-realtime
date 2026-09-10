"""Independent acceptance of a real capture queue overflow during conversation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
from scripts.candidate_wheel import VerifiedCandidateWheelV1
from scripts.capacity_rollover import (
    _digest,
    _digests,
    _validate_consent_callback,
    _validate_live_bindings,
    _validate_thread_owners,
)
from scripts.deterministic_equivalence import _expected_close, _keys, _require
from scripts.packaged_scenario import (
    PackagedScenarioEvidenceV1,
    _evidence,
    _observe_packaged_run,
    _ObservedRun,
    _validate_packaged_run,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1


def _validate_capacity(rows: Any) -> None:
    _require(type(rows) is list and len(rows) == 120, "overflow capacity trace is incomplete")
    kinds = (
        ["ordinary_admitted"] * 58
        + ["ordinary_rejected"]
        + ["ordinary_completed"] * 59
        + ["drain_accepted", "drain_terminal"]
    )
    counts = (
        list(zip(range(7, 65), range(2, 60), strict=True))
        + [(64, 59)]
        + list(zip(range(61, 2, -1), range(58, -1, -1), strict=True))
        + [(3, 0), (0, 0)]
    )
    for index, (row, kind, (records, physical)) in enumerate(zip(rows, kinds, counts, strict=True)):
        _keys(row, {"kind", "source", "records", "bytes", "physical", "released"})
        _require(
            row["kind"] == kind
            and row["source"] == ("record_capacity" if index == 58 else "none")
            and type(row["records"]) is int
            and row["records"] == records
            and type(row["physical"]) is int
            and row["physical"] == physical
            and row["released"] is (index == 119)
            and type(row["bytes"]) is int,
            "overflow capacity ownership or refusal differs",
        )
        if index < 119:
            _require(98_304 <= row["bytes"] <= 2_097_152, "overflow queue bytes exceed authority")
        else:
            _require(row["bytes"] == 0, "overflow drain retained byte credits")
    _require(
        all(rows[n]["bytes"] < rows[n + 1]["bytes"] for n in range(57))
        and rows[58]["bytes"] == rows[57]["bytes"]
        and rows[59]["bytes"] < rows[58]["bytes"] - 65_536
        and all(rows[n]["bytes"] > rows[n + 1]["bytes"] for n in range(59, 117))
        and rows[117]["bytes"] == rows[118]["bytes"] == 98_304,
        "overflow did not drain its accepted prefix and unused terminal credits",
    )


def _validate_observations(row: Any) -> None:
    _keys(row, {"arm", "baseline", "captured"})
    _require(row["arm"] == "over_budget_turn", "overflow scenario differs")
    common = {"arm", "complete", "records", "terminals", "close", "user", "completed_inputs"}
    extra = {
        "consent",
        "snapshots",
        "commands",
        "dispatch",
        "queue",
        "spool_records",
        "scan",
        "capacity",
        "threads",
    }
    baseline, captured = row["baseline"], row["captured"]
    _keys(baseline, common)
    _keys(captured, common | extra)
    kinds = (
        ["committed_conversation_context_snapshot"]
        + ["generated_text"] * 80
        + ["transport_confirmed_chunk"] * 80
        + [
            "committed_conversation_context_snapshot",
            "generated_text",
            "transport_confirmed_chunk",
            "host_return",
        ]
    )
    for arm, name in ((baseline, "disabled"), (captured, "consented")):
        _require(
            arm["arm"] == name
            and arm["complete"] is True
            and arm["terminals"] == []
            and arm["close"] == _expected_close(name)
            and type(arm["completed_inputs"]) is list
            and all(type(n) is int for n in arm["completed_inputs"])
            and arm["completed_inputs"] == [1, 2],
            "overflow conversation, trace, or owned close is incomplete",
        )
        records = arm["records"]
        _require(type(records) is list and len(records) == len(kinds), "source trace size differs")
        for record, kind in zip(records, kinds, strict=True):
            _keys(record, {"kind", "value"})
            _require(record["kind"] == kind, "overflow source trace order differs")
            if kind == "host_return":
                _require(record["value"] == "returned", "overflow host failed to return")
            else:
                _digest(record["value"])
        _digests(arm["user"], 2)
    _require(
        baseline["records"] == captured["records"] and baseline["user"] == captured["user"],
        "capture overflow changed production conversation",
    )
    generated = [r["value"] for r in captured["records"] if r["kind"] == "generated_text"]
    _require(
        len(set(generated)) == 81 and len(set(captured["user"])) == 2, "synthetic sources repeat"
    )
    _validate_capacity(captured["capacity"])
    _validate_thread_owners(captured["threads"], scenario="over_budget_turn")
    commands, dispatch, consent = captured["commands"], captured["dispatch"], captured["consent"]
    _keys(commands, {"create", "create_dto", "request", "binding"})
    _keys(dispatch, {"create_dto", "request", "ordinals", "records"})
    _keys(consent, {"consent", "request", "binding", "consent_callback"})
    _digests(commands["create"], 2)
    for field in ("create_dto", "request"):
        _digest(commands[field])
        _require(commands[field] == dispatch[field], "overflow transport changed create authority")
    _validate_live_bindings(consent["binding"], commands["binding"])
    _validate_consent_callback(consent["consent_callback"], consent, commands["binding"])
    _digest(consent["consent"])
    _require(commands["request"] == consent["request"], "overflow accepted request differs")
    _require(
        type(dispatch["ordinals"]) is list
        and all(type(n) is int for n in dispatch["ordinals"])
        and dispatch["ordinals"] == list(range(3, 62)),
        "overflow prefix lost admission order",
    )
    _digests(dispatch["records"], 59)
    records = captured["spool_records"]
    _require(type(records) is list and len(records) == 59, "overflow spool record count differs")
    for index, record in enumerate(records):
        _keys(record, {"dto", "snapshot", "result"})
        _digest(record["snapshot"])
        _require(
            record["dto"] == dispatch["records"][index] and record["result"] == "committed",
            "overflow spool received a different or uncommitted record",
        )
    queue = captured["queue"]
    _require(type(queue) is list and len(queue) == 61, "overflow queue envelopes are incomplete")
    for index, item in enumerate(queue):
        _keys(item, {"version", "ordinal", "kind", "lane", "payload"})
        _digest(item["payload"])
        kind = "create" if index == 0 else "drain" if index == 60 else "record"
        expected_payload = (
            commands["create_dto"]
            if index == 0
            else dispatch["records"][index - 1]
            if index < 60
            else item["payload"]
        )
        _require(
            type(item["version"]) is int
            and item["version"] == 1
            and type(item["ordinal"]) is int
            and item["ordinal"] == index + 2
            and item["kind"] == kind
            and item["lane"] == ("drain" if index == 60 else "ordered")
            and item["payload"] == expected_payload,
            "overflow dequeued envelope differs from transport",
        )
    snapshots = captured["snapshots"]
    _require(
        type(snapshots) is list and len(snapshots) == 2 and snapshots[0] == snapshots[1],
        "owned close changed the persisted incomplete prefix",
    )
    snapshot = snapshots[0]
    _keys(snapshot, {"sessions", "clock", "authority"})
    _digest(snapshot["clock"])
    _digest(snapshot["authority"])
    _require(
        type(snapshot["sessions"]) is list and len(snapshot["sessions"]) == 1,
        "overflow session count differs",
    )
    session = snapshot["sessions"][0]
    _keys(
        session,
        {
            "session",
            "epoch",
            "consent",
            "consent_request",
            "controls",
            "records",
            "opened",
            "expires",
            "retention_lag_us",
            "last_event_at",
            "predecessor",
            "state",
            "events",
            "chain",
            "kinds",
            "user",
            "generated",
            "transport",
        },
    )
    for field in (
        "session",
        "epoch",
        "consent",
        "consent_request",
        "opened",
        "expires",
        "last_event_at",
    ):
        _digest(session[field])
    _require(
        session["predecessor"] == ""
        and session["state"] == "open"
        and type(session["events"]) is int
        and session["events"] == 61
        and type(session["retention_lag_us"]) is int
        and session["retention_lag_us"] == 0
        and session["controls"] == commands["create"]
        and session["consent_request"] == consent["consent"]
        and snapshot["clock"] == session["last_event_at"],
        "overflow durable session or consent authority differs",
    )
    _require(
        session["kinds"]
        == ["session_opened", "binding_opened", "turn_opened", "user_final_accepted"]
        + ["assistant_segment_generated"] * 57
        and session["user"] == captured["user"][:1]
        and session["generated"] == generated[:57]
        and session["transport"] == [],
        "persisted prefix differs from accepted production source",
    )
    _digests(session["chain"], 61)
    _require(
        session["records"] == [record["snapshot"] for record in records],
        "overflow durable snapshots differ from dispatched records",
    )
    # Retained text without a snapshot, settlement, seal, or closed epoch cannot
    # satisfy the production eligibility conjunction. Derive exclusion from the
    # independently validated durable history, never a supplied success flag.
    scan = captured["scan"]
    _keys(scan, {"generated", "user", "matches", "files", "bytes"})
    _require(
        scan["generated"] == generated[57:]
        and scan["user"] == captured["user"][1:]
        and scan["matches"] == []
        and type(scan["files"]) is int
        and 1 <= scan["files"] <= 10
        and type(scan["bytes"]) is int
        and 0 < scan["bytes"] <= 64 * 1024 * 1024,
        "rejected source was persisted or its closed-store scan is incomplete",
    )


class ObservedOverBudgetTurnV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("overflow receipts are producer-minted only")


_RUNS: WeakKeyDictionary[ObservedOverBudgetTurnV1, _ObservedRun] = WeakKeyDictionary()


def produce_over_budget_turn_v1(
    archive: VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
    *,
    livekit_executable: Path,
    livekit_sha256: str,
) -> ObservedOverBudgetTurnV1:
    record = _observe_packaged_run(
        archive,
        identity,
        wheel,
        scenario="over_budget_turn",
        livekit_executable=livekit_executable,
        livekit_sha256=livekit_sha256,
    )
    receipt = object.__new__(ObservedOverBudgetTurnV1)
    _RUNS[receipt] = record
    return receipt


def validate_over_budget_turn_v1(receipt: ObservedOverBudgetTurnV1) -> PackagedScenarioEvidenceV1:
    if type(receipt) is not ObservedOverBudgetTurnV1:
        raise TypeError("overflow receipt type is invalid")
    _require(receipt in _RUNS, "overflow receipt is unregistered")
    record = _RUNS[receipt]
    _validate_observations(_validate_packaged_run(record, scenario="over_budget_turn"))
    return _evidence(record, ("persisted_but_excluded", "rejected_source_absent"))
