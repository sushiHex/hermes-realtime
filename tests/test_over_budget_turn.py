"""Overflow acceptance binds real source, durable exclusion and complete cleanup."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest


def _observations() -> dict:
    from scripts.deterministic_equivalence import _expected_close
    from tests.test_capacity_rollover import _observations as rollover

    def digest(n):
        return f"{n:064x}"

    generated = [digest(n) for n in range(100, 181)]
    transport = [digest(n) for n in range(200, 281)]
    records = (
        [{"kind": "committed_conversation_context_snapshot", "value": digest(1)}]
        + [{"kind": "generated_text", "value": value} for value in generated[:80]]
        + [{"kind": "transport_confirmed_chunk", "value": value} for value in transport[:80]]
        + [
            {"kind": "committed_conversation_context_snapshot", "value": digest(2)},
            {"kind": "generated_text", "value": generated[-1]},
            {"kind": "transport_confirmed_chunk", "value": transport[-1]},
            {"kind": "host_return", "value": "returned"},
        ]
    )
    users = [digest(400), digest(401)]
    baseline = dict(
        arm="disabled",
        complete=True,
        records=records,
        terminals=[],
        close=_expected_close("disabled"),
        user=users,
        completed_inputs=[1, 2],
    )
    captured = copy.deepcopy(baseline)
    captured.update(arm="consented", close=_expected_close("consented"))
    sample = rollover()
    session = sample["snapshots"][0]["sessions"][0]
    session.update(
        events=61,
        chain=[digest(n) for n in range(500, 561)],
        kinds=["session_opened", "binding_opened", "turn_opened", "user_final_accepted"]
        + ["assistant_segment_generated"] * 57,
        records=[digest(n) for n in range(900, 959)],
        user=users[:1],
        generated=generated[:57],
        transport=[],
    )
    snapshot = dict(sessions=[session], clock=session["last_event_at"], authority=digest(4))

    def capacity(kind, records, physical, size, *, source="none", released=False):
        return dict(
            kind=kind,
            records=records,
            physical=physical,
            bytes=size,
            source=source,
            released=released,
        )

    rows = [capacity("ordinary_admitted", 7 + n, 2 + n, 164_840 + 100 * n) for n in range(58)]
    rows.append(capacity("ordinary_rejected", 64, 59, rows[-1]["bytes"], source="record_capacity"))
    rows += [
        capacity("ordinary_completed", 61 - n, 58 - n, 98_304 + (58 - n) * 100) for n in range(59)
    ]
    rows += [
        capacity("drain_accepted", 3, 0, 98_304),
        capacity("drain_terminal", 0, 0, 0, released=True),
    ]
    threads = sample["threads"]
    threads.update(
        dequeues=61,
        calls=[
            dict(stage=stage, thread=threads["sqlite"])
            for stage in ["factory", "create_epoch", "active_session_expiry"]
            + ["append_record"] * 59
            + ["drain_and_close", "close"]
        ],
    )
    captured.update(
        threads=threads,
        consent={
            "consent": session["consent_request"],
            "request": sample["commands"]["request"],
            "binding": copy.deepcopy(sample["source"]["binding"]),
            "consent_callback": copy.deepcopy(sample["source"]["consent_callback"]),
        },
        snapshots=[snapshot, copy.deepcopy(snapshot)],
        commands={k: sample["commands"][k] for k in ("create", "create_dto", "request", "binding")},
        dispatch={
            **{k: sample["commands"][k] for k in ("create_dto", "request")},
            "ordinals": list(range(3, 62)),
            "records": [digest(n) for n in range(700, 759)],
        },
        spool_records=[
            dict(dto=digest(n + 700), snapshot=digest(n + 900), result="committed")
            for n in range(59)
        ],
        queue=[
            dict(
                version=1,
                ordinal=n + 2,
                lane="drain" if n == 60 else "ordered",
                kind="create" if n == 0 else "drain" if n == 60 else "record",
                payload=sample["commands"]["create_dto"]
                if n == 0
                else digest(n + 699)
                if n < 60
                else digest(999),
            )
            for n in range(61)
        ],
        scan=dict(generated=generated[57:], user=users[1:], matches=[], files=1, bytes=262_144),
        capacity=rows,
    )
    return dict(arm="over_budget_turn", baseline=baseline, captured=captured)


def test_overflow_accepts_equal_conversation_and_excluded_durable_prefix() -> None:
    from scripts.over_budget_turn import _validate_observations

    _validate_observations(_observations())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_arm",
        "extra_claim",
        "conversation_changed",
        "trace_incomplete",
        "missing_completion",
        "bool_completion",
        "terminal_present",
        "host_failed",
        "close_missing",
        "trace_reordered",
        "empty_commitment",
        "repeated_source",
        "foreign_source",
        "capacity_missing",
        "false_refusal",
        "false_record_bound",
        "false_physical_bound",
        "bool_records",
        "leaked_capacity",
        "leaked_bytes",
        "early_completion",
        "bytes_reappear",
        "not_durable",
        "queue_missing",
        "queue_ordinal",
        "queue_payload",
        "spool_dto",
        "spool_snapshot",
        "durable_snapshot",
        "foreign_dto",
        "foreign_request",
        "fifo_gap",
        "bool_ordinal",
        "close_changed_store",
        "missing_store",
        "extra_session",
        "wrong_clock",
        "sealed",
        "bool_event_count",
        "retention_extended",
        "foreign_consent",
        "foreign_control",
        "foreign_user",
        "source_tail_persisted",
        "transport_persisted",
        "terminal_persisted",
        "chain_truncated",
        "rejected_source_found",
        "scan_source_substituted",
        "scan_user_missing",
        "scan_empty",
        "scan_unbounded",
    ],
)
def test_overflow_rejects_incomplete_or_contradictory_observations(mutation: str) -> None:
    from scripts.over_budget_turn import _validate_observations

    row = _observations()
    arm = row["captured"]
    session = arm["snapshots"][0]["sessions"][0]
    if mutation == "missing_arm":
        del row["baseline"]
    elif mutation == "extra_claim":
        row["accepted"] = True
    elif mutation == "conversation_changed":
        arm["records"][1]["value"] = "f" * 64
    elif mutation == "trace_incomplete":
        arm["complete"] = False
    elif mutation == "missing_completion":
        arm["completed_inputs"].pop()
    elif mutation == "bool_completion":
        arm["completed_inputs"][0] = True
    elif mutation == "terminal_present":
        arm["terminals"].append({"disposition": "completed"})
    elif mutation == "host_failed":
        arm["records"][-1]["value"] = "raised"
    elif mutation == "close_missing":
        arm["close"].pop()
    elif mutation == "trace_reordered":
        arm["records"][0], arm["records"][1] = arm["records"][1], arm["records"][0]
    elif mutation == "empty_commitment":
        arm["records"][0]["value"] = ""
    elif mutation == "repeated_source":
        arm["records"][2]["value"] = arm["records"][1]["value"]
        row["baseline"]["records"] = copy.deepcopy(arm["records"])
    elif mutation == "foreign_source":
        arm["user"][0] = "e" * 64
    elif mutation == "capacity_missing":
        arm["capacity"].pop()
    elif mutation == "false_refusal":
        arm["capacity"][58]["source"] = "none"
    elif mutation == "false_record_bound":
        arm["capacity"][58]["records"] = 63
    elif mutation == "false_physical_bound":
        arm["capacity"][58]["physical"] = 64
    elif mutation == "bool_records":
        arm["capacity"][0]["records"] = True
    elif mutation == "leaked_capacity":
        arm["capacity"][-1]["released"] = False
    elif mutation == "leaked_bytes":
        arm["capacity"][-1]["bytes"] = 1
    elif mutation == "early_completion":
        arm["capacity"][20]["kind"] = "ordinary_completed"
    elif mutation == "bytes_reappear":
        arm["capacity"][90]["bytes"] = arm["capacity"][89]["bytes"] + 1
    elif mutation == "queue_missing":
        arm["queue"].pop()
    elif mutation == "queue_ordinal":
        arm["queue"][0]["ordinal"] = 1
    elif mutation == "queue_payload":
        arm["queue"][1]["payload"] = "f" * 64
    elif mutation == "spool_dto":
        arm["spool_records"][0]["dto"] = "f" * 64
    elif mutation == "spool_snapshot":
        arm["spool_records"][0]["snapshot"] = "f" * 64
    elif mutation == "durable_snapshot":
        session["records"][0] = "f" * 64
    elif mutation == "not_durable":
        arm["spool_records"][-1]["result"] = "writer_fault"
    elif mutation == "foreign_dto":
        arm["commands"]["create_dto"] = "f" * 64
    elif mutation == "foreign_request":
        arm["consent"]["request"] = "f" * 64
    elif mutation == "fifo_gap":
        arm["dispatch"]["ordinals"][5] += 1
    elif mutation == "bool_ordinal":
        arm["dispatch"]["ordinals"][0] = True
    elif mutation == "close_changed_store":
        arm["snapshots"][1]["clock"] = "f" * 64
    elif mutation == "missing_store":
        arm["snapshots"].pop()
    elif mutation == "extra_session":
        arm["snapshots"][0]["sessions"].append(copy.deepcopy(session))
    elif mutation == "wrong_clock":
        arm["snapshots"][0]["clock"] = "f" * 64
    elif mutation == "sealed":
        session["state"] = "sealed"
    elif mutation == "bool_event_count":
        session["events"] = True
    elif mutation == "retention_extended":
        session["retention_lag_us"] = -1
    elif mutation == "foreign_consent":
        session["consent_request"] = "d" * 64
    elif mutation == "foreign_control":
        session["controls"][0] = "f" * 64
    elif mutation == "foreign_user":
        session["user"][0] = "f" * 64
    elif mutation == "source_tail_persisted":
        session["generated"][-1] = arm["scan"]["generated"][0]
    elif mutation == "transport_persisted":
        session["transport"] = ["a" * 64]
    elif mutation == "terminal_persisted":
        session["kinds"][-1] = "turn_settled"
    elif mutation == "chain_truncated":
        session["chain"].pop()
    elif mutation == "rejected_source_found":
        arm["scan"]["matches"] = [arm["scan"]["generated"][0]]
    elif mutation == "scan_source_substituted":
        arm["scan"]["generated"][0] = "f" * 64
    elif mutation == "scan_user_missing":
        arm["scan"]["user"] = []
    elif mutation == "scan_empty":
        arm["scan"]["files"] = 0
    elif mutation == "scan_unbounded":
        arm["scan"]["bytes"] = 64 * 1024 * 1024 + 1
    else:
        raise AssertionError(mutation)
    # Test independent semantic checks even when both store reads agree.
    if mutation not in {"missing_store", "close_changed_store"}:
        arm["snapshots"][1] = copy.deepcopy(arm["snapshots"][0])
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal", "-vacuum", "-tmp"])
def test_rejected_source_scan_reads_every_present_sqlite_file(tmp_path: Path, suffix: str) -> None:
    from scripts.over_budget_turn_worker import _scan_rejected_source

    database = tmp_path / "capture-v1.sqlite3"
    generated = [f"Synthetic generated segment {n:02d}.".encode() for n in range(81)]
    users = ["Synthetic first user input.", "Synthetic second user input."]
    database.write_bytes(b"synthetic accepted prefix")
    file = database.with_name(database.name + suffix)
    file.write_bytes(generated[-1] + b"\n" + users[-1].encode())
    result = _scan_rejected_source(database, generated, users, b"synthetic key")
    assert result["matches"] == [result["generated"][-1], result["user"][0]]


def test_overflow_receipts_and_registration_reject_supplied_success() -> None:
    from scripts.over_budget_turn import ObservedOverBudgetTurnV1, validate_over_budget_turn_v1
    from scripts.qualify_evidence_slice_zero import (
        OVER_BUDGET_TURN_REGISTRATION_V1,
        SCENARIO_REGISTRY_V1,
        OverBudgetTurnRegistrationV1,
    )

    assert SCENARIO_REGISTRY_V1[12] is OVER_BUDGET_TURN_REGISTRATION_V1
    with pytest.raises(TypeError):
        ObservedOverBudgetTurnV1()
    with pytest.raises(TypeError):
        validate_over_budget_turn_v1({"accepted": True})
    with pytest.raises(ValueError):
        validate_over_budget_turn_v1(object.__new__(ObservedOverBudgetTurnV1))
    with pytest.raises(ValueError, match="canonical"):
        OverBudgetTurnRegistrationV1().produce(
            None, None, None, livekit_executable=Path("missing"), livekit_sha256="a" * 64
        )


@pytest.mark.parametrize("mutation", ["missing", "role", "dequeues", "call", "live", "crash"])
def test_overflow_rejects_incomplete_or_failed_writer_owners(mutation: str) -> None:
    from scripts.over_budget_turn import _validate_observations

    row = _observations()
    captured = row["captured"]
    threads = captured["threads"]
    if mutation == "missing":
        del captured["threads"]
    elif mutation == "role":
        threads["sqlite"] = threads["dispatcher"]
    elif mutation == "dequeues":
        threads["dequeues"] = 21
    elif mutation == "call":
        threads["calls"].pop()
    elif mutation == "live":
        threads["sqlite_stopped"] = False
    else:
        threads["dispatcher_clean"] = False
    with pytest.raises(ValueError):
        _validate_observations(row)


@pytest.mark.parametrize("mutation", ["none", "early_drain", "watermark", "foreign_owner"])
def test_overflow_queue_observer_retains_actual_envelopes(mutation: str) -> None:
    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence import models as m
    from scripts.evidence_observation import _ObserveQueue
    from tests.evidence.test_sqlite_spool import (
        make_create_epoch,
        ordinary_record,
        turn_opened_snapshot,
    )

    queue = a.BoundedEvidenceWriterQueueV1()
    observer = _ObserveQueue(
        queue, b"synthetic key", owner_generation=41, scenario="over_budget_turn"
    )
    items = [
        a.EvidenceWriterQueueItemV1(
            protocol_version=1,
            lane=a.WriterQueueLane.ORDERED,
            payload=make_create_epoch(),
            admission_ordinal=2,
        )
    ]
    for ordinal in range(3, 62):
        items.append(
            a.EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=a.WriterQueueLane.ORDERED,
                payload=ordinary_record(turn_opened_snapshot(3, event_index=3), ordinal),
                admission_ordinal=ordinal,
            )
        )
    items.append(
        a.EvidenceWriterQueueItemV1(
            protocol_version=1,
            lane=a.WriterQueueLane.DRAIN,
            admission_ordinal=62,
            payload=m.DrainAndStopV1(
                protocol_version=1,
                owner_generation=42 if mutation == "foreign_owner" else 41,
                final_admission_ordinal=60 if mutation == "watermark" else 61,
            ),
        )
    )
    if mutation == "early_drain":
        del items[20:-1]

    def consume():
        for item in items:
            queue.put_nowait(item)
            assert queue.get_blocking() is item

    try:
        if mutation == "none":
            consume()
            assert [r["ordinal"] for r in observer.records] == list(range(2, 63))
            assert [r["kind"] for r in observer.records] == ["create"] + ["record"] * 59 + ["drain"]
        else:
            with pytest.raises(ValueError):
                consume()
    finally:
        observer.restore()
