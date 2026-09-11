"""Drive real packaged admission and spool paths under five bounded injections."""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _read_frame, _require, _require_ack, _write_frame
from scripts.evidence_protocol_oracle import canonical_json_bytes
from scripts.synthetic_fault_oracle import (
    ADJACENT,
    CASES_V1,
    DATABASE_NAMES,
    DECOYS,
    digest,
    inventory,
    observe_database,
    scan_rejected_source,
)


class _InjectedConnection(sqlite3.Connection):
    inject = False
    injections = 0
    injection_transactions: tuple[bool, ...] = ()
    rollback_transactions: tuple[bool, ...] = ()

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if self.inject and sql.startswith("INSERT INTO evidence_events"):
            self.injections += 1
            self.injection_transactions += (self.in_transaction,)
            raise sqlite3.OperationalError("synthetic event insertion failure")
        return super().execute(sql, *args, **kwargs)

    def rollback(self) -> None:
        self.rollback_transactions += (self.in_transaction,)
        super().rollback()


def _spool(case: Path, clock: Any, cls: Any = None) -> Any:
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool
    from hermes_realtime.evidence.storage_security import WindowsStorageProbeV1
    from tests.evidence import spool_crash_worker as driver

    root = case / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    return (cls or SQLiteEvidenceSpool)(
        root / "capture-v1.sqlite3",
        clock=clock,
        uuid_factory=driver._Uuids(),
        probe=WindowsStorageProbeV1(),
        connection_factory=_InjectedConnection,
    )


def _user(text: str) -> Any:
    from hermes_realtime.evidence import models as m
    from tests.evidence import spool_crash_worker as driver

    snapshot = driver._snapshot(
        m.EventKind.USER_FINAL_ACCEPTED,
        4,
        m.UserFinalAcceptedPayloadV1(
            utterance_id="10000000-0000-4000-8000-000000000008",
            evidence_turn_id="10000000-0000-4000-8000-000000000007",
            source=m.InputSource.TYPED,
            routing_disposition="response",
            text=text,
        ),
        11,
    )
    return m.QueuedEvidenceRecordV1(
        protocol_version=1,
        snapshot=snapshot,
        admission_ordinal=4,
        reservation_class=m.QueueReservationClass.ORDINARY,
        lease_open_ordinal=None,
    )


def _generated(ordinal: int) -> Any:
    from hermes_realtime.evidence import models as m
    from tests.evidence import spool_crash_worker as driver

    return driver._snapshot(
        m.EventKind.ASSISTANT_SEGMENT_GENERATED,
        4 + ordinal,
        m.AssistantSegmentGeneratedPayloadV1(
            evidence_turn_id="10000000-0000-4000-8000-000000000007",
            evidence_segment_id=driver._event_id(200 + ordinal),
            segment_ordinal=ordinal,
            text=f"Synthetic segment {ordinal:02d}.",
        ),
        20 + ordinal,
    )


def _seed(owned: Any, case: Path, *, queue: bool) -> tuple[Any, ...]:
    from hermes_realtime.evidence import models as m
    from tests.evidence import spool_crash_worker as driver

    create, turn = driver._make_create_epoch(), driver._ordinary_record()
    _require(owned.create_epoch(create) is m.StoreDisposition.COMMITTED, "synthetic epoch refused")
    _require(owned.append_record(turn) is m.StoreDisposition.COMMITTED, "synthetic turn refused")
    snapshots: tuple[Any, ...] = (create.session_opened, create.binding_opened, turn.snapshot)
    if queue:
        user = _user("Synthetic baseline.")
        _require(
            owned.append_record(user) is m.StoreDisposition.COMMITTED, "synthetic user refused"
        )
        snapshots += (user.snapshot,)
    for name, raw in DECOYS.items():
        with (case / "evidence" / name).open("xb") as file:
            file.write(raw)
    adjacent = case / "adjacent"
    adjacent.mkdir()
    for name, raw in ADJACENT.items():
        with (adjacent / name).open("xb") as file:
            file.write(raw)
    return snapshots


def _queue(owned: Any, seed: tuple[Any, ...]) -> tuple[dict[str, Any], str]:
    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence import models as m

    scheduler = a._new_production_evidence_scheduler_v1()
    # Prime the real ordinal allocator with the independently committed fixture
    # prefix; no record/byte limit, counter, or ordinal is assigned by this driver.
    for snapshot in seed:
        _require(
            scheduler.try_admit(scheduler.prepare(snapshot)) is not None, "synthetic prefix refused"
        )
        scheduler.complete(scheduler.dequeue_nowait())
    for ordinal in range(1, 65):
        _require(
            scheduler.try_admit(scheduler.prepare(_generated(ordinal))) is not None,
            "synthetic lawful admission refused",
        )
    before = scheduler.credits
    source = _generated(65)
    prepared = scheduler.prepare(source)
    item, reason = scheduler._try_admit_observed(prepared)
    after = scheduler.credits
    _require(item is None and reason is not None, "synthetic capacity did not refuse")
    decisions = []
    for _ in range(64):
        item = scheduler.dequeue_nowait()
        result = owned.append_record(item.payload)
        decisions.append(result.value)
        _require(
            result is m.StoreDisposition.COMMITTED, "synthetic admitted prefix did not persist"
        )
        scheduler.complete(item)
    capacity = scheduler._capacity
    return {
        "source_sha256": digest(prepared.canonical_bytes),
        "reason": reason,
        "before": list(before),
        "after": list(after),
        "drained": list(scheduler.credits),
        "decisions": decisions,
        "limits": {
            "maxQueueRecords": capacity.max_records,
            "maxQueueCanonicalBytes": capacity.max_canonical_bytes,
            "maxQueuePhysicalItems": capacity.max_physical_items,
            "maxCanonicalRecordBytes": a.MAX_CANONICAL_RECORD_BYTES,
        },
    }, source.payload.text


def _blocked_drain(case: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.sqlite_spool import (
        SQLiteEvidenceSpool,
        SQLiteEvidenceWriterDaemonV1,
    )
    from tests.evidence import spool_crash_worker as driver

    entered, release = threading.Event(), threading.Event()
    stages: list[str] = []

    # The script gate skips the installed candidate's implementation imports.
    class BlockedSpool(SQLiteEvidenceSpool):  # type: ignore[misc]
        def drain_and_close(self, command: Any) -> Any:
            stages.append("drain_entered")
            entered.set()
            _require(release.wait(5), "synthetic drain gate timed out")
            stages.append("drain_released")
            result = super().drain_and_close(command)
            stages.append("drain_" + result.value)
            return result

    daemon = SQLiteEvidenceWriterDaemonV1(lambda: _spool(case, driver._Clock(), BlockedSpool))
    thread = None
    outcomes: list[str] = []
    try:
        _seed(daemon, case, queue=False)
        before = observe_database(case)
        command = m.DrainAndStopV1(
            protocol_version=1, owner_generation=daemon.owner_generation, final_admission_ordinal=3
        )

        def drain() -> None:
            outcomes.append(daemon.drain_and_close(command).value)

        thread = threading.Thread(target=drain)
        thread.start()
        _require(entered.wait(5), "synthetic drain was not entered")
        pending = (
            daemon.drain_is_pending and thread.is_alive() and daemon.is_running and not outcomes
        )
        _require(pending, "synthetic drain did not remain pending")
        stages.append("pending_observed")
        release.set()
        thread.join(5)
        _require(not thread.is_alive() and daemon.join(5), "synthetic drain owner did not stop")
        stages.append("joined")
        _require(not daemon.is_running, "synthetic writer remains running")
        return before, {"stages": stages, "outcomes": outcomes}
    finally:
        release.set()
        if thread is not None:
            thread.join(5)
        daemon.close()
        _require(daemon.join(5), "synthetic writer cleanup failed")


def _run(case: Path, case_id: str, prepared: Callable[[], None]) -> dict[str, Any]:
    from hermes_realtime.evidence import models as m
    from tests.evidence import spool_crash_worker as driver

    _require(
        os.name == "nt" and case_id in CASES_V1 and case.is_absolute() and not case.exists(),
        "synthetic case ownership differs",
    )
    source = ""
    if case_id == "writer_drain_blocked":
        before, fault = _blocked_drain(case)
    else:
        clock = driver._Clock()
        owned = _spool(case, clock)
        try:
            seed = _seed(owned, case, queue=case_id == "queue_capacity_coupled")
            before = observe_database(case)
            if case_id == "queue_capacity_coupled":
                fault, source = _queue(owned, seed)
            else:
                source = (
                    "Bearer " + "x" * 24
                    if case_id == "deny_filter"
                    else "Synthetic rejected input."
                )
                record = _user(source)
                if case_id == "clock_rollback":
                    clock.set(driver.START - timedelta(hours=1))
                if case_id == "sqlite_injected_fault":
                    owned.connection.inject = True
                result = owned.append_record(record)
                owned.connection.inject = False
                fault = {
                    "result": result.value,
                    "source_sha256": digest(
                        canonical_json_bytes(m.evidence_snapshot_to_primitive(record.snapshot))
                    ),
                    "sticky_fault": getattr(owned.diagnostics().sticky_fault, "value", "none"),
                    "injections": owned.connection.injections,
                    "injection_transactions": list(owned.connection.injection_transactions),
                    "rollback_transactions": list(owned.connection.rollback_transactions),
                    "in_transaction": owned.connection.in_transaction,
                }
        finally:
            owned.close()
    after = observe_database(case)
    scan = scan_rejected_source(case, source) if source else {}
    # Exact synthetic deletion fixtures make all six pre-purge file identities
    # observable by the retained parent. They make no journal-recovery claim.
    for name in DATABASE_NAMES[1:]:
        with (case / "evidence" / name).open("xb") as file:
            file.write(b"synthetic-sidecar")
    cleanup_before = inventory(case)
    prepared()
    purger = _spool(case, driver._Clock(driver.START + timedelta(hours=2)))
    try:
        if case_id == "clock_rollback":
            result = purger.recover_existing().value
        else:
            result = purger.purge_full_store(
                m.FullPurgeV1(
                    protocol_version=1,
                    full_purge_generation_id="40000000-0000-4000-8000-000000000096",
                    sentinel_state=m.SentinelState.FULL_PURGE_PENDING,
                    artifact_manifest_version=1,
                )
            ).value
    finally:
        purger.close()
    return {
        "caseId": case_id,
        "before": before,
        "fault": fault,
        "after": after,
        "scan": scan,
        "cleanup": {"before": cleanup_before, "result": result, "after": inventory(case)},
    }


def main() -> None:
    _require(
        os.name == "nt"
        and bool(sys.flags.isolated)
        and not sys.flags.optimize
        and len(sys.argv) == 3,
        "synthetic worker launch differs",
    )
    import msvcrt

    import hermes_realtime

    _require(
        Path(hermes_realtime.__file__)
        .resolve()
        .is_relative_to(Path(sys.path[0]).resolve(strict=True)),
        "synthetic worker imported source instead of wheel",
    )
    handles = tuple(int(value) for value in sys.argv[1:])
    for handle in handles:
        os.set_handle_inheritable(handle, False)
    request = msvcrt.open_osfhandle(handles[0], os.O_RDONLY | os.O_BINARY)
    response = msvcrt.open_osfhandle(handles[1], os.O_WRONLY | os.O_BINARY)
    try:
        config = _read_frame(request, time.monotonic() + 10)
        _require(
            type(config) is dict
            and set(config)
            == {"version", "nonce", "point", "mode", "action", "clock", "workspace"},
            "synthetic configuration differs",
        )
        _require(
            type(config["version"]) is int
            and config["version"] == 1
            and type(config["mode"]) is int
            and config["mode"] == 0
            and config["point"] in CASES_V1
            and config["action"] == "synthetic"
            and config["clock"] == "caught-up"
            and type(config["nonce"]) is str
            and re.fullmatch("[0-9a-f]{64}", config["nonce"]) is not None,
            "synthetic operation differs",
        )
        workspace = Path(config["workspace"]).resolve(strict=True)
        _require(workspace == Path.cwd().resolve(strict=True), "synthetic workspace differs")

        def send(observation: dict[str, Any]) -> None:
            _write_frame(
                response,
                {key: config[key] for key in ("version", "nonce", "point", "mode", "action")}
                | {"pid": os.getpid(), "observation": observation},
            )

        def prepared() -> None:
            send({"phase": "prepared"})
            _require_ack(_read_frame(request, time.monotonic() + 10), config["nonce"], 0)

        failed = False
        try:
            observation = _run(
                workspace / f"{config['point']}-exit0-caught-up", config["point"], prepared
            )
        except Exception as error:
            observation = {
                "failure": "value"
                if isinstance(error, ValueError)
                else "os"
                if isinstance(error, OSError)
                else "other",
                "source_line": traceback.extract_tb(error.__traceback__)[-1].lineno,
            }
            failed = True
        send(observation)
        _require_ack(_read_frame(request, time.monotonic() + 10), config["nonce"], 1)
        _require(not failed, "synthetic worker failed")
    finally:
        os.close(request)
        os.close(response)


if __name__ == "__main__":
    main()
