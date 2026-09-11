"""Independent, bounded observation of the closed synthetic fault fixtures."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _require
from scripts.evidence_protocol_oracle import canonical_json_bytes, hre1_record_hash
from scripts.spool_crash_oracle import (
    SCHEMA_DIGEST_V1,
    _sentinel_image_v1,
    _source_events,
    sentinel_state_v1,
)
from scripts.windows_storage_oracle import audit_storage

CASES_V1 = (
    "queue_capacity_coupled",
    "deny_filter",
    "clock_rollback",
    "sqlite_injected_fault",
    "writer_drain_blocked",
)
DATABASE_NAMES = tuple(
    "capture-v1.sqlite3" + suffix for suffix in ("", "-journal", "-wal", "-shm", "-vacuum", "-tmp")
)
MARKER = ".hermes-realtime-evidence-root-v1"
SENTINEL = "capture-v1.owner"
DECOYS = {
    "purge-decoy.bin": b"not-owned",
    "capture-v1.sqlite3.backup": b"database-prefix-decoy",
    "capture-v1.sqlite3-wal.backup": b"sidecar-prefix-decoy",
}
ADJACENT = {"capture-v1.sqlite3": b"adjacent-database-decoy", "keep.bin": b"adjacent-decoy"}
ALLOWED = frozenset((MARKER, SENTINEL, *DATABASE_NAMES, *DECOYS))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def expected_sentinel_digest(case: str, *, after: bool) -> str:
    """Pin both fixture slots, including the predecessor's exact authority."""
    _require(case in CASES_V1 and type(after) is bool, "synthetic sentinel phase differs")
    if case == "clock_rollback":
        active = (3, 2, "50000000-0000-4000-8000-000000000003")
        clear_generation = 4 if after else 2
    elif after:
        active = (3, 1, "40000000-0000-4000-8000-000000000096")
        clear_generation = 4
    else:
        active = (1, 3, "50000000-0000-4000-8000-000000000002")
        clear_generation = 2
    return digest(_sentinel_image_v1(active, (clear_generation, 0, None)))


def expected_history(count: int) -> list[dict[str, Any]]:
    _require(type(count) is int and count in {3, 4, 68, 69}, "synthetic history bound differs")
    decoded = (json.loads(raw) for raw in _source_events())
    history = sorted(
        (
            row
            for row in decoded
            if row["logical_session_id"] == "10000000-0000-4000-8000-000000000004"
            and row["event_kind"] in {"session_opened", "binding_opened", "turn_opened"}
        ),
        key=lambda item: item["event_sequence"],
    )
    _require(len(history) == 3, "synthetic baseline oracle differs")
    if count >= 4:
        history.append(
            {
                **history[2],
                "event_sequence": 4,
                "event_id": "40000000-0000-4000-8000-000000000011",
                "event_kind": "user_final_accepted",
                "payload": {
                    "utterance_id": "10000000-0000-4000-8000-000000000008",
                    "evidence_turn_id": "10000000-0000-4000-8000-000000000007",
                    "source": "typed",
                    "routing_disposition": "response",
                    "text": "Synthetic baseline.",
                },
            }
        )
    for ordinal in range(1, count - 3):
        history.append(
            {
                **history[2],
                "event_sequence": 4 + ordinal,
                "event_id": f"40000000-0000-4000-8000-{20 + ordinal:012d}",
                "event_kind": "assistant_segment_generated",
                "payload": {
                    "evidence_turn_id": "10000000-0000-4000-8000-000000000007",
                    "evidence_segment_id": f"40000000-0000-4000-8000-{200 + ordinal:012d}",
                    "segment_ordinal": ordinal,
                    "text": f"Synthetic segment {ordinal:02d}.",
                },
            }
        )
    _require(len(history) == count, "synthetic source cardinality differs")
    return history


def rejected_snapshot(case: str) -> dict[str, Any]:
    if case == "queue_capacity_coupled":
        return expected_history(69)[-1]
    row = expected_history(4)[-1]
    row["payload"]["text"] = (
        "Bearer " + "x" * 24 if case == "deny_filter" else "Synthetic rejected input."
    )
    return row


def inventory(case: Path) -> dict[str, Any]:
    with audit_storage(case / "evidence", ALLOWED) as files:
        _require(
            all(p.stat().st_size <= 4 * 1024**2 for p in files), "synthetic file bound differs"
        )
        values = {p.name: digest(p.read_bytes()) for p in files}
        sentinel = sentinel_state_v1((case / "evidence" / SENTINEL).read_bytes())
    with audit_storage(case / "adjacent", frozenset(ADJACENT)) as neighbors:
        adjacent = {p.name: digest(p.read_bytes()) for p in neighbors}
    return {"files": values, "sentinel": sentinel, "adjacent": adjacent}


def _utc(value: str) -> datetime:
    result = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    _require(result.strftime("%Y-%m-%dT%H:%M:%S.%fZ") == value, "synthetic timestamp differs")
    return result


def observe_database(case: Path) -> dict[str, Any]:
    database = case / "evidence" / DATABASE_NAMES[0]
    with audit_storage(database.parent, ALLOWED):
        _require(database.stat().st_size <= 4 * 1024**2, "synthetic database exceeds its bound")
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            _require(
                connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
                "synthetic integrity differs",
            )
            _require(
                not connection.execute("PRAGMA foreign_key_check").fetchall(),
                "synthetic foreign keys differ",
            )
            schema = [
                list(row)
                for row in connection.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
                )
            ]
            _require(
                digest(canonical_json_bytes(schema)) == SCHEMA_DIGEST_V1, "synthetic schema differs"
            )
            _require(
                connection.execute("PRAGMA application_id").fetchone() == (0x48524531,),
                "synthetic application identity differs",
            )
            _require(
                connection.execute("PRAGMA user_version").fetchone() == (1,),
                "synthetic schema version differs",
            )
            installations = connection.execute(
                "SELECT installation_id,purge_required,purge_reason,purge_scope,"
                "created_at_utc,clock_high_water_utc FROM producer_installation"
            ).fetchall()
            epochs = connection.execute(
                "SELECT consent_epoch_id,producer_instance_id,state,"
                "opened_at_utc,closed_at_utc FROM consent_epochs"
            ).fetchall()
            sessions = connection.execute(
                "SELECT logical_session_id,consent_epoch_id,producer_instance_id,state,"
                "event_count,canonical_bytes,taint_code,final_event_sequence,head_hash,"
                "opened_at_utc,expires_at_utc FROM evidence_sessions"
            ).fetchall()
            _require(
                len(installations) == len(epochs) == len(sessions) == 1,
                "synthetic authority cardinality differs",
            )
            installation, epoch, session = installations[0], epochs[0], sessions[0]
            _require(
                installation[0] == "10000000-0000-4000-8000-000000000001"
                and installation[1:4] in {(0, None, None), (1, "clock_rollback", "store")},
                "synthetic installation differs",
            )
            _require(
                epoch[:3]
                == (
                    "10000000-0000-4000-8000-000000000003",
                    "10000000-0000-4000-8000-000000000002",
                    "active",
                )
                and epoch[4] is None,
                "synthetic epoch differs",
            )
            _require(
                session[:3] == ("10000000-0000-4000-8000-000000000004", epoch[0], epoch[1])
                and session[7:9] == (None, None),
                "synthetic session differs",
            )
            _require(
                (session[3], session[6]) in {("open", None), ("tainted", "deny_filter")},
                "synthetic taint differs",
            )
            for table in ("evidence_conflicts", "erasure_requests", "erasure_tombstones"):
                _require(
                    connection.execute("SELECT COUNT(*) FROM " + table).fetchone() == (0,),
                    "synthetic auxiliary history differs",
                )
            events = connection.execute(
                "SELECT event_id,event_sequence,event_kind,canonical_payload,payload_hash,"
                "previous_hash,record_hash,recorded_at_utc,canonical_bytes "
                "FROM evidence_events ORDER BY event_sequence"
            ).fetchall()
            _require(
                len(events) in {3, 4, 68} and session[4] == len(events),
                "synthetic event count differs",
            )
            snapshots, previous, times = [], None, []
            for ordinal, event in enumerate(events, 1):
                eid, sequence, kind, payload, payload_hash, prior, record_hash, at, size = event
                parsed = json.loads(payload)
                raw = canonical_json_bytes(parsed)
                _require(
                    raw == payload.encode("utf-8")
                    and len(raw) == size
                    and digest(raw) == payload_hash
                    and sequence == ordinal
                    and prior == previous,
                    "synthetic canonical record differs",
                )
                _require(
                    hre1_record_hash(
                        installation_id=installation[0],
                        producer_instance_id=epoch[1],
                        event_id=eid,
                        logical_session_id=session[0],
                        event_sequence=sequence,
                        event_kind=kind,
                        recorded_at_utc=at,
                        payload_hash=payload_hash,
                        previous_hash=prior,
                    )
                    == record_hash,
                    "synthetic record chain differs",
                )
                snapshots.append(
                    {
                        "schema_version": 1,
                        "installation_id": installation[0],
                        "producer_instance_id": epoch[1],
                        "logical_session_id": session[0],
                        "event_id": eid,
                        "event_sequence": sequence,
                        "event_kind": kind,
                        "payload": parsed,
                    }
                )
                previous = record_hash
                times.append(_utc(at))
            _require(
                canonical_json_bytes(snapshots)
                == canonical_json_bytes(expected_history(len(events))),
                "synthetic persisted source differs",
            )
            _require(
                session[5] == sum(e[8] for e in events) and times == sorted(times),
                "synthetic accounting differs",
            )
            _require(
                _utc(installation[4]) == _utc(epoch[3]) == _utc(session[9]) == times[0] == times[1]
                and _utc(session[10]) == times[0] + timedelta(hours=24)
                and _utc(installation[5]) >= times[-1],
                "synthetic time authority differs",
            )
    return {
        "events": len(events),
        "source_sha256": digest(canonical_json_bytes(snapshots)),
        "state": session[3],
        "taint": session[6] or "none",
        "purge_required": installation[1],
    }


def scan_rejected_source(case: Path, source: str) -> dict[str, int]:
    needles = (source.encode("utf-8"), source.encode("utf-16le"))
    with audit_storage(case / "evidence", ALLOWED) as files:
        _require(
            all(p.stat().st_size <= 4 * 1024**2 for p in files), "synthetic scan bound differs"
        )
        contents = (p.read_bytes() for p in files)
        matches = sum(any(needle in raw for needle in needles) for raw in contents)
        return {"files": len(files), "matches": matches}
