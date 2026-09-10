"""Bounded, content-free inspection independent of the spool recovery methods."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _require


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _database_state(database: Path) -> dict[str, Any]:
    from hermes_realtime.evidence.models import parse_evidence_snapshot_json
    from hermes_realtime.evidence.sqlite_spool import canonical_json_bytes, hre1_record_hash

    # SQLite may roll back its own hot rollback journal on this cold open. No
    # application SQL writes or production recovery methods run in this observer.
    with closing(sqlite3.connect(database.as_uri() + "?mode=rw", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        _require(
            connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
            "storage integrity differs",
        )
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not tables:
            return {"schema": False}
        _require(
            not connection.execute("PRAGMA foreign_key_check").fetchall(),
            "storage foreign keys differ",
        )
        installations = connection.execute(
            "SELECT installation_id,purge_required FROM producer_installation"
        ).fetchall()
        _require(len(installations) <= 1, "storage installation count differs")
        epochs = connection.execute(
            "SELECT consent_epoch_id,producer_instance_id,state FROM consent_epochs ORDER BY rowid"
        ).fetchall()
        rows = connection.execute(
            "SELECT logical_session_id,consent_epoch_id,producer_instance_id,state,event_count,"
            "canonical_bytes,final_event_sequence,head_hash,taint_code "
            "FROM evidence_sessions ORDER BY rowid"
        ).fetchall()
        _require(len(rows) <= 2 and len(epochs) <= 2, "storage history exceeds its bound")
        total = 0
        seals = []
        for session in rows:
            sid, epoch, producer, state, count, size, final, head, taint = session
            events = connection.execute(
                "SELECT event_id,event_sequence,event_kind,canonical_payload,payload_hash,"
                "previous_hash,record_hash,recorded_at_utc,canonical_bytes FROM evidence_events "
                "WHERE logical_session_id=? ORDER BY event_sequence",
                (sid,),
            ).fetchall()
            _require(
                len(installations) == 1 and 2 <= len(events) <= 4, "storage event count differs"
            )
            previous = None
            payloads = []
            for ordinal, event in enumerate(events, 1):
                eid, sequence, kind, payload, payload_hash, prior, digest, at, n = event
                parsed = json.loads(payload)
                raw = canonical_json_bytes(parsed)
                _require(
                    raw == payload.encode("utf-8")
                    and len(raw) == n
                    and sequence == ordinal
                    and prior == previous
                    and _digest(raw) == payload_hash
                    and hre1_record_hash(
                        installation_id=installations[0][0],
                        producer_instance_id=producer,
                        event_id=eid,
                        logical_session_id=sid,
                        event_sequence=sequence,
                        event_kind=kind,
                        recorded_at_utc=at,
                        payload_hash=payload_hash,
                        previous_hash=previous,
                    )
                    == digest,
                    "storage event chain differs",
                )
                parse_evidence_snapshot_json(
                    canonical_json_bytes(
                        {
                            "schema_version": 1,
                            "installation_id": installations[0][0],
                            "producer_instance_id": producer,
                            "logical_session_id": sid,
                            "event_id": eid,
                            "event_sequence": sequence,
                            "event_kind": kind,
                            "payload": parsed,
                        }
                    )
                )
                previous = digest
                payloads.append(parsed)
            _require(
                count == len(events) and size == sum(e[8] for e in events),
                "storage aggregates differ",
            )
            _require(
                (state == "open" and final is head is taint is None)
                or (state == "sealed" and final == count and head == previous and taint is None),
                "storage partial seal is present",
            )
            if state == "sealed":
                _require(
                    [e[2] for e in events]
                    == [
                        "session_opened",
                        "binding_opened",
                        "binding_closed",
                        "session_seal_requested",
                    ]
                    and payloads[-1]["final_event_sequence"] == count
                    and payloads[-2]["binding_id"] == payloads[0]["binding_id"]
                    and all(
                        payloads[-1][k] == payloads[0][k]
                        for k in ("consent_epoch_id", "consent_version", "disclosure_digest")
                    )
                    and (epoch, producer, "closed") in epochs,
                    "storage complete seal lineage differs",
                )
                seals.append(
                    _digest(canonical_json_bytes([list(session), [list(e) for e in events]]))
                )
            total += len(events)
        _require(
            connection.execute("SELECT COUNT(*) FROM evidence_events").fetchone() == (total,),
            "storage has orphan events",
        )
        tombstones = connection.execute(
            "SELECT reason_code,erased_session_count,erased_event_count "
            "FROM erasure_tombstones ORDER BY rowid"
        ).fetchall()
        _require(len(tombstones) <= 2, "storage tombstone bound differs")
        erasures = connection.execute(
            "SELECT state FROM erasure_requests ORDER BY rowid"
        ).fetchall()
        _require(len(erasures) <= 2, "storage erasure request bound differs")
        return {
            "schema": True,
            "events": total,
            "sessions": [r[3] for r in rows],
            "epochs": [r[2] for r in epochs],
            "seals": seals,
            "purge_required": installations[0][1] if installations else -1,
            "tombstones": [list(row) for row in tombstones],
            "erasures": [row[0] for row in erasures],
            "logical_digest": _digest("\n".join(connection.iterdump()).encode("utf-8")),
        }


def observe_storage(root: Path) -> dict[str, Any]:
    from hermes_realtime.evidence.storage_security import (
        EvidenceArtifactManifestV1,
        WindowsStorageProbeV1,
        decode_sentinel_image,
        parse_root_marker,
    )

    manifest = EvidenceArtifactManifestV1()
    probe = WindowsStorageProbeV1()
    _require(
        probe.platform_is_supported() and probe.volume_is_fixed_local(root),
        "storage is not fixed local Windows",
    )
    _require(probe.path_grants_only_owner(root), "storage root DACL differs")
    files = {}
    allowed = set(manifest.all_names) | {
        "recreation-decoy.bin",
        "rollback-decoy.bin",
        "purge-decoy.bin",
    }
    for path in root.iterdir():
        _require(
            path.name in allowed
            and path.is_file()
            and not probe.path_has_reparse_point(path)
            and not probe.path_has_alternate_data_streams(path)
            and probe.path_grants_only_owner(path)
            and path.stat().st_size <= 4 * 1024 * 1024,
            "storage artifact is outside the bounded safe inventory",
        )
        files[path.name] = _digest(path.read_bytes())
    sentinel = root / manifest.sentinel
    state = (
        decode_sentinel_image(sentinel.read_bytes()).active.state.value
        if sentinel.exists()
        else "no_final_sentinel"
    )
    marker = root / manifest.root_marker
    if marker.exists():
        parse_root_marker(marker.read_bytes())
    database = root / manifest.database
    readable = database.exists() and state not in {"full_purge_pending"}
    db = _database_state(database) if readable else {"schema": False}
    return {"files": files, "sentinel": state, "database": db}
