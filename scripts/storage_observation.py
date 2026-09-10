"""Bounded, content-free inspection independent of the spool recovery methods."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _require
from scripts.evidence_protocol_oracle import canonical_json_bytes, hre1_record_hash
from scripts.spool_crash_oracle import (
    SCHEMA_DIGEST_V1,
    initialization_digest_v1,
    sentinel_state_v1,
    validate_spool_snapshot_v1,
)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _database_state(database: Path, *, include_journal: bool = True) -> dict[str, Any]:
    """Read a disposable image; only production recovery opens the crash files."""
    sources = [database]
    journal = database.with_name(database.name + "-journal")
    if include_journal and journal.exists():
        sources.append(journal)
    images = {}
    for source in sources:
        _require(source.stat().st_size <= 4 * 1024 * 1024, "database image exceeds its bound")
        images[source.name] = source.read_bytes()
    parent = database.parent.parent.resolve(strict=True)
    snapshot = Path(tempfile.mkdtemp(prefix="hermes-storage-observer-", dir=parent))
    try:
        for name, raw in images.items():
            (snapshot / name).write_bytes(raw)
        return _read_database_snapshot(snapshot / database.name)
    finally:
        _require(
            snapshot.resolve(strict=True).parent == parent and not snapshot.is_symlink(),
            "database snapshot cleanup escaped its owner",
        )
        shutil.rmtree(snapshot)
        _require(
            all(
                source.is_file() and source.read_bytes() == images[source.name]
                for source in sources
            ),
            "database observation changed the original crash image",
        )


def _read_database_snapshot(database: Path) -> dict[str, Any]:
    # SQLite may recover a hot journal here, inside the disposable copy only.
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
            tables
            == {
                "producer_installation",
                "consent_epochs",
                "evidence_sessions",
                "evidence_events",
                "evidence_conflicts",
                "erasure_requests",
                "erasure_tombstones",
            }
            and connection.execute("PRAGMA application_id").fetchone() == (0x48524531,)
            and connection.execute("PRAGMA user_version").fetchone() == (1,),
            "storage schema version or table inventory differs",
        )
        schema = [
            list(row)
            for row in connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
            )
        ]
        _require(
            _digest(canonical_json_bytes(schema)) == SCHEMA_DIGEST_V1
            and connection.execute("PRAGMA page_size").fetchone() == (4096,)
            and connection.execute("PRAGMA encoding").fetchone() == ("UTF-8",)
            and connection.execute("PRAGMA auto_vacuum").fetchone() == (0,)
            and connection.execute("PRAGMA journal_mode").fetchone() == ("delete",),
            "storage committed schema differs from the independent V1 image",
        )
        _require(
            not connection.execute("PRAGMA foreign_key_check").fetchall(),
            "storage foreign keys differ",
        )
        installations = connection.execute(
            "SELECT installation_id,purge_required,singleton,purge_reason,purge_scope,"
            "created_at_utc,clock_high_water_utc "
            "FROM producer_installation"
        ).fetchall()
        _require(len(installations) <= 1, "storage installation count differs")
        for (
            installation,
            required,
            singleton,
            reason,
            scope,
            _created,
            _high_water,
        ) in installations:
            _require(
                installation == "10000000-0000-4000-8000-000000000001"
                and singleton == 1
                and (required, reason, scope) in {(0, None, None), (1, "clock_rollback", "store")},
                "storage installation authority differs",
            )
        epochs = connection.execute(
            "SELECT consent_epoch_id,producer_instance_id,state,opened_at_utc,closed_at_utc "
            "FROM consent_epochs ORDER BY rowid"
        ).fetchall()
        rows = connection.execute(
            "SELECT logical_session_id,consent_epoch_id,producer_instance_id,state,event_count,"
            "canonical_bytes,final_event_sequence,head_hash,taint_code,opened_at_utc,expires_at_utc,"
            "consent_version "
            "FROM evidence_sessions ORDER BY rowid"
        ).fetchall()
        _require(len(rows) <= 2 and len(epochs) <= 2, "storage history exceeds its bound")
        _require(
            len(rows) == len(epochs)
            and {row[1] for row in rows} == {epoch[0] for epoch in epochs}
            and connection.execute("SELECT COUNT(*) FROM evidence_conflicts").fetchone() == (0,),
            "storage has orphan epochs or unexpected conflicts",
        )
        total = 0
        seals = []
        source_histories = []
        for session in rows:
            (
                sid,
                epoch,
                producer,
                state,
                count,
                size,
                final,
                head,
                taint,
                opened,
                expires,
                consent,
            ) = session
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
            source_events = []
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
                snapshot = {
                    "schema_version": 1,
                    "installation_id": installations[0][0],
                    "producer_instance_id": producer,
                    "logical_session_id": sid,
                    "event_id": eid,
                    "event_sequence": sequence,
                    "event_kind": kind,
                    "payload": parsed,
                }
                validate_spool_snapshot_v1(snapshot)
                source_events.append({"snapshot": snapshot, "recorded_at_utc": at})
                previous = digest
                payloads.append(parsed)
            opening = payloads[0]
            moment = datetime.strptime(events[0][7], "%Y-%m-%dT%H:%M:%S.%fZ")
            expected_expiry = (moment + timedelta(hours=opening["retention_hours"])).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            epoch_row = next(row for row in epochs if row[0] == epoch)
            _require(
                epoch == opening["consent_epoch_id"]
                and consent == opening["consent_version"]
                and opened == events[0][7] == epoch_row[3]
                and expires == expected_expiry
                and producer == epoch_row[1],
                "storage persisted consent lineage differs from its source",
            )
            for ordinal, event in enumerate(events):
                expected_at = (moment + timedelta(seconds=max(0, ordinal - 1))).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                _require(event[7] == expected_at, "storage fixture event chronology differs")
            _require(
                (state == "sealed" and epoch_row[2:] == ("closed", opened, events[-1][7]))
                or (
                    state == "open"
                    and epoch_row[2:]
                    in {
                        ("active", opened, None),
                        ("revoked", opened, "2026-08-08T00:00:01.000000Z"),
                    }
                ),
                "storage epoch lifecycle differs from its source",
            )
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
                    and (epoch, producer, "closed", opened, events[-1][7]) in epochs,
                    "storage complete seal lineage differs",
                )
                seals.append(
                    _digest(
                        canonical_json_bytes(
                            [list(session), list(epoch_row), [list(e) for e in events]]
                        )
                    )
                )
            total += len(events)
            source_histories.append(source_events)
        _require(
            connection.execute("SELECT COUNT(*) FROM evidence_events").fetchone() == (total,),
            "storage has orphan events",
        )
        tombstones = connection.execute(
            "SELECT erasure_request_id,scope_kind,scope_key,reason_code,control_sequence,"
            "control_fingerprint_hash,ttl_consent_epoch_id,ttl_expires_at_utc,"
            "last_admission_ordinal,final_admission_ordinal,erased_at_utc,"
            "erased_session_count,erased_event_count "
            "FROM erasure_tombstones ORDER BY rowid"
        ).fetchall()
        _require(len(tombstones) <= 2, "storage tombstone bound differs")
        erasures = connection.execute(
            "SELECT erasure_request_id,scope_kind,scope_key,requested_at_utc,reason_code,"
            "state,control_sequence,control_fingerprint_hash,ttl_consent_epoch_id,"
            "ttl_expires_at_utc,last_admission_ordinal,final_admission_ordinal,resume_state,"
            "erased_session_count,erased_event_count FROM erasure_requests ORDER BY rowid"
        ).fetchall()
        _require(len(erasures) <= 2, "storage erasure request bound differs")
        return {
            "schema": True,
            "events": total,
            "sessions": [r[3] for r in rows],
            "epochs": [r[2] for r in epochs],
            "seals": seals,
            "source_digest": _digest(canonical_json_bytes(source_histories)),
            "installation_digest": _digest(
                canonical_json_bytes(list(installations[0]) if installations else [])
            ),
            "purge_required": installations[0][1] if installations else -1,
            "tombstones": [
                [row[3], row[11], row[12], _digest(canonical_json_bytes(list(row)))]
                for row in tombstones
            ],
            "erasures": [[row[5], _digest(canonical_json_bytes(list(row)))] for row in erasures],
            "logical_digest": _digest("\n".join(connection.iterdump()).encode("utf-8")),
        }


def observe_storage(root: Path) -> dict[str, Any]:
    from hermes_realtime.evidence.storage_security import (
        EvidenceArtifactManifestV1,
        WindowsStorageProbeV1,
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
    state = sentinel_state_v1(sentinel.read_bytes()) if sentinel.exists() else "no_final_sentinel"
    marker = root / manifest.root_marker
    if marker.exists():
        _require(
            _digest(marker.read_bytes()) == initialization_digest_v1("root_marker", "full_write"),
            "root marker differs from its independent fixture image",
        )
    database = root / manifest.database
    # Full-purge fixtures deliberately seed non-SQLite sidecar bytes. Their
    # committed database is inspected alone, before production deletes it.
    db = (
        _database_state(database, include_journal=state != "full_purge_pending")
        if database.exists()
        else {"schema": False}
    )
    return {"files": files, "sentinel": state, "database": db}
