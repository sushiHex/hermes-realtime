"""Observe the closed full-purge fixture without candidate parsing or probe code."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scripts.spool_crash_oracle import sentinel_state_v1
from scripts.storage_observation import _database_state
from scripts.windows_storage_oracle import audit_storage


def observe_full_purge(case: Path) -> dict[str, Any]:
    root = case / "evidence"
    database = root / "capture-v1.sqlite3"
    allowed = frozenset(
        {
            ".hermes-realtime-evidence-root-v1",
            "capture-v1.owner",
            "purge-decoy.bin",
            "capture-v1.sqlite3.backup",
            "capture-v1.sqlite3-wal.backup",
            *(
                "capture-v1.sqlite3" + suffix
                for suffix in ("", "-journal", "-wal", "-shm", "-vacuum", "-tmp")
            ),
        }
    )
    with (
        audit_storage(root, allowed) as paths,
        audit_storage(
            case / "adjacent", frozenset({"capture-v1.sqlite3", "keep.bin"})
        ) as neighbors,
    ):
        return {
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "sentinel": sentinel_state_v1((root / "capture-v1.owner").read_bytes()),
            # These five sidecars are deliberately synthetic deletion inputs;
            # this scenario makes no journal-recovery claim.
            "database": _database_state(database, include_journal=False)
            if database.exists()
            else {"schema": False},
            "adjacent": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in neighbors},
        }
