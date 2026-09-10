"""Full-purge acceptance requires exact deletion and an unchanged repeated purge."""

from __future__ import annotations

import os
from copy import deepcopy
from hashlib import sha256

import pytest


def _observations():
    from scripts.spool_crash_oracle import (
        checkpoint_installation_digest_v1,
        checkpoint_sentinel_digest_v1,
        checkpoint_source_digest_v1,
        initialization_digest_v1,
    )

    marker, sentinel, database = (
        ".hermes-realtime-evidence-root-v1",
        "capture-v1.owner",
        "capture-v1.sqlite3",
    )
    decoys = {
        "purge-decoy.bin": sha256(b"not-owned").hexdigest(),
        "capture-v1.sqlite3.backup": sha256(b"database-prefix-decoy").hexdigest(),
        "capture-v1.sqlite3-wal.backup": sha256(b"sidecar-prefix-decoy").hexdigest(),
    }
    adjacent = {
        "capture-v1.sqlite3": sha256(b"adjacent-database-decoy").hexdigest(),
        "keep.bin": sha256(b"adjacent-decoy").hexdigest(),
    }
    before = {
        "files": {
            marker: initialization_digest_v1("root_marker", "full_write"),
            sentinel: checkpoint_sentinel_digest_v1(0, "caught-up", after=False),
            database: "d" * 64,
            **{
                database + suffix: sha256(b"owned-sidecar").hexdigest()
                for suffix in ("-journal", "-wal", "-shm", "-vacuum", "-tmp")
            },
            **decoys,
        },
        "sentinel": "clear",
        "adjacent": adjacent,
        "database": {
            "schema": True,
            "events": 2,
            "sessions": ["open"],
            "epochs": ["active"],
            "source_digest": checkpoint_source_digest_v1(0),
            "installation_digest": checkpoint_installation_digest_v1(0),
            "purge_required": 0,
            "seals": [],
            "tombstones": [],
            "erasures": [],
            "logical_digest": "e" * 64,
        },
    }
    after = {
        "files": {
            marker: before["files"][marker],
            sentinel: checkpoint_sentinel_digest_v1(40, "caught-up", after=True),
            **decoys,
        },
        "sentinel": "clear",
        "database": {"schema": False},
        "adjacent": adjacent,
    }
    return [
        {"phase": "purge", "before": before, "result": "purge_completed", "after": after},
        {
            "phase": "repeat_purge",
            "before": deepcopy(after),
            "result": "already_absent",
            "after": deepcopy(after),
        },
    ]


def test_registration_preserves_the_governed_ordinal_and_refuses_forged_receipts() -> None:
    from scripts.full_purge_cleanup import ObservedFullPurgeV1, validate_full_purge_v1
    from scripts.qualify_evidence_slice_zero import (
        FULL_PURGE_REGISTRATION_V1,
        SCENARIO_REGISTRY_V1,
        FullPurgeRegistrationV1,
    )

    assert SCENARIO_REGISTRY_V1[18] is FULL_PURGE_REGISTRATION_V1
    with pytest.raises(ValueError, match="canonical"):
        FullPurgeRegistrationV1().produce(None, None, None)
    with pytest.raises(TypeError):
        validate_full_purge_v1({"purged": True})
    with pytest.raises(ValueError, match="unregistered"):
        validate_full_purge_v1(object.__new__(ObservedFullPurgeV1))


@pytest.mark.parametrize("phase", ["purge", "repeat_purge"])
def test_normal_exit_cannot_substitute_for_live_storage_release(phase) -> None:
    from types import SimpleNamespace

    from scripts.qualify_evidence_slice_zero import canonical_json_bytes
    from scripts.storage_process import _StorageInvocation, _validate_invocation

    record = _StorageInvocation(
        canonical_json_bytes({"phase": phase}),
        SimpleNamespace(process_handle=71),
        SimpleNamespace(
            closed=True,
            zero_active_observed=True,
            failures=(),
            failed_handles=(),
            waited_handles=(71,),
        ),
        0,
        0,
    )
    with pytest.raises(ValueError, match="storage release observation differs"):
        _validate_invocation(record)


@pytest.mark.parametrize(
    "action,mode,point,clock",
    [
        ("purge", 197, "full_purge_cleanup", "caught-up"),
        ("purge", False, "full_purge_cleanup", "caught-up"),
        ("crash", 0, "before_begin", "caught-up"),
        ("purge", 0, "before_begin", "caught-up"),
        ("repeat_purge", 0, "full_purge_cleanup", "regressed"),
    ],
)
def test_purge_and_crash_launch_authorities_cannot_be_confused(action, mode, point, clock) -> None:
    from scripts.storage_process import _run_storage_worker

    with pytest.raises(
        ValueError, match="storage exit mode differs|full-purge worker configuration"
    ):
        _run_storage_worker(None, action=action, mode=mode, point=point, clock=clock)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "sidecar",
        "prefix",
        "decoy",
        "adjacent",
        "marker",
        "sentinel",
        "entry",
        "result",
        "repeat_result",
        "repeat_mutation",
        "missing_phase",
        "source",
        "schema",
    ],
)
def test_acceptance_requires_exact_cleanup_and_durable_idempotence(mutation) -> None:
    from scripts.full_purge_cleanup import _validate_observations

    rows = _observations()
    if mutation == "sidecar":
        rows[0]["after"]["files"]["capture-v1.sqlite3-journal"] = "f" * 64
    elif mutation in {"prefix", "decoy"}:
        del rows[0]["after"]["files"][
            "capture-v1.sqlite3.backup" if mutation == "prefix" else "purge-decoy.bin"
        ]
    elif mutation == "adjacent":
        rows[0]["after"]["adjacent"] = {}
    elif mutation in {"marker", "sentinel"}:
        rows[0]["after"]["files"][
            ".hermes-realtime-evidence-root-v1" if mutation == "marker" else "capture-v1.owner"
        ] = "f" * 64
    elif mutation == "entry":
        del rows[0]["before"]["files"]["capture-v1.sqlite3-wal"]
    elif mutation in {"result", "repeat_result"}:
        rows[mutation == "repeat_result"]["result"] = "purge_failed"
    elif mutation == "repeat_mutation":
        rows[1]["after"]["files"]["capture-v1.owner"] = "f" * 64
    elif mutation == "missing_phase":
        rows.pop()
    elif mutation == "source":
        rows[0]["before"]["database"]["source_digest"] = "f" * 64
    elif mutation == "schema":
        rows[0]["before"]["database"]["schema"] = False
    if mutation is None:
        _validate_observations(rows)
    else:
        with pytest.raises(ValueError):
            _validate_observations(rows)


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows storage and process ownership")
@pytest.mark.parametrize("fault", [None, "no_delete", "repeat_mutation", "unclosed"])
def test_owned_workers_observe_real_purge_and_reject_false_completion(tmp_path, fault) -> None:
    import shutil
    from pathlib import Path

    from scripts.full_purge_cleanup import _PurgeRun, _validate_run
    from scripts.storage_process import _run_storage_worker, _StorageArchive

    source = Path(__file__).resolve().parents[1]
    package = tmp_path / "package"
    shutil.copytree(source / "src/hermes_realtime", package / "hermes_realtime")
    if fault is not None:
        patch = {
            "no_delete": (
                "\ndef _fake_purge(self, command):\n"
                "    return PurgeDisposition.PURGE_COMPLETED\n"
                "SQLiteEvidenceSpool.purge_full_store = _fake_purge\n"
            ),
            "repeat_mutation": (
                "\n_purge = SQLiteEvidenceSpool.purge_full_store\n"
                "def _fake_purge(self, command):\n"
                "    result = _purge(self, command)\n"
                "    if result is PurgeDisposition.ALREADY_ABSENT:\n"
                "        (self._database.parent / 'purge-decoy.bin').write_bytes(b'changed')\n"
                "    return result\n"
                "SQLiteEvidenceSpool.purge_full_store = _fake_purge\n"
            ),
            "unclosed": (
                "\n_purge = SQLiteEvidenceSpool.purge_full_store\n"
                "_close = SQLiteEvidenceSpool.close\n"
                "def _fake_purge(self, command):\n"
                "    result = _purge(self, command)\n"
                "    self._retain_after_purge = True\n"
                "    return result\n"
                "def _fake_close(self):\n"
                "    if not getattr(self, '_retain_after_purge', False):\n"
                "        _close(self)\n"
                "SQLiteEvidenceSpool.purge_full_store = _fake_purge\n"
                "SQLiteEvidenceSpool.close = _fake_close\n"
            ),
        }[fault]
        with (package / "hermes_realtime/evidence/sqlite_spool.py").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(patch)
    archive = _StorageArchive(tmp_path, source, package, None, "0" * 64)

    def qualify():
        invocations = tuple(
            _run_storage_worker(archive, point="full_purge_cleanup", mode=0, action=phase)
            for phase in ("purge", "repeat_purge")
        )
        return _validate_run(_PurgeRun("a" * 40, "b" * 40, "c" * 64, "d" * 64, invocations))

    if fault is None:
        evidence = qualify()
        assert evidence.process_count == 2
        assert evidence.machine_assertions == (
            "adjacent_decoys_preserved",
            "purge_verified",
            "purged",
        )
    else:
        with pytest.raises(
            ValueError,
            match={
                "no_delete": "full-purge disposition differs",
                "repeat_mutation": "repeated full purge mutated storage",
                "unclosed": "storage ownership was not released",
            }[fault],
        ):
            qualify()
