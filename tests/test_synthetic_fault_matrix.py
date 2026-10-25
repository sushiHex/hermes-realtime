"""Acceptance needs real fault observations and a producer-minted receipt."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest

CASES = (
    "queue_capacity_coupled",
    "deny_filter",
    "clock_rollback",
    "sqlite_injected_fault",
    "writer_drain_blocked",
)


def test_canonical_registration_rejects_foreign_inputs_and_forged_receipts():
    from scripts.qualify_evidence_slice_zero import (
        SCENARIO_REGISTRY_V1,
        SYNTHETIC_FAULT_REGISTRATION_V1,
        SyntheticFaultRegistrationV1,
    )
    from scripts.synthetic_fault_matrix import (
        ObservedSyntheticFaultV1,
        validate_synthetic_fault_v1,
    )

    assert SCENARIO_REGISTRY_V1[15] is SYNTHETIC_FAULT_REGISTRATION_V1
    assert SYNTHETIC_FAULT_REGISTRATION_V1.scenario_id.value == "synthetic_fault_matrix"
    with pytest.raises(ValueError, match="canonical"):
        SyntheticFaultRegistrationV1().produce(None, None, None)
    with pytest.raises(TypeError):
        validate_synthetic_fault_v1({"passed": True})
    with pytest.raises(TypeError):
        ObservedSyntheticFaultV1()
    with pytest.raises(ValueError, match="unregistered"):
        validate_synthetic_fault_v1(object.__new__(ObservedSyntheticFaultV1))
    with pytest.raises(TypeError):
        SyntheticFaultRegistrationV1(proof_class="physical")


@pytest.mark.parametrize(
    "rows",
    [None, {}, [], [{"caseId": case, "passed": True} for case in CASES]],
)
def test_case_labels_and_supplied_success_cannot_establish_observations(rows):
    from scripts.synthetic_fault_matrix import _validate_observations

    with pytest.raises((TypeError, ValueError)):
        _validate_observations(rows)


def test_assertions_and_case_order_match_the_governed_schema():
    import json
    from pathlib import Path

    from scripts.synthetic_fault_matrix import ASSERTIONS_V1

    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "scripts/schemas/qualification-report-v1.schema.json").read_bytes())
    row = schema["properties"]["scenarios"]["prefixItems"][15]["allOf"][1]["properties"]
    assert row["scenarioId"]["const"] == "synthetic_fault_matrix"
    assert row["proofClass"]["const"] == "synthetic_injected"
    assert tuple(item["const"] for item in row["machineAssertions"]["prefixItems"]) == ASSERTIONS_V1
    assert (
        tuple(
            item["allOf"][1]["properties"]["caseId"]["const"]
            for item in row["caseResults"]["prefixItems"]
        )
        == CASES
    )


@pytest.mark.parametrize(
    ("case_index", "path", "replacement"),
    [
        (0, ("fault", "reason"), "canonical_byte_capacity"),
        (0, ("fault", "limits", "maxQueueRecords"), 63),
        (0, ("fault", "before"), [64, 2097152, 64]),
        (0, ("fault", "after"), [63, 31666, 64]),
        (0, ("fault", "drained"), [0, 1, 0]),
        (0, ("fault", "source_sha256"), "0" * 64),
        (0, ("after", "source_sha256"), "0" * 64),
        (0, ("after", "events"), 67),
        (0, ("scan", "matches"), 1),
        (0, ("scan", "files"), 0),
        (1, ("after", "taint"), "none"),
        (1, ("fault", "result"), "committed"),
        (1, ("fault", "injections"), False),
        (2, ("after", "purge_required"), 0),
        (2, ("after", "purge_required"), True),
        (2, ("cleanup", "before", "sentinel"), "clear"),
        (2, ("cleanup", "result"), "already_absent"),
        (3, ("fault", "injections"), 0),
        (3, ("fault", "in_transaction"), True),
        (3, ("fault", "sticky_fault"), "none"),
        (4, ("fault", "stages"), ["drain_entered", "drain_stopped", "joined"]),
        (4, ("fault", "outcomes"), ["already_queued"]),
        (4, ("cleanup", "after", "sentinel"), "full_purge_pending"),
        (4, ("cleanup", "after", "adjacent", "keep.bin"), "0" * 64),
        (0, ("cleanup", "before", "files", "capture-v1.owner"), "0" * 64),
        (0, ("cleanup", "after", "files", "capture-v1.owner"), "0" * 64),
    ],
)
def test_changed_fault_source_capacity_or_cleanup_is_rejected(
    observed_cases, case_index, path, replacement
):
    from scripts.synthetic_fault_matrix import _validate_observations

    rows = deepcopy(observed_cases)
    target = rows[case_index]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises((TypeError, ValueError)):
        _validate_observations(rows)


def test_observations_exclude_source_text_paths_and_fixture_identifiers(observed_cases):
    import json

    rendered = json.dumps(observed_cases)
    for marker in (
        "Synthetic baseline.",
        "Synthetic segment",
        "Synthetic rejected input.",
        "Bearer " + "x" * 24,
        "10000000-0000-4000-8000",
        "40000000-0000-4000-8000",
        "C:\\",
        "workspace",
        "nonce",
    ):
        assert marker not in rendered


@pytest.fixture(scope="module")
def observed_cases(tmp_path_factory):
    if os.name != "nt":
        pytest.skip("real Windows storage observations")
    from scripts.synthetic_fault_worker import _run

    workspace = tmp_path_factory.mktemp("synthetic-fault-observations")
    return [_run(workspace / case, case, lambda: None) for case in CASES]


def test_all_five_real_fault_paths_supply_independently_valid_observations(observed_cases):
    from scripts.synthetic_fault_matrix import _validate_observations

    measurements = _validate_observations(observed_cases)
    assert measurements["queueRecordCount"] == 64
    assert measurements["maxQueueRecords"] == 64
    assert measurements["maxQueuePhysicalItems"] == 64
    assert measurements["maxCanonicalRecordBytes"] == 32768
    assert measurements["maxQueueCanonicalBytes"] == 2097152
    assert measurements["queueCanonicalBytes"] < measurements["maxQueueCanonicalBytes"]
    assert measurements["queuePhysicalCount"] <= measurements["maxQueuePhysicalItems"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "extra_flag"])
def test_real_observations_cannot_be_omitted_reordered_or_extended(observed_cases, mutation):
    from scripts.synthetic_fault_matrix import _validate_observations

    rows = deepcopy(observed_cases)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["passed"] = True
    with pytest.raises((TypeError, ValueError)):
        _validate_observations(rows)


@pytest.mark.parametrize("case", CASES)
def test_normal_exit_without_live_storage_release_is_not_evidence(case):
    from types import SimpleNamespace

    from scripts.qualify_evidence_slice_zero import canonical_json_bytes
    from scripts.storage_process import _StorageInvocation, _validate_invocation

    record = _StorageInvocation(
        canonical_json_bytes({"caseId": case}),
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
    ("point", "mode", "clock"),
    [
        (CASES[0], 197, "caught-up"),
        (CASES[0], False, "caught-up"),
        ("full_purge_cleanup", 0, "caught-up"),
        (CASES[0], 0, "regressed"),
    ],
)
def test_synthetic_launch_cannot_borrow_crash_or_recovery_authority(point, mode, clock):
    from scripts.storage_process import _run_storage_worker

    with pytest.raises(
        ValueError, match="storage exit mode differs|synthetic worker configuration"
    ):
        _run_storage_worker(None, action="synthetic", mode=mode, point=point, clock=clock)


@pytest.mark.skipif(os.name != "nt", reason="requires retained Windows process/storage identities")
@pytest.mark.parametrize(
    "fault",
    [
        None, "no_delete", "unclosed", "ignore_sqlite", "no_transaction",
        "commit_instead_of_rollback", "wrong_purge_authority", "wrong_initial_authority",
        "earlier_rollback", "commit_then_other_rollback",
    ],
)
def test_owned_processes_refuse_false_purge_release_and_transaction_success(tmp_path, fault):
    import shutil
    from pathlib import Path

    from scripts.storage_process import _run_storage_worker, _StorageArchive
    from scripts.synthetic_fault_matrix import ASSERTIONS_V1, _FaultRun, _validate_run

    source = Path(__file__).resolve().parents[1]
    package = tmp_path / "package"
    shutil.copytree(
        source / "src/hermes_realtime",
        package / "hermes_realtime",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    patches = {
        "no_delete": (
            "\ndef _fake_purge(self, command):\n"
            "    return PurgeDisposition.PURGE_COMPLETED\n"
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
        "ignore_sqlite": (
            "\n_append = SQLiteEvidenceSpool.append_record\n"
            "def _fake_append(self, record):\n"
            "    result = _append(self, record)\n"
            "    if result is StoreDisposition.FAULTED:\n"
            "        return StoreDisposition.COMMITTED\n"
            "    return result\n"
            "SQLiteEvidenceSpool.append_record = _fake_append\n"
        ),
        "no_transaction": (
            "\n_append = SQLiteEvidenceSpool.append_record\n"
            "def _fake_append(self, record):\n"
            "    if self.connection.inject:\n"
            "        try:\n"
            "            self.connection.execute('INSERT INTO evidence_events')\n"
            "        except sqlite3.Error:\n"
            "            self._rollback()\n"
            "            self._latch(WriterFault.SQLITE_FAULT)\n"
            "            return StoreDisposition.FAULTED\n"
            "    return _append(self, record)\n"
            "SQLiteEvidenceSpool.append_record = _fake_append\n"
        ),
        "commit_instead_of_rollback": (
            "\ndef _fake_rollback(self):\n"
            "    if self._connection is not None:\n"
            "        self._connection.commit()\n"
            "SQLiteEvidenceSpool._rollback = _fake_rollback\n"
        ),
        "earlier_rollback": (
            "\n_append = SQLiteEvidenceSpool.append_record\n"
            "def _fake_append(self, record):\n"
            "    if self.connection.inject:\n"
            "        self.connection.execute('BEGIN IMMEDIATE')\n"
            "        self.connection.rollback()\n"
            "    return _append(self, record)\n"
            "def _fake_rollback(self):\n"
            "    if self._connection is not None:\n"
            "        self._connection.commit()\n"
            "SQLiteEvidenceSpool.append_record = _fake_append\n"
            "SQLiteEvidenceSpool._rollback = _fake_rollback\n"
        ),
        "commit_then_other_rollback": (
            "\ndef _fake_rollback(self):\n"
            "    if self._connection is not None:\n"
            "        self._connection.execute('\\ufeff;/* synthetic */ COMMIT')\n"
            "        self._connection.execute('-- synthetic\\n;\\ufeffBEGIN IMMEDIATE')\n"
            "        self._connection.rollback()\n"
            "SQLiteEvidenceSpool._rollback = _fake_rollback\n"
        ),
        "wrong_purge_authority": (
            "\nfrom dataclasses import replace as _replace\n"
            "_purge = SQLiteEvidenceSpool.purge_full_store\n"
            "def _fake_purge(self, command):\n"
            "    return _purge(self, _replace(command, full_purge_generation_id="
            "'40000000-0000-4000-8000-000000000097'))\n"
            "SQLiteEvidenceSpool.purge_full_store = _fake_purge\n"
        ),
        "wrong_initial_authority": (
            "\n_initial = initial_sentinel_image\n"
            "def _fake_initial(identifier):\n"
            "    return _initial('50000000-0000-4000-8000-000000000099')\n"
            "initial_sentinel_image = _fake_initial\n"
        ),
    }
    if fault is not None:
        with (package / "hermes_realtime/evidence/sqlite_spool.py").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(patches[fault])
    archive = _StorageArchive(tmp_path, source, package, None, "0" * 64)

    def qualify():
        invocations = tuple(
            _run_storage_worker(archive, point=case, mode=0, action="synthetic") for case in CASES
        )
        return _validate_run(_FaultRun("a" * 40, "b" * 40, "c" * 64, "d" * 64, invocations))

    if fault is None:
        evidence = qualify()
        assert evidence.process_count == 5
        assert evidence.machine_assertions == ASSERTIONS_V1
    else:
        with pytest.raises(
            ValueError,
            match={
                "no_delete": "storage worker retains a deleted file identity",
                "unclosed": "storage ownership was not released",
                "ignore_sqlite": "synthetic injection or capacity observation differs",
                "no_transaction": "synthetic injection or capacity observation differs",
                "commit_instead_of_rollback": "synthetic injection or capacity observation differs",
                "earlier_rollback": "synthetic injection or capacity observation differs",
                "commit_then_other_rollback": "synthetic injection or capacity observation differs",
                "wrong_purge_authority": "synthetic purge inventory differs",
                "wrong_initial_authority": "synthetic purge inventory differs",
            }[fault],
        ):
            qualify()


def _diagnostic_archive(tmp_path, *, pause=False):
    import shutil
    from pathlib import Path

    from scripts.storage_process import _StorageArchive

    source = Path(__file__).resolve().parents[1]
    package = tmp_path / "package"
    shutil.copytree(
        source / "src/hermes_realtime", package / "hermes_realtime",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    if pause:
        with (package / "hermes_realtime/evidence/sqlite_spool.py").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(
                "\n_initial = SQLiteEvidenceSpool.__init__\n"
                "def _paused_initial(self, *args, **kwargs):\n"
                "    import time\n"
                "    time.sleep(12)\n"
                "    _initial(self, *args, **kwargs)\n"
                "SQLiteEvidenceSpool.__init__ = _paused_initial\n"
            )
    return _StorageArchive(tmp_path, source, package, None, "0" * 64)


@pytest.mark.skipif(os.name != "nt", reason="requires retained Windows process identities")
def test_storage_timeout_reports_last_worker_stage_once(tmp_path, capsys):
    import json

    from scripts.storage_process import _run_storage_worker

    with pytest.raises(TimeoutError, match="equivalence frame deadline expired"):
        _run_storage_worker(
            _diagnostic_archive(tmp_path, pause=True),
            point="deny_filter", mode=0, action="synthetic",
        )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("[storage-worker]")
    evidence = json.loads(lines[0].removeprefix("[storage-worker]"))
    assert evidence == {
        "version": 1, "action": "synthetic", "case": "deny_filter",
        "failure": "timeout", "parent_stage": "first_observation",
        "worker_stage": "opening_storage", "stage_count": 4,
        "elapsed_ms": evidence["elapsed_ms"],
    }
    # Floored floating-point milliseconds can land one tick below the 10s deadline.
    assert type(evidence["elapsed_ms"]) is int and 9_999 <= evidence["elapsed_ms"] <= 60_000


@pytest.mark.skipif(os.name != "nt", reason="requires retained Windows process identities")
def test_storage_success_does_not_emit_refusal(tmp_path, capsys):
    from scripts.storage_process import _run_storage_worker

    result = _run_storage_worker(
        _diagnostic_archive(tmp_path), point="deny_filter", mode=0, action="synthetic",
    )
    assert result.exit_code == 0
    assert capsys.readouterr().out == ""


@pytest.mark.skipif(os.name != "nt", reason="requires the Windows nonblocking pipe reader")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"", (0, "unavailable")),
        (b"BICO", (4, "opening_storage")),
        (b"BICOSFDRPUEZ", (12, "complete")),
        (b"BICOSFDRPUEZZ", (13, "invalid")),
        (b"private", (7, "invalid")),
    ],
)
def test_storage_worker_stage_channel_is_bounded_and_closed(raw, expected):
    from scripts.storage_process import _read_worker_stages

    read, write = os.pipe()
    try:
        if raw:
            os.write(write, raw)
        assert _read_worker_stages(read) == expected
    finally:
        os.close(write)
        os.close(read)


@pytest.mark.skipif(os.name != "nt", reason="requires the Windows nonblocking pipe reader")
def test_storage_failed_stage_reader_is_unavailable():
    from scripts.storage_process import _read_worker_stages

    read, write = os.pipe()
    os.close(read)
    os.close(write)
    assert _read_worker_stages(read) == (0, "unavailable")


@pytest.mark.skipif(os.name != "nt", reason="requires retained Windows process identities")
@pytest.mark.parametrize(
    ("error", "category"),
    [(ValueError("private"), "value"), (OSError("private"), "os"),
     (RuntimeError("private"), "other")],
)
def test_storage_refusal_preserves_error_and_closed_category(
    tmp_path, capsys, monkeypatch, error, category,
):
    import json

    from scripts import storage_process

    def refused(*args):
        raise error

    monkeypatch.setattr(storage_process, "_read_frame", refused)
    with pytest.raises(type(error)) as raised:
        storage_process._run_storage_worker(
            _diagnostic_archive(tmp_path), point="deny_filter", mode=0, action="synthetic",
        )
    assert raised.value is error
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    evidence = json.loads(lines[0].removeprefix("[storage-worker]"))
    assert evidence["failure"] == category
    assert "private" not in lines[0]


def test_storage_worker_emits_only_closed_stage_codes(monkeypatch):
    from scripts import synthetic_fault_worker

    writes = []
    monkeypatch.setattr(synthetic_fault_worker.sys, "_storage_phase_fd", 1, raising=False)
    monkeypatch.setattr(synthetic_fault_worker.os, "write", lambda fd, raw: writes.append(raw))
    monkeypatch.setattr(synthetic_fault_worker, "_stage_count", 0, raising=False)
    for code in (b"private", b"", b"B"):
        synthetic_fault_worker._stage(code)
    synthetic_fault_worker._stage(b"S")
    assert writes == [b"S"]


def test_storage_worker_stage_attempts_stop_at_lifetime_limit(monkeypatch):
    from scripts import synthetic_fault_worker

    writes = []
    monkeypatch.setattr(synthetic_fault_worker.sys, "_storage_phase_fd", 1, raising=False)
    monkeypatch.setattr(synthetic_fault_worker.os, "write", lambda fd, raw: writes.append(raw))
    monkeypatch.setattr(synthetic_fault_worker, "_stage_count", 0, raising=False)
    for _ in range(20):
        synthetic_fault_worker._stage(b"I")
    assert writes == [b"I"] * 11


def test_storage_worker_failed_writes_are_bounded_and_nonfatal(monkeypatch):
    from scripts import synthetic_fault_worker

    calls = []

    def broken(fd, raw):
        calls.append(raw)
        raise OSError("private")

    monkeypatch.setattr(synthetic_fault_worker.sys, "_storage_phase_fd", 1, raising=False)
    monkeypatch.setattr(synthetic_fault_worker.os, "write", broken)
    monkeypatch.setattr(synthetic_fault_worker, "_stage_count", 0, raising=False)
    for _ in range(20):
        synthetic_fault_worker._stage(b"S")
    assert calls == [b"S"] * 11


def test_storage_refusal_normalizes_unknown_categories(capsys):
    import json

    from scripts.storage_process import _storage_refusal_evidence

    error = ValueError("private")
    with (
        pytest.raises(ValueError) as raised,
        _storage_refusal_evidence(None, "private", "private") as stage,
    ):
        stage["parent_stage"] = "private"
        raise error
    assert raised.value is error
    line = capsys.readouterr().out.strip()
    evidence = json.loads(line.removeprefix("[storage-worker]"))
    assert evidence["action"] == evidence["case"] == evidence["parent_stage"] == "unknown"
    assert "private" not in line


@pytest.mark.parametrize(("clock", "expected"), [(float("nan"), 0), (float("inf"), 60_000),
                                                  (-1, 0), (1_000_000, 60_000),
                                                  (9.999999999999998, 9_999)])
def test_storage_refusal_elapsed_is_bounded_finite(clock, expected, capsys, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts import storage_process

    ticks = iter([0, clock])
    monkeypatch.setattr(storage_process, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with (
        pytest.raises(ValueError),
        storage_process._storage_refusal_evidence(None, "synthetic", "deny_filter"),
    ):
        raise ValueError("private")
    evidence = json.loads(capsys.readouterr().out.strip().removeprefix("[storage-worker]"))
    assert type(evidence["elapsed_ms"]) is int and evidence["elapsed_ms"] == expected


def test_storage_failed_marker_cannot_mask_original_error(monkeypatch):
    from scripts import storage_process

    attempts = []

    def broken(*args, **kwargs):
        attempts.append(1)
        raise OSError("private")

    monkeypatch.setattr("builtins.print", broken)
    error = TimeoutError("original")
    with (
        pytest.raises(TimeoutError) as raised,
        storage_process._storage_refusal_evidence(None, "synthetic", "deny_filter"),
    ):
        raise error
    assert raised.value is error
    assert attempts == [1]


@pytest.mark.skipif(os.name != "nt", reason="requires the real protected Git archive authority")
@pytest.mark.parametrize("fault", ["candidate", "wheel"])
def test_foreign_candidate_or_unminted_wheel_refuses_before_dispatch(tmp_path, fault, monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    from runpy import run_path

    from scripts import storage_process
    from scripts.candidate_source_archive_oracle import (
        CandidateSourceArchiveError,
        capture_candidate_source_archive,
    )
    from scripts.candidate_wheel import VerifiedCandidateWheelV1
    from scripts.qualify_evidence_slice_zero import SYNTHETIC_FAULT_REGISTRATION_V1

    helpers = run_path(
        str(Path(__file__).resolve().parent / "test_candidate_source_archive_oracle.py")
    )
    repository, baseline = helpers["_repository"](tmp_path)
    identity = helpers["_identity"](repository, baseline)
    archive = capture_candidate_source_archive(repository, identity, helpers["_pin"]())
    wheel = object.__new__(VerifiedCandidateWheelV1)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid candidate or artifact reached workspace creation")

    monkeypatch.setattr(storage_process.tempfile, "mkdtemp", forbidden)
    if fault == "candidate":
        identity = replace(identity, candidate_tree_oid="0" * 40)
    with pytest.raises(
        CandidateSourceArchiveError if fault == "candidate" else ValueError,
        match="candidate identity differs" if fault == "candidate" else "no verified authority",
    ):
        SYNTHETIC_FAULT_REGISTRATION_V1.produce(archive, identity, wheel)
