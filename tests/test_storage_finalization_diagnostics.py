"""Synthetic failure evidence only; never launch a storage worker or change cleanup."""

from __future__ import annotations

import json

import pytest

from scripts import qualify_evidence_slice_zero as core
from scripts.storage_process import _storage_refusal_evidence


def refusal() -> core._WindowsFinalizationError:
    return core._WindowsFinalizationError(core._WindowsFinalizationResultV1(
        pre_cleanup_snapshot=None,
        terminated_job=True,
        waited_handles=(981234567, 981234568),
        zero_active_observed=True,
        closed_handles=(981234568,),
        failed_handles=(981234567,),
        failures=(
            core._WindowsOperationFailureV1("close", 981234567, "private token and C:/private"),
            core._WindowsOperationFailureV1("close", 981234567, "private process 981234569"),
            core._WindowsOperationFailureV1("private operation", None, "private transcript"),
            *(core._WindowsOperationFailureV1(operation, 981234567, "private message")
              for operation in (
                  "pre_cleanup_snapshot", "terminate_job", "wait", "identity", "zero_active",
                  "transient_close",
              )),
        ),
        closed=False,
    ))


@pytest.mark.parametrize("nested", [False, True])
def test_finalization_evidence_preserves_failure_and_only_reports_closed_categories(capsys, nested):
    failure = refusal()
    if nested:
        failure = ExceptionGroup("private group", [
            RuntimeError("private primary"), ExceptionGroup("private nested", [failure]),
        ])
    with (
        pytest.raises(BaseException) as caught,
        _storage_refusal_evidence(None, "crash", "private case") as stage,
    ):
        stage["parent_stage"] = "process_exit"
        raise failure
    assert caught.value is failure
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[storage-worker]")
    assert json.loads(lines[0].removeprefix("[storage-worker]"))["parent_stage"] == "process_exit"
    assert json.loads(lines[1].removeprefix("[storage-finalization]")) == {
        "version": 1,
        "finalization_count": 1,
        "operations": {
            "close": 2, "unknown": 1, "pre_cleanup_snapshot": 1, "terminate_job": 1,
            "wait": 1, "identity": 1, "zero_active": 1, "transient_close": 1,
        },
        "terminated_job": True,
        "zero_active_observed": True,
        "closed": False,
        "waited_handle_count": 2,
        "closed_handle_count": 1,
        "failed_handle_count": 1,
        "truncated": False,
    }
    assert "private" not in "\n".join(lines)
    assert "98123456" not in "\n".join(lines)


def test_finalization_group_scan_is_bounded_without_hiding_original_failure(capsys):
    failure = ExceptionGroup("private group", [refusal(), *(
        RuntimeError("private error") for _ in range(100)
    )])
    with (
        pytest.raises(ExceptionGroup) as caught,
        _storage_refusal_evidence(None, "crash", "private case"),
    ):
        raise failure
    assert caught.value is failure
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    evidence = json.loads(lines[1].removeprefix("[storage-finalization]"))
    assert evidence["finalization_count"] == 1
    assert evidence["truncated"] is True
    assert len(lines[1]) < 1024 and "private" not in lines[1]


def test_success_does_not_emit_refusal_evidence(capsys):
    with _storage_refusal_evidence(None, "crash", "private case"):
        pass
    assert capsys.readouterr().out == ""


def test_finalization_diagnostic_sink_failure_cannot_mask_original_error(monkeypatch):
    attempts = []

    def broken_finalization(line, **kwargs):
        attempts.append(line.split("]", 1)[0])
        if line.startswith("[storage-finalization]"):
            raise OSError("private diagnostic sink failure")

    monkeypatch.setattr("builtins.print", broken_finalization)
    failure = refusal()
    with (
        pytest.raises(core._WindowsFinalizationError) as caught,
        _storage_refusal_evidence(None, "crash", "private case"),
    ):
        raise failure
    assert caught.value is failure
    assert attempts == ["[storage-worker", "[storage-finalization"]
