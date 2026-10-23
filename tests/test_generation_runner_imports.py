"""The reviewed proof helpers must execute source, not finder/cache overlays."""

import contextlib
import errno
import hashlib
import json
import os
import py_compile
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


def test_reviewed_helper_ignores_extension_and_bytecode_overlay(tmp_path):
    # Mutation: replace exact source execution with a name/finder or cached loader.
    namespace = runpy.run_path(str(Path(__file__).with_name("offline_plugin_generation_proof.py")))
    path = tmp_path / "helper.py"
    old = b'VALUE = "stale"\n'
    new = b'VALUE = "fresh"\n'
    path.write_bytes(old)
    timestamp = path.stat().st_mtime
    py_compile.compile(str(path), doraise=True)
    path.write_bytes(new)
    os.utime(path, (timestamp, timestamp))  # Same-size, timestamp-valid stale cache.
    (tmp_path / "helper.pyd").write_bytes(b"not-an-executable-extension")
    module, digest = namespace["load_reviewed"]("helper", tmp_path)
    assert (module.VALUE, module.__file__, digest) == (
        "fresh",
        str(path.resolve()),
        hashlib.sha256(new).hexdigest(),
    )


@pytest.mark.parametrize("value", [7, 11])
def test_reviewed_helper_healthy_control(tmp_path, value):
    namespace = runpy.run_path(str(Path(__file__).with_name("offline_plugin_generation_proof.py")))
    payload = f"VALUE = {value}\n".encode()
    (tmp_path / "helper.py").write_bytes(payload)
    module, digest = namespace["load_reviewed"]("helper", tmp_path)
    assert value == module.VALUE and digest == hashlib.sha256(payload).hexdigest()


def test_no_bytecode_startup_required(tmp_path):
    # Mutation: remove the -B startup requirement before source preparation.
    receipt = tmp_path / "receipt.json"
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("offline_plugin_generation_proof.py")),
            "--hermes-source",
            str(tmp_path),
            "--receipt",
            str(receipt),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    result = json.loads(receipt.read_bytes())
    assert (
        child.returncode == 1 and result.get("category") == "isolated-no-site-no-bytecode-required"
    )


def profile_namespace():
    return runpy.run_path(str(Path(__file__).with_name("offline_plugin_generation_proof.py")))


def test_cleanup_pending_is_not_overall_success(tmp_path):
    # Mutation: retain ok=True in the pre-cleanup receipt.
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    result = dict(
        child_settled=True, ok=True, stage="native-proof", case="synthetic", mutation="none"
    )
    seen = []

    class Directory:
        name = str(tmp_path / "profile")

        def __init__(self, **_kwargs):
            pass

        def cleanup(self):
            seen.append(json.loads(receipt.read_bytes()))

    with (
        patch.dict(namespace["checkpoint"].__globals__, new_profile=Directory),
        namespace["proof_profile"](receipt, result),
    ):
        assert not receipt.exists()
    assert json.loads(receipt.read_bytes())["ok"] is True
    assert seen[0]["ok"] is False and seen[0]["semantic_ok"] is True
    assert seen[0]["profile_cleanup"] == "pending"


def test_cleanup_failure_preserves_primary_refusal(tmp_path, capsys):
    # Mutation: mark a cleanup exception as overall success.
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    result = dict(
        child_settled=True,
        ok=True,
        stage="native-proof",
        case="synthetic",
        mutation="none",
        error_kind="PrimaryKind",
    )

    class Directory:
        name = str(tmp_path / "profile")

        def __init__(self, **_kwargs):
            pass

        def cleanup(self):
            raise PermissionError("synthetic")

    with (
        patch.dict(namespace["checkpoint"].__globals__, new_profile=Directory),
        namespace["proof_profile"](receipt, result),
    ):
        pass
    observed = json.loads(receipt.read_bytes())
    assert observed["ok"] is False and observed["semantic_ok"] is True
    assert observed["profile_cleanup"] == "refused"
    assert observed["error_kind"] == "PrimaryKind"
    assert observed["cleanup_error_kind"] == "PermissionError"
    assert (
        json.loads(capsys.readouterr().out.split(" ", 1)[1])["category"]
        == "profile-cleanup-refused"
    )


def test_pending_receipt_failure_still_cleans_profile(tmp_path, capsys):
    # Mutation: skip directory cleanup when the pending evidence write fails.
    namespace = profile_namespace()
    result = dict(
        child_settled=True, ok=True, stage="native-proof", case="synthetic", mutation="none"
    )

    def refuse_write(*_args):
        raise PermissionError("synthetic")

    class Directory:
        def __init__(self, **_kwargs):
            self.name = str(tmp_path / "profile")
            Path(self.name).mkdir()

        def cleanup(self):
            Path(self.name).rmdir()

    with (
        patch.dict(namespace["checkpoint"].__globals__, write_receipt=refuse_write),
        patch.dict(namespace["checkpoint"].__globals__, new_profile=Directory),
        namespace["proof_profile"](tmp_path / "receipt.json", result) as profile,
    ):
        assert Path(profile).is_dir()
    assert not Path(profile).exists()
    assert result["ok"] is False and result["profile_cleanup"] == "completed"
    assert result["receipt_error_kind"] == "PermissionError"
    assert (
        json.loads(capsys.readouterr().out.split(" ", 1)[1])["category"] == "proof-receipt-refused"
    )


def test_profile_body_failure_cannot_publish_success(tmp_path):
    # Mutation: leave the preceding successful semantic result after body failure.
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    result = dict(child_settled=True, ok=True, stage="setup", case="synthetic", mutation="none")
    with pytest.raises(ValueError), namespace["proof_profile"](receipt, result):
        raise ValueError("synthetic")
    observed = json.loads(receipt.read_bytes())
    assert observed["ok"] is False and observed["semantic_ok"] is False
    assert observed["error_kind"] == "ValueError"
    assert observed["profile_cleanup"] == "completed"


@pytest.mark.parametrize("primary_failure", [False, True])
def test_final_receipt_failure_cannot_publish_success(tmp_path, capsys, primary_failure):
    # Mutation: retain successful overall state when the final evidence write fails.
    namespace = profile_namespace()
    result = dict(
        child_settled=True, ok=True, stage="native-proof", case="synthetic", mutation="none"
    )

    calls = []
    original_write = namespace["write_receipt"]

    def refuse_final_write(*args):
        calls.append(1)
        if len(calls) == 2:
            raise PermissionError("synthetic")
        original_write(*args)

    expectation = pytest.raises(ValueError) if primary_failure else contextlib.nullcontext()
    with (
        expectation,
        patch.dict(namespace["checkpoint"].__globals__, write_receipt=refuse_final_write),
        namespace["proof_profile"](tmp_path / "receipt.json", result) as profile,
    ):
        assert Path(profile).is_dir()
        if primary_failure:
            raise ValueError("synthetic")
    assert not Path(profile).exists() and result["ok"] is False
    assert result["profile_cleanup"] == "completed"
    if primary_failure:
        assert result["error_kind"] == "ValueError"
    assert (
        json.loads(capsys.readouterr().out.split(" ", 1)[1])["category"] == "proof-receipt-refused"
    )


def test_checkpoint_precedes_work_and_never_publishes_success(tmp_path):
    # Mutations: omit checkpoint publication or retain semantic ok=True in it.
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    result = dict(child_settled=True, ok=True, stage="setup")
    with patch.object(namespace["time"], "monotonic", side_effect=[10, 12]):
        namespace["checkpoint"](receipt, result, "export", 5)
        first = json.loads(receipt.read_bytes())
        assert first["ok"] is False and first["stage"] == "export"
        namespace["checkpoint"](receipt, result, "import-bootstrap", 5)
    observed = json.loads(receipt.read_bytes())
    assert observed["ok"] is False and result["ok"] is True
    assert observed["phase_timings"] == [
        {"stage": "export", "elapsed_seconds": 5},
        {"stage": "import-bootstrap", "elapsed_seconds": 7},
    ]


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_interrupted_checkpoint_preserves_previous_evidence(tmp_path, failure):
    # Mutation: write directly over the retained receipt instead of a temporary file.
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    namespace["write_receipt"](receipt, {"ok": False, "stage": "source-verification"})
    previous = receipt.read_bytes()
    result = dict(child_settled=True, ok=False, stage="setup")
    work = []
    original_write = Path.write_text

    def interrupted_write(path, *_args, **_kwargs):
        original_write(path, "partial synthetic evidence")
        raise PermissionError("synthetic")

    fault = (
        patch.object(Path, "write_text", interrupted_write)
        if failure == "write"
        else patch.object(namespace["os"], "replace", side_effect=PermissionError("synthetic"))
    )
    with fault, pytest.raises(PermissionError):
        namespace["checkpoint"](receipt, result, "export", 0)
        work.append(1)
    assert not work and receipt.read_bytes() == previous
    assert result["receipt_category"] == "proof-receipt-refused"


@pytest.mark.parametrize("semantic_ok", [False, True])
def test_cleanup_healthy_control(tmp_path, semantic_ok):
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    result = dict(
        child_settled=True, ok=semantic_ok, stage="native-proof", case="synthetic", mutation="none"
    )
    with namespace["proof_profile"](receipt, result) as profile:
        assert Path(profile).is_dir()
    observed = json.loads(receipt.read_bytes())
    assert observed["ok"] is semantic_ok and observed["profile_cleanup"] == "completed"
    assert not Path(profile).exists()


@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("write_failure", [False, True])
def test_cleanup_target_metadata_is_owned_and_private(tmp_path, outside, write_failure):
    namespace = profile_namespace()
    profile = tmp_path / "profile"
    target = (tmp_path if outside else profile) / "state.db"
    sidecar = tmp_path / "private.json"
    result = dict(child_settled=True, ok=True, stage="test", case="synthetic", mutation="none")
    error = PermissionError(errno.EACCES, "synthetic", str(target))
    error.winerror = 32

    class Directory:
        name = str(profile)

        def __init__(self, **_kwargs):
            pass

        def cleanup(self):
            raise error

    original = namespace["write_receipt"]

    def write(path, record):
        if path == sidecar and write_failure:
            raise OSError("synthetic diagnostic write failure")
        original(path, record)

    with (
        patch.dict(namespace["checkpoint"].__globals__, new_profile=Directory),
        patch.dict(namespace["checkpoint"].__globals__, write_receipt=write),
        namespace["proof_profile"](
            tmp_path / "receipt.json", result, private_cleanup_evidence=sidecar
        ),
    ):
        pass
    assert result["ok"] is False and result["cleanup_error_kind"] == "PermissionError"
    assert result["cleanup_target_scope"] == ("outside" if outside else "owned")
    assert result["cleanup_target_kind"] == ("unknown" if outside else "database")
    assert (result["cleanup_errno"], result["cleanup_winerror"]) == (errno.EACCES, 32)
    assert json.dumps(str(tmp_path)) not in json.dumps(result)
    assert "state.db" not in json.dumps(result)
    if write_failure:
        assert result["private_cleanup_evidence_error_kind"] == "OSError"
    else:
        assert json.loads(sidecar.read_bytes())["target_relative"] == (
            None if outside else "state.db"
        )


def test_cleanup_operation_uses_actual_stdlib_failure(tmp_path):
    namespace = profile_namespace()
    profile = tmp_path / "profile"
    profile.mkdir()
    target = profile / "state.db"
    target.write_bytes(b"synthetic")
    error = PermissionError(errno.EACCES, "synthetic", str(target))
    with patch.object(os, "unlink", side_effect=error), pytest.raises(PermissionError) as caught:
        shutil.rmtree(profile)
    detail, _ = namespace["cleanup_failure_details"](caught.value, profile)
    assert detail["cleanup_operation"] == "unlink" and detail["cleanup_operation_frames"]


def test_cleanup_classifier_failure_preserves_primary_refusal(tmp_path):
    namespace = profile_namespace()
    result = dict(child_settled=True, ok=True, stage="test", case="synthetic", mutation="none")

    class Directory:
        name = str(tmp_path / "profile")

        def __init__(self, **_kwargs):
            pass

        def cleanup(self):
            raise PermissionError("synthetic primary")

    with (
        patch.dict(namespace["checkpoint"].__globals__, new_profile=Directory),
        patch.dict(namespace["checkpoint"].__globals__, cleanup_failure_details=lambda *_: 1 / 0),
        namespace["proof_profile"](tmp_path / "receipt.json", result),
    ):
        pass
    assert result["ok"] is False and result["cleanup_error_kind"] == "PermissionError"
    assert result["cleanup_evidence_category"] == "classification-refused"


def test_cleanup_private_output_cannot_alias_receipt(tmp_path):
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    with (
        pytest.raises(ValueError, match="cleanup-evidence-path-conflict"),
        namespace["proof_profile"](receipt, {"ok": True}, private_cleanup_evidence=receipt),
    ):
        pass
    assert not receipt.exists()


@pytest.mark.parametrize("drain_failure", [False, True])
def test_logging_owner_drains_before_handler_close(tmp_path, drain_failure):
    namespace = profile_namespace()
    observed = []
    result = dict(child_settled=True, ok=True, stage="test", case="synthetic", mutation="none")

    def drain(*, timeout):
        observed.append(("drain", timeout))
        if drain_failure:
            raise RuntimeError("synthetic drain failure")

    with (
        patch.dict(sys.modules, hermes_logging=types.SimpleNamespace(drain_log_queue=drain)),
        patch.object(namespace["logging"], "shutdown", lambda: observed.append("close")),
    ):
        if drain_failure:
            with (
                pytest.raises(RuntimeError, match="synthetic drain failure"),
                namespace["proof_profile"](tmp_path / "receipt.json", result),
            ):
                namespace["shutdown_logging"]()
        else:
            with namespace["proof_profile"](tmp_path / "receipt.json", result):
                namespace["shutdown_logging"]()
    assert observed == [("drain", 1.0), "close"]
    assert result["ok"] is (not drain_failure)


def test_unsettled_child_preserves_profile_and_refuses_success(tmp_path):
    namespace = profile_namespace()
    result = dict(ok=True, child_settled=False, stage="worker", case="synthetic", mutation="none")
    profile = None
    try:
        with namespace["proof_profile"](tmp_path / "receipt.json", result) as profile:
            pass
        assert Path(profile).exists(), "unsettled-profile-was-deleted"
        assert result["ok"] is False and result["profile_cleanup"] == "preserved"
    finally:
        if profile is not None:
            shutil.rmtree(profile, ignore_errors=True)


@pytest.mark.parametrize(
    "fault", ["none", "source", "loaded", "nonce", "job", "exit", "semantic", "timeout"]
)
def test_parent_cleanup_requires_settled_bound_worker(tmp_path, fault):
    namespace = profile_namespace()
    events = []
    receipt = tmp_path / "receipt.json"
    args = types.SimpleNamespace(
        receipt=receipt, hermes_source=tmp_path, case="healthy", mutation="none"
    )
    result = dict(ok=False, child_settled=False, stage="worker", case="healthy", mutation="none")
    expected = namespace["reviewed_bindings"]()
    observed = dict(
        ok=False,
        semantic_ok=fault != "semantic",
        worker_nonce="expected",
        case="healthy",
        mutation="none",
        reviewed_source_bindings=expected.copy(),
    )
    observed.update(
        {
            field: expected[name]
            for field, name in {
                "runner_sha256": "offline_plugin_generation_proof.py",
                "source_export_helper_sha256": "generation_source.py",
                "source_import_helper_sha256": "generation_imports.py",
                "cases_sha256": "generation_native_cases.py",
                "guard_checks_sha256": "generation_policy_checks.py",
                "fixture_sha256": "fixtures/plugin_generation_candidate/__init__.py",
                "policy_sha256": "fixtures/plugin_generation_candidate/policy.py",
                "entrypoint_sha256": "fixtures/plugin_generation_candidate/entrypoint.py",
                "fixture_manifest_sha256": "fixtures/plugin_generation_candidate/plugin.yaml",
            }.items()
        }
    )
    if fault == "loaded":
        observed["source_import_helper_sha256"] = "changed"
    if fault == "source":
        observed["reviewed_source_bindings"] = {"source": "changed"}
    if fault == "nonce":
        observed["worker_nonce"] = "stale"

    class Child:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("handle-closed")

        def wait(self, *, timeout):
            assert 0 < timeout <= 60
            self.calls += 1
            events.append("wait")
            if fault == "timeout" and self.calls == 1:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            receipt.with_name(receipt.name + ".worker.json").write_text(json.dumps(observed))
            return 2 if fault == "exit" else 1 if fault == "semantic" else 0

        def kill(self):
            events.append("kill")

    def members():
        events.append("members")
        return {os.getpid(), -1} if fault == "job" else {os.getpid()}

    original_remove = shutil.rmtree

    def remove(path):
        events.append("delete")
        original_remove(path)

    globals_ = namespace["run_owned_worker"].__globals__
    child = Child()
    profile = None
    with (
        patch.dict(globals_, reviewed_bindings=lambda: expected),
        patch.object(namespace["secrets"], "token_hex", return_value="expected"),
        patch.object(subprocess, "Popen", return_value=child),
        patch.object(subprocess, "CREATE_NO_WINDOW", 0, create=True),
        patch.object(shutil, "rmtree", side_effect=remove),
    ):
        try:
            with namespace["proof_profile"](receipt, result) as profile:
                try:
                    namespace["run_owned_worker"](
                        args,
                        profile,
                        result,
                        types.SimpleNamespace(members=members),
                        namespace["time"].monotonic(),
                    )
                except (RuntimeError, subprocess.TimeoutExpired):
                    result["ok"] = False
            if fault == "none":
                assert events == ["wait", "members", "handle-closed", "delete"]
                assert result["ok"] is True and not Path(profile).exists()
            else:
                assert "delete" not in events and Path(profile).exists()
                assert result["ok"] is False and result["profile_cleanup"] == "preserved"
                if fault == "timeout":
                    assert events == ["wait", "kill", "wait", "handle-closed"]
        finally:
            if profile is not None and Path(profile).exists():
                original_remove(profile)
