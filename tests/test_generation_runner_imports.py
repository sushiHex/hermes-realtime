"""The reviewed proof helpers must execute source, not finder/cache overlays."""

import contextlib
import hashlib
import json
import os
import py_compile
import runpy
import subprocess
import sys
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
    result = dict(ok=True, stage="native-proof", case="synthetic", mutation="none")
    seen = []

    class Directory:
        name = str(tmp_path / "profile")

        def __init__(self, **_kwargs):
            pass

        def cleanup(self):
            seen.append(json.loads(receipt.read_bytes()))

    with (
        patch.object(namespace["tempfile"], "TemporaryDirectory", Directory),
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
        ok=True, stage="native-proof", case="synthetic", mutation="none", error_kind="PrimaryKind"
    )

    class Directory:
        name = str(tmp_path / "profile")

        def __init__(self, **_kwargs):
            pass

        def cleanup(self):
            raise PermissionError("synthetic")

    with (
        patch.object(namespace["tempfile"], "TemporaryDirectory", Directory),
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
    result = dict(ok=True, stage="native-proof", case="synthetic", mutation="none")

    class BrokenReceipt:
        def write_text(self, *_args, **_kwargs):
            raise PermissionError("synthetic")

    class Directory:
        def __init__(self, **_kwargs):
            self.name = str(tmp_path / "profile")
            Path(self.name).mkdir()

        def cleanup(self):
            Path(self.name).rmdir()

    with (
        patch.object(namespace["tempfile"], "TemporaryDirectory", Directory),
        namespace["proof_profile"](BrokenReceipt(), result) as profile,
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
    result = dict(ok=True, stage="setup", case="synthetic", mutation="none")
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
    result = dict(ok=True, stage="native-proof", case="synthetic", mutation="none")

    class Receipt:
        calls = 0

        def write_text(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise PermissionError("synthetic")

    expectation = pytest.raises(ValueError) if primary_failure else contextlib.nullcontext()
    with expectation, namespace["proof_profile"](Receipt(), result) as profile:
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


@pytest.mark.parametrize("semantic_ok", [False, True])
def test_cleanup_healthy_control(tmp_path, semantic_ok):
    namespace = profile_namespace()
    receipt = tmp_path / "receipt.json"
    result = dict(ok=semantic_ok, stage="native-proof", case="synthetic", mutation="none")
    with namespace["proof_profile"](receipt, result) as profile:
        assert Path(profile).is_dir()
    observed = json.loads(receipt.read_bytes())
    assert observed["ok"] is semantic_ok and observed["profile_cleanup"] == "completed"
    assert not Path(profile).exists()
