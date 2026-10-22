"""The reviewed proof helpers must execute source, not finder/cache overlays."""

import hashlib
import json
import os
import py_compile
import runpy
import subprocess
import sys
from pathlib import Path

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
