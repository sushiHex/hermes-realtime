"""Linux candidate-wheel proof that unsupported capture leaves no local residue."""

from __future__ import annotations

import importlib
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="linux-null-capture gate")


def _run(*arguments: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_linux_candidate_has_no_capture_artifacts_threads_or_local_cli_surface() -> None:
    import hermes_realtime

    names = sorted(
        module.name
        for module in pkgutil.walk_packages(
            hermes_realtime.__path__, "hermes_realtime."
        )
    )
    for name in names:
        importlib.import_module(name)

    host = shutil.which("hermes-realtime-host")
    local = shutil.which("hermes-realtime-local")
    assert host is not None and local is not None
    with tempfile.TemporaryDirectory(prefix="hermes-linux-null-capture-") as temporary:
        root = Path(temporary)
        before_paths = {path.relative_to(root) for path in root.rglob("*")}
        before_threads = {thread.ident for thread in threading.enumerate()}
        environment = {
            "HOME": str(root / "home"),
            "PATH": os.environ["PATH"],
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(root),
        }
        assert _run(host, "--help", environment=environment).returncode == 0
        assert _run(local, "--help", environment=environment).returncode == 0
        local_evidence = _run(local, "--evidence-capture", environment=environment)
        assert local_evidence.returncode == 2
        assert "unrecognized arguments" in local_evidence.stderr
        for flag in ("--evidence-capture", "--evidence-status", "--purge-evidence"):
            response = _run(host, flag, environment=environment)
            assert response.returncode == 2
            assert response.stdout == '{"error":"unsupported_platform","version":1}\n'
        after_paths = {path.relative_to(root) for path in root.rglob("*")}
        assert after_paths == before_paths
        assert {thread.ident for thread in threading.enumerate()} == before_threads
        forbidden = ("capture-v1.sqlite3", "evidence", ".lock", "installation_id", "writer")
        assert not any(any(token in path.as_posix() for token in forbidden) for path in after_paths)
