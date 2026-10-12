from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_provider_package_import_does_not_import_numpy() -> None:
    # A fresh interpreter proves import purity without replacing classes held by
    # other collected tests or weakening production's exact-type boundaries.
    completed = subprocess.run(
        [sys.executable, "-I", "-c", """
import builtins
import importlib
import sys

sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__

def reject_numpy(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise AssertionError("base provider package imported numpy")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_numpy
package = importlib.import_module("hermes_realtime.providers")
assert "PlaybackEchoGuard" in package.__all__
""", str(Path(__file__).resolve().parents[2] / "src")],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
