"""The CUDA worker closure is generated output, so its pins must be what generation yields.

`requirements/kokoro-cuda-worker-win-py311.txt` is the resolution of
`requirements/kokoro-cuda-worker.in` for Windows CPython 3.11, produced by the procedure in
`requirements/README.md`. Its validity is a property of the whole resolution — the target
interpreter and platform, and every transitive dependency — so no check of a single entry can
establish it. Two edits that carried every published wheel hash still broke it: a joblib bump
that omitted the cloudpickle it newly required, and a numpy bump to a release requiring
Python 3.12. Neither was visible to a passing run, because nothing installs this closure in CI.

Re-running the procedure against the committed file keeps every compatible pin and changes any
that cannot hold, so the committed closure is valid exactly when the two agree. Pins are
compared rather than bytes, because an index may publish another wheel for an existing
version and that changes the hash set without anything being wrong.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from packaging.utils import canonicalize_name

_ROOT = Path(__file__).resolve().parents[1]
_ROOTS = _ROOT / "requirements" / "kokoro-cuda-worker.in"
_CLOSURE = _ROOT / "requirements" / "kokoro-cuda-worker-win-py311.txt"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)", re.MULTILINE)

# The procedure in requirements/README.md, less its output path.
_COMPILE = (
    "pip",
    "compile",
    str(_ROOTS),
    "--python-version",
    "3.11",
    "--python-platform",
    "x86_64-pc-windows-msvc",
    "--only-binary",
    ":all:",
    "--generate-hashes",
    "--no-emit-package",
    "kokoro-onnx",
    "--no-emit-package",
    "onnxruntime",
    "--no-header",
    "--no-annotate",
)


def _pins(closure: str) -> dict[str, str]:
    return {canonicalize_name(name): version for name, version in _PIN.findall(closure)}


def test_the_worker_closure_is_its_own_regeneration(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv must be on PATH to regenerate the worker closure"
    regenerated = tmp_path / _CLOSURE.name
    shutil.copyfile(_CLOSURE, regenerated)
    subprocess.run(
        [uv, *_COMPILE, "--output-file", str(regenerated)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )

    committed = _pins(_CLOSURE.read_text(encoding="utf-8"))
    expected = _pins(regenerated.read_text(encoding="utf-8"))
    assert committed == expected, (
        "requirements/kokoro-cuda-worker-win-py311.txt is not a valid resolution of its roots "
        "for Windows CPython 3.11. Regenerate it with the procedure in requirements/README.md. "
        f"Committed only: {sorted(committed.items() - expected.items())}; "
        f"regeneration only: {sorted(expected.items() - committed.items())}"
    )
