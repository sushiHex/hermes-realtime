"""The CUDA worker closure is generated output, so it must be what generation yields.

`requirements/kokoro-cuda-worker-win-py311.txt` is the resolution of
`requirements/kokoro-cuda-worker.in` for Windows CPython 3.11, produced by the procedure in
`requirements/README.md` and installed by `scripts/setup-kokoro-cuda-worker.sh` with
`--require-hashes`. Its validity is a property of the whole resolution — the target interpreter
and platform, every transitive dependency, and every artifact hash — so no check of a single
entry can establish it. Two edits that carried every published wheel hash still broke it: a
joblib bump that omitted the cloudpickle it newly required, and a numpy bump to a release
requiring Python 3.12. Neither was visible to a passing run, because nothing installs this
closure in CI.

Re-running the procedure against the committed file keeps every requirement that can hold and
changes any that cannot. So the requirements themselves — name, version and marker — must agree
exactly. Hashes are held to one allowed divergence: an index may publish another wheel for a
version already pinned, which grows the regenerated set without anything being wrong. Each
committed hash set must therefore be non-empty and contained in the regenerated one. That
refuses a corrupted or stripped hash, which `--require-hashes` would reject at setup. It cannot
tell a hash removed from the committed file apart from one published after it was written; the
installer's own hash check remains the backstop for that.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_ROOT = Path(__file__).resolve().parents[1]
_ROOTS = _ROOT / "requirements" / "kokoro-cuda-worker.in"
_CLOSURE = _ROOT / "requirements" / "kokoro-cuda-worker-win-py311.txt"
_HASH = re.compile(r"\s+--hash=")

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

_Closure = dict[str, tuple[str, frozenset[str]]]


def _parse(text: str) -> _Closure:
    """Map each requirement to its version-and-marker text and its artifact hashes."""
    parsed: _Closure = {}
    for logical in text.replace("\\\n", " ").splitlines():
        line = logical.split("#", 1)[0].strip()
        if not line:
            continue
        requirement, *hashes = _HASH.split(line)
        parsed_requirement = Requirement(requirement)
        identity = str(parsed_requirement.specifier)
        if parsed_requirement.marker is not None:
            identity += f"; {parsed_requirement.marker}"
        parsed[canonicalize_name(parsed_requirement.name)] = (identity, frozenset(hashes))
    return parsed


def _compile(uv: str, output: Path, *extra: str) -> _Closure:
    subprocess.run(
        [uv, *_COMPILE, *extra, "--output-file", str(output)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return _parse(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def closures(tmp_path_factory: pytest.TempPathFactory) -> tuple[_Closure, _Closure, _Closure]:
    """Return the committed closure, its regeneration, and independently fetched hashes.

    The regeneration reads the committed file as its preference source, which is what keeps
    every compatible pin — but uv also carries that file's hashes forward for any pin it
    keeps, so a corrupted committed hash would reappear in it. Hashes are therefore taken from
    a second compile that never reads the committed file, constrained to the pins the first
    one validated.
    """
    uv = shutil.which("uv")
    assert uv is not None, "uv must be on PATH to regenerate the worker closure"
    workspace = tmp_path_factory.mktemp("closure")
    preferred = workspace / "preferred.txt"
    shutil.copyfile(_CLOSURE, preferred)
    regenerated = _compile(uv, preferred)
    constraints = workspace / "constraints.txt"
    constraints.write_text(
        "".join(f"{name}{identity}\n" for name, (identity, _) in regenerated.items()),
        encoding="utf-8",
    )
    fresh = _compile(uv, workspace / "fresh.txt", "--constraint", str(constraints))
    return _parse(_CLOSURE.read_text(encoding="utf-8")), regenerated, fresh


def test_the_worker_closure_resolves_as_its_own_regeneration(
    closures: tuple[_Closure, _Closure, _Closure],
) -> None:
    committed, regenerated, _ = closures
    have = {name: identity for name, (identity, _) in committed.items()}
    want = {name: identity for name, (identity, _) in regenerated.items()}
    assert have == want, (
        "requirements/kokoro-cuda-worker-win-py311.txt is not a valid resolution of its roots "
        "for Windows CPython 3.11. Regenerate it with the procedure in requirements/README.md. "
        f"Committed only: {sorted(have.items() - want.items())}; "
        f"regeneration only: {sorted(want.items() - have.items())}"
    )


def test_every_worker_closure_hash_is_one_the_index_publishes(
    closures: tuple[_Closure, _Closure, _Closure],
) -> None:
    committed, _, published = closures
    # Judged only where the requirement itself agrees; a differing version is the other
    # test's failure, and its hashes are expected to differ with it.
    invalid = {
        name: sorted(hashes - published[name][1]) or "no hashes"
        for name, (identity, hashes) in committed.items()
        if name in published
        and identity == published[name][0]
        and (not hashes or not hashes <= published[name][1])
    }
    assert not invalid, (
        "requirements/kokoro-cuda-worker-win-py311.txt carries hashes the index does not publish "
        "for the pinned versions, or none at all; scripts/setup-kokoro-cuda-worker.sh installs "
        "with --require-hashes and would reject it. Regenerate it with the procedure in "
        f"requirements/README.md. {invalid}"
    )
