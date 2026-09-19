"""Dependabot may only propose changes it can complete in the files it writes.

Every failing bot pull request in this repository has had one shape: the change needs a
second artifact regenerated that the bot does not write. `uv.lock` after a manifest edit,
the tracked browser assets after a `/web` bump, the reviewed action pins in
`tests/test_release_workflow.py` after an action bump, the recompiled worker closure after
one of its declared roots moves. The configuration exists to keep those classes apart, and
these tests bind the parts of it that silently rot.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / ".github" / "dependabot.yml"
_WORKER_ROOTS = _ROOT / "requirements" / "kokoro-cuda-worker.in"
_IGNORED = re.compile(r'^\s*- dependency-name: "([^"]+)"\s*$', re.MULTILINE)

# The one mirrored root an ignore cannot cover. cryptography is pinned by the qualified
# upstream commit *and* constrained independently by the dev group, and an ignore applies
# to a dependency, not to one dependency group. Silencing it would silence dev's own
# security updates, so the bump is refused downstream instead.
_UNSCOPABLE = frozenset({"cryptography"})


def _entries() -> list[tuple[str, str, str]]:
    """Return (ecosystem, directory, body) for each update entry, in file order."""
    text = _CONFIG.read_text(encoding="utf-8")
    parsed: list[tuple[str, str, str]] = []
    for entry in re.split(r"^  - (?=package-ecosystem:)", text, flags=re.MULTILINE)[1:]:
        lines = [line.strip() for line in entry.splitlines()]
        ecosystem = lines[0].removeprefix("package-ecosystem: ")
        directory = next(
            line.removeprefix("directory: ") for line in lines if line.startswith("directory: ")
        )
        parsed.append((ecosystem, directory, entry))
    return parsed


def _entry_body(ecosystem: str, directory: str) -> str:
    """Return one entry's text, so an ignore cannot be read from a neighbour's.

    An ignore belongs to the entry that encloses it. Reading the file as a flat list of
    `dependency-name` lines would accept one that had drifted under another ecosystem,
    where it covers nothing, while still reading as present.
    """
    for candidate, target, body in _entries():
        if (candidate, target) == (ecosystem, directory):
            return body
    raise AssertionError(f"no {ecosystem} entry for {directory} in {_CONFIG.name}")


def _mirrored_roots() -> frozenset[str]:
    manifest = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return frozenset(
        Requirement(item).name
        for item in manifest["dependency-groups"]["qualification-hermes"]
    )


def _declared_worker_roots() -> frozenset[str]:
    """Return the CUDA worker's declared roots, following its `-r` include."""
    names: set[str] = set()
    pending = [_WORKER_ROOTS]
    while pending:
        current = pending.pop()
        for raw in current.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("--"):
                continue
            if line.startswith("-r "):
                pending.append(current.parent / line.removeprefix("-r ").strip())
                continue
            line = line.split("--hash", 1)[0].rstrip().removesuffix("\\").strip()
            if line:
                names.add(Requirement(line).name)
    return frozenset(names)


def _ignored_names(ecosystem: str = "uv", directory: str = "/") -> frozenset[str]:
    return frozenset(_IGNORED.findall(_entry_body(ecosystem, directory)))


# Every ignore list is derived, never authored. Each entry names the set it must equal and
# the roots it may leave out, so the binding holds in both directions at once: a source
# root that stops being ignored is caught, and so is an ignore that outlives its root.
_IGNORE_BINDINGS = (
    pytest.param(
        "uv",
        "/",
        "the qualification-hermes group in pyproject.toml",
        _mirrored_roots,
        _UNSCOPABLE,
        id="uv-mirrors-the-qualified-commit",
    ),
    pytest.param(
        "pip",
        "/requirements",
        f"the roots declared in {_WORKER_ROOTS.name}",
        _declared_worker_roots,
        frozenset(),
        id="pip-mirrors-the-worker-closure-roots",
    ),
)


@pytest.mark.parametrize(("ecosystem", "directory", "source", "derive", "exempt"), _IGNORE_BINDINGS)
def test_each_ignore_list_equals_the_set_it_is_derived_from(
    ecosystem: str,
    directory: str,
    source: str,
    derive: Callable[[], frozenset[str]],
    exempt: frozenset[str],
) -> None:
    """An ignore list states which versions are not this repository's to choose.

    Both lists are computed elsewhere. The uv entry follows the group that mirrors the
    qualified upstream commit; the /requirements entry follows the roots the worker
    closure is compiled from. Equality is what binds them, because either direction alone
    leaves a hole: a subset check misses a stale ignore that goes on suppressing security
    updates after its root is gone, and a superset check misses a new root that a bot is
    then free to propose.
    """
    expected = derive() - exempt
    ignored = _ignored_names(ecosystem, directory)
    assert ignored == expected, (
        f"the {ecosystem} entry for {directory} no longer matches {source}. "
        f"It ignores {sorted(ignored)} but should ignore exactly {sorted(expected)}: "
        f"{sorted(expected - ignored)} is unignored and {sorted(ignored - expected)} is "
        "ignored without a source. An ignore also suppresses security updates, so both "
        "directions are deliberate."
    )


def test_the_root_python_ecosystem_owns_its_lockfile() -> None:
    """`pip` edits pyproject.toml and leaves uv.lock stale; `uv` writes both.

    Under the pip ecosystem every manifest bump failed `uv lock --check` before any test
    ran, which made the pull request unreviewable rather than merely wrong. Reverting this
    line reintroduces that whole class, so it is asserted rather than remembered.
    """
    assert ("uv", "/") in {(ecosystem, directory) for ecosystem, directory, _ in _entries()}, (
        "the root Python ecosystem must be uv, which maintains pyproject.toml and uv.lock "
        "together; pip rewrites the manifest alone and every such update fails uv lock --check"
    )
