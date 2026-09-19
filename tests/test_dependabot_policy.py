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
from pathlib import Path

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


def test_every_mirrored_root_is_ignored_except_the_unscopable_one() -> None:
    """A derived pin is not a version a bot may propose.

    The qualification-hermes group mirrors the commit pinned by
    scripts/qualification_hermes_source.py, so each of its roots is computed rather than
    chosen. Adding a root without ignoring it reopens a class of pull requests that cannot
    resolve against that commit, and this binding is the only thing that notices.
    """
    unignored = _mirrored_roots() - _ignored_names()
    assert unignored == _UNSCOPABLE, (
        "the Dependabot ignore list no longer matches the qualification-hermes group in "
        f"pyproject.toml; unignored mirrored roots are {sorted(unignored)} but only "
        f"{sorted(_UNSCOPABLE)} can be left out. Ignore the new root, or record here why it "
        "cannot be scoped."
    )


def test_the_uv_ignore_list_holds_nothing_but_mirrored_roots() -> None:
    """An ignore hides a dependency from security updates too, so it needs a stated reason.

    The mirrored group is that reason for the root project. Anything else accumulating
    there would be silently unwatched, which is the failure this file exists to prevent
    rather than to permit.
    """
    unexplained = _ignored_names() - _mirrored_roots()
    assert not unexplained, (
        f"{sorted(unexplained)} is ignored under the root uv entry but is not a "
        "qualification-hermes root. An ignore also suppresses security updates, so add the "
        "reason here deliberately rather than leaving it unexplained in the configuration."
    )


def test_no_mirrored_root_is_ignored_outside_the_root_uv_entry() -> None:
    """A mirrored ignore under another entry reads as present and covers nothing.

    The mirrored roots belong to the root project, so their ignores are only effective
    under the uv entry. Moved beneath the /requirements pip entry or the /web npm entry
    they would still satisfy a flat reading of this file while Dependabot resumed
    proposing the very bumps these tests exist to refuse. Other entries keep their own
    ignores for their own reasons; only the mirrored group is bound here.
    """
    mirrored = _mirrored_roots()
    for ecosystem, directory, body in _entries():
        if (ecosystem, directory) == ("uv", "/"):
            continue
        strays = frozenset(_IGNORED.findall(body)) & mirrored
        assert not strays, (
            f"{sorted(strays)} is ignored under the {ecosystem} entry for {directory}, "
            "where it does not cover the root project; move it to the uv entry"
        )


def test_the_worker_closure_roots_are_never_bot_updated() -> None:
    """A declared root moves by recompiling the closure, never by editing its output.

    requirements/kokoro-cuda-worker-win-py311.txt is generated from the roots in
    kokoro-cuda-worker.in. Its derived entries are safe for a bot, which supplies the
    version and every published wheel hash, but a root changed in the output alone would
    contradict the file it was compiled from. Attempting them also fails the whole
    evaluation, because the pinned kokoro-onnx is a win32-only wheel that cannot resolve
    on Dependabot's Linux runner.
    """
    unignored = _declared_worker_roots() - _ignored_names("pip", "/requirements")
    assert not unignored, (
        f"{sorted(unignored)} is declared in {_WORKER_ROOTS.name} but not ignored for the "
        "/requirements entry; a bot changing a declared root in the compiled output would "
        "contradict the roots it was compiled from"
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
