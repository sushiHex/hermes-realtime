"""Dependabot may only propose changes it can complete in the files it writes.

Every failing bot pull request in this repository has had one shape: the change needs a
second artifact regenerated that the bot does not write. `uv.lock` after a manifest edit,
the tracked browser assets after a `/web` bump, the reviewed action pins in
`tests/test_release_workflow.py` after an action bump. The configuration exists to keep
those classes apart, and these tests bind the parts of it that silently rot.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / ".github" / "dependabot.yml"
_IGNORED = re.compile(r'^\s*- dependency-name: "([^"]+)"\s*$', re.MULTILINE)

# The one mirrored root an ignore cannot cover. cryptography is pinned by the qualified
# upstream commit *and* constrained independently by the dev group, and an ignore applies
# to a dependency, not to one dependency group. Silencing it would silence dev's own
# security updates, so the bump is refused downstream instead.
_UNSCOPABLE = frozenset({"cryptography"})


def _mirrored_roots() -> frozenset[str]:
    manifest = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    group = manifest["dependency-groups"]["qualification-hermes"]
    return frozenset(Requirement(item).name for item in group)


def _ignored_names() -> frozenset[str]:
    return frozenset(_IGNORED.findall(_CONFIG.read_text(encoding="utf-8")))


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


def test_the_ignore_list_holds_nothing_but_mirrored_roots() -> None:
    """An ignore hides a dependency from security updates too, so it needs a stated reason.

    The mirrored group is that reason. Anything else accumulating here would be silently
    unwatched, which is the failure this file exists to prevent rather than to permit.
    """
    unexplained = _ignored_names() - _mirrored_roots()
    assert not unexplained, (
        f"{sorted(unexplained)} is ignored but is not a qualification-hermes root. An ignore "
        "also suppresses security updates, so add the reason here deliberately rather than "
        "leaving it unexplained in the configuration."
    )


def test_the_root_python_ecosystem_owns_its_lockfile() -> None:
    """`pip` edits pyproject.toml and leaves uv.lock stale; `uv` writes both.

    Under the pip ecosystem every manifest bump failed `uv lock --check` before any test
    ran, which made the pull request unreviewable rather than merely wrong. Reverting this
    line reintroduces that whole class, so it is asserted rather than remembered.
    """
    config = _CONFIG.read_text(encoding="utf-8")
    assert "package-ecosystem: uv\n    directory: /\n" in config, (
        "the root Python ecosystem must be uv, which maintains pyproject.toml and uv.lock "
        "together; pip rewrites the manifest alone and every such update fails uv lock --check"
    )
