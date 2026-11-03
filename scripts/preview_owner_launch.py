"""Start the maintained preview owner independently of its Windows caller.

The caller binds the interpreter/active plan and supplies an existing private
output directory. A spawn is only a launch request; serve owns readiness, its
Containment and settlement. No fallback to an attached process is permitted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

from scripts.qualification_launch import WINDOWLESS_BREAKAWAY_FLAGS, _outside_jobs

Category = Literal[
    "windows_required", "output_exists", "owner_output_unavailable",
    "breakaway_not_permitted", "owner_spawn_failed", "owner_inherited_job",
    "membership_unavailable",
]


class PreviewOwnerLaunchRefusal(RuntimeError):
    def __init__(self, category: Category) -> None:
        super().__init__(category)
        self.category = category


def assert_independent_owner() -> None:
    """Call in the actual serve process before creating its own Containment."""
    try:
        outside = _outside_jobs()
    except OSError:
        raise PreviewOwnerLaunchRefusal("membership_unavailable") from None
    if not outside:
        raise PreviewOwnerLaunchRefusal("owner_inherited_job")


def launch_preview_owner(
    *, interpreter: Path, active: Path, output: Path,
) -> subprocess.Popen[bytes]:
    if sys.platform != "win32":
        raise PreviewOwnerLaunchRefusal("windows_required")
    try:
        # Inherit the prepared output directory's private ACL. Exclusive files
        # preserve earlier evidence and never depend on a caller-owned pipe.
        with (output / "owner.private.log").open("xb") as stdout, (
            output / "owner.stderr.private.log"
        ).open("xb") as stderr:
            try:
                return subprocess.Popen(
                    [str(interpreter), "-I", "-B",
                     str(Path(__file__).with_name("preview_cutover.py")),
                     "serve", "--active", str(active)],
                    cwd=Path(__file__).resolve().parents[1],
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=WINDOWLESS_BREAKAWAY_FLAGS, close_fds=True,
                )
            except OSError as error:
                category: Category = (
                    "breakaway_not_permitted" if getattr(error, "winerror", None) == 5
                    else "owner_spawn_failed"
                )
                raise PreviewOwnerLaunchRefusal(category) from None
    except FileExistsError:
        raise PreviewOwnerLaunchRefusal("output_exists") from None
    except OSError:
        raise PreviewOwnerLaunchRefusal("owner_output_unavailable") from None
