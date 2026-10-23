"""Stage an experimental native plugin only into an empty, caller-owned profile.

No configuration, installation, subprocess, provider or model action is performed.
The caller owns settlement/cleanup and must retain a partial profile on failure.
Use the existing hidden Processes/Containment owner for a later loader observation.
This module does not launch that observation or qualify real-provider behavior.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

REVIEWED_REVISION = "e7477d9d3df1aa6319566d911b097c6f69bf3c15"
REVIEWED_SOURCES = {
    "entrypoint.py": "eb998e0b8a517aa5bd6e905d17bbd0dd873e2a96961379743fafdc3bf33ac282",
    "policy.py": "7967d036cb9117f8a927cd950d81f690307f7ef49700fbf558228fde1e34733b",
}
REVIEWED_SIZES = {"entrypoint.py": 2366, "policy.py": 17574}
PLUGIN = "ordinary-lookup-experimental"
POLICY = "ordinary-lookup/v1"
BOOTSTRAP = b"from .entrypoint import register\n"
MANIFEST = (
    b'name: ordinary-lookup-experimental\nversion: "0.0.1"\n'
    b'description: "Experimental ordinary lookup; dedicated disposable profile only."\n'
)


class StagingRefusal(RuntimeError):
    """A fixed-category refusal; no source content or private path is exposed."""


def stage_lookup_policy(owned_profile: Path, reviewed_source: Path) -> dict[str, object]:
    """Write only the source-bound native artifact; never copy the synthetic fixture.

    ``owned_profile`` must already be an empty directory created by its process
    owner. This API neither discovers nor creates a profile and cannot update one.
    The caller separately supplies profile configuration and explicit override
    consent before any real pinned-loader observation. No loader runs here.
    """
    category = "profile-not-fresh"
    try:
        if owned_profile.is_symlink() or not owned_profile.is_dir() or any(owned_profile.iterdir()):
            raise StagingRefusal(category)
        category = "reviewed-source-unreadable"
        try:
            payloads = {}
            for name in REVIEWED_SOURCES:
                with (reviewed_source / name).open("rb") as stream:
                    payloads[name] = stream.read(REVIEWED_SIZES[name] + 1)
        except OSError:
            raise StagingRefusal(category) from None
        category = "reviewed-source-mismatch"
        if any(
            hashlib.sha256(payloads[name]).hexdigest() != expected
            for name, expected in REVIEWED_SOURCES.items()
        ):
            raise StagingRefusal(category)
        payloads.update({"__init__.py": BOOTSTRAP, "plugin.yaml": MANIFEST})
        category = "artifact-write-refused"
        package = owned_profile / "plugins" / PLUGIN
        package.mkdir(parents=True)
        for name, content in payloads.items():
            with (package / name).open("xb") as stream:
                stream.write(content)
        category = "artifact-integrity-refused"
        if {path.name for path in package.iterdir()} != set(payloads):
            raise StagingRefusal(category)
        for name, content in payloads.items():
            with (package / name).open("rb") as stream:
                if stream.read(len(content) + 1) != content:
                    raise StagingRefusal(category)
        category = None
        return {
            "plugin": PLUGIN,
            "policy": POLICY,
            "reviewed_revision": REVIEWED_REVISION,
            "files_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(payloads.items())
            },
        }
    finally:
        if category is not None:
            with contextlib.suppress(OSError, ValueError):
                print(
                    "[lookup-policy-stage] "
                    + json.dumps(
                        {"category": category, "version": 1}, separators=(",", ":"), sort_keys=True
                    ),
                    flush=True,
                )
