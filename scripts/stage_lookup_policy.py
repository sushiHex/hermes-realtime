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

REVIEWED_REVISION = "dc9e2994c414b2e4ceb7310a6e44662786c82400"
REVIEWED_SOURCES = {
    "entrypoint.py": "484ff3b2743e824abe22d3c6a87d3af26912ccc07f891286f088a6971d32ae50",
    "policy.py": "ed4d13e39b6ef01557b36bac5a0fbc577ae75f2c62dd4c14d3dcd8e12897a9b9",
}
REVIEWED_SIZES = {'entrypoint.py': 2612, 'policy.py': 53739}
REVIEWED_SHARED_SOURCES = {
    "evidence_presentation": "123d7b61967b6e942a1b05bc1073e029de59fb8b04d4b9d6b973aecadead8519",
    "presentation_transport": "fdc8f816487cfab3b6f1b341f4d86b37ade0b5986e335cb41a5f47d0d367061f",
    "result_limits": "cdb48e4dd7d001b00086a7d816f2d9821e4419ef0799f619f776e16a065adfdb",
}
REVIEWED_SHARED_SIZES = {
    "evidence_presentation": 29825, "presentation_transport": 23508, "result_limits": 833,
}
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
        # Pin the shared source too, without importing an ambient installed
        # runtime. The preview's source/wheel/installed comparator separately
        # establishes that these are the bytes actually executed.
        category = "reviewed-shared-source-mismatch"
        shared_root = Path(__file__).resolve().parents[1] / "src/hermes_realtime/integration"
        for name, expected in REVIEWED_SHARED_SOURCES.items():
            module_root = shared_root.parent if name == "result_limits" else shared_root
            with (module_root / (name + ".py")).open("rb") as stream:
                raw = stream.read(REVIEWED_SHARED_SIZES[name] + 1)
            if hashlib.sha256(raw).hexdigest() != expected:
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
