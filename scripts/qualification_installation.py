"""Shared offline installer and immutable package snapshot for governed recipes.

This private mechanism does not authenticate caller-supplied wheels or authorize
imports. The build and runtime recipes must first establish their input authority
and retain it in the work owner through every invocation and cleanup observation.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
    _security,
    owned_execution_files,
)
from scripts.qualification_file_seals import _retain, retain_file_seals, sealed_file_bytes
from scripts.qualification_installed_files import InstalledFileMetadataV1, inspect_installed_wheels
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_tool_environment import (
    ImmutableToolEnvironmentV1,
    _tool_image_for_consumer,
)
from scripts.qualification_tool_process import CompletedToolInvocationV1
from scripts.windows_storage_oracle import _identity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@contextmanager
def _workspace() -> Iterator[Path]:
    parent = Path(tempfile.gettempdir()).resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="hermes-build-environment-", dir=parent)).resolve(
        strict=True
    )
    _require(root.parent == parent, "build workspace parent differs")
    with _retain(root, directory=True) as (_, info):
        identity = _identity(info)
    try:
        _security(root, directory=True, frozen=False)
        yield root
    finally:
        _require(
            root.resolve(strict=True) == root and root.parent == parent,
            "build workspace cleanup target differs",
        )
        with _retain(root, directory=True) as (_, current):
            _require(_identity(current) == identity, "build workspace cleanup identity differs")
        shutil.rmtree(root)
        _require(not root.exists(), "build workspace cleanup is incomplete")


@dataclass(frozen=True, slots=True)
class _InstalledPackages:
    files: ImmutableExecutionFilesV1
    resources: ImmutableExecutionFilesV1
    installer: CompletedToolInvocationV1
    inventory: InstalledFileMetadataV1
    workspace: Path


def _install_packages(
    work: OwnedQualificationWorkV1,
    tools: ImmutableToolEnvironmentV1,
    *,
    wheels: dict[str, bytes],
    requirements: bytes,
    constraints: bytes,
    worker: bytes,
) -> _InstalledPackages:
    python, _, python_version = _tool_image_for_consumer(tools, "build_python")
    workspace = work.enter(_workspace())
    resources = work.enter(
        owned_execution_files(
            {
                "requirements.txt": requirements,
                "constraints.txt": constraints,
                "worker.py": worker,
                **{"wheels/" + name: raw for name, raw in wheels.items()},
            }
        )
    )
    source = _execution_files_for_consumer(resources)
    target = workspace / "installed"
    installer = work.run_tool(
        tools,
        "uv",
        (
            "pip",
            "install",
            "--python",
            str(python),
            "--target",
            str(target),
            "--no-index",
            "--find-links",
            str(source / "wheels"),
            "--require-hashes",
            "--no-deps",
            "--no-build",
            "--link-mode",
            "copy",
            "-r",
            str(source / "requirements.txt"),
            "-c",
            str(source / "constraints.txt"),
        ),
        workspace,
    )
    members: list[str] = []
    for path in target.rglob("*"):
        if path.is_file():
            members.append(path.relative_to(target).as_posix())
            _require(len(members) <= 16384, "installed file count exceeds its bound")
    seals = work.enter(retain_file_seals(target, tuple(sorted(members))))
    contents: dict[str, bytes] = {}
    total = 0
    for name in members:
        raw = sealed_file_bytes(seals, name, 128 * 1024**2)
        total += len(raw)
        _require(total <= 512 * 1024**2, "installed bytes exceed their bound")
        contents[name] = raw
    inventory = inspect_installed_wheels(
        requirements=requirements,
        constraints=constraints,
        wheels=wheels,
        python_version=python_version,
        platform="windows_amd64",
        installed=contents,
    )
    files = work.enter(owned_execution_files(contents))
    return _InstalledPackages(files, resources, installer, inventory, workspace)
