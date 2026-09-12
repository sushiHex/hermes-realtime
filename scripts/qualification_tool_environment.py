"""Own complete admitted Windows tool trees, without claiming their execution.

The controller must own every consumer and observe its final exit before leaving
this context. Build dependencies, invocation receipts and durable recovery remain
separate authorities; this file owner cannot substitute for them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
    owned_execution_files,
)
from scripts.qualification_tool_distributions import (
    AdmittedToolDistributionV1,
    ToolDistributionMetadataV1,
    _distribution,
    _tool_distribution_files,
    tool_distribution_metadata,
)

_ROLES = ("build_python", "git", "uv")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ImmutableToolEnvironmentV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("tool environments are created by their owner only")


@dataclass(frozen=True, slots=True)
class _Environment:
    files: ImmutableExecutionFilesV1
    distributions: tuple[AdmittedToolDistributionV1, ...]


_LIVE: WeakKeyDictionary[ImmutableToolEnvironmentV1, _Environment] = WeakKeyDictionary()


def _environment(receipt: ImmutableToolEnvironmentV1) -> _Environment:
    if type(receipt) is not ImmutableToolEnvironmentV1:
        raise TypeError("tool environment capability type differs")
    _require(receipt in _LIVE, "tool environment capability is closed or unregistered")
    value = _LIVE[receipt]
    _execution_files_for_consumer(value.files)
    for distribution in value.distributions:
        tool_distribution_metadata(distribution)
    return value


@contextmanager
def owned_tool_environment(
    distributions: dict[str, AdmittedToolDistributionV1],
) -> Iterator[ImmutableToolEnvironmentV1]:
    """Seal every admitted member under a distinct role namespace, with no pruning."""
    _require(
        type(distributions) is dict and set(distributions) == set(_ROLES),
        "tool environment requires all three governed distributions",
    )
    selected = tuple(distributions[role] for role in _ROLES)
    for role, distribution in zip(_ROLES, selected, strict=True):
        _require(
            tool_distribution_metadata(distribution).role == role, "tool role identity differs"
        )
    contents = {
        role + "/" + member: raw
        for role, distribution in zip(_ROLES, selected, strict=True)
        for member, raw in _tool_distribution_files(distribution)
    }
    with owned_execution_files(contents) as files:
        receipt = object.__new__(ImmutableToolEnvironmentV1)
        _LIVE[receipt] = _Environment(files, selected)
        try:
            yield receipt
        finally:
            del _LIVE[receipt]


def tool_environment_metadata(
    receipt: ImmutableToolEnvironmentV1,
) -> tuple[ToolDistributionMetadataV1, ...]:
    return tuple(tool_distribution_metadata(item) for item in _environment(receipt).distributions)


def _tool_image_for_consumer(
    receipt: ImmutableToolEnvironmentV1,
    role: str,
) -> tuple[Path, str, str]:
    _require(type(role) is str and role in _ROLES, "tool role is unavailable")
    value = _environment(receipt)
    distribution = _distribution(value.distributions[_ROLES.index(role)])
    member = distribution.policy.executable
    raw = dict(distribution.files)[member]
    root = _execution_files_for_consumer(value.files)
    return (
        root.joinpath(role, *member.split("/")),
        hashlib.sha256(raw).hexdigest(),
        distribution.policy.version,
    )
