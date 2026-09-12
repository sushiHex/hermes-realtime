"""Compose the complete #47 input authority from independently minted leaves.

This is the final stage-three input boundary.  It neither launches the full host
nor derives scenario/report outcomes; those remain the #62 composition owner's
responsibility.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from weakref import WeakKeyDictionary

from scripts import qualification_benchmark as benchmark
from scripts import qualification_build_inputs as build_inputs
from scripts import qualification_builds as builds
from scripts import qualification_candidate_files as candidate_files
from scripts import qualification_chrome_files as chrome_files
from scripts import qualification_codex_files as codex_files
from scripts import qualification_dependency_files as dependency_files
from scripts import qualification_hermes_pluginmanager_runtime as hermes_runtime
from scripts import qualification_hermes_source as hermes_source
from scripts import qualification_kokoro_resources as kokoro_resources
from scripts import qualification_linux_prerequisite as linux_prerequisite
from scripts import qualification_livekit_files as livekit_files
from scripts import qualification_moonshine_catalog as moonshine_catalog
from scripts import qualification_moonshine_resources as moonshine_resources
from scripts import qualification_runtime_environment as runtime_environment
from scripts import qualification_tool_environment as tool_environment
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualify_evidence_slice_zero import QualificationInputClosure
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    retained_input_metadata,
)

_RUNTIME_PURPOSES = frozenset(
    {
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
        "hermes_v020_pluginmanager_runtime",
    }
)
_PROVIDER_PURPOSES = frozenset(
    {"realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime"}
)
_DEPENDENCY_PURPOSES = frozenset(
    {
        "build",
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
        "realtime_linux_runtime",
        "hermes_v020_pluginmanager_runtime",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _closed(owner: OwnedQualificationWorkV1, label: str) -> None:
    _require(
        type(owner) is OwnedQualificationWorkV1
        and owner._closing
        and owner._closed
        and owner._unrecoverable is None,
        f"{label} cleanup is incomplete",
    )


class AcceptedQualificationInputsV1:
    """Opaque complete input authority; validity follows the final file lease."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("accepted qualification inputs are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Acceptance:
    files: RetainedQualificationInputFilesV1
    tool_environment: tool_environment.CompletedToolEnvironmentV1
    candidate: candidate_files.BoundCandidateFilesV1
    build_files: build_inputs.BoundBuildFilesV1
    build_outputs: builds.BoundBuildOutputsV1
    dependencies: dependency_files.BoundDependencyFilesV1
    linux: linux_prerequisite.BoundPrefinalLinuxReceiptV1
    runtimes: tuple[runtime_environment.CompletedRuntimeEnvironmentV1, ...]
    moonshine: tuple[moonshine_resources.BoundMoonshineResourcesV1, ...]
    kokoro: tuple[kokoro_resources.BoundKokoroResourcesV1, ...]
    benchmark: benchmark.BoundBenchmarkFilesV1
    hermes_source: hermes_source.BoundHermesPublisherSourceV1
    hermes_pluginmanager: hermes_runtime.CompletedHermesPluginManagerRuntimeV1
    chrome: chrome_files.BoundChromeFilesV1
    livekit: livekit_files.BoundLiveKitFilesV1
    codex: codex_files.BoundCodexFilesV1
    metadata: QualificationInputClosure


_ACCEPTED: WeakKeyDictionary[AcceptedQualificationInputsV1, _Acceptance] = WeakKeyDictionary()


def _purpose_map(
    values: tuple[Any, ...],
    metadata: Callable[[Any], Any],
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        purpose = metadata(value).purpose
        _require(purpose not in result, f"{label} purpose is duplicated")
        result[purpose] = value
    _require(set(result) == expected, f"{label} purposes are incomplete")
    return result


def _moonshine_shared(value: object) -> tuple[object, ...]:
    names = (
        "qualification_input_sha256",
        "source_commit",
        "source_tree",
        "source_archive_sha256",
        "worker_sha256",
        "distribution_sha256",
        "python_api_sha256",
        "native_library_sha256",
        "catalog_identity_sha256",
        "model_identity_sha256",
        "resource_count",
        "resource_bytes",
    )
    return tuple(getattr(value, name) for name in names)


def _kokoro_shared(value: object) -> tuple[object, ...]:
    names = (
        "qualification_input_sha256",
        "source_commit",
        "provider_source_sha256",
        "distribution_sha256",
        "model_identity_sha256",
        "resource_count",
        "resource_bytes",
    )
    return tuple(getattr(value, name) for name in names)


def _validate(value: _Acceptance) -> QualificationInputClosure:
    files = value.files
    closure = retained_input_metadata(files)
    candidate = candidate_files._candidate_files_for_consumer(value.candidate)
    _require(candidate.files is files, "candidate final input authority differs")

    final_build_inputs = build_inputs._build_files_for_consumer(value.build_files)
    inputs = build_inputs._build_input_facts(final_build_inputs.inputs)
    _require(final_build_inputs.files is files, "build final input authority differs")
    _require(
        inputs.archive is candidate.archive and inputs.identity is candidate.identity,
        "build and candidate source authorities differ",
    )
    completed_tools = tool_environment._completed_tool_environment_for_consumer(
        value.tool_environment
    )
    _require(
        completed_tools.tools is inputs.tools
        and completed_tools.historical.metadata == inputs.tool_distributions,
        "completed build tool authority differs",
    )

    final_outputs = builds._build_outputs_for_consumer(value.build_outputs)
    produced_builds = builds._candidate_builds_for_consumer(final_outputs.builds)
    _require(
        final_outputs.files is files
        and produced_builds.inputs is final_build_inputs.inputs
        and produced_builds.archive is candidate.archive
        and produced_builds.identity is candidate.identity,
        "build output lineage differs",
    )
    _closed(produced_builds.output_owner, "build output staging")

    dependencies = dependency_files._dependency_files_for_consumer(value.dependencies)
    _require(
        dependencies.files is files
        and dependencies.candidate is value.candidate
        and {item.purpose for item in dependencies.metadata.wheelhouses} == _DEPENDENCY_PURPOSES
        and dependencies.linux_target is not None,
        "complete dependency authority differs",
    )

    linux = linux_prerequisite._bound_linux_receipt_for_consumer(value.linux)
    linux_dependencies = dependency_files._dependency_purpose_for_consumer(linux.linux_runtime)
    _require(
        linux_dependencies.files is files
        and linux_dependencies.candidate is value.candidate
        and {item.purpose for item in linux_dependencies.metadata.wheelhouses}
        == {"realtime_linux_runtime"}
        and linux_dependencies.linux_target is dependencies.linux_target,
        "Linux prerequisite graph differs",
    )
    _closed(linux.preparation_owner, "Linux input preparation")

    runtime_values = tuple(
        runtime_environment._completed_runtime_for_consumer(item) for item in value.runtimes
    )
    runtimes = _purpose_map(
        runtime_values, lambda item: item.metadata, _RUNTIME_PURPOSES, "runtime"
    )
    _require(
        len({id(item) for item in runtime_values}) == len(runtime_values)
        and len({id(item.files) for item in runtime_values}) == len(runtime_values)
        and len({id(item.resources) for item in runtime_values}) == len(runtime_values)
        and len({id(invocation) for item in runtime_values for invocation in item.invocations})
        == 2 * len(runtime_values),
        "runtime environments are not distinct",
    )
    for runtime in runtime_values:
        _require(
            runtime.dependencies is value.dependencies
            and runtime.outputs is value.build_outputs
            and runtime.tools is inputs.tools,
            "runtime authority graph differs",
        )
        _closed(runtime.work, f"{runtime.metadata.purpose} runtime")

    moonshine_bindings = tuple(moonshine_resources._binding(item) for item in value.moonshine)
    moonshine_by_purpose = _purpose_map(
        moonshine_bindings, lambda item: item.metadata, _PROVIDER_PURPOSES, "Moonshine"
    )
    _require(
        len({_moonshine_shared(item.metadata) for item in moonshine_bindings}) == 1,
        "Moonshine shared resource authority differs",
    )
    for purpose, resources in moonshine_by_purpose.items():
        catalog = moonshine_catalog._binding(resources.catalog)
        _require(
            resources.files is files
            and resources.candidate is value.candidate
            and resources.dependencies is value.dependencies
            and catalog.runtime is runtimes[purpose]
            and catalog.work is runtimes[purpose].work
            and catalog.dependencies is value.dependencies
            and catalog.outputs is value.build_outputs
            and catalog.candidate is value.candidate,
            "Moonshine runtime or resource graph differs",
        )

    kokoro_bindings = tuple(kokoro_resources._binding(item) for item in value.kokoro)
    kokoro_by_purpose = _purpose_map(
        kokoro_bindings, lambda item: item.metadata, _PROVIDER_PURPOSES, "Kokoro"
    )
    _require(
        len({_kokoro_shared(item.metadata) for item in kokoro_bindings}) == 1,
        "Kokoro shared resource authority differs",
    )
    for purpose, resources in kokoro_by_purpose.items():
        _require(
            resources.files is files
            and resources.candidate is value.candidate
            and resources.dependencies is value.dependencies
            and purpose in runtimes,
            "Kokoro runtime or resource graph differs",
        )

    benchmark_files = benchmark._benchmark_files_for_consumer(value.benchmark)
    produced_benchmark = benchmark._completed(benchmark_files.produced)
    _require(
        benchmark_files.files is files
        and benchmark_files.candidate is value.candidate
        and produced_benchmark.inputs is final_build_inputs.inputs,
        "benchmark authority graph differs",
    )
    _closed(produced_benchmark.output_owner, "benchmark output staging")

    source = hermes_source._bound_hermes_source_for_consumer(value.hermes_source)
    _require(
        source.inputs is final_build_inputs.inputs and source.files is files,
        "Hermes source authority graph differs",
    )
    _closed(source.preparation_owner, "Hermes source preparation")
    plugin = hermes_runtime._completed(value.hermes_pluginmanager)
    _require(
        plugin.source is value.hermes_source
        and plugin.runtime is runtimes["hermes_v020_pluginmanager_runtime"],
        "Hermes PluginManager runtime graph differs",
    )

    chrome = chrome_files._chrome_files_for_consumer(value.chrome)
    codex = codex_files._codex_files_for_consumer(value.codex)
    livekit = livekit_files._livekit_files_for_consumer(value.livekit)
    _require(
        chrome.files is files
        and chrome.candidate is value.candidate
        and codex.files is files
        and codex.candidate is value.candidate
        and livekit.files is files
        and livekit.inputs is final_build_inputs.inputs,
        "publisher file authority graph differs",
    )
    _closed(livekit.preparation_owner, "LiveKit preparation")

    current = retained_input_metadata(files)
    _require(current == closure, "qualification input closure changed during acceptance")
    return current


def accept_qualification_inputs(
    files: RetainedQualificationInputFilesV1,
    *,
    tool_environment: tool_environment.CompletedToolEnvironmentV1,
    candidate: candidate_files.BoundCandidateFilesV1,
    build_files: build_inputs.BoundBuildFilesV1,
    build_outputs: builds.BoundBuildOutputsV1,
    dependencies: dependency_files.BoundDependencyFilesV1,
    linux: linux_prerequisite.BoundPrefinalLinuxReceiptV1,
    runtimes: tuple[runtime_environment.CompletedRuntimeEnvironmentV1, ...],
    moonshine: tuple[moonshine_resources.BoundMoonshineResourcesV1, ...],
    kokoro: tuple[kokoro_resources.BoundKokoroResourcesV1, ...],
    benchmark: benchmark.BoundBenchmarkFilesV1,
    hermes_source: hermes_source.BoundHermesPublisherSourceV1,
    hermes_pluginmanager: hermes_runtime.CompletedHermesPluginManagerRuntimeV1,
    chrome: chrome_files.BoundChromeFilesV1,
    livekit: livekit_files.BoundLiveKitFilesV1,
    codex: codex_files.BoundCodexFilesV1,
) -> AcceptedQualificationInputsV1:
    """Accept only one exact, complete, cleaned stage-three input graph."""
    _require(
        type(runtimes) is tuple and type(moonshine) is tuple and type(kokoro) is tuple,
        "qualification input capability collections differ",
    )
    pending = _Acceptance(
        files,
        tool_environment,
        candidate,
        build_files,
        build_outputs,
        dependencies,
        linux,
        runtimes,
        moonshine,
        kokoro,
        benchmark,
        hermes_source,
        hermes_pluginmanager,
        chrome,
        livekit,
        codex,
        retained_input_metadata(files),
    )
    metadata = _validate(pending)
    receipt = object.__new__(AcceptedQualificationInputsV1)
    _ACCEPTED[receipt] = replace(pending, metadata=metadata)
    return receipt


def accepted_qualification_input_metadata(
    receipt: AcceptedQualificationInputsV1,
) -> QualificationInputClosure:
    if type(receipt) is not AcceptedQualificationInputsV1:
        raise TypeError("accepted qualification input capability type differs")
    _require(receipt in _ACCEPTED, "accepted qualification input capability is unregistered")
    value = _ACCEPTED[receipt]
    _require(_validate(value) == value.metadata, "accepted qualification input facts differ")
    return value.metadata


def _accepted_qualification_inputs_for_consumer(
    receipt: AcceptedQualificationInputsV1,
) -> _Acceptance:
    """Return the exact validated leaves to the later #62 composition owner."""
    accepted_qualification_input_metadata(receipt)
    return _ACCEPTED[receipt]
