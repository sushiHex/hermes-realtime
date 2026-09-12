"""Final #47 composition checks; upstream observations here are explicitly synthetic."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path
from types import SimpleNamespace as Row

import pytest

_helpers = run_path(str(Path(__file__).with_name("test_qualification_candidate_files.py")))
source = _helpers["source"]
bound_graph = _helpers["bound_graph"]
dependency_graph = _helpers["dependency_graph"]


def _token(cls):
    return object.__new__(cls)


def _owner(*, closed: bool = True, failed: bool = False):
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    owner = OwnedQualificationWorkV1()
    if failed:
        owner._retain_failure(RuntimeError("synthetic cleanup failure"))
    elif closed:
        owner.close()
    return owner


def _patch(monkeypatch, module, name, values):
    monkeypatch.setattr(module, name, lambda receipt: values[receipt])


def _synthetic_capabilities(monkeypatch, files):
    """Model genuine opaque leaves without claiming that their native recipes ran."""
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
    from scripts import qualification_linux_prerequisite as linux
    from scripts import qualification_livekit_files as livekit_files
    from scripts import qualification_moonshine_catalog as moonshine_catalog
    from scripts import qualification_moonshine_resources as moonshine_resources
    from scripts import qualification_runtime_environment as runtime
    from scripts import qualification_tool_environment as tools
    from scripts.qualification_wheelhouse import AuthenticatedLinuxWheelTargetV1

    candidate = _token(candidate_files.BoundCandidateFilesV1)
    candidate_binding = Row(files=files, archive=object(), identity=object())
    _patch(
        monkeypatch,
        candidate_files,
        "_candidate_files_for_consumer",
        {candidate: candidate_binding},
    )

    original_tools = _token(tools.ImmutableToolEnvironmentV1)
    completed_tools = _token(tools.CompletedToolEnvironmentV1)
    tool_facts = ()
    _patch(
        monkeypatch,
        tools,
        "_completed_tool_environment_for_consumer",
        {completed_tools: Row(tools=original_tools, historical=Row(metadata=tool_facts))},
    )

    inputs = _token(build_inputs.BoundBuildInputsV1)
    input_binding = Row(
        archive=candidate_binding.archive,
        identity=candidate_binding.identity,
        tools=original_tools,
        tool_distributions=tool_facts,
    )
    build_files = _token(build_inputs.BoundBuildFilesV1)
    build_file_binding = Row(inputs=inputs, files=files)
    _patch(monkeypatch, build_inputs, "_build_input_facts", {inputs: input_binding})
    _patch(
        monkeypatch, build_inputs, "_build_files_for_consumer", {build_files: build_file_binding}
    )

    build_receipt = _token(builds.CandidateBuildsV1)
    build_binding = Row(
        inputs=inputs,
        archive=candidate_binding.archive,
        identity=candidate_binding.identity,
        output_owner=_owner(),
    )
    build_outputs = _token(builds.BoundBuildOutputsV1)
    output_binding = Row(builds=build_receipt, files=files)
    _patch(monkeypatch, builds, "_candidate_builds_for_consumer", {build_receipt: build_binding})
    _patch(monkeypatch, builds, "_build_outputs_for_consumer", {build_outputs: output_binding})

    target = _token(AuthenticatedLinuxWheelTargetV1)
    purposes = (
        "build",
        "hermes_v020_pluginmanager_runtime",
        "realtime_linux_runtime",
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
    )
    dependencies = _token(dependency_files.BoundDependencyFilesV1)
    dependency_binding = Row(
        files=files,
        candidate=candidate,
        linux_target=target,
        metadata=Row(wheelhouses=tuple(Row(purpose=item) for item in purposes)),
    )
    linux_dependencies = _token(dependency_files.BoundDependencyPurposeV1)
    linux_dependency_binding = Row(
        files=files,
        candidate=candidate,
        linux_target=target,
        metadata=Row(wheelhouses=(Row(purpose="realtime_linux_runtime"),)),
    )
    _patch(
        monkeypatch,
        dependency_files,
        "_dependency_files_for_consumer",
        {dependencies: dependency_binding},
    )
    _patch(
        monkeypatch,
        dependency_files,
        "_dependency_purpose_for_consumer",
        {linux_dependencies: linux_dependency_binding},
    )

    linux_receipt = _token(linux.BoundPrefinalLinuxReceiptV1)
    linux_binding = Row(
        linux_runtime=linux_dependencies,
        preparation_owner=_owner(),
    )
    _patch(monkeypatch, linux, "_bound_linux_receipt_for_consumer", {linux_receipt: linux_binding})

    runtime_receipts = []
    runtime_bindings = {}
    runtime_by_purpose = {}
    for purpose in (
        "realtime_windows_sdist_built_runtime",
        "hermes_v020_pluginmanager_runtime",
        "realtime_windows_direct_runtime",
    ):
        receipt = _token(runtime.CompletedRuntimeEnvironmentV1)
        binding = Row(
            dependencies=dependencies,
            outputs=build_outputs,
            tools=original_tools,
            work=_owner(),
            files=object(),
            resources=object(),
            invocations=(object(), object()),
            metadata=Row(purpose=purpose),
        )
        runtime_receipts.append(receipt)
        runtime_bindings[receipt] = binding
        runtime_by_purpose[purpose] = binding
    _patch(monkeypatch, runtime, "_completed_runtime_for_consumer", runtime_bindings)

    moonshine_receipts = []
    moonshine_bindings = {}
    catalog_bindings = {}
    moonshine_shared = dict(
        qualification_input_sha256="a" * 64,
        source_commit="b" * 40,
        source_tree="c" * 40,
        source_archive_sha256="d" * 64,
        worker_sha256="e" * 64,
        distribution_sha256="f" * 64,
        python_api_sha256="1" * 64,
        native_library_sha256="2" * 64,
        catalog_identity_sha256="3" * 64,
        model_identity_sha256="4" * 64,
        resource_count=9,
        resource_bytes=99,
    )
    for purpose in (
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
    ):
        catalog = _token(moonshine_catalog.BoundMoonshineCatalogV1)
        runtime_binding = runtime_by_purpose[purpose]
        catalog_bindings[catalog] = Row(
            runtime=runtime_binding,
            work=runtime_binding.work,
            dependencies=dependencies,
            outputs=build_outputs,
            candidate=candidate,
        )
        receipt = _token(moonshine_resources.BoundMoonshineResourcesV1)
        moonshine_bindings[receipt] = Row(
            files=files,
            candidate=candidate,
            dependencies=dependencies,
            catalog=catalog,
            metadata=Row(purpose=purpose, **moonshine_shared),
        )
        moonshine_receipts.append(receipt)
    _patch(monkeypatch, moonshine_catalog, "_binding", catalog_bindings)
    _patch(monkeypatch, moonshine_resources, "_binding", moonshine_bindings)

    kokoro_receipts = []
    kokoro_bindings = {}
    kokoro_shared = dict(
        qualification_input_sha256="a" * 64,
        source_commit="b" * 40,
        provider_source_sha256="5" * 64,
        distribution_sha256="6" * 64,
        model_identity_sha256="7" * 64,
        resource_count=2,
        resource_bytes=22,
    )
    for purpose in (
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
    ):
        receipt = _token(kokoro_resources.BoundKokoroResourcesV1)
        kokoro_bindings[receipt] = Row(
            files=files,
            candidate=candidate,
            dependencies=dependencies,
            metadata=Row(purpose=purpose, **kokoro_shared),
        )
        kokoro_receipts.append(receipt)
    _patch(monkeypatch, kokoro_resources, "_binding", kokoro_bindings)

    benchmark_produced = _token(benchmark.ProducedBenchmarkArtifactsV1)
    benchmark_production = Row(inputs=inputs, output_owner=_owner())
    benchmark_receipt = _token(benchmark.BoundBenchmarkFilesV1)
    benchmark_binding = Row(produced=benchmark_produced, files=files, candidate=candidate)
    _patch(monkeypatch, benchmark, "_completed", {benchmark_produced: benchmark_production})
    _patch(
        monkeypatch,
        benchmark,
        "_benchmark_files_for_consumer",
        {benchmark_receipt: benchmark_binding},
    )

    source_receipt = _token(hermes_source.BoundHermesPublisherSourceV1)
    source_binding = Row(inputs=inputs, files=files, preparation_owner=_owner())
    _patch(
        monkeypatch,
        hermes_source,
        "_bound_hermes_source_for_consumer",
        {source_receipt: source_binding},
    )
    plugin_receipt = _token(hermes_runtime.CompletedHermesPluginManagerRuntimeV1)
    plugin_binding = Row(
        source=source_receipt,
        runtime=runtime_by_purpose["hermes_v020_pluginmanager_runtime"],
    )
    _patch(monkeypatch, hermes_runtime, "_completed", {plugin_receipt: plugin_binding})

    chrome_receipt = _token(chrome_files.BoundChromeFilesV1)
    chrome_binding = Row(files=files, candidate=candidate)
    _patch(
        monkeypatch, chrome_files, "_chrome_files_for_consumer", {chrome_receipt: chrome_binding}
    )
    codex_receipt = _token(codex_files.BoundCodexFilesV1)
    codex_binding = Row(files=files, candidate=candidate)
    _patch(monkeypatch, codex_files, "_codex_files_for_consumer", {codex_receipt: codex_binding})
    livekit_receipt = _token(livekit_files.BoundLiveKitFilesV1)
    livekit_binding = Row(files=files, inputs=inputs, preparation_owner=_owner())
    _patch(
        monkeypatch,
        livekit_files,
        "_livekit_files_for_consumer",
        {livekit_receipt: livekit_binding},
    )

    arguments = dict(
        files=files,
        tool_environment=completed_tools,
        candidate=candidate,
        build_files=build_files,
        build_outputs=build_outputs,
        dependencies=dependencies,
        linux=linux_receipt,
        runtimes=tuple(runtime_receipts),
        moonshine=tuple(moonshine_receipts),
        kokoro=tuple(kokoro_receipts),
        benchmark=benchmark_receipt,
        hermes_source=source_receipt,
        hermes_pluginmanager=plugin_receipt,
        chrome=chrome_receipt,
        livekit=livekit_receipt,
        codex=codex_receipt,
    )
    bindings = Row(
        input=input_binding,
        build=build_binding,
        dependencies=dependency_binding,
        linux_dependencies=linux_dependency_binding,
        linux=linux_binding,
        runtimes=runtime_bindings,
        runtime_by_purpose=runtime_by_purpose,
        moonshine=moonshine_bindings,
        catalogs=catalog_bindings,
        kokoro=kokoro_bindings,
        benchmark=benchmark_production,
        hermes_source=source_binding,
        plugin=plugin_binding,
        chrome=chrome_binding,
        codex=codex_binding,
        livekit=livekit_binding,
    )
    return arguments, bindings


def test_complete_input_capability_is_opaque_and_registered_only():
    from scripts.qualification_input_acceptance import (
        AcceptedQualificationInputsV1,
        accepted_qualification_input_metadata,
    )

    with pytest.raises(TypeError):
        AcceptedQualificationInputsV1()
    with pytest.raises(TypeError):
        accepted_qualification_input_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        accepted_qualification_input_metadata(object.__new__(AcceptedQualificationInputsV1))


def test_complete_input_accepts_exact_clean_graph_and_expires_with_final_seals(
    dependency_graph, monkeypatch
):
    from scripts.qualification_input_acceptance import (
        _accepted_qualification_inputs_for_consumer,
        accept_qualification_inputs,
        accepted_qualification_input_metadata,
    )

    with _helpers["freeze"](dependency_graph) as files:
        arguments, _ = _synthetic_capabilities(monkeypatch, files)
        receipt = accept_qualification_inputs(**arguments)
        metadata = accepted_qualification_input_metadata(receipt)
        assert _accepted_qualification_inputs_for_consumer(receipt).candidate is arguments[
            "candidate"
        ]
        assert metadata.qualification_input_sha256
        assert len(metadata.verified_artifacts) > 33
    with pytest.raises(ValueError, match="closed"):
        accepted_qualification_input_metadata(receipt)


@pytest.mark.parametrize(
    "fault",
    [
        "candidate",
        "build_inputs",
        "missing_runtime",
        "runtime_dependencies",
        "moonshine_runtime",
        "linux_target",
        "pending_cleanup",
        "failed_cleanup",
    ],
)
def test_complete_input_refuses_mixed_partial_stale_or_unclean_graph(
    dependency_graph, monkeypatch, fault
):
    from scripts.qualification_input_acceptance import accept_qualification_inputs

    with _helpers["freeze"](dependency_graph) as files:
        arguments, bindings = _synthetic_capabilities(monkeypatch, files)
        if fault == "candidate":
            bindings.chrome.candidate = object()
        elif fault == "build_inputs":
            bindings.build.inputs = object()
        elif fault == "missing_runtime":
            arguments["runtimes"] = arguments["runtimes"][:-1]
        elif fault == "runtime_dependencies":
            bindings.runtime_by_purpose["realtime_windows_direct_runtime"].dependencies = object()
        elif fault == "moonshine_runtime":
            next(iter(bindings.catalogs.values())).runtime = object()
        elif fault == "linux_target":
            bindings.linux_dependencies.linux_target = object()
        elif fault == "pending_cleanup":
            bindings.benchmark.output_owner = _owner(closed=False)
        elif fault == "failed_cleanup":
            bindings.linux.preparation_owner = _owner(failed=True)
        with pytest.raises(ValueError):
            accept_qualification_inputs(**arguments)
