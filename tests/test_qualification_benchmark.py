"""Benchmark input authority requires an owned invocation and live output transfer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace as Row

import pytest

_helpers = run_path(str(Path(__file__).with_name("test_qualification_candidate_files.py")))
_build_helpers = run_path(str(Path(__file__).with_name("test_qualification_build_inputs.py")))
bound_graph = _helpers["bound_graph"]
source = _helpers["source"]
tools = _build_helpers["tools"]


def test_supplied_benchmark_json_cannot_mint_authority() -> None:
    from scripts.qualification_benchmark import (
        BoundBenchmarkFilesV1,
        ProducedBenchmarkArtifactsV1,
        benchmark_artifact_metadata,
        benchmark_file_metadata,
    )

    with pytest.raises(TypeError):
        ProducedBenchmarkArtifactsV1()
    with pytest.raises(TypeError):
        BoundBenchmarkFilesV1()
    with pytest.raises(TypeError):
        benchmark_artifact_metadata({"passed": True})
    with pytest.raises(TypeError):
        benchmark_file_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        benchmark_artifact_metadata(object.__new__(ProducedBenchmarkArtifactsV1))
    with pytest.raises(ValueError, match="unregistered"):
        benchmark_file_metadata(object.__new__(BoundBenchmarkFilesV1))


@pytest.mark.parametrize(
    "fault",
    [None, "script", "machine", "python", "architecture", "thresholds", "passed", "pid", "extra"],
)
def test_benchmark_outputs_bind_source_machine_report_and_actual_invocation(
    fault: str | None,
) -> None:
    from scripts import benchmark_evidence_admission as benchmark
    from scripts.qualification_benchmark import _validate_outputs

    script = Path(benchmark.__file__).read_bytes()
    script_digest = hashlib.sha256(script).hexdigest()
    machine = {
        "schemaVersion": 1,
        "cpuModel": "Synthetic Test CPU",
        "logicalCpuCount": 8,
        "installedRamBytes": 17_179_869_184,
        "windowsEdition": "Windows Test Edition",
        "windowsBuild": "26000.1",
        "windowsArchitecture": "AMD64",
        "pythonFullVersion": "3.11.16",
        "pythonArchitecture": "64bit",
        "acPower": True,
        "powerScheme": "high_performance",
        "benchmarkScriptSha256": script_digest,
    }
    machine_raw = benchmark.canonical_json_bytes(machine)
    report = benchmark.make_report_for_testing(required=True, passed=True)
    report["benchmarkScriptSha256"] = script_digest
    report["machineManifestSha256"] = hashlib.sha256(machine_raw).hexdigest()
    report_raw = benchmark.canonical_json_bytes(report)
    observation = {
        "version": 1,
        "pid": 73,
        "benchmarkScriptSha256": script_digest,
        "machineManifestSha256": hashlib.sha256(machine_raw).hexdigest(),
        "benchmarkReportSha256": hashlib.sha256(report_raw).hexdigest(),
        "pythonVersion": "3.11.16",
        "sourceFallback": False,
        "fileOrigins": 12,
    }
    if fault == "script":
        machine["benchmarkScriptSha256"] = "f" * 64
        machine_raw = benchmark.canonical_json_bytes(machine)
    elif fault == "machine":
        report["machineManifestSha256"] = "f" * 64
        report_raw = benchmark.canonical_json_bytes(report)
    elif fault == "python":
        machine["pythonFullVersion"] = "3.11.15"
        machine_raw = benchmark.canonical_json_bytes(machine)
    elif fault == "architecture":
        machine["pythonArchitecture"] = "32bit"
        machine_raw = benchmark.canonical_json_bytes(machine)
    elif fault == "thresholds":
        report["thresholds"]["required"] = False
        report["passed"] = False
        report_raw = benchmark.canonical_json_bytes(report)
    elif fault == "passed":
        report["passed"] = False
        report_raw = benchmark.canonical_json_bytes(report)
    elif fault == "pid":
        observation["pid"] = 74
    elif fault == "extra":
        observation["privatePath"] = "C:/private"
    if fault in {"script", "python", "architecture"}:
        observation["machineManifestSha256"] = hashlib.sha256(machine_raw).hexdigest()
        report["machineManifestSha256"] = observation["machineManifestSha256"]
        report_raw = benchmark.canonical_json_bytes(report)
    if fault in {"machine", "thresholds", "passed", "script", "python", "architecture"}:
        observation["benchmarkReportSha256"] = hashlib.sha256(report_raw).hexdigest()

    if fault is None:
        result = _validate_outputs(
            script=script,
            machine=machine_raw,
            report=report_raw,
            observation=benchmark.canonical_json_bytes(observation),
            python_version="3.11.16",
            pid=73,
        )
        assert tuple(item.role for item in result) == (
            "benchmark_machine_manifest",
            "benchmark_report",
        )
    else:
        with pytest.raises(ValueError):
            _validate_outputs(
                script=script,
                machine=machine_raw,
                report=report_raw,
                observation=benchmark.canonical_json_bytes(observation),
                python_version="3.11.16",
                pid=73,
            )


def test_final_binder_requires_live_produced_bytes_and_retains_only_final_seals(
    bound_graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import qualification_benchmark as owner
    from scripts.qualification_candidate_files import bind_candidate_files

    root, _, document, archive, identity = bound_graph
    raw = {
        item["role"]: (root / item["relativePath"]).read_bytes()
        for item in document["files"]
        if item["role"] in owner._OUTPUT_ROLES
    }
    produced_metadata = Row(
        source_commit=identity.candidate_head_oid,
        source_tree=identity.candidate_tree_oid,
        python_version=document["expected"]["pythonFullVersion"],
    )
    monkeypatch.setattr(owner, "_benchmark_output_bytes", lambda _: raw)
    monkeypatch.setattr(owner, "benchmark_artifact_metadata", lambda _: produced_metadata)

    with _helpers["freeze"](bound_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        produced = object.__new__(owner.ProducedBenchmarkArtifactsV1)
        receipt = owner.bind_benchmark_output_files(produced, files, candidate)
        metadata = owner.benchmark_file_metadata(receipt)
        with pytest.raises(ValueError, match="already transferred"):
            owner.bind_benchmark_output_files(produced, files, candidate)
        monkeypatch.setattr(
            owner,
            "_benchmark_output_bytes",
            lambda _: pytest.fail("final acceptance reopened disposable benchmark output"),
        )
        assert owner.benchmark_file_metadata(receipt) == metadata
    with pytest.raises(ValueError, match="closed"):
        owner.benchmark_file_metadata(receipt)


def test_producer_refuses_unbound_inputs_before_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import qualification_benchmark as owner
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    monkeypatch.setattr(
        owner,
        "_benchmark_source_files",
        lambda _: pytest.fail("unbound input reached source materialization"),
    )
    with OwnedQualificationWorkV1() as work:
        with pytest.raises((TypeError, ValueError)):
            owner.produce_benchmark_artifacts(work, {"passed": True})
        with pytest.raises(RuntimeError, match="closing"):
            work._accepting()


@pytest.mark.skipif(os.name != "nt", reason="genuine Windows source authority")
def test_benchmark_source_is_selected_from_the_candidate_archive(
    tmp_path_factory,
    tools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as archives
    from scripts import qualification_benchmark as owner
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
    from scripts.qualification_build_inputs import bind_build_inputs
    from scripts.qualification_tool_environment import tool_environment_metadata

    repository, _, original, _ = _helpers["_make_source"](tmp_path_factory)
    for relative in owner._SOURCE_FILES:
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path.cwd() / relative).read_bytes())
    oracle = run_path(str(Path(__file__).with_name("test_candidate_source_archive_oracle.py")))
    oracle["_git"]("add", ".", cwd=repository)
    oracle["_git"]("commit", "-qm", "add benchmark source closure", cwd=repository)
    identity = oracle["_identity"](repository, original.canonical_baseline_oid)
    archive = capture_candidate_source_archive(repository, identity, oracle["_pin"]())
    captured = tool_environment_metadata(tools)
    monkeypatch.setattr(
        archives,
        "_archive_tool_capture_for_consumer",
        lambda archive, identity: captured,
    )
    wheel_helpers, wheels = _build_helpers["values"]()
    inputs = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels=wheels,
        requirements=wheel_helpers["pins"](wheels),
        constraints=b"",
    )
    expected = (repository / "scripts/benchmark_evidence_admission.py").read_bytes()
    (repository / "scripts/benchmark_evidence_admission.py").write_bytes(b"changed checkout\n")

    selected = owner._benchmark_source_files(inputs)

    assert tuple(selected) == owner._SOURCE_FILES
    assert selected["scripts/benchmark_evidence_admission.py"] == expected
