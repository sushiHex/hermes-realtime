"""Own source-bound benchmark production and bind its two final input roles.

The benchmark process uses candidate source and the admitted Python distribution.
Its completed facts outlive the disposable execution tree. The produced artifact
snapshot must remain live until a later final-input binding copies the exact bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts import benchmark_evidence_admission as benchmark
from scripts.qualification_build_environment import _workspace
from scripts.qualification_build_inputs import (
    BoundBuildInputsV1,
    _build_input_facts,
    _build_inputs_for_consumer,
    build_input_metadata,
)
from scripts.qualification_builds import _archive_source_files
from scripts.qualification_candidate_files import (
    BoundCandidateFilesV1,
    candidate_file_metadata,
)
from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
    execution_file_metadata,
    owned_execution_files,
)
from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)

_SOURCE_FILES = (
    "scripts/__init__.py",
    "scripts/benchmark_evidence_admission.py",
    "scripts/qualification_benchmark_worker.py",
    "src/hermes_realtime/__init__.py",
    "src/hermes_realtime/evidence/__init__.py",
    "src/hermes_realtime/evidence/admission.py",
    "src/hermes_realtime/evidence/lifecycle.py",
    "src/hermes_realtime/evidence/models.py",
)
_OUTPUT_ROLES = ("benchmark_machine_manifest", "benchmark_report")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactMetadataV1:
    role: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class BenchmarkProductionMetadataV1:
    source_commit: str
    source_tree: str
    python_version: str
    script_sha256: str
    worker_sha256: str
    artifacts: tuple[BenchmarkArtifactMetadataV1, ...]
    invocation_count: int


class ProducedBenchmarkArtifactsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("benchmark artifacts are recipe-minted only")


@dataclass(frozen=True, slots=True)
class _Produced:
    inputs: BoundBuildInputsV1
    files: ImmutableExecutionFilesV1
    contents: tuple[tuple[str, bytes], ...]
    invocation: CompletedToolInvocationV1
    output_owner: OwnedQualificationWorkV1
    execution_work: OwnedQualificationWorkV1
    metadata: BenchmarkProductionMetadataV1


_PRODUCED: WeakKeyDictionary[ProducedBenchmarkArtifactsV1, _Produced] = WeakKeyDictionary()
_TRANSFERRED: WeakKeyDictionary[ProducedBenchmarkArtifactsV1, str] = WeakKeyDictionary()


def _benchmark_source_files(inputs: BoundBuildInputsV1) -> dict[str, bytes]:
    source = _archive_source_files(inputs)
    _require(
        all(name in source and 0 < len(source[name]) <= 2 * 1024**2 for name in _SOURCE_FILES),
        "candidate benchmark source closure is incomplete",
    )
    return {name: source[name] for name in _SOURCE_FILES}


def _observation(raw: bytes) -> dict[str, object]:
    _require(type(raw) is bytes and 0 < len(raw) <= 4096, "benchmark observation is unbounded")
    value = benchmark.parse_canonical_json_bytes(raw)
    _require(
        type(value) is dict
        and set(value)
        == {
            "benchmarkReportSha256",
            "benchmarkScriptSha256",
            "fileOrigins",
            "machineManifestSha256",
            "pid",
            "pythonVersion",
            "sourceFallback",
            "version",
        },
        "benchmark observation fields differ",
    )
    assert isinstance(value, dict)
    return value


def _validate_outputs(
    *,
    script: bytes,
    machine: bytes,
    report: bytes,
    observation: bytes,
    python_version: str,
    pid: int,
) -> tuple[BenchmarkArtifactMetadataV1, ...]:
    script_sha256 = hashlib.sha256(script).hexdigest()
    machine_sha256 = hashlib.sha256(machine).hexdigest()
    report_sha256 = hashlib.sha256(report).hexdigest()
    machine_value = benchmark.validate_machine_manifest(
        benchmark.parse_canonical_json_bytes(machine)
    )
    report_value = benchmark.validate_report(benchmark.parse_canonical_json_bytes(report))
    thresholds = report_value["thresholds"]
    _require(type(thresholds) is dict, "benchmark thresholds shape differs")
    assert isinstance(thresholds, dict)
    observed = _observation(observation)
    _require(
        machine_value["benchmarkScriptSha256"] == script_sha256
        and machine_value["pythonFullVersion"] == python_version
        and machine_value["pythonArchitecture"] == "64bit"
        and machine_value["windowsArchitecture"] == "AMD64",
        "benchmark machine identity differs from the admitted runtime",
    )
    _require(
        report_value["benchmarkScriptSha256"] == script_sha256
        and report_value["machineManifestSha256"] == machine_sha256
        and thresholds["required"] is True
        and report_value["passed"] is True,
        "benchmark report does not bind a passing required run",
    )
    _require(
        observed
        == {
            "benchmarkReportSha256": report_sha256,
            "benchmarkScriptSha256": script_sha256,
            "fileOrigins": observed["fileOrigins"],
            "machineManifestSha256": machine_sha256,
            "pid": pid,
            "pythonVersion": python_version,
            "sourceFallback": False,
            "version": 1,
        }
        and type(observed["fileOrigins"]) is int
        and 8 <= observed["fileOrigins"] <= 16384,
        "benchmark observation is not process and source bound",
    )
    return tuple(
        BenchmarkArtifactMetadataV1(role, hashlib.sha256(raw).hexdigest(), len(raw))
        for role, raw in zip(_OUTPUT_ROLES, (machine, report), strict=True)
    )


def produce_benchmark_artifacts(
    work: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
) -> ProducedBenchmarkArtifactsV1:
    _require(type(work) is OwnedQualificationWorkV1, "benchmark work owner type differs")
    work._accepting()
    try:
        bound = _build_inputs_for_consumer(inputs)
        input_metadata = build_input_metadata(inputs)
        source_files = _benchmark_source_files(inputs)
        script = source_files["scripts/benchmark_evidence_admission.py"]
        worker = source_files["scripts/qualification_benchmark_worker.py"]
        with OwnedQualificationWorkV1() as execution:
            source = execution.enter(owned_execution_files(source_files))
            source_root = _execution_files_for_consumer(source)
            workspace = execution.enter(_workspace())
            output = workspace / "output"
            output.mkdir()
            invocation = execution.run_tool(
                bound.tools,
                "build_python",
                (
                    str(source_root / "scripts" / "qualification_benchmark_worker.py"),
                    str(source_root / "src"),
                    str(output / "machine.json"),
                    str(output / "report.json"),
                    str(output / "observation.json"),
                ),
                workspace,
                timeout_milliseconds=600_000,
            )
            _execution_files_for_consumer(source)
            _require(
                sorted(path.name for path in output.iterdir())
                == ["machine.json", "observation.json", "report.json"],
                "benchmark output namespace differs",
            )
            seals = execution.enter(
                retain_file_seals(output, ("machine.json", "observation.json", "report.json"))
            )
            machine = sealed_file_bytes(seals, "machine.json", 4 * 1024**2)
            report = sealed_file_bytes(seals, "report.json", 16 * 1024**2)
            observed = sealed_file_bytes(seals, "observation.json", 4096)
            artifacts = _validate_outputs(
                script=script,
                machine=machine,
                report=report,
                observation=observed,
                python_version=input_metadata.python_version,
                pid=_tool_invocation_for_consumer(invocation).process.pid,
            )
            contents = dict(zip(_OUTPUT_ROLES, (machine, report), strict=True))
            retained = work.enter(owned_execution_files(contents))
        _require(
            execution._closed and execution._unrecoverable is None,
            "benchmark execution cleanup is incomplete",
        )
        _require(
            tool_invocation_metadata(invocation).role == "build_python",
            "benchmark tool differs",
        )
        receipt = object.__new__(ProducedBenchmarkArtifactsV1)
        _PRODUCED[receipt] = _Produced(
            inputs,
            retained,
            tuple(contents.items()),
            invocation,
            work,
            execution,
            BenchmarkProductionMetadataV1(
                input_metadata.source_commit,
                input_metadata.source_tree,
                input_metadata.python_version,
                hashlib.sha256(script).hexdigest(),
                hashlib.sha256(worker).hexdigest(),
                artifacts,
                1,
            ),
        )
        return receipt
    except BaseException as error:
        work._retain_failure(error)
        raise


def _completed(receipt: ProducedBenchmarkArtifactsV1) -> _Produced:
    if type(receipt) is not ProducedBenchmarkArtifactsV1:
        raise TypeError("benchmark artifact capability type differs")
    _require(receipt in _PRODUCED, "benchmark artifact capability is unregistered")
    value = _PRODUCED[receipt]
    inputs = _build_input_facts(value.inputs).metadata
    _require(
        (inputs.source_commit, inputs.source_tree, inputs.python_version)
        == (
            value.metadata.source_commit,
            value.metadata.source_tree,
            value.metadata.python_version,
        ),
        "benchmark source or Python facts differ",
    )
    _require(
        value.execution_work._closed
        and value.execution_work._unrecoverable is None
        and tool_invocation_metadata(value.invocation).role == "build_python",
        "benchmark invocation cleanup is incomplete",
    )
    for item, (role, raw) in zip(value.metadata.artifacts, value.contents, strict=True):
        _require(
            item.role == role
            and item.sha256 == hashlib.sha256(raw).hexdigest()
            and item.size == len(raw),
            "benchmark retained output facts differ",
        )
    return value


def benchmark_artifact_metadata(
    receipt: ProducedBenchmarkArtifactsV1,
) -> BenchmarkProductionMetadataV1:
    return _completed(receipt).metadata


def _benchmark_output_bytes(receipt: ProducedBenchmarkArtifactsV1) -> dict[str, bytes]:
    value = _completed(receipt)
    _require(
        execution_file_metadata(value.files)
        == tuple((item.role, item.sha256, item.size) for item in value.metadata.artifacts),
        "benchmark output seals differ",
    )
    return dict(value.contents)


@dataclass(frozen=True, slots=True)
class BenchmarkFileMetadataV1:
    qualification_input_sha256: str
    benchmark: BenchmarkProductionMetadataV1


class BoundBenchmarkFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("benchmark file bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _BoundFiles:
    produced: ProducedBenchmarkArtifactsV1
    files: RetainedQualificationInputFilesV1
    candidate: BoundCandidateFilesV1
    metadata: BenchmarkFileMetadataV1


_BOUND: WeakKeyDictionary[BoundBenchmarkFilesV1, _BoundFiles] = WeakKeyDictionary()


def bind_benchmark_output_files(
    produced: ProducedBenchmarkArtifactsV1,
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
) -> BoundBenchmarkFilesV1:
    if type(produced) is not ProducedBenchmarkArtifactsV1:
        raise TypeError("benchmark artifact capability type differs")
    _require(produced not in _TRANSFERRED, "benchmark artifacts were already transferred")
    contents = _benchmark_output_bytes(produced)
    production = benchmark_artifact_metadata(produced)
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    source = candidate_file_metadata(candidate)
    _require(
        source.qualification_input_sha256
        == retained_input_metadata(files).qualification_input_sha256
        and (source.source_commit, source.source_tree)
        == (production.source_commit, production.source_tree)
        and (
            document["candidate"]["candidateCommit"],
            document["candidate"]["tree"],
            document["expected"]["pythonFullVersion"],
        )
        == (production.source_commit, production.source_tree, production.python_version),
        "final benchmark candidate or Python differs",
    )
    references = {item["role"]: item for item in document["files"]}
    for role, raw in contents.items():
        _require(
            sealed_file_bytes(selected.seals, references[role]["relativePath"], 16 * 1024**2)
            == raw,
            "final benchmark role differs from its produced artifact",
        )
    receipt = object.__new__(BoundBenchmarkFilesV1)
    _BOUND[receipt] = _BoundFiles(
        produced,
        files,
        candidate,
        BenchmarkFileMetadataV1(source.qualification_input_sha256, production),
    )
    _TRANSFERRED[produced] = source.qualification_input_sha256
    return receipt


def benchmark_file_metadata(receipt: BoundBenchmarkFilesV1) -> BenchmarkFileMetadataV1:
    if type(receipt) is not BoundBenchmarkFilesV1:
        raise TypeError("benchmark file binding type differs")
    _require(receipt in _BOUND, "benchmark file binding is unregistered")
    value = _BOUND[receipt]
    _require(
        benchmark_artifact_metadata(value.produced) == value.metadata.benchmark
        and candidate_file_metadata(value.candidate).qualification_input_sha256
        == value.metadata.qualification_input_sha256
        and retained_input_metadata(value.files).qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "final benchmark input seals or facts differ",
    )
    return value.metadata


def _benchmark_files_for_consumer(receipt: BoundBenchmarkFilesV1) -> _BoundFiles:
    benchmark_file_metadata(receipt)
    return _BOUND[receipt]
