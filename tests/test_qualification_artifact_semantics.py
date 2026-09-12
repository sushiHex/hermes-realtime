"""Controlled in-memory fixtures for pure Task 12 artifact semantic validators.

These temporary fixtures deliberately are not qualification or release evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]


def _load_test_helpers(stem: str) -> ModuleType:
    """Load sibling fixture helpers without making tests a production package."""
    path = ROOT / "tests" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


_SCHEMA_HELPERS = _load_test_helpers("test_evidence_artifact_schemas")
_INPUT_HELPERS = _load_test_helpers("test_qualify_evidence_slice_zero")
_governed_qualification_report = _SCHEMA_HELPERS._governed_qualification_report
_make_input_root = _INPUT_HELPERS._make_input_root
_ref = _INPUT_HELPERS._ref
_write_json = _INPUT_HELPERS._write_json
REPORT_SCHEMA = ROOT / "scripts" / "schemas" / "qualification-report-v1.schema.json"
RELEASE_SCHEMA = ROOT / "scripts" / "schemas" / "release-manifest-v1.schema.json"
WHEELHOUSE_ROLES = (
    "build_wheelhouse_manifest",
    "hermes_runtime_wheelhouse_manifest",
    "linux_runtime_wheelhouse_manifest",
    "windows_direct_runtime_wheelhouse_manifest",
    "windows_sdist_built_runtime_wheelhouse_manifest",
)
GATES = (
    "windowsTests",
    "ruff",
    "mypy",
    "compileall",
    "webTestsBuild",
    "generatedStaticClean",
    "releaseGate",
    "deterministicEquivalence",
    "benchmark",
    "reproducibleBuild",
    "artifactParity",
    "cleanInstallWindows",
    "cleanInstallLinux",
    "linuxNullCapture",
    "hermesPluginManager",
    "physicalQualification",
    "uninstallRetention",
)


def _canonical(document: object) -> bytes:
    return (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file(document: dict[str, Any], role: str) -> dict[str, Any]:
    return next(item for item in document["files"] if item["role"] == role)


def _governed_jobs(report: dict[str, Any], input_document: dict[str, Any]) -> list[dict[str, Any]]:
    build_python = next(
        item["artifact"]
        for item in input_document["toolIdentities"]
        if item["artifact"]["role"] == "build_python_executable"
    )
    jobs: list[dict[str, Any]] = []
    for ordinal, scenario in enumerate(report["scenarios"][1:], start=1):
        pid = 1000 + ordinal
        jobs.append(
            {
                "scenarioId": scenario["scenarioId"],
                "limitFlags": {"breakawayAllowed": False, "killOnJobClose": True},
                "activeProcessCountAfterCleanup": 0,
                "processes": [
                    {
                        "scenarioId": scenario["scenarioId"],
                        "pid": pid,
                        "parentPid": report["topology"]["runnerPid"],
                        "parentCreationFiletime": report["topology"]["runnerCreationFiletime"],
                        "creationFiletime": 10_000 + ordinal,
                        "processGroupId": pid,
                        "imageBasename": build_python["basename"],
                        "imageSha256": build_python["sha256"],
                        "role": "host_root",
                    }
                ],
            }
        )
    return jobs


def _context(tmp_path: Path) -> dict[str, Any]:
    """Make schema-valid ephemeral data only; it never represents release evidence."""
    from scripts import qualify_evidence_slice_zero as runner

    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    input_document = json.loads(manifest.read_text(encoding="utf-8"))
    for role, source in (
        ("qualification_report_schema", REPORT_SCHEMA),
        ("release_manifest_schema", RELEASE_SCHEMA),
    ):
        destination = root / "schemas" / source.name
        destination.write_bytes(source.read_bytes())
        replacement = _ref(root, destination, role)
        input_document["files"][input_document["files"].index(_file(input_document, role))] = (
            replacement
        )
        expected_key = {
            "qualification_report_schema": "qualificationReportSchemaSha256",
            "release_manifest_schema": "releaseManifestSchemaSha256",
        }[role]
        input_document["expected"][expected_key] = replacement["sha256"]
    # Align synthetic input pins with the reviewed report schema's representable values.
    # These in-memory test values are not observations or release evidence.
    input_document["expected"].update(
        codexVersion="1.0.0",
        codexModel="gpt-5.6-terra",
        codexEffort="low",
        pythonArchitecture="64bit",
        livekitVersion="1.13.4",
        moonshineVersion="0.1.0",
        kokoroVersion="1.0.0",
    )
    _write_json(manifest, input_document)
    input_bytes = manifest.read_bytes()
    closure = runner.verify_qualification_input_closure(
        qualification_input_root=root,
        qualification_input_manifest=manifest,
        expected_qualification_input_sha256=_sha(input_bytes),
        plan=plan,
        candidate_source_archive=archive,
        runner_path=archived_runner,
    )
    report = _governed_qualification_report()
    report["qualificationInputSha256"] = _sha(input_bytes)
    report["governingPlanSha256"] = _file(input_document, "governing_plan")["sha256"]
    report["benchmarkReportSha256"] = _file(input_document, "benchmark_report")["sha256"]
    report["candidate"] = deepcopy(input_document["candidate"])
    report["verifiedArtifacts"] = [
        {"logicalId": item.logical_id, "sha256": item.sha256, "bytes": item.bytes}
        for item in closure.verified_artifacts
    ]
    expected = input_document["expected"]
    report["environment"].update(
        pythonFullVersion=expected["pythonFullVersion"],
        chromeVersion=expected["chromeVersion"],
        chromeVersionDirectoryManifestSha256=_file(
            input_document, "chrome_version_directory_manifest"
        )["sha256"],
    )
    report["topology"].update(
        livekitVersion=expected["livekitVersion"],
        rtcSocketInventory=[{"protocol": "tcp", "port": 7880, "addressClass": "loopback"}],
        jobs=_governed_jobs(report, input_document),
    )
    report["providers"].update(
        codexExecutableVersion=expected["codexVersion"],
        codexExecutableSha256=_file(input_document, "codex_executable")["sha256"],
        moonshineDistributionVersion=expected["moonshineVersion"],
        moonshineDistributionSha256=_file(input_document, "moonshine_distribution")["sha256"],
        moonshineModelIdentitySha256=json.loads(
            (root / _file(input_document, "moonshine_model_manifest")["relativePath"]).read_text()
        )["modelIdentitySha256"],
        kokoroDistributionVersion=expected["kokoroVersion"],
        kokoroDistributionSha256=_file(input_document, "kokoro_distribution")["sha256"],
        kokoroModelIdentitySha256=json.loads(
            (root / _file(input_document, "kokoro_model_manifest")["relativePath"]).read_text()
        )["modelIdentitySha256"],
    )
    attestation_id = "01234567-89ab-cdef-0123-456789abcdef"
    report["operatorAttestations"] = [
        {
            "attestationId": attestation_id,
            "code": "disclosure_visible_controls_accessible",
            "confirmed": True,
            "confirmedAtUtc": "2026-08-19T00:00:00Z",
        }
    ]
    report["scenarios"][1]["attestationIds"] = [attestation_id]
    report_bytes = _canonical(report)
    release = {
        "schemaVersion": 1,
        "candidate": deepcopy(input_document["candidate"]),
        "sourceArchive": deepcopy(_file(input_document, "candidate_source_archive")),
        "artifacts": deepcopy(report["verifiedArtifacts"]),
        "wheelhouseManifests": [deepcopy(_file(input_document, role)) for role in WHEELHOUSE_ROLES],
        "qualificationInputSha256": _sha(input_bytes),
        "qualificationReportSha256": _sha(report_bytes),
        "benchmarkReportSha256": _file(input_document, "benchmark_report")["sha256"],
        "gateResults": {
            gate: {"outcome": "pass", "commandSha256": "a" * 64, "outputSha256": "b" * 64}
            for gate in GATES
        },
        "environment": {
            "windowsEdition": report["environment"]["windowsEdition"],
            "windowsBuild": report["environment"]["windowsBuild"],
            "windowsArchitecture": report["environment"]["windowsArchitecture"],
            "pythonFullVersion": expected["pythonFullVersion"],
            "pythonArchitecture": report["environment"]["pythonArchitecture"],
            "sqliteVersion": report["environment"]["sqliteVersion"],
            "toolIdentities": deepcopy(input_document["toolIdentities"]),
            "linuxWorkflow": {
                key: "c" * 64
                for key in (
                    "runIdSha256",
                    "artifactNameSha256",
                    "artifactSha256",
                    "commandOutputSha256",
                )
            },
        },
        "passed": True,
    }
    return {
        "runner": runner,
        "root": root,
        "manifest": manifest,
        "plan": plan,
        "archive": archive,
        "archived_runner": archived_runner,
        "input_bytes": input_bytes,
        "report": report,
        "report_bytes": report_bytes,
        "release": release,
    }


def _report_validate(context: dict[str, Any], report_bytes: bytes | None = None) -> dict[str, Any]:
    return context["runner"].validate_qualification_report(
        qualification_input_root=context["root"],
        qualification_input_manifest=context["manifest"],
        expected_qualification_input_sha256=_sha(context["input_bytes"]),
        plan=context["plan"],
        candidate_source_archive=context["archive"],
        runner_path=context["archived_runner"],
        report_bytes=context["report_bytes"] if report_bytes is None else report_bytes,
    )


def _release_validate(
    context: dict[str, Any],
    release: dict[str, Any] | None = None,
    report_bytes: bytes | None = None,
) -> dict[str, Any]:
    return context["runner"].validate_release_manifest(
        qualification_input_root=context["root"],
        qualification_input_manifest=context["manifest"],
        expected_qualification_input_sha256=_sha(context["input_bytes"]),
        plan=context["plan"],
        candidate_source_archive=context["archive"],
        runner_path=context["archived_runner"],
        qualification_report_bytes=context["report_bytes"]
        if report_bytes is None
        else report_bytes,
        release_manifest_bytes=_canonical(context["release"] if release is None else release),
    )


def test_controlled_fixture_is_schema_valid_and_semantically_closed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    report = _report_validate(context)
    release = _release_validate(context)
    assert report["passed"] is True
    assert release["passed"] is True


def test_ordinary_chrome_publisher_name_survives_report_and_release_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _make_input_root

    def with_publisher_resource(parent: Path):
        result = original(parent)
        root, manifest, _, _, _ = result
        document = json.loads(manifest.read_bytes())
        reference = _file(document, "chrome_version_directory_manifest")
        chrome_manifest = root / reference["relativePath"]
        chrome = json.loads(chrome_manifest.read_bytes())
        raw = b"synthetic publisher first-run metadata"
        (chrome_manifest.parent / "First Run").write_bytes(raw)
        chrome["files"].append({"name": "First Run", "bytes": len(raw), "sha256": _sha(raw)})
        chrome["files"].sort(key=lambda row: row["name"])
        _write_json(chrome_manifest, chrome)
        reference.update(_ref(root, chrome_manifest, "chrome_version_directory_manifest"))
        _write_json(manifest, document)
        return result

    monkeypatch.setitem(_context.__globals__, "_make_input_root", with_publisher_resource)
    context = _context(tmp_path)
    report = _report_validate(context)
    release = _release_validate(context)
    assert report["verifiedArtifacts"] == release["artifacts"]
    chrome_ids = [
        row["logicalId"]
        for row in release["artifacts"]
        if row["logicalId"].startswith("chrome:file:")
    ]
    assert "chrome:file:" + _sha(b"First Run") in chrome_ids
    assert all("First Run" not in value for value in chrome_ids)


@pytest.mark.parametrize(
    "field", ("qualificationInputSha256", "governingPlanSha256", "benchmarkReportSha256")
)
def test_report_rejects_named_direct_hash_equations(tmp_path: Path, field: str) -> None:
    context = _context(tmp_path)
    report = deepcopy(context["report"])
    report[field] = "0" * 64
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(report))


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate",
        "artifacts",
        "environment",
        "python_architecture",
        "topology",
        "topology_loopback",
        "topology_inventory",
        "providers",
        "scenario",
        "case",
        "cleanup",
        "passed",
        "attestation_order",
        "attestation_reference",
        "timestamp",
    ),
)
def test_report_semantic_equations_fail_closed(tmp_path: Path, mutation: str) -> None:
    context = _context(tmp_path)
    report = deepcopy(context["report"])
    if mutation == "candidate":
        report["candidate"]["tree"] = "0" * 40
    elif mutation == "artifacts":
        report["verifiedArtifacts"][0]["bytes"] += 1
    elif mutation == "environment":
        report["environment"]["chromeVersion"] = "0.0"
    elif mutation == "python_architecture":
        report["environment"]["pythonArchitecture"] = (
            "32bit" if report["environment"]["pythonArchitecture"] == "64bit" else "64bit"
        )
    elif mutation == "topology":
        report["topology"]["livekitVersion"] = "0.0"
    elif mutation == "topology_loopback":
        report["topology"]["signalingLoopbackOnly"] = False
    elif mutation == "topology_inventory":
        report["topology"]["rtcSocketInventoryRecorded"] = False
    elif mutation == "providers":
        report["providers"]["codexModel"] = "other"
    elif mutation == "scenario":
        report["scenarios"][0]["outcome"] = "fail"
    elif mutation == "case":
        report["scenarios"][15]["caseResults"][0]["outcome"] = "fail"
    elif mutation == "cleanup":
        report["cleanup"]["jobsZeroActive"] = False
    elif mutation == "passed":
        report["passed"] = False
    elif mutation == "attestation_order":
        duplicate = deepcopy(report["operatorAttestations"][0])
        duplicate["attestationId"] = "11234567-89ab-cdef-0123-456789abcdef"
        report["operatorAttestations"] = [duplicate, *report["operatorAttestations"]]
    elif mutation == "attestation_reference":
        report["scenarios"][1]["attestationIds"] = ["11234567-89ab-cdef-0123-456789abcdef"]
    else:
        report["finishedAtUtc"] = report["startedAtUtc"]
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(report))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("maxQueueRecords", 63),
        ("maxQueueCanonicalBytes", 2_097_151),
        ("maxQueuePhysicalItems", 63),
        ("maxCanonicalRecordBytes", 32_767),
        ("queueRecordCount", 63),
        ("queueCanonicalBytes", 2_097_152),
        ("queuePhysicalCount", 65),
    ),
)
def test_report_rejects_false_coupled_capacity_measurements(
    tmp_path: Path, field: str, replacement: int
) -> None:
    context = _context(tmp_path)
    report = deepcopy(context["report"])
    scenario = next(
        item for item in report["scenarios"] if item["scenarioId"] == "synthetic_fault_matrix"
    )
    scenario["measurements"][field] = replacement
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(report))


@pytest.mark.parametrize(
    "mutation",
    (
        "job_order",
        "duplicate_job",
        "empty_job",
        "process_scenario",
        "unresolved_parent",
        "parent_creation",
        "duplicate_identity",
        "pid_reuse",
        "other_role",
        "root_parent",
        "process_group",
        "process_order",
        "chrome_identity",
        "livekit_identity",
        "codex_identity",
        "rtc_non_loopback",
        "rtc_duplicate",
        "rtc_order",
    ),
)
def test_report_rejects_unclosed_process_and_socket_topology(tmp_path: Path, mutation: str) -> None:
    context = _context(tmp_path)
    report = deepcopy(context["report"])
    jobs = report["topology"]["jobs"]
    process = jobs[0]["processes"][0]
    if mutation == "job_order":
        jobs.reverse()
    elif mutation == "duplicate_job":
        jobs.append(deepcopy(jobs[-1]))
    elif mutation == "empty_job":
        jobs[0]["processes"] = []
    elif mutation == "process_scenario":
        process["scenarioId"] = "deterministic_equivalence"
    elif mutation == "unresolved_parent":
        process["parentPid"] = 999_999
        process["parentCreationFiletime"] = 999_999
    elif mutation == "parent_creation":
        process["parentCreationFiletime"] += 1
    elif mutation == "duplicate_identity":
        jobs[0]["processes"].append(deepcopy(process))
    elif mutation == "pid_reuse":
        reused = deepcopy(process)
        reused["creationFiletime"] += 1
        jobs[0]["processes"].append(reused)
    elif mutation == "other_role":
        process["role"] = "other_owned_descendant"
    elif mutation == "root_parent":
        process["parentPid"] = process["pid"]
        process["parentCreationFiletime"] = process["creationFiletime"]
    elif mutation == "process_group":
        process["processGroupId"] = 999_999
    elif mutation == "process_order":
        child = deepcopy(process)
        child["pid"] += 100
        child["creationFiletime"] += 100
        child["parentPid"] = process["pid"]
        child["parentCreationFiletime"] = process["creationFiletime"]
        child["role"] = "host_runtime"
        jobs[0]["processes"] = [child, process]
    elif mutation in {"chrome_identity", "livekit_identity", "codex_identity"}:
        input_document = json.loads(context["input_bytes"])
        role, artifact_role = {
            "chrome_identity": ("chrome_owned_descendant", "chrome_executable"),
            "livekit_identity": ("livekit_owned_descendant", "livekit_executable"),
            "codex_identity": ("codex_owned_descendant", "codex_executable"),
        }[mutation]
        reference = _file(input_document, artifact_role)
        process["role"] = role
        process["imageBasename"] = reference["basename"]
        process["imageSha256"] = "0" * 64
    elif mutation == "rtc_non_loopback":
        report["topology"]["rtcSocketInventory"][0]["addressClass"] = "wildcard"
    elif mutation == "rtc_duplicate":
        report["topology"]["rtcSocketInventory"].append(
            deepcopy(report["topology"]["rtcSocketInventory"][0])
        )
    else:
        report["topology"]["rtcSocketInventory"] = [
            {"protocol": "udp", "port": 7881, "addressClass": "loopback"},
            {"protocol": "tcp", "port": 7880, "addressClass": "loopback"},
        ]
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(report))


@pytest.mark.parametrize("raw", (b'{"b":1,"a":2}\n', b'{"a":1,"a":2}\n', b'{"a":1}\r\n', b"{}\n"))
def test_report_rejects_canonical_or_schema_invalid_bytes(tmp_path: Path, raw: bytes) -> None:
    context = _context(tmp_path)
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, raw)


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate",
        "report_hash",
        "source",
        "wheelhouse",
        "artifacts",
        "tools",
        "gate",
        "passed",
        "environment",
    ),
)
def test_release_semantic_equations_fail_closed(tmp_path: Path, mutation: str) -> None:
    context = _context(tmp_path)
    release = deepcopy(context["release"])
    if mutation == "candidate":
        release["candidate"]["tree"] = "0" * 40
    elif mutation == "report_hash":
        release["qualificationReportSha256"] = "0" * 64
    elif mutation == "source":
        release["sourceArchive"]["bytes"] += 1
    elif mutation == "wheelhouse":
        release["wheelhouseManifests"] = list(reversed(release["wheelhouseManifests"]))
    elif mutation == "artifacts":
        release["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "tools":
        release["environment"]["toolIdentities"][0]["version"] = "other"
    elif mutation == "gate":
        release["gateResults"]["ruff"]["outcome"] = "fail"
    elif mutation == "passed":
        release["passed"] = False
    else:
        release["environment"]["pythonArchitecture"] = (
            "32bit" if release["environment"]["pythonArchitecture"] == "64bit" else "64bit"
        )
    with pytest.raises(context["runner"].QualificationInputError):
        _release_validate(context, release)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("windowsEdition", "Windows Server 2022"),
        ("windowsBuild", "19045.1"),
        ("windowsArchitecture", "ARM64"),
        ("pythonFullVersion", "3.11.14"),
        ("pythonArchitecture", "32bit"),
        ("sqliteVersion", "3.99.0"),
    ),
)
def test_release_environment_is_cross_bound_to_report(
    tmp_path: Path, field: str, replacement: str
) -> None:
    context = _context(tmp_path)
    release = deepcopy(context["release"])
    if release["environment"][field] == replacement:
        replacement = {
            "windowsEdition": "Windows 10 Pro",
            "windowsBuild": "22631.1",
            "windowsArchitecture": "x86",
            "pythonFullVersion": "3.11.13",
            "pythonArchitecture": "64bit",
            "sqliteVersion": "3.98.0",
        }[field]
    release["environment"][field] = replacement
    with pytest.raises(context["runner"].QualificationInputError):
        _release_validate(context, release)


@pytest.mark.parametrize("validator", ("report", "release"))
def test_validators_repeat_full_input_revalidation_immediately_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validator: str
) -> None:
    context = _context(tmp_path)
    runner = context["runner"]
    original = runner._validate_schema_document
    target_label = "qualification report" if validator == "report" else "release manifest"

    def mutate_after_schema_validation(document: object, *, schema_path: Path, label: str) -> None:
        original(document, schema_path=schema_path, label=label)
        if label == target_label:
            context["plan"].write_bytes(context["plan"].read_bytes() + b"changed")

    monkeypatch.setattr(runner, "_validate_schema_document", mutate_after_schema_validation)
    with pytest.raises(runner.QualificationInputError):
        if validator == "report":
            _report_validate(context)
        else:
            _release_validate(context)


@pytest.mark.parametrize(
    "target", ("report_schema", "release_schema", "candidate_archive", "input_path")
)
def test_validators_revalidate_schema_and_input_paths_before_return(
    tmp_path: Path, target: str
) -> None:
    context = _context(tmp_path)
    if target == "report_schema":
        (
            context["root"]
            / _file(json.loads(context["input_bytes"]), "qualification_report_schema")[
                "relativePath"
            ]
        ).write_bytes(b"changed")
    elif target == "release_schema":
        (
            context["root"]
            / _file(json.loads(context["input_bytes"]), "release_manifest_schema")["relativePath"]
        ).write_bytes(b"changed")
    elif target == "candidate_archive":
        context["archive"].write_bytes(b"changed")
    else:
        item = _file(json.loads(context["input_bytes"]), "governing_plan")
        (context["root"] / item["relativePath"]).rename(context["root"] / "files" / "moved.bin")
    with pytest.raises(context["runner"].QualificationInputError):
        _release_validate(context)


def _controlled_descendant(report: dict[str, Any], role: str, basename: str, digest: str) -> None:
    root = report["topology"]["jobs"][0]["processes"][0]
    child = deepcopy(root)
    child.update(
        pid=root["pid"] + 50_000,
        creationFiletime=root["creationFiletime"] + 1,
        parentPid=root["pid"],
        parentCreationFiletime=root["creationFiletime"],
        processGroupId=root["pid"],
        role=role,
        imageBasename=basename,
        imageSha256=digest,
    )
    report["topology"]["jobs"][0]["processes"].append(child)


@pytest.mark.parametrize("role", ("host_runtime", "hermes_pluginmanager_owned_descendant"))
def test_report_accepts_python_bound_controlled_descendant_roles(tmp_path: Path, role: str) -> None:
    context = _context(tmp_path)
    report = deepcopy(context["report"])
    document = json.loads(context["input_bytes"])
    python = next(
        item["artifact"]
        for item in document["toolIdentities"]
        if item["artifact"]["role"] == "build_python_executable"
    )
    _controlled_descendant(report, role, python["basename"], python["sha256"])
    assert _report_validate(context, _canonical(report))["passed"] is True


@pytest.mark.parametrize(
    ("role", "field"),
    (
        ("host_root", "imageBasename"),
        ("host_runtime", "imageSha256"),
        ("hermes_pluginmanager_owned_descendant", "imageBasename"),
    ),
)
def test_report_rejects_arbitrary_python_host_or_pluginmanager_binding(
    tmp_path: Path, role: str, field: str
) -> None:
    context = _context(tmp_path)
    report = deepcopy(context["report"])
    root = report["topology"]["jobs"][0]["processes"][0]
    if role != "host_root":
        _controlled_descendant(report, role, root["imageBasename"], root["imageSha256"])
    target = report["topology"]["jobs"][0]["processes"][-1]
    target[field] = "arbitrary.exe" if field == "imageBasename" else "f" * 64
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(report))


def test_report_rejects_host_root_as_descendant_and_runtime_as_root(tmp_path: Path) -> None:
    context = _context(tmp_path)
    descendant = deepcopy(context["report"])
    root = descendant["topology"]["jobs"][0]["processes"][0]
    _controlled_descendant(descendant, "host_root", root["imageBasename"], root["imageSha256"])
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(descendant))
    root_misuse = deepcopy(context["report"])
    root_misuse["topology"]["jobs"][0]["processes"][0]["role"] = "host_runtime"
    with pytest.raises(context["runner"].QualificationInputError):
        _report_validate(context, _canonical(root_misuse))
