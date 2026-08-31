"""Focused contract tests for the release-manifest-v1 JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "scripts" / "schemas" / "release-manifest-v1.schema.json"
DIGEST = "a" * 64

GATE_NAMES = [
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
]

WHEELHOUSE_ROLES = [
    "build_wheelhouse_manifest",
    "hermes_runtime_wheelhouse_manifest",
    "linux_runtime_wheelhouse_manifest",
    "windows_direct_runtime_wheelhouse_manifest",
    "windows_sdist_built_runtime_wheelhouse_manifest",
]

TOOL_ROLES = ["build_python", "git", "hatchling", "uv"]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate object keys are noncanonical: {keys}"
    return dict(pairs)


def _load_schema() -> dict[str, Any]:
    raw = SCHEMA_PATH.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b"\r" not in raw
    schema: dict[str, Any] = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    canonical = json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    assert raw == canonical + b"\n"
    return schema


def _artifact(role: str, *, name: str | None = None) -> dict[str, object]:
    basename = name or f"{role}.bin"
    return {
        "role": role,
        "relativePath": f"artifacts/{basename}",
        "basename": basename,
        "sha256": DIGEST,
        "bytes": 1,
    }


def _tool_identity(role: str) -> dict[str, object]:
    artifact_role = {
        "build_python": "build_python_executable",
        "git": "git_executable",
        "hatchling": "hatchling_wheel",
        "uv": "uv_executable",
    }[role]
    return {
        "role": role,
        "version": "1.0.0",
        "artifact": _artifact(artifact_role),
    }


def _valid_manifest() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "candidate": {
            "baselineCommit": "b" * 40,
            "candidateCommit": "c" * 40,
            "tree": "d" * 40,
            "canonicalDiffSha256": DIGEST,
            "version": "0.0.3",
        },
        "sourceArchive": _artifact("candidate_source_archive", name="candidate.tar.gz"),
        "artifacts": [
            {"logicalId": "file:candidate_source_archive", "sha256": DIGEST, "bytes": 1},
            {"logicalId": "tool:git", "sha256": DIGEST, "bytes": 1},
            {
                "logicalId": "wheelhouse:build:wheel:example.whl",
                "sha256": DIGEST,
                "bytes": 1,
            },
            {
                "logicalId": "provider:moonshine:resource:model/file.bin",
                "sha256": DIGEST,
                "bytes": 1,
            },
            {"logicalId": "chrome:file:chrome.exe", "sha256": DIGEST, "bytes": 1},
        ],
        "wheelhouseManifests": [_artifact(role) for role in WHEELHOUSE_ROLES],
        "qualificationInputSha256": DIGEST,
        "qualificationReportSha256": DIGEST,
        "benchmarkReportSha256": DIGEST,
        "gateResults": {
            gate: {"outcome": "pass", "commandSha256": DIGEST, "outputSha256": DIGEST}
            for gate in GATE_NAMES
        },
        "environment": {
            "windowsEdition": "Windows 11 Pro",
            "windowsBuild": "26100.1",
            "windowsArchitecture": "AMD64",
            "pythonFullVersion": "3.11.15 (main)",
            "pythonArchitecture": "64bit",
            "sqliteVersion": "3.46.1",
            "toolIdentities": [_tool_identity(role) for role in TOOL_ROLES],
            "linuxWorkflow": {
                "runIdSha256": DIGEST,
                "artifactNameSha256": DIGEST,
                "artifactSha256": DIGEST,
                "commandOutputSha256": DIGEST,
            },
        },
        "passed": True,
    }


def _assert_closed_object(definition: dict[str, Any]) -> None:
    assert definition["type"] == "object"
    assert definition["additionalProperties"] is False
    assert set(definition["properties"]) == set(definition["required"])


def test_release_manifest_schema_freezes_complete_closed_contract() -> None:
    schema = _load_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://hermes-realtime.invalid/schemas/release-manifest-v1.schema.json"
    )
    _assert_closed_object(schema)
    assert schema["required"] == [
        "schemaVersion",
        "candidate",
        "sourceArchive",
        "artifacts",
        "wheelhouseManifests",
        "qualificationInputSha256",
        "qualificationReportSha256",
        "benchmarkReportSha256",
        "gateResults",
        "environment",
        "passed",
    ]
    assert schema["properties"]["schemaVersion"] == {"const": 1, "type": "integer"}
    assert schema["properties"]["passed"] == {"type": "boolean"}

    definitions = schema["$defs"]
    for definition in definitions.values():
        if definition.get("type") == "object":
            _assert_closed_object(definition)

    candidate = definitions["CandidateV1"]
    assert candidate["required"] == [
        "baselineCommit",
        "candidateCommit",
        "tree",
        "canonicalDiffSha256",
        "version",
    ]
    assert candidate["properties"]["version"] == {
        "const": "0.0.3",
        "type": "string",
    }

    source_archive = definitions["SourceArchiveRefV1"]
    assert source_archive["properties"]["role"] == {
        "const": "candidate_source_archive",
        "type": "string",
    }

    wheelhouses = schema["properties"]["wheelhouseManifests"]
    assert wheelhouses["type"] == "array"
    assert wheelhouses["minItems"] == wheelhouses["maxItems"] == 5
    assert wheelhouses["items"] is False
    prefix_names = [
        item["$ref"].removeprefix("#/$defs/").removesuffix("RefV1")
        for item in wheelhouses["prefixItems"]
    ]
    assert prefix_names == [
        "BuildWheelhouseManifest",
        "HermesRuntimeWheelhouseManifest",
        "LinuxRuntimeWheelhouseManifest",
        "WindowsDirectRuntimeWheelhouseManifest",
        "WindowsSdistBuiltRuntimeWheelhouseManifest",
    ]

    gate_results = definitions["GateResultsV1"]
    assert gate_results["required"] == GATE_NAMES
    assert set(gate_results["properties"]) == set(GATE_NAMES)
    assert gate_results["properties"]["windowsTests"] == {"$ref": "#/$defs/GateResultV1"}
    assert definitions["GateResultV1"]["properties"]["outcome"] == {
        "enum": ["pass", "fail"],
        "type": "string",
    }

    environment = definitions["ReleaseEnvironmentV1"]
    assert environment["required"] == [
        "windowsEdition",
        "windowsBuild",
        "windowsArchitecture",
        "pythonFullVersion",
        "pythonArchitecture",
        "sqliteVersion",
        "toolIdentities",
        "linuxWorkflow",
    ]
    assert definitions["LinuxWorkflowV1"]["required"] == [
        "runIdSha256",
        "artifactNameSha256",
        "artifactSha256",
        "commandOutputSha256",
    ]


def test_release_manifest_schema_validates_closed_contextual_references() -> None:
    schema = _load_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = _valid_manifest()
    assert list(validator.iter_errors(valid)) == []

    invalid_cases = [
        {**valid, "unexpected": None},
        {**valid, "qualificationInputSha256": "A" * 64},
        {**valid, "sourceArchive": {**valid["sourceArchive"], "role": "sdist"}},
        {
            **valid,
            "wheelhouseManifests": list(reversed(valid["wheelhouseManifests"])),
        },
        {
            **valid,
            "gateResults": {
                **valid["gateResults"],
                "windowsTests": {
                    "outcome": "skipped",
                    "commandSha256": DIGEST,
                    "outputSha256": DIGEST,
                },
            },
        },
        {
            **valid,
            "environment": {
                **valid["environment"],
                "toolIdentities": list(reversed(valid["environment"]["toolIdentities"])),
            },
        },
        {
            **valid,
            "artifacts": [{"logicalId": "file:../escape", "sha256": DIGEST, "bytes": 1}],
        },
    ]
    for invalid in invalid_cases:
        assert list(validator.iter_errors(invalid)), invalid


@pytest.mark.parametrize(
    ("logical_id", "valid"),
    [
        ("file:release_manifest_schema", True),
        ("tool:build_python", True),
        ("wheelhouse:realtime_linux_runtime:requirements", True),
        ("wheelhouse:build:wheel:example-1.0.whl", True),
        ("provider:kokoro:resource:model/voice.bin", True),
        ("chrome:file:123.0.0.0/chrome.exe", True),
        ("file:unknown_role", False),
        ("tool:python", False),
        ("wheelhouse:build:wheel:../escape.whl", False),
        ("provider:moonshine:resource:../secret", False),
        ("chrome:file:C:/chrome.exe", False),
    ],
)
def test_verified_artifact_logical_id_is_closed_and_safe(logical_id: str, valid: bool) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    artifact = {"logicalId": logical_id, "sha256": DIGEST, "bytes": 1}
    artifact_validator = validator.evolve(schema=schema["$defs"]["VerifiedArtifactV1"])
    errors = list(artifact_validator.iter_errors(artifact))
    assert not errors if valid else errors


@pytest.mark.parametrize(
    "purpose",
    [
        "build",
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
        "realtime_linux_runtime",
        "hermes_v020_pluginmanager_runtime",
    ],
)
def test_wheel_logical_ids_require_a_safe_basename_not_a_path(purpose: str) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    logical_id_validator = validator.evolve(
        schema=schema["$defs"]["VerifiedArtifactV1"]["properties"]["logicalId"]
    )

    accepted = f"wheelhouse:{purpose}:wheel:example-1.0-py3-none-any.whl"
    assert list(logical_id_validator.iter_errors(accepted)) == []

    for unsafe_path_tail in (
        "nested/example-1.0-py3-none-any.whl",
        "nested/deeper/example-1.0-py3-none-any.whl",
        "../example-1.0-py3-none-any.whl",
        "example-1.0-py3-none-any.whl/",
    ):
        candidate = f"wheelhouse:{purpose}:wheel:{unsafe_path_tail}"
        assert list(logical_id_validator.iter_errors(candidate)), candidate


SHA256_IDENTITY_PROPERTIES = [
    ("ReleaseManifestV1", "qualificationInputSha256"),
    ("ReleaseManifestV1", "qualificationReportSha256"),
    ("ReleaseManifestV1", "benchmarkReportSha256"),
    ("CandidateV1", "canonicalDiffSha256"),
    ("GateResultV1", "commandSha256"),
    ("GateResultV1", "outputSha256"),
    ("LinuxWorkflowV1", "runIdSha256"),
    ("LinuxWorkflowV1", "artifactNameSha256"),
    ("LinuxWorkflowV1", "artifactSha256"),
    ("LinuxWorkflowV1", "commandOutputSha256"),
    ("VerifiedArtifactV1", "sha256"),
    *[
        (definition_name, "sha256")
        for definition_name in (
            "SourceArchiveRefV1",
            "BuildWheelhouseManifestRefV1",
            "HermesRuntimeWheelhouseManifestRefV1",
            "LinuxRuntimeWheelhouseManifestRefV1",
            "WindowsDirectRuntimeWheelhouseManifestRefV1",
            "WindowsSdistBuiltRuntimeWheelhouseManifestRefV1",
            "build_python_executableArtifactRefV1",
            "git_executableArtifactRefV1",
            "hatchling_wheelArtifactRefV1",
            "uv_executableArtifactRefV1",
        )
    ],
]

GIT_IDENTITY_PROPERTIES = [
    ("CandidateV1", "baselineCommit"),
    ("CandidateV1", "candidateCommit"),
    ("CandidateV1", "tree"),
]


@pytest.mark.parametrize(
    ("definition_name", "property_name"),
    SHA256_IDENTITY_PROPERTIES,
)
@pytest.mark.parametrize("line_ending", ["\r", "\n", "\r\n"])
def test_sha256_identity_properties_reject_cr_lf_and_crlf(
    definition_name: str, property_name: str, line_ending: str
) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    property_schema = (
        schema["properties"][property_name]
        if definition_name == "ReleaseManifestV1"
        else schema["$defs"][definition_name]["properties"][property_name]
    )
    property_validator = validator.evolve(schema=property_schema)

    assert list(property_validator.iter_errors(DIGEST)) == []
    assert list(property_validator.iter_errors(DIGEST + line_ending))


@pytest.mark.parametrize(
    ("definition_name", "property_name"),
    GIT_IDENTITY_PROPERTIES,
)
@pytest.mark.parametrize("line_ending", ["\r", "\n", "\r\n"])
def test_git_identity_properties_reject_cr_lf_and_crlf(
    definition_name: str, property_name: str, line_ending: str
) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    property_validator = validator.evolve(
        schema=schema["$defs"][definition_name]["properties"][property_name]
    )

    commit = "c" * 40
    assert list(property_validator.iter_errors(commit)) == []
    assert list(property_validator.iter_errors(commit + line_ending))


FREE_TEXT_PIN_PROPERTIES = [
    ("BuildPythonToolIdentityV1", "version", "3.11.15 (main) [MSC v.1938 64 bit (AMD64)]"),
    ("GitToolIdentityV1", "version", "git version 2.46.0.windows.1"),
    ("HatchlingToolIdentityV1", "version", "1.27.0 (hatchling build backend)"),
    ("ToolIdentityV1", "version", "0.5.0 (release build)"),
    ("UvToolIdentityV1", "version", "uv 0.5.0 (abcdef123456)"),
    ("ReleaseEnvironmentV1", "windowsEdition", "Windows 11 Pro for Workstations"),
    ("ReleaseEnvironmentV1", "windowsBuild", "26100.1"),
    ("ReleaseEnvironmentV1", "pythonFullVersion", "3.11.15 (main) [MSC v.1938 64 bit (AMD64)]"),
    ("ReleaseEnvironmentV1", "sqliteVersion", "3.46.1"),
]


@pytest.mark.parametrize(
    ("definition_name", "property_name", "realistic_value"),
    FREE_TEXT_PIN_PROPERTIES,
)
def test_free_text_pins_reject_controls_and_edge_padding_but_allow_interior_spaces(
    definition_name: str, property_name: str, realistic_value: str
) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    property_validator = validator.evolve(
        schema=schema["$defs"][definition_name]["properties"][property_name]
    )

    assert list(property_validator.iter_errors(realistic_value)) == []
    for invalid in (
        f" {realistic_value}",
        f"{realistic_value} ",
        f"{realistic_value}\t",
        f"{realistic_value}\x00",
        f"{realistic_value}\x1f",
        f"{realistic_value}\x7f",
        f"{realistic_value}\r",
        f"{realistic_value}\n",
        f"{realistic_value}\r\n",
    ):
        assert list(property_validator.iter_errors(invalid)), repr(invalid)


def test_release_manifest_integer_fields_reject_booleans() -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    integer_fields: list[tuple[str, dict[str, Any]]] = [
        ("ReleaseManifestV1.schemaVersion", schema["properties"]["schemaVersion"])
    ]
    for definition_name, definition in schema["$defs"].items():
        properties = definition.get("properties", {})
        if "bytes" in properties:
            integer_fields.append((f"{definition_name}.bytes", properties["bytes"]))

    assert {name for name, _ in integer_fields} == {
        "ReleaseManifestV1.schemaVersion",
        "VerifiedArtifactV1.bytes",
        "SourceArchiveRefV1.bytes",
        "BuildWheelhouseManifestRefV1.bytes",
        "HermesRuntimeWheelhouseManifestRefV1.bytes",
        "LinuxRuntimeWheelhouseManifestRefV1.bytes",
        "WindowsDirectRuntimeWheelhouseManifestRefV1.bytes",
        "WindowsSdistBuiltRuntimeWheelhouseManifestRefV1.bytes",
        "build_python_executableArtifactRefV1.bytes",
        "git_executableArtifactRefV1.bytes",
        "hatchling_wheelArtifactRefV1.bytes",
        "uv_executableArtifactRefV1.bytes",
    }
    for name, property_schema in integer_fields:
        property_validator = validator.evolve(schema=property_schema)
        for boolean in (False, True):
            assert list(property_validator.iter_errors(boolean)), (name, boolean)
