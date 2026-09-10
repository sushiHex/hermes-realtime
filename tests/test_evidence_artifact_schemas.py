"""Task 12 contract tests for the checked-in evidence qualification artifact schemas."""

from __future__ import annotations

import json
import re
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
WHEELHOUSE_SCHEMA = ROOT / "scripts" / "schemas" / "wheelhouse-manifest-v1.schema.json"
QUALIFICATION_INPUT_SCHEMA = ROOT / "scripts" / "schemas" / "qualification-input-v1.schema.json"
QUALIFICATION_REPORT_SCHEMA = ROOT / "scripts" / "schemas" / "qualification-report-v1.schema.json"
ARTIFACT_REF_KEYS = ["role", "relativePath", "basename", "sha256", "bytes"]
CONTEXTUAL_WHEELHOUSE_ROLES = {
    "WheelhouseConstraintsRefV1": "constraints",
    "WheelhouseRequirementsRefV1": "requirements",
    "WheelhouseWheelRefV1": "wheel",
}
QUALIFICATION_INPUT_FILE_ROLES = (
    "governing_plan",
    "candidate_source_archive",
    "qualification_runner",
    "direct_wheel",
    "direct_wheel_repeat",
    "sdist",
    "sdist_repeat",
    "sdist_built_wheel",
    "sdist_built_wheel_repeat",
    "build_wheelhouse_manifest",
    "windows_direct_runtime_wheelhouse_manifest",
    "windows_sdist_built_runtime_wheelhouse_manifest",
    "linux_runtime_wheelhouse_manifest",
    "hermes_runtime_wheelhouse_manifest",
    "benchmark_machine_schema",
    "benchmark_report_schema",
    "wheelhouse_manifest_schema",
    "qualification_input_schema",
    "qualification_report_schema",
    "release_manifest_schema",
    "benchmark_report",
    "benchmark_machine_manifest",
    "hermes_source_archive",
    "hermes_pluginmanager_runner",
    "chrome_executable",
    "chrome_version_directory_manifest",
    "livekit_archive",
    "livekit_executable",
    "codex_executable",
    "moonshine_distribution",
    "moonshine_model_manifest",
    "kokoro_distribution",
    "kokoro_model_manifest",
)
QUALIFICATION_INPUT_TOOL_ROLES = ("git", "uv", "build_python", "hatchling")
QUALIFICATION_INPUT_SCENARIO_IDS = (
    "deterministic_equivalence",
    "physical_capture_disabled",
    "physical_available_unconsented",
    "physical_microphone_response",
    "physical_typed_response",
    "physical_unmuted_transport",
    "physical_muted_transport",
    "physical_interruption_matrix",
    "physical_reconnect",
    "physical_media_replacement",
    "revoke_race",
    "capacity_rollover",
    "over_budget_turn",
    "spool_crash_matrix",
    "installed_host_crash_matrix",
    "synthetic_fault_matrix",
    "windows_filesystem_matrix",
    "windows_volume_full",
    "full_purge_cleanup",
    "owned_close_faults",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate object keys are noncanonical: {keys}"
    return dict(pairs)


def _load_canonical_schema(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM is noncanonical"
    assert raw.endswith(b"\n"), "canonical schema bytes end with exactly one LF"
    assert raw.count(b"\n") == 1, "canonical schema bytes contain exactly one LF"
    assert b"\r" not in raw, "canonical schema bytes contain no CR"
    document: dict[str, Any] = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert canonical + b"\n" == raw, "schema bytes are not compact sorted-key canonical JSON"
    return document


def test_wheelhouse_manifest_schema_freezes_canonical_contextual_artifact_contract() -> None:
    schema = _load_canonical_schema(WHEELHOUSE_SCHEMA)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://hermes-realtime.invalid/schemas/wheelhouse-manifest-v1.schema.json"
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schemaVersion",
        "purpose",
        "pythonVersion",
        "platform",
        "requirements",
        "constraints",
        "wheels",
    ]

    properties = schema["properties"]
    assert set(properties) == set(schema["required"])
    assert properties["schemaVersion"] == {"const": 1, "type": "integer"}
    assert properties["purpose"] == {
        "enum": [
            "build",
            "realtime_windows_direct_runtime",
            "realtime_windows_sdist_built_runtime",
            "realtime_linux_runtime",
            "hermes_v020_pluginmanager_runtime",
        ]
    }
    assert properties["platform"] == {"enum": ["windows_amd64", "linux_x86_64", "any"]}

    python_version = properties["pythonVersion"]
    assert python_version["type"] == "string"
    assert python_version["minLength"] == 1
    assert python_version["maxLength"] == 32

    definitions = schema["$defs"]
    assert set(definitions) == set(CONTEXTUAL_WHEELHOUSE_ROLES)
    assert properties["requirements"] == {"$ref": "#/$defs/WheelhouseRequirementsRefV1"}
    assert properties["constraints"] == {"$ref": "#/$defs/WheelhouseConstraintsRefV1"}

    wheels = properties["wheels"]
    assert wheels["type"] == "array"
    assert wheels["minItems"] == 1
    assert wheels["maxItems"] == 4096
    assert wheels["items"] == {"$ref": "#/$defs/WheelhouseWheelRefV1"}
    assert "uniqueItems" not in wheels, "a repeated wheel role is not a rejection ground"
    assert "contains" not in wheels, "a repeated wheel role is not a rejection ground"

    for name, role in CONTEXTUAL_WHEELHOUSE_ROLES.items():
        definition = definitions[name]
        assert "$ref" not in definition, name
        assert definition["type"] == "object", name
        assert definition["additionalProperties"] is False, name
        assert definition["required"] == ARTIFACT_REF_KEYS, name
        ref_properties = definition["properties"]
        assert set(ref_properties) == set(ARTIFACT_REF_KEYS), name
        assert ref_properties["role"] == {"const": role, "type": "string"}, name
        assert ref_properties["sha256"] == {
            "pattern": "^(?![\\s\\S]*[\\r\\n])[0-9a-f]{64}$",
            "type": "string",
        }, name
        assert ref_properties["bytes"] == {
            "maximum": 9223372036854775807,
            "minimum": 1,
            "type": "integer",
        }, name

        relative_path = ref_properties["relativePath"]
        assert relative_path["type"] == "string", name
        assert relative_path["minLength"] == 1, name
        assert relative_path["maxLength"] == 512, name
        path_pattern = re.compile(relative_path["pattern"])
        for accepted in (
            "requirements.txt",
            "wheels/example-1.0.0-py3-none-any.whl",
            "a/b/c-1_0.txt",
            ".hidden.txt",
        ):
            assert path_pattern.search(accepted) is not None, (name, accepted)
        for rejected in (
            "",
            ".",
            "..",
            "/absolute.txt",
            "wheels/",
            "wheels//x.whl",
            "../escape.txt",
            "wheels/../escape.whl",
            "wheels/./x.whl",
            "C:/wheels/x.whl",
            "wheels\\x.whl",
            "wheels/x y.whl",
            "wheels/na\u00efve.whl",
        ):
            assert path_pattern.search(rejected) is None, (name, rejected)

        basename = ref_properties["basename"]
        assert basename["type"] == "string", name
        assert basename["minLength"] == 1, name
        assert basename["maxLength"] == 256, name
        basename_pattern = re.compile(basename["pattern"])
        for accepted in ("constraints.txt", "example-1.0.0-py3-none-any.whl"):
            assert basename_pattern.search(accepted) is not None, (name, accepted)
        for rejected in ("", ".", "..", "wheels/x.whl", "x y.whl", "wheels\\x.whl"):
            assert basename_pattern.search(rejected) is None, (name, rejected)


def _requirement_distribution(entry: str) -> str:
    return re.split(r"[<>=!~\[;( ]", entry, maxsplit=1)[0].strip().lower().replace("_", "-")


def test_dev_group_declares_direct_jsonschema_accepted_by_draft_2020_12_metaschema() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev_group = pyproject["dependency-groups"]["dev"]
    declared = [entry for entry in dev_group if _requirement_distribution(entry) == "jsonschema"]
    assert declared == ["jsonschema>=4.23,<5"], (
        "Draft 2020-12 validation must use a directly declared development dependency, "
        "never incidental transitive availability"
    )

    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    Draft202012Validator.check_schema(_load_canonical_schema(WHEELHOUSE_SCHEMA))


def test_wheelhouse_schema_rejects_terminal_newlines_in_artifact_refs() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    validator = Draft202012Validator(_load_canonical_schema(WHEELHOUSE_SCHEMA))
    digest = "a" * 64
    artifact = {
        "role": "requirements",
        "relativePath": "requirements.txt",
        "basename": "requirements.txt",
        "sha256": digest,
        "bytes": 1,
    }
    instance = {
        "schemaVersion": 1,
        "purpose": "build",
        "pythonVersion": "3.11.15",
        "platform": "windows_amd64",
        "requirements": artifact,
        "constraints": {**artifact, "role": "constraints"},
        "wheels": [
            {
                **artifact,
                "role": "wheel",
                "relativePath": "wheels/x.whl",
                "basename": "x.whl",
            }
        ],
    }

    for field, value in (
        ("relativePath", "requirements.txt"),
        ("basename", "requirements.txt"),
        ("sha256", digest),
    ):
        for suffix in ("\n", "\r", "\r\n"):
            for reference_key in ("requirements", "constraints"):
                invalid = {
                    **instance,
                    reference_key: {**instance[reference_key], field: f"{value}{suffix}"},
                }
                assert list(validator.iter_errors(invalid)), (reference_key, field, repr(suffix))

            invalid_wheel = {
                **instance,
                "wheels": [{**instance["wheels"][0], field: f"{value}{suffix}"}],
            }
            assert list(validator.iter_errors(invalid_wheel)), ("wheels", field, repr(suffix))


def _qualification_input_instance() -> dict[str, Any]:
    digest = "a" * 64
    files = [
        {
            "role": role,
            "relativePath": f"artifacts/{role}.bin",
            "basename": f"{role}.bin",
            "sha256": digest,
            "bytes": 1,
        }
        for role in QUALIFICATION_INPUT_FILE_ROLES
    ]
    tool_artifact_roles = {
        "git": "git_executable",
        "uv": "uv_executable",
        "build_python": "build_python_executable",
        "hatchling": "hatchling_wheel",
    }
    return {
        "schemaVersion": 1,
        "candidate": {
            "baselineCommit": "b" * 40,
            "candidateCommit": "c" * 40,
            "tree": "d" * 40,
            "canonicalDiffSha256": digest,
            "version": "0.0.3",
        },
        "files": files,
        "toolIdentities": [
            {
                "role": role,
                "version": "1.27.0" if role == "hatchling" else "1.0.0",
                "artifact": {
                    "role": artifact_role,
                    "relativePath": f"tools/{artifact_role}.bin",
                    "basename": f"{artifact_role}.bin",
                    "sha256": digest,
                    "bytes": 1,
                },
            }
            for role, artifact_role in tool_artifact_roles.items()
        ],
        "expected": {
            "pythonFullVersion": (
                "3.11.15 (main, Dec  1 2025, 00:00:00) [MSC v.1944 64 bit (AMD64)]"
            ),
            "pythonArchitecture": "AMD64",
            "chromeVersion": "140.0.0.0",
            "livekitVersion": "1.13.4",
            "codexVersion": "0.1.0",
            "moonshineVersion": "0.1.0",
            "kokoroVersion": "0.1.0",
            "codexModel": "gpt-5.6-terra",
            "codexEffort": "low",
            "sourceArchivePrefix": "hermes-realtime-0.0.3/",
            "benchmarkMachineSchemaSha256": digest,
            "benchmarkReportSchemaSha256": digest,
            "wheelhouseManifestSchemaSha256": digest,
            "qualificationInputSchemaSha256": digest,
            "qualificationReportSchemaSha256": digest,
            "releaseManifestSchemaSha256": digest,
        },
        "requestedScenarioIds": list(QUALIFICATION_INPUT_SCENARIO_IDS),
    }


def test_qualification_input_schema_freezes_the_complete_closed_input_contract() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    schema = _load_canonical_schema(QUALIFICATION_INPUT_SCHEMA)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://hermes-realtime.invalid/schemas/qualification-input-v1.schema.json"
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schemaVersion",
        "candidate",
        "files",
        "toolIdentities",
        "expected",
        "requestedScenarioIds",
    ]
    assert set(schema["properties"]) == set(schema["required"])

    definitions = schema["$defs"]
    candidate = definitions["CandidateV1"]
    assert candidate["required"] == [
        "baselineCommit",
        "candidateCommit",
        "tree",
        "canonicalDiffSha256",
        "version",
    ]
    assert candidate["properties"]["version"] == {"const": "0.0.3", "type": "string"}

    file_ref = definitions["QualificationInputFileRefV1"]
    assert file_ref["required"] == ARTIFACT_REF_KEYS
    assert file_ref["properties"]["role"]["enum"] == list(QUALIFICATION_INPUT_FILE_ROLES)
    files = schema["properties"]["files"]
    assert files["minItems"] == files["maxItems"] == len(QUALIFICATION_INPUT_FILE_ROLES)
    assert len(files["allOf"]) == len(QUALIFICATION_INPUT_FILE_ROLES)

    tool_identity = definitions["ToolIdentityV1"]
    assert tool_identity["required"] == ["role", "version", "artifact"]
    assert tool_identity["properties"]["role"]["enum"] == list(QUALIFICATION_INPUT_TOOL_ROLES)
    tools = schema["properties"]["toolIdentities"]
    assert tools["minItems"] == tools["maxItems"] == len(QUALIFICATION_INPUT_TOOL_ROLES)
    assert len(tools["allOf"]) == len(QUALIFICATION_INPUT_TOOL_ROLES)

    expected = definitions["ExpectedQualificationEnvironmentV1"]
    assert expected["required"] == [
        "pythonFullVersion",
        "pythonArchitecture",
        "chromeVersion",
        "livekitVersion",
        "codexVersion",
        "moonshineVersion",
        "kokoroVersion",
        "codexModel",
        "codexEffort",
        "sourceArchivePrefix",
        "benchmarkMachineSchemaSha256",
        "benchmarkReportSchemaSha256",
        "wheelhouseManifestSchemaSha256",
        "qualificationInputSchemaSha256",
        "qualificationReportSchemaSha256",
        "releaseManifestSchemaSha256",
    ]
    assert expected["properties"]["codexModel"] == {"$ref": "#/$defs/VersionV1"}
    assert expected["properties"]["codexEffort"] == {"$ref": "#/$defs/VersionV1"}
    assert expected["properties"]["sourceArchivePrefix"] == {
        "const": "hermes-realtime-0.0.3/",
        "type": "string",
    }

    requested = schema["properties"]["requestedScenarioIds"]
    assert requested["items"]["enum"] == list(QUALIFICATION_INPUT_SCENARIO_IDS)
    assert requested["minItems"] == requested["maxItems"] == len(QUALIFICATION_INPUT_SCENARIO_IDS)
    assert requested["uniqueItems"] is True

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    instance = _qualification_input_instance()
    assert list(validator.iter_errors(instance)) == []

    invalid_instances = [
        {**instance, "unexpected": True},
        {**instance, "schemaVersion": True},
        {
            **instance,
            "candidate": {**instance["candidate"], "canonicalDiffSha256": "A" * 64},
        },
        {**instance, "candidate": {**instance["candidate"], "version": "0.0.2"}},
        {
            **instance,
            "files": [
                {**instance["files"][0], "relativePath": "../outside.bin"},
                *instance["files"][1:],
            ],
        },
        {
            **instance,
            "files": [
                {**instance["files"][0], "bytes": 0},
                *instance["files"][1:],
            ],
        },
        {
            **instance,
            "toolIdentities": [
                *instance["toolIdentities"][:-1],
                {**instance["toolIdentities"][-1], "version": "1.26.0"},
            ],
        },
        {**instance, "files": instance["files"][:-1]},
        {**instance, "files": [*instance["files"][:-1], instance["files"][0]]},
        {
            **instance,
            "toolIdentities": [
                {
                    **instance["toolIdentities"][0],
                    "artifact": {
                        **instance["toolIdentities"][0]["artifact"],
                        "role": "uv_executable",
                    },
                },
                *instance["toolIdentities"][1:],
            ],
        },
        {**instance, "requestedScenarioIds": instance["requestedScenarioIds"][:-1]},
        {
            **instance,
            "requestedScenarioIds": [
                *instance["requestedScenarioIds"][:-1],
                "not_a_scenario",
            ],
        },
    ]
    for invalid in invalid_instances:
        assert list(validator.iter_errors(invalid)), invalid


def test_qualification_input_schema_rejects_terminal_newlines_in_identities() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    validator = Draft202012Validator(_load_canonical_schema(QUALIFICATION_INPUT_SCHEMA))
    instance = _qualification_input_instance()
    candidate_fields = (
        "baselineCommit",
        "candidateCommit",
        "tree",
        "canonicalDiffSha256",
    )
    expected_hash_fields = (
        "benchmarkMachineSchemaSha256",
        "benchmarkReportSchemaSha256",
        "wheelhouseManifestSchemaSha256",
        "qualificationInputSchemaSha256",
        "qualificationReportSchemaSha256",
        "releaseManifestSchemaSha256",
    )

    for suffix in ("\n", "\r", "\r\n"):
        for field in candidate_fields:
            invalid_candidate = {
                **instance,
                "candidate": {
                    **instance["candidate"],
                    field: f"{instance['candidate'][field]}{suffix}",
                },
            }
            assert list(validator.iter_errors(invalid_candidate)), (field, repr(suffix))

        invalid_file = {
            **instance,
            "files": [
                {**instance["files"][0], "sha256": f"{'a' * 64}{suffix}"},
                *instance["files"][1:],
            ],
        }
        assert list(validator.iter_errors(invalid_file)), ("files", repr(suffix))

        invalid_tool = {
            **instance,
            "toolIdentities": [
                {
                    **instance["toolIdentities"][0],
                    "artifact": {
                        **instance["toolIdentities"][0]["artifact"],
                        "sha256": f"{'a' * 64}{suffix}",
                    },
                },
                *instance["toolIdentities"][1:],
            ],
        }
        assert list(validator.iter_errors(invalid_tool)), ("toolIdentities", repr(suffix))

        for field in expected_hash_fields:
            invalid_expected = {
                **instance,
                "expected": {
                    **instance["expected"],
                    field: f"{instance['expected'][field]}{suffix}",
                },
            }
            assert list(validator.iter_errors(invalid_expected)), (field, repr(suffix))


def test_qualification_input_schema_rejects_control_or_padded_identity_pins() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    validator = Draft202012Validator(_load_canonical_schema(QUALIFICATION_INPUT_SCHEMA))
    instance = _qualification_input_instance()
    expected_identity_fields = (
        "pythonFullVersion",
        "pythonArchitecture",
        "chromeVersion",
        "livekitVersion",
        "codexVersion",
        "moonshineVersion",
        "kokoroVersion",
        "codexModel",
        "codexEffort",
    )
    invalid_suffixes = ("\n", "\r", "\r\n", "\v", "\f", "\x85", "\u2028", "\t", " ")

    for suffix in invalid_suffixes:
        for field in expected_identity_fields:
            invalid_expected = {
                **instance,
                "expected": {
                    **instance["expected"],
                    field: f"{instance['expected'][field]}{suffix}",
                },
            }
            assert list(validator.iter_errors(invalid_expected)), (field, repr(suffix))

        for index in range(3):
            tool_identities = [*instance["toolIdentities"]]
            tool_identities[index] = {
                **tool_identities[index],
                "version": f"{tool_identities[index]['version']}{suffix}",
            }
            invalid_tool_version = {**instance, "toolIdentities": tool_identities}
            assert list(validator.iter_errors(invalid_tool_version)), (index, repr(suffix))

    for invalid_value in ("", " ", " padded", "padded "):
        invalid_architecture = {
            **instance,
            "expected": {**instance["expected"], "pythonArchitecture": invalid_value},
        }
        assert list(validator.iter_errors(invalid_architecture)), repr(invalid_value)

    full_python_version = "3.11.15 (main, Dec  1 2025, 00:00:00) [MSC v.1944 64 bit (AMD64)]"
    valid_full_version = {
        **instance,
        "expected": {
            **instance["expected"],
            "pythonFullVersion": full_python_version,
        },
    }
    assert not list(validator.iter_errors(valid_full_version))


def test_qualification_report_schema_freezes_the_closed_canonical_report_dtos() -> None:
    """The report schema is a closed wire contract, not a permissive report template."""

    schema = _load_canonical_schema(QUALIFICATION_REPORT_SCHEMA)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://hermes-realtime.invalid/schemas/qualification-report-v1.schema.json"
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schemaVersion",
        "qualificationInputSha256",
        "governingPlanSha256",
        "candidate",
        "verifiedArtifacts",
        "environment",
        "topology",
        "providers",
        "benchmarkReportSha256",
        "scenarios",
        "operatorAttestations",
        "cleanup",
        "startedAtUtc",
        "finishedAtUtc",
        "passed",
    ]
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["schemaVersion"] == {"const": 1, "type": "integer"}

    definitions = schema["$defs"]
    assert set(definitions) == {
        "CandidateV1",
        "CleanupV1",
        "EnvironmentV1",
        "InstalledHostCrashCaseV1",
        "MeasurementV1",
        "OperatorAttestationV1",
        "PhysicalInterruptionCaseV1",
        "ProcessV1",
        "ProviderV1",
        "RtcSocketV1",
        "ScenarioV1",
        "SpoolCrashCaseV1",
        "SyntheticFaultCaseV1",
        "TopologyJobV1",
        "TopologyV1",
        "VerifiedArtifactV1",
        "WindowsFilesystemCaseV1",
    }

    for name, definition in definitions.items():
        assert definition["type"] == "object", name
        assert definition["additionalProperties"] is False, name
        assert set(definition["required"]) <= set(definition["properties"]), name

    # Measurements deliberately permit a closed optional subset rather than
    # requiring unrelated units for every scenario.
    assert definitions["MeasurementV1"]["required"] == []

    candidate = definitions["CandidateV1"]
    assert candidate["required"] == [
        "baselineCommit",
        "candidateCommit",
        "tree",
        "canonicalDiffSha256",
        "version",
    ]
    assert candidate["properties"]["version"] == {"const": "0.0.3", "type": "string"}

    environment = definitions["EnvironmentV1"]
    assert environment["required"] == [
        "windowsEdition",
        "windowsBuild",
        "windowsArchitecture",
        "pythonFullVersion",
        "pythonArchitecture",
        "sqliteVersion",
        "cpuModel",
        "logicalCpuCount",
        "installedRamBytes",
        "acPower",
        "powerScheme",
        "chromeVersion",
        "chromeVersionDirectoryManifestSha256",
        "systemVolumeIdentitySha256",
        "vhdxBackingVolumeIdentitySha256",
        "vhdxBackingDiskExtentsSha256",
        "vhdxBackingNonSystem",
    ]
    assert environment["properties"]["powerScheme"] == {
        "const": "high_performance",
        "type": "string",
    }
    assert environment["properties"]["vhdxBackingNonSystem"] == {"const": True}

    topology = definitions["TopologyV1"]
    assert topology["required"] == [
        "hermesTaskMode",
        "livekitVersion",
        "livekitCredentialProfileId",
        "signalingLoopbackOnly",
        "rtcSocketInventoryRecorded",
        "rtcSocketInventory",
        "firewallRulesCreated",
        "runnerPid",
        "runnerCreationFiletime",
        "jobs",
    ]
    assert topology["properties"]["hermesTaskMode"] == {
        "const": "qualification_disabled",
        "type": "string",
    }
    assert topology["properties"]["livekitCredentialProfileId"] == {
        "const": "livekit_local_v1",
        "type": "string",
    }
    assert topology["properties"]["firewallRulesCreated"] == {"const": False}

    scenario = definitions["ScenarioV1"]
    assert scenario["required"] == [
        "scenarioId",
        "proofClass",
        "outcome",
        "exitCode",
        "caseResults",
        "measurements",
        "machineAssertions",
        "attestationIds",
    ]
    assert scenario["properties"]["outcome"] == {
        "enum": ["pass", "fail", "blocked"],
        "type": "string",
    }
    assert scenario["properties"]["caseResults"]["items"]["oneOf"] == [
        {"$ref": "#/$defs/PhysicalInterruptionCaseV1"},
        {"$ref": "#/$defs/SpoolCrashCaseV1"},
        {"$ref": "#/$defs/InstalledHostCrashCaseV1"},
        {"$ref": "#/$defs/SyntheticFaultCaseV1"},
        {"$ref": "#/$defs/WindowsFilesystemCaseV1"},
    ]

    spool_failpoints = [
        "before_begin",
        "after_event_insert_before_commit",
        "after_event_commit",
        "before_seal_commit",
        "after_seal_commit_before_ack",
        "after_revoke_request_commit",
        "after_logical_purge_before_vacuum",
        "after_vacuum_before_ack",
        "after_drain_ack_before_exit",
        "after_root_marker_init_create",
        "after_root_marker_init_partial_write",
        "after_root_marker_init_full_write",
        "after_root_marker_init_flush",
        "after_root_marker_activation",
        "after_sentinel_init_create",
        "after_sentinel_init_partial_write",
        "after_sentinel_init_full_write",
        "after_sentinel_init_flush",
        "after_sentinel_activation",
        "after_first_db_create",
        "after_first_schema_commit",
        "after_first_epoch_commit",
        "after_first_marker_clear",
        "after_recreate_pending_fsync",
        "after_recreate_db_create",
        "after_recreate_schema_commit",
        "after_recreate_epoch_commit",
        "after_recreate_marker_clear",
        "before_rollback_sentinel_write",
        "after_rollback_sentinel_fsync",
        "before_rollback_db_latch",
        "after_rollback_db_latch",
        "after_full_purge_marker_fsync",
        "after_full_purge_db_delete",
        "after_full_purge_journal_delete",
        "after_full_purge_wal_delete",
        "after_full_purge_shm_delete",
        "after_full_purge_vacuum_delete",
        "after_full_purge_tmp_delete",
        "after_full_purge_absence_verify",
        "after_full_purge_marker_clear",
    ]
    spool_case = definitions["SpoolCrashCaseV1"]
    assert spool_case["properties"]["faultMechanism"] == {
        "enum": spool_failpoints,
        "type": "string",
    }
    assert spool_case["properties"]["caseId"] == {
        "enum": [
            f"{failpoint}@exit{exit_mode}"
            for failpoint in spool_failpoints
            for exit_mode in (197, 198)
        ],
        "type": "string",
    }

    coupled_fault = definitions["SyntheticFaultCaseV1"]
    assert coupled_fault["properties"]["caseId"] == coupled_fault["properties"]["faultMechanism"]
    assert coupled_fault["properties"]["caseId"]["enum"] == [
        "queue_capacity_coupled",
        "deny_filter",
        "clock_rollback",
        "sqlite_injected_fault",
        "writer_drain_blocked",
    ]
    assert "queue_bytes_full" not in coupled_fault["properties"]["caseId"]["enum"]

    measurements = definitions["MeasurementV1"]
    assert set(measurements["properties"]) == {
        "durationMs",
        "admissionP50Ns",
        "admissionP95Ns",
        "admissionP99Ns",
        "admissionMaxNs",
        "eventLoopLagRegressionP99Ns",
        "ownedAllocatedBytes",
        "requiredFreeBytes",
        "volumeFreeBytes",
        "vhdxCapacityBytes",
        "probeWriteBytes",
        "windowsErrorCode",
        "queueRecordCount",
        "queueCanonicalBytes",
        "queuePhysicalCount",
        "maxQueueRecords",
        "maxQueueCanonicalBytes",
        "maxQueuePhysicalItems",
        "maxCanonicalRecordBytes",
    }

    cleanup = definitions["CleanupV1"]
    assert cleanup["required"] == [
        "chromeProfileRemoved",
        "chromeImageUnchanged",
        "inputExtractionRemoved",
        "evidenceRootsRemoved",
        "cleanVenvsRemoved",
        "vhdxDismounted",
        "vhdxBackingRemoved",
        "vhdxBackingNonSystem",
        "jobsZeroActive",
        "retainedProcessHandlesWaited",
        "retainedProcessHandlesClosed",
        "livekitProcessesExited",
        "hostProcessesExited",
        "chromeProcessesExited",
        "codexProcessesExited",
        "pidReuseSafe",
        "daclRestored",
        "reportTemporaryRemoved",
        "rawDatabasePurged",
        "rawAudioPurged",
        "rawScreenshotsPurged",
        "adjacentDecoysPreserved",
        "firewallUnchanged",
    ]
    assert all(value == {"type": "boolean"} for value in cleanup["properties"].values())


def test_qualification_report_schema_is_a_draft_2020_12_schema() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    Draft202012Validator.check_schema(_load_canonical_schema(QUALIFICATION_REPORT_SCHEMA))


def _minimal_qualification_report() -> dict[str, Any]:
    digest = "a" * 64
    commit = "b" * 40
    scenarios = [
        "deterministic_equivalence",
        "physical_capture_disabled",
        "physical_available_unconsented",
        "physical_microphone_response",
        "physical_typed_response",
        "physical_unmuted_transport",
        "physical_muted_transport",
        "physical_interruption_matrix",
        "physical_reconnect",
        "physical_media_replacement",
        "revoke_race",
        "capacity_rollover",
        "over_budget_turn",
        "spool_crash_matrix",
        "installed_host_crash_matrix",
        "synthetic_fault_matrix",
        "windows_filesystem_matrix",
        "windows_volume_full",
        "full_purge_cleanup",
        "owned_close_faults",
    ]
    return {
        "schemaVersion": 1,
        "qualificationInputSha256": digest,
        "governingPlanSha256": digest,
        "candidate": {
            "baselineCommit": commit,
            "candidateCommit": commit,
            "tree": commit,
            "canonicalDiffSha256": digest,
            "version": "0.0.3",
        },
        "verifiedArtifacts": [{"logicalId": "file:governing_plan", "sha256": digest, "bytes": 1}],
        "environment": {
            "windowsEdition": "Windows 10 Pro",
            "windowsBuild": "19045.1",
            "windowsArchitecture": "AMD64",
            "pythonFullVersion": "3.11.15",
            "pythonArchitecture": "64bit",
            "sqliteVersion": "3.46.1",
            "cpuModel": "Example CPU",
            "logicalCpuCount": 1,
            "installedRamBytes": 1,
            "acPower": True,
            "powerScheme": "high_performance",
            "chromeVersion": "127.0.0.0",
            "chromeVersionDirectoryManifestSha256": digest,
            "systemVolumeIdentitySha256": digest,
            "vhdxBackingVolumeIdentitySha256": digest,
            "vhdxBackingDiskExtentsSha256": digest,
            "vhdxBackingNonSystem": True,
        },
        "topology": {
            "hermesTaskMode": "qualification_disabled",
            "livekitVersion": "1.13.4",
            "livekitCredentialProfileId": "livekit_local_v1",
            "signalingLoopbackOnly": True,
            "rtcSocketInventoryRecorded": True,
            "rtcSocketInventory": [],
            "firewallRulesCreated": False,
            "runnerPid": 1,
            "runnerCreationFiletime": 1,
            "jobs": [],
        },
        "providers": {
            "codexExecutableVersion": "1.0.0",
            "codexExecutableSha256": digest,
            "codexModel": "gpt-5.6-terra",
            "codexEffort": "low",
            "moonshineDistributionVersion": "0.1.0",
            "moonshineDistributionSha256": digest,
            "moonshineModelIdentitySha256": digest,
            "moonshineModelTier": "medium",
            "kokoroDistributionVersion": "1.0.0",
            "kokoroDistributionSha256": digest,
            "kokoroModelIdentitySha256": digest,
            "kokoroVoice": "bf_isabella",
        },
        "benchmarkReportSha256": digest,
        "scenarios": [
            {
                "scenarioId": scenario_id,
                "proofClass": "packaged_process",
                "outcome": "pass",
                "exitCode": 0,
                "caseResults": [],
                "measurements": {},
                "machineAssertions": [],
                "attestationIds": [],
            }
            for scenario_id in scenarios
        ],
        "operatorAttestations": [],
        "cleanup": {
            "chromeProfileRemoved": True,
            "chromeImageUnchanged": True,
            "inputExtractionRemoved": True,
            "evidenceRootsRemoved": True,
            "cleanVenvsRemoved": True,
            "vhdxDismounted": True,
            "vhdxBackingRemoved": True,
            "vhdxBackingNonSystem": True,
            "jobsZeroActive": True,
            "retainedProcessHandlesWaited": True,
            "retainedProcessHandlesClosed": True,
            "livekitProcessesExited": True,
            "hostProcessesExited": True,
            "chromeProcessesExited": True,
            "codexProcessesExited": True,
            "pidReuseSafe": True,
            "daclRestored": True,
            "reportTemporaryRemoved": True,
            "rawDatabasePurged": True,
            "rawAudioPurged": True,
            "rawScreenshotsPurged": True,
            "adjacentDecoysPreserved": True,
            "firewallUnchanged": True,
        },
        "startedAtUtc": "2026-08-19T00:00:00Z",
        "finishedAtUtc": "2026-08-19T00:00:01Z",
        "passed": True,
    }


def test_qualification_report_schema_rejects_extra_wrong_type_and_retired_capacity_values() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    validator = Draft202012Validator(_load_canonical_schema(QUALIFICATION_REPORT_SCHEMA))
    report = _governed_qualification_report()
    assert not list(validator.iter_errors(report))

    extra = deepcopy(report)
    extra["operatorName"] = "forbidden"
    assert list(validator.iter_errors(extra))

    boolean_pid = deepcopy(report)
    boolean_pid["topology"]["runnerPid"] = True
    assert list(validator.iter_errors(boolean_pid))

    malformed_digest = deepcopy(report)
    malformed_digest["benchmarkReportSha256"] = "A" * 64
    assert list(validator.iter_errors(malformed_digest))

    retired_capacity_case = deepcopy(report)
    retired_capacity_case["scenarios"][15]["caseResults"] = [
        {
            "caseId": "queue_bytes_full",
            "faultMechanism": "queue_bytes_full",
            "outcome": "pass",
            "assertions": ["purge_verified"],
        }
    ]
    assert list(validator.iter_errors(retired_capacity_case))


_SPOOL_FAILPOINTS = [
    "before_begin",
    "after_event_insert_before_commit",
    "after_event_commit",
    "before_seal_commit",
    "after_seal_commit_before_ack",
    "after_revoke_request_commit",
    "after_logical_purge_before_vacuum",
    "after_vacuum_before_ack",
    "after_drain_ack_before_exit",
    "after_root_marker_init_create",
    "after_root_marker_init_partial_write",
    "after_root_marker_init_full_write",
    "after_root_marker_init_flush",
    "after_root_marker_activation",
    "after_sentinel_init_create",
    "after_sentinel_init_partial_write",
    "after_sentinel_init_full_write",
    "after_sentinel_init_flush",
    "after_sentinel_activation",
    "after_first_db_create",
    "after_first_schema_commit",
    "after_first_epoch_commit",
    "after_first_marker_clear",
    "after_recreate_pending_fsync",
    "after_recreate_db_create",
    "after_recreate_schema_commit",
    "after_recreate_epoch_commit",
    "after_recreate_marker_clear",
    "before_rollback_sentinel_write",
    "after_rollback_sentinel_fsync",
    "before_rollback_db_latch",
    "after_rollback_db_latch",
    "after_full_purge_marker_fsync",
    "after_full_purge_db_delete",
    "after_full_purge_journal_delete",
    "after_full_purge_wal_delete",
    "after_full_purge_shm_delete",
    "after_full_purge_vacuum_delete",
    "after_full_purge_tmp_delete",
    "after_full_purge_absence_verify",
    "after_full_purge_marker_clear",
]


def _spool_sentinel_at_entry(point: str) -> str:
    index = _SPOOL_FAILPOINTS.index(point)
    if 9 <= index <= 17:
        return "no_final_sentinel"
    if 18 <= index <= 21 or 23 <= index <= 26:
        return "first_create_pending"
    if 29 <= index <= 31:
        return "clock_rollback_purge_pending"
    if 32 <= index <= 39:
        return "full_purge_pending"
    return "clear"


_SCENARIO_MATRIX = {
    "deterministic_equivalence": ("deterministic_equivalence", ["conversation_trace_equal"]),
    "physical_capture_disabled": (
        "physical_browser",
        [
            "no_evidence_artifact",
            "owned_process_cleanup",
            "rtc_socket_inventory_recorded",
            "signaling_loopback_only",
        ],
    ),
    "physical_available_unconsented": (
        "physical_browser",
        [
            "no_evidence_artifact",
            "owned_process_cleanup",
            "rtc_socket_inventory_recorded",
            "signaling_loopback_only",
        ],
    ),
    "physical_microphone_response": (
        "physical_browser",
        [
            "medium_backend_selected",
            "owned_process_cleanup",
            "persisted_source_equal",
            "physical_phrase_stt_match",
            "rtc_socket_inventory_recorded",
            "signaling_loopback_only",
        ],
    ),
    "physical_typed_response": (
        "physical_browser",
        [
            "owned_process_cleanup",
            "persisted_source_equal",
            "rtc_socket_inventory_recorded",
            "signaling_loopback_only",
        ],
    ),
    "physical_unmuted_transport": (
        "physical_browser",
        [
            "owned_process_cleanup",
            "rtc_socket_inventory_recorded",
            "server_transport_equal",
            "signaling_loopback_only",
        ],
    ),
    "physical_muted_transport": (
        "physical_browser",
        [
            "owned_process_cleanup",
            "rtc_socket_inventory_recorded",
            "server_transport_equal",
            "signaling_loopback_only",
        ],
    ),
    "physical_interruption_matrix": (
        "physical_browser",
        [
            "never_persisted",
            "owned_process_cleanup",
            "persisted_but_excluded",
            "rtc_socket_inventory_recorded",
            "server_transport_equal",
            "signaling_loopback_only",
        ],
    ),
    "physical_reconnect": (
        "physical_browser",
        [
            "never_persisted",
            "owned_process_cleanup",
            "rtc_socket_inventory_recorded",
            "signaling_loopback_only",
        ],
    ),
    "physical_media_replacement": (
        "physical_browser",
        [
            "never_persisted",
            "owned_process_cleanup",
            "rtc_socket_inventory_recorded",
            "signaling_loopback_only",
        ],
    ),
    "revoke_race": ("packaged_process", ["purge_verified", "purged", "revocation_request_durable"]),
    "capacity_rollover": ("packaged_process", ["persisted_source_equal", "rollover_atomic"]),
    "over_budget_turn": ("packaged_process", ["persisted_but_excluded", "rejected_source_absent"]),
    "spool_crash_matrix": (
        "packaged_process",
        [
            "partial_seal_absent",
            "postcommit_seal_revalidated",
            "precommit_session_unsealed",
            "purge_verified",
        ],
    ),
    "installed_host_crash_matrix": (
        "packaged_process",
        ["checkpoint_observed", "owned_process_cleanup", "purge_verified"],
    ),
    "synthetic_fault_matrix": (
        "synthetic_injected",
        [
            "aggregate_byte_capacity_unreachable",
            "configured_capacity_algebra_equal",
            "physical_capacity_unreachable_before_record_capacity",
            "purge_verified",
            "purged",
            "record_capacity_rejected",
            "rejected_source_absent",
        ],
    ),
    "windows_filesystem_matrix": (
        "real_windows_filesystem",
        ["adjacent_decoys_preserved", "dacl_restored", "owned_process_cleanup"],
    ),
    "windows_volume_full": (
        "real_windows_filesystem",
        ["headroom_formula_equal", "owned_process_cleanup", "vhdx_backing_non_system"],
    ),
    "full_purge_cleanup": (
        "real_windows_filesystem",
        ["adjacent_decoys_preserved", "purge_verified", "purged"],
    ),
    "owned_close_faults": (
        "packaged_process",
        ["conversation_trace_equal_to_revised_close_baseline", "owned_process_cleanup"],
    ),
}

_COUPLED_CAPACITY_ASSERTIONS = [
    "aggregate_byte_capacity_unreachable",
    "configured_capacity_algebra_equal",
    "physical_capacity_unreachable_before_record_capacity",
    "purge_verified",
    "record_capacity_rejected",
    "rejected_source_absent",
]


def _governed_qualification_report() -> dict[str, Any]:
    report = _minimal_qualification_report()
    cases_by_scenario: dict[str, list[dict[str, Any]]] = {
        "physical_interruption_matrix": [
            {
                "caseId": "barge_in_first_chunk",
                "faultMechanism": "none",
                "outcome": "pass",
                "assertions": [
                    "owned_process_cleanup",
                    "persisted_but_excluded",
                    "rtc_socket_inventory_recorded",
                    "signaling_loopback_only",
                ],
            },
            {
                "caseId": "barge_in_between_chunks",
                "faultMechanism": "none",
                "outcome": "pass",
                "assertions": [
                    "owned_process_cleanup",
                    "persisted_but_excluded",
                    "rtc_socket_inventory_recorded",
                    "server_transport_equal",
                    "signaling_loopback_only",
                ],
            },
            {
                "caseId": "stop_speaking",
                "faultMechanism": "none",
                "outcome": "pass",
                "assertions": [
                    "owned_process_cleanup",
                    "persisted_but_excluded",
                    "rtc_socket_inventory_recorded",
                    "server_transport_equal",
                    "signaling_loopback_only",
                ],
            },
            {
                "caseId": "replay_interrupted",
                "faultMechanism": "none",
                "outcome": "pass",
                "assertions": [
                    "never_persisted",
                    "owned_process_cleanup",
                    "rtc_socket_inventory_recorded",
                    "signaling_loopback_only",
                ],
            },
        ],
        "installed_host_crash_matrix": [
            {
                "caseId": case_id,
                "faultMechanism": case_id,
                "outcome": "pass",
                "assertions": ["checkpoint_observed", "owned_process_cleanup", "purge_verified"],
            }
            for case_id in (
                "host_consent_active",
                "host_response_completed_before_shutdown",
                "host_drain_started",
            )
        ],
        "synthetic_fault_matrix": [
            {
                "caseId": "queue_capacity_coupled",
                "faultMechanism": "queue_capacity_coupled",
                "outcome": "pass",
                "assertions": _COUPLED_CAPACITY_ASSERTIONS,
            },
            {
                "caseId": "deny_filter",
                "faultMechanism": "deny_filter",
                "outcome": "pass",
                "assertions": ["purge_verified", "rejected_source_absent"],
            },
            {
                "caseId": "clock_rollback",
                "faultMechanism": "clock_rollback",
                "outcome": "pass",
                "assertions": ["purge_verified", "purged"],
            },
            {
                "caseId": "sqlite_injected_fault",
                "faultMechanism": "sqlite_injected_fault",
                "outcome": "pass",
                "assertions": ["purge_verified", "rejected_source_absent"],
            },
            {
                "caseId": "writer_drain_blocked",
                "faultMechanism": "writer_drain_blocked",
                "outcome": "pass",
                "assertions": ["purge_verified"],
            },
        ],
        "windows_filesystem_matrix": [
            {
                "caseId": "writer_lease_contended",
                "faultMechanism": "writer_lease_contended",
                "outcome": "pass",
                "assertions": ["adjacent_decoys_preserved", "owned_process_cleanup"],
            },
            {
                "caseId": "windows_dacl_denied",
                "faultMechanism": "windows_dacl_denied",
                "outcome": "pass",
                "assertions": [
                    "adjacent_decoys_preserved",
                    "dacl_restored",
                    "owned_process_cleanup",
                ],
            },
            {
                "caseId": "windows_read_only",
                "faultMechanism": "windows_read_only",
                "outcome": "pass",
                "assertions": ["adjacent_decoys_preserved", "owned_process_cleanup"],
            },
            {
                "caseId": "purge_lease_contended",
                "faultMechanism": "purge_lease_contended",
                "outcome": "pass",
                "assertions": ["adjacent_decoys_preserved", "owned_process_cleanup"],
            },
        ],
    }
    cases_by_scenario["spool_crash_matrix"] = [
        {
            "caseId": f"{failpoint}@exit{exit_mode}",
            "faultMechanism": failpoint,
            "exitMode": exit_mode,
            "sentinelStateAtRecoveryEntry": _spool_sentinel_at_entry(failpoint),
            "outcome": "pass",
            "assertions": ["purge_verified"],
        }
        for failpoint in _SPOOL_FAILPOINTS
        for exit_mode in (197, 198)
    ]
    for scenario in report["scenarios"]:
        proof_class, assertions = _SCENARIO_MATRIX[scenario["scenarioId"]]
        scenario["proofClass"] = proof_class
        scenario["machineAssertions"] = assertions
        scenario["caseResults"] = cases_by_scenario.get(scenario["scenarioId"], [])
    report["scenarios"][15]["measurements"] = {
        "queueRecordCount": 64,
        "queueCanonicalBytes": 193497,
        "queuePhysicalCount": 59,
        "maxQueueRecords": 64,
        "maxQueueCanonicalBytes": 2097152,
        "maxQueuePhysicalItems": 64,
        "maxCanonicalRecordBytes": 32768,
    }
    return report


def test_qualification_report_schema_closes_governed_scenario_case_capacity_matrix() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    validator = Draft202012Validator(_load_canonical_schema(QUALIFICATION_REPORT_SCHEMA))
    report = _governed_qualification_report()
    assert not list(validator.iter_errors(report))

    missing_scenario = deepcopy(report)
    missing_scenario["scenarios"].pop()
    assert list(validator.iter_errors(missing_scenario))

    duplicate_scenario = deepcopy(report)
    duplicate_scenario["scenarios"][-1] = deepcopy(duplicate_scenario["scenarios"][0])
    assert list(validator.iter_errors(duplicate_scenario))

    wrong_scenario_order = deepcopy(report)
    wrong_scenario_order["scenarios"][0], wrong_scenario_order["scenarios"][1] = (
        wrong_scenario_order["scenarios"][1],
        wrong_scenario_order["scenarios"][0],
    )
    assert list(validator.iter_errors(wrong_scenario_order))

    wrong_proof = deepcopy(report)
    wrong_proof["scenarios"][0]["proofClass"] = "packaged_process"
    assert list(validator.iter_errors(wrong_proof))

    wrong_scenario_assertions = deepcopy(report)
    wrong_scenario_assertions["scenarios"][15]["machineAssertions"].pop()
    assert list(validator.iter_errors(wrong_scenario_assertions))

    wrong_case_family = deepcopy(report)
    wrong_case_family["scenarios"][15]["caseResults"] = []
    assert list(validator.iter_errors(wrong_case_family))

    wrong_case_order = deepcopy(report)
    wrong_case_order["scenarios"][15]["caseResults"] = list(
        reversed(wrong_case_order["scenarios"][15]["caseResults"])
    )
    assert list(validator.iter_errors(wrong_case_order))

    wrong_case_binding = deepcopy(report)
    wrong_case_binding["scenarios"][15]["caseResults"][0]["faultMechanism"] = "deny_filter"
    assert list(validator.iter_errors(wrong_case_binding))

    unsorted_case_assertions = deepcopy(report)
    unsorted_case_assertions["scenarios"][15]["caseResults"][0]["assertions"] = list(
        reversed(_COUPLED_CAPACITY_ASSERTIONS)
    )
    assert list(validator.iter_errors(unsorted_case_assertions))

    missing_coupled_measurement = deepcopy(report)
    del missing_coupled_measurement["scenarios"][15]["measurements"]["queuePhysicalCount"]
    assert list(validator.iter_errors(missing_coupled_measurement))

    obsolete_capacity_alias = deepcopy(report)
    obsolete_capacity_alias["scenarios"][15]["measurements"]["queueBytesFull"] = 1
    assert list(validator.iter_errors(obsolete_capacity_alias))

    obsolete_synthetic_alias = deepcopy(report)
    obsolete_synthetic_alias["scenarios"][15]["caseResults"][0]["caseId"] = "queue_bytes_full"
    obsolete_synthetic_alias["scenarios"][15]["caseResults"][0]["faultMechanism"] = (
        "queue_bytes_full"
    )
    assert list(validator.iter_errors(obsolete_synthetic_alias))


def test_qualification_report_schema_hardens_canonical_scalar_patterns_and_integer_types() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    schema = _load_canonical_schema(QUALIFICATION_REPORT_SCHEMA)
    pattern_guard = "^(?![\\s\\S]*[\\r\\n])"

    def walk(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [value, *[item for child in value.values() for item in walk(child)]]
        if isinstance(value, list):
            return [item for child in value for item in walk(child)]
        return []

    patterned = [value for value in walk(schema) if "pattern" in value]
    assert patterned
    assert all(value["pattern"].startswith(pattern_guard) for value in patterned)

    printable_fields = [
        schema["$defs"]["EnvironmentV1"]["properties"]["cpuModel"],
        schema["$defs"]["EnvironmentV1"]["properties"]["pythonFullVersion"],
        schema["$defs"]["ProviderV1"]["properties"]["codexExecutableVersion"],
        schema["$defs"]["ProviderV1"]["properties"]["kokoroDistributionVersion"],
    ]
    printable_pattern = r"^(?![\s\S]*[\r\n])(?=[ -~]+\Z)\S(?:[ -~]*\S)?$"
    assert all(value["pattern"] == printable_pattern for value in printable_fields)
    validator = Draft202012Validator(schema)
    for property_schema in printable_fields:
        property_validator = validator.evolve(schema=property_schema)
        assert not list(property_validator.iter_errors("identity with interior spaces"))
        for invalid in (
            " identity",
            "identity ",
            "identity\tvalue",
            "identity\x00value",
            "identity\x7fvalue",
            "identity\r",
            "identity\n",
            "identity\r\n",
        ):
            assert list(property_validator.iter_errors(invalid)), repr(invalid)

    integer_fields = [value for value in walk(schema) if value.get("type") == "integer"]
    assert integer_fields
    for property_schema in integer_fields:
        property_validator = validator.evolve(schema=property_schema)
        for boolean in (False, True):
            assert list(property_validator.iter_errors(boolean)), property_schema
