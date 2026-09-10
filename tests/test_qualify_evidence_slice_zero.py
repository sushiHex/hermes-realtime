"""Focused REDâ†’GREEN tests for the Task 12 qualification input verifier."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest


def _canonical(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _ref(root: Path, path: Path, role: str) -> dict[str, object]:
    value = path.read_bytes()
    return {
        "role": role,
        "relativePath": path.relative_to(root).as_posix(),
        "basename": path.name,
        "sha256": hashlib.sha256(value).hexdigest(),
        "bytes": len(value),
    }


def _write_json(path: Path, document: object) -> None:
    _write(path, _canonical(document))


FILE_ROLES = (
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


WHEELHOUSE_PURPOSES = (
    ("build", "build_wheelhouse_manifest"),
    ("realtime_windows_direct_runtime", "windows_direct_runtime_wheelhouse_manifest"),
    (
        "realtime_windows_sdist_built_runtime",
        "windows_sdist_built_runtime_wheelhouse_manifest",
    ),
    ("realtime_linux_runtime", "linux_runtime_wheelhouse_manifest"),
    ("hermes_v020_pluginmanager_runtime", "hermes_runtime_wheelhouse_manifest"),
)

EXPECTED_SCHEMA_ROLES = {
    "benchmarkMachineSchemaSha256": "benchmark_machine_schema",
    "benchmarkReportSchemaSha256": "benchmark_report_schema",
    "wheelhouseManifestSchemaSha256": "wheelhouse_manifest_schema",
    "qualificationInputSchemaSha256": "qualification_input_schema",
    "qualificationReportSchemaSha256": "qualification_report_schema",
    "releaseManifestSchemaSha256": "release_manifest_schema",
}

REQUESTED_SCENARIO_IDS = (
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


def _make_input_root(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    root = tmp_path / "qualification-input"
    root.mkdir()
    paths: dict[str, Path] = {}
    for role in FILE_ROLES:
        path = root / "files" / f"{role}.bin"
        _write(path, f"{role}\n".encode("ascii"))
        paths[role] = path

    runner = tmp_path / "archived-candidate" / "scripts" / "qualify_evidence_slice_zero.py"
    _write(runner, paths["qualification_runner"].read_bytes())

    chrome_root = root / "chrome" / "123.0.0.1"
    chrome_executable = chrome_root / "chrome.exe"
    chrome_resource = chrome_root / "resources.pak"
    _write(chrome_executable, b"chrome executable")
    _write(chrome_resource, b"chrome resource")
    chrome_manifest = chrome_root / "manifest.json"
    _write_json(
        chrome_manifest,
        {
            "schemaVersion": 1,
            "chromeVersion": "123.0.0.1",
            "files": [
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in sorted(
                    (("chrome.exe", chrome_executable), ("resources.pak", chrome_resource))
                )
            ],
        },
    )
    paths["chrome_executable"] = chrome_executable
    paths["chrome_version_directory_manifest"] = chrome_manifest

    for provider in ("moonshine", "kokoro"):
        provider_root = root / "providers" / provider
        resource = provider_root / "model.bin"
        _write(resource, f"{provider} model".encode("ascii"))
        resource_hash = hashlib.sha256(resource.read_bytes()).hexdigest()
        manifest = provider_root / "model-manifest.json"
        _write_json(
            manifest,
            {
                "schemaVersion": 1,
                "provider": provider,
                "modelIdentitySha256": hashlib.sha256(
                    json.dumps(
                        {"model.bin": resource_hash},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "resources": [
                    {
                        "name": "model.bin",
                        "bytes": resource.stat().st_size,
                        "sha256": resource_hash,
                    }
                ],
            },
        )
        paths[f"{provider}_model_manifest"] = manifest

    reviewed_schema = (
        Path(__file__).parents[1] / "scripts" / "schemas" / "qualification-input-v1.schema.json"
    )
    _write(root / "schemas" / reviewed_schema.name, reviewed_schema.read_bytes())
    paths["qualification_input_schema"] = root / "schemas" / reviewed_schema.name

    for purpose, role in WHEELHOUSE_PURPOSES:
        wheel_dir = root / "wheelhouses" / purpose / "wheels"
        wheel = wheel_dir / f"{purpose}-1.0.whl"
        _write(wheel, f"wheel for {purpose}".encode("ascii"))
        wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
        requirements = wheel_dir.parent / "requirements.txt"
        constraints = wheel_dir.parent / "constraints.txt"
        _write(requirements, f"package==1 --hash=sha256:{wheel_hash}\n".encode("ascii"))
        _write(constraints, b"package==1\n")
        manifest = wheel_dir.parent / "manifest.json"
        _write_json(
            manifest,
            {
                "schemaVersion": 1,
                "purpose": purpose,
                "pythonVersion": "3.11.15",
                "platform": "any",
                "requirements": _ref(root, requirements, "requirements"),
                "constraints": _ref(root, constraints, "constraints"),
                "wheels": [_ref(root, wheel, "wheel")],
            },
        )
        paths[role] = manifest

    tool_identities = []
    for role, artifact_role in (
        ("build_python", "build_python_executable"),
        ("git", "git_executable"),
        ("hatchling", "hatchling_wheel"),
        ("uv", "uv_executable"),
    ):
        artifact = root / "tools" / f"{role}.bin"
        _write(artifact, f"{role} tool".encode("ascii"))
        tool_identities.append(
            {
                "role": role,
                "version": "1.27.0" if role == "hatchling" else "1.0",
                "artifact": _ref(root, artifact, artifact_role),
            }
        )

    files = sorted(
        (_ref(root, paths[role], role) for role in FILE_ROLES),
        key=lambda item: item["role"],
    )
    plan = paths["governing_plan"]
    archive = paths["candidate_source_archive"]
    manifest = root / "qualification-input-v1.json"
    document = {
        "schemaVersion": 1,
        "candidate": {
            "baselineCommit": "a" * 40,
            "candidateCommit": "b" * 40,
            "tree": "c" * 40,
            "canonicalDiffSha256": "d" * 64,
            "version": "0.0.3",
        },
        "files": files,
        "toolIdentities": tool_identities,
        "expected": {
            "pythonFullVersion": "3.11.15",
            "pythonArchitecture": "AMD64",
            "chromeVersion": "123.0.0.1",
            "livekitVersion": "1.0",
            "codexVersion": "1.0",
            "moonshineVersion": "1.0",
            "kokoroVersion": "1.0",
            "codexModel": "gpt-5",
            "codexEffort": "medium",
            "sourceArchivePrefix": "hermes-realtime-0.0.3/",
            **{
                expected_key: next(file["sha256"] for file in files if file["role"] == direct_role)
                for expected_key, direct_role in EXPECTED_SCHEMA_ROLES.items()
            },
        },
        "requestedScenarioIds": list(REQUESTED_SCENARIO_IDS),
    }
    _write_json(manifest, document)
    actual_plan = tmp_path / "governing-plan.md"
    actual_archive = tmp_path / "candidate-source.tar.gz"
    _write(actual_plan, plan.read_bytes())
    _write(actual_archive, archive.read_bytes())
    return root, manifest, actual_plan, actual_archive, runner


def _load_runner() -> Any:
    from scripts import qualify_evidence_slice_zero

    return qualify_evidence_slice_zero


SCENARIO_IDS_V1 = (
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


def _candidate_binding(module: Any) -> Any:
    return module.CandidateBindingV1("a" * 64, "b" * 40, "c" * 40)


def _attempt(module: Any, ordinal: int = 0) -> Any:
    return module.ScenarioAttemptIdentityV1(
        tuple(module.ScenarioIdV1)[ordinal],
        ordinal,
        _candidate_binding(module),
        101,
        202,
    )


def test_scenario_registry_has_exact_frozen_order_and_no_raw_protocol() -> None:
    module = _load_runner()
    assert module._REQUESTED_SCENARIO_IDS == SCENARIO_IDS_V1
    assert tuple(value.value for value in module.ScenarioIdV1) == SCENARIO_IDS_V1
    assert not hasattr(module, "ProducerRoleV1")
    assert not hasattr(module, "RawRecordV1")
    assert not hasattr(module, "RawScenarioObservationV1")


def test_candidate_and_attempt_identity_are_exact_and_fail_closed() -> None:
    module = _load_runner()
    binding = _candidate_binding(module)
    assert binding.qualification_input_sha256 == "a" * 64
    invalid_bindings = (
        (True, "b" * 40, "c" * 40),
        ("A" * 64, "b" * 40, "c" * 40),
        ("a" * 63, "b" * 40, "c" * 40),
        ("a" * 64, "g" * 40, "c" * 40),
        ("a" * 64, "b" * 40, "c" * 39),
    )
    for values in invalid_bindings:
        with pytest.raises((TypeError, ValueError)):
            module.CandidateBindingV1(*values)
    assert _attempt(module).scenario_id is module.ScenarioIdV1.DETERMINISTIC_EQUIVALENCE
    for attempt_values in (
        ("deterministic_equivalence", 0, binding, 101, 202),
        (module.ScenarioIdV1.DETERMINISTIC_EQUIVALENCE, True, binding, 101, 202),
        (module.ScenarioIdV1.DETERMINISTIC_EQUIVALENCE, 1, binding, 101, 202),
        (module.ScenarioIdV1.DETERMINISTIC_EQUIVALENCE, 0, object(), 101, 202),
        (module.ScenarioIdV1.DETERMINISTIC_EQUIVALENCE, 0, binding, True, 202),
        (module.ScenarioIdV1.DETERMINISTIC_EQUIVALENCE, 0, binding, 101, 0),
    ):
        with pytest.raises((TypeError, ValueError)):
            module.ScenarioAttemptIdentityV1(*attempt_values)


def test_unavailable_registry_is_exact_closed_and_returns_nothing() -> None:
    module = _load_runner()
    registry = module.UNAVAILABLE_SCENARIO_REGISTRY_V1
    assert type(registry) is tuple
    assert tuple(item.scenario_id.value for item in registry) == (
        *SCENARIO_IDS_V1[1:10], *SCENARIO_IDS_V1[14:],
    )
    assert module.validate_unavailable_scenario_registry_v1(registry) is registry
    for registration in registry:
        ordinal = SCENARIO_IDS_V1.index(registration.scenario_id.value)
        with pytest.raises(module.ProducerUnavailableV1) as raised:
            module.invoke_unavailable_scenario_v1(registration, _attempt(module, ordinal))
        assert raised.value.scenario_id is registration.scenario_id
        assert raised.value.args == (f"producer unavailable: {registration.scenario_id.value}",)


def test_unavailable_registry_rejects_open_or_confused_registration() -> None:
    module = _load_runner()
    registry = module.UNAVAILABLE_SCENARIO_REGISTRY_V1
    with pytest.raises(TypeError):
        module.validate_unavailable_scenario_registry_v1(list(registry))
    with pytest.raises(ValueError):
        module.validate_unavailable_scenario_registry_v1(registry[:-1])
    confused = (registry[1], registry[0], *registry[2:])
    with pytest.raises(ValueError):
        module.validate_unavailable_scenario_registry_v1(confused)
    reconstructed = tuple(
        module.UnavailableScenarioRegistrationV1(
            item.scenario_id,
            module._UnavailableScenarioProducerV1(item.scenario_id),
        )
        for item in registry
    )
    assert all(
        rebuilt is not canonical for rebuilt, canonical in zip(reconstructed, registry, strict=True)
    )
    with pytest.raises(ValueError):
        module.validate_unavailable_scenario_registry_v1(reconstructed)
    with pytest.raises((TypeError, ValueError)):
        module.invoke_unavailable_scenario_v1(reconstructed[0], _attempt(module, 0))
    with pytest.raises((TypeError, ValueError)):
        module.invoke_unavailable_scenario_v1(registry[1], _attempt(module, 0))


def _verify(
    runner: Any,
    root: Path,
    manifest: Path,
    plan: Path,
    archive: Path,
    archived_runner: Path,
) -> None:
    runner.verify_qualification_input_closure(
        qualification_input_root=root,
        qualification_input_manifest=manifest,
        expected_qualification_input_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        plan=plan,
        candidate_source_archive=archive,
        runner_path=archived_runner,
    )


def test_strict_canonical_json_accepts_only_exact_compact_sorted_utf8_with_one_lf() -> None:
    runner = _load_runner()
    document = {"alpha": [1, True, "Ï€"], "zeta": {"value": 2}}
    raw = _canonical(document)

    assert runner.load_strict_canonical_json(raw, source="fixture") == document

    for malformed in (
        b'{"zeta":{"value":2},"alpha":[1,true,"\xcf\x80"]}\n',
        b'{"alpha":[1,true,"\\u03c0"],"zeta":{"value":2}}\n',
        b'{"alpha":[1,true,"\xcf\x80"],"zeta":{"value":2}}',
        b'\xef\xbb\xbf{"alpha":1}\n',
        b'{"alpha":1,"alpha":2}\n',
        b'{"alpha":1.0}\n',
        b'{"alpha":NaN}\n',
        b'{"alpha":null}\n',
    ):
        with pytest.raises(runner.QualificationInputError):
            runner.load_strict_canonical_json(malformed, source="fixture")


def test_input_closure_revalidates_direct_and_transitive_artifacts_before_execution(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    expected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    closure = runner.verify_qualification_input_closure(
        qualification_input_root=root,
        qualification_input_manifest=manifest,
        expected_qualification_input_sha256=expected_digest,
        plan=plan,
        candidate_source_archive=archive,
        runner_path=archived_runner,
    )

    logical_ids = [artifact.logical_id for artifact in closure.verified_artifacts]
    assert logical_ids == sorted(logical_ids)
    assert "file:governing_plan" in logical_ids
    assert "tool:git" in logical_ids
    assert "wheelhouse:build:wheel:build-1.0.whl" in logical_ids
    assert "provider:moonshine:resource:model.bin" in logical_ids
    assert "chrome:file:chrome.exe" in logical_ids


def test_input_closure_rejects_schema_invalid_empty_requested_scenarios(tmp_path: Path) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["requestedScenarioIds"] = []
    _write_json(manifest, document)

    with pytest.raises(runner.QualificationInputError):
        _verify(runner, root, manifest, plan, archive, archived_runner)


def test_input_closure_rejects_schema_valid_wrong_requested_scenario_order(tmp_path: Path) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["requestedScenarioIds"][0], document["requestedScenarioIds"][1] = (
        document["requestedScenarioIds"][1],
        document["requestedScenarioIds"][0],
    )
    _write_json(manifest, document)

    with pytest.raises(runner.QualificationInputError):
        _verify(runner, root, manifest, plan, archive, archived_runner)


def test_input_closure_rejects_schema_valid_unsorted_files(tmp_path: Path) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["files"][0], document["files"][1] = document["files"][1], document["files"][0]
    _write_json(manifest, document)

    with pytest.raises(runner.QualificationInputError):
        _verify(runner, root, manifest, plan, archive, archived_runner)


def test_input_closure_rejects_schema_valid_unsorted_tool_identities(tmp_path: Path) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["toolIdentities"][0], document["toolIdentities"][1] = (
        document["toolIdentities"][1],
        document["toolIdentities"][0],
    )
    _write_json(manifest, document)

    with pytest.raises(runner.QualificationInputError):
        _verify(runner, root, manifest, plan, archive, archived_runner)


@pytest.mark.parametrize("expected_key,direct_role", EXPECTED_SCHEMA_ROLES.items())
def test_input_closure_rejects_expected_schema_hash_not_bound_to_direct_ref(
    tmp_path: Path, expected_key: str, direct_role: str
) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    direct_hash = next(file["sha256"] for file in document["files"] if file["role"] == direct_role)
    document["expected"][expected_key] = "0" * 64 if direct_hash != "0" * 64 else "1" * 64
    _write_json(manifest, document)

    with pytest.raises(runner.QualificationInputError):
        _verify(runner, root, manifest, plan, archive, archived_runner)


@pytest.mark.parametrize("mutation", ("direct", "provider", "chrome", "unlisted_wheel"))
def test_input_closure_fails_closed_on_direct_or_transitive_input_drift(
    tmp_path: Path, mutation: str
) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    expected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    if mutation == "direct":
        _write(plan, b"changed plan")
    elif mutation == "provider":
        _write(root / "providers" / "moonshine" / "model.bin", b"changed model")
    elif mutation == "chrome":
        _write(root / "chrome" / "123.0.0.1" / "resources.pak", b"changed resource")
    else:
        _write(root / "wheelhouses" / "build" / "wheels" / "unlisted.whl", b"unlisted")

    with pytest.raises(runner.QualificationInputError):
        runner.verify_qualification_input_closure(
            qualification_input_root=root,
            qualification_input_manifest=manifest,
            expected_qualification_input_sha256=expected_digest,
            plan=plan,
            candidate_source_archive=archive,
            runner_path=archived_runner,
        )


def test_input_closure_rejects_reparse_observation_in_wheel_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    linked_wheel = root / "wheelhouses" / "build" / "wheels" / "linked.whl"
    _write(linked_wheel, b"simulated linked wheel")
    original = runner._is_reparse_or_link
    monkeypatch.setattr(
        runner,
        "_is_reparse_or_link",
        lambda path: path == linked_wheel or original(path),
    )

    with pytest.raises(runner.QualificationInputError, match="reparse point or symlink"):
        _verify(runner, root, manifest, plan, archive, archived_runner)


def test_input_closure_rejects_path_escape_and_duplicate_transitive_path(tmp_path: Path) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["files"][0]["relativePath"] = "../outside.bin"
    _write_json(manifest, document)
    expected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    with pytest.raises(runner.QualificationInputError):
        runner.verify_qualification_input_closure(
            qualification_input_root=root,
            qualification_input_manifest=manifest,
            expected_qualification_input_sha256=expected_digest,
            plan=plan,
            candidate_source_archive=archive,
            runner_path=archived_runner,
        )


def test_input_closure_binds_chrome_manifest_version_to_its_directory_leaf(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    root, manifest, plan, archive, archived_runner = _make_input_root(tmp_path)
    chrome_manifest = root / "chrome" / "123.0.0.1" / "manifest.json"
    chrome_document = json.loads(chrome_manifest.read_text(encoding="utf-8"))
    chrome_document["chromeVersion"] = "123.0.0.2"
    _write_json(chrome_manifest, chrome_document)

    input_document = json.loads(manifest.read_text(encoding="utf-8"))
    chrome_ref = next(
        ref for ref in input_document["files"] if ref["role"] == "chrome_version_directory_manifest"
    )
    chrome_ref.update(_ref(root, chrome_manifest, "chrome_version_directory_manifest"))
    _write_json(manifest, input_document)
    expected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    with pytest.raises(runner.QualificationInputError):
        runner.verify_qualification_input_closure(
            qualification_input_root=root,
            qualification_input_manifest=manifest,
            expected_qualification_input_sha256=expected_digest,
            plan=plan,
            candidate_source_archive=archive,
            runner_path=archived_runner,
        )


# Task-12 t12-3: deterministic Win32 process-authority substrate only.
class _FakeWindowsKernel:
    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.events: list[tuple[Any, ...]] = []
        self.members = [42]
        self.fail: set[str] = set()
        self.fail_close: set[int] = set()
        self.launch_value: Any = None
        self.job_handle: Any = 101
        self.active_count: Any = 0
        self.identities: dict[int, Any] = {
            201: self._identity(42, 7, 420, "root.exe", "a" * 64),
            203: self._identity(43, 42, 430, "child.exe", "b" * 64),
        }
        self.opened = {42: 201, 43: 203}

    def _identity(
        self,
        pid: int,
        parent_pid: int,
        creation: int,
        image: str,
        digest: str,
        scenario: str | None = None,
        parent_creation: int | None = None,
    ) -> Any:
        if parent_creation is None:
            parent_creation = {7: 70, 42: 420, 43: 430}.get(parent_pid, 1)
        return self.runner._WindowsKernelProcessV1(
            pid, parent_pid, parent_creation, creation, image, digest, scenario
        )

    def create_job(self, limits: Any) -> int:
        self.events.append(("create_job", limits))
        if "create_job" in self.fail:
            raise OSError("create_job")
        return self.job_handle

    def create_process_suspended(
        self, command: Any, environment: Any, cwd: str, handles: Any, flags: int
    ) -> Any:
        self.events.append(("create_process", command, environment, cwd, handles, flags))
        if "create_process" in self.fail:
            raise OSError("create_process")
        if self.launch_value is not None:
            return self.launch_value
        return self.runner._WindowsKernelLaunchV1(42, 201, 202)

    def assign_process_to_job(self, job: int, process: int) -> None:
        self.events.append(("assign", job, process))
        if "assign" in self.fail:
            raise OSError("assign")

    def resume_thread(self, thread: int) -> None:
        self.events.append(("resume", thread))
        if "resume" in self.fail:
            raise OSError("resume")

    def query_process_identity(self, handle: int) -> Any:
        self.events.append(("identity", handle))
        if f"identity:{handle}" in self.fail:
            raise OSError("identity")
        return self.identities[handle]

    def query_job_processes(self, job: int) -> tuple[int, ...]:
        self.events.append(("members", job))
        if "members" in self.fail:
            raise OSError("members")
        return tuple(self.members)

    def open_process(self, pid: int, access: int) -> int:
        self.events.append(("open", pid, access))
        if f"open:{pid}" in self.fail:
            raise OSError("open")
        return self.opened[pid]

    def terminate_process(self, handle: int) -> None:
        self.events.append(("terminate_process", handle))
        if "terminate_process" in self.fail:
            raise OSError("terminate_process")

    def terminate_job(self, job: int) -> None:
        self.events.append(("terminate_job", job))
        if "terminate_job" in self.fail:
            raise OSError("terminate_job")
        self.members = []

    def wait(self, handle: int, timeout_ms: int) -> None:
        self.events.append(("wait", handle, timeout_ms))
        if f"wait:{handle}" in self.fail:
            raise OSError("wait")

    def query_job_active_process_count(self, job: int) -> int:
        self.events.append(("active", job))
        if "active" in self.fail:
            raise OSError("active")
        return self.active_count

    def close_handle(self, handle: int) -> None:
        self.events.append(("close", handle))
        if handle in self.fail_close:
            raise OSError(f"close:{handle}")


def _job_spec(runner: Any, **changes: Any) -> Any:
    values = {
        "scenario_id": "synthetic_fault_matrix",
        "command": ("C:\\candidate\\python.exe", "-I", "worker.py"),
        "environment": (("LANG", "C"), ("PYTHONUTF8", "1")),
        "working_directory": "C:\\candidate",
        "timeout_milliseconds": 2_000,
        "inherited_handles": (11, 12),
        "limits": runner._WindowsJobLimitsV1(4, 1_024, 2_048),
    }
    values.update(changes)
    return runner._WindowsScenarioSpecV1(**values)


def _job(
    runner: Any,
    kernel: _FakeWindowsKernel,
    *,
    runner_identity: Any = None,
    role_rules: Any = None,
    **changes: Any,
) -> Any:
    rules = role_rules or (
        runner._WindowsRoleRuleV1("root", "root.exe", "a" * 64, frozenset(), True),
        runner._WindowsRoleRuleV1(
            "descendant", "child.exe", "b" * 64, frozenset({"root", "descendant"}), False
        ),
    )
    return runner._WindowsScenarioJobV1(
        kernel,
        _job_spec(runner, **changes),
        runner_identity or runner._WindowsRunnerIdentityV1(7, 70),
        rules,
    )


_UNBOUNDED_JOB_LIMITS_SENTINEL = object()
_INHERITED_HANDLE_TUPLE_SUBCLASS = type("_InheritedHandleTuple", (tuple,), {})


@pytest.mark.parametrize(
    "changes",
    (
        {"scenario_id": "bad id"},
        {"timeout_milliseconds": 0},
        {"command": ()},
        {"environment": (("A", "1"), ("A", "2"))},
        {"environment": (("PATH", "two"), ("Path", "one"))},
        {"working_directory": "relative"},
        {"working_directory": "C:\\candidate\\..\\escape"},
        {"inherited_handles": ()},
        {"inherited_handles": (11, 11)},
        {"inherited_handles": _INHERITED_HANDLE_TUPLE_SUBCLASS((11, 12))},
        {"limits": None},
        {"limits": _UNBOUNDED_JOB_LIMITS_SENTINEL},
    ),
)
def test_windows_job_validates_every_input_before_native_action(changes: dict[str, Any]) -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    if changes.get("limits") is _UNBOUNDED_JOB_LIMITS_SENTINEL:
        changes = {
            **changes,
            "limits": runner._WindowsJobLimitsV1(4_097, 1_024, 2_048),
        }
    with pytest.raises(runner._WindowsScenarioJobError):
        _job(runner, kernel, **changes)
    assert kernel.events == []


def test_windows_job_launches_suspended_with_exact_allowlist_assigns_then_resumes() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    root = _job(runner, kernel).launch_root()
    assert [event[0] for event in kernel.events][:5] == [
        "create_job",
        "create_process",
        "identity",
        "assign",
        "resume",
    ]
    create = kernel.events[1]
    assert create[4] == (11, 12)
    assert (
        create[5]
        == runner._CREATE_SUSPENDED
        | runner._CREATE_NEW_PROCESS_GROUP
        | runner._CREATE_UNICODE_ENVIRONMENT
        | runner._EXTENDED_STARTUPINFO_PRESENT
    )
    assert root.identity.pid == 42


def test_windows_assignment_failure_preserves_cleanup_failure_and_retry_authority() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.fail.update({"assign", "wait:201"})
    job = _job(runner, kernel)
    with pytest.raises(BaseExceptionGroup) as raised:
        job.launch_root()
    assert any("assign" in str(error) for error in raised.value.exceptions)
    assert any(
        isinstance(error, runner._WindowsFinalizationError) for error in raised.value.exceptions
    )
    assert ("terminate_process", 201) in kernel.events
    assert ("close", 201) not in kernel.events
    kernel.fail.remove("wait:201")
    result = job.finalize()
    assert result.closed is True
    assert ("close", 201) in kernel.events


def test_windows_job_assignment_failure_directly_terminates_waits_and_closes_unassigned_root() -> (
    None
):
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.fail.add("assign")
    with pytest.raises(OSError, match="assign"):
        _job(runner, kernel).launch_root()
    names = [event[0] for event in kernel.events]
    assert "terminate_job" in names
    assert names.index("terminate_process") < names.index("wait")
    assert ("terminate_process", 201) in kernel.events
    assert ("wait", 201, 2_000) in kernel.events
    assert ("wait", 202, 2_000) in kernel.events
    assert ("active", 101) in kernel.events
    assert {event for event in kernel.events if event[0] == "close"} >= {
        ("close", 101),
        ("close", 201),
        ("close", 202),
    }


def test_windows_job_resume_failure_finalizes_assigned_root_without_bare_pid_termination() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.fail.add("resume")
    with pytest.raises(OSError, match="resume"):
        _job(runner, kernel).launch_root()
    assert ("terminate_job", 101) in kernel.events
    assert all(event[0] != "terminate_process_pid" for event in kernel.events)


@pytest.mark.parametrize(
    "bad_member", ("pid_reuse", "cross_scenario", "unexpected_role", "unresolved_parent")
)
def test_windows_membership_rejects_conflicting_or_unclassified_members(bad_member: str) -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    if bad_member == "pid_reuse":
        kernel.identities[201] = kernel._identity(42, 7, 999, "root.exe", "a" * 64)
    elif bad_member == "cross_scenario":
        kernel.members = [42, 43]
        kernel.identities[203] = kernel._identity(43, 42, 430, "child.exe", "b" * 64, "other")
    elif bad_member == "unexpected_role":
        kernel.members = [42, 43]
        kernel.identities[203] = kernel._identity(43, 42, 430, "evil.exe", "c" * 64)
    else:
        kernel.members = [42, 43]
        kernel.identities[203] = kernel._identity(43, 999, 430, "child.exe", "b" * 64)
    with pytest.raises(runner._WindowsScenarioJobError):
        job.checkpoint("late-descendant")


def test_windows_membership_resolves_child_first_kernel_order() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.opened[44] = 204
    kernel.identities[204] = kernel._identity(44, 43, 440, "child.exe", "b" * 64)
    kernel.members = [44, 43, 42]
    snapshot = job.checkpoint("child-first")
    assert [(member.identity.pid, member.role) for member in snapshot.members] == [
        (42, "root"),
        (43, "descendant"),
        (44, "descendant"),
    ]


def test_windows_membership_identity_failure_retains_handle_for_job_cleanup() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.members = [42, 43]
    kernel.fail.add("identity:203")
    with pytest.raises(OSError, match="identity"):
        job.checkpoint("identity-failure")
    kernel.fail.remove("identity:203")
    result = job.finalize()
    assert result.closed is True
    assert ("terminate_job", 101) in kernel.events
    assert ("wait", 203, 2_000) in kernel.events
    assert ("close", 203) in kernel.events


def test_windows_membership_binds_parent_identity_role_and_late_descendant() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.members = [42, 43]
    snapshot = job.checkpoint("late-descendant")
    assert [(member.identity.pid, member.role) for member in snapshot.members] == [
        (42, "root"),
        (43, "descendant"),
    ]
    assert (
        "open",
        43,
        runner._SYNCHRONIZE | runner._PROCESS_QUERY_LIMITED_INFORMATION,
    ) in kernel.events


def test_windows_finalization_retries_failed_handle_after_all_microphases() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    job.retain_auxiliary_handle(301, "stdout")
    job.retain_auxiliary_handle(302, "console")
    kernel.fail_close.add(202)
    with pytest.raises(runner._WindowsFinalizationError) as raised:
        job.finalize()
    first = raised.value.result
    assert first.pre_cleanup_snapshot is not None and first.zero_active_observed is True
    assert 202 in first.failed_handles
    assert all(("close", handle) in kernel.events for handle in (202, 201, 301, 302, 101))
    assert kernel.events.index(("active", 101)) < kernel.events.index(("close", 101))
    kernel.fail_close.clear()
    result = job.finalize()
    assert result.closed is True and result.zero_active_observed is True
    assert [event for event in kernel.events if event[0] == "active"] == [("active", 101)]


def test_windows_finalization_retains_job_until_zero_active_is_proven() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.fail.add("active")
    with pytest.raises(runner._WindowsFinalizationError) as raised:
        job.finalize()
    assert 101 in raised.value.result.failed_handles
    assert raised.value.result.zero_active_observed is False
    assert ("close", 101) not in kernel.events
    kernel.fail.remove("active")
    result = job.finalize()
    assert result.zero_active_observed is True
    assert result.closed is True
    assert ("close", 101) in kernel.events


def test_windows_finalization_retains_wait_failure_until_bounded_retry() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.fail.add("wait:201")
    with pytest.raises(runner._WindowsFinalizationError) as raised:
        job.finalize()
    assert 201 in raised.value.result.failed_handles
    assert ("close", 201) not in kernel.events
    assert raised.value.result.zero_active_observed is True
    kernel.fail.remove("wait:201")
    result = job.finalize()
    assert result.closed is True
    assert [event for event in kernel.events if event[0] == "wait" and event[1] == 201] == [
        ("wait", 201, 2_000),
        ("wait", 201, 2_000),
    ]
    assert ("close", 201) in kernel.events


def test_windows_finalization_waits_after_identity_failure_and_preserves_dual_error() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.fail.add("identity:201")
    kernel.fail_close.add(202)
    with pytest.raises(runner._WindowsFinalizationError):
        job.finalize()
    assert ("wait", 201, 2_000) in kernel.events
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.fail_close.add(202)
    with pytest.raises(ExceptionGroup) as raised:
        runner._run_with_windows_scenario_job_finalization_v1(
            job, lambda: (_ for _ in ()).throw(ValueError("body"))
        )
    assert any(isinstance(error, ValueError) for error in raised.value.exceptions)
    assert any(
        isinstance(error, runner._WindowsFinalizationError) for error in raised.value.exceptions
    )


def test_ctypes_windows_job_layouts_match_documented_64_bit_offsets() -> None:
    runner = _load_runner()
    assert ctypes.sizeof(runner._JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1) == 144
    assert runner._JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1.BasicLimitInformation.offset == 0
    assert runner._JOBOBJECT_BASIC_LIMIT_INFORMATION_V1.LimitFlags.offset == 16
    assert runner._JOBOBJECT_BASIC_LIMIT_INFORMATION_V1.ActiveProcessLimit.offset == 40
    assert runner._JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1.ProcessMemoryLimit.offset == 112
    assert runner._JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1.JobMemoryLimit.offset == 120
    assert ctypes.sizeof(runner._JOBOBJECT_BASIC_ACCOUNTING_INFORMATION_V1) == 48
    assert runner._JOBOBJECT_BASIC_ACCOUNTING_INFORMATION_V1.ActiveProcesses.offset == 40
    assert ctypes.sizeof(runner._JOBOBJECT_BASIC_PROCESS_ID_LIST_HEADER_V1) == 8
    assert runner._JOBOBJECT_BASIC_PROCESS_ID_LIST_HEADER_V1.NumberOfProcessIdsInList.offset == 4


def test_windows_job_pid_list_buffer_is_closed_and_limit_bounded() -> None:
    runner = _load_runner()
    assert runner._job_pid_list_buffer_bytes_v1(4) == 8 + 4 * ctypes.sizeof(ctypes.c_size_t)
    assert runner._job_pid_list_buffer_bytes_v1(4_096) == 8 + 4_096 * ctypes.sizeof(ctypes.c_size_t)
    for value in (True, 0, 4_097):
        with pytest.raises(runner._WindowsScenarioJobError):
            runner._job_pid_list_buffer_bytes_v1(value)


def test_ctypes_windows_api_configures_all_used_prototypes() -> None:
    runner = _load_runner()

    class _Function:
        argtypes: object = None
        restype: object = None

    class _Api:
        pass

    names = (
        "CreateJobObjectW",
        "SetInformationJobObject",
        "CloseHandle",
        "InitializeProcThreadAttributeList",
        "UpdateProcThreadAttribute",
        "DeleteProcThreadAttributeList",
        "CreateProcessW",
        "AssignProcessToJobObject",
        "ResumeThread",
        "CreateToolhelp32Snapshot",
        "Process32FirstW",
        "Process32NextW",
        "GetProcessTimes",
        "QueryFullProcessImageNameW",
        "GetProcessId",
        "QueryInformationJobObject",
        "OpenProcess",
        "TerminateProcess",
        "TerminateJobObject",
        "WaitForSingleObject",
    )
    api = _Api()
    for name in names:
        setattr(api, name, _Function())
    runner._configure_windows_api_v1(api)
    assert api.CreateJobObjectW.restype is ctypes.c_void_p
    assert api.CreateToolhelp32Snapshot.restype is ctypes.c_void_p
    assert api.OpenProcess.restype is ctypes.c_void_p
    for name in names:
        assert getattr(api, name).argtypes is not None


def test_ctypes_windows_kernel_is_import_safe_and_fails_closed_off_windows() -> None:
    runner = _load_runner()
    kernel = runner._CtypesWindowsKernelV1(platform="posix")
    with pytest.raises(runner._WindowsPlatformError):
        kernel.create_job(runner._WindowsJobLimitsV1(1, 1, 1))


def test_windows_membership_rejects_runner_pid_reuse_even_when_pid_matches() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.identities[201] = kernel._identity(42, 7, 420, "root.exe", "a" * 64, parent_creation=71)
    with pytest.raises(BaseExceptionGroup) as raised:
        _job(runner, kernel).launch_root()
    assert "runner identity" in str(raised.value.exceptions[0])


def test_windows_membership_rejects_descendant_parent_pid_reuse_even_when_pid_matches() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.members = [42, 43]
    kernel.identities[203] = kernel._identity(
        43, 42, 430, "child.exe", "b" * 64, parent_creation=421
    )
    with pytest.raises(runner._WindowsScenarioJobError, match="parent identity"):
        job.checkpoint("pid-reuse")


@pytest.mark.parametrize(
    "launch",
    (
        None,
        "wrong-dto",
        "bad-pid",
        "bad-process",
        "bad-thread",
        "duplicate-handles",
    ),
)
def test_windows_launch_creation_or_malformed_dto_finalizes_all_plausible_authority(
    launch: str | None,
) -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.members = []
    if launch is None:
        kernel.fail.add("create_process")
    else:
        kernel.launch_value = {
            "wrong-dto": object(),
            "bad-pid": runner._WindowsKernelLaunchV1(0, 201, 202),
            "bad-process": runner._WindowsKernelLaunchV1(42, 0, 202),
            "bad-thread": runner._WindowsKernelLaunchV1(42, 201, 0),
            "duplicate-handles": runner._WindowsKernelLaunchV1(42, 201, 201),
        }[launch]
    job = _job(runner, kernel)
    with pytest.raises((runner._WindowsScenarioJobError, OSError, BaseExceptionGroup)):
        job.launch_root()
    assert ("close", 101) in kernel.events
    if launch in {"bad-pid", "bad-thread", "duplicate-handles"}:
        assert ("terminate_process", 201) in kernel.events
        assert ("close", 201) in kernel.events
    if launch in {"bad-pid", "bad-process"}:
        assert ("close", 202) in kernel.events


def test_windows_assignment_fallback_failure_remains_retryable() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.fail.update({"assign", "terminate_job"})
    job = _job(runner, kernel)
    with pytest.raises(BaseExceptionGroup) as raised:
        job.launch_root()
    assert any(
        isinstance(error, runner._WindowsFinalizationError) for error in raised.value.exceptions
    )
    assert ("terminate_process", 201) in kernel.events
    assert ("terminate_job", 101) in kernel.events
    assert ("close", 101) not in kernel.events
    kernel.fail.remove("terminate_job")
    assert job.finalize().closed is True


def test_windows_assignment_direct_termination_failure_retains_job_for_retry() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.fail.update({"assign", "terminate_process"})
    job = _job(runner, kernel)
    with pytest.raises(BaseExceptionGroup):
        job.launch_root()
    assert ("close", 101) not in kernel.events
    kernel.fail.remove("terminate_process")
    assert job.finalize().closed is True
    assert [event for event in kernel.events if event == ("terminate_process", 201)] == [
        ("terminate_process", 201),
        ("terminate_process", 201),
    ]


def test_windows_unsigned_invalid_handle_is_rejected_before_native_action() -> None:
    runner = _load_runner()
    sentinel = ctypes.c_void_p(-1).value
    assert type(sentinel) is int
    kernel = _FakeWindowsKernel(runner)
    with pytest.raises(runner._WindowsScenarioJobError):
        _job(runner, kernel, inherited_handles=(11, sentinel))
    assert kernel.events == []
    job = _job(runner, kernel)
    with pytest.raises(runner._WindowsScenarioJobError):
        job.retain_auxiliary_handle(sentinel, "stdout")


def test_windows_unsigned_invalid_job_handle_never_enters_authority() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    kernel.job_handle = ctypes.c_void_p(-1).value
    with pytest.raises(runner._WindowsScenarioJobError):
        _job(runner, kernel).launch_root()
    assert [event[0] for event in kernel.events] == ["create_job"]


def test_windows_unsigned_invalid_launch_handles_are_never_operated() -> None:
    runner = _load_runner()
    sentinel = ctypes.c_void_p(-1).value
    for launch in (
        runner._WindowsKernelLaunchV1(42, sentinel, 202),
        runner._WindowsKernelLaunchV1(42, 201, sentinel),
    ):
        kernel = _FakeWindowsKernel(runner)
        kernel.launch_value = launch
        with pytest.raises((runner._WindowsScenarioJobError, BaseExceptionGroup)):
            _job(runner, kernel).launch_root()
        assert not any(sentinel in event[1:] for event in kernel.events)


def test_windows_unsigned_invalid_open_process_handle_is_not_retained() -> None:
    runner = _load_runner()
    sentinel = ctypes.c_void_p(-1).value
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.members = [42, 43]
    kernel.opened[43] = sentinel
    with pytest.raises(runner._WindowsScenarioJobError):
        job.checkpoint("invalid-open")
    assert not any(sentinel in event[1:] for event in kernel.events)


def test_windows_job_setup_failure_is_finalized_by_direct_launch_lifecycle() -> None:
    runner = _load_runner()

    class SetupFailureKernel(_FakeWindowsKernel):
        def create_job(self, limits: Any) -> int:
            self.events.append(("create_job", limits))
            raise OSError("set_information_and_close")

        def finalize_transient_handles(self) -> None:
            self.events.append(("finalize_transient_handles",))

    kernel = SetupFailureKernel(runner)
    with pytest.raises(OSError, match="set_information"):
        _job(runner, kernel).launch_root()
    assert ("finalize_transient_handles",) in kernel.events


def test_windows_boolean_runner_identity_is_rejected_before_native_action() -> None:
    runner = _load_runner()
    for identity in (
        runner._WindowsRunnerIdentityV1(True, 70),
        runner._WindowsRunnerIdentityV1(7, True),
    ):
        kernel = _FakeWindowsKernel(runner)
        with pytest.raises(runner._WindowsScenarioJobError):
            _job(runner, kernel, runner_identity=identity)
        assert kernel.events == []


def test_windows_integer_root_policy_is_rejected_before_native_action() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    rules = (runner._WindowsRoleRuleV1("root", "root.exe", "a" * 64, frozenset(), 1),)
    with pytest.raises(runner._WindowsScenarioJobError):
        _job(runner, kernel, role_rules=rules)
    assert kernel.events == []


def test_windows_false_active_count_is_not_zero_active_proof() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    kernel.active_count = False
    with pytest.raises(runner._WindowsFinalizationError):
        job.finalize()
    assert ("close", 101) not in kernel.events


class _ApiFunction:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: Any) -> Any:
        return self.callback(*args)


class _ConcreteApiShim:
    def __init__(
        self, *, failure: str | None = None, process: int = 501, thread: int = 502
    ) -> None:
        self.failure = failure
        self.process = process
        self.thread = thread
        self.events: list[tuple[Any, ...]] = []
        self._initialize_calls = 0
        for name in (
            "CreateJobObjectW",
            "SetInformationJobObject",
            "CloseHandle",
            "InitializeProcThreadAttributeList",
            "UpdateProcThreadAttribute",
            "DeleteProcThreadAttributeList",
            "CreateProcessW",
            "CreateToolhelp32Snapshot",
            "Process32FirstW",
            "Process32NextW",
        ):
            setattr(self, name, _ApiFunction(getattr(self, f"_{name}")))

    def __getattr__(self, name: str) -> Any:
        function = _ApiFunction(lambda *_: 1)
        setattr(self, name, function)
        return function

    def _CreateJobObjectW(self, *_: Any) -> int:
        self.events.append(("create_job",))
        return 101

    def _SetInformationJobObject(self, *_: Any) -> int:
        self.events.append(("set_job",))
        return int(self.failure != "set_job")

    def _CloseHandle(self, handle: Any) -> int:
        value = int(handle.value)
        self.events.append(("close", value))
        return int(self.failure != f"close:{value}")

    def _InitializeProcThreadAttributeList(self, attributes: Any, *_: Any) -> int:
        self._initialize_calls += 1
        size = _[-1]
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t)).contents.value = 64
        self.events.append(("initialize", bool(attributes)))
        return int(self.failure != "initialize" or self._initialize_calls == 1)

    def _UpdateProcThreadAttribute(
        self,
        _attributes: Any,
        _flags: int,
        attribute: int,
        value: Any,
        size: int,
        *_: Any,
    ) -> int:
        handles = tuple(
            int(ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p))[index] or 0)
            for index in range(size // ctypes.sizeof(ctypes.c_void_p))
        )
        self.events.append(("update", attribute, size, handles))
        return int(self.failure != "update")

    def _DeleteProcThreadAttributeList(self, *_: Any) -> None:
        self.events.append(("delete",))

    def _CreateProcessW(
        self,
        _application: Any,
        _command: Any,
        _sa_process: Any,
        _sa_thread: Any,
        inherit: Any,
        flags: int,
        _environment: Any,
        _cwd: Any,
        _startup: Any,
        process: Any,
    ) -> int:
        self.events.append(("create_process", bool(inherit), int(flags)))
        if self.failure == "create_process":
            return 0
        info = ctypes.cast(process, ctypes.POINTER(_load_runner()._PROCESS_INFORMATION_V1)).contents
        info.hProcess = self.process
        info.hThread = self.thread
        info.dwProcessId = 42
        return 1

    def _CreateToolhelp32Snapshot(self, *_: Any) -> int:
        self.events.append(("snapshot",))
        return ctypes.c_void_p(-1).value if self.failure == "snapshot" else 900

    def _Process32FirstW(self, *_: Any) -> int:
        self.events.append(("first",))
        return 0

    def _Process32NextW(self, *_: Any) -> int:
        self.events.append(("next",))
        return 0


def _concrete_kernel(runner: Any, api: _ConcreteApiShim) -> Any:
    kernel = runner._CtypesWindowsKernelV1(platform="nt")
    kernel._kernel32 = api
    runner._configure_windows_api_v1(api)
    return kernel


def test_ctypes_invalid_toolhelp_sentinel_fails_before_process32_calls() -> None:
    runner = _load_runner()
    api = _ConcreteApiShim(failure="snapshot")
    with pytest.raises(OSError):
        _concrete_kernel(runner, api)._parent_pid(42)
    assert [event[0] for event in api.events] == ["snapshot"]


def test_ctypes_parent_identity_uses_exact_parent_handle_and_retries_failed_transient_close() -> (
    None
):
    runner = _load_runner()
    api = _ConcreteApiShim()
    kernel = _concrete_kernel(runner, api)
    kernel._parent_pid = lambda _pid: 7
    api.OpenProcess = _ApiFunction(lambda _access, _inherit, pid: 700 if pid == 7 else 0)
    api.GetProcessId = _ApiFunction(lambda _handle: 7)

    def process_times(_handle: Any, creation: Any, *_: Any) -> int:
        ctypes.cast(creation, ctypes.POINTER(ctypes.c_uint64)).contents.value = 70
        return 1

    api.GetProcessTimes = _ApiFunction(process_times)
    api.failure = "close:700"
    with pytest.raises(OSError):
        kernel._parent_identity(42)
    assert kernel.retryable_handles == (700,)
    api.failure = None
    kernel.finalize_transient_handles()
    assert kernel.retryable_handles == ()


def test_ctypes_set_information_and_close_failure_preserve_retry_authority() -> None:
    runner = _load_runner()
    api = _ConcreteApiShim(failure="set_job")
    api.failure = "close:101"
    # The shim needs both calls to fail; its callback is deliberately stateful.
    api._SetInformationJobObject = lambda *_: (api.events.append(("set_job",)), 0)[1]
    api.SetInformationJobObject.callback = api._SetInformationJobObject
    kernel = _concrete_kernel(runner, api)
    with pytest.raises(BaseExceptionGroup):
        kernel.create_job(runner._WindowsJobLimitsV1(1, 1, 1))
    assert ("close", 101) in api.events
    assert kernel.retryable_handles == (101,)
    api.failure = None
    kernel.finalize_transient_handles()
    assert kernel.retryable_handles == ()


@pytest.mark.parametrize(
    "handles",
    (
        (),
        (11, 11),
        (11, True),
        (11, ctypes.c_void_p(-1).value),
    ),
)
def test_ctypes_invalid_allowlist_is_rejected_before_any_api_event(
    handles: tuple[int, ...],
) -> None:
    runner = _load_runner()
    api = _ConcreteApiShim()
    with pytest.raises(runner._WindowsScenarioJobError):
        _concrete_kernel(runner, api).create_process_suspended(
            ("C:\\candidate\\python.exe",), (("LANG", "C"),), "C:\\candidate", handles, 4
        )
    assert api.events == []


def test_ctypes_handle_conversion_rejects_boolean_without_coercion() -> None:
    runner = _load_runner()
    with pytest.raises(OSError):
        runner._CtypesWindowsKernelV1._handle(True)


@pytest.mark.parametrize("failure", ("initialize", "update", "create_process"))
def test_ctypes_attribute_setup_failures_delete_initialized_list(failure: str) -> None:
    runner = _load_runner()
    api = _ConcreteApiShim(failure=failure)
    with pytest.raises(OSError):
        _concrete_kernel(runner, api).create_process_suspended(
            ("C:\\candidate\\python.exe", "-I"), (("LANG", "C"),), "C:\\candidate", (11, 12), 0xA55
        )
    if failure != "initialize":
        assert ("delete",) in api.events


def test_ctypes_attribute_list_uses_exact_allowlist_inheritance_and_required_flags() -> None:
    runner = _load_runner()
    api = _ConcreteApiShim()
    launch = _concrete_kernel(runner, api).create_process_suspended(
        ("C:\\candidate\\python.exe", "-I"), (("LANG", "C"),), "C:\\candidate", (11, 12), 0xA55
    )
    assert launch == runner._WindowsKernelLaunchV1(42, 501, 502)
    assert ("update", 0x00020002, 2 * ctypes.sizeof(ctypes.c_void_p), (11, 12)) in api.events
    assert ("create_process", True, 0xA55) in api.events
    assert ("delete",) in api.events


@pytest.mark.parametrize("process,thread,expected", ((501, 0, (501,)), (0, 502, (502,))))
def test_ctypes_post_create_handle_validation_closes_every_valid_process_information_handle(
    process: int, thread: int, expected: tuple[int, ...]
) -> None:
    runner = _load_runner()
    api = _ConcreteApiShim(process=process, thread=thread)
    with pytest.raises(OSError):
        _concrete_kernel(runner, api).create_process_suspended(
            ("C:\\candidate\\python.exe",), (("LANG", "C"),), "C:\\candidate", (11,), 4
        )
    assert {event[1] for event in api.events if event[0] == "close"} >= set(expected)


def test_finalization_waits_before_rechecking_terminated_process_identity() -> None:
    runner = _load_runner()
    kernel = _FakeWindowsKernel(runner)
    job = _job(runner, kernel)
    job.launch_root()
    job.finalize()
    termination = kernel.events.index(("terminate_job", 101))
    events = kernel.events[termination:]
    assert events.index(("wait", 201, 2000)) < events.index(("identity", 201))
