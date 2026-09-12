"""Execute the source-bound Hermes v0.20 PluginManager qualification.

The completed receipt proves cleanup of this qualifier's worker, child stages,
and disposable roots.  The installed runtime may remain operational; final
qualification separately requires its matching completed-runtime receipt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from weakref import WeakKeyDictionary

from scripts import candidate_wheel as wheels
from scripts.qualification_candidate_files import (
    _candidate_files_for_consumer,
    candidate_file_metadata,
)
from scripts.qualification_dependency_files import _dependency_binding_for_consumer
from scripts.qualification_execution_files import (
    _execution_files_for_consumer,
    execution_file_metadata,
    owned_execution_files,
)
from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes
from scripts.qualification_hermes_source import (
    BoundHermesPublisherSourceV1,
    HermesPublisherFileMetadataV1,
    _bound_hermes_execution_inputs,
    _bound_hermes_source_for_consumer,
    hermes_publisher_file_metadata,
)
from scripts.qualification_installation import _workspace
from scripts.qualification_owned_work import (
    OwnedQualificationCleanupError,
    OwnedQualificationWorkV1,
)
from scripts.qualification_runtime_environment import (
    InstalledRuntimeEnvironmentV1,
    RuntimeEnvironmentMetadataV1,
    _installed_runtime_for_consumer,
    _retained_facts,
    _Runtime,
    installed_runtime_metadata,
)
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    ToolInvocationMetadataV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)

_PURPOSE = "hermes_v020_pluginmanager_runtime"
_REPORT = "hermes-pluginmanager.json"
_MAX_REPORT_BYTES = 16 * 1024
_INSTALLER_DIST_INFO = frozenset({"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"})
_SOURCE_PROFILE_DIRECTORIES = (
    "cron",
    "sessions",
    "logs",
    "logs/curator",
    "memories",
    "pairing",
    "hooks",
    "image_cache",
    "audio_cache",
    "skills",
)
_PROFILE_DIRECTORIES = tuple(sorted((*_SOURCE_PROFILE_DIRECTORIES, "plugins")))
_PROFILE_OUTPUT_FILES = frozenset({"config.yaml"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class HermesPluginManagerResultV1:
    active_profile_unchanged: bool
    default_root_unchanged: bool
    disabled_entry_point_discovered: bool
    disabled_import_or_registration: bool
    distribution_version: str
    enabled_entry_point_discovered: bool
    evidence_root_unchanged: bool
    exact_config_delta: bool
    plugin_context_registration_observed: bool


@dataclass(frozen=True, slots=True)
class HermesPluginManagerRuntimeMetadataV1:
    qualification_input_sha256: str
    candidate_commit: str
    candidate_tree: str
    source: HermesPublisherFileMetadataV1
    runtime: RuntimeEnvironmentMetadataV1
    worker_sha256: str
    harness_sha256: str
    report_sha256: str
    invocation: ToolInvocationMetadataV1
    result: HermesPluginManagerResultV1


class CompletedHermesPluginManagerRuntimeV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("completed Hermes PluginManager receipts are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Completed:
    source: BoundHermesPublisherSourceV1
    runtime: _Runtime
    invocation: CompletedToolInvocationV1
    execution: OwnedQualificationWorkV1
    metadata: HermesPluginManagerRuntimeMetadataV1


_COMPLETED: WeakKeyDictionary[CompletedHermesPluginManagerRuntimeV1, _Completed] = (
    WeakKeyDictionary()
)


def _metadata(
    source: BoundHermesPublisherSourceV1, runtime: InstalledRuntimeEnvironmentV1
) -> tuple[HermesPublisherFileMetadataV1, RuntimeEnvironmentMetadataV1]:
    bound_source = _bound_hermes_source_for_consumer(source)
    source_metadata = hermes_publisher_file_metadata(source)
    installed = _installed_runtime_for_consumer(runtime)
    runtime_metadata = installed_runtime_metadata(runtime)
    dependency = _dependency_binding_for_consumer(installed.dependencies)
    candidate = candidate_file_metadata(dependency.candidate)
    _require(
        runtime_metadata.purpose == _PURPOSE,
        "Hermes PluginManager installed runtime purpose differs",
    )
    _require(
        bound_source.files is dependency.files
        and (
            source_metadata.qualification_input_sha256,
            source_metadata.source.candidate_commit,
            source_metadata.source.candidate_tree,
            source_metadata.source.candidate_archive_sha256,
        )
        == (
            runtime_metadata.qualification_input_sha256,
            runtime_metadata.source_commit,
            runtime_metadata.source_tree,
            candidate.source_archive_sha256,
        ),
        "Hermes publisher source and installed runtime differ",
    )
    return source_metadata, runtime_metadata


def _strict_object(raw: bytes) -> dict[str, Any]:
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_REPORT_BYTES, "report is unbounded")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        _require(len(dict(items)) == len(items), "report contains duplicate fields")
        return dict(items)

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("Hermes PluginManager report is invalid") from None
    _require(
        type(value) is dict
        and json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n" == raw,
        "Hermes PluginManager report is noncanonical",
    )
    return cast(dict[str, Any], value)


def _worker_failure(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > 4096:
        return None
    try:
        value = _strict_object(path.read_bytes())
    except (OSError, ValueError):
        return None
    if (
        set(value) == {"error", "module", "pid", "stage", "version"}
        and value["version"] == 1
        and type(value["pid"]) is int
        and value["pid"] > 0
        and value["error"] in {"ModuleNotFoundError", "StageFailure"}
        and value["stage"] in {"disabled", "enable", "enabled"}
        and (
            value["module"] is None
            or (
                type(value["module"]) is str
                and value["module"].replace("_", "a").replace(".", "a").isalnum()
                and len(value["module"]) <= 128
            )
        )
    ):
        suffix = f":{value['module']}" if value["module"] is not None else ""
        return f"{value['stage']} ({value['error']}{suffix})"
    return None


def _stage(value: object, keys: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == keys, f"Hermes {label} stage differs")
    assert isinstance(value, dict)
    _require(
        all(
            type(item) in {bool, str}
            and not (type(item) is str and (not item or len(item) > 128))
            for item in value.values()
        ),
        f"Hermes {label} stage value differs",
    )
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class _CandidatePlugin:
    dist_info: str
    immutable_dist_info: tuple[str, ...]
    module_sha256: str
    metadata_sha256: str
    entry_points_sha256: str
    inventory_sha256: str


def _candidate_plugin(runtime: _Runtime) -> _CandidatePlugin:
    dependency = _dependency_binding_for_consumer(runtime.dependencies)
    candidate = _candidate_files_for_consumer(dependency.candidate)
    wheel = wheels._wheel_for_consumer(candidate.wheels[0], candidate.archive, candidate.identity)
    members = {
        name: (hashlib.sha256(payload).hexdigest(), len(payload))
        for name, payload in wheel.members.items()
    }
    metadata = [name for name in members if name.endswith(".dist-info/METADATA")]
    entry_points = [name for name in members if name.endswith(".dist-info/entry_points.txt")]
    _require(
        len(metadata) == len(entry_points) == 1
        and "hermes_realtime/hermes_plugin.py" in members,
        "candidate PluginManager distribution provenance is incomplete",
    )
    dist_info = metadata[0].rsplit("/", 1)[0]
    _require(
        dist_info == "hermes_realtime-0.0.3.dist-info"
        and entry_points[0].rsplit("/", 1)[0] == dist_info,
        "candidate PluginManager distribution identity differs",
    )
    immutable = tuple(
        sorted(
            name
            for name in members
            if name.startswith(dist_info + "/")
            and name.rsplit("/", 1)[-1] not in _INSTALLER_DIST_INFO
        )
    )
    _require(bool(immutable), "candidate immutable dist-info inventory is empty")
    installed = {
        name: (digest, size)
        for name, digest, size in execution_file_metadata(runtime.files)
    }
    _require(
        all(installed.get(name) == members[name] for name in immutable)
        and installed.get("hermes_realtime/hermes_plugin.py")
        == members["hermes_realtime/hermes_plugin.py"],
        "installed candidate PluginManager distribution differs",
    )
    inventory = {name: members[name][0] for name in immutable}
    return _CandidatePlugin(
        dist_info,
        immutable,
        members["hermes_realtime/hermes_plugin.py"][0],
        members[metadata[0]][0],
        members[entry_points[0]][0],
        hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def _tree(root: Path) -> tuple[tuple[str, str, int], ...]:
    root_metadata = root.lstat()
    _require(
        stat.S_ISDIR(root_metadata.st_mode)
        and not root.is_symlink()
        and not getattr(root_metadata, "st_file_attributes", 0) & 0x400,
        "disposable PluginManager root is redirected",
    )
    selected: list[tuple[Path, os.stat_result]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                _require(
                    len(selected) < 1024,
                    "disposable PluginManager root exceeds its bound",
                )
                path = Path(entry.path)
                metadata = path.lstat()
                _require(
                    not entry.is_symlink()
                    and not getattr(metadata, "st_file_attributes", 0) & 0x400,
                    "disposable PluginManager root is redirected",
                )
                _require(
                    stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode),
                    "disposable PluginManager root contains a non-ordinary entry",
                )
                selected.append((path, metadata))
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)

    rows: list[tuple[str, str, int]] = []
    total = 0
    for path, discovered in sorted(
        selected,
        key=lambda item: (
            item[0].relative_to(root).as_posix().casefold(),
            item[0].relative_to(root).as_posix(),
        ),
    ):
        metadata = path.lstat()
        _require(
            (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size)
            == (discovered.st_dev, discovered.st_ino, discovered.st_mode, discovered.st_size),
            "disposable PluginManager root changed while it was read",
        )
        name = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            rows.append((name + "/", "directory", 0))
            continue
        _require(
            metadata.st_size <= 1024**2 and total + metadata.st_size <= 4 * 1024**2,
            "disposable PluginManager root exceeds its bound",
        )
        raw = path.read_bytes()
        _require(
            len(raw) == metadata.st_size,
            "disposable PluginManager root changed while it was read",
        )
        total += len(raw)
        rows.append((name, hashlib.sha256(raw).hexdigest(), len(raw)))
    return tuple(rows)


def _bootstrap_profile(profile: Path, source: dict[str, bytes]) -> tuple[tuple[str, str, int], ...]:
    _require(_tree(profile) == (), "PluginManager disposable profile is not fresh")
    module = source.get("hermes_cli/default_soul.py")
    config_module = source.get("hermes_cli/config.py")
    plugins_module = source.get("hermes_cli/plugins_cmd.py")
    _require(
        type(module) is bytes
        and 0 < len(module) <= 64 * 1024
        and type(config_module) is bytes
        and 0 < len(config_module) <= 1024**2
        and type(plugins_module) is bytes
        and 0 < len(plugins_module) <= 1024**2,
        "Hermes profile bootstrap source is unavailable",
    )
    assert isinstance(module, bytes) and isinstance(config_module, bytes)
    assert isinstance(plugins_module, bytes)
    tree = ast.parse(module.decode("utf-8"), filename="hermes_cli/default_soul.py")
    assignments = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "DEFAULT_SOUL_MD"
        and isinstance(node.value, ast.Constant)
        and type(node.value.value) is str
    ]
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "DEFAULT_SOUL_MD"
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    _require(
        len(assignments) == len(writes) == 1,
        "Hermes default soul source differs",
    )
    config_tree = ast.parse(config_module.decode("utf-8"), filename="hermes_cli/config.py")
    ensure = [
        node
        for node in config_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "ensure_hermes_home"
    ]
    directory_literals: list[tuple[str, ...]] = []
    if len(ensure) == 1:
        for node in ast.walk(ensure[0]):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "subdir"
                and isinstance(node.iter, (ast.Tuple, ast.List))
                and all(
                    isinstance(item, ast.Constant) and type(item.value) is str
                    for item in node.iter.elts
                )
                ):
                directory_literals.append(
                    tuple(
                        cast(str, cast(ast.Constant, item).value)
                        for item in node.iter.elts
                    )
                )
    _require(
        directory_literals == [_SOURCE_PROFILE_DIRECTORIES],
        "Hermes profile directory bootstrap source differs",
    )
    plugins_tree = ast.parse(
        plugins_module.decode("utf-8"), filename="hermes_cli/plugins_cmd.py"
    )
    plugins_functions = [
        node
        for node in plugins_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_plugins_dir"
    ]
    plugins_function = plugins_functions[0] if len(plugins_functions) == 1 else None
    plugin_body = list(plugins_function.body) if plugins_function is not None else []
    if (
        plugin_body
        and isinstance(plugin_body[0], ast.Expr)
        and isinstance(plugin_body[0].value, ast.Constant)
        and type(plugin_body[0].value.value) is str
    ):
        plugin_body.pop(0)
    plugin_assignment, plugin_mkdir, plugin_return = (
        plugin_body if len(plugin_body) == 3 else (None, None, None)
    )
    _require(
        isinstance(plugin_assignment, ast.Assign)
        and len(plugin_assignment.targets) == 1
        and isinstance(plugin_assignment.targets[0], ast.Name)
        and plugin_assignment.targets[0].id == "plugins"
        and isinstance(plugin_assignment.value, ast.BinOp)
        and isinstance(plugin_assignment.value.op, ast.Div)
        and isinstance(plugin_assignment.value.left, ast.Call)
        and isinstance(plugin_assignment.value.left.func, ast.Name)
        and plugin_assignment.value.left.func.id == "get_hermes_home"
        and not plugin_assignment.value.left.args
        and not plugin_assignment.value.left.keywords
        and isinstance(plugin_assignment.value.right, ast.Constant)
        and plugin_assignment.value.right.value == "plugins"
        and isinstance(plugin_mkdir, ast.Expr)
        and isinstance(plugin_mkdir.value, ast.Call)
        and isinstance(plugin_mkdir.value.func, ast.Attribute)
        and isinstance(plugin_mkdir.value.func.value, ast.Name)
        and plugin_mkdir.value.func.value.id == "plugins"
        and plugin_mkdir.value.func.attr == "mkdir"
        and not plugin_mkdir.value.args
        and {
            keyword.arg: keyword.value.value
            for keyword in plugin_mkdir.value.keywords
            if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
        }
        == {"parents": True, "exist_ok": True}
        and len(plugin_mkdir.value.keywords) == 2
        and isinstance(plugin_return, ast.Return)
        and isinstance(plugin_return.value, ast.Name)
        and plugin_return.value.id == "plugins",
        "Hermes plugins directory source differs",
    )
    soul = assignments[0].encode("utf-8")
    _require(0 < len(soul) <= 64 * 1024, "Hermes default soul bytes are unbounded")
    for relative in _PROFILE_DIRECTORIES:
        profile.joinpath(*relative.split("/")).mkdir()
    (profile / "SOUL.md").write_bytes(soul)
    expected = tuple((name + "/", "directory", 0) for name in _PROFILE_DIRECTORIES) + (
        ("SOUL.md", hashlib.sha256(soul).hexdigest(), len(soul)),
    )
    baseline = _tree(profile)
    _require(
        baseline == expected,
        "Hermes disposable profile bootstrap differs",
    )
    return baseline


def _exact_profile(
    baseline: tuple[tuple[str, str, int], ...],
    current: tuple[tuple[str, str, int], ...],
) -> bool:
    previous = {name: (digest, size) for name, digest, size in baseline}
    observed = {name: (digest, size) for name, digest, size in current}
    additions = set(observed) - set(previous)
    return (
        len(observed) == len(current)
        and all(observed.get(name) == value for name, value in previous.items())
        and additions == _PROFILE_OUTPUT_FILES
    )


def _result(
    raw: bytes,
    *,
    pid: int,
    source: Path,
    packages: Path,
    candidate: _CandidatePlugin,
    roots_unchanged: bool,
    exact_profile: bool,
) -> HermesPluginManagerResultV1:
    report = _strict_object(raw)
    _require(
        set(report) == {"pid", "stages", "version"}
        and type(report["pid"]) is int
        and report["pid"] == pid
        and report["version"] == 1
        and type(report["stages"]) is dict
        and set(report["stages"]) == {"disabled", "enable", "enabled"},
        "Hermes PluginManager report identity differs",
    )
    stages = report["stages"]
    assert isinstance(stages, dict)
    disabled = _stage(
        stages["disabled"],
        {"discovered", "importOrRegistration", "sourceOriginSha256"},
        "disabled",
    )
    enabled_config = _stage(
        stages["enable"], {"argvExact", "exactConfigDelta", "stdinClosed"}, "enable"
    )
    enabled = _stage(
        stages["enabled"],
        {
            "contextObserved",
            "discovered",
            "distributionVersion",
            "distEntryPointsContentSha256",
            "distInfoInventorySha256",
            "distMetadataContentSha256",
            "distOriginSha256",
            "moduleContentSha256",
            "moduleOriginSha256",
        },
        "enabled",
    )
    expected_source = hashlib.sha256(str(source).encode()).hexdigest()
    module = packages / "hermes_realtime" / "hermes_plugin.py"
    dist_info = packages / candidate.dist_info
    _require(
        disabled
        == {
            "discovered": True,
            "importOrRegistration": False,
            "sourceOriginSha256": expected_source,
        }
        and enabled_config
        == {"argvExact": True, "exactConfigDelta": True, "stdinClosed": True}
        and enabled
        == {
            "contextObserved": True,
            "discovered": True,
            "distributionVersion": "0.0.3",
            "distEntryPointsContentSha256": candidate.entry_points_sha256,
            "distInfoInventorySha256": candidate.inventory_sha256,
            "distMetadataContentSha256": candidate.metadata_sha256,
            "distOriginSha256": hashlib.sha256(str(dist_info).encode()).hexdigest(),
            "moduleContentSha256": candidate.module_sha256,
            "moduleOriginSha256": hashlib.sha256(str(module).encode()).hexdigest(),
        },
        "Hermes PluginManager stage facts differ from parent authority",
    )
    _require(roots_unchanged and exact_profile, "PluginManager disposable roots changed")
    return HermesPluginManagerResultV1(True, True, True, False, "0.0.3", True, True, True, True)


def qualify_hermes_pluginmanager_runtime(
    work: OwnedQualificationWorkV1,
    source: BoundHermesPublisherSourceV1,
    runtime: InstalledRuntimeEnvironmentV1,
) -> CompletedHermesPluginManagerRuntimeV1:
    """Run the three archived stages and mint facts only after owned cleanup."""
    _require(type(work) is OwnedQualificationWorkV1, "PluginManager work owner type differs")
    work._accepting()
    source_metadata, runtime_metadata = _metadata(source, runtime)
    installed = _installed_runtime_for_consumer(runtime)
    candidate = _candidate_plugin(installed)
    execution = OwnedQualificationWorkV1()
    try:
        with execution:
            source_contents, harness, worker = _bound_hermes_execution_inputs(source)
            source_inventory = tuple(
                sorted(
                    (name, hashlib.sha256(payload).hexdigest(), len(payload))
                    for name, payload in source_contents.items()
                )
            )
            source_files = execution.enter(owned_execution_files(source_contents))
            harness_files = execution.enter(owned_execution_files({"harness.py": harness}))
            worker_files = execution.enter(owned_execution_files({"worker.py": worker}))
            source_root = _execution_files_for_consumer(source_files)
            harness_root = _execution_files_for_consumer(harness_files)
            worker_root = _execution_files_for_consumer(worker_files)
            packages = _execution_files_for_consumer(installed.files)
            workspace = execution.enter(_workspace())
            profile = workspace / "profile"
            protected = tuple(workspace / name for name in ("default", "active", "evidence"))
            profile.mkdir()
            for root in protected:
                root.mkdir()
            profile_before = _bootstrap_profile(profile, source_contents)
            del source_contents
            before = tuple(_tree(root) for root in protected)
            report = workspace / _REPORT
            try:
                invocation = execution.run_tool(
                    installed.tools,
                    "build_python",
                    (
                        str(worker_root / "worker.py"),
                        str(harness_root / "harness.py"),
                        str(source_root),
                        str(packages),
                        str(profile),
                        str(protected[0]),
                        str(report),
                        json.dumps(candidate.immutable_dist_info, separators=(",", ":")),
                    ),
                    workspace,
                    timeout_milliseconds=600_000,
                )
            except BaseException as error:
                failure = _worker_failure(report)
                if failure is not None:
                    raise ValueError(f"PluginManager worker failed at {failure}") from error
                raise
            report_seals = execution.enter(retain_file_seals(workspace, (_REPORT,)))
            raw = sealed_file_bytes(report_seals, _REPORT, _MAX_REPORT_BYTES)
            process = _tool_invocation_for_consumer(invocation).process
            profile_tree = _tree(profile)
            exact_profile = _exact_profile(profile_before, profile_tree)
            result = _result(
                raw,
                pid=process.pid,
                source=source_root,
                packages=packages,
                candidate=candidate,
                roots_unchanged=before == tuple(_tree(root) for root in protected),
                exact_profile=exact_profile,
            )
            _require(
                execution_file_metadata(source_files) == source_inventory
                and _execution_files_for_consumer(installed.files) == packages,
                "PluginManager execution inputs changed",
            )
            _metadata(source, runtime)
            report_sha256 = hashlib.sha256(raw).hexdigest()
    except OwnedQualificationCleanupError as error:
        work._retain_failure(error)
        raise
    except BaseException as error:
        work._retain_failure(error)
        raise
    _require(
        execution._closed and execution._unrecoverable is None,
        "PluginManager execution cleanup is incomplete",
    )
    metadata = HermesPluginManagerRuntimeMetadataV1(
        source_metadata.qualification_input_sha256,
        source_metadata.source.candidate_commit,
        source_metadata.source.candidate_tree,
        source_metadata,
        runtime_metadata,
        hashlib.sha256(worker).hexdigest(),
        hashlib.sha256(harness).hexdigest(),
        report_sha256,
        tool_invocation_metadata(invocation),
        result,
    )
    receipt = object.__new__(CompletedHermesPluginManagerRuntimeV1)
    _COMPLETED[receipt] = _Completed(source, installed, invocation, execution, metadata)
    return receipt


def _completed(receipt: CompletedHermesPluginManagerRuntimeV1) -> _Completed:
    if type(receipt) is not CompletedHermesPluginManagerRuntimeV1:
        raise TypeError("completed Hermes PluginManager receipt type differs")
    _require(receipt in _COMPLETED, "completed Hermes PluginManager receipt is unregistered")
    value = _COMPLETED[receipt]
    _require(
        value.execution._closed and value.execution._unrecoverable is None,
        "PluginManager execution cleanup is incomplete",
    )
    source = hermes_publisher_file_metadata(value.source)
    runtime = _retained_facts(value.runtime)
    invocation = tool_invocation_metadata(value.invocation)
    _require(
        value.runtime.work._closing == value.runtime.work._closed
        and value.runtime.work._unrecoverable is None,
        "PluginManager installed runtime cleanup is incomplete",
    )
    _require(
        source == value.metadata.source
        and runtime == value.metadata.runtime
        and invocation == value.metadata.invocation
        and invocation.exit_code == 0,
        "completed PluginManager authority differs",
    )
    return value


def hermes_pluginmanager_runtime_metadata(
    receipt: CompletedHermesPluginManagerRuntimeV1,
) -> HermesPluginManagerRuntimeMetadataV1:
    return _completed(receipt).metadata


def hermes_pluginmanager_result(
    receipt: CompletedHermesPluginManagerRuntimeV1,
) -> dict[str, object]:
    value = _completed(receipt).metadata.result
    return {
        "activeProfileUnchanged": value.active_profile_unchanged,
        "defaultRootUnchanged": value.default_root_unchanged,
        "disabledEntryPointDiscovered": value.disabled_entry_point_discovered,
        "disabledImportOrRegistration": value.disabled_import_or_registration,
        "distributionVersion": value.distribution_version,
        "enabledEntryPointDiscovered": value.enabled_entry_point_discovered,
        "evidenceRootUnchanged": value.evidence_root_unchanged,
        "exactConfigDelta": value.exact_config_delta,
        "pluginContextRegistrationObserved": value.plugin_context_registration_observed,
    }
