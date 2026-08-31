"""Focused Task12 contracts for the private Hermes v0.20 qualification harness."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
import os
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from runpy import run_path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "qualify_hermes_v020_pluginmanager.py"


def runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qualify_hermes_v020_pluginmanager", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_archive_authority() -> ModuleType:
    path = ROOT / "scripts" / "source_archive_authority.py"
    assert path.is_file(), "shared source-archive authority module must exist"
    spec = importlib.util.spec_from_file_location("scripts.source_archive_authority", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_archive_policy_rejects_malformed_authority_values() -> None:
    authority = source_archive_authority()
    valid = {
        "prefix": "hermes-agent-v0.20.0",
        "workspace_child": ".pluginmanager-hermes-v020-source",
        "owner_marker": ".pluginmanager-source-owner",
        "max_members": 100_000,
        "max_file_bytes": 16 * 1024 * 1024,
        "max_tree_bytes": 256 * 1024 * 1024,
        "error_label": "Hermes source archive",
    }
    invalid = (
        {"prefix": "../escape"},
        {"workspace_child": "bad/name"},
        {"owner_marker": ".."},
        {"max_members": True},
        {"max_file_bytes": -1},
        {"max_tree_bytes": 0},
        {"max_file_bytes": 2, "max_tree_bytes": 1},
        {"error_label": ""},
    )

    for change in invalid:
        with pytest.raises(ValueError, match="source archive policy"):
            authority.SourceArchivePolicyV1(**(valid | change))


def test_source_archive_authority_is_shared_without_pluginmanager_policy_coupling() -> None:
    authority = source_archive_authority()
    module = runner()

    assert authority.SourceArchivePolicyV1.__module__ == "scripts.source_archive_authority"
    assert authority.WindowsSourceWorkspaceAuthorityV1.__module__ == "scripts.source_archive_authority"
    assert "ParentRequest" not in vars(authority)
    assert "PluginManager" not in vars(authority)
    assert module._validated_archive_members.__module__ == "scripts.source_archive_authority"
    assert module._ExtractedSourceV1 is authority.WindowsSourceWorkspaceAuthorityV1
    assert tuple(
        inspect.signature(authority.WindowsSourceWorkspaceAuthorityV1).parameters
    ) == ("workspace", "policy", "watcher_factory")
    expected_policy = authority.SourceArchivePolicyV1(
        prefix="hermes-agent-v0.20.0",
        workspace_child=".pluginmanager-hermes-v020-source",
        owner_marker=".pluginmanager-source-owner",
        max_members=100_000,
        max_file_bytes=16 * 1024 * 1024,
        max_tree_bytes=256 * 1024 * 1024,
        error_label="Hermes source archive",
    )
    assert expected_policy == module._SOURCE_ARCHIVE_POLICY


def valid_result() -> dict[str, object]:
    return {
        "activeProfileUnchanged": True,
        "defaultRootUnchanged": True,
        "disabledEntryPointDiscovered": True,
        "disabledImportOrRegistration": False,
        "distributionVersion": "0.0.3",
        "enabledEntryPointDiscovered": True,
        "evidenceRootUnchanged": True,
        "exactConfigDelta": True,
        "pluginContextRegistrationObserved": True,
    }


def add_tar_member(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes = b"",
    *,
    kind: bytes = tarfile.REGTYPE,
    linkname: str = "",
) -> None:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
    info.linkname = linkname
    info.size = len(payload) if kind == tarfile.REGTYPE else 0
    archive.addfile(info, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)


def write_official_source_archive(path: Path, *, member: str = "hermes_cli/plugins.py") -> Path:
    """Create the smallest safe pinned-source-shaped archive for parent tests."""
    with tarfile.open(path, "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/", kind=tarfile.DIRTYPE)
        add_tar_member(
            archive,
            "hermes-agent-v0.20.0/" + member,
            b"# official source fixture\n",
        )
    return path


def write_request_inputs(module: ModuleType, tmp_path: Path) -> tuple[object, dict[str, Path]]:
    roots = {
        name: (tmp_path / name).resolve()
        for name in ("wheelhouse", "workspace", "default", "evidence")
    }
    for path in roots.values():
        path.mkdir(parents=True)
    roots["active"] = (roots["default"] / "profiles" / "live").resolve()
    roots["active"].mkdir(parents=True)
    roots["profile"] = (roots["workspace"] / "profiles" / "gate-profile").resolve()
    roots["profile"].mkdir(parents=True)
    files = {
        "archive": write_official_source_archive((tmp_path / "hermes-v020.tar").resolve()),
        "candidate": (roots["wheelhouse"] / "candidate.whl").resolve(),
        "requirements": (roots["wheelhouse"] / "requirements.txt").resolve(),
        "constraints": (roots["wheelhouse"] / "constraints.txt").resolve(),
        "build_python": (tmp_path / "build-python.exe").resolve(),
    }
    files["candidate"].write_bytes(b"candidate-wheel")
    files["requirements"].write_bytes(b"hermes-realtime==0.0.3\n")
    files["constraints"].write_bytes(b"hermes-realtime==0.0.3\n")
    files["build_python"].write_bytes(b"python")
    manifest, manifest_digest = write_wheelhouse_manifest(files, roots["wheelhouse"])
    files["manifest"] = manifest
    request = module.ParentRequest(
        files["archive"],
        files["candidate"],
        roots["wheelhouse"],
        files["requirements"],
        files["constraints"],
        roots["workspace"],
        roots["profile"],
        roots["default"],
        roots["active"],
        roots["evidence"],
        roots["workspace"] / "result.json",
        files["build_python"],
        hashlib.sha256(files["archive"].read_bytes()).hexdigest(),
        manifest,
        manifest_digest,
    )
    return request, files


def write_wheelhouse_manifest(files: dict[str, Path], wheelhouse: Path) -> tuple[Path, str]:
    """Write the parent-owned canonical v1 inventory outside the wheelhouse."""

    def entry(role: str, path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "role": role,
            "relativePath": path.relative_to(wheelhouse).as_posix(),
            "basename": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    manifest_value = {
        "schemaVersion": 1,
        "purpose": "hermes_v020_pluginmanager_runtime",
        "pythonVersion": "3.11",
        "platform": "windows_amd64",
        "requirements": entry("requirements", files["requirements"]),
        "constraints": entry("constraints", files["constraints"]),
        "wheels": [entry("wheel", files["candidate"])],
    }
    payload = (
        json.dumps(
            manifest_value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    manifest = (wheelhouse.parent / "wheelhouse-manifest-v1.json").resolve()
    manifest.write_bytes(payload)
    return manifest, hashlib.sha256(payload).hexdigest()


def refresh_wheelhouse_manifest(request: object, files: dict[str, Path]) -> object:
    manifest, digest = write_wheelhouse_manifest(files, request.wheelhouse)
    return request._replace(wheelhouse_manifest=manifest, wheelhouse_manifest_sha256=digest)


def write_candidate_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hermes_realtime/hermes_plugin.py", b"candidate module")
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/METADATA",
            b"Name: hermes-realtime\nVersion: 0.0.3\n",
        )
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/WHEEL",
            b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/entry_points.txt",
            b"[hermes_agent.plugins]\nhermes-realtime = hermes_realtime.hermes_plugin\n",
        )
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/RECORD",
            b"installer rewrites this",
        )


def bare_windows_owner(module: ModuleType) -> object:
    owner = module.WindowsRetainedImmutableInputs.__new__(module.WindowsRetainedImmutableInputs)
    owner._kernel = module.ctypes.WinDLL("kernel32", use_last_error=True)
    owner._handles = []
    owner._watchers = []
    owner._watch_lock = threading.Lock()
    owner._changed = threading.Event()
    owner._closed = False
    owner._inventories = {}
    owner._file_seals = {}
    owner._retained = {}
    owner._request = None
    owner._bind()
    return owner


def test_obsolete_in_process_facade_is_absent() -> None:
    module = runner()
    for name in (
        "build_isolated_environment",
        "validate_isolated_roots",
        "validate_pluginmanager_result",
        "suppress_profile_file_logging",
        "invoke_exact_enable",
        "run_enablement_sequence",
        "run_pluginmanager_probe",
        "probe_installed_environment",
    ):
        assert not hasattr(module, name)


def test_private_result_protocol_is_canonical_bounded_closed_and_path_free() -> None:
    module = runner()
    payload = module.canonical_child_result(valid_result())
    assert payload.endswith(b"\n") and len(payload) <= 4096
    assert module.parse_child_result(payload) == valid_result()
    mutations = (
        payload[:-1],
        b" " + payload,
        payload.replace(b'"0.0.3"', b'"C:\\\\Users\\\\name"'),
        payload.replace(b"false", b"true"),
        payload + b"x" * 4096,
    )
    for mutation in mutations:
        with pytest.raises(ValueError):
            module.parse_child_result(mutation)


def test_allowlisted_environment_scrubs_all_ambient_hermes_python_and_profile_selectors(
    tmp_path: Path,
) -> None:
    module = runner()
    scripts = (tmp_path / "venv" / "Scripts").resolve()
    profile = (tmp_path / "workspace" / "profiles" / "gate-profile").resolve()
    scripts.mkdir(parents=True)
    profile.mkdir(parents=True)
    base = {
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "USERNAME": "gate-user",
        "USERPROFILE": r"C:\Users\gate-user",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\gate-user",
        "PATH": "ambient",
        "PYTHONPATH": "hostile",
        "PYTHONHOME": "hostile",
        "VIRTUAL_ENV": "hostile",
        "HERMES_PROFILE": "hostile",
        "HERMES_CONFIG": "hostile",
        "HERMES_ENV": "hostile",
        "HERMES_SAFE_MODE": "1",
        "HERMES_ENABLE_PROJECT_PLUGINS": "1",
        "HERMES_BUNDLED_PLUGINS": "hostile",
    }
    temporary = (tmp_path / "workspace" / "temp").resolve()
    temporary.mkdir()
    environment = module.build_child_environment(
        base=base, scripts=scripts, profile=profile, temporary=temporary
    )
    assert set(environment) == {
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "HERMES_HOME",
        "PATH",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_NO_COLOR",
        "PIP_PROGRESS_BAR",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
    assert environment["HERMES_HOME"] == str(profile)
    assert environment["PATH"].split(__import__("os").pathsep)[0] == str(scripts)
    assert environment["TEMP"] == environment["TMP"] == str(temporary)


def test_child_programs_use_nonmutating_profile_observation_and_exact_eof_checks() -> None:
    module = runner()
    programs = module_source_literals(module)
    compile(module._DISABLED_CHILD, "<disabled>", "exec")
    compile(module._ENABLE_CHILD, "<enable>", "exec")
    compile(module._ENABLED_CHILD, "<enabled>", "exec")
    assert "sys.setprofile" in module._DISABLED_CHILD
    assert "sys.setprofile" in module._ENABLED_CHILD
    assert "PluginContext.__init__=" not in programs
    assert "setattr(hp.PluginContext" not in programs
    assert 'sys.stdin.read(1)!=""' in programs
    assert "get_hermes_home" in programs and "get_config_path" in programs


def test_harness_has_no_bare_process_launcher_or_direct_config_write_fallback() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "StringIO" not in source
    assert "write_text" not in module_source_literals(runner())


def module_source_literals(module: ModuleType) -> str:
    return module._DISABLED_CHILD + module._ENABLE_CHILD + module._ENABLED_CHILD


def test_stage_parser_rejects_wrong_origin_duplicate_fields_and_noncanonical_bytes() -> None:
    module = runner()
    valid = (
        b'{"discovered":true,"importOrRegistration":false,"sourceOriginSha256":"'
        + b"a" * 64
        + b'"}\n'
    )
    assert (
        module._canonical_stage(
            valid, {"discovered", "importOrRegistration", "sourceOriginSha256"}
        )["discovered"]
        is True
    )
    for payload in (
        valid.replace(b"a" * 64, b"C:\\\\repo"),
        valid.replace(b"}\n", b',"extra":true}\n'),
        valid.replace(b",", b", ", 1),
    ):
        with pytest.raises(ValueError):
            module._canonical_stage(
                payload, {"discovered", "importOrRegistration", "sourceOriginSha256"}
            )


def test_protected_snapshot_rejects_changes_case_collisions_links_and_ads(tmp_path: Path) -> None:
    module = runner()
    root = tmp_path.resolve() / "protected"
    root.mkdir()
    (root / "config.yaml").write_text("plugins: {}\n", encoding="utf-8")
    before = module.snapshot_regular_tree(root)
    (root / "created").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unchanged"):
        module.require_tree_unchanged(before, module.snapshot_regular_tree(root))


def test_snapshot_regular_tree_reads_each_file_through_one_retained_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    root = (tmp_path / "protected").resolve()
    root.mkdir()
    payload = root / "value.bin"
    payload.write_bytes(b"original")
    original_open = Path.open
    opened = 0

    def counted_open(path: Path, *args: object, **kwargs: object):
        nonlocal opened
        if path == payload:
            opened += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    snapshot = module.snapshot_regular_tree(root)
    assert opened == 1
    assert snapshot["value.bin"][3:5] == (8, hashlib.sha256(b"original").hexdigest())
    assert snapshot["value.bin"][5] == os.path.normcase(str(payload))


def test_snapshot_regular_tree_rejects_same_byte_path_replacement_during_retained_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    root = (tmp_path / "protected").resolve()
    root.mkdir()
    payload = root / "value.bin"
    payload.write_bytes(b"identical")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"identical")
    original_open = Path.open

    class SwappingReader:
        def __init__(self, stream: object) -> None:
            self.stream = stream
            self.swapped = False

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args: object):
            return self.stream.__exit__(*args)

        def fileno(self) -> int:
            return self.stream.fileno()

        def read(self, size: int = -1) -> bytes:
            data = self.stream.read(size)
            if data and not self.swapped:
                self.swapped = True
                payload.unlink()
                replacement.replace(payload)
            return data

    def swapping_open(path: Path, *args: object, **kwargs: object):
        stream = original_open(path, *args, **kwargs)
        return SwappingReader(stream) if path == payload else stream

    monkeypatch.setattr(Path, "open", swapping_open)
    with pytest.raises((ValueError, PermissionError)):
        module.snapshot_regular_tree(root)


def test_parent_request_requires_distinct_external_owned_roots_and_initially_absent_output(
    tmp_path: Path,
) -> None:
    module = runner()
    names = ("wheelhouse", "workspace", "default", "evidence")
    roots = {name: (tmp_path / name).resolve() for name in names}
    for root in roots.values():
        root.mkdir()
    roots["active"] = (roots["default"] / "profiles" / "live").resolve()
    roots["active"].mkdir(parents=True)
    roots["profile"] = (roots["workspace"] / "profiles" / "gate-profile").resolve()
    roots["profile"].mkdir(parents=True)
    files = {
        "archive": write_official_source_archive((tmp_path / "hermes-v020.tar").resolve()),
        "build_python": (tmp_path / "build-python.exe").resolve(),
    }
    files["build_python"].write_bytes(b"python")
    for name in ("candidate", "requirements", "constraints"):
        filename = "candidate.whl" if name == "candidate" else name
        files[name] = (roots["wheelhouse"] / filename).resolve()
        files[name].write_bytes(b"x")
    manifest, manifest_digest = write_wheelhouse_manifest(files, roots["wheelhouse"])
    request = module.ParentRequest(
        files["archive"],
        files["candidate"],
        roots["wheelhouse"],
        files["requirements"],
        files["constraints"],
        roots["workspace"],
        roots["profile"],
        roots["default"],
        roots["active"],
        roots["evidence"],
        roots["workspace"] / "result.json",
        files["build_python"],
        hashlib.sha256(files["archive"].read_bytes()).hexdigest(),
        manifest,
        manifest_digest,
    )
    assert module.validate_parent_request(request) is request
    assert module.validate_parent_request(request._replace(active_profile=roots["default"]))
    with pytest.raises(ValueError, match="alias"):
        module.validate_parent_request(request._replace(active_profile=roots["evidence"]))


def test_parent_supplied_manifest_binds_the_exact_closed_wheelhouse_before_any_child(
    tmp_path: Path,
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    closure = module._verify_bound_wheelhouse_closure(request)
    assert closure["candidate"] == files["candidate"]
    assert closure["requirements"] == files["requirements"]
    assert closure["constraints"] == files["constraints"]
    assert closure["wheels"] == (files["candidate"],)
    assert module.validate_parent_request(request) is request


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_digest",
        "manifest_digest_uppercase",
        "manifest_noncanonical",
        "manifest_duplicate_key",
        "manifest_wrong_purpose",
        "manifest_bool_bytes",
        "manifest_nested_path",
        "manifest_case_collision",
        "candidate_hash",
        "candidate_not_listed",
        "listed_wheel_missing",
        "listed_nonwheel",
        "requirements_path",
        "extra_wheelhouse_file",
        "missing_wheelhouse_file",
    ),
)
def test_parent_manifest_rejects_each_closure_mutation_before_any_child(
    tmp_path: Path, mutation: str
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    value = json.loads(files["manifest"].read_text(encoding="utf-8"))
    if mutation == "manifest_digest":
        request = request._replace(wheelhouse_manifest_sha256="0" * 64)
    elif mutation == "manifest_digest_uppercase":
        request = request._replace(
            wheelhouse_manifest_sha256=request.wheelhouse_manifest_sha256.upper()
        )
    elif mutation == "manifest_noncanonical":
        files["manifest"].write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "manifest_duplicate_key":
        files["manifest"].write_bytes(
            b'{"constraints":{},"constraints":{},"platform":"windows_amd64"}\n'
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "manifest_wrong_purpose":
        value["purpose"] = "build"
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "manifest_bool_bytes":
        value["wheels"][0]["bytes"] = True
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "manifest_nested_path":
        value["wheels"][0]["relativePath"] = "nested/candidate.whl"
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "manifest_case_collision":
        value["wheels"].append(
            {**value["wheels"][0], "relativePath": "CANDIDATE.WHL", "basename": "CANDIDATE.WHL"}
        )
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "candidate_hash":
        files["candidate"].write_bytes(b"mutated candidate")
    elif mutation == "candidate_not_listed":
        extra = request.wheelhouse / "other.whl"
        extra.write_bytes(files["candidate"].read_bytes())
        value["wheels"][0]["relativePath"] = "other.whl"
        value["wheels"][0]["basename"] = "other.whl"
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "listed_wheel_missing":
        value["wheels"].append(
            {**value["wheels"][0], "relativePath": "missing.whl", "basename": "missing.whl"}
        )
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "listed_nonwheel":
        nonwheel = request.wheelhouse / "payload.bin"
        nonwheel.write_bytes(b"not a wheel")
        value["wheels"].append(
            {
                "role": "wheel",
                "relativePath": nonwheel.name,
                "basename": nonwheel.name,
                "sha256": hashlib.sha256(nonwheel.read_bytes()).hexdigest(),
                "bytes": nonwheel.stat().st_size,
            }
        )
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "requirements_path":
        value["requirements"]["relativePath"] = "constraints.txt"
        value["requirements"]["basename"] = "constraints.txt"
        files["manifest"].write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        request = request._replace(
            wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
        )
    elif mutation == "extra_wheelhouse_file":
        (request.wheelhouse / "unlisted.whl").write_bytes(b"unlisted")
    elif mutation == "missing_wheelhouse_file":
        files["constraints"].unlink()
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="wheelhouse|manifest|closure"):
        module._verify_bound_wheelhouse_closure(request)


def test_parent_manifest_python_pin_is_frozen_to_311(tmp_path: Path) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    value = json.loads(files["manifest"].read_text(encoding="utf-8"))
    value["pythonVersion"] = "3.12"
    files["manifest"].write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    request = request._replace(
        wheelhouse_manifest_sha256=hashlib.sha256(files["manifest"].read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="Python pin"):
        module._verify_bound_wheelhouse_closure(request)


def test_source_archive_uses_a_retained_random_root_without_publication_destination(
    tmp_path: Path,
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    assert module.validate_parent_request(request) is request
    extracted = module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert extracted.root.parent.parent == request.workspace
    assert extracted.root.parent.name.startswith(
        module._SOURCE_ARCHIVE_POLICY.workspace_child + "."
    )
    assert extracted.root.parent.name.endswith(".tmp")
    assert extracted.root.parent != request.workspace / ".pluginmanager-hermes-v020-source"
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()
    assert "hermes-agent-v0.20.0/hermes_cli/plugins.py" in extracted.inventory
    assert set(extracted.inventory) == {
        ".",
        ".pluginmanager-source-owner",
        "hermes-agent-v0.20.0/",
        "hermes-agent-v0.20.0/hermes_cli/",
        "hermes-agent-v0.20.0/hermes_cli/plugins.py",
    }
    extracted.assert_unchanged()
    extracted.close()
    assert not extracted.root.parent.exists()
    assert request.hermes_source_archive == files["archive"]


def test_source_archive_rejects_obsolete_prefix_before_creating_source_tree(
    tmp_path: Path,
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    with tarfile.open(files["archive"], "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20/", kind=tarfile.DIRTYPE)
        add_tar_member(archive, "hermes-agent-v0.20/hermes_cli/plugins.py", b"obsolete")
    with pytest.raises(ValueError, match="prefix"):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()


@pytest.mark.parametrize(
    "member",
    (
        "../escape.py",
        "hermes_cli/../../escape.py",
        "HERmes_cli/plugins.py",
        "hermes_cli/plugins.py.",
        "hermes_cli/plugins.py ",
        "hermes_cli/CON.py",
        "hermes_cli/plugins.py/child.py",
    ),
)
def test_source_archive_rejects_traversal_and_case_collisions_before_creating_source_tree(
    tmp_path: Path, member: str
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    with tarfile.open(files["archive"], "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/", kind=tarfile.DIRTYPE)
        add_tar_member(archive, "hermes-agent-v0.20.0/hermes_cli/plugins.py", b"first")
        add_tar_member(archive, "hermes-agent-v0.20.0/" + member, b"second")
    with pytest.raises(ValueError, match="archive"):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()


def test_source_archive_preserves_win32_unsafe_member_error_contract(tmp_path: Path) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    with tarfile.open(files["archive"], "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/", kind=tarfile.DIRTYPE)
        add_tar_member(archive, "hermes-agent-v0.20.0/con.py", b"reserved")

    with pytest.raises(
        ValueError,
        match=r"^Hermes source archive has a Win32-unsafe member component$",
    ):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not tuple(request.workspace.glob(".pluginmanager-hermes-v020-source.*.tmp"))


def test_source_archive_rejects_links_and_duplicate_members_before_creating_source_tree(
    tmp_path: Path,
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    with tarfile.open(files["archive"], "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/", kind=tarfile.DIRTYPE)
        add_tar_member(
            archive,
            "hermes-agent-v0.20.0/hermes_cli/link.py",
            kind=tarfile.SYMTYPE,
            linkname="../../outside",
        )
    with pytest.raises(ValueError, match="archive"):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()
    with tarfile.open(files["archive"], "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/", kind=tarfile.DIRTYPE)
        for payload in (b"first", b"second"):
            add_tar_member(
                archive,
                "hermes-agent-v0.20.0/hermes_cli/plugins.py",
                payload,
            )
    with pytest.raises(ValueError, match="duplicate"):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()


def test_source_archive_extraction_failure_removes_all_owned_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)

    def fail_extractfile(*_args: object, **_kwargs: object) -> None:
        raise OSError("controlled read failure")

    monkeypatch.setattr(module.tarfile.TarFile, "extractfile", fail_extractfile)
    with pytest.raises(OSError, match="controlled read failure"):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()
    assert not tuple(request.workspace.glob(".pluginmanager-hermes-v020-source.*.tmp"))


def test_source_archive_rejects_legacy_contiguous_header_before_destination_creation(
    tmp_path: Path,
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    with tarfile.open(files["archive"], "w:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/", kind=tarfile.DIRTYPE)
        add_tar_member(
            archive,
            "hermes-agent-v0.20.0/hermes_cli/plugins.py",
            b"legacy",
            kind=tarfile.CONTTYPE,
        )
    with pytest.raises(ValueError, match="ordinary regular"):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert not (request.workspace / ".pluginmanager-hermes-v020-source").exists()


@pytest.mark.skipif(os.name != "nt", reason="native source ownership is Windows-only")
def test_owned_source_prearm_insertion_fails_closed_and_preserves_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    original = module._ExtractedSourceV1._write_file
    inserted: list[Path] = []

    def write_then_insert(
        self: object, relative: str, parent: int, name: str, payload: bytes
    ) -> None:
        original(self, relative, parent, name, payload)
        if not inserted:
            foreign = self._workspace / self._root_name / "foreign.txt"
            foreign.write_bytes(b"not owned")
            inserted.append(foreign)

    monkeypatch.setattr(module._ExtractedSourceV1, "_write_file", write_then_insert)
    with pytest.raises(BaseExceptionGroup):
        module._extract_official_source_archive(request, files["archive"].read_bytes())
    assert len(inserted) == 1
    assert inserted[0].read_bytes() == b"not owned"


@pytest.mark.skipif(os.name != "nt", reason="native source ownership is Windows-only")
def test_owned_source_insertion_fails_closed_and_preserves_residue(tmp_path: Path) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    extracted = module._extract_official_source_archive(request, files["archive"].read_bytes())
    foreign = extracted.root.parent / "foreign.txt"
    foreign.write_bytes(b"not owned")
    with pytest.raises(BaseExceptionGroup):
        extracted.close()
    assert foreign.read_bytes() == b"not owned"


@pytest.mark.skipif(os.name != "nt", reason="native source ownership is Windows-only")
def test_owned_source_retained_handles_deny_replacement_and_mutation(tmp_path: Path) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    extracted = module._extract_official_source_archive(request, files["archive"].read_bytes())
    with pytest.raises(PermissionError):
        extracted.root.parent.rename(request.workspace / "replacement")
    with pytest.raises(PermissionError):
        (extracted.root / "hermes_cli" / "plugins.py").write_bytes(b"mutated")
    extracted.assert_unchanged()
    extracted.close()


@pytest.mark.skipif(os.name != "nt", reason="native source ownership is Windows-only")
def test_source_close_rejects_change_observed_during_watcher_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    extracted = module._extract_official_source_archive(request, files["archive"].read_bytes())
    root = extracted.root.parent
    stop = extracted._stop_watcher

    def mutate_then_stop() -> None:
        transient = root / "transient.txt"
        transient.write_bytes(b"transient")
        transient.unlink()
        deadline = time.monotonic() + 5
        while not extracted._watcher._changed.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert extracted._watcher._changed.is_set()
        stop()

    monkeypatch.setattr(extracted, "_stop_watcher", mutate_then_stop)
    with pytest.raises(BaseExceptionGroup, match="finalization"):
        extracted.close()
    assert root.exists()


@pytest.mark.skipif(os.name != "nt", reason="native source ownership is Windows-only")
def test_source_close_preserves_disposition_and_close_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    request, files = write_request_inputs(module, tmp_path)
    extracted = module._extract_official_source_archive(request, files["archive"].read_bytes())
    original = extracted._dispose_and_close

    def fail_disposition(relative: str) -> None:
        raise PermissionError("controlled disposition failure")

    monkeypatch.setattr(extracted, "_dispose_and_close", fail_disposition)
    with pytest.raises(BaseExceptionGroup, match="finalization") as raised:
        extracted.close()
    assert any("controlled disposition failure" in str(error) for error in raised.value.exceptions)
    monkeypatch.setattr(extracted, "_dispose_and_close", original)


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_retained_owner_denies_existing_writes_and_records_restored_insertions(
    tmp_path: Path,
) -> None:
    module = runner()
    wheelhouse = (tmp_path / "wheelhouse").resolve()
    workspace = (tmp_path / "workspace").resolve()
    default = (tmp_path / "default").resolve()
    active = default / "profiles" / "live"
    profile = workspace / "profiles" / "gate-profile"
    for directory in (wheelhouse, active, profile):
        directory.mkdir(parents=True, exist_ok=True)
    source_archive = write_official_source_archive((tmp_path / "hermes-v020.tar").resolve())
    candidate = wheelhouse / "candidate.whl"
    requirements = wheelhouse / "requirements.txt"
    constraints = wheelhouse / "constraints.txt"
    build_python = (tmp_path / "build-python.exe").resolve()
    for path in (candidate, requirements, constraints, build_python):
        path.write_bytes(b"sealed")
    manifest, manifest_digest = write_wheelhouse_manifest(
        {"candidate": candidate, "requirements": requirements, "constraints": constraints},
        wheelhouse,
    )
    request = module.ParentRequest(
        source_archive,
        candidate,
        wheelhouse,
        requirements,
        constraints,
        workspace,
        profile,
        default,
        active,
        tmp_path / "absent-evidence",
        workspace / "result.json",
        build_python,
        hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        manifest,
        manifest_digest,
    )
    expected_source_payload = source_archive.read_bytes()
    expected_manifest_payload = request.wheelhouse_manifest.read_bytes()
    with (
        pytest.raises(BaseExceptionGroup, match="qualification body and retained cleanup failed"),
        module.WindowsRetainedImmutableInputs(request) as retained,
    ):
        assert retained.read_retained_bytes(source_archive) == expected_source_payload
        assert (
            retained.read_retained_bytes(request.wheelhouse_manifest) == expected_manifest_payload
        )
        assert request.wheelhouse_manifest in retained._retained
        with pytest.raises(PermissionError):
            source_archive.write_bytes(b"changed")
        with pytest.raises(PermissionError):
            request.wheelhouse_manifest.write_bytes(b"changed")
        replacement_manifest = tmp_path / "replacement-manifest.json"
        replacement_manifest.write_bytes(request.wheelhouse_manifest.read_bytes())
        with pytest.raises(PermissionError):
            os.replace(replacement_manifest, request.wheelhouse_manifest)
        inserted = wheelhouse / "transient.py"
        inserted.write_bytes(b"transient")
        inserted.unlink()
        deadline = time.monotonic() + 2
        while not retained._changed.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        retained.assert_unchanged()


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_entry_waits_for_authoritative_arm_and_prearm_change_denies_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    wheelhouse, workspace, default = (
        tmp_path / name for name in ("wheelhouse", "workspace", "default")
    )
    active, profile = default / "profiles/live", workspace / "profiles/gate"
    for directory in (wheelhouse, active, profile):
        directory.mkdir(parents=True, exist_ok=True)
    source_archive = write_official_source_archive((tmp_path / "hermes-v020.tar").resolve())
    candidate, requirements, constraints = (
        wheelhouse / name for name in ("candidate.whl", "requirements.txt", "constraints.txt")
    )
    build_python = (tmp_path / "build-python.exe").resolve()
    for path in (candidate, requirements, constraints, build_python):
        path.write_bytes(b"same")
    manifest, manifest_digest = write_wheelhouse_manifest(
        {"candidate": candidate, "requirements": requirements, "constraints": constraints},
        wheelhouse,
    )
    request = module.ParentRequest(
        source_archive,
        candidate.resolve(),
        wheelhouse.resolve(),
        requirements.resolve(),
        constraints.resolve(),
        workspace.resolve(),
        profile.resolve(),
        default.resolve(),
        active.resolve(),
        (tmp_path / "absent-evidence").resolve(),
        (workspace / "result.json").resolve(),
        build_python,
        hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        manifest,
        manifest_digest,
    )
    reached, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = module.WindowsRetainedImmutableInputs._post_read
    calls = 0

    def paused(self: object, watcher: object, index: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            reached.set()
            assert release.wait(5)
        original(self, watcher, index)

    monkeypatch.setattr(module.WindowsRetainedImmutableInputs, "_post_read", paused)
    errors: list[BaseException] = []
    entered: list[bool] = []

    def enter() -> None:
        try:
            with module.WindowsRetainedImmutableInputs(request):
                entered.append(True)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=enter)
    thread.start()
    assert reached.wait(5) and not finished.is_set()
    with pytest.raises(PermissionError):
        source_archive.write_bytes(source_archive.read_bytes())
    release.set()
    thread.join(10)
    assert finished.is_set() and entered == [True] and errors == []
    assert calls >= 1


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_watcher_posts_both_reads_before_thread_launch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "watched").resolve()
    root.mkdir()
    launched = threading.Event()
    original_start = threading.Thread.start

    def checked_start(thread: threading.Thread) -> None:
        watcher = owner._watchers[0]
        assert watcher["pending"] == [True, True]
        launched.set()
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", checked_start)
    owner._start_watcher(root, None)
    assert launched.is_set()
    owner.close(validate=False)


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_peer_read_captures_transient_change_while_completed_slot_pauses_before_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "watched").resolve()
    root.mkdir()
    parse_reached = threading.Event()
    release_parse = threading.Event()
    original_parse = module.WindowsRetainedImmutableInputs._parse_records
    calls = 0

    def paused_parse(self: object, buffer: bytearray, count: int, prefix: str | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            watcher = self._watchers[0]
            assert watcher["pending"] == [True, True]
            parse_reached.set()
            assert release_parse.wait(5)
        original_parse(self, buffer, count, prefix)

    monkeypatch.setattr(module.WindowsRetainedImmutableInputs, "_parse_records", paused_parse)
    owner._start_watcher(root, "target.tmp")
    (root / "unrelated.tmp").write_bytes(b"first")
    assert parse_reached.wait(5)
    target = root / "target.tmp"
    target.write_bytes(b"transient")
    target.unlink()
    release_parse.set()
    assert owner._changed.wait(5)
    with pytest.raises(BaseExceptionGroup, match="finalization") as captured:
        owner.close(validate=True)
    assert any("changed" in str(error) for error in captured.value.exceptions)


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_shutdown_drains_completed_peer_record_before_final_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "watched").resolve()
    root.mkdir()
    first_parse = threading.Event()
    release_first = threading.Event()
    closing_validation = threading.Event()
    original_parse = module.WindowsRetainedImmutableInputs._parse_records
    original_assert = owner.assert_unchanged
    parse_calls = 0

    def paused_first_parse(self: object, buffer: bytearray, count: int, prefix: str | None) -> None:
        nonlocal parse_calls
        parse_calls += 1
        if parse_calls == 1:
            first_parse.set()
            assert release_first.wait(5)
        original_parse(self, buffer, count, prefix)

    def observed_assert() -> None:
        original_assert()
        closing_validation.set()

    monkeypatch.setattr(module.WindowsRetainedImmutableInputs, "_parse_records", paused_first_parse)
    monkeypatch.setattr(owner, "assert_unchanged", observed_assert)
    owner._start_watcher(root, "target.tmp")
    (root / "unrelated.tmp").write_bytes(b"first")
    assert first_parse.wait(5)
    target = root / "target.tmp"
    target.write_bytes(b"transient")
    target.unlink()
    outcomes: list[BaseException | None] = []

    def close_owner() -> None:
        try:
            owner.close(validate=True)
        except BaseException as error:
            outcomes.append(error)
        else:
            outcomes.append(None)

    closer = threading.Thread(target=close_owner)
    closer.start()
    assert closing_validation.wait(5)
    release_first.set()
    closer.join(10)
    assert not closer.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], BaseExceptionGroup)
    assert "changed" in repr(outcomes[0])


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_nonterminal_watcher_retains_directory_and_both_events_after_join_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "watched").resolve()
    root.mkdir()
    parsing = threading.Event()
    release = threading.Event()
    original_parse = module.WindowsRetainedImmutableInputs._parse_records

    def blocked_parse(self: object, buffer: bytearray, count: int, prefix: str | None) -> None:
        parsing.set()
        assert release.wait(5)
        original_parse(self, buffer, count, prefix)

    monkeypatch.setattr(module.WindowsRetainedImmutableInputs, "_parse_records", blocked_parse)
    owner._start_watcher(root, None)
    associated = {owner._watchers[0]["handle"], *owner._watchers[0]["events"]}
    (root / "change.tmp").write_bytes(b"x")
    assert parsing.wait(5)
    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: None)
    with pytest.raises(BaseExceptionGroup, match="finalization") as captured:
        owner.close(validate=False)
    assert any("terminal" in str(error) for error in captured.value.exceptions)
    assert associated <= set(owner._handles)
    release.set()
    assert owner._watchers[0]["terminal"].wait(5)
    owner.close(validate=False)
    assert owner._handles == []


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_thread_start_failure_cancels_both_reads_and_closes_all_safe_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "watched").resolve()
    root.mkdir()

    def failed_start(self: threading.Thread) -> None:
        raise RuntimeError("injected start failure")

    monkeypatch.setattr(threading.Thread, "start", failed_start)
    with pytest.raises(RuntimeError, match="start failure"):
        owner._start_watcher(root, None)
    assert owner._watchers[0]["started"] is False
    assert owner._watchers[0]["pending"] == [True, True]
    owner.close(validate=False)
    assert owner._handles == []
    assert owner._watchers[0]["pending"] == [False, False]


@pytest.mark.skipif(os.name != "nt", reason="Win32 sharing and change journals are Windows-only")
def test_join_failure_is_accumulated_after_terminal_watcher_handles_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "watched").resolve()
    root.mkdir()

    def failed_parse(self: object, buffer: bytearray, count: int, prefix: str | None) -> None:
        raise RuntimeError("injected watcher terminal")

    monkeypatch.setattr(module.WindowsRetainedImmutableInputs, "_parse_records", failed_parse)
    owner._start_watcher(root, None)
    (root / "change.tmp").write_bytes(b"x")
    assert owner._watchers[0]["terminal"].wait(5)
    monkeypatch.setattr(
        threading.Thread,
        "join",
        lambda self, timeout=None: (_ for _ in ()).throw(RuntimeError("injected join failure")),
    )
    with pytest.raises(BaseExceptionGroup, match="finalization") as captured:
        owner.close(validate=False)
    assert owner._handles == []
    assert any("join failure" in str(error) for error in captured.value.exceptions)


@pytest.mark.skipif(os.name != "nt", reason="Win32 retained reads are Windows-only")
def test_actual_retained_file_validation_uses_bound_x64_pointer_and_read_apis(
    tmp_path: Path,
) -> None:
    module = runner()
    owner = bare_windows_owner(module)
    root = (tmp_path / "inventory").resolve()
    root.mkdir()
    retained = root / "payload.bin"
    retained.write_bytes(b"bound-read" * 1024)
    seal = module.snapshot_regular_tree(root)["payload.bin"]
    assert module.ctypes.sizeof(module.wintypes.HANDLE) == module.ctypes.sizeof(
        module.ctypes.c_void_p
    )
    assert owner._kernel.SetFilePointerEx.argtypes is not None
    assert owner._kernel.ReadFile.argtypes is not None
    owner._retain(retained, seal)
    owner._validate_handle(retained, owner._retained[retained][0], seal)
    owner.close(validate=False)


def test_journal_parser_fails_closed_on_zero_overflow_and_malformed_records() -> None:
    if os.name != "nt":
        pytest.skip("Win32 journal parser is Windows-only")
    module = runner()
    owner = module.WindowsRetainedImmutableInputs.__new__(module.WindowsRetainedImmutableInputs)
    owner._changed = threading.Event()
    for buffer, count in (
        (bytearray(32), 0),
        (bytearray(8), 9),
        (bytearray(32), 12),
        (bytearray(b"\x04\0\0\0" + b"\0" * 28), 32),
    ):
        with pytest.raises(RuntimeError, match="journal"):
            owner._parse_records(buffer, count, None)


def test_stage_failure_always_finalizes_job_and_preserves_dual_failure(
    tmp_path: Path,
) -> None:
    module = runner()

    class FailingJob:
        def __init__(self, *, fail_finalization: bool) -> None:
            self.fail_finalization = fail_finalization
            self.finalize_calls = 0

        def launch(self, command: tuple[str, ...], **_: object) -> int:
            del command
            raise RuntimeError("stage failed")

        def wait_for_exit(self, child: int, **_: object) -> tuple[int, bytes]:
            del child
            raise AssertionError("wait must not follow failed launch")

        def force_finalize(self, *, timeout_seconds: int) -> None:
            assert timeout_seconds == 30
            self.finalize_calls += 1
            if self.fail_finalization:
                raise RuntimeError("finalization failed")

    request, files = write_request_inputs(module, tmp_path / "primary")
    write_candidate_wheel(files["candidate"])
    request = refresh_wheelhouse_manifest(request, files)
    base_environment = {
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "USERNAME": "gate",
        "USERPROFILE": r"C:\Users\gate",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\gate",
    }

    validation_job = FailingJob(fail_finalization=False)
    with pytest.raises(ValueError, match="alias"):
        module._qualify_governed_request(
            request._replace(active_profile=request.wheelhouse),
            job=validation_job,
            base_environment=base_environment,
            revalidate_inputs=lambda: None,
            venv_name="pluginmanager-venv-validation",
        )
    assert validation_job.finalize_calls == 1

    digest_request, digest_files = write_request_inputs(module, tmp_path / "digest")
    write_candidate_wheel(digest_files["candidate"])
    digest_request = refresh_wheelhouse_manifest(digest_request, digest_files)
    with tarfile.open(digest_files["archive"], "a:") as archive:
        add_tar_member(archive, "hermes-agent-v0.20.0/digest-drift.txt", b"drift")
    digest_job = FailingJob(fail_finalization=False)
    with pytest.raises(ValueError, match="archive digest changed"):
        module._qualify_governed_request(
            digest_request,
            job=digest_job,
            base_environment=base_environment,
            revalidate_inputs=lambda: None,
            venv_name="pluginmanager-venv-digest",
        )
    assert digest_job.finalize_calls == 1

    primary_job = FailingJob(fail_finalization=False)
    primary_revalidations: list[None] = []
    with pytest.raises(RuntimeError, match="stage failed"):
        module._qualify_governed_request(
            request,
            job=primary_job,
            base_environment=base_environment,
            revalidate_inputs=lambda: primary_revalidations.append(None),
            venv_name="pluginmanager-venv-primary",
        )
    assert primary_job.finalize_calls == 1
    assert len(primary_revalidations) == 4

    post_request, post_files = write_request_inputs(module, tmp_path / "post")
    write_candidate_wheel(post_files["candidate"])
    post_request = refresh_wheelhouse_manifest(post_request, post_files)
    post_failure_job = FailingJob(fail_finalization=False)
    post_revalidations = 0

    def fail_post_revalidation() -> None:
        nonlocal post_revalidations
        post_revalidations += 1
        if post_revalidations == 4:
            raise RuntimeError("post revalidation failed")

    with pytest.raises(BaseExceptionGroup) as post_captured:
        module._qualify_governed_request(
            post_request,
            job=post_failure_job,
            base_environment=base_environment,
            revalidate_inputs=fail_post_revalidation,
            venv_name="pluginmanager-venv-post-failure",
        )
    assert [str(error) for error in post_captured.value.exceptions] == [
        "stage failed",
        "post revalidation failed",
    ]
    assert post_failure_job.finalize_calls == 1

    request, files = write_request_inputs(module, tmp_path / "dual")
    write_candidate_wheel(files["candidate"])
    request = refresh_wheelhouse_manifest(request, files)
    dual_job = FailingJob(fail_finalization=True)
    with pytest.raises(BaseExceptionGroup) as captured:
        module._qualify_governed_request(
            request,
            job=dual_job,
            base_environment=base_environment,
            revalidate_inputs=lambda: None,
            venv_name="pluginmanager-venv-dual",
        )
    assert dual_job.finalize_calls == 1
    assert [str(error) for error in captured.value.exceptions] == [
        "stage failed",
        "finalization failed",
    ]


def test_governed_aggregation_requires_disabled_false_and_publishes_reopened_output(
    tmp_path: Path,
) -> None:
    module = runner()
    request, request_files = write_request_inputs(module, tmp_path)
    files = {
        "candidate.whl": request_files["candidate"],
        "requirements.txt": request_files["requirements"],
        "constraints.txt": request_files["constraints"],
        "build_python": request_files["build_python"],
    }
    import zipfile

    with zipfile.ZipFile(files["candidate.whl"], "w") as archive:
        archive.writestr("hermes_realtime/hermes_plugin.py", b"candidate module")
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/METADATA", b"Name: hermes-realtime\nVersion: 0.0.3\n"
        )
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/WHEEL", b"Wheel-Version: 1.0\nTag: py3-none-any\n"
        )
        archive.writestr(
            "hermes_realtime-0.0.3.dist-info/entry_points.txt",
            b"[hermes_agent.plugins]\nhermes-realtime = hermes_realtime.hermes_plugin\n",
        )
        archive.writestr("hermes_realtime-0.0.3.dist-info/RECORD", b"installer rewrites this")
    request = refresh_wheelhouse_manifest(request, request_files)
    extracted_source = (
        request.workspace / ".pluginmanager-hermes-v020-source" / "hermes-agent-v0.20.0"
    )
    source_hash = hashlib.sha256(str(extracted_source).encode()).hexdigest()
    expected_module = (
        request.workspace
        / "pluginmanager-venv-fixed"
        / "Lib"
        / "site-packages"
        / "hermes_realtime"
        / "hermes_plugin.py"
    )
    module_hash = hashlib.sha256(str(expected_module).encode()).hexdigest()
    module_content_hash = hashlib.sha256(b"candidate module").hexdigest()
    dist_info = (
        request.workspace
        / "pluginmanager-venv-fixed"
        / "Lib"
        / "site-packages"
        / "hermes_realtime-0.0.3.dist-info"
    )
    dist_origin_hash = hashlib.sha256(str(dist_info).encode()).hexdigest()
    metadata_hash = hashlib.sha256(b"Name: hermes-realtime\nVersion: 0.0.3\n").hexdigest()
    entrypoints_hash = hashlib.sha256(
        b"[hermes_agent.plugins]\nhermes-realtime = hermes_realtime.hermes_plugin\n"
    ).hexdigest()
    wheel_hash = hashlib.sha256(b"Wheel-Version: 1.0\nTag: py3-none-any\n").hexdigest()
    dist_inventory_hash = hashlib.sha256(
        json.dumps(
            {
                "hermes_realtime-0.0.3.dist-info/METADATA": metadata_hash,
                "hermes_realtime-0.0.3.dist-info/WHEEL": wheel_hash,
                "hermes_realtime-0.0.3.dist-info/entry_points.txt": entrypoints_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    outputs = [
        b"",
        b"",
        json.dumps(
            {"discovered": True, "importOrRegistration": False, "sourceOriginSha256": source_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n",
        b'{"argvExact":true,"exactConfigDelta":true,"stdinClosed":true}\n',
        json.dumps(
            {
                "contextObserved": True,
                "discovered": True,
                "distributionVersion": "0.0.3",
                "distEntryPointsContentSha256": entrypoints_hash,
                "distInfoInventorySha256": dist_inventory_hash,
                "distMetadataContentSha256": metadata_hash,
                "distOriginSha256": dist_origin_hash,
                "moduleContentSha256": module_content_hash,
                "moduleOriginSha256": module_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n",
    ]

    class Job:
        def launch(self, command: tuple[str, ...], **_: object) -> int:
            self.commands.append(command)
            if command[1:3] == ("-m", "venv"):
                assert not Path(command[3]).exists()
                python = Path(command[3]) / (
                    "Scripts/python.exe" if os.name == "nt" else "bin/python"
                )
                python.parent.mkdir(parents=True)
                python.write_bytes(b"python")
            if command[3] == module._DISABLED_CHILD:
                self.pending[0] = (
                    json.dumps(
                        {
                            "discovered": True,
                            "importOrRegistration": False,
                            "sourceOriginSha256": hashlib.sha256(command[4].encode()).hexdigest(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
            return len(outputs) - len(self.pending)

        def wait_for_exit(self, child: int, **_: object) -> tuple[int, bytes]:
            del child
            return 0, self.pending.pop(0)

        def force_finalize(self, *, timeout_seconds: int) -> None:
            assert timeout_seconds == 30
            self.finalize_calls += 1

        pending = outputs.copy()
        finalize_calls = 0
        commands: list[tuple[str, ...]] = []

    job = Job()
    revalidations: list[None] = []
    facts = module._qualify_governed_request(
        request,
        job=job,
        base_environment={
            "SYSTEMROOT": r"C:\Windows",
            "WINDIR": r"C:\Windows",
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
            "USERNAME": "gate",
            "USERPROFILE": r"C:\Users\gate",
            "HOMEDRIVE": "C:",
            "HOMEPATH": r"\Users\gate",
        },
        revalidate_inputs=lambda: revalidations.append(None),
        venv_name="pluginmanager-venv-fixed",
    )
    payload = request.output.read_bytes()
    assert module.parse_child_result(payload)["disabledImportOrRegistration"] is False
    assert facts == {
        "commandSha256": facts["commandSha256"],
        "outputSha256": hashlib.sha256(payload).hexdigest(),
        "result": module.parse_child_result(payload),
    }
    assert job.finalize_calls == 1
    assert job.commands[0][0] == str(files["build_python"])
    install = job.commands[1]
    disabled = job.commands[2]
    enabled = job.commands[4]
    assert install[-1] == "--no-deps"
    assert str(files["candidate.whl"]) not in install
    assert disabled[-1] == str(request.workspace / "pluginmanager-venv-fixed")
    assert json.loads(enabled[-1]) == [
        "hermes_realtime-0.0.3.dist-info/METADATA",
        "hermes_realtime-0.0.3.dist-info/WHEEL",
        "hermes_realtime-0.0.3.dist-info/entry_points.txt",
    ]
    assert len(revalidations) == 15


def test_release_gate_keeps_the_pluginmanager_harness_in_sdist_and_static_checks() -> None:
    release_gate_path = ROOT / "scripts" / "release_gate.py"
    release_gate = run_path(str(release_gate_path))
    source = release_gate_path.read_text(encoding="utf-8")

    harness_path = '"scripts/qualify_hermes_v020_pluginmanager.py"'
    typecheck_block = source.split("script_type_env =", 1)[1].split("cwd=root", 1)[0]
    assert "scripts/qualify_hermes_v020_pluginmanager.py" in release_gate["required_sdist_paths"]()
    assert harness_path in typecheck_block
