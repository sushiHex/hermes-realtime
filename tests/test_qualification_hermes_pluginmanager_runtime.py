import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _source_metadata():
    from scripts.qualification_hermes_source import (
        HermesPublisherFileMetadataV1,
        HermesPublisherSourceMetadataV1,
    )

    return HermesPublisherFileMetadataV1(
        "a" * 64,
        HermesPublisherSourceMetadataV1(
            "NousResearch/hermes-agent",
            "v2026.8.3",
            "b" * 40,
            "c" * 40,
            "d" * 40,
            "e" * 64,
            1,
            "f" * 64,
            1,
            "0" * 64,
            1,
            1,
            1,
            "1" * 40,
            "2" * 40,
            "3" * 64,
        ),
    )


def _runtime_metadata(*, purpose="hermes_v020_pluginmanager_runtime", source="1" * 40):
    from scripts.qualification_runtime_environment import RuntimeEnvironmentMetadataV1

    return RuntimeEnvironmentMetadataV1(
        "a" * 64,
        source,
        "2" * 40,
        purpose,
        "4" * 64,
        SimpleNamespace(),
        4,
        5,
    )


def _profile_source(soul: str = "source soul") -> dict[str, bytes]:
    return {
        "hermes_cli/default_soul.py": f"DEFAULT_SOUL_MD = {soul!r}\n".encode(),
        "hermes_cli/config.py": b'''def ensure_hermes_home():
    for subdir in (
        "cron", "sessions", "logs", "logs/curator", "memories",
        "pairing", "hooks", "image_cache", "audio_cache", "skills",
    ):
        pass
''',
        "hermes_cli/plugins_cmd.py": b'''def _plugins_dir():
    plugins = get_hermes_home() / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins
''',
    }


def test_qualifier_requires_live_matching_source_and_installed_runtime(monkeypatch):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter
    from scripts.qualification_hermes_source import BoundHermesPublisherSourceV1
    from scripts.qualification_runtime_environment import InstalledRuntimeEnvironmentV1

    source = object.__new__(BoundHermesPublisherSourceV1)
    runtime = object.__new__(InstalledRuntimeEnvironmentV1)
    source_metadata = _source_metadata()
    runtime_metadata = _runtime_metadata()
    files, candidate, dependencies = object(), object(), object()
    monkeypatch.setattr(
        adapter,
        "_bound_hermes_source_for_consumer",
        lambda selected: SimpleNamespace(files=files),
    )
    monkeypatch.setattr(adapter, "hermes_publisher_file_metadata", lambda selected: source_metadata)
    monkeypatch.setattr(
        adapter,
        "_installed_runtime_for_consumer",
        lambda selected: SimpleNamespace(work=object(), dependencies=dependencies),
    )
    monkeypatch.setattr(adapter, "installed_runtime_metadata", lambda selected: runtime_metadata)
    monkeypatch.setattr(
        adapter,
        "_dependency_binding_for_consumer",
        lambda selected: SimpleNamespace(files=files, candidate=candidate),
    )
    monkeypatch.setattr(
        adapter,
        "candidate_file_metadata",
        lambda selected: SimpleNamespace(source_archive_sha256="3" * 64),
    )

    assert adapter._metadata(source, runtime)[1] == runtime_metadata


@pytest.mark.parametrize("fault", ["purpose", "source", "closed"])
def test_qualifier_refuses_wrong_or_expired_authority(monkeypatch, fault):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter
    from scripts.qualification_hermes_source import BoundHermesPublisherSourceV1
    from scripts.qualification_runtime_environment import InstalledRuntimeEnvironmentV1

    source = object.__new__(BoundHermesPublisherSourceV1)
    runtime = object.__new__(InstalledRuntimeEnvironmentV1)
    source_metadata = _source_metadata()
    runtime_metadata = _runtime_metadata(
        purpose="realtime_windows_direct_runtime"
        if fault == "purpose"
        else "hermes_v020_pluginmanager_runtime",
        source="9" * 40 if fault == "source" else "1" * 40,
    )
    if fault == "closed":
        monkeypatch.setattr(
            adapter,
            "_bound_hermes_source_for_consumer",
            lambda selected: (_ for _ in ()).throw(ValueError("final source seals are closed")),
        )
    else:
        files = object()
        monkeypatch.setattr(
            adapter,
            "_bound_hermes_source_for_consumer",
            lambda selected: SimpleNamespace(files=files),
        )
    monkeypatch.setattr(adapter, "hermes_publisher_file_metadata", lambda selected: source_metadata)
    monkeypatch.setattr(
        adapter,
        "_installed_runtime_for_consumer",
        lambda selected: SimpleNamespace(work=object(), dependencies=object()),
    )
    monkeypatch.setattr(adapter, "installed_runtime_metadata", lambda selected: runtime_metadata)
    if fault != "closed":
        monkeypatch.setattr(
            adapter,
            "_dependency_binding_for_consumer",
            lambda selected: SimpleNamespace(files=files, candidate=object()),
        )
        monkeypatch.setattr(
            adapter,
            "candidate_file_metadata",
            lambda selected: SimpleNamespace(source_archive_sha256="3" * 64),
        )

    with pytest.raises(ValueError):
        adapter._metadata(source, runtime)


def test_completed_runtime_capability_cannot_be_forged():
    from scripts.qualification_hermes_pluginmanager_runtime import (
        CompletedHermesPluginManagerRuntimeV1,
        hermes_pluginmanager_runtime_metadata,
    )

    with pytest.raises(TypeError):
        CompletedHermesPluginManagerRuntimeV1()
    with pytest.raises(ValueError, match="unregistered"):
        hermes_pluginmanager_runtime_metadata(
            object.__new__(CompletedHermesPluginManagerRuntimeV1)
        )


def test_candidate_plugin_uses_authenticated_wheel_member_bytes(monkeypatch):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter

    payloads = {
        "hermes_realtime/hermes_plugin.py": b"plugin\n",
        "hermes_realtime-0.0.3.dist-info/METADATA": b"metadata\n",
        "hermes_realtime-0.0.3.dist-info/entry_points.txt": b"entry points\n",
        "hermes_realtime-0.0.3.dist-info/WHEEL": b"wheel\n",
        "hermes_realtime-0.0.3.dist-info/RECORD": b"mutable\n",
    }
    wheel = SimpleNamespace(members=payloads)
    candidate = SimpleNamespace(wheels=(object(),), archive=object(), identity=object())
    runtime = SimpleNamespace(dependencies=object(), files=object())
    monkeypatch.setattr(
        adapter,
        "_dependency_binding_for_consumer",
        lambda selected: SimpleNamespace(candidate=candidate),
    )
    monkeypatch.setattr(adapter, "_candidate_files_for_consumer", lambda selected: candidate)
    monkeypatch.setattr(adapter.wheels, "_wheel_for_consumer", lambda *unused: wheel)
    monkeypatch.setattr(
        adapter,
        "execution_file_metadata",
        lambda selected: tuple(
            (name, hashlib.sha256(payload).hexdigest(), len(payload))
            for name, payload in payloads.items()
            if not name.endswith("/RECORD")
        ),
    )

    value = adapter._candidate_plugin(runtime)

    assert value.module_sha256 == hashlib.sha256(
        payloads["hermes_realtime/hermes_plugin.py"]
    ).hexdigest()
    assert value.immutable_dist_info == (
        "hermes_realtime-0.0.3.dist-info/METADATA",
        "hermes_realtime-0.0.3.dist-info/WHEEL",
        "hermes_realtime-0.0.3.dist-info/entry_points.txt",
    )


@pytest.mark.parametrize(
    "fault", [None, "source", "plugins_source", "missing", "extra", "drift", "symlink"]
)
def test_source_bootstrap_and_profile_namespace_are_exact(tmp_path, monkeypatch, fault):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter

    profile = tmp_path / "profile"
    profile.mkdir()
    source = _profile_source()
    if fault == "source":
        source["hermes_cli/config.py"] = source["hermes_cli/config.py"].replace(
            b'"skills",', b'"skills", "foreign",'
        )
        with pytest.raises(ValueError, match="directory bootstrap source differs"):
            adapter._bootstrap_profile(profile, source)
        return
    if fault == "plugins_source":
        source["hermes_cli/plugins_cmd.py"] = source["hermes_cli/plugins_cmd.py"].replace(
            b"plugins.mkdir", b"foreign.mkdir"
        )
        with pytest.raises(ValueError, match="plugins directory source differs"):
            adapter._bootstrap_profile(profile, source)
        return
    baseline = adapter._bootstrap_profile(profile, source)
    assert (profile / "SOUL.md").read_bytes() == b"source soul"
    (profile / "config.yaml").write_bytes(b"config\n")
    if fault == "missing":
        (profile / "plugins").rmdir()
    elif fault == "extra":
        (profile / "foreign").write_bytes(b"foreign")
    elif fault == "drift":
        (profile / "SOUL.md").write_bytes(b"changed")
    elif fault == "symlink":
        original = Path.lstat

        def redirected(selected):
            metadata = original(selected)
            if selected != profile / "SOUL.md":
                return metadata
            return SimpleNamespace(
                st_dev=metadata.st_dev,
                st_file_attributes=0x400,
                st_ino=metadata.st_ino,
                st_mode=metadata.st_mode,
                st_size=metadata.st_size,
            )

        monkeypatch.setattr(Path, "lstat", redirected)
        with pytest.raises(ValueError, match="redirected"):
            adapter._tree(profile)
        return
    assert adapter._exact_profile(baseline, adapter._tree(profile)) is (fault is None)


def test_profile_tree_bounds_entries_and_files_before_reading(tmp_path, monkeypatch):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter

    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *unused, **unused_keywords: (_ for _ in ()).throw(AssertionError("eager tree")),
    )
    oversized = profile / "oversized"
    with oversized.open("wb") as stream:
        stream.truncate(1024**2 + 1)
    original = Path.read_bytes
    read = False

    def observed(path):
        nonlocal read
        if path == oversized:
            read = True
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observed)
    with pytest.raises(ValueError, match="exceeds its bound"):
        adapter._tree(profile)
    assert read is False

    oversized.unlink()
    for index in range(1025):
        (profile / f"d{index:04d}").mkdir()
    with pytest.raises(ValueError, match="exceeds its bound"):
        adapter._tree(profile)


def _stage_report(adapter, pid, source, packages, candidate):
    value = {
        "pid": pid,
        "stages": {
            "disabled": {
                "discovered": True,
                "importOrRegistration": False,
                "sourceOriginSha256": hashlib.sha256(str(source).encode()).hexdigest(),
            },
            "enable": {"argvExact": True, "exactConfigDelta": True, "stdinClosed": True},
            "enabled": {
                "contextObserved": True,
                "discovered": True,
                "distributionVersion": "0.0.3",
                "distEntryPointsContentSha256": candidate.entry_points_sha256,
                "distInfoInventorySha256": candidate.inventory_sha256,
                "distMetadataContentSha256": candidate.metadata_sha256,
                "distOriginSha256": hashlib.sha256(
                    str(packages / candidate.dist_info).encode()
                ).hexdigest(),
                "moduleContentSha256": candidate.module_sha256,
                "moduleOriginSha256": hashlib.sha256(
                    str(packages / "hermes_realtime/hermes_plugin.py").encode()
                ).hexdigest(),
            },
        },
        "version": 1,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def test_report_requires_exact_pid_stage_semantics_and_parent_origins(tmp_path):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter

    source, packages = tmp_path / "source", tmp_path / "packages"
    source.mkdir()
    packages.mkdir()
    candidate = adapter._CandidatePlugin(
        "hermes_realtime-0.0.3.dist-info",
        ("hermes_realtime-0.0.3.dist-info/METADATA",),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    raw = _stage_report(adapter, 42, source, packages, candidate)
    result = adapter._result(
        raw,
        pid=42,
        source=source,
        packages=packages,
        candidate=candidate,
        roots_unchanged=True,
        exact_profile=True,
    )
    assert result.plugin_context_registration_observed is True
    report = json.loads(raw)
    for fault in ("pid", "disabled", "origin", "profile"):
        changed = json.loads(raw)
        if fault == "pid":
            changed["pid"] = 43
        elif fault == "disabled":
            changed["stages"]["disabled"]["importOrRegistration"] = True
        elif fault == "origin":
            changed["stages"]["enabled"]["moduleOriginSha256"] = "9" * 64
        else:
            changed = report
        payload = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with pytest.raises(ValueError):
            adapter._result(
                payload,
                pid=42,
                source=source,
                packages=packages,
                candidate=candidate,
                roots_unchanged=True,
                exact_profile=fault != "profile",
            )


@pytest.mark.skipif(os.name != "nt", reason="owned execution snapshots require Windows")
def test_parent_executes_archived_worker_and_cleans_disposable_roots_before_receipt(
    tmp_path: Path, monkeypatch
):
    from scripts import qualification_hermes_pluginmanager_runtime as adapter
    from scripts.qualification_execution_files import owned_execution_files
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    source_receipt, runtime_receipt = object(), object()
    source_metadata, runtime_metadata = _source_metadata(), _runtime_metadata()
    candidate = adapter._CandidatePlugin(
        "hermes_realtime-0.0.3.dist-info",
        ("hermes_realtime-0.0.3.dist-info/METADATA",),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    invocation = object()
    process = SimpleNamespace(pid=4242)
    invocation_metadata = SimpleNamespace(exit_code=0)
    runtime_work = SimpleNamespace(_closing=False, _closed=False, _unrecoverable=None)
    calls = []
    with owned_execution_files({"installed.txt": b"installed"}) as installed_files:
        installed = SimpleNamespace(files=installed_files, tools=object(), work=runtime_work)
        monkeypatch.setattr(
            adapter, "_metadata", lambda selected_source, selected_runtime: (
                source_metadata,
                runtime_metadata,
            )
        )
        monkeypatch.setattr(adapter, "_installed_runtime_for_consumer", lambda selected: installed)
        monkeypatch.setattr(adapter, "_candidate_plugin", lambda selected: candidate)
        monkeypatch.setattr(
            adapter,
            "_bound_hermes_execution_inputs",
            lambda selected: (
                {
                    "hermes_cli/__init__.py": b"",
                    **_profile_source("soul"),
                },
                b"harness",
                b"worker",
            ),
        )
        monkeypatch.setattr(
            adapter,
            "_tool_invocation_for_consumer",
            lambda selected: SimpleNamespace(process=process),
        )
        monkeypatch.setattr(
            adapter, "tool_invocation_metadata", lambda selected: invocation_metadata
        )
        monkeypatch.setattr(
            adapter, "hermes_publisher_file_metadata", lambda selected: source_metadata
        )
        monkeypatch.setattr(adapter, "_retained_facts", lambda selected: runtime_metadata)

        def run_tool(self, tools, role, arguments, workspace, **options):
            del self, tools
            calls.append((role, arguments, workspace, options))
            source = Path(arguments[2])
            packages = Path(arguments[3])
            profile = Path(arguments[4])
            assert Path(arguments[5]).name == "default"
            profile.joinpath("config.yaml").write_text("{}", encoding="utf-8")
            Path(arguments[6]).write_bytes(
                _stage_report(adapter, process.pid, source, packages, candidate)
            )
            return invocation

        monkeypatch.setattr(OwnedQualificationWorkV1, "run_tool", run_tool)
        work = OwnedQualificationWorkV1()
        try:
            receipt = adapter.qualify_hermes_pluginmanager_runtime(
                work, source_receipt, runtime_receipt
            )
            metadata = adapter.hermes_pluginmanager_runtime_metadata(receipt)
            assert metadata.result.exact_config_delta is True
            assert adapter.hermes_pluginmanager_result(receipt)["distributionVersion"] == "0.0.3"
            assert calls[0][0] == "build_python"
            assert calls[0][1][-1] == '["hermes_realtime-0.0.3.dist-info/METADATA"]'
            assert metadata.invocation is invocation_metadata
            assert adapter._COMPLETED[receipt].execution._closed is True
            runtime_work._closing = True
            with pytest.raises(ValueError, match="cleanup is incomplete"):
                adapter.hermes_pluginmanager_runtime_metadata(receipt)
            runtime_work._closed = True
            assert adapter.hermes_pluginmanager_runtime_metadata(receipt) == metadata
            runtime_work._unrecoverable = RuntimeError("cleanup failed")
            with pytest.raises(ValueError, match="cleanup is incomplete"):
                adapter.hermes_pluginmanager_runtime_metadata(receipt)
            runtime_work._unrecoverable = None
        finally:
            work.close()
