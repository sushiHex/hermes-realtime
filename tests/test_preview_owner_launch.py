from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import preview_owner_launch as launch
from scripts.qualification_launch import WINDOWLESS_BREAKAWAY_FLAGS


def _fake_runtime(monkeypatch, interpreter):
    binding = {"executable": str(interpreter),
               "base_executable": str(interpreter.parent / "base.exe"),
               "prefix": str(interpreter.parent), "base_prefix": str(interpreter.parent),
               "executable_sha256": "a" * 64, "base_sha256": "b" * 64}
    monkeypatch.setattr(launch, "_probe_interpreter", lambda selected: binding, raising=False)
    return binding


def test_launch_uses_fixed_windowless_independent_owner_and_private_files(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    interpreter = tmp_path / "bound-python.exe"
    active = tmp_path / "active.private.json"
    binding = _fake_runtime(monkeypatch, interpreter)
    selected = object()
    observed = {}

    def popen(command, **kwargs):
        observed.update(command=command, **kwargs)
        assert kwargs["stdout"].name == str(tmp_path / "owner.private.log")
        assert kwargs["stderr"].name == str(tmp_path / "owner.stderr.private.log")
        assert not kwargs["stdout"].closed and not kwargs["stderr"].closed
        return selected

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    assert launch.launch_preview_owner(
        interpreter=interpreter, active=active, output=tmp_path,
    ) is selected
    assert observed["command"] == [
        binding["base_executable"], "-I", "-B",
        str(Path(launch.__file__).with_name("preview_cutover.py")),
        "serve", "--active", str(active),
    ]
    assert observed["cwd"] == Path(launch.__file__).resolve().parents[1]
    assert observed["creationflags"] == WINDOWLESS_BREAKAWAY_FLAGS
    assert observed["stdin"] == subprocess.DEVNULL
    assert observed["close_fds"] is True
    assert observed["env"]["__PYVENV_LAUNCHER__"] == str(interpreter)
    assert observed["stdout"].closed and observed["stderr"].closed


@pytest.mark.parametrize("name", ["owner.private.log", "owner.stderr.private.log"])
def test_launch_never_overwrites_existing_output(tmp_path, monkeypatch, name):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    _fake_runtime(monkeypatch, Path("python.exe"))
    path = tmp_path / name
    path.write_bytes(b"retained")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as error:
        launch.launch_preview_owner(interpreter=Path("python.exe"), active=Path("active"),
                                    output=tmp_path)
    assert error.value.category == "output_exists"
    assert path.read_bytes() == b"retained"


@pytest.mark.parametrize("winerror,category", [(5, "breakaway_not_permitted"),
                                              (87, "owner_spawn_failed")])
def test_spawn_refuses_without_fallback_and_closes_parent_handles(
    tmp_path, monkeypatch, winerror, category,
):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    _fake_runtime(monkeypatch, Path("python.exe"))
    observed = []

    def popen(*args, **kwargs):
        observed.append(kwargs)
        error = OSError("synthetic private error")
        error.winerror = winerror
        raise error

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.launch_preview_owner(interpreter=Path("python.exe"), active=Path("active"),
                                    output=tmp_path)
    assert failure.value.category == category
    assert str(failure.value) == category
    assert len(observed) == 1
    assert observed[0]["stdout"].closed and observed[0]["stderr"].closed


def test_output_failure_never_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    _fake_runtime(monkeypatch, Path("python.exe"))
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.launch_preview_owner(interpreter=Path("python.exe"), active=Path("active"),
                                    output=tmp_path / "absent")
    assert failure.value.category == "owner_output_unavailable"


@pytest.mark.parametrize("outside", [True, False])
def test_actual_owner_requires_no_inherited_jobs(monkeypatch, outside):
    monkeypatch.setattr(launch, "_outside_jobs", lambda: outside)
    monkeypatch.setattr(launch, "_verify_owner_runtime", lambda interpreter: None, raising=False)
    if outside:
        launch.assert_independent_owner(interpreter=Path("bound-python.exe"))
    else:
        with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
            launch.assert_independent_owner(interpreter=Path("bound-python.exe"))
        assert failure.value.category == "owner_inherited_job"


def test_unknown_membership_is_refused(monkeypatch):
    def unavailable():
        raise OSError("synthetic private membership error")

    monkeypatch.setattr(launch, "_outside_jobs", unavailable)
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.assert_independent_owner(interpreter=Path("bound-python.exe"))
    assert failure.value.category == "membership_unavailable"
    assert str(failure.value) == "membership_unavailable"


def test_non_windows_launch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.launch_preview_owner(interpreter=Path("python"), active=Path("active"),
                                    output=tmp_path)
    assert failure.value.category == "windows_required"


@pytest.fixture
def runtime_binding(tmp_path):
    interpreter, base = tmp_path / "venv-python.exe", tmp_path / "base-python.exe"
    interpreter.write_bytes(b"venv-image")
    base.write_bytes(b"base-image")
    return {"executable": str(interpreter), "base_executable": str(base),
            "prefix": str(tmp_path / "venv"), "base_prefix": str(tmp_path / "base"),
            "executable_sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
            "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest()}


def test_probe_derives_only_from_selected_interpreter(runtime_binding, monkeypatch):
    binding = runtime_binding
    for key in launch._IDENTITY_ENV:
        monkeypatch.setenv(key, "ambient-override")
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps(binding).encode())

    monkeypatch.setattr(launch.subprocess, "run", run)
    assert launch._probe_interpreter(Path(binding["executable"])) == binding
    assert observed["command"] == [binding["executable"], "-I", "-B", "-c", launch._PROBE]
    assert observed["timeout"] == 10
    assert observed["stdin"] == observed["stderr"] == subprocess.DEVNULL
    assert observed["creationflags"] == WINDOWLESS_BREAKAWAY_FLAGS
    assert observed["close_fds"] is True
    assert all(key not in observed["env"] for key in launch._IDENTITY_ENV)


@pytest.mark.parametrize("fault", ["exit", "json", "bound", "foreign", "venv_hash", "base_hash"])
def test_probe_refuses_invalid_binding_without_fallback(runtime_binding, monkeypatch, fault):
    binding = dict(runtime_binding)
    if fault == "foreign":
        foreign = Path(binding["executable"]).with_name("foreign.exe")
        foreign.write_bytes(Path(binding["executable"]).read_bytes())
        binding["executable"] = str(foreign)
    elif fault == "venv_hash":
        binding["executable_sha256"] = "0" * 64
    elif fault == "base_hash":
        binding["base_sha256"] = "0" * 64
    if fault == "bound":
        binding["prefix"] = "C:/" + "p" * 4090
        binding["base_prefix"] = "C:/" + "b" * 4090
    raw = json.dumps(binding).encode()
    raw = b"not-json" if fault == "json" else raw
    calls = []

    def run(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(returncode=1 if fault == "exit" else 0, stdout=raw)

    monkeypatch.setattr(launch.subprocess, "run", run)
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch._probe_interpreter(Path(runtime_binding["executable"]))
    assert failure.value.category == "interpreter_probe_failed"
    assert len(calls) == 1


def test_probe_refuses_venv_changed_during_probe(runtime_binding, monkeypatch):
    def run(*args, **kwargs):
        Path(runtime_binding["executable"]).write_bytes(b"changed-image")
        changed = runtime_binding | {
            "executable_sha256": hashlib.sha256(b"changed-image").hexdigest()}
        return SimpleNamespace(returncode=0, stdout=json.dumps(changed).encode())

    monkeypatch.setattr(launch.subprocess, "run", run)
    with pytest.raises(launch.PreviewOwnerLaunchRefusal):
        launch._probe_interpreter(Path(runtime_binding["executable"]))


@pytest.mark.parametrize("fault", [None, "missing", "image", "venv", "prefix", "base_prefix",
                                  "plan", "base_hash", "venv_hash", "unconsumed"])
def test_actual_owner_binds_plan_image_and_venv(runtime_binding, monkeypatch, fault):
    binding = dict(runtime_binding)
    monkeypatch.setattr(launch, "_outside_jobs", lambda: True)
    monkeypatch.setattr(launch, "_actual_image", lambda: Path(
        binding["executable"] if fault == "image" else binding["base_executable"]))
    for name in ("executable", "prefix", "base_prefix"):
        value = binding[name]
        if fault == {"executable": "venv", "prefix": "prefix", "base_prefix": "base_prefix"}[name]:
            value += "-foreign"
        monkeypatch.setattr(launch.sys, name, value)
    monkeypatch.delenv("__PYVENV_LAUNCHER__", raising=False)
    if fault == "unconsumed":
        monkeypatch.setenv("__PYVENV_LAUNCHER__", binding["executable"])
    if fault in {"base_hash", "venv_hash"}:
        binding["base_sha256" if fault == "base_hash" else "executable_sha256"] = "0" * 64
    if fault != "missing":
        monkeypatch.setenv(launch._BINDING_ENV, json.dumps(binding))
    else:
        monkeypatch.delenv(launch._BINDING_ENV, raising=False)
    selected = Path(binding["base_executable"] if fault == "plan" else binding["executable"])
    if fault is None:
        launch.assert_independent_owner(interpreter=selected)
    else:
        with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
            launch.assert_independent_owner(interpreter=selected)
        assert failure.value.category == "owner_runtime_mismatch"
    assert launch._BINDING_ENV not in launch.os.environ
