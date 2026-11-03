from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import preview_owner_launch as launch
from scripts.qualification_launch import WINDOWLESS_BREAKAWAY_FLAGS


def test_launch_uses_fixed_windowless_independent_owner_and_private_files(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    interpreter = tmp_path / "bound-python.exe"
    active = tmp_path / "active.private.json"
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
        str(interpreter), "-I", "-B", str(Path(launch.__file__).with_name("preview_cutover.py")),
        "serve", "--active", str(active),
    ]
    assert observed["cwd"] == Path(launch.__file__).resolve().parents[1]
    assert observed["creationflags"] == WINDOWLESS_BREAKAWAY_FLAGS
    assert observed["stdin"] == subprocess.DEVNULL
    assert observed["close_fds"] is True
    assert observed["stdout"].closed and observed["stderr"].closed


@pytest.mark.parametrize("name", ["owner.private.log", "owner.stderr.private.log"])
def test_launch_never_overwrites_existing_output(tmp_path, monkeypatch, name):
    monkeypatch.setattr(launch.sys, "platform", "win32")
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
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.launch_preview_owner(interpreter=Path("python.exe"), active=Path("active"),
                                    output=tmp_path / "absent")
    assert failure.value.category == "owner_output_unavailable"


@pytest.mark.parametrize("outside", [True, False])
def test_actual_owner_requires_no_inherited_jobs(monkeypatch, outside):
    monkeypatch.setattr(launch, "_outside_jobs", lambda: outside)
    if outside:
        launch.assert_independent_owner()
    else:
        with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
            launch.assert_independent_owner()
        assert failure.value.category == "owner_inherited_job"


def test_unknown_membership_is_refused(monkeypatch):
    def unavailable():
        raise OSError("synthetic private membership error")

    monkeypatch.setattr(launch, "_outside_jobs", unavailable)
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.assert_independent_owner()
    assert failure.value.category == "membership_unavailable"
    assert str(failure.value) == "membership_unavailable"


def test_non_windows_launch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(launch.PreviewOwnerLaunchRefusal) as failure:
        launch.launch_preview_owner(interpreter=Path("python"), active=Path("active"),
                                    output=tmp_path)
    assert failure.value.category == "windows_required"
