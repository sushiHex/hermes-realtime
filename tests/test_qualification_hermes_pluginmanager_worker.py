"""The archived PluginManager worker runs only three literal isolated stages."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _harness(path: Path, *, duplicate: bool = False, fail: bool = False) -> None:
    values = {
        "_DISABLED_CHILD": {
            "discovered": True,
            "importOrRegistration": False,
            "sourceOriginSha256": "a" * 64,
        },
        "_ENABLE_CHILD": {
            "argvExact": True,
            "exactConfigDelta": True,
            "stdinClosed": True,
        },
        "_ENABLED_CHILD": {
            "contextObserved": True,
            "discovered": True,
            "distributionVersion": "0.0.3",
        },
    }
    lines = []
    for name, value in values.items():
        code = "raise RuntimeError('failed')" if fail and name == "_ENABLE_CHILD" else (
            "import json,os,pathlib,sys;assert os.environ['LOCALAPPDATA'].endswith('default');"
            "assert pathlib.Path.home()==pathlib.Path(os.environ['LOCALAPPDATA']);"
            "sys.stdout.buffer.write(json.dumps("
            + repr(value)
            + ",sort_keys=True,separators=(',',':')).encode()+b'\\n')"
        )
        lines.append(f"{name} = {code!r}\n")
    if duplicate:
        lines.append("def replace():\n    _DISABLED_CHILD = 'foreign'\n")
    path.write_text("".join(lines), encoding="utf-8")


def _arguments(tmp_path: Path) -> tuple[list[str], Path]:
    harness = tmp_path / "harness.py"
    _harness(harness)
    source, packages, profile, default = (
        tmp_path / name for name in ("source", "packages", "profile", "default")
    )
    for path in (source, packages, profile, default):
        path.mkdir()
    output = tmp_path / "report.json"
    inventory = '["hermes_realtime-0.0.3.dist-info/METADATA"]'
    return [
        str(harness),
        str(source),
        str(packages),
        str(profile),
        str(default),
        str(output),
        inventory,
    ], output


def test_worker_runs_exactly_three_isolated_literal_stages(tmp_path: Path) -> None:
    from scripts import qualification_hermes_pluginmanager_worker as worker

    arguments, output = _arguments(tmp_path)
    assert worker.main(arguments) == 0
    report = json.loads(output.read_bytes())
    assert report["version"] == 1 and report["pid"] > 0
    assert set(report["stages"]) == {"disabled", "enable", "enabled"}
    assert report["stages"]["disabled"]["importOrRegistration"] is False


@pytest.mark.parametrize("fault", ["duplicate", "inventory", "stage", "existing_output"])
def test_worker_refuses_changed_harness_inventory_stage_or_output(
    tmp_path: Path, fault: str
) -> None:
    from scripts import qualification_hermes_pluginmanager_worker as worker

    arguments, output = _arguments(tmp_path)
    if fault == "duplicate":
        _harness(Path(arguments[0]), duplicate=True)
    elif fault == "inventory":
        arguments[-1] = '["../foreign"]'
    elif fault == "stage":
        _harness(Path(arguments[0]), fail=True)
    else:
        output.write_bytes(b"foreign")
    assert worker.main(arguments) == 1


def test_stage_output_is_bounded_while_drained(tmp_path: Path) -> None:
    from scripts import qualification_hermes_pluginmanager_worker as worker

    environment = os.environ.copy()
    environment["TEMP"] = str(tmp_path)
    with pytest.raises(worker._StageFailure):
        worker._stage(
            "enable",
            "import sys; sys.stdout.buffer.write(b'x' * (8 * 1024 * 1024))",
            (),
            environment,
        )
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert "capture_output" not in source and ".communicate(" not in source


def test_stage_refuses_pipe_error_after_canonical_prefix(tmp_path: Path, monkeypatch) -> None:
    from scripts import qualification_hermes_pluginmanager_worker as worker

    class BrokenPipe:
        def __init__(self) -> None:
            self._first = True

        def read(self, size: int) -> bytes:
            assert size == 4096
            if self._first:
                self._first = False
                return b"{}\n"
            raise OSError("incomplete pipe")

    class Child:
        stdout = BrokenPipe()
        stderr = io.BytesIO()

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 180
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: Child())
    environment = os.environ.copy()
    environment["TEMP"] = str(tmp_path)
    with pytest.raises(worker._StageFailure):
        worker._stage("enable", "pass", (), environment)


def test_worker_process_has_no_checkout_import_requirement(tmp_path: Path) -> None:
    arguments, output = _arguments(tmp_path)
    worker = Path(__file__).parents[1] / "scripts/qualification_hermes_pluginmanager_worker.py"
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", str(worker), *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert json.loads(output.read_bytes())["version"] == 1
