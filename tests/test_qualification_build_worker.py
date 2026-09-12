"""Exercise the isolated worker protocol; synthetic packages are not build proof."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "scripts/qualification_build_worker.py"


@pytest.fixture
def layout(tmp_path):
    packages = tmp_path / "packages"
    packages.mkdir()
    for name in ("hatchling", "packaging", "pathspec", "pluggy", "trove_classifiers"):
        package = packages / name
        package.mkdir()
        (package / "__init__.py").write_text("# synthetic worker fixture\n")
    (packages / "hatchling/build.py").write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "def build_wheel(output):\n"
        "    name='hermes_realtime-0.0.3-py3-none-any.whl'\n"
        "    with zipfile.ZipFile(Path(output,name), 'w') as wheel:\n"
        "        info=zipfile.ZipInfo('payload.txt', (2024,1,2,3,4,6))\n"
        "        info.create_system=0\n"
        "        info.comment=b'variant'\n"
        "        wheel.writestr(info,b'synthetic wheel, not qualification',"
        "compress_type=zipfile.ZIP_DEFLATED)\n"
        "        wheel.writestr('hermes_realtime-0.0.3.dist-info/RECORD',"
        "b'exact record payload\\n')\n"
        "    return name\n"
        "def build_sdist(output):\n"
        "    name='hermes_realtime-0.0.3.tar.gz'\n"
        "    Path(output,name).write_bytes(b'synthetic sdist, not qualification')\n"
        "    return name\n"
    )
    source = tmp_path / "source"
    source.mkdir()
    # A source module must never shadow the installed backend under -I -S.
    (source / "hatchling.py").write_text("raise RuntimeError('unbound source fallback')\n")
    output = tmp_path / "output"
    output.mkdir()
    return packages, source, output, tmp_path / "observation.json"


def invoke(layout, kind, *, isolated=True):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(layout[1])
    return subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            *(["-I", "-S", "-B"] if isolated else []),
            str(WORKER),
            kind,
            *map(str, layout),
        ],
        cwd=layout[1],
        env=environment,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("kind", ["imports", "wheel", "sdist"])
def test_worker_uses_installed_namespace_and_reports_only_closed_metadata(layout, kind):
    result = invoke(layout, kind)
    assert result.returncode == 0
    assert result.stdout == result.stderr == b""
    observed = json.loads(layout[3].read_bytes())
    assert observed["version"] == 1 and observed["kind"] == kind
    assert type(observed["pid"]) is int and observed["pid"] > 0
    assert observed["imports"] == 5 and observed["source_fallback"] is False
    assert 5 <= observed["file_origins"] <= 16384
    expected = {
        "imports": None,
        "wheel": "hermes_realtime-0.0.3-py3-none-any.whl",
        "sdist": "hermes_realtime-0.0.3.tar.gz",
    }[kind]
    assert observed["artifact"] == expected
    assert set(observed) == {
        "version",
        "kind",
        "pid",
        "imports",
        "file_origins",
        "source_fallback",
        "artifact",
    }
    if kind == "wheel":
        with zipfile.ZipFile(layout[2] / expected) as wheel:
            assert wheel.comment == b""
            assert wheel.namelist() == sorted(wheel.namelist())
            assert wheel.read("hermes_realtime-0.0.3.dist-info/RECORD") == b"exact record payload\n"
            assert all(
                item.create_system == 3
                and item.compress_type == zipfile.ZIP_STORED
                and item.date_time == (2020, 2, 2, 0, 0, 0)
                for item in wheel.infolist()
            )
    assert not list(layout[0].rglob("*.pyc"))


def test_worker_never_processes_installed_path_configuration(layout):
    marker = layout[2] / "hook-ran.txt"
    (layout[0] / "synthetic.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('hook executed')\n"
    )
    result = invoke(layout, "imports")
    assert result.returncode == 0 and not marker.exists()
    # Positive control: the same hook executes if site processing is requested.
    control = subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-I",
            "-S",
            "-B",
            "-c",
            "import site,sys; site.addsitedir(sys.argv[1])",
            str(layout[0]),
        ],
        capture_output=True,
        timeout=10,
    )
    assert control.returncode == 0 and marker.read_text() == "hook executed"


@pytest.mark.parametrize("fault", ["kind", "isolation", "origin", "artifact"])
def test_worker_refuses_unbound_or_incorrect_execution_without_private_output(layout, fault):
    if fault == "origin":
        (layout[0] / "hatchling/__init__.py").write_text(
            '__file__=__import__("sys").argv[3]+"/hatchling.py"\n'
        )
    if fault == "artifact":
        (layout[0] / "hatchling/build.py").write_text(
            "def build_wheel(output): return '../../private-fixture.whl'\n"
        )
    result = invoke(
        layout, "private-fixture" if fault == "kind" else "wheel", isolated=fault != "isolation"
    )
    assert result.returncode != 0
    assert result.stdout == result.stderr == b""
    assert not layout[3].exists()


@pytest.mark.parametrize(
    "purpose",
    [
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
        "hermes_v020_pluginmanager_runtime",
    ],
)
@pytest.mark.parametrize("fault", [None, "missing", "origin", "version"])
def test_runtime_import_profile_uses_only_installed_candidate_modules(layout, purpose, fault):
    package = layout[0] / "hermes_realtime"
    package.mkdir()
    (package / "__init__.py").write_text(
        '__version__ = "0.0.4"\n' if fault == "version" else '__version__ = "0.0.3"\n'
    )
    for name in ("host_launcher", "launcher", "hermes_plugin"):
        (package / (name + ".py")).write_text(
            "def main(): raise AssertionError('import must not activate a runtime')\n"
        )
    if fault == "missing":
        (package / "hermes_plugin.py").unlink()
    elif fault == "origin":
        (package / "hermes_plugin.py").write_text(
            '__file__=__import__("sys").argv[3]+"/hatchling.py"\n'
        )
    # Build dependencies have no place in a runtime import recipe.
    (layout[0] / "hatchling/__init__.py").write_text("raise AssertionError('build fallback')\n")
    result = invoke(layout, purpose)
    if fault is None:
        assert result.returncode == 0
        observed = json.loads(layout[3].read_bytes())
        assert observed["imports"] == 4 and observed["kind"] == purpose
        assert observed["artifact"] is None and observed["source_fallback"] is False
    else:
        assert result.returncode != 0 and not layout[3].exists()
    assert result.stdout == result.stderr == b""
