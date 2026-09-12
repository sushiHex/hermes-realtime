"""The fixed worker observes the native catalog without model construction."""

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import qualification_moonshine_worker as worker

WORKER = Path(__file__).resolve().parents[1] / "scripts/qualification_moonshine_worker.py"


def _group(base: str, names: tuple[str, ...]) -> dict[str, object]:
    return {
        "base_url": base,
        "files": [
            {
                "checksum": "AQIDBA==",
                "checksum_type": "crc32c",
                "name": name,
                "size": index + 1,
                "url": f"{base}/{name}",
            }
            for index, name in enumerate(names)
        ],
    }


def _package(root: Path, *, version: str = "0.1.0") -> Path:
    packages = root / "packages"
    package = packages / "moonshine_voice"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"__version__ = {version!r}\n", encoding="utf-8", newline="\n"
    )
    primary = _group(
        "https://download.moonshine.ai/model/medium-streaming-en/quantized",
        ("a.ort", "b.ort", "c.ort", "d.ort", "e.ort", "f.json", "g.bin"),
    )
    spelling = _group(
        "https://download.moonshine.ai/model/spelling-en", ("a.ort", "b.json")
    )
    module = f'''from enum import IntEnum
import json
from pathlib import Path

class ModelArch(IntEnum):
    MEDIUM_STREAMING = 5

class _Library:
    _name = str(Path(__file__).with_name("moonshine.dll"))

class _MoonshineLib:
    lib = _Library()

_calls = 0
def moonshine_get_stt_dependencies_string(language, options):
    global _calls
    expected = {{"model_arch": 5, "include_spelling": bool(_calls)}}
    if language != "en" or options != expected or _calls >= 2:
        raise RuntimeError("worker called an unapproved native profile")
    _calls += 1
    groups = {primary!r} if not options["include_spelling"] else [{primary!r}, {spelling!r}]
    if isinstance(groups, dict):
        groups = [groups]
    return json.dumps({{"groups": groups}}, separators=(",", ":"))

class Transcriber:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("model construction is forbidden")

def download_model(*args, **kwargs):
    raise RuntimeError("model download is forbidden")
'''
    (package / "moonshine_api.py").write_text(module, encoding="utf-8", newline="\n")
    (package / "moonshine.dll").write_bytes(b"synthetic native catalog library")
    metadata = packages / f"moonshine_voice-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: moonshine-voice\nVersion: {version}\n",
        encoding="utf-8",
        newline="\n",
    )
    return packages


def _run(tmp_path: Path, *, flags: tuple[str, ...] = ("-I", "-S", "-B")):
    packages = _package(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = workspace / "moonshine-catalog.json"
    result = subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            *flags,
            str(WORKER),
            str(packages),
            str(workspace),
            str(report),
        ],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result, report


def test_worker_makes_only_the_fixed_native_catalog_call():
    calls = []
    allocations = []

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def query(language, options, count, output):
        pairs = tuple((options[index].name, options[index].value) for index in range(count))
        calls.append((language, pairs))
        payload = json.dumps({"groups": []}, separators=(",", ":")).encode()
        allocation = ctypes.create_string_buffer(payload)
        allocations.append(allocation)
        output._obj.value = ctypes.addressof(allocation)
        return 0

    freed = []
    library = type("Library", (), {})()
    library.moonshine_get_stt_dependencies = Function(query)
    library.moonshine_free_buffer = Function(lambda pointer: freed.append(pointer.value))
    assert worker._native_catalog(library, include_spelling=False) == {"groups": []}
    assert worker._native_catalog(library, include_spelling=True) == {"groups": []}
    assert calls == [
        (b"en", ((b"model_arch", b"5"),)),
        (b"en", ((b"model_arch", b"5"), (b"include_spelling", b"true"))),
    ]
    assert freed == [ctypes.addressof(allocation) for allocation in allocations]


@pytest.mark.parametrize("flags", [(), ("-I", "-B"), ("-I", "-S")])
def test_worker_requires_the_closed_python_profile(tmp_path, flags):
    result, report = _run(tmp_path, flags=flags)
    assert result.returncode == 2
    assert not report.exists()


def test_worker_refuses_wrong_distribution_before_native_catalog_call(tmp_path):
    packages = _package(tmp_path, version="0.1.1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = workspace / "moonshine-catalog.json"
    result = subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-I",
            "-S",
            "-B",
            str(WORKER),
            str(packages),
            str(workspace),
            str(report),
        ],
        cwd=workspace,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert not report.exists()


def test_worker_will_not_overwrite_a_supplied_observation(tmp_path):
    packages = _package(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = workspace / "moonshine-catalog.json"
    report.write_bytes(b'{"supplied":true}\n')
    result = subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-I",
            "-S",
            "-B",
            str(WORKER),
            str(packages),
            str(workspace),
            str(report),
        ],
        cwd=workspace,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert report.read_bytes() == b'{"supplied":true}\n'


def test_worker_frees_a_native_result_that_is_not_valid_json():
    allocation = ctypes.create_string_buffer(b"not-json")
    freed = []

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def query(language, options, count, output):
        output._obj.value = ctypes.addressof(allocation)
        return 0

    library = type("Library", (), {})()
    library.moonshine_get_stt_dependencies = Function(query)
    library.moonshine_free_buffer = Function(lambda pointer: freed.append(pointer.value))
    with pytest.raises(json.JSONDecodeError):
        worker._native_catalog(library, include_spelling=False)
    assert freed == [ctypes.addressof(allocation)]
