"""Native catalog observations must be strict and cannot be caller-minted."""

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

from scripts import qualification_moonshine_catalog as catalog


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


def _value() -> dict[str, object]:
    primary = _group(
        catalog._MODEL_BASE,
        ("one.ort", "three.ort", "two.ort", "x.bin", "y.bin", "z.bin", "zz.json"),
    )
    spelling = _group(catalog._SPELLING_BASE, ("letters.json", "letters.ort"))
    return {
        "version": 1,
        "pid": 81,
        "language": "en",
        "model_arch": 5,
        "native_library_sha256": "a" * 64,
        "native_library_bytes": 123,
        "python_api_sha256": "b" * 64,
        "python_api_bytes": 456,
        "file_origins": 50,
        "source_fallback": False,
        "without_spelling": {"groups": [primary]},
        "with_spelling": {"groups": [copy.deepcopy(primary), spelling]},
    }


def _observe(value: dict[str, object]):
    return catalog._observation(
        json.dumps(value, separators=(",", ":")).encode(),
        pid=81,
        native=("a" * 64, 123),
        python_api=("b" * 64, 456),
    )


def test_native_catalog_owns_names_while_controller_closes_profile_and_bounds():
    primary, spelling, origins, identity = _observe(_value())
    assert tuple(row.group for row in primary + spelling) == ("primary",) * 7 + (
        "spelling",
    ) * 2
    assert primary[0].cache_path.startswith(
        "download.moonshine.ai/model/medium-streaming-en/quantized/"
    )
    assert spelling[0].cache_path.startswith("download.moonshine.ai/model/spelling-en/")
    assert origins == 50 and len(identity) == 64


@pytest.mark.parametrize(
    "fault",
    [
        "pid",
        "architecture",
        "native",
        "api",
        "fallback",
        "extra",
        "primary_origin",
        "resource_url",
        "checksum_type",
        "checksum",
        "count",
        "order",
        "primary_changed",
        "spelling_origin",
        "bool_size",
    ],
)
def test_changed_or_ambiguous_native_observation_refuses(fault):
    value = _value()
    primary = value["without_spelling"]["groups"][0]
    with_primary = value["with_spelling"]["groups"][0]
    spelling = value["with_spelling"]["groups"][1]
    if fault == "pid":
        value["pid"] = 82
    elif fault == "architecture":
        value["model_arch"] = 4
    elif fault == "native":
        value["native_library_sha256"] = "f" * 64
    elif fault == "api":
        value["python_api_bytes"] = 455
    elif fault == "fallback":
        value["source_fallback"] = True
    elif fault == "extra":
        value["unbound"] = True
    elif fault == "primary_origin":
        primary["base_url"] = "https://example.invalid/model"
    elif fault == "resource_url":
        primary["files"][0]["url"] = "https://example.invalid/one.ort"
    elif fault == "checksum_type":
        primary["files"][0]["checksum_type"] = "sha256"
    elif fault == "checksum":
        primary["files"][0]["checksum"] = "not-base64"
    elif fault == "count":
        primary["files"].pop()
    elif fault == "order":
        primary["files"].reverse()
    elif fault == "primary_changed":
        with_primary["files"][0]["size"] += 1
    elif fault == "spelling_origin":
        spelling["base_url"] = catalog._MODEL_BASE
    else:
        primary["files"][0]["size"] = True
    with pytest.raises(ValueError):
        _observe(value)


def test_duplicate_observation_fields_refuse():
    raw = json.dumps(_value(), separators=(",", ":")).encode()
    raw = raw[:-1] + b',"pid":81}'
    with pytest.raises(ValueError, match="ambiguous"):
        catalog._observation(
            raw,
            pid=81,
            native=("a" * 64, 123),
            python_api=("b" * 64, 456),
        )


def test_probe_json_or_ordinary_object_cannot_mint_catalog_authority():
    with pytest.raises(TypeError):
        catalog.BoundMoonshineCatalogV1()
    with pytest.raises(TypeError):
        catalog.moonshine_catalog_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        catalog.moonshine_catalog_metadata(object.__new__(catalog.BoundMoonshineCatalogV1))
    with pytest.raises(TypeError):
        catalog._moonshine_catalog_for_consumer({"catalog": _value()}, include_spelling=True)
    with pytest.raises(ValueError, match="selection"):
        catalog._moonshine_catalog_for_consumer({}, include_spelling=1)


@pytest.mark.parametrize("change", [None, "changed", "missing"])
def test_catalog_worker_bytes_come_from_genuine_candidate_archive(
    tmp_path_factory, monkeypatch, change
):
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive

    candidate_helpers = run_path(
        str(Path.cwd() / "tests/test_qualification_candidate_files.py")
    )
    source_helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
    repository, _, old_identity, _ = candidate_helpers["_make_source"](tmp_path_factory)
    path = repository / catalog._WORKER
    expected = (Path.cwd() / "scripts/qualification_moonshine_worker.py").read_bytes()
    if change != "missing":
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = b"raise SystemExit(7)\n" if change == "changed" else expected
        path.write_bytes(expected)
        source_helpers["_git"]("add", catalog._WORKER, cwd=repository)
        source_helpers["_git"]("commit", "-qm", "synthetic catalog worker", cwd=repository)
    identity = source_helpers["_identity"](
        repository, old_identity.canonical_baseline_oid
    )
    archive = capture_candidate_source_archive(repository, identity, source_helpers["_pin"]())
    candidate = object()
    monkeypatch.setattr(
        catalog,
        "_candidate_files_for_consumer",
        lambda selected: SimpleNamespace(archive=archive, identity=identity),
    )
    if change == "missing":
        with pytest.raises(ValueError, match="worker"):
            catalog._candidate_worker(candidate)
    else:
        assert catalog._candidate_worker(candidate) == expected


def _installed_package() -> dict[str, bytes]:
    primary = _group(
        catalog._MODEL_BASE,
        ("a.ort", "b.ort", "c.ort", "d.ort", "e.ort", "f.json", "g.bin"),
    )
    spelling = _group(catalog._SPELLING_BASE, ("a.ort", "b.json"))
    api = f'''from enum import IntEnum
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
        raise RuntimeError("catalog profile differs")
    _calls += 1
    groups = [{primary!r}]
    if options["include_spelling"]:
        groups.append({spelling!r})
    return json.dumps({{"groups": groups}}, separators=(",", ":"))
'''.encode()
    return {
        "moonshine_voice/__init__.py": b'__version__ = "0.1.0"\n',
        "moonshine_voice/moonshine_api.py": api,
        "moonshine_voice/moonshine.dll": b"synthetic native catalog library",
        "moonshine_voice-0.1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: moonshine-voice\nVersion: 0.1.0\n"
        ),
    }


@dataclass
class _ControllerFixture:
    runtime: object
    value: object
    work: object
    calls: list[tuple[object, ...]]
    closure: object


@pytest.fixture
def controller(tmp_path, monkeypatch):
    from scripts.qualification_execution_files import owned_execution_files
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    contents = _installed_package()
    work = OwnedQualificationWorkV1()
    files = work.enter(owned_execution_files(contents))
    candidate = object()
    dependencies = object()
    outputs = object()
    purpose = "realtime_windows_direct_runtime"
    members = tuple(
        (name, hashlib.sha256(raw).hexdigest(), len(raw)) for name, raw in sorted(contents.items())
    )
    provider = SimpleNamespace(
        name="moonshine-voice", version="0.1.0", sha256="c" * 64, members=members
    )
    closure = SimpleNamespace(
        candidate=candidate,
        metadata=SimpleNamespace(
            qualification_input_sha256="d" * 64,
            wheelhouses=(SimpleNamespace(purpose=purpose, distributions=(provider,)),),
        ),
    )
    value = SimpleNamespace(
        work=work,
        files=files,
        tools=object(),
        dependencies=dependencies,
        outputs=outputs,
        metadata=SimpleNamespace(purpose=purpose),
    )
    runtime = object()
    calls: list[tuple[object, ...]] = []
    invocations: dict[object, object] = {}

    def installed(selected):
        assert selected is runtime
        work._accepting()
        return value

    def run_tool(self, tools, role, arguments, workspace, **kwargs):
        assert self is work and tools is value.tools and role == "build_python"
        calls.append(arguments)
        process = subprocess.Popen(
            [getattr(sys, "_base_executable", sys.executable), "-I", "-S", "-B", *arguments],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0 and stdout == stderr == b""
        receipt = object()
        invocations[receipt] = SimpleNamespace(process=SimpleNamespace(pid=process.pid))
        return receipt

    metadata = SimpleNamespace(
        qualification_input_sha256="d" * 64,
        source_commit="e" * 40,
        source_tree="f" * 40,
    )
    monkeypatch.setattr(catalog, "_installed_runtime_for_consumer", installed)
    monkeypatch.setattr(catalog, "_dependency_binding_for_consumer", lambda selected: closure)
    monkeypatch.setattr(catalog, "candidate_file_metadata", lambda selected: metadata)
    monkeypatch.setattr(
        catalog,
        "build_output_metadata",
        lambda selected: SimpleNamespace(qualification_input_sha256="d" * 64),
    )
    primary = _group(
        catalog._MODEL_BASE,
        ("a.ort", "b.ort", "c.ort", "d.ort", "e.ort", "f.json", "g.bin"),
    )
    spelling = _group(catalog._SPELLING_BASE, ("a.ort", "b.json"))
    worker = f'''import hashlib,json,os,sys
from pathlib import Path
packages=Path(sys.argv[1])
api=packages/"moonshine_voice/moonshine_api.py"
native=packages/"moonshine_voice/moonshine.dll"
value={{
    "version":1,
    "pid":os.getpid(),
    "language":"en",
    "model_arch":5,
    "native_library_sha256":hashlib.sha256(native.read_bytes()).hexdigest(),
    "native_library_bytes":native.stat().st_size,
    "python_api_sha256":hashlib.sha256(api.read_bytes()).hexdigest(),
    "python_api_bytes":api.stat().st_size,
    "file_origins":2,
    "source_fallback":False,
    "without_spelling":{{"groups":[{primary!r}]}},
    "with_spelling":{{"groups":[{primary!r},{spelling!r}]}},
}}
Path(sys.argv[3]).write_text(json.dumps(value,separators=(",",":")),encoding="utf-8")
'''.encode()
    monkeypatch.setattr(
        catalog,
        "_candidate_worker",
        lambda selected: worker,
    )
    monkeypatch.setattr(OwnedQualificationWorkV1, "run_tool", run_tool)
    monkeypatch.setattr(
        catalog, "_tool_invocation_for_consumer", lambda selected: invocations[selected]
    )
    monkeypatch.setattr(
        catalog,
        "tool_invocation_metadata",
        lambda selected: SimpleNamespace(exit_code=0),
    )
    result = _ControllerFixture(runtime, value, work, calls, closure)
    try:
        yield result
    finally:
        if not work._closed:
            work.close()


def test_controller_observes_owned_runtime_and_retains_facts_after_clean_cleanup(controller):
    receipt = catalog.observe_moonshine_catalog(controller.runtime)
    metadata = catalog.moonshine_catalog_metadata(receipt)
    assert metadata.purpose == "realtime_windows_direct_runtime"
    assert metadata.primary_resource_count == 7 and metadata.spelling_resource_count == 2
    assert len(controller.calls) == 1
    arguments = controller.calls[0]
    assert arguments[0].endswith("worker.py") and arguments[3].endswith(
        "moonshine-catalog.json"
    )
    assert len(catalog._moonshine_catalog_for_consumer(receipt, include_spelling=False)) == 7
    assert len(catalog._moonshine_catalog_for_consumer(receipt, include_spelling=True)) == 9
    controller.work.close()
    assert catalog.moonshine_catalog_metadata(receipt) == metadata
    original = catalog._dependency_binding_for_consumer
    catalog._dependency_binding_for_consumer = lambda selected: (_ for _ in ()).throw(
        ValueError("final input files are closed")
    )
    try:
        with pytest.raises(ValueError, match="closed"):
            catalog.moonshine_catalog_metadata(receipt)
    finally:
        catalog._dependency_binding_for_consumer = original


@pytest.mark.parametrize("fault", ["purpose", "source", "output"])
def test_controller_refuses_wrong_purpose_source_or_changed_output(controller, monkeypatch, fault):
    if fault == "purpose":
        controller.value.metadata.purpose = "hermes_v020_pluginmanager_runtime"
    elif fault == "source":
        monkeypatch.setattr(
            catalog,
            "_candidate_worker",
            lambda selected: (_ for _ in ()).throw(ValueError("candidate worker differs")),
        )
    else:
        original = catalog._observation

        def changed(*args, **kwargs):
            raise ValueError("catalog observation changed")

        monkeypatch.setattr(catalog, "_observation", changed)
        assert original is not changed
    with pytest.raises(ValueError):
        catalog.observe_moonshine_catalog(controller.runtime)
    assert controller.work._closing and not controller.work._closed
    if fault in {"purpose", "source"}:
        assert controller.calls == []
    with pytest.raises(RuntimeError, match="closing"):
        catalog.observe_moonshine_catalog(controller.runtime)
