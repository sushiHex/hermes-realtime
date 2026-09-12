"""Pre-final Moonshine resources require owned authenticated acquisition."""

import base64
import hashlib
import inspect
import io
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts import qualification_moonshine_preparation as preparation
from scripts import qualification_moonshine_worker as worker


class _Response:
    def __init__(self, url: str, payload: bytes, *, status: int = 200) -> None:
        self.url = url
        self.payload = io.BytesIO(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return None

    def geturl(self):
        return self.url

    def read(self, size: int):
        return self.payload.read(size)


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request, timeout))
        return self.response


def _crc(raw: bytes) -> str:
    return base64.b64encode(worker._crc32c(raw).to_bytes(4, "big")).decode("ascii")


def test_crc32c_uses_the_catalogs_castagnoli_encoding():
    assert worker._crc32c(b"123456789") == 0xE3069283
    assert _crc(b"123456789") == "4waSgw=="


def test_worker_streams_one_exact_https_resource_and_mints_sha256(tmp_path):
    payload = b"publisher model bytes"
    url = "https://download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json"
    opener = _Opener(_Response(url, payload))
    target = tmp_path / "download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json"
    digest = worker._download_resource(opener, url, target, len(payload), _crc(payload))
    assert digest == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert opener.requests[0][0].full_url == url
    assert opener.requests[0][0].get_header("User-agent") == (
        "hermes-realtime-qualification/1"
    )


@pytest.mark.parametrize("fault", ["redirect", "status", "size", "crc", "existing"])
def test_worker_refuses_changed_https_response_or_target(tmp_path, fault):
    payload = b"publisher model bytes"
    url = "https://download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json"
    response_url = "https://example.invalid/resource" if fault == "redirect" else url
    opener = _Opener(_Response(response_url, payload, status=404 if fault == "status" else 200))
    target = tmp_path / "download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json"
    if fault == "existing":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"supplied")
    expected_size = len(payload) + (1 if fault == "size" else 0)
    expected_crc = "AQIDBA==" if fault == "crc" else _crc(payload)
    with pytest.raises((ValueError, FileExistsError)):
        worker._download_resource(opener, url, target, expected_size, expected_crc)


def test_preparation_capability_cannot_be_caller_minted():
    with pytest.raises(TypeError):
        preparation.PreparedMoonshineResourcesV1()
    with pytest.raises(TypeError):
        preparation.moonshine_preparation_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        preparation.moonshine_preparation_metadata(
            object.__new__(preparation.PreparedMoonshineResourcesV1)
        )


def test_preparation_has_no_final_input_or_spelling_weakening_parameter():
    names = tuple(inspect.signature(preparation.prepare_moonshine_resources).parameters)
    assert names == ("work", "inputs", "distribution_basename", "distribution")


@pytest.mark.parametrize(
    "basename",
    [
        "moonshine_voice-0.1.1-py3-none-win_amd64.whl",
        "moonshine_voice-0.1.0-py3-none-manylinux_2_17_x86_64.whl",
        "kokoro_onnx-0.1.0-py3-none-win_amd64.whl",
    ],
)
def test_preparation_refuses_wrong_distribution_profile(monkeypatch, basename):
    bound = SimpleNamespace(metadata=SimpleNamespace(python_version="3.11.16"))
    monkeypatch.setattr(preparation, "_build_inputs_for_consumer", lambda selected: bound)
    with pytest.raises(ValueError, match="profile"):
        preparation._wheel(object(), basename, b"not-a-wheel")


def test_preparation_refuses_an_unlocked_distribution(monkeypatch):
    bound = SimpleNamespace(
        archive=object(),
        identity=object(),
        metadata=SimpleNamespace(python_version="3.11.16"),
    )
    monkeypatch.setattr(preparation, "_build_inputs_for_consumer", lambda selected: bound)
    monkeypatch.setattr(
        preparation,
        "_archive_locked_wheels",
        lambda *args: ("a" * 64, frozenset()),
    )
    with pytest.raises(ValueError, match="source lock"):
        preparation._wheel(
            object(),
            "moonshine_voice-0.1.0-py3-none-win_amd64.whl",
            b"not-the-locked-wheel",
        )


def test_preparation_refuses_forged_work_owner():
    with (
        pytest.raises(ValueError, match="owner type"),
        preparation.prepare_moonshine_resources(
            object(),
            object(),
            distribution_basename="moonshine_voice-0.1.0-py3-none-win_amd64.whl",
            distribution=b"not-a-wheel",
        ),
    ):
        raise AssertionError("unreachable")


def _controller_worker() -> tuple[bytes, dict[str, bytes]]:
    primary_names = ("a.ort", "b.ort", "c.ort", "d.ort", "e.ort", "f.json", "g.bin")
    spelling_names = ("a.ort", "b.json")
    resources = {}

    def group(base: str, names: tuple[str, ...]) -> dict[str, object]:
        files = []
        for name in names:
            payload = f"publisher:{base}:{name}".encode()
            cache_path = f"{base.removeprefix('https://')}/{name}"
            resources[cache_path] = payload
            files.append(
                {
                    "checksum": _crc(payload),
                    "checksum_type": "crc32c",
                    "name": name,
                    "size": len(payload),
                    "url": f"{base}/{name}",
                }
            )
        return {"base_url": base, "files": files}

    primary = group(worker._MODEL_BASE, primary_names)
    spelling = group(worker._SPELLING_BASE, spelling_names)
    script = f'''import hashlib,json,os,sys
from pathlib import Path
packages=Path(sys.argv[1])
root=Path(sys.argv[3])
resources={resources!r}
rows=[]
for cache_path,payload in resources.items():
    target=root.joinpath(*cache_path.split("/"))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(payload)
    rows.append({{
        "cache_path":cache_path,
        "url":"https://"+cache_path,
        "size":len(payload),
        "crc32c":{dict((key, _crc(value)) for key, value in resources.items())!r}[cache_path],
        "sha256":hashlib.sha256(payload).hexdigest(),
    }})
api=packages/"moonshine_voice/moonshine_api.py"
native=packages/"moonshine_voice/moonshine.dll"
catalog={{
    "version":1,"pid":os.getpid(),"language":"en","model_arch":5,
    "native_library_sha256":hashlib.sha256(native.read_bytes()).hexdigest(),
    "native_library_bytes":native.stat().st_size,
    "python_api_sha256":hashlib.sha256(api.read_bytes()).hexdigest(),
    "python_api_bytes":api.stat().st_size,
    "file_origins":2,"source_fallback":False,
    "without_spelling":{{"groups":[{primary!r}]}},
    "with_spelling":{{"groups":[{primary!r},{spelling!r}]}},
}}
(root/"moonshine-preparation.json").write_text(
    json.dumps({{"catalog":catalog,"resources":rows}},separators=(",",":")),encoding="utf-8")
'''.encode()
    return script, resources


@pytest.fixture
def controller(monkeypatch):
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    work = OwnedQualificationWorkV1()
    inputs = object()
    tools = object()
    source = SimpleNamespace(
        source_commit="a" * 40,
        source_tree="b" * 40,
        source_archive_sha256="c" * 64,
        python_version="3.11.16",
    )
    bound = SimpleNamespace(
        tools=tools,
        archive=object(),
        identity=object(),
        metadata=source,
    )
    api = b"fixed python API"
    native = b"fixed native library"
    contents = {
        "moonshine_voice/moonshine_api.py": api,
        "moonshine_voice/moonshine.dll": native,
    }
    members = tuple(
        (name, hashlib.sha256(raw).hexdigest(), len(raw))
        for name, raw in sorted(contents.items())
    )
    package = SimpleNamespace(sha256="d" * 64, members=members)
    archived_worker, resources = _controller_worker()
    invocations = {}

    def run_tool(self, selected_tools, role, arguments, workspace, **kwargs):
        assert self is not work and selected_tools is tools and role == "build_python"
        assert arguments[-1] == "acquire" and kwargs["timeout_milliseconds"] >= 60_000
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

    monkeypatch.setattr(preparation, "_build_inputs_for_consumer", lambda selected: bound)
    monkeypatch.setattr(preparation, "_build_input_facts", lambda selected: bound)
    monkeypatch.setattr(preparation, "_wheel", lambda *args: (package, contents))
    monkeypatch.setattr(preparation, "_archive_worker", lambda *args: archived_worker)
    monkeypatch.setattr(OwnedQualificationWorkV1, "run_tool", run_tool)
    monkeypatch.setattr(
        preparation, "_tool_invocation_for_consumer", lambda selected: invocations[selected]
    )
    monkeypatch.setattr(
        preparation, "tool_invocation_metadata", lambda selected: SimpleNamespace(exit_code=0)
    )
    try:
        yield SimpleNamespace(
            work=work,
            inputs=inputs,
            resources=resources,
            package=package,
        )
    finally:
        if not work._closed:
            work.close()


def test_owned_preparation_retains_sealed_resources_after_execution_cleanup(controller):
    context = preparation.prepare_moonshine_resources(
        controller.work,
        controller.inputs,
        distribution_basename="moonshine_voice-0.1.0-py3-none-win_amd64.whl",
        distribution=b"source-locked wheel",
    )
    with context as receipt:
        metadata = preparation.moonshine_preparation_metadata(receipt)
        assert metadata.resource_count == 9
        assert metadata.distribution_sha256 == controller.package.sha256
        selected = preparation._moonshine_preparation_for_consumer(receipt)
        resource_root = selected.resource_root
        first = selected.resources[0]
        assert preparation._prepared_resource_bytes(receipt, first.cache_path) == (
            controller.resources[first.cache_path]
        )
        controller.work.close()
        assert selected.execution_work._closed and resource_root.is_dir()
        assert preparation.moonshine_preparation_metadata(receipt) == metadata
        assert preparation._prepared_resource_bytes(receipt, first.cache_path) == (
            controller.resources[first.cache_path]
        )
    assert not resource_root.exists()
    with pytest.raises(ValueError, match="unregistered"):
        preparation.moonshine_preparation_metadata(receipt)


def test_changed_preparation_output_latches_execution_owner(controller, monkeypatch):
    monkeypatch.setattr(
        preparation,
        "_prepared_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("changed report")),
    )
    with (
        pytest.raises(ValueError, match="changed"),
        preparation.prepare_moonshine_resources(
            controller.work,
            controller.inputs,
            distribution_basename="moonshine_voice-0.1.0-py3-none-win_amd64.whl",
            distribution=b"source-locked wheel",
        ),
    ):
        raise AssertionError("unreachable")
    assert controller.work._closing and not controller.work._closed


def test_incomplete_execution_cleanup_retains_resource_owner_and_tool_work(
    controller, monkeypatch
):
    from scripts.qualification_owned_work import (
        OwnedQualificationCleanupError,
        OwnedQualificationWorkV1,
    )

    created = []
    paths = []
    original_init = OwnedQualificationWorkV1.__init__
    original_close = OwnedQualificationWorkV1.close
    original_workspace = preparation._workspace
    blocked = True

    def initialize(self):
        original_init(self)
        created.append(self)

    def close(self):
        if blocked and len(created) >= 2 and self is created[1]:
            raise OwnedQualificationCleanupError(self)
        return original_close(self)

    @contextmanager
    def workspace():
        with original_workspace() as path:
            paths.append(path)
            yield path

    monkeypatch.setattr(OwnedQualificationWorkV1, "__init__", initialize)
    monkeypatch.setattr(OwnedQualificationWorkV1, "close", close)
    monkeypatch.setattr(preparation, "_workspace", workspace)
    with (
        pytest.raises(OwnedQualificationCleanupError),
        preparation.prepare_moonshine_resources(
            controller.work,
            controller.inputs,
            distribution_basename="moonshine_voice-0.1.0-py3-none-win_amd64.whl",
            distribution=b"source-locked wheel",
        ),
    ):
        raise AssertionError("unreachable")
    assert len(created) == 2 and len(controller.work._children) == 1
    resource_work, execution_work = created
    assert controller.work._children == [resource_work]
    assert resource_work._children == [execution_work]
    assert paths[0].is_dir()
    blocked = False
    controller.work.close()
    assert all(not path.exists() for path in paths)
