"""Offline binding and owned readiness checks; never launch Ollama or contact HTTP."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import lookup_backend_model as model


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def selected(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        members = {"ollama.exe": b"synthetic-executable", "lib/runner.dll": b"runner"}
        for name, raw in members.items():
            bundle.writestr(name, raw)
            path = runtime / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
    store = tmp_path / "models"
    blobs = store / "blobs"
    blobs.mkdir(parents=True)

    def blob(kind, raw):
        identity = digest(raw)
        (blobs / ("sha256-" + identity)).write_bytes(raw)
        return {"mediaType": kind, "digest": "sha256:" + identity, "size": len(raw)}

    layers = [blob("application/vnd.ollama.image.model", b"synthetic-gguf"),
              blob("application/vnd.ollama.image.template", b"synthetic-template"),
              blob("application/vnd.ollama.image.params", b"{}")]
    config = blob("application/vnd.docker.container.image.v1+json", json.dumps({
        "model_format": "gguf", "model_family": "synthetic",
        "rootfs": {"diff_ids": [row["digest"] for row in layers]},
    }).encode())
    manifest_raw = json.dumps({"schemaVersion": 2, "config": config, "layers": layers}).encode()
    manifest = store / "manifests/registry.ollama.ai/library/local-model/tag"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(manifest_raw)
    return {"model": "local-model:tag", "model_store": str(store),
        "model_manifest_sha256": digest(manifest_raw), "model_port": 11440,
        "ollama_runtime": str(runtime), "ollama_archive": str(archive),
        "ollama_archive_sha256": digest(archive.read_bytes())}


def verify(selected):
    return model.verify_artifacts(selected, model.time.monotonic() + 10).facts


def test_complete_artifact_binding(selected):
    value = verify(selected)
    assert value["runtime_files"] == 2
    assert value["model_blob_count"] == 4
    assert value["model_manifest_sha256"] == selected["model_manifest_sha256"]
    assert value["model_bytes"] == len(b"synthetic-gguf")


def test_archive_digest_refused(selected):
    selected["ollama_archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive-digest"):
        verify(selected)


def test_runtime_member_set_refused(selected):
    (Path(selected["ollama_runtime"]) / "extra.dll").write_bytes(b"extra")
    with pytest.raises(ValueError, match="runtime-content"):
        verify(selected)


def test_runtime_content_refused(selected):
    (Path(selected["ollama_runtime"]) / "ollama.exe").write_bytes(b"different-executable")
    with pytest.raises(ValueError, match="runtime-content"):
        verify(selected)


def test_archive_path_escape_refused(selected):
    archive = Path(selected["ollama_archive"])
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("../outside.dll", b"escape")
    selected["ollama_archive_sha256"] = digest(archive.read_bytes())
    with pytest.raises(ValueError):
        verify(selected)


def test_archive_duplicate_refused(selected):
    archive = Path(selected["ollama_archive"])
    with zipfile.ZipFile(archive, "a") as bundle, pytest.warns(UserWarning):
        bundle.writestr("ollama.exe", b"synthetic-executable")
    selected["ollama_archive_sha256"] = digest(archive.read_bytes())
    with pytest.raises(ValueError, match="archive-member-kind"):
        verify(selected)


def test_archive_link_refused(selected):
    archive = Path(selected["ollama_archive"])
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("ollama.exe")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        bundle.writestr(info, b"synthetic-executable")
        bundle.writestr("lib/runner.dll", b"runner")
    selected["ollama_archive_sha256"] = digest(archive.read_bytes())
    with pytest.raises(ValueError, match="archive-member-kind"):
        verify(selected)


def test_reparse_point_refused():
    path = SimpleNamespace(lstat=lambda: SimpleNamespace(st_mode=0o100644,
                                                       st_file_attributes=0x400))
    with pytest.raises(ValueError, match="artifact-file-kind"):
        model._ordinary(path)


def test_model_manifest_digest_refused(selected):
    selected["model_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model-manifest-digest"):
        verify(selected)


def test_model_blob_size_refused(selected):
    path = Path(selected["model_store"]) / "manifests/registry.ollama.ai/library/local-model/tag"
    content = json.loads(path.read_bytes())
    content["layers"][0]["size"] += 1
    path.write_text(json.dumps(content))
    selected["model_manifest_sha256"] = digest(path.read_bytes())
    with pytest.raises(ValueError, match="model-blob-size"):
        verify(selected)


def test_model_blob_digest_refused(selected):
    blobs = Path(selected["model_store"]) / "blobs"
    path = next(path for path in blobs.iterdir() if path.read_bytes() == b"synthetic-gguf")
    path.write_bytes(b"synthetic-FAKE")
    with pytest.raises(ValueError, match="model-blob-digest"):
        verify(selected)


def rewrite_blob(selected, change, *, params=False):
    store = Path(selected["model_store"])
    path = store / "manifests/registry.ollama.ai/library/local-model/tag"
    manifest = json.loads(path.read_bytes())
    config_path = store / "blobs" / manifest["config"]["digest"].replace(":", "-")
    config = json.loads(config_path.read_bytes())
    if params:
        row = manifest["layers"][-1]
        raw = json.dumps(change).encode()
        row.update(digest="sha256:" + digest(raw), size=len(raw))
        (store / "blobs" / row["digest"].replace(":", "-")).write_bytes(raw)
        config["rootfs"]["diff_ids"] = [item["digest"] for item in manifest["layers"]]
    else:
        config.update(change)
    raw = json.dumps(config).encode()
    manifest["config"].update(digest="sha256:" + digest(raw), size=len(raw))
    (store / "blobs" / manifest["config"]["digest"].replace(":", "-")).write_bytes(raw)
    path.write_text(json.dumps(manifest))
    selected["model_manifest_sha256"] = digest(path.read_bytes())


def rewrite_params_layers(selected, count):
    store = Path(selected["model_store"])
    path = store / "manifests/registry.ollama.ai/library/local-model/tag"
    manifest = json.loads(path.read_bytes())
    params = [row for row in manifest["layers"]
              if row["mediaType"] == "application/vnd.ollama.image.params"]
    manifest["layers"] = [row for row in manifest["layers"]
                          if row["mediaType"] != "application/vnd.ollama.image.params"]
    manifest["layers"].extend(params * count)
    config_path = store / "blobs" / manifest["config"]["digest"].replace(":", "-")
    config = json.loads(config_path.read_bytes())
    config["rootfs"]["diff_ids"] = [row["digest"] for row in manifest["layers"]]
    raw = json.dumps(config).encode()
    manifest["config"].update(digest="sha256:" + digest(raw), size=len(raw))
    (store / "blobs" / manifest["config"]["digest"].replace(":", "-")).write_bytes(raw)
    path.write_text(json.dumps(manifest))
    selected["model_manifest_sha256"] = digest(path.read_bytes())


def test_stock_shape_without_params_is_admitted(selected):
    rewrite_params_layers(selected, 0)
    assert verify(selected)["model_blob_count"] == 3


def test_duplicate_params_layers_are_refused(selected):
    rewrite_params_layers(selected, 2)
    with pytest.raises(ValueError, match="model-layer-layout"):
        verify(selected)


def test_model_configuration_refused(selected):
    rewrite_blob(selected, {"renderer": "other-route"})
    with pytest.raises(ValueError, match="model-config"):
        verify(selected)


def test_artifact_context_override_refused(selected):
    rewrite_blob(selected, {"num_ctx": 4096}, params=True)
    with pytest.raises(ValueError, match="artifact-context-override"):
        verify(selected)


@pytest.mark.parametrize("name", ["../bad:tag", "name:../bad", "name/tag:tag"])
def test_model_reference_cannot_escape(selected, name):
    selected["model"] = name
    with pytest.raises(ValueError, match="model-reference"):
        verify(selected)


class Owner:
    def __init__(self):
        self.roots = {}
        self.calls = []
        self.killed = []

    def spawn(self, name, argv, **kwargs):
        self.calls.append((name, argv, kwargs))
        process = SimpleNamespace(pid=4321, poll=lambda: None)
        self.roots[name] = process
        return process

    def kill(self, name):
        self.killed.append(name)


@pytest.fixture
def owned(selected, monkeypatch):
    owner = Owner()
    monkeypatch.setattr(model, "_listeners", lambda port, deadline:
        [(f"127.0.0.1:{port}", 4321)] if owner.calls else [])
    monkeypatch.setattr(model, "_image",
                        lambda pid: Path(selected["ollama_runtime"]) / "ollama.exe")
    replies = [{"version": "0.40.4"}, {"models": [{"name": selected["model"],
        "digest": selected["model_manifest_sha256"]}]}, {"template": "synthetic-template",
        "details": {"format": "gguf", "family": "synthetic"},
        "capabilities": ["completion", "tools"]}]
    calls = []

    class Response(io.BytesIO):
        status = 200

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            assert 0 < timeout <= 8
            return Response(json.dumps(replies[len(calls) - 1]).encode())

    monkeypatch.setattr(model.urllib.request, "build_opener", lambda *args: Opener())
    ready_calls = []
    ready = {"load": {"model": selected["model"], "done": True, "done_reason": "load",
                      "message": {"role": "assistant", "content": ""}},
             "ps": {"models": [{"model": selected["model"], "name": selected["model"],
                 "digest": selected["model_manifest_sha256"], "context_length": 64000}]}}

    class Dribble(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{"
            await asyncio.sleep(2)
            yield b"}"

        async def aclose(self):
            ready["stream_closed"] = True

    class Transport(httpx.MockTransport):
        async def aclose(self):
            ready["client_closed"] = True
            await super().aclose()

    async def ready_reply(request):
        ready_calls.append(request)
        kind = "load" if request.url.path == "/api/chat" else "ps"
        if kind == "load" and ready.get("after_load"):
            ready["after_load"]()
        if kind == "ps" and ready.get("after_ps"):
            ready["after_ps"]()
        if ready.get("dribble"):
            return httpx.Response(200, stream=Dribble())
        if ready.get("oversize"):
            return httpx.Response(200, content=b"x" * (1024 * 1024 + 1))
        if ready.get("redirect"):
            return httpx.Response(302, headers={"location": "http://example.invalid/api/chat"})
        return httpx.Response(200, json=ready[kind])

    def transport(**kwargs):
        assert kwargs == {"retries": 0}
        return Transport(ready_reply)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    containment = SimpleNamespace(members=lambda: {os.getpid()})
    return owner, containment, calls, replies, ready_calls, ready


_DEFAULT_PROFILE = object()


def start(selected, owned, tmp_path, *, startup_profile=_DEFAULT_PROFILE):
    owner, containment = owned[:2]
    kwargs = {} if startup_profile is _DEFAULT_PROFILE else {"startup_profile": startup_profile}
    return model.start_model(selected, owner, containment, tmp_path / "evidence",
                             model.time.monotonic() + 10, **kwargs)


def test_native_tools_profile_binds_environment_load_and_readback(
        selected, owned, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_GO_TEMPLATE", "0")
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "9000")
    owned[5]["ps"]["models"][0]["context_length"] = 4096
    result = start(selected, owned, tmp_path,
                   startup_profile=model.StartupProfile.NATIVE_TOOLS_4K)
    env = owned[0].calls[0][2]["env"]
    assert env["OLLAMA_CONTEXT_LENGTH"] == "4096"
    assert env["OLLAMA_GO_TEMPLATE"] == "1"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert json.loads(owned[4][0].content)["options"] == {"num_ctx": 4096}
    assert env["OLLAMA_KEEP_ALIVE"] == "-1"
    assert "keep_alive" not in json.loads(owned[4][0].content)
    assert result["context_default_tokens"] == result["configured_context_tokens"] == 4096
    assert result["startup_profile"] == "NATIVE_TOOLS_4K"


def test_native_startup_admits_load_after_sixty_before_120(
        selected, owned, tmp_path, monkeypatch):
    now = [1000.0]
    monotonic_ns = model.time.monotonic_ns
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=monotonic_ns))
    proof = model.verify_artifacts(selected, 1200.0)
    owned[5]["ps"]["models"][0]["context_length"] = 4096
    owned[5]["after_load"] = lambda: now.__setitem__(0, 1065.0)
    result = model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
        1200.0, verified=proof, startup_profile=model.StartupProfile.NATIVE_TOOLS_4K)
    assert result["model_ready"] is True
    assert result["startup_budget_ceiling_seconds"] == 120
    assert result["startup_timeout_requested_seconds"] is None
    assert result["startup_effective_budget_seconds"] == 120
    assert len(owned[4]) == 2


def test_native_startup_parent_deadline_wins(selected, owned, tmp_path, monkeypatch):
    now = [1000.0]
    monotonic_ns = model.time.monotonic_ns
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=monotonic_ns))
    proof = model.verify_artifacts(selected, 1200.0)
    owned[5]["ps"]["models"][0]["context_length"] = 4096
    owned[5]["after_load"] = lambda: now.__setitem__(0, 1031.0)
    with pytest.raises(TimeoutError, match="model-deadline"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            1030.0, verified=proof,
            startup_profile=model.StartupProfile.NATIVE_TOOLS_4K)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]
    assert json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())[
        "startup_budget_ceiling_seconds"] == 120


def test_native_startup_budget_is_shared_after_metadata(
        selected, owned, tmp_path, monkeypatch):
    now = [1000.0]
    monotonic_ns = model.time.monotonic_ns
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=monotonic_ns))
    proof = model.verify_artifacts(selected, 1200.0)
    owned[5]["ps"]["models"][0]["context_length"] = 4096
    original_read = model._read

    def metadata_read(*args):
        result = original_read(*args)
        if args[0] == "show":
            now[0] = 1030.0
        return result

    monkeypatch.setattr(model, "_read", metadata_read)
    owned[5]["after_load"] = lambda: now.__setitem__(0, 1121.0)
    with pytest.raises(TimeoutError, match="model-deadline"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            1200.0, verified=proof,
            startup_profile=model.StartupProfile.NATIVE_TOOLS_4K)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_default_lookup_startup_stays_sixty_seconds(
        selected, owned, tmp_path, monkeypatch):
    now = [1000.0]
    monotonic_ns = model.time.monotonic_ns
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=monotonic_ns))
    proof = model.verify_artifacts(selected, 1200.0)
    owned[5]["after_load"] = lambda: now.__setitem__(0, 1061.0)
    with pytest.raises(TimeoutError, match="model-deadline"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            1200.0, verified=proof)
    state = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert state["startup_budget_ceiling_seconds"] == 60
    assert state["startup_timeout_requested_seconds"] is None
    assert state["startup_effective_budget_seconds"] == 60
    assert state["context_default_tokens"] == 64000
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


@pytest.mark.parametrize("seconds", [1, 300, 600])
def test_explicit_startup_timeout_and_receipt(selected, owned, tmp_path, monkeypatch, seconds):
    now = [1000.0]
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=model.time.monotonic_ns))
    proof = model.verify_artifacts(selected, 1700.0)
    owned[5]["after_load"] = lambda: now.__setitem__(0, 1000.0 + seconds - 0.5)
    result = model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
        1700.0, verified=proof, startup_timeout_seconds=seconds)
    assert result["model_ready"] is True
    assert result["startup_timeout_requested_seconds"] == seconds
    assert result["startup_budget_ceiling_seconds"] == seconds
    assert result["startup_effective_budget_seconds"] == seconds
    assert result["context_default_tokens"] == 64000
    assert len(owned[4]) == 2


@pytest.mark.parametrize("caller_budget, load_elapsed, effective", [(30, 31, 30), (500, 301, 300)])
def test_explicit_startup_timeout_and_caller_bound_are_shared(
        selected, owned, tmp_path, monkeypatch, caller_budget, load_elapsed, effective):
    now = [1000.0]
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=model.time.monotonic_ns))
    proof = model.verify_artifacts(selected, 1700.0)
    owned[5]["after_load"] = lambda: now.__setitem__(0, 1000.0 + load_elapsed)
    with pytest.raises(TimeoutError, match="model-deadline"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            1000.0 + caller_budget, verified=proof, startup_timeout_seconds=300)
    receipt = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert receipt["startup_timeout_requested_seconds"] == 300
    assert receipt["startup_budget_ceiling_seconds"] == 300
    assert receipt["startup_effective_budget_seconds"] == effective
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


class _StartupSeconds(int):
    pass


@pytest.mark.parametrize("seconds", [
    pytest.param(True, id="bool"), pytest.param(300.0, id="float"),
    pytest.param(float("inf"), id="infinity"), pytest.param(float("nan"), id="nan"),
    pytest.param(_StartupSeconds(300), id="subclass"), pytest.param(0, id="zero"),
    pytest.param(601, id="over-cap"), pytest.param(object(), id="object"),
])
def test_invalid_startup_timeout_refuses_without_serializing_input(
        selected, owned, tmp_path, capsys, seconds):
    with pytest.raises(ValueError, match="startup-timeout"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 700, startup_timeout_seconds=seconds)
    receipt = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert receipt["failure_category"] == "startup-timeout"
    assert receipt["startup_timeout_requested_seconds"] is None
    assert receipt["startup_effective_budget_seconds"] is None
    assert receipt["keep_alive_configured"] == -1 and receipt["load_request_count"] == 0
    assert not owned[0].calls and not owned[4]
    assert json.loads(capsys.readouterr().out.removeprefix("[lookup-model] "))[
        "failure_category"] == "startup-timeout"


def test_startup_profile_settings_cannot_drift(selected, owned, tmp_path):
    profile = model.StartupProfile.NATIVE_TOOLS_4K
    with pytest.raises(AttributeError):
        profile.context_tokens = 64000
    with pytest.raises(AttributeError):
        profile.go_template = None
    with pytest.raises(AttributeError):
        profile.profile_id = "LOOKUP_64K"
    with pytest.raises(AttributeError):
        profile.startup_seconds = 60
    with pytest.raises(AttributeError):
        profile.value = (64000, None)
    original_value, original_name = profile._value_, profile._name_
    try:
        profile._value_ = (64000, None)
        profile._name_ = "LOOKUP_64K"
        owned[5]["ps"]["models"][0]["context_length"] = 4096
        result = start(selected, owned, tmp_path, startup_profile=profile)
    finally:
        profile._value_, profile._name_ = original_value, original_name
    assert result["configured_context_tokens"] == 4096
    assert result["startup_profile"] == "NATIVE_TOOLS_4K"
    assert result["startup_budget_ceiling_seconds"] == 120
    assert owned[0].calls[0][2]["env"]["OLLAMA_GO_TEMPLATE"] == "1"


@pytest.mark.parametrize("invalid", [None, "NATIVE_TOOLS_4K", 4096, object()])
def test_invalid_startup_profile_refused_before_spawn(selected, owned, tmp_path, invalid):
    with pytest.raises(ValueError, match="startup-profile"):
        start(selected, owned, tmp_path, startup_profile=invalid)
    assert not owned[0].calls
    receipt = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert receipt["category"] == "refused"
    assert receipt["failure_category"] == "startup-profile"
    assert receipt["startup_profile"] == "invalid"


@pytest.mark.parametrize("context", [64000, 4096.0, None])
def test_native_tools_profile_requires_exact_readback(selected, owned, tmp_path, context):
    owned[5]["ps"]["models"][0]["context_length"] = context
    with pytest.raises(ValueError, match="loaded-context"):
        start(selected, owned, tmp_path,
              startup_profile=model.StartupProfile.NATIVE_TOOLS_4K)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_owned_startup_and_closed_options(selected, owned, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_GO_TEMPLATE", "1")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")
    monkeypatch.setenv("LLAMA_ARG_GPU_LAYERS", "0")
    result = start(selected, owned, tmp_path)
    owner, _, calls, _ = owned[:4]
    assert result["category"] == "metadata_ready"
    assert result["inference_request_count"] == 0
    assert [request.full_url for request in calls] == [
        "http://127.0.0.1:11440/api/version", "http://127.0.0.1:11440/api/tags",
        "http://127.0.0.1:11440/api/show"]
    assert [request.get_method() for request in calls] == ["GET", "GET", "POST"]
    assert json.loads(calls[-1].data) == {"model": selected["model"]}
    assert len(owner.calls) == 1 and not owner.killed
    env = owner.calls[0][2]["env"]
    assert "OLLAMA_GO_TEMPLATE" not in env and "LLAMA_ARG_GPU_LAYERS" not in env
    assert env["OLLAMA_CONTEXT_LENGTH"] == "64000"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_KEEP_ALIVE"] == "-1"


def test_foreign_containment_refused_before_start(selected, owned, tmp_path):
    owned[1].members = lambda: set()
    with pytest.raises(RuntimeError, match="owner-containment"):
        start(selected, owned, tmp_path)
    assert not owned[0].calls
    assert not owned[4]


def test_occupied_port_never_adopted(selected, owned, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_listeners", lambda *args: [("127.0.0.1:11440", 999)])
    with pytest.raises(RuntimeError, match="port-occupied"):
        start(selected, owned, tmp_path)
    assert not owned[0].calls
    assert not owned[4]


@pytest.mark.parametrize("listener", [("0.0.0.0:11440", 4321), ("127.0.0.1:11440", 999)])
def test_listener_must_match_owned_process(selected, owned, tmp_path, monkeypatch, listener):
    monkeypatch.setattr(model, "_listeners", lambda *args: [listener] if owned[0].calls else [])
    with pytest.raises(RuntimeError, match="listener-owner"):
        start(selected, owned, tmp_path)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]
    assert not owned[2]
    assert not owned[4]


def test_process_image_must_match_runtime(selected, owned, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_image", lambda pid: tmp_path / "foreign.exe")
    with pytest.raises(RuntimeError, match="listener-owner"):
        start(selected, owned, tmp_path)
    assert not owned[2]
    assert not owned[4]


def test_show_template_bound_to_artifact(selected, owned, tmp_path):
    owned[3][-1]["template"] = "different-template"
    with pytest.raises(ValueError, match="metadata-template"):
        start(selected, owned, tmp_path)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_version_refused_before_further_requests(selected, owned, tmp_path):
    owned[3][0]["version"] = "different-version"
    with pytest.raises(ValueError, match="metadata-version"):
        start(selected, owned, tmp_path)
    assert len(owned[2]) == 1


def test_served_model_digest_refused(selected, owned, tmp_path):
    owned[3][1]["models"][0]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="metadata-model"):
        start(selected, owned, tmp_path)


def test_served_capabilities_refused(selected, owned, tmp_path):
    owned[3][-1]["capabilities"] = ["completion"]
    with pytest.raises(ValueError, match="metadata-capabilities"):
        start(selected, owned, tmp_path)


def test_unknown_failure_does_not_publish_details(selected, owned, tmp_path, monkeypatch, capsys):
    def refuse(*args):
        raise OSError("private-untrusted-error-detail")
    monkeypatch.setattr(model, "verify_artifacts", refuse)
    with pytest.raises(OSError):
        start(selected, owned, tmp_path)
    receipt = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert receipt["stage"] == "artifacts" and receipt["failure_category"] == "other"
    assert "private-untrusted-error-detail" not in capsys.readouterr().out
    assert "private-untrusted-error-detail" not in json.dumps(receipt)


def test_public_projection_excludes_endpoint(selected, owned, tmp_path, capsys):
    start(selected, owned, tmp_path)
    output = capsys.readouterr().out
    value = json.loads(output.removeprefix("[lookup-model] "))
    assert "port" not in value and "artifact" not in value
    assert value["category"] == "metadata_ready"
    private = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert private["port"] == 11440 and "artifact" in private


def test_cached_verification_has_no_second_large_hash(selected, owned, tmp_path, monkeypatch):
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    def no_second_read(*args):
        raise AssertionError("unexpected-second-artifact-read")
    monkeypatch.setattr(model, "verify_artifacts", no_second_read)
    result = model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
                               model.time.monotonic() + 10, verified=proof)
    assert result["category"] == "metadata_ready"


def test_serialized_verification_cannot_be_reused(selected, owned, tmp_path):
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    with pytest.raises(ValueError, match="artifact-proof"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
                          model.time.monotonic() + 10, verified=dict(proof.facts))
    assert not owned[0].calls


def test_cached_verification_selection_cannot_drift(selected, owned, tmp_path):
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    selected["model_port"] = 11441
    with pytest.raises(ValueError, match="artifact-proof"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
                          model.time.monotonic() + 10, verified=proof)
    assert not owned[0].calls


def test_verification_is_immutable(selected):
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    with pytest.raises(TypeError):
        proof.facts["template_sha256"] = "0" * 64
    with pytest.raises(FrozenInstanceError):
        proof.selection = ()


def test_expired_launch_does_not_start(selected, owned, tmp_path):
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    with pytest.raises(TimeoutError, match="model-deadline"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
                          model.time.monotonic() - 1, verified=proof)
    assert not owned[0].calls and not owned[2]


def test_metadata_redirect_is_refused():
    with pytest.raises(ValueError, match="metadata-redirect"):
        model.NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.invalid")


def test_load_and_readback_precede_ready(selected, owned, tmp_path):
    result = start(selected, owned, tmp_path)
    assert result["model_ready"] is True
    assert result["load_request_count"] == 1 and result["inference_request_count"] == 0
    assert result["metadata_request_count"] == 4
    assert result["configured_context_tokens"] == 64000
    assert result["allocation_observed"] is False
    assert result["keep_alive_configured"] == -1
    assert result["load_mechanism_configured"] == "native-http"
    assert result["mcp_callback_invocation_count"] == 0
    assert result["mcp_callback_acknowledged"] is False
    assert owned[0].calls[0][2]["env"]["OLLAMA_KEEP_ALIVE"] == "-1"
    calls = owned[4]
    assert [(r.method, r.url.path) for r in calls] == [("POST", "/api/chat"), ("GET", "/api/ps")]
    assert json.loads(calls[0].content) == {
        "model": selected["model"], "messages": [], "stream": False,
        "options": {"num_ctx": 64000}}


@pytest.mark.parametrize("profile", list(model.StartupProfile), ids=["lookup", "native"])
def test_mcp_loader_replaces_native_load_and_shares_readback(selected, owned, tmp_path, profile):
    calls = []
    owned[5]["ps"]["models"][0]["context_length"] = profile.context_tokens

    class Loader:
        def __bool__(self):
            return False  # Selection is by None, never callback truthiness.

        async def __call__(self, **kwargs):
            calls.append(kwargs)
            return True

    deadline = model.time.monotonic() + 10
    result = model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
        deadline, startup_profile=profile, mcp_loader=Loader())
    assert calls == [{"base_url": "http://127.0.0.1:11440", "model": selected["model"],
                      "startup_profile": profile, "deadline": deadline}]
    assert [(r.method, r.url.path) for r in owned[4]] == [("GET", "/api/ps")]
    assert result["load_mechanism_configured"] == "mcp-callback"
    assert result["mcp_callback_invocation_count"] == 1
    assert result["mcp_callback_acknowledged"] is True
    assert result["load_request_count"] == 0 and result["load_acknowledged"] is False
    assert result["model_ready"] is True


@pytest.mark.parametrize("answer", [False, None, 1, "accepted"])
def test_mcp_loader_requires_exact_ack_without_native_fallback(selected, owned, tmp_path, answer):
    async def loader(**_kwargs):
        return answer

    with pytest.raises(ValueError, match="mcp-load-ack"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 10, mcp_loader=loader)
    state = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert state["mcp_callback_invocation_count"] == 1
    assert state["load_request_count"] == 0 and state["model_ready"] is False
    assert not owned[4] and owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_mcp_loader_exception_has_no_fallback_or_private_detail(selected, owned, tmp_path, capsys):
    async def loader(**_kwargs):
        raise RuntimeError("private-loader-detail")

    with pytest.raises(RuntimeError):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 10, mcp_loader=loader)
    state = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert not owned[4] and owned[0].killed == [model.MODEL_PROCESS_NAME]
    assert "private-loader-detail" not in json.dumps(state) + capsys.readouterr().out


def test_mcp_loader_timeout_cancels_callback_without_native_fallback(selected, owned, tmp_path):
    observed = {"cancelled": False, "completed": False}

    async def loader(**_kwargs):
        try:
            await asyncio.sleep(1)
            observed["completed"] = True
            return True
        except asyncio.CancelledError:
            observed["cancelled"] = True
            raise

    with pytest.raises(TimeoutError):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 0.2, mcp_loader=loader)
    assert observed == {"cancelled": True, "completed": False}
    assert not owned[4] and owned[0].killed == [model.MODEL_PROCESS_NAME]


@pytest.mark.parametrize("field,value,category", [
    ("digest", "0" * 64, "loaded-model"), ("context_length", 4096, "loaded-context"),
], ids=["digest", "context"])
def test_mcp_loader_ack_cannot_replace_readback(selected, owned, tmp_path, field, value, category):
    owned[5]["ps"]["models"][0][field] = value

    async def loader(**_kwargs):
        return True

    with pytest.raises(ValueError, match=category):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 10, mcp_loader=loader)
    assert [(r.method, r.url.path) for r in owned[4]] == [("GET", "/api/ps")]
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_mcp_loader_checks_ownership_before_invocation(selected, owned, tmp_path, monkeypatch):
    calls = []
    original = model._read

    def read(kind, *args):
        reply = original(kind, *args)
        if kind == "show":
            monkeypatch.setattr(model, "_image", lambda _pid: tmp_path / "foreign.exe")
        return reply

    async def loader(**_kwargs):
        calls.append(True)
        return True

    monkeypatch.setattr(model, "_read", read)
    with pytest.raises(RuntimeError, match="listener-owner"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 10, mcp_loader=loader)
    assert not calls and not owned[4]


def test_invalid_mcp_loader_refuses_before_start(selected, owned, tmp_path):
    with pytest.raises(ValueError, match="mcp-loader"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 10, mcp_loader=object())
    assert not owned[0].calls and not owned[4]


def test_mcp_loader_cannot_change_owned_process_before_readback(
        selected, owned, tmp_path, monkeypatch):
    async def loader(**_kwargs):
        monkeypatch.setattr(model, "_image", lambda _pid: tmp_path / "foreign.exe")
        return True

    with pytest.raises(RuntimeError, match="listener-owner"):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
            model.time.monotonic() + 10, mcp_loader=loader)
    assert not owned[4] and owned[0].killed == [model.MODEL_PROCESS_NAME]


@pytest.mark.parametrize("change", [
    {"done": False}, {"done_reason": "stop"}, {"model": "other:tag"},
    {"message": None}, {"message": {"role": "assistant", "content": "generated"}},
    {"message": {"role": "assistant", "content": "", "tool_calls": [{}]}},
    {"message": {"role": "assistant", "content": "", "thinking": "reasoning"}},
    {"message": {"role": "system", "content": ""}},
], ids=["done", "reason", "model", "missing", "content", "tools", "thinking", "role"])
def test_load_ack_must_be_positive_and_empty(selected, owned, tmp_path, change):
    owned[5]["load"].update(change)
    with pytest.raises(ValueError, match="load-ack"):
        start(selected, owned, tmp_path)
    assert len(owned[4]) == 1 and owned[0].killed == [model.MODEL_PROCESS_NAME]
    state = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert state["model_ready"] is False and state["load_request_count"] == 1


def test_missing_load_ack_refuses(selected, owned, tmp_path):
    owned[5]["load"] = {}
    with pytest.raises(ValueError, match="load-ack"):
        start(selected, owned, tmp_path)
    assert len(owned[4]) == 1


@pytest.mark.parametrize("change", [
    {"digest": "0" * 64}, {"name": "other:tag"}, {"model": "other:tag"},
], ids=["digest", "name", "model"])
def test_loaded_identity_must_equal_selection(selected, owned, tmp_path, change):
    owned[5]["ps"]["models"][0].update(change)
    with pytest.raises(ValueError, match="loaded-model"):
        start(selected, owned, tmp_path)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]


@pytest.mark.parametrize("context", [4096, 64000.0, None], ids=["wrong", "float", "missing"])
def test_loaded_context_must_be_exact_integer(selected, owned, tmp_path, context):
    owned[5]["ps"]["models"][0]["context_length"] = context
    with pytest.raises(ValueError, match="loaded-context"):
        start(selected, owned, tmp_path)


def test_readback_cannot_add_another_model(selected, owned, tmp_path):
    owned[5]["ps"]["models"].append(dict(owned[5]["ps"]["models"][0]))
    with pytest.raises(ValueError, match="loaded-model"):
        start(selected, owned, tmp_path)


def test_load_ownership_cannot_change(selected, owned, tmp_path, monkeypatch):
    owned[5]["after_load"] = lambda: monkeypatch.setattr(
        model, "_image", lambda pid: tmp_path / "foreign.exe")
    with pytest.raises(RuntimeError, match="listener-owner"):
        start(selected, owned, tmp_path)
    assert len(owned[4]) == 1 and owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_readback_ownership_cannot_change(selected, owned, tmp_path, monkeypatch):
    owned[5]["after_ps"] = lambda: monkeypatch.setattr(
        model, "_image", lambda pid: tmp_path / "foreign.exe")
    with pytest.raises(RuntimeError, match="listener-owner"):
        start(selected, owned, tmp_path)
    assert len(owned[4]) == 2 and owned[0].killed == [model.MODEL_PROCESS_NAME]


def test_load_dribble_cannot_extend_shared_deadline(selected, owned, tmp_path):
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    owned[5]["dribble"] = True
    started = model.time.monotonic()
    with pytest.raises(TimeoutError):
        model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
                          started + 0.2, verified=proof)
    assert model.time.monotonic() - started < 1.0
    assert len(owned[4]) == 1 and owned[0].killed == [model.MODEL_PROCESS_NAME]
    assert owned[5]["stream_closed"] is True and owned[5]["client_closed"] is True
    state = json.loads((tmp_path / "evidence/model-metadata.private.json").read_text())
    assert state["category"] == "refused" and state["model_ready"] is False
    assert state["load_request_count"] == 1


def test_load_response_is_bounded(selected, owned, tmp_path):
    owned[5]["oversize"] = True
    with pytest.raises(ValueError, match="readiness-bound"):
        start(selected, owned, tmp_path)
    assert len(owned[4]) == 1


def test_load_cannot_follow_redirect(selected, owned, tmp_path):
    owned[5]["redirect"] = True
    with pytest.raises((ValueError, httpx.TooManyRedirects)):
        start(selected, owned, tmp_path)
    assert len(owned[4]) == 1


def test_load_uses_remaining_budget_but_metadata_keeps_eight_seconds(
        selected, owned, tmp_path, monkeypatch):
    # Keep model deadline arithmetic exact without changing asyncio's own clock.
    monkeypatch.setattr(model, "time", SimpleNamespace(
        monotonic=lambda: 1000.0, monotonic_ns=model.time.monotonic_ns))
    proof = model.verify_artifacts(selected, model.time.monotonic() + 10)
    model.start_model(selected, owned[0], owned[1], tmp_path / "evidence",
                      model.time.monotonic() + 60, verified=proof)
    load, readback = owned[4]
    assert 8 < load.extensions["timeout"]["read"] <= 60
    assert 0 < readback.extensions["timeout"]["read"] <= 8
