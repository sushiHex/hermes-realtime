"""Offline binding and owned metadata checks; never launch Ollama or contact HTTP."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

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
    containment = SimpleNamespace(members=lambda: {os.getpid()})
    return owner, containment, calls, replies


def start(selected, owned, tmp_path):
    owner, containment, _, _ = owned
    return model.start_model(selected, owner, containment, tmp_path / "evidence",
                             model.time.monotonic() + 10)


def test_owned_metadata_only_and_closed_options(selected, owned, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_GO_TEMPLATE", "1")
    monkeypatch.setenv("LLAMA_ARG_GPU_LAYERS", "0")
    result = start(selected, owned, tmp_path)
    owner, _, calls, _ = owned
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


def test_foreign_containment_refused_before_start(selected, owned, tmp_path):
    owned[1].members = lambda: set()
    with pytest.raises(RuntimeError, match="owner-containment"):
        start(selected, owned, tmp_path)
    assert not owned[0].calls


def test_occupied_port_never_adopted(selected, owned, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_listeners", lambda *args: [("127.0.0.1:11440", 999)])
    with pytest.raises(RuntimeError, match="port-occupied"):
        start(selected, owned, tmp_path)
    assert not owned[0].calls


@pytest.mark.parametrize("listener", [("0.0.0.0:11440", 4321), ("127.0.0.1:11440", 999)])
def test_listener_must_match_owned_process(selected, owned, tmp_path, monkeypatch, listener):
    monkeypatch.setattr(model, "_listeners", lambda *args: [listener] if owned[0].calls else [])
    with pytest.raises(RuntimeError, match="listener-owner"):
        start(selected, owned, tmp_path)
    assert owned[0].killed == [model.MODEL_PROCESS_NAME]
    assert not owned[2]


def test_process_image_must_match_runtime(selected, owned, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_image", lambda pid: tmp_path / "foreign.exe")
    with pytest.raises(RuntimeError, match="listener-owner"):
        start(selected, owned, tmp_path)
    assert not owned[2]


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
