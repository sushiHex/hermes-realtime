"""Managed asset preparation is constructor-only: no SDK, model or host starts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_realtime.providers import kokoro_mcp
from scripts import preview_cutover as module


@pytest.fixture
def managed(tmp_path, monkeypatch):
    def pin(name, data):
        return SimpleNamespace(
            filename=name, sha256=hashlib.sha256(data).hexdigest(), size=len(data),
        )
    model = pin("model.onnx", b"synthetic model")
    voices = pin("voices.bin", b"synthetic voices")
    prefix = Path(".cache/hermes-realtime/kokoro") / model.sha256
    home = tmp_path / "next" / "input" / "home"
    assets = []
    for asset, data in ((model, b"synthetic model"), (voices, b"synthetic voices")):
        source = tmp_path / asset.filename
        source.write_bytes(data)
        relative = prefix / asset.filename
        target = home / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assets.append((source, relative, asset.sha256, asset.size))
    worker = tmp_path / "worker.py"
    worker.write_text("# inert fixture", encoding="utf-8")
    profile = {"version": 1, "gpu_index": 0, "gpu_uuid": "synthetic-device",
               "reserve_gb": 2, "headroom_mb": 1024, "voice": "bf_isabella",
               "language": "en-gb"}
    for name, path, sha in (
        ("python", Path(sys.executable), "a" * 64),
        ("venv_python", Path(sys.executable), "a" * 64),
        ("worker", worker, "b" * 64),
        ("model", home / prefix / model.filename, model.sha256),
        ("voices", home / prefix / voices.filename, voices.sha256),
    ):
        profile[name], profile[name + "_sha256"] = str(path), sha
    profile_path = tmp_path / "operator.json"
    state_dir = tmp_path / "ledger"
    state_dir.mkdir()
    config = {"version": 1, "server_python": sys.executable,
              "server_python_sha256": "a" * 64, "server_venv_python": sys.executable,
              "server_venv_python_sha256": "a" * 64, "server_source": str(tmp_path),
              "server_commit": "c" * 40, "server_tree": "d" * 40,
              "profile": str(profile_path), "profile_sha256": "",
              "server_state_dir": str(state_dir)}
    config_path = tmp_path / "managed.json"
    options = {}
    def save():
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        config["profile_sha256"] = module.digest(profile_path)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        options.update(kokoro_mcp_config=str(config_path),
                       kokoro_mcp_config_sha256=module.digest(config_path))
    save()
    plan = SimpleNamespace(host_options=options, speech={"tts_voice": "bf_isabella"},
                           runtime_target=Path(kokoro_mcp.__file__).parents[2],
                           ports={name: 12345 for name in ("api", "companion", "livekit", "host")})
    monkeypatch.setattr(module, "_runtime_modules", lambda plan: None)
    monkeypatch.setattr(module, "_kokoro_asset_pins", lambda plan: (model, voices))
    monkeypatch.setattr(module, "_speech_asset_manifest", lambda plan: assets)
    def forbidden(*a, **k):
        pytest.fail("preflight must never start a worker/transport")
    monkeypatch.setattr(kokoro_mcp.KokoroMcpWorkerFactory, "_start_transport", forbidden)
    monkeypatch.setattr(kokoro_mcp.KokoroMcpWorkerFactory, "launch", forbidden)
    monkeypatch.setattr(kokoro_mcp.KokoroWorkerClient, "_start", forbidden)
    return SimpleNamespace(plan=plan, home=home, profile=profile, save=save, assets=assets,
                           model=model, voices=voices, config_path=config_path)


def test_managed_asset_preflight_constructs_and_closes_without_launch(managed, monkeypatch, capsys):
    s = managed
    snapshots = (s.config_path.read_bytes(), dict(s.profile))
    clients, factories = [], []
    create = kokoro_mcp.KokoroMcpWorkerFactory.create
    close = kokoro_mcp.KokoroMcpWorkerFactory.close
    def tracked_create(self, **kwargs):
        client = create(self, **kwargs)
        clients.append(client)
        return client
    def tracked_close(self):
        close(self)
        factories.append(self)
    monkeypatch.setattr(kokoro_mcp.KokoroMcpWorkerFactory, "create", tracked_create)
    monkeypatch.setattr(kokoro_mcp.KokoroMcpWorkerFactory, "close", tracked_close)
    module._verify_managed_kokoro_assets(s.plan, s.home)
    assert len(clients) == len(factories) == 1
    assert clients[0].closed and factories[0]._closed
    assert factories[0]._thread is None and not factories[0]._launch_attempted
    assert snapshots == (s.config_path.read_bytes(), s.profile)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("fault", ["model", "voices", "model_sha256", "voices_sha256",
                                    "manifest", "extra_manifest", "duplicate_manifest", "origin"])
def test_managed_asset_preflight_refuses_stale_bindings(managed, monkeypatch, capsys, fault):
    s = managed
    if fault in {"model", "voices"}:
        old = s.home.parents[2] / ("old-" + fault)
        old.write_bytes(s.assets[0 if fault == "model" else 1][0].read_bytes())
        s.profile[fault] = str(old)
        s.save()
    elif fault.endswith("sha256"):
        s.profile[fault] = "e" * 64
        s.save()
    elif fault == "manifest":
        source, relative, sha, size = s.assets[0]
        s.assets[0] = (source, relative.parent / "nested" / relative.name, sha, size)
    elif fault == "extra_manifest":
        source, relative, sha, size = s.assets[0]
        s.assets.append((source, relative.parent / "extra.onnx", sha, size))
    elif fault == "duplicate_manifest":
        s.assets.append(s.assets[0])
    else:
        s.plan.runtime_target = s.home / "unrelated-runtime"
    with pytest.raises(module.PreviewRefusal, match="kokoro_mcp_asset_binding"):
        module._verify_managed_kokoro_assets(s.plan, s.home)
    lines = capsys.readouterr().err.splitlines()
    marker = '[kokoro-mcp-preview-refused] '
    assert json.loads(lines[-1].removeprefix(marker)) == {
        "version": 1, "category": "asset_binding_refused"}
    assert str(s.home) not in "".join(lines)


def test_cpu_asset_preflight_does_not_import_factory_or_manifest(monkeypatch, tmp_path):
    plan = SimpleNamespace(host_options={})
    monkeypatch.setattr(module, "_runtime_modules", lambda *a: pytest.fail("CPU import"))
    monkeypatch.setattr(module, "_speech_asset_manifest", lambda *a: pytest.fail("CPU manifest"))
    module._verify_managed_kokoro_assets(plan, tmp_path)


def test_preflight_invokes_managed_binding_before_ready(managed, monkeypatch):
    s = managed
    for name in ("verify_realtime_runtime", "verify_profile", "verify_cpu_speech",
                 "verify_host_profile_mode", "verify_speech_assets"):
        monkeypatch.setattr(module, name, lambda *a: None)
    s.profile["model"] = str(s.assets[0][0])
    s.save()
    with pytest.raises(module.PreviewRefusal, match="kokoro_mcp_asset_binding"):
        module.preflight(s.plan, s.home)


@pytest.mark.parametrize("failure", ["create", "close"])
def test_managed_asset_preflight_cleanup_failure_is_not_ready(
    managed, monkeypatch, capsys, failure,
):
    s = managed
    closed = []
    original_close = kokoro_mcp.KokoroMcpWorkerFactory.close
    def fail(*a, **k):
        raise RuntimeError("synthetic_private_value_not_for_output")
    def close(self):
        original_close(self)
        closed.append(self)
        if failure == "close":
            fail()
    monkeypatch.setattr(kokoro_mcp.KokoroMcpWorkerFactory, "close", close)
    if failure == "create":
        monkeypatch.setattr(kokoro_mcp.KokoroMcpWorkerFactory, "create", fail)
    with pytest.raises(module.PreviewRefusal, match="kokoro_mcp_asset_binding"):
        module._verify_managed_kokoro_assets(s.plan, s.home)
    assert len(closed) == 1 and closed[0]._closed
    assert "synthetic_private_value_not_for_output" not in capsys.readouterr().err
