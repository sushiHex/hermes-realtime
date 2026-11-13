from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import preview_cutover


def _mcp_plan(tmp_path):
    plan_path, profile = _plan(tmp_path)
    config = tmp_path / "gpu-operator.json"
    config.write_text("{}", encoding="utf-8")
    value = json.loads(plan_path.read_text())
    value["host_options"].update({
        "kokoro_mcp_config": str(config),
        "kokoro_mcp_config_sha256": preview_cutover.digest(config),
    })
    plan_path.write_text(json.dumps(value))
    return plan_path, profile, config


@pytest.mark.parametrize("fault", [None, "partial", "digest", "relative", "extra"])
def test_managed_mcp_plan_requires_complete_bound_configuration(tmp_path, fault):
    path, _, _ = _mcp_plan(tmp_path)
    value = json.loads(path.read_text())
    options = value["host_options"]
    if fault == "partial":
        del options["kokoro_mcp_config_sha256"]
    elif fault == "digest":
        options["kokoro_mcp_config_sha256"] = "invalid"
    elif fault == "relative":
        options["kokoro_mcp_config"] = "relative.json"
    elif fault == "extra":
        options["arbitrary_command"] = "refused"
    path.write_text(json.dumps(value))
    if fault is not None:
        category = {"partial": "kokoro_mcp_config_fields", "digest": "kokoro_mcp_config_digest",
                    "relative": "kokoro_mcp_config_path", "extra": "host_options_fields"}[fault]
        with pytest.raises(ValueError, match=category):
            preview_cutover.load_plan(path)
    else:
        assert preview_cutover.load_plan(path).host_options == options


def test_managed_mcp_config_drift_refused_before_host_construction(tmp_path):
    path, _, config = _mcp_plan(tmp_path)
    plan = preview_cutover.load_plan(path)
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="kokoro_mcp_config_binding"):
        preview_cutover._bound_kokoro_mcp_config(plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False])
async def test_managed_mcp_config_forwards_to_real_host_builder(tmp_path, monkeypatch, supported):
    path, profile, config = _mcp_plan(tmp_path)
    plan = preview_cutover.load_plan(path)
    plan.output.mkdir()
    (plan.output / "host.stop").touch()
    observed = []

    class Launcher:
        async def start(self):
            observed.append("start")
        async def close(self):
            observed.append("close")

    def build(*, hermes_discover_tools=False, kokoro_mcp_config=None, **kwargs):
        assert kokoro_mcp_config == config
        return Launcher()

    def unsupported(*, hermes_discover_tools=False, **kwargs):
        return Launcher()

    host = SimpleNamespace(
        _load_api_bearer=lambda _: "synthetic-not-a-credential",
        _load_voice_companion=lambda *_: None,
        LoopbackPeerAuthorizer=object,
        build_local_host_launcher=build if supported else unsupported,
    )
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda _: (plan, None, profile))
    monkeypatch.setattr(preview_cutover, "_runtime_modules", lambda _: (None, host))
    if supported:
        await preview_cutover._host_child(tmp_path / "active")
        assert observed == ["start", "close"]
    else:
        with pytest.raises(ValueError, match="kokoro_mcp_host_unavailable"):
            await preview_cutover._host_child(tmp_path / "active")
        assert observed == []


def test_native_evidence_host_selection_is_explicit():
    from scripts.preview_cutover import native_web_evidence_contract

    assert native_web_evidence_contract({}) is None
    config = {"plugins": {"entries": {"hermes-realtime": {"settings": {
        "web_evidence": True,
        "presentation": {
            "provider": "synthetic", "base_url": "https://example.invalid/v1",
            "model": "synthetic", "api_key_env": "SYNTHETIC_KEY",
        },
    }}}}}
    assert native_web_evidence_contract(config) == "hermes.web-evidence/v1"
    config["plugins"]["entries"]["hermes-realtime"]["settings"]["web_evidence"] = "true"
    with pytest.raises(ValueError, match="evidence"):
        native_web_evidence_contract(config)


@pytest.mark.asyncio
async def test_native_preview_forwards_evidence_contract_to_actual_launcher(tmp_path, monkeypatch):
    plan_path, profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    config_path = profile / "config.yaml"
    config = json.loads(config_path.read_text())
    config["plugins"]["entries"]["hermes-realtime"]["settings"].update({
        "web_evidence": True,
        "presentation": {
            "provider": "synthetic", "base_url": "https://example.invalid/v1",
            "model": "synthetic", "api_key_env": "SYNTHETIC_KEY",
        },
    })
    config_path.write_text(json.dumps(config))
    plan.output.mkdir()
    (plan.output / "host.stop").touch()
    calls = []

    class Launcher:
        async def start(self):
            calls.append("start")

        async def close(self):
            calls.append("close")

    def build(*, hermes_discover_tools=False, hermes_web_evidence_contract=None, **kwargs):
        assert hermes_discover_tools is True
        assert hermes_web_evidence_contract == "hermes.web-evidence/v1"
        assert kwargs["hermes_lookup_policy"] is None
        return Launcher()

    host = SimpleNamespace(
        _load_api_bearer=lambda _: "synthetic-not-a-credential",
        _load_voice_companion=lambda *_: None,
        LoopbackPeerAuthorizer=object,
        build_local_host_launcher=build,
    )
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda _: (plan, None, profile))
    monkeypatch.setattr(preview_cutover, "_runtime_modules", lambda _: (None, host))
    assert await preview_cutover._host_child(tmp_path / "active.json") == 0
    assert calls == ["start", "close"]


@pytest.mark.parametrize("fault", [None, "origin", "config"])
@pytest.mark.parametrize("staged", [False, True])
def test_presentation_validation_binds_verified_runtime_before_use(
    tmp_path, monkeypatch, fault, staged,
):
    from hermes_realtime.integration.presentation_transport import Presentation

    plan_path, profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.runtime_target.mkdir()
    config_path = profile / "config.yaml"
    config = json.loads(config_path.read_text())
    route = {
        "provider": "synthetic", "base_url": "https://example.invalid/v1",
        "model": "synthetic", "api_key_env": "SYNTHETIC_KEY",
    }
    if fault == "config":
        del route["model"]
    config["plugins"]["entries"]["hermes-realtime"]["settings"].update(
        web_evidence=True, presentation=route
    )
    config_path.write_text(json.dumps(config))
    home = profile
    if staged:
        home = tmp_path / "staged-profile"
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps(config))
        config["plugins"]["entries"]["hermes-realtime"]["settings"]["web_evidence"] = False
        config_path.write_text(json.dumps(config))

    def build(*, hermes_discover_tools=False, hermes_web_evidence_contract=None):
        pass

    host = SimpleNamespace(
        __file__=str(plan.runtime_target / "hermes_realtime/host_launcher.py"),
        build_local_host_launcher=build,
    )
    presentation = SimpleNamespace(
        __file__=str((tmp_path if fault == "origin" else plan.runtime_target)
                     / "hermes_realtime/integration/presentation_transport.py"),
        Presentation=Presentation,
    )
    monkeypatch.setattr(
        preview_cutover.importlib, "import_module",
        lambda name: host if name.endswith("host_launcher") else presentation,
    )
    if fault is None:
        preview_cutover.verify_host_profile_mode(plan, home)
    else:
        with pytest.raises(ValueError, match="native_web_presentation_" + fault):
            preview_cutover.verify_host_profile_mode(plan, home)



def test_failure_metadata_keeps_operator_category_without_echoing_dependency_errors():
    try:
        preview_cutover.require(False, "runtime_import_origin")
    except ValueError as error:
        assert preview_cutover._failure_status(error)["failure_category"] == "runtime_import_origin"
    private = "synthetic_private_exception_value"
    assert private not in json.dumps(preview_cutover._failure_status(ValueError(private)))


def test_receipt_is_invisible_until_fsynced_and_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "ready.json"
    observed = []
    fsync = preview_cutover.os.fsync

    def inspect_before_publish(fd):
        observed.append(path.exists())
        fsync(fd)

    monkeypatch.setattr(preview_cutover.os, "fsync", inspect_before_publish)
    preview_cutover._write_new(path, {"ready": True})
    assert observed == [False]
    assert json.loads(path.read_text()) == {"ready": True}
    with pytest.raises(ValueError, match="output_receipt_refused"):
        preview_cutover._write_new(path, {"ready": False})
    assert json.loads(path.read_text()) == {"ready": True}
    assert list(tmp_path.iterdir()) == [path]


def test_receipt_publication_cannot_replace_a_competing_writer(tmp_path, monkeypatch):
    path = tmp_path / "ready.json"
    fdopen = preview_cutover.os.fdopen
    def competitor(*args, **kwargs):
        path.write_text('{"owner":"other"}')
        return fdopen(*args, **kwargs)
    monkeypatch.setattr(preview_cutover.os, "fdopen", competitor)
    with pytest.raises(ValueError, match="output_receipt_refused"):
        preview_cutover._write_new(path, {"owner": "this"})
    assert json.loads(path.read_text()) == {"owner": "other"}
    assert list(tmp_path.iterdir()) == [path]


def test_preflight_passes_the_staged_home_to_every_profile_check(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    home = tmp_path / "staged"
    called = []
    monkeypatch.setattr(preview_cutover, "verify_realtime_runtime", lambda p: None)
    monkeypatch.setattr(preview_cutover, "verify_cpu_speech", lambda p: None)
    for name in ("verify_profile", "verify_host_profile_mode", "verify_speech_assets",
                 "_verify_native_credential_runtime"):
        monkeypatch.setattr(preview_cutover, name, lambda p, h, name=name: called.append((name, h)))
    preview_cutover.preflight(plan, home)
    assert called == [(name, home) for name in (
        "verify_profile", "verify_host_profile_mode", "verify_speech_assets",
        "_verify_native_credential_runtime")]


def test_base_environment_preserves_programfiles_without_passing_credentials(monkeypatch):
    monkeypatch.setenv("PATH", "C:\\Windows\\System32")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("PROGRAMFILES", "C:\\Program Files")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-key")

    env = preview_cutover._base_environment()

    assert env["PROGRAMFILES"] == "C:\\Program Files"
    assert "OPENAI_API_KEY" not in env
    assert set(env) <= {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PROGRAMFILES"}

_FAKE_MODEL_BYTES = b"fixture-kokoro-model"
_FAKE_VOICE_BYTES = b"fixture-kokoro-voices"
_FAKE_MODEL = SimpleNamespace(
    filename="kokoro-v1.0.onnx",
    size=len(_FAKE_MODEL_BYTES),
    sha256=hashlib.sha256(_FAKE_MODEL_BYTES).hexdigest(),
)
_FAKE_VOICES = SimpleNamespace(
    filename="voices-v1.0.bin",
    size=len(_FAKE_VOICE_BYTES),
    sha256=hashlib.sha256(_FAKE_VOICE_BYTES).hexdigest(),
)


@pytest.fixture(autouse=True)
def _fixture_kokoro_pins(monkeypatch):
    monkeypatch.setattr(
        preview_cutover, "_kokoro_asset_pins", lambda _plan: (_FAKE_MODEL, _FAKE_VOICES)
    )


def test_prefixed_preflight_success_is_the_prepared_completion_gate(tmp_path):
    prepared = tmp_path / "prepared.json"
    completed = subprocess.CompletedProcess(
        ["preview", "preflight"],
        0,
        stdout=(
            '[voice-preview] {"category":"preflight_ready","launch_authorized":false,'
            '"ports":[59876,59877]}\n'
        ),
        stderr="",
    )

    result = preview_cutover.record_prepared_after_preflight(
        {"version": 1, "profile_mode": "native"}, completed, prepared
    )

    assert result == {"version": 1, "profile_mode": "native", "preflight": "accepted"}
    assert json.loads(prepared.read_text(encoding="utf-8")) == result


def test_refused_prefixed_preflight_does_not_mark_preparation_complete(tmp_path):
    prepared = tmp_path / "prepared.json"
    completed = subprocess.CompletedProcess(
        ["preview", "preflight"],
        1,
        stdout=(
            '[voice-preview] {"category":"refused","failure_kind":"RuntimeError",'
            '"failure_category":"wheel_binding"}\n'
        ),
        stderr="private path omitted",
    )

    with pytest.raises(ValueError, match="preflight_refused"):
        preview_cutover.record_prepared_after_preflight(
            {"version": 1, "profile_mode": "lookup"}, completed, prepared
        )

    assert not prepared.exists()


def _plan(tmp_path: Path) -> tuple[Path, Path]:
    profile = tmp_path / "source-profile"
    profile.mkdir()
    status_command = tmp_path / "vram-status.exe"
    status_command.write_bytes(b"synthetic launcher fixture")
    state_dir = tmp_path / "vram-state"
    state_dir.mkdir()
    asset_root = tmp_path / "speech-assets"
    moonshine_cache = asset_root / "moonshine"
    moonshine_file = moonshine_cache / "medium" / "synthetic-model.bin"
    moonshine_file.parent.mkdir(parents=True)
    moonshine_file.write_bytes(b"fixture-moonshine-model")
    kokoro_cache = asset_root / "kokoro" / _FAKE_MODEL.sha256
    kokoro_cache.mkdir(parents=True)
    model_file = kokoro_cache / _FAKE_MODEL.filename
    model_file.write_bytes(_FAKE_MODEL_BYTES)
    voice_file = kokoro_cache / _FAKE_VOICES.filename
    voice_file.write_bytes(_FAKE_VOICE_BYTES)
    asset_files = [moonshine_file, model_file, voice_file]
    receipt = {
        "version": 1,
        "moonshine_tier": "medium",
        "moonshine_package_version": "0.1.5",
        "moonshine_cache_root": str(moonshine_cache.resolve()),
        "kokoro_cache_root": str(kokoro_cache.resolve()),
        "kokoro_model_sha256": _FAKE_MODEL.sha256,
        "kokoro_model_bytes": _FAKE_MODEL.size,
        "kokoro_voice": "bf_isabella",
        "kokoro_voice_sha256": _FAKE_VOICES.sha256,
        "kokoro_voice_bytes": _FAKE_VOICES.size,
        "files": {
            str(item.resolve()): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in asset_files
        },
        "file_sizes": {str(item.resolve()): item.stat().st_size for item in asset_files},
    }
    receipt_path = asset_root / "speech-assets.json"
    asset_root.mkdir(exist_ok=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (profile / "config.yaml").write_text(
        json.dumps(
            {
                "model": {"default": "gpt-6-luna"},
                "agent": {"reasoning_effort": "low"},
                "approvals": {"mode": "manual", "unattended_mode": "deny"},
                "plugins": {
                    "enabled": ["hermes-realtime"],
                    "entries": {
                        "hermes-realtime": {
                            "allow_tool_override": False,
                            "settings": {"background_capabilities": True},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan = {
        "version": 1,
        "output": str(tmp_path / "preview"),
        "state_root": str(tmp_path / "state-root"),
        "realtime": {
            "source": str(tmp_path / "repo"),
            "revision": "a" * 40,
            "tree": "b" * 40,
            "wheel": str(tmp_path / "runtime.whl"),
            "wheel_sha256": "c" * 64,
            "runtime_target": str(tmp_path / "runtime"),
            "record_sha256": "d" * 64,
            "git_executable": str(tmp_path / "git.exe"),
        },
        "hermes": {
            "source": str(tmp_path / "hermes"),
            "revision": "e" * 40,
            "tree": "f" * 40,
            "python": str(tmp_path / "python.exe"),
            "profile_source": str(profile),
        },
        "native_status": {
            "command": str(status_command),
            "sha256": hashlib.sha256(status_command.read_bytes()).hexdigest(),
            "state_dir": str(state_dir),
            "python_user_base": None,
        },
        "speech_assets": {
            "receipt": str(receipt_path),
            "moonshine_cache": str(moonshine_cache),
            "kokoro_cache": str(kokoro_cache),
        },
        "profile_mode": "native",
        "ports": {
            "api": 59876,
            "companion": 59877,
            "livekit": 7880,
            "host": 28765,
            "protected": [8765],
        },
        "models": {
            "foreground": "gpt-5.6-terra",
            "foreground_effort": "low",
            "background": "gpt-6-luna",
            "background_effort": "low",
        },
        "speech": {
            "stt_provider": "moonshine",
            "stt_tier": "medium",
            "tts_provider": "kokoro",
            "tts_voice": "bf_isabella",
            "tts_threads": 8,
            "tts_chunk_chars": 128,
            "moonshine_version": "0.1.5",
            "kokoro_version": "0.6.1",
            "onnxruntime_version": "1.22.0",
        },
        "executables": {
            "codex": str(tmp_path / "codex.exe"),
            "livekit": str(tmp_path / "livekit.exe"),
            "host_python": str(tmp_path / "host-python.exe"),
        },
        "host_options": {
            "allow_unsandboxed_tasks": False,
            "natural_work_tools": True,
            "task_start_policy": "automatic",
            "conversation_profile": "natural_v1",
        },
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, profile


def test_prepare_keeps_refusal_evidence_and_never_writes_prepared(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    monkeypatch.setattr(
        preview_cutover,
        "run_preflight",
        lambda *_: subprocess.CompletedProcess(
            ["preflight"],
            1,
            stdout=(
                '[voice-preview] {"category":"refused","failure_kind":"RuntimeError",'
                '"failure_category":"source_binding"}\n'
            ),
            stderr="private diagnostic text",
        ),
    )

    with pytest.raises(ValueError, match="preflight_refused"):
        preview_cutover.prepare(plan_path)

    output = tmp_path / "preview"
    assert not (output / "prepared.private.json").exists()
    refusal = json.loads((output / "preparation-refused.private.json").read_text())
    assert refusal == {
        "version": 1,
        "category": "preparation_refused",
        "failure_kind": "RuntimeError",
        "failure_category": "source_binding",
        "prepared": False,
    }
    assert "private diagnostic text" not in json.dumps(refusal)


def test_prepare_success_runs_the_prefixed_preflight_before_receipt(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    monkeypatch.setattr(
        preview_cutover,
        "run_preflight",
        lambda *_: subprocess.CompletedProcess(
            ["preflight"],
            0,
            stdout=(
                '[voice-preview] {"category":"preflight_ready","launch_authorized":false,'
                '"ports":[59876,59877,7880,28765]}\n'
            ),
            stderr="",
        ),
    )

    result = preview_cutover.prepare(plan_path)

    prepared = json.loads((tmp_path / "preview" / "prepared.private.json").read_text())
    assert result["category"] == "prepared"
    assert prepared["preflight"] == "accepted"
    assert prepared["profile_complete"] is False
    assert prepared["memory_provider_config_inspection"] == "absent"
    assert not (tmp_path / "preview" / "input" / "home" / ".env").exists()


def test_prepare_refuses_a_profile_selecting_an_external_memory_provider(tmp_path):
    plan_path, profile = _plan(tmp_path)
    config = json.loads((profile / "config.yaml").read_text(encoding="utf-8"))
    config["memory"] = {"provider": "synthetic-provider", "synthetic-provider": {"k": "v"}}
    (profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="preflight_refused"):
        preview_cutover.prepare(plan_path)

    output = tmp_path / "preview"
    assert not (output / "prepared.private.json").exists()
    assert not (output / "input" / "home" / "config.yaml").exists()
    refusal = (output / "preparation-refused.private.json").read_text(encoding="utf-8")
    assert json.loads(refusal) == {
        "version": 1,
        "category": "preparation_refused",
        "failure_kind": "ValueError",
        "failure_category": "memory_provider_selected",
        "prepared": False,
    }
    assert "synthetic-provider" not in refusal


@pytest.mark.parametrize(
    ("memory", "category"),
    [
        ({"provider": "synthetic-provider", "synthetic-provider": {"k": "secret-v"}},
         "memory_provider_selected"),
        ({"provider": ["synthetic-provider"]}, "memory_provider_refused"),
    ],
)
def test_preflight_marker_carries_only_the_memory_refusal_category(
    tmp_path, monkeypatch, capsys, memory, category
):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.output.mkdir()
    home = preview_cutover.stage_preflight_profile(plan)
    config = json.loads((home / "config.yaml").read_text(encoding="utf-8"))
    config["memory"] = memory
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "verify_realtime_runtime", lambda _: None)

    code = preview_cutover.main(["preflight", "--plan", str(plan_path), "--home", str(home)])

    output = capsys.readouterr().out
    assert code == 1
    assert output.startswith("[voice-preview] ") and output.count("\n") == 1
    assert json.loads(output.removeprefix("[voice-preview] ")) == {
        "category": "refused",
        "failure_kind": "ValueError",
        "failure_category": category,
    }
    for private in ("synthetic-provider", "secret-v"):
        assert private not in output


def test_native_staging_uses_the_reviewed_status_profile_projection(tmp_path):
    plan_path, source_profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.output.mkdir()
    isolated_import = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import runpy,sys; from pathlib import Path; "
            "ns=runpy.run_path(sys.argv[1],run_name='preview_import'); "
            "ns['load_plan'](Path(sys.argv[2]))",
            str(Path(__file__).parents[1] / "scripts" / "preview_cutover.py"),
            str(plan_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated_import.returncode == 0, isolated_import.stderr

    staged_home = preview_cutover.stage_preflight_profile(plan)

    source = json.loads((source_profile / "config.yaml").read_text(encoding="utf-8"))
    staged = json.loads((staged_home / "config.yaml").read_text(encoding="utf-8"))
    assert staged != source
    assert staged["mcp_servers"]["vram"]["tools"]["include"] == ["vram_status"]
    assert staged["mcp_servers"]["vram"]["env"]["VRAM_MCP_AUDIT"] == "0"
    assert staged["platform_toolsets"]["api_server"] == ["web", "mcp-vram"]
    assert staged["plugins"]["enabled"] == ["hermes-realtime"]
    assert (
        staged["plugins"]["entries"]["hermes-realtime"]["settings"]["background_capabilities"]
        is True
    )
    staged_moonshine = staged_home / ".cache" / "moonshine" / "medium" / "synthetic-model.bin"
    staged_kokoro = (
        staged_home
        / ".cache"
        / "hermes-realtime"
        / "kokoro"
        / _FAKE_MODEL.sha256
        / _FAKE_MODEL.filename
    )
    assert staged_moonshine.read_bytes() == b"fixture-moonshine-model"
    assert staged_kokoro.read_bytes() == _FAKE_MODEL_BYTES
    assert "MOONSHINE_VOICE_CACHE" in inspect.getsource(preview_cutover._host_environment)
    preview_cutover.verify_profile(plan, staged_home)


def test_native_profile_accepts_bounded_yaml_for_staging_and_verification(tmp_path):
    yaml = pytest.importorskip("yaml")

    plan_path, source_profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    config = json.loads((source_profile / "config.yaml").read_text(encoding="utf-8"))
    (source_profile / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    plan.output.mkdir()

    staged_home = preview_cutover.stage_preflight_profile(plan)

    staged_path = staged_home / "config.yaml"
    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    assert staged["model"]["default"] == "gpt-6-luna"
    staged_path.write_text(yaml.safe_dump(staged), encoding="utf-8")
    preview_cutover.verify_profile(plan, staged_home)


def test_native_profile_json_duplicate_fields_still_refuse(tmp_path):
    plan_path, source_profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    (source_profile / "config.yaml").write_text(
        '{"model":{"default":"gpt-6-luna"},"model":{"default":"gpt-6-luna"}}',
        encoding="utf-8",
    )
    plan.output.mkdir()

    with pytest.raises(ValueError, match="json_duplicate_field"):
        preview_cutover.stage_preflight_profile(plan)


@pytest.mark.parametrize(
    ("model", "effort"),
    [("gpt-5.6-terra", "low"), ("gpt-6-luna", "high")],
)
def test_native_profile_model_and_effort_must_match_plan(tmp_path, model, effort):
    plan_path, source_profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    config = json.loads((source_profile / "config.yaml").read_text(encoding="utf-8"))
    config["model"]["default"] = model
    config["agent"]["reasoning_effort"] = effort
    (source_profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    plan.output.mkdir()

    with pytest.raises(ValueError, match="native_model_binding"):
        preview_cutover.stage_preflight_profile(plan)


def test_native_profile_refuses_provider_prefixed_reasoning_override(tmp_path):
    plan_path, source_profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    config = json.loads((source_profile / "config.yaml").read_text(encoding="utf-8"))
    config["agent"]["reasoning_overrides"] = {"openai/gpt-6-luna": "high"}
    (source_profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    plan.output.mkdir()

    with pytest.raises(ValueError, match="native_model_binding"):
        preview_cutover.stage_preflight_profile(plan)


def test_livekit_command_binds_planned_port(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.local_livekit.server_command",
        lambda executable: [str(executable), "--dev", "--bind", "127.0.0.1"],
    )

    assert preview_cutover._livekit_command(tmp_path / "livekit-server", 17880) == [
        str(tmp_path / "livekit-server"), "--dev", "--bind", "127.0.0.1", "--port", "17880"
    ]


def test_speech_asset_hash_drift_refuses_preflight(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    speech_file = next(plan.speech_assets["moonshine_cache"].rglob("synthetic-model.bin"))
    speech_file.write_bytes(b"drifted-moonshine-model")
    plan.output.mkdir()

    with pytest.raises(ValueError, match="speech_cache_integrity"):
        preview_cutover.stage_preflight_profile(plan)


def test_unadmitted_shared_cache_metadata_is_not_copied(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    extra = plan.speech_assets["moonshine_cache"] / "unrelated-cache-metadata.json"
    extra.write_bytes(b"unadmitted shared-cache data")
    plan.output.mkdir()
    home = preview_cutover.stage_preflight_profile(plan)
    assert not (home / ".cache" / "moonshine" / extra.name).exists()
    preview_cutover.verify_speech_assets(plan, home)


def test_prepare_retains_fixed_stage_refusal_category(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)

    def refuse(_):
        raise preview_cutover.PreviewRefusal("speech_cache_integrity")

    monkeypatch.setattr(preview_cutover, "stage_preflight_profile", refuse)
    with pytest.raises(ValueError, match="preflight_refused"):
        preview_cutover.prepare(plan_path)
    receipt = json.loads((tmp_path / "preview/preparation-refused.private.json").read_text())
    assert receipt["failure_category"] == "speech_cache_integrity"


def test_staged_cache_refuses_unadmitted_files(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.output.mkdir()
    home = preview_cutover.stage_preflight_profile(plan)
    (home / ".cache/moonshine/unadmitted.json").write_bytes(b"not in receipt")
    with pytest.raises(ValueError, match="speech_asset_stage_inventory"):
        preview_cutover.verify_speech_assets(plan, home)


def test_native_profile_revalidation_refuses_removed_platform_toolset(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.output.mkdir()
    staged_home = preview_cutover.stage_preflight_profile(plan)
    config_path = staged_home / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["platform_toolsets"]["api_server"] = ["web"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="native platform capability refused"):
        preview_cutover.verify_profile(plan, staged_home)


def test_runtime_import_binds_target_directory_for_host_and_gateway(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    runtime_target = tmp_path / "isolated-runtime"
    runtime_target.mkdir()
    plan_value["realtime"]["runtime_target"] = str(runtime_target)
    plan_path.write_text(json.dumps(plan_value), encoding="utf-8")
    plan = preview_cutover.load_plan(plan_path)
    expected_package = runtime_target / "hermes_realtime" / "__init__.py"
    import_paths: list[list[str]] = []

    def fake_import(name):
        import_paths.append(list(sys.path))
        return SimpleNamespace(__file__=str(expected_package))

    original_path = list(sys.path)
    monkeypatch.setattr(preview_cutover.importlib, "import_module", fake_import)
    try:
        preview_cutover._runtime_modules(plan)
    finally:
        sys.path[:] = original_path
    assert str(runtime_target.resolve()) in import_paths[0]

    cli_source = plan.hermes_source / "hermes_cli" / "main.py"
    cli_source.parent.mkdir(parents=True)
    cli_source.write_text("", encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda _path: (plan, {}, tmp_path))
    monkeypatch.setattr(preview_cutover, "_runtime_modules", lambda _plan: (None, None))
    monkeypatch.setattr(
        preview_cutover.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(cli_source), main=lambda: 0),
    )
    original_path = list(sys.path)
    try:
        assert preview_cutover._gateway_child(tmp_path / "active.json") == 0
        assert str(runtime_target.resolve()) in sys.path
    finally:
        sys.path[:] = original_path


def test_host_spawn_follows_gateway_readiness_and_protected_snapshot_is_diagnostic():
    tree = ast.parse(inspect.getsource(preview_cutover.serve))
    ready_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "gateway_ready"
    )
    host_spawn_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "spawn"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "host"
    )
    stopped = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "stopped" for target in node.targets)
    )
    assert host_spawn_line > ready_line
    assert "protected_unchanged" not in ast.unparse(stopped)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "require_owned_listeners" in calls
    assert "_require_owned_endpoint_rows" in calls
    assert "_require_gateway_health_owner" in calls
    assert "'pid': host_pid" in ast.unparse(tree)

    host_tree = ast.parse(inspect.getsource(preview_cutover._run_host_child))
    host_ready = next(
        node
        for node in ast.walk(host_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_new"
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ready"
    )
    ready_fields = {
        key.value: value
        for key, value in zip(host_ready.args[1].keys, host_ready.args[1].values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert set(ready_fields) == {"version", "ready", "pid"}
    assert ast.unparse(ready_fields["pid"]) == "os.getpid()"


@pytest.mark.parametrize(
    ("rows", "expected_pid", "members"),
    [([("127.0.0.1", 502)], 501, {501, 502}), ([("127.0.0.1", 501)], 501, {502})],
)
def test_endpoint_listener_refuses_wrong_pid_or_missing_job_membership(rows, expected_pid, members):
    with pytest.raises(ValueError, match="endpoint_owner"):
        preview_cutover._require_owned_endpoint_rows(rows, expected_pid, members, 1)


@pytest.mark.parametrize(
    ("health", "expected_pid", "members"),
    [({"pid": 502}, 501, {501, 502}), ({"pid": 501}, 501, {502})],
)
def test_gateway_health_refuses_wrong_pid_or_missing_job_membership(health, expected_pid, members):
    with pytest.raises(ValueError, match="gateway_health_owner"):
        preview_cutover._require_gateway_health_owner(health, expected_pid, members)


@pytest.mark.parametrize("legacy_ports", [False, True])
def test_prepared_replacement_follows_clean_normal_stop_without_path_constants(
    tmp_path, monkeypatch, legacy_ports
):
    plan_path, source_profile = _plan(tmp_path)
    (source_profile / ".env").write_bytes(b"opaque-auth-value")
    (source_profile / "SOUL.md").write_text("profile note", encoding="utf-8")
    lookup_plugin = source_profile / "plugins" / preview_cutover._PLUGIN
    lookup_plugin.mkdir(parents=True)
    (lookup_plugin / "__init__.py").write_text("# obsolete lookup policy")
    preserved_plugin = source_profile / "plugins" / "preserved-helper"
    preserved_plugin.mkdir()
    (preserved_plugin / "__init__.py").write_text("# preserved opaque file")
    (source_profile / "skills").mkdir()
    (source_profile / "skills" / "example.md").write_text("skill", encoding="utf-8")
    success = subprocess.CompletedProcess(
        ["preflight"],
        0,
        stdout=(
            '[voice-preview] {"category":"preflight_ready","launch_authorized":false,'
            '"ports":[59876,59877,7880,28765]}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(preview_cutover, "run_preflight", lambda *_: success)
    preview_cutover.prepare(plan_path)

    previous = tmp_path / "previous"
    (previous / "state").mkdir(parents=True)
    (previous / "result-receipt.private.json").write_text(
        json.dumps(
            {
                "version": 1,
                "category": "stopped",
                "cleanup_settled": True,
                "cleanup_failed": False,
                "owned_survivor_count": 0,
                "remaining_owned_listeners": 0,
                "protected_listeners_unchanged": False,
                "ports": [59876, 59877, 7880, 28765],
            }
        ),
        encoding="utf-8",
    )
    if legacy_ports:
        receipt_path = previous / "result-receipt.private.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.pop("ports")
        receipt.update(api_port=59876, companion_port=59877, livekit_port=7880, browser_port=28765)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (previous / "pids.json").write_text(
        json.dumps(
            {
                "gateway": 101,
                "host": 102,
                "livekit": 103,
            }
        ),
        encoding="utf-8",
    )
    (previous / "supervisor.pid.private.txt").write_text("104\n", encoding="ascii")
    (previous / "state" / "hermes-runs-v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "pending": [],
                "admitted": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    output = tmp_path / "preview"
    result = preview_cutover.finalize(output / "prepared.private.json", previous)

    assert result["category"] == "ready_to_launch"
    # The previous run is evidence only: nothing of its conversation enters the new run.
    assert not (output / "state").exists()
    assert not (tmp_path / "state-root").exists()
    staged_home = output / "input" / "home"
    assert not (staged_home / "plugins" / preview_cutover._PLUGIN).exists()
    assert (staged_home / "plugins/preserved-helper/__init__.py").is_file()
    assert (staged_home / ".env").read_bytes() == b"opaque-auth-value"
    assert (staged_home / "SOUL.md").read_text(encoding="utf-8") == "profile note"
    assert (staged_home / "skills" / "example.md").read_text(encoding="utf-8") == "skill"
    assert json.loads((output / "active.private.json").read_text())["profile_complete"] is True
    handoff = json.loads((output / "handoff.private.json").read_text())
    assert handoff["previous_stop"]["protected_listeners_unchanged"] is False


@pytest.mark.parametrize("failure", ["cleanup", "live_pid", "occupied_port", "journal"])
def test_finalize_refuses_unsettled_or_still_owned_previous_run(tmp_path, monkeypatch, failure):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    previous = tmp_path / "previous"
    (previous / "state").mkdir(parents=True)
    receipt = {
        "version": 1,
        "category": "stopped",
        "cleanup_settled": True,
        "cleanup_failed": False,
        "owned_survivor_count": 0,
        "remaining_owned_listeners": 0,
        "protected_listeners_unchanged": True,
        "ports": [59876, 59877, 7880, 28765],
    }
    if failure == "cleanup":
        receipt["cleanup_failed"] = True
    (previous / "result-receipt.private.json").write_text(json.dumps(receipt), encoding="utf-8")
    (previous / "pids.json").write_text(
        json.dumps({"gateway": 101, "host": 102, "livekit": 103}), encoding="utf-8"
    )
    (previous / "supervisor.pid.private.txt").write_text("104\n", encoding="ascii")
    journal = {"version": 1, "pending": [], "admitted": []}
    if failure == "journal":
        journal["pending"] = [{"run_id": "synthetic"}]
    (previous / "state" / "hermes-runs-v1.json").write_text(json.dumps(journal), encoding="utf-8")
    monkeypatch.setattr(
        preview_cutover,
        "_started_at",
        lambda pid: 123 if failure == "live_pid" and pid == 101 else None,
    )
    monkeypatch.setattr(
        preview_cutover,
        "_listener_rows",
        lambda port: [("synthetic", 123)] if failure == "occupied_port" and port == 59876 else [],
    )

    with pytest.raises(ValueError):
        preview_cutover.verify_normal_stop(previous, plan)


def test_legacy_ports_keep_strict_port_validation(tmp_path):
    plan_path, _ = _plan(tmp_path)
    previous = tmp_path / "legacy-preview"
    previous.mkdir()
    (previous / "result-receipt.private.json").write_text(json.dumps({
        "category": "stopped", "cleanup_settled": True, "cleanup_failed": False,
        "owned_survivor_count": 0, "remaining_owned_listeners": 0,
        "protected_listeners_unchanged": True,
        "api_port": True, "companion_port": 59877, "livekit_port": 7880, "browser_port": 28765,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="previous_ports_shape"):
        preview_cutover.verify_normal_stop(previous, preview_cutover.load_plan(plan_path))


# The abandoned run listened on its own ports, distinct from the replacement plan's.
_ABANDONED_PORTS = {
    "api": 59886, "companion": 59887, "livekit": 7890, "host": 28775, "protected": [8765],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _abandoned_previous(
    tmp_path: Path, plan_path: Path, *, journal: object = "empty", policy: object = None
) -> Path:
    """A run directory as `serve` leaves it when its owner dies before writing a receipt."""
    previous = tmp_path / "previous"
    (previous / "state").mkdir(parents=True)
    previous_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    previous_plan["output"] = str(previous)
    previous_plan["ports"] = dict(_ABANDONED_PORTS)
    if policy is not None:
        previous_plan["readiness_policy"] = policy
    (previous / "plan.private.json").write_text(json.dumps(previous_plan), encoding="utf-8")
    (previous / "prepared.private.json").write_text(json.dumps({
        "version": 1, "plan": previous_plan, "plan_sha256": _sha256(previous / "plan.private.json"),
        "output": str(previous), "policy_mode": "native", "profile_complete": False,
        "preflight": "accepted",
    }), encoding="utf-8")
    (previous / "handoff.private.json").write_text(
        json.dumps({"version": 1, "category": "normal_stop_handoff_verified"}), encoding="utf-8"
    )
    (previous / "active.private.json").write_text(json.dumps({
        "version": 1, "category": "ready_to_launch",
        "prepared_sha256": _sha256(previous / "prepared.private.json"),
        "handoff_sha256": _sha256(previous / "handoff.private.json"),
        "profile_complete": True, "launch_authorized": False,
    }), encoding="utf-8")
    # A host leaves both lock files behind; only a live one holds them.
    (previous / "state" / "hermes-runs-v1.json.lock").write_bytes(b"")
    (previous / "state" / "voice-tail-v1.json.lock").write_bytes(b"")
    (previous / "pids.json").write_text(
        json.dumps({"gateway": 101, "host": 102, "livekit": 103}), encoding="utf-8"
    )
    (previous / "supervisor.pid.private.txt").write_text("104\n", encoding="ascii")
    if journal == "empty":
        journal = {"version": 1, "pending": [], "admitted": []}
    if type(journal) is dict:
        (previous / "state" / "hermes-runs-v1.json").write_text(
            json.dumps(journal), encoding="utf-8"
        )
    return previous


def _tree_snapshot(root: Path) -> dict[str, object]:
    entries: dict[str, object] = {".": root.stat().st_mtime_ns}
    for item in sorted(root.rglob("*")):
        # Windows region locks are mandatory: a held lock file can be sized, never read.
        content = (
            None if item.is_dir()
            else item.stat().st_size if item.name.endswith(".lock")
            else item.read_bytes()
        )
        entries[item.relative_to(root).as_posix()] = (
            item.is_dir(), content, item.stat().st_mtime_ns,
        )
    return entries


@pytest.mark.parametrize("journal", ["empty", "absent"])
def test_finalize_accepts_abandoned_run_without_writing_it(tmp_path, monkeypatch, journal):
    plan_path, source_profile = _plan(tmp_path)
    (source_profile / ".env").write_bytes(b"opaque-auth-value")
    success = subprocess.CompletedProcess(
        ["preflight"],
        0,
        stdout=(
            '[voice-preview] {"category":"preflight_ready","launch_authorized":false,'
            '"ports":[59876,59877,7880,28765]}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(preview_cutover, "run_preflight", lambda *_: success)
    preview_cutover.prepare(plan_path)
    previous = _abandoned_previous(tmp_path, plan_path, journal=journal)
    before = _tree_snapshot(previous)
    observed: list[int] = []
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(
        preview_cutover, "_listener_rows", lambda port: observed.append(port) or []
    )
    output = tmp_path / "preview"

    result = preview_cutover.finalize(output / "prepared.private.json", previous)

    assert result["category"] == "ready_to_launch"
    assert not (output / "state").exists()
    assert sorted(observed) == sorted(
        {59876, 59877, 7880, 28765} | {59886, 59887, 7890, 28775}
    )
    handoff = json.loads((output / "handoff.private.json").read_text())
    assert handoff["category"] == "abandoned_stop_handoff_verified"
    stop = handoff["previous_stop"]
    assert stop["category"] == "abandoned_stop_verified"
    assert stop["receipt_sha256"] is None
    assert stop["plan_sha256"] == _sha256(previous / "plan.private.json")
    assert stop["processes_absent"] is True
    assert stop["listeners_absent"] is True
    # Only absence was observed; no owner was there to verify.
    assert stop["listener_ownership"] == "unavailable"
    assert stop["work_journal_empty"] is (True if journal == "empty" else None)
    assert stop["work_journal_absent"] is (journal == "absent")
    assert stop["protected_listeners_unchanged"] is None
    assert _tree_snapshot(previous) == before


def test_normal_stop_alone_still_requires_its_receipt(tmp_path):
    plan_path, _ = _plan(tmp_path)
    previous = _abandoned_previous(tmp_path, plan_path)
    with pytest.raises(ValueError, match="receipt_file"):
        preview_cutover.verify_normal_stop(previous, preview_cutover.load_plan(plan_path))


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        ("live_pid", "previous_process_alive"),
        ("live_supervisor", "previous_process_alive"),
        ("old_listener", "previous_listener_alive"),
        ("new_listener", "previous_listener_alive"),
        ("pending", "previous_work_unsettled"),
        ("admitted", "previous_work_unsettled"),
        ("journal_directory", "receipt_file"),
        ("pid_shape", "previous_pid_shape"),
        ("missing_supervisor", "previous_supervisor_pid_file"),
        ("plan_drift", "previous_plan_binding"),
        ("prepared_drift", "previous_plan_binding"),
        ("handoff_drift", "previous_plan_binding"),
        ("not_launched", "previous_plan_binding"),
        ("lock_directory", "previous_lock_file"),
        ("receipt_appeared", "previous_receipt_appeared"),
        ("foreign_output", "previous_plan_binding"),
        ("plan_ports", "previous_ports_shape"),
    ],
)
def test_abandoned_run_refuses_live_or_unsettled_or_unbound_state(
    tmp_path, monkeypatch, failure, category
):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    journal: object = "empty"
    if failure == "pending":
        journal = {"version": 1, "pending": [{"run_id": "synthetic"}], "admitted": []}
    if failure == "admitted":
        journal = {"version": 1, "pending": [], "admitted": ["run_synthetic"]}
    previous = _abandoned_previous(
        tmp_path, plan_path, journal="absent" if failure == "journal_directory" else journal
    )
    if failure == "journal_directory":
        (previous / "state" / "hermes-runs-v1.json").mkdir()
    if failure == "pid_shape":
        (previous / "pids.json").write_text(json.dumps({"gateway": 101}), encoding="utf-8")
    if failure == "missing_supervisor":
        (previous / "supervisor.pid.private.txt").unlink()
    if failure == "plan_drift":
        # Well formed, but not the plan prepared bound: only the hash can tell.
        plan_value = json.loads((previous / "plan.private.json").read_text())
        plan_value["ports"]["api"] = 59896
        (previous / "plan.private.json").write_text(json.dumps(plan_value), encoding="utf-8")
    if failure == "prepared_drift":
        prepared = json.loads((previous / "prepared.private.json").read_text())
        prepared["policy_mode"] = "lookup"
        (previous / "prepared.private.json").write_text(json.dumps(prepared), encoding="utf-8")
    if failure == "handoff_drift":
        (previous / "handoff.private.json").write_text("{}", encoding="utf-8")
    if failure == "lock_directory":
        (previous / "state" / "voice-tail-v1.json.lock").unlink()
        (previous / "state" / "voice-tail-v1.json.lock").mkdir()
    if failure == "not_launched":
        active = json.loads((previous / "active.private.json").read_text())
        active["category"] = "prepared"
        (previous / "active.private.json").write_text(json.dumps(active), encoding="utf-8")
    if failure in {"foreign_output", "plan_ports"}:
        plan_value = json.loads((previous / "plan.private.json").read_text())
        if failure == "foreign_output":
            plan_value["output"] = str(tmp_path / "elsewhere")
        else:
            plan_value["ports"]["api"] = plan_value["ports"]["companion"]
        (previous / "plan.private.json").write_text(json.dumps(plan_value), encoding="utf-8")
        prepared = json.loads((previous / "prepared.private.json").read_text())
        prepared["plan_sha256"] = _sha256(previous / "plan.private.json")
        if failure == "foreign_output":
            prepared["output"] = str(tmp_path / "elsewhere")
        (previous / "prepared.private.json").write_text(json.dumps(prepared), encoding="utf-8")
        active = json.loads((previous / "active.private.json").read_text())
        active["prepared_sha256"] = _sha256(previous / "prepared.private.json")
        (previous / "active.private.json").write_text(json.dumps(active), encoding="utf-8")
    live = {"live_pid": 102, "live_supervisor": 104}.get(failure)
    listening = {"old_listener": 59886, "new_listener": 59876}.get(failure)
    monkeypatch.setattr(preview_cutover, "_started_at", lambda pid: 1 if pid == live else None)
    receipt = previous / "result-receipt.private.json"

    def listener_rows(port):
        if failure == "receipt_appeared" and not receipt.exists():
            # A late owner settles while the abandoned checks run; its receipt is clean.
            receipt.write_text(json.dumps({
                "version": 1, "category": "stopped", "cleanup_settled": True,
                "cleanup_failed": False, "owned_survivor_count": 0,
                "remaining_owned_listeners": 0, "protected_listeners_unchanged": True,
                "ports": [59886, 59887, 7890, 28775],
            }), encoding="utf-8")
        return [("127.0.0.1", 999)] if port == listening else []

    monkeypatch.setattr(preview_cutover, "_listener_rows", listener_rows)
    before = _tree_snapshot(previous)

    with pytest.raises(ValueError, match=category):
        preview_cutover.verify_previous_stop(previous, plan)
    if failure == "receipt_appeared":
        # The appeared receipt alone would have been accepted as a normal stop.
        assert preview_cutover.verify_previous_stop(previous, plan)["category"] == (
            "normal_stop_verified"
        )
    else:
        assert _tree_snapshot(previous) == before


@pytest.mark.parametrize("stop", ["abandoned", "normal"])
@pytest.mark.parametrize("record", ["hermes-runs-v1.json", "voice-tail-v1.json"])
def test_a_held_host_lock_refuses_the_previous_run_without_touching_it(
    tmp_path, monkeypatch, stop, record
):
    from hermes_realtime.integration.run_record import lock_run_record, unlock_run_record

    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    previous = _abandoned_previous(tmp_path, plan_path)
    if stop == "normal":
        (previous / "result-receipt.private.json").write_text(json.dumps({
            "version": 1, "category": "stopped", "cleanup_settled": True,
            "cleanup_failed": False, "owned_survivor_count": 0,
            "remaining_owned_listeners": 0, "protected_listeners_unchanged": True,
            "ports": [59886, 59887, 7890, 28775],
        }), encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    # The host's own mechanism, on the host's own path: a second handle must not get it.
    owner = lock_run_record(previous / "state" / record)
    assert owner is not None
    try:
        before = _tree_snapshot(previous)
        with pytest.raises(ValueError, match="previous_lock_held"):
            preview_cutover.verify_previous_stop(previous, plan)
        assert _tree_snapshot(previous) == before
    finally:
        unlock_run_record(owner)
    # Released, the same run is accepted, and the probe leaves the lock files as they were.
    before = _tree_snapshot(previous)
    assert preview_cutover.verify_previous_stop(previous, plan)["category"] == (
        f"{stop}_stop_verified"
    )
    assert _tree_snapshot(previous) == before
    if stop == "abandoned":
        for name in ("hermes-runs-v1.json.lock", "voice-tail-v1.json.lock"):
            (previous / "state" / name).unlink()
        # A missing lock file is never created by the probe.
        assert preview_cutover.verify_previous_stop(previous, plan)["category"] == (
            "abandoned_stop_verified"
        )
        assert not (previous / "state" / "hermes-runs-v1.json.lock").exists()
        assert not (previous / "state" / "voice-tail-v1.json.lock").exists()


def _real_runtime() -> Path:
    from hermes_realtime.integration import voice_tail as voice_tail_module

    return Path(voice_tail_module.__file__).resolve().parents[2]


_STOPPED_RECEIPT = {
    "version": 1, "category": "stopped", "cleanup_settled": True, "cleanup_failed": False,
    "owned_survivor_count": 0, "remaining_owned_listeners": 0,
    "protected_listeners_unchanged": True, "ports": [59886, 59887, 7890, 28775],
}
_SAID = "synthetic-heard-before-the-restart"
_TAIL = "voice-tail-v1.json"
_DELETES = "voice-tail-v1.deletes.json"


def _prepared(root: Path, monkeypatch) -> Path:
    root.mkdir()
    plan_path, source_profile = _plan(root)
    (source_profile / ".env").write_bytes(b"opaque-auth-value")
    success = subprocess.CompletedProcess(
        ["preflight"], 0,
        stdout=(
            '[voice-preview] {"category":"preflight_ready","launch_authorized":false,'
            '"ports":[59876,59877,7880,28765]}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(preview_cutover, "run_preflight", lambda *_: success)
    preview_cutover.prepare(plan_path)
    return root / "preview"


def _state_plan(tmp_path: Path, output: str = "preview") -> Path:
    """A plan for one more run directory on the same state root, bound to the real runtime."""
    base = tmp_path / "plan.json"
    if not base.exists():
        _plan(tmp_path)
    value = json.loads(base.read_text(encoding="utf-8"))
    value["output"] = str(tmp_path / output)
    value["realtime"]["runtime_target"] = str(_real_runtime())
    path = tmp_path / f"plan-{output}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


async def _say(tail: Path, text: str, conversation_id: str = "old") -> None:
    """What a host does to its store: open under the lock, hear something, close."""
    from hermes_realtime.conversation import ConversationContextStore
    from hermes_realtime.integration.voice_tail import VoiceTailWriter
    from hermes_realtime.speech import Transcript

    writer = VoiceTailWriter(tail, conversation_ids=iter([conversation_id]).__next__)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    store.record_user_transcript(Transcript(text=text, final=True))
    await writer.close()


async def _restored(tail: Path) -> list[str]:
    """What the next host restores from the store, by the host's own open."""
    from hermes_realtime.conversation import ConversationContextStore
    from hermes_realtime.integration.voice_tail import VoiceTailWriter

    writer = VoiceTailWriter(tail, conversation_ids=iter(["new"]).__next__)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        return [message.text for message in store.snapshot().messages]
    finally:
        await writer.close()


async def _saved_run(root: Path, record: str = "none") -> Path:
    """A stopped run directory of the copying era, holding a saved conversation."""
    from hermes_realtime.integration.voice_deletes import VoiceDeletes, voice_deletes_bytes

    state = root / "state"
    state.mkdir(parents=True)
    await _say(state / _TAIL, _SAID)
    if record == "pending_forget":
        # The crash came after the record was written and before the cleared tail was.
        (state / _DELETES).write_bytes(
            voice_deletes_bytes(VoiceDeletes(pending=(("old", 0),)))
        )
    elif record == "other_live":
        (state / _DELETES).write_bytes(
            voice_deletes_bytes(VoiceDeletes(outcome=("older", 0), live=("newer", 1)))
        )
    return root


@pytest.mark.parametrize(
    ("edit", "category"),
    [
        ("missing", "plan_fields"),
        ("relative", "state_root_path"),
        ("parent_reference", "state_root_path"),
        ("not_text", "state_root_path"),
        ("unc", "state_root_path"),
        ("unc_forward", "state_root_path"),
        ("device", "state_root_path"),
        ("the_run_directory", "state_root_path"),
        ("inside_the_run_directory", "state_root_path"),
        ("around_the_run_directory", "state_root_path"),
    ],
)
def test_plan_requires_one_state_root_outside_every_run_directory(tmp_path, edit, category):
    path, _ = _plan(tmp_path)
    assert preview_cutover.load_plan(path).state_root == tmp_path / "state-root"
    value = json.loads(path.read_text(encoding="utf-8"))
    output = Path(value["output"])
    if edit == "missing":
        del value["state_root"]
    else:
        value["state_root"] = {
            "relative": "state-root",
            "parent_reference": str(tmp_path / "elsewhere" / ".." / "state-root"),
            "not_text": 7,
            "unc": "\\\\fileserver\\share\\state-root",
            "unc_forward": "//fileserver/share/state-root",
            "device": "\\\\?\\C:\\state-root",
            "the_run_directory": str(output),
            "inside_the_run_directory": str(output / "state"),
            "around_the_run_directory": str(output.parent),
        }[edit]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(preview_cutover.PreviewRefusal, match=category):
        preview_cutover.load_plan(path)


@pytest.mark.parametrize("kind", [0, 1, 2, 4, 5, 6, True, None])
def test_state_root_on_a_volume_that_is_not_a_local_fixed_disk_refuses(
    tmp_path, monkeypatch, kind
):
    plan = preview_cutover.load_plan(_plan(tmp_path)[0])
    anchors = []
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    monkeypatch.setattr(preview_cutover, "_drive_type", lambda anchor: anchors.append(anchor) or 3)
    assert preview_cutover._conversation_store(plan) == plan.state_root / _TAIL
    assert anchors == [plan.state_root.anchor]
    # Removable, network, optical, RAM or unknown: locks and atomic replace are not dependable.
    monkeypatch.setattr(preview_cutover, "_drive_type", lambda _anchor: kind)
    for use in (preview_cutover._conversation_store, preview_cutover._require_state_root_available):
        with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_not_local"):
            use(plan)
    with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_not_local"):
        preview_cutover.adopt_state(_state_plan(tmp_path), tmp_path / "source")
    assert not plan.state_root.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="native volume query")
def test_the_volume_of_a_local_temporary_directory_is_a_fixed_disk(tmp_path):
    assert preview_cutover._drive_type(tmp_path.anchor) == 3
    # No volume is mounted there: DRIVE_NO_ROOT_DIR, which is not a fixed disk either.
    assert preview_cutover._drive_type(str(tmp_path / "no-such-volume") + "\\") == 1


@pytest.mark.asyncio
async def test_two_consecutive_runs_share_one_tail_and_nothing_is_copied(tmp_path):
    import shutil

    first = preview_cutover.load_plan(_state_plan(tmp_path, "first-run"))
    second = preview_cutover.load_plan(_state_plan(tmp_path, "second-run"))
    store = preview_cutover._conversation_store(first)
    assert store == preview_cutover._conversation_store(second) == tmp_path / "state-root" / _TAIL
    # Genesis: the first run needs no predecessor and no existing store.
    assert not first.state_root.exists()
    preview_cutover._require_state_root_available(first)
    first.output.mkdir()
    (first.output / "evidence.private.json").write_text("{}", encoding="utf-8")

    await _say(store, _SAID)

    assert sorted(item.name for item in first.output.iterdir()) == ["evidence.private.json"]
    # Nothing in any run directory is needed to restore the conversation.
    shutil.rmtree(first.output)
    preview_cutover._require_state_root_available(second)
    assert await _restored(preview_cutover._conversation_store(second)) == [_SAID]
    assert not second.output.exists()


@pytest.mark.asyncio
async def test_a_second_owner_on_the_same_root_refuses_while_the_first_holds_it(tmp_path):
    from hermes_realtime.conversation import ConversationContextStore
    from hermes_realtime.integration.voice_tail import VoiceTailWriter

    plan = preview_cutover.load_plan(_state_plan(tmp_path, "second-run"))
    store = preview_cutover._conversation_store(plan)
    await _say(store, _SAID)
    holder = VoiceTailWriter(store, conversation_ids=iter(["held"]).__next__)
    await holder.open(ConversationContextStore(on_change=holder.update))
    try:
        before = _tree_snapshot(plan.state_root)
        with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_held"):
            preview_cutover._require_state_root_available(plan)
        # The probe is a courtesy. With it bypassed, the host's own open is what refuses.
        with pytest.raises(RuntimeError, match="another host holds the voice tail"):
            await _restored(store)
        assert _tree_snapshot(plan.state_root) == before
    finally:
        await holder.close()
    preview_cutover._require_state_root_available(plan)
    assert await _restored(store) == [_SAID]


_HOLDER = """
import asyncio, sys
from pathlib import Path
from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.integration.voice_tail import VoiceTailWriter
from hermes_realtime.speech import Transcript


async def main():
    tail, ready, said = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    writer = VoiceTailWriter(tail, conversation_ids=iter(["old"]).__next__)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    store.record_user_transcript(Transcript(text=said, final=True))
    while not (tail.exists() and said.encode() in tail.read_bytes()):
        await asyncio.sleep(0.05)
    ready.write_text("held", encoding="utf-8")
    await asyncio.sleep(600)


asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_a_killed_owner_leaves_a_root_the_next_run_acquires_with_the_tail_intact(tmp_path):
    import asyncio

    plan = preview_cutover.load_plan(_state_plan(tmp_path, "next-run"))
    store = preview_cutover._conversation_store(plan)
    script = tmp_path / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")
    ready = tmp_path / "holder.ready"
    with (tmp_path / "holder.log").open("wb") as log:
        owner = subprocess.Popen(
            [sys.executable, str(script), str(store), str(ready), _SAID],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(600):
                if ready.exists() or owner.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            assert ready.exists(), (tmp_path / "holder.log").read_text(errors="replace")
            # Another process holds the store: refused across the process boundary.
            with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_held"):
                preview_cutover._require_state_root_available(plan)
            with pytest.raises(RuntimeError, match="another host holds the voice tail"):
                await _restored(store)
        finally:
            owner.kill()
            owner.wait(timeout=30)
    # No stop, no receipt, no cleanup: the kernel releases the lock with the process. The
    # interpreter may be a child of the launcher that was killed, so its end is awaited.
    for _ in range(300):
        try:
            preview_cutover._require_state_root_available(plan)
            break
        except preview_cutover.PreviewRefusal:
            await asyncio.sleep(0.1)
    stale = plan.state_root / ".voice-tail-v1.json.abcd1234.tmp"
    stale.write_text("synthetic-unfinished-write", encoding="utf-8")
    preview_cutover._require_state_root_available(plan)

    assert await _restored(store) == [_SAID]
    # The next lock holder clears a killed writer's temporaries; none outlive it in a run.
    assert not stale.exists()


@pytest.mark.parametrize(
    ("fault", "category"),
    [
        ("root_is_a_file", "state_root_directory"),
        ("lock_is_a_directory", "state_root_lock_file"),
        ("deletes_without_tail", "state_root_incomplete"),
        ("empty_directory", None),
        ("tail_only", None),
        ("tail_and_deletes", None),
    ],
)
def test_state_root_is_checked_before_a_run_owns_anything(tmp_path, fault, category):
    plan = preview_cutover.load_plan(_plan(tmp_path)[0])
    root = plan.state_root
    if fault == "root_is_a_file":
        root.write_bytes(b"")
    else:
        root.mkdir()
    if fault == "lock_is_a_directory":
        (root / (_TAIL + ".lock")).mkdir()
    if fault in {"deletes_without_tail", "tail_and_deletes"}:
        (root / _DELETES).write_text("{}", encoding="utf-8")
    if fault in {"tail_only", "tail_and_deletes"}:
        (root / _TAIL).write_text("{}", encoding="utf-8")
    before = _tree_snapshot(tmp_path)

    if category is None:
        preview_cutover._require_state_root_available(plan)
    else:
        with pytest.raises(preview_cutover.PreviewRefusal, match=category):
            preview_cutover._require_state_root_available(plan)
    assert _tree_snapshot(tmp_path) == before


def test_serve_refuses_a_held_state_root_before_it_owns_anything(tmp_path, monkeypatch):
    from hermes_realtime.integration.run_record import lock_run_record, unlock_run_record
    from scripts import lookup_backend_task, rehearse_desktop_mvp

    plan = preview_cutover.load_plan(_plan(tmp_path)[0])
    plan.output.mkdir()
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda path: (plan, {}, plan.output))
    monkeypatch.setattr(preview_cutover, "preflight", lambda *_: None)
    monkeypatch.setattr(lookup_backend_task, "listener_rows", lambda _: [])
    monkeypatch.setattr(
        rehearse_desktop_mvp, "Containment",
        lambda: pytest.fail("a run that cannot have the store must not own children"),
    )
    owner = lock_run_record(preview_cutover._conversation_store(plan))
    assert owner is not None
    try:
        before = _tree_snapshot(tmp_path)
        with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_held"):
            preview_cutover.serve(plan.output / "active.private.json")
        # Nothing was created, so the run directory can be launched again later.
        assert _tree_snapshot(tmp_path) == before
    finally:
        unlock_run_record(owner)
    tree = ast.parse(inspect.getsource(preview_cutover.serve))
    (probe_line,) = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node) == "_require_state_root_available(plan)"
    ]
    # The later of the two mentions is the exclusive create that makes the run single-use.
    pid_line = max(
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "supervisor.pid.private.txt"
    )
    assert probe_line < pid_line


@pytest.mark.asyncio
async def test_host_gets_the_state_root_tail_and_its_own_run_record(tmp_path, monkeypatch):
    path, profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(path)
    plan.output.mkdir()
    (plan.output / "host.stop").touch()
    seen = {}

    class Launcher:
        async def start(self):
            pass

        async def close(self):
            pass

    def build(*, hermes_discover_tools=False, **kwargs):
        seen.update(kwargs)
        return Launcher()

    host = SimpleNamespace(
        _load_api_bearer=lambda _: "synthetic-not-a-credential",
        _load_voice_companion=lambda *_: None,
        LoopbackPeerAuthorizer=object,
        build_local_host_launcher=build,
    )
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda _: (plan, None, profile))
    monkeypatch.setattr(preview_cutover, "_runtime_modules", lambda _: (None, host))

    await preview_cutover._host_child(tmp_path / "active")

    assert seen["voice_tail"] == tmp_path / "state-root" / _TAIL
    assert not seen["voice_tail"].is_relative_to(plan.output)
    # The run record stays with the run: its Hermes home does not outlive it either.
    assert seen["hermes_run_record"] == plan.output / "state" / "hermes-runs-v1.json"


def test_first_launch_needs_no_predecessor(tmp_path, monkeypatch, capsys):
    output = _prepared(tmp_path / "first", monkeypatch)
    plan = preview_cutover.load_plan(output / "plan.private.json")
    assert not plan.state_root.exists()

    assert preview_cutover.main(
        ["finalize", "--prepared", str(output / "prepared.private.json")]
    ) == 0

    status = json.loads(capsys.readouterr().out.removeprefix("[voice-preview] "))
    assert status["category"] == "ready_to_launch"
    handoff = json.loads((output / "handoff.private.json").read_text())
    assert handoff["category"] == "no_previous_handoff"
    assert handoff["previous_stop"] is None
    assert set(handoff) == {"version", "category", "previous_stop", "profile"}
    assert (output / "input" / "home" / ".env").read_bytes() == b"opaque-auth-value"
    # Finalizing neither creates nor needs conversation state.
    assert not (output / "state").exists()
    assert not plan.state_root.exists()
    calls = []
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    helper = SimpleNamespace(
        PreviewOwnerLaunchRefusal=_OwnerLaunchRefusal,
        launch_preview_owner=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "scripts.preview_owner_launch", helper)
    active = output / "active.private.json"

    assert preview_cutover.main(["launch", "--active", str(active)]) == 0

    assert calls == [{
        "interpreter": plan.executables["host_python"], "active": active.resolve(),
        "output": plan.output,
    }]
    launched = json.loads(capsys.readouterr().out.removeprefix("[voice-preview] "))
    assert launched["category"] == "launch_requested"
    preview_cutover._require_state_root_available(plan)


@pytest.mark.asyncio
async def test_a_run_that_ended_abnormally_does_not_block_the_next_run_on_its_state_root(
    tmp_path, monkeypatch, capsys
):
    output = _prepared(tmp_path / "new", monkeypatch)
    plan = preview_cutover.load_plan(output / "plan.private.json")
    store = preview_cutover._conversation_store(plan)
    # The earlier run's host heard something, then exited on a fault: a refused receipt,
    # work still recorded in its own run record, and no clean stop.
    await _say(store, _SAID)
    ended = _abandoned_previous(
        tmp_path, tmp_path / "new" / "plan.json",
        journal={"version": 1, "pending": [{"run_id": "synthetic"}],
                 "admitted": ["run_synthetic"]},
    )
    (ended / "result-receipt.private.json").write_text(json.dumps({
        "version": 1, "category": "refused", "cleanup_settled": False,
        "cleanup_failed": True, "owned_survivor_count": 0, "remaining_owned_listeners": 0,
        "protected_listeners_unchanged": True, "run_journal_empty": False,
        "failure_category": "owned_child_exited", "ports": [59886, 59887, 7890, 28775],
    }), encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    before = _tree_snapshot(ended)
    # As evidence that run is still refused, and the refusal binds nothing.
    with pytest.raises(preview_cutover.PreviewRefusal, match="previous_normal_stop"):
        preview_cutover.finalize(output / "prepared.private.json", ended)
    assert not (output / "handoff.private.json").exists()

    # Without it, nothing reads that run: its receipt and its journal decide nothing.
    assert preview_cutover.main(
        ["finalize", "--prepared", str(output / "prepared.private.json")]
    ) == 0
    assert json.loads((output / "handoff.private.json").read_text())["category"] == (
        "no_previous_handoff"
    )
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    helper = SimpleNamespace(
        PreviewOwnerLaunchRefusal=_OwnerLaunchRefusal, launch_preview_owner=lambda **_: None,
    )
    monkeypatch.setitem(sys.modules, "scripts.preview_owner_launch", helper)
    assert preview_cutover.main(
        ["launch", "--active", str(output / "active.private.json")]
    ) == 0
    statuses = [
        json.loads(line.removeprefix("[voice-preview] "))["category"]
        for line in capsys.readouterr().out.splitlines()
    ]
    assert statuses == ["ready_to_launch", "launch_requested"]
    # What serve checks before it owns anything, then what the new host restores.
    preview_cutover._require_state_root_available(plan)
    assert await _restored(store) == [_SAID]
    assert _tree_snapshot(ended) == before


@pytest.mark.parametrize("store", ["empty", "adopted"])
def test_finalize_does_not_leave_an_earlier_runs_conversation_behind(
    tmp_path, monkeypatch, store
):
    output = _prepared(tmp_path / "new", monkeypatch)
    previous = _abandoned_previous(tmp_path, tmp_path / "new" / "plan.json")
    # A run of the copying era: its conversation is still in its own directory.
    (previous / "state" / _TAIL).write_text("{}", encoding="utf-8")
    plan = preview_cutover.load_plan(output / "plan.private.json")
    if store == "adopted":
        plan.state_root.mkdir()
        (plan.state_root / _TAIL).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    before = _tree_snapshot(previous)

    if store == "adopted":
        preview_cutover.finalize(output / "prepared.private.json", previous)
        assert (output / "active.private.json").is_file()
    else:
        with pytest.raises(
            preview_cutover.PreviewRefusal, match="previous_conversation_not_adopted"
        ):
            preview_cutover.finalize(output / "prepared.private.json", previous)
        # Refused before the launch is bound, so the same prepared run finalizes later.
        assert not (output / "handoff.private.json").exists()
        assert not (output / "input" / "home" / ".env").exists()
        assert not plan.state_root.exists()
    assert _tree_snapshot(previous) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("record", ["pending_forget", "other_live", "none"])
async def test_adopt_carries_a_saved_conversation_with_its_forget_record(
    tmp_path, capsys, record
):
    source = await _saved_run(tmp_path / "stopped-run", record)
    plan_path = _state_plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    before = _tree_snapshot(source)
    assert not plan.state_root.exists()

    assert preview_cutover.main(
        ["adopt-state", "--plan", str(plan_path), "--source", str(source)]
    ) == 0

    raw = capsys.readouterr().out
    result = json.loads(raw.removeprefix("[voice-preview] "))
    assert _tree_snapshot(source) == before
    assert str(tmp_path) not in raw and _SAID not in raw
    store = plan.state_root / _TAIL
    assert result["category"] == "state_adopted" and result["already_adopted"] is False
    assert result["sha256"] == _sha256(store) == _sha256(source / "state" / _TAIL)
    assert result["bytes"] == store.stat().st_size
    if record == "none":
        assert result["deletes_sha256"] is None and result["deletes_bytes"] is None
        assert not (plan.state_root / _DELETES).exists()
    else:
        assert result["deletes_sha256"] == _sha256(plan.state_root / _DELETES)
        assert (plan.state_root / _DELETES).read_bytes() == (
            source / "state" / _DELETES
        ).read_bytes()
    adopted = _tree_snapshot(plan.state_root)
    # Running it again changes nothing and says so.
    again = preview_cutover.adopt_state(plan_path, source)
    assert again == dict(result, already_adopted=True)
    assert _tree_snapshot(plan.state_root) == adopted
    assert _tree_snapshot(source) == before
    preview_cutover._require_state_root_available(plan)
    # The host's own open decides: a forgotten conversation is retired, not restored.
    assert await _restored(preview_cutover._conversation_store(plan)) == (
        [_SAID] if record == "none" else []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "occupant",
    [
        "other_tail",
        "other_tail_same_deletes",
        "other_deletes_without_tail",
        "deletes_the_source_lacks",
        "same_tail_without_the_source_deletes",
        "same_tail_with_other_deletes",
        "tail_is_a_directory",
    ],
)
async def test_adopt_refuses_a_state_root_that_holds_anything_else(tmp_path, occupant):
    from hermes_realtime.integration.voice_deletes import VoiceDeletes, voice_deletes_bytes

    source_has_deletes = occupant != "deletes_the_source_lacks"
    source = await _saved_run(
        tmp_path / "stopped-run", "pending_forget" if source_has_deletes else "none"
    )
    plan_path = _state_plan(tmp_path)
    root = preview_cutover.load_plan(plan_path).state_root
    root.mkdir()
    source_tail = (source / "state" / _TAIL).read_bytes()
    other_deletes = voice_deletes_bytes(VoiceDeletes(pending=(("another", 3),)))
    if occupant in {"other_tail", "other_tail_same_deletes"}:
        await _say(root / _TAIL, "synthetic-newer-conversation", "newer")
    if occupant == "other_tail_same_deletes":
        (root / _DELETES).write_bytes((source / "state" / _DELETES).read_bytes())
    if occupant in {"other_deletes_without_tail", "deletes_the_source_lacks",
                    "same_tail_with_other_deletes"}:
        (root / _DELETES).write_bytes(other_deletes)
    if occupant in {"same_tail_without_the_source_deletes", "same_tail_with_other_deletes"}:
        (root / _TAIL).write_bytes(source_tail)
    if occupant == "tail_is_a_directory":
        (root / _TAIL).mkdir()

    def contents():
        # Taking the store's lock creates its lock file; nothing else may change.
        return {
            item.name: None if item.is_dir() else item.read_bytes()
            for item in root.iterdir() if item.name != _TAIL + ".lock"
        }

    before = contents()
    source_before = _tree_snapshot(source)

    with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_not_empty"):
        preview_cutover.adopt_state(plan_path, source)

    # Refused before anything is written: nothing there is overwritten or completed.
    assert contents() == before
    assert _tree_snapshot(source) == source_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "killed_at"),
    [
        ("pending_forget", "before_any_write"),
        ("pending_forget", "inside_first_write"),
        ("pending_forget", "between_writes"),
        ("pending_forget", "after_last_write"),
        ("none", "before_any_write"),
        ("none", "inside_first_write"),
        ("none", "after_last_write"),
    ],
)
async def test_adopt_killed_at_any_step_is_completed_by_running_it_again(
    tmp_path, monkeypatch, record, killed_at
):
    from hermes_realtime.integration import run_record

    source = await _saved_run(tmp_path / "stopped-run", record)
    plan_path = _state_plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    before = _tree_snapshot(source)
    write = run_record.write_run_record
    written: list[str] = []

    def killed(path, data):
        if killed_at == "inside_first_write" and not written:
            # The process died after mkstemp wrote plaintext and before the replace.
            (path.parent / f".{path.name}.abcd1234.tmp").write_bytes(data)
            raise KeyboardInterrupt
        if killed_at == "before_any_write" or (killed_at == "between_writes" and written):
            raise KeyboardInterrupt
        write(path, data)
        written.append(path.name)
        if killed_at == "after_last_write" and path.name == _TAIL:
            # Written and replaced, but never verified or reported.
            raise KeyboardInterrupt

    monkeypatch.setattr(run_record, "write_run_record", killed)
    with pytest.raises(KeyboardInterrupt):
        preview_cutover.adopt_state(plan_path, source)
    monkeypatch.setattr(run_record, "write_run_record", write)

    assert _tree_snapshot(source) == before
    present = {item.name for item in plan.state_root.iterdir()} - {_TAIL + ".lock"}
    first = _DELETES if record == "pending_forget" else _TAIL
    if killed_at == "before_any_write":
        assert present == set()
    elif killed_at == "inside_first_write":
        assert present == {f".{first}.abcd1234.tmp"}
    elif killed_at == "between_writes":
        # Only one order exists: the forget record never trails its tail.
        assert present == {_DELETES}
    else:
        assert present == {_TAIL} | ({_DELETES} if record == "pending_forget" else set())
    if present == {_DELETES}:
        # A host must not start on the forget record alone and begin a new conversation.
        with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_incomplete"):
            preview_cutover._require_state_root_available(plan)
    # The lock died with the process, so the same command finishes the same adoption.
    result = preview_cutover.adopt_state(plan_path, source)

    assert result["already_adopted"] is (killed_at == "after_last_write")
    assert _tree_snapshot(source) == before
    expected = {_TAIL, _TAIL + ".lock"} | ({_DELETES} if record == "pending_forget" else set())
    assert {item.name for item in plan.state_root.iterdir()} == expected
    assert (plan.state_root / _TAIL).read_bytes() == (source / "state" / _TAIL).read_bytes()
    preview_cutover._require_state_root_available(plan)
    assert await _restored(preview_cutover._conversation_store(plan)) == (
        [_SAID] if record == "none" else []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "category"),
    [
        ("source_tail_lock_held", "adopt_source_held"),
        ("source_run_lock_held", "adopt_source_held"),
        ("source_lock_directory", "adopt_source_lock_file"),
        ("store_held", "state_root_held"),
        ("no_tail", "adopt_source_tail_missing"),
        ("tail_directory", "adopt_source_tail"),
        ("tail_oversized", "adopt_source_tail"),
        ("deletes_directory", "adopt_source_deletes"),
        ("deletes_oversized", "adopt_source_deletes"),
        ("root_is_a_file", "state_root_directory"),
    ],
)
async def test_adopt_refuses_a_live_or_unsafe_source_or_store(tmp_path, fault, category):
    from hermes_realtime.integration.run_record import lock_run_record, unlock_run_record
    from hermes_realtime.integration.voice_deletes import MAX_VOICE_DELETES_BYTES
    from hermes_realtime.integration.voice_tail import max_voice_tail_bytes

    source = await _saved_run(tmp_path / "stopped-run", "pending_forget")
    state = source / "state"
    plan_path = _state_plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    owner = None
    if fault == "source_tail_lock_held":
        owner = lock_run_record(state / _TAIL)
    elif fault == "source_run_lock_held":
        owner = lock_run_record(state / "hermes-runs-v1.json")
    elif fault == "store_held":
        owner = lock_run_record(preview_cutover._conversation_store(plan))
    elif fault == "source_lock_directory":
        (state / (_TAIL + ".lock")).unlink()
        (state / (_TAIL + ".lock")).mkdir()
    elif fault == "no_tail":
        (state / _TAIL).unlink()
    elif fault == "tail_directory":
        (state / _TAIL).unlink()
        (state / _TAIL).mkdir()
    elif fault == "tail_oversized":
        (state / _TAIL).write_bytes(b" " * (max_voice_tail_bytes(32, 1024) + 1))
    elif fault == "deletes_directory":
        (state / _DELETES).unlink()
        (state / _DELETES).mkdir()
    elif fault == "deletes_oversized":
        (state / _DELETES).write_bytes(b" " * (MAX_VOICE_DELETES_BYTES + 1))
    elif fault == "root_is_a_file":
        plan.state_root.write_bytes(b"")
    assert owner is not None or "held" not in fault
    before = _tree_snapshot(tmp_path)
    try:
        with pytest.raises(preview_cutover.PreviewRefusal, match=category):
            preview_cutover.adopt_state(plan_path, source)
        assert _tree_snapshot(tmp_path) == before
    finally:
        if owner is not None:
            unlock_run_record(owner)


@pytest.mark.asyncio
async def test_adopt_refuses_a_source_that_changed_underneath_it(tmp_path, monkeypatch):
    from hermes_realtime.integration import run_record

    source = await _saved_run(tmp_path / "stopped-run")
    plan_path = _state_plan(tmp_path)
    write = run_record.write_run_record

    def a_live_host_writes(path, data):
        write(path, data)
        (source / "state" / _TAIL).write_bytes(data + b" ")

    monkeypatch.setattr(run_record, "write_run_record", a_live_host_writes)

    with pytest.raises(preview_cutover.PreviewRefusal, match="adopt_source_changed"):
        preview_cutover.adopt_state(plan_path, source)


def test_adopt_refuses_a_store_that_does_not_hold_what_was_read(tmp_path, monkeypatch):
    import asyncio

    from hermes_realtime.integration import run_record

    source = asyncio.run(_saved_run(tmp_path / "stopped-run"))
    plan_path = _state_plan(tmp_path)
    write = run_record.write_run_record
    monkeypatch.setattr(
        run_record, "write_run_record", lambda path, data: write(path, data[:-1])
    )

    with pytest.raises(preview_cutover.PreviewRefusal, match="state_root_copy"):
        preview_cutover.adopt_state(plan_path, source)


@pytest.mark.parametrize(
    ("previous_policy", "new_policy", "category"),
    [
        ("dev_owned_services", "kernel_listeners", "previous_readiness_policy"),
        ("dev_owned_services", None, "previous_readiness_policy"),
        ("dev_owned_services", "dev_owned_services", None),
        ("kernel_listeners", "dev_owned_services", None),
        (None, "kernel_listeners", None),
        ("best_effort", "dev_owned_services", "previous_readiness_policy"),
        (True, "dev_owned_services", "previous_readiness_policy"),
    ],
)
def test_abandoned_run_pairs_readiness_policy_as_a_clean_stop_does(
    tmp_path, monkeypatch, previous_policy, new_policy, category
):
    plan_path, _ = _plan(tmp_path)
    previous = _abandoned_previous(tmp_path, plan_path, policy=previous_policy)
    if new_policy is not None:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
        value["readiness_policy"] = new_policy
        plan_path.write_text(json.dumps(value), encoding="utf-8")
    plan = preview_cutover.load_plan(plan_path)
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    monkeypatch.setattr(preview_cutover, "_tcp_accepting", lambda _port: False)
    before = _tree_snapshot(previous)

    if category is None:
        assert preview_cutover.verify_previous_stop(previous, plan)["category"] == (
            "abandoned_stop_verified"
        )
    else:
        with pytest.raises(preview_cutover.PreviewRefusal, match=category):
            preview_cutover.verify_previous_stop(previous, plan)
    assert _tree_snapshot(previous) == before
    # The clean-stop path decides the same pairing for the same two policies.
    if type(previous_policy) is str and previous_policy in {
        "kernel_listeners", "dev_owned_services"
    }:
        receipt = dict(_STOPPED_RECEIPT, readiness_policy=previous_policy)
        if previous_policy == "dev_owned_services":
            receipt |= {
                "listener_ownership": "unavailable", "remaining_owned_listeners": None,
                "protected_listeners_unchanged": None, "remaining_accepting_endpoints": 0,
                "protected_endpoints_unchanged": None,
            }
        (previous / "result-receipt.private.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        if category is None:
            assert preview_cutover.verify_previous_stop(previous, plan)["category"] == (
                "normal_stop_verified"
            )
        else:
            with pytest.raises(preview_cutover.PreviewRefusal):
                preview_cutover.verify_previous_stop(previous, plan)


@pytest.mark.parametrize("stop", ["abandoned", "normal"])
def test_handoff_counts_crashed_writer_temporaries_without_touching_them(
    tmp_path, monkeypatch, stop
):
    output = _prepared(tmp_path / "new", monkeypatch)
    previous = _abandoned_previous(tmp_path, tmp_path / "new" / "plan.json")
    if stop == "normal":
        (previous / "result-receipt.private.json").write_text(
            json.dumps(_STOPPED_RECEIPT), encoding="utf-8"
        )
    state = previous / "state"
    marker = "synthetic-heard-text"
    # Exactly the names run_record.write_run_record leaves behind for the three host records.
    for name in (
        ".hermes-runs-v1.json.abcd1234.tmp",
        ".voice-tail-v1.json.a_b_c_d0.tmp",
        ".voice-tail-v1.json.zzzzzzzz.tmp",
        ".voice-tail-v1.deletes.json.00000000.tmp",
    ):
        (state / name).write_text(marker, encoding="utf-8")
    # Neighbours that are not such temporaries.
    for name in (
        ".voice-tail-v1.json.abcd123.tmp",
        ".voice-tail-v1.json.ABCD1234.tmp",
        ".other-record.json.abcd1234.tmp",
        "voice-tail-v1.json.abcd1234.tmp",
        ".voice-tail-v1.json.abcd1234.tmp.bak",
    ):
        (state / name).write_text(marker, encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    before = _tree_snapshot(previous)

    preview_cutover.finalize(output / "prepared.private.json", previous)

    assert _tree_snapshot(previous) == before
    raw = (output / "handoff.private.json").read_text(encoding="utf-8")
    assert json.loads(raw)["previous_stop"]["state_temporaries"] == 4
    # Counts only: no name and no content of a temporary reaches the handoff.
    assert marker not in raw and ".tmp" not in raw
    assert not (output / "state").exists()


def test_state_temporaries_scan_is_bounded(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    previous = _abandoned_previous(tmp_path, plan_path)
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    assert preview_cutover.verify_previous_stop(previous, plan)["state_temporaries"] == 0
    monkeypatch.setattr(preview_cutover, "_MAX_STATE_ENTRIES", 4)
    for index in range(2):
        (previous / "state" / f"extra-{index}").write_bytes(b"")

    with pytest.raises(preview_cutover.PreviewRefusal, match="previous_state_bound"):
        preview_cutover.verify_previous_stop(previous, plan)


@pytest.mark.parametrize("stop", ["abandoned", "normal"])
@pytest.mark.parametrize("unobservable", [101, 104])
def test_a_process_that_cannot_be_observed_is_not_read_as_dead(
    tmp_path, monkeypatch, stop, unobservable
):
    from scripts import rehearse_desktop_mvp

    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    previous = _abandoned_previous(tmp_path, plan_path)
    if stop == "normal":
        (previous / "result-receipt.private.json").write_text(
            json.dumps(_STOPPED_RECEIPT), encoding="utf-8"
        )

    def started_at(pid):
        if pid == unobservable:
            raise rehearse_desktop_mvp.ProcessUnobservable("synthetic private detail")
        return None

    monkeypatch.setattr(rehearse_desktop_mvp, "started_at", started_at)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    before = _tree_snapshot(previous)

    with pytest.raises(preview_cutover.PreviewRefusal, match="previous_process_unobservable"):
        preview_cutover.verify_previous_stop(previous, plan)
    assert _tree_snapshot(previous) == before


@pytest.mark.parametrize(
    "receipt", ["directory", "dangling_junction", "dangling_link", "settled_link"]
)
def test_receipt_that_is_not_a_plain_file_is_never_treated_as_absent(
    tmp_path, monkeypatch, receipt
):
    plan_path, _ = _plan(tmp_path)
    previous = _abandoned_previous(tmp_path, plan_path)
    path = previous / "result-receipt.private.json"
    if receipt == "directory":
        path.mkdir()
    elif receipt == "dangling_junction":
        if sys.platform != "win32":
            pytest.skip("junctions are Windows-only")
        import _winapi

        target_dir = tmp_path / "removed-target"
        target_dir.mkdir()
        _winapi.CreateJunction(str(target_dir), str(path))
        target_dir.rmdir()
        # Path.exists() follows the reparse point and would call this receipt absent.
        assert not path.exists()
    else:
        target = tmp_path / "outside-receipt.json"
        if receipt == "settled_link":
            target.write_text(json.dumps({
                "version": 1, "category": "stopped", "cleanup_settled": True,
                "cleanup_failed": False, "owned_survivor_count": 0,
                "remaining_owned_listeners": 0, "protected_listeners_unchanged": True,
                "ports": [59886, 59887, 7890, 28775],
            }), encoding="utf-8")
        try:
            path.symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])

    with pytest.raises(ValueError, match="receipt_file"):
        preview_cutover.verify_previous_stop(previous, preview_cutover.load_plan(plan_path))


@pytest.mark.parametrize(
    ("receipt", "category"),
    [
        ("unsettled", "previous_normal_stop"),
        ("malformed", None),
        # Absence of the journal is accepted only for a run that never wrote a receipt.
        ("settled_without_journal", "receipt_file"),
    ],
)
def test_present_receipt_keeps_its_refusal_even_when_the_run_looks_abandoned(
    tmp_path, monkeypatch, receipt, category
):
    plan_path, _ = _plan(tmp_path)
    previous = _abandoned_previous(
        tmp_path, plan_path, journal="absent" if receipt == "settled_without_journal" else "empty"
    )
    path = previous / "result-receipt.private.json"
    if receipt == "settled_without_journal":
        path.write_text(json.dumps({
            "version": 1, "category": "stopped", "cleanup_settled": True,
            "cleanup_failed": False, "owned_survivor_count": 0,
            "remaining_owned_listeners": 0, "protected_listeners_unchanged": True,
            "ports": [59886, 59887, 7890, 28775],
        }), encoding="utf-8")
    elif receipt == "unsettled":
        path.write_text(json.dumps({
            "version": 1, "category": "refused", "cleanup_settled": False,
            "cleanup_failed": True, "owned_survivor_count": 1,
            "remaining_owned_listeners": 0, "protected_listeners_unchanged": True,
            "ports": [59886, 59887, 7890, 28775],
        }), encoding="utf-8")
    else:
        path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])
    plan = preview_cutover.load_plan(plan_path)

    for verify in (preview_cutover.verify_previous_stop, preview_cutover.verify_normal_stop):
        if category is not None:
            with pytest.raises(ValueError, match=category):
                verify(previous, plan)
        else:
            with pytest.raises(json.JSONDecodeError):
                verify(previous, plan)


def test_realtime_source_binding_refuses_before_wheel_acceptance(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_value["realtime"]["source"] = str(preview_cutover.ROOT)
    plan_path.write_text(json.dumps(plan_value), encoding="utf-8")
    plan = preview_cutover.load_plan(plan_path)
    candidate_wheel = __import__("scripts.candidate_wheel", fromlist=["unused"])
    monkeypatch.setattr(
        "scripts.task13_artifact_orchestrator.verify_clean_tracked_candidate",
        lambda _path: SimpleNamespace(candidate_head_oid="0" * 40, candidate_tree_oid="b" * 40),
    )
    monkeypatch.setattr(candidate_wheel, "verify_candidate_wheel_v1", lambda *_: pytest.fail())

    with pytest.raises(ValueError, match="realtime_source_binding"):
        preview_cutover.verify_realtime_runtime(plan)


def test_candidate_wheel_verifier_refusal_is_not_accepted(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_value["realtime"]["source"] = str(preview_cutover.ROOT)
    plan_path.write_text(json.dumps(plan_value), encoding="utf-8")
    plan = preview_cutover.load_plan(plan_path)
    monkeypatch.setattr(
        "scripts.task13_artifact_orchestrator.verify_clean_tracked_candidate",
        lambda _path: SimpleNamespace(
            candidate_head_oid=plan.realtime_revision, candidate_tree_oid=plan.realtime_tree
        ),
    )
    monkeypatch.setattr(
        "scripts.setup_lookup_backend.verify_runtime",
        lambda *_, **__: (_ for _ in ()).throw(ValueError("candidate wheel rejected")),
    )

    with pytest.raises(ValueError, match="candidate wheel rejected"):
        preview_cutover.verify_realtime_runtime(plan)


@pytest.mark.parametrize("drift", [None, "record", "source"])
def test_preview_runtime_uses_scoped_comparator_without_release_archive(
    tmp_path, monkeypatch, drift
):
    plan_path, _ = _plan(tmp_path)
    value = json.loads(plan_path.read_text())
    value["realtime"]["source"] = str(preview_cutover.ROOT)
    plan_path.write_text(json.dumps(value))
    plan = preview_cutover.load_plan(plan_path)
    identity = SimpleNamespace(candidate_head_oid=plan.realtime_revision,
                               candidate_tree_oid=plan.realtime_tree)
    identities = iter([identity, SimpleNamespace(candidate_head_oid="0" * 40,
                       candidate_tree_oid=plan.realtime_tree) if drift == "source" else identity])
    monkeypatch.setattr("scripts.task13_artifact_orchestrator.verify_clean_tracked_candidate",
                        lambda _: next(identities))
    monkeypatch.setattr("scripts.candidate_source_archive_oracle.capture_candidate_source_archive",
                        lambda *_: pytest.fail("release archive must stay outside DEV preflight"))
    seen = []
    def compare(manifest, **kwargs):
        seen.append((manifest, kwargs))
        return {"recordSha256": "0" * 64 if drift == "record" else plan.runtime_record_sha256,
                "packageFiles": 114}
    monkeypatch.setattr("scripts.setup_lookup_backend.verify_runtime", compare)
    monkeypatch.setattr("scripts.setup_lookup_backend.git", lambda path, *args:
                        plan.hermes_revision if args[-1] == "HEAD^{commit}" else
                        plan.hermes_tree if args[-1] == "HEAD^{tree}" else "")
    monkeypatch.setattr("scripts.real_gate_support.installed_hermes_identity",
                        lambda *_: {"commit": plan.hermes_revision})
    for path in (*plan.executables.values(), plan.hermes_python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if drift is not None:
        category = "runtime_record_binding" if drift == "record" else "realtime_source_changed"
        with pytest.raises(ValueError, match=category):
            preview_cutover.verify_realtime_runtime(plan)
        return
    result = preview_cutover.verify_realtime_runtime(plan)
    assert result["package_files"] == 114
    assert seen[0][1] == {"source_root": plan.realtime_source,
                          "hermes_revision": plan.hermes_revision}
    assert seen[0][0]["wheel_sha256"] == plan.wheel_sha256
    assert seen[0][0]["runtime_target"] == plan.runtime_target


def test_lookup_gateway_manifest_uses_verified_distribution_record(tmp_path):
    path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(path)
    record = plan.runtime_target / "hermes_realtime-1.2.3.dist-info" / "RECORD"
    record.parent.mkdir(parents=True)
    record.write_bytes(b"fixture")
    plan.output.mkdir()
    manifest_path = preview_cutover._gateway_manifest(plan)
    assert json.loads(manifest_path.read_text())["recordPath"] == str(record)


def test_isolated_operator_can_import_existing_process_owner():
    code = (
        "import runpy; runpy.run_path(" + repr(str(Path(preview_cutover.__file__))) + "); "
        "from scripts.rehearse_desktop_mvp import Processes, Containment, started_at"
    )
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_windows_python_listener_may_be_an_owned_role_descendant():
    assert preview_cutover._require_owned_endpoint_rows(
        [("127.0.0.1", 502), ("127.0.0.1", 502)], 501, {501, 502, 503}, 2,
        role_pids={501, 502},
    ) == 502


@pytest.mark.parametrize("role_pids,members", [({501}, {501, 502}), ({501, 502}, {501})])
def test_listener_refuses_wrong_role_or_foreign_job(role_pids, members):
    with pytest.raises(ValueError, match="endpoint_owner"):
        preview_cutover._require_owned_endpoint_rows(
            [("127.0.0.1", 502)], 501, members, 1, role_pids=role_pids)


class _OwnerLaunchRefusal(RuntimeError):
    def __init__(self, category):
        self.category = category
        super().__init__(category)


@pytest.mark.parametrize("relative", [False, True])
def test_persistent_launch_binds_plan_and_never_claims_readiness(tmp_path, monkeypatch, relative):
    active = tmp_path / "active.private.json"
    plan = SimpleNamespace(output=tmp_path, executables={"host_python": tmp_path / "bound-python"})
    calls = []
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda path: (plan, {}, tmp_path))
    helper = SimpleNamespace(
        PreviewOwnerLaunchRefusal=_OwnerLaunchRefusal,
        launch_preview_owner=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "scripts.preview_owner_launch", helper)

    monkeypatch.chdir(tmp_path)
    result = preview_cutover.launch_preview(Path(active.name) if relative else active)

    assert calls == [{"interpreter": plan.executables["host_python"],
                      "active": active, "output": tmp_path}]
    assert result == {"category": "launch_requested", "launch_authorized": True, "ready": False}
    assert not (tmp_path / "supervisor.pid.private.txt").exists()
    assert not (tmp_path / "ready-receipt.private.json").exists()


def test_launch_cli_reports_only_the_launch_request(tmp_path, monkeypatch, capsys):
    calls = []

    def launch(active):
        calls.append(active)
        return {"category": "launch_requested", "launch_authorized": True, "ready": False}

    monkeypatch.setattr(preview_cutover, "launch_preview", launch)
    active = tmp_path / "active.private.json"
    assert preview_cutover.main(["launch", "--active", str(active)]) == 0
    assert calls == [active]
    output = capsys.readouterr().out
    assert output.startswith("[voice-preview] ")
    assert json.loads(output.removeprefix("[voice-preview] "))["ready"] is False


def test_persistent_launch_preserves_fixed_spawn_refusal(tmp_path, monkeypatch):
    plan = SimpleNamespace(output=tmp_path, executables={"host_python": tmp_path / "bound-python"})
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda path: (plan, {}, tmp_path))

    def refuse(**kwargs):
        raise _OwnerLaunchRefusal("breakaway_not_permitted")

    helper = SimpleNamespace(
        PreviewOwnerLaunchRefusal=_OwnerLaunchRefusal, launch_preview_owner=refuse)
    monkeypatch.setitem(sys.modules, "scripts.preview_owner_launch", helper)
    with pytest.raises(preview_cutover.PreviewRefusal, match="breakaway_not_permitted"):
        preview_cutover.launch_preview(tmp_path / "active.private.json")


def test_actual_serve_refuses_inherited_job_before_containment_or_children(tmp_path, monkeypatch):
    from scripts import lookup_backend_task, rehearse_desktop_mvp

    (tmp_path / "state").mkdir()
    (tmp_path / "state/hermes-runs-v1.json").write_text(
        json.dumps({"version": 1, "pending": [], "admitted": []}), encoding="utf-8")
    ports = {"api": 49001, "companion": 49002, "livekit": 49003, "host": 49004,
             "protected": [49005]}
    plan = SimpleNamespace(output=tmp_path, ports=ports, readiness_policy="kernel_listeners",
                           state_root=tmp_path.parent / (tmp_path.name + "-state"),
                           executables={"host_python": tmp_path / "bound-python"})
    calls = []
    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda path: (plan, {}, tmp_path))
    monkeypatch.setattr(preview_cutover, "preflight", lambda *_: None)
    monkeypatch.setattr(lookup_backend_task, "listener_rows", lambda _: [])

    def inherited(*, interpreter):
        assert interpreter == plan.executables["host_python"]
        raise _OwnerLaunchRefusal("owner_inherited_job")

    def forbidden_containment():
        calls.append("containment")
        raise AssertionError("child ownership must not start")

    monkeypatch.setattr(rehearse_desktop_mvp, "Containment", forbidden_containment)
    helper = SimpleNamespace(
        PreviewOwnerLaunchRefusal=_OwnerLaunchRefusal, assert_independent_owner=inherited)
    monkeypatch.setitem(sys.modules, "scripts.preview_owner_launch", helper)

    assert preview_cutover.serve(tmp_path / "active.private.json") == 1

    assert calls == []
    result = json.loads((tmp_path / "result-receipt.private.json").read_text(encoding="utf-8"))
    assert result["failure_category"] == "owner_inherited_job"
    assert result["cleanup_settled"] is True
    assert not (tmp_path / "ready-receipt.private.json").exists()


def test_dev_readiness_policy_is_explicit_and_default_stays_strict(tmp_path):
    path, _ = _plan(tmp_path)
    assert preview_cutover.load_plan(path).readiness_policy == "kernel_listeners"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["readiness_policy"] = "dev_owned_services"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert preview_cutover.load_plan(path).readiness_policy == "dev_owned_services"


@pytest.mark.parametrize("policy", [True, None, "best_effort", {}, ""])
def test_readiness_policy_rejects_unknown_or_coerced_values(tmp_path, policy):
    path, _ = _plan(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["readiness_policy"] = policy
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="readiness_policy"):
        preview_cutover.load_plan(path)


@pytest.mark.parametrize("policy", ["kernel_listeners", "dev_owned_services"])
@pytest.mark.parametrize("receipt_patch", [
    {}, {"remaining_accepting_endpoints": 1}, {"remaining_accepting_endpoints": False},
    {"listener_ownership": "verified"}, {"remaining_owned_listeners": 0},
    {"protected_listeners_unchanged": True}, {"protected_endpoints_unchanged": "unknown"},
    {"protected_endpoints_unchanged": None},
])
def test_dev_stop_never_becomes_kernel_listener_proof(
    tmp_path, monkeypatch, policy, receipt_patch,
):
    path, _ = _plan(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["readiness_policy"] = policy
    path.write_text(json.dumps(value), encoding="utf-8")
    plan = preview_cutover.load_plan(path)
    previous = tmp_path / "previous"
    (previous / "state").mkdir(parents=True)
    receipt = {"version": 1, "category": "stopped", "cleanup_settled": True,
               "cleanup_failed": False, "owned_survivor_count": 0,
               "readiness_policy": "dev_owned_services", "listener_ownership": "unavailable",
               "remaining_owned_listeners": None, "protected_listeners_unchanged": None,
               "remaining_accepting_endpoints": 0, "protected_endpoints_unchanged": True,
               "ports": [59876, 59877, 7880, 28765]}
    receipt.update(receipt_patch)
    (previous / "result-receipt.private.json").write_text(json.dumps(receipt), encoding="utf-8")
    (previous / "pids.json").write_text(json.dumps({"gateway": 101, "livekit": 102,
                                                   "host": 103}), encoding="utf-8")
    (previous / "supervisor.pid.private.txt").write_text("104", encoding="ascii")
    (previous / "state/hermes-runs-v1.json").write_text(json.dumps(
        {"version": 1, "pending": [], "admitted": []}), encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _: None)
    monkeypatch.setattr(preview_cutover, "_tcp_accepting", lambda _: False, raising=False)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _: pytest.fail(
        "DEV finalization must not claim an unavailable kernel observation"))
    optional_only = receipt_patch == {"protected_endpoints_unchanged": None}
    if policy == "kernel_listeners" or (receipt_patch and not optional_only):
        with pytest.raises(ValueError, match="previous_normal_stop"):
            preview_cutover.verify_normal_stop(previous, plan)
    else:
        result = preview_cutover.verify_normal_stop(previous, plan)
        assert result["listeners_absent"] is None
        assert result["listener_ownership"] == "unavailable"
        assert result["endpoints_not_accepting"] is True


def test_dev_refusal_keeps_its_bounded_category():
    from scripts.preview_readiness import PreviewReadinessRefusal

    refusal = PreviewReadinessRefusal("tcp_observation_unavailable")
    assert preview_cutover._failure_status(refusal) == {
        "category": "refused", "failure_kind": "ValueError",
        "failure_category": "tcp_observation_unavailable",
    }


def _runtime_plan_with_real_hermes_checkout(tmp_path, monkeypatch):
    from dataclasses import replace

    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    checkout = tmp_path / "hermes-checkout"
    checkout.mkdir()

    def git(*arguments):
        return subprocess.run(
            ("git", *arguments), cwd=checkout, capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    (checkout / "entry.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "entry.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "source")
    (checkout / "spare.txt").write_text("VALUE = 2\n", encoding="utf-8")
    other_blob = git("hash-object", "-w", "spare.txt")
    (checkout / "spare.txt").unlink()
    git("update-index", "--add", "--cacheinfo", f"100644,{other_blob},ENTRY.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "case collision")
    git("checkout", "-q", "-f", "HEAD")
    plan = replace(
        plan, realtime_source=preview_cutover.ROOT, hermes_source=checkout,
        hermes_revision=git("rev-parse", "HEAD"), hermes_tree=git("rev-parse", "HEAD^{tree}"),
    )
    identity = SimpleNamespace(
        candidate_head_oid=plan.realtime_revision, candidate_tree_oid=plan.realtime_tree,
    )
    monkeypatch.setattr("scripts.task13_artifact_orchestrator.verify_clean_tracked_candidate",
                        lambda _: identity)
    monkeypatch.setattr("scripts.setup_lookup_backend.verify_runtime",
                        lambda *_, **__: {"recordSha256": plan.runtime_record_sha256,
                                          "packageFiles": 114})
    for executable in (*plan.executables.values(), plan.hermes_python):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    return plan, checkout, git


def test_preview_source_accepts_committed_case_aliases(tmp_path, monkeypatch):
    plan, checkout, git = _runtime_plan_with_real_hermes_checkout(tmp_path, monkeypatch)
    # The fixture uses a real two-spelling Git tree. On Windows one on-disk
    # file aliases both paths; its bytes still match exactly one committed blob.
    assert "ENTRY.py" in git("ls-tree", "-r", "--name-only", "HEAD")
    assert "entry.py" in git("ls-tree", "-r", "--name-only", "HEAD")
    assert preview_cutover.verify_realtime_runtime(plan)["package_files"] == 114


@pytest.mark.parametrize("fault", ["edited_bytes", "hidden_edit", "revision", "tree"])
def test_preview_source_case_aliases_never_hide_drift(tmp_path, monkeypatch, fault):
    from dataclasses import replace

    plan, checkout, git = _runtime_plan_with_real_hermes_checkout(tmp_path, monkeypatch)
    if fault in {"edited_bytes", "hidden_edit"}:
        if fault == "hidden_edit":
            git("update-index", "--assume-unchanged", "entry.py")
        (checkout / "entry.py").write_text("VALUE = 99\n", encoding="utf-8")
    elif fault == "revision":
        plan = replace(plan, hermes_revision="0" * 40)
    else:
        plan = replace(plan, hermes_tree="0" * 40)
    with pytest.raises((ValueError, RuntimeError)):
        preview_cutover.verify_realtime_runtime(plan)


@pytest.mark.parametrize("selector", ["environment", "replacement"])
@pytest.mark.parametrize("binding", ["actual", "foreign"])
def test_preview_source_tree_ignores_foreign_git_selection(
    tmp_path, monkeypatch, binding, selector,
):
    from dataclasses import replace

    plan, checkout, source_git = _runtime_plan_with_real_hermes_checkout(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    def git(*arguments):
        return subprocess.run(
            ("git", *arguments), cwd=foreign, capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    (foreign / "foreign.py").write_text("FOREIGN = True\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "foreign source")
    foreign_tree = git("rev-parse", "HEAD^{tree}")
    if selector == "environment":
        monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    else:
        (checkout / "extra.py").write_text("EXTRA = True\n", encoding="utf-8")
        source_git("add", "extra.py")
        source_git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                   "commit", "-qm", "replacement source")
        replacement = source_git("rev-parse", "HEAD")
        foreign_tree = source_git("rev-parse", "HEAD^{tree}")
        source_git("reset", "--hard", plan.hermes_revision)
        source_git("replace", plan.hermes_revision, replacement)
    assert foreign_tree != plan.hermes_tree
    if binding == "foreign":
        plan = replace(plan, hermes_tree=foreign_tree)
        with pytest.raises(ValueError, match="hermes_source_binding"):
            preview_cutover.verify_realtime_runtime(plan)
    else:
        assert preview_cutover.verify_realtime_runtime(plan)["package_files"] == 114


@pytest.mark.parametrize("mapping", [{}, {"A P I": "API"}, {"x" * 64: "y" * 128}])
def test_optional_pronunciations_preserve_exact_mapping(tmp_path, mapping):
    path, _ = _plan(tmp_path)
    value = json.loads(path.read_text())
    value["speech"]["pronunciations"] = mapping
    path.write_text(json.dumps(value))
    assert preview_cutover.load_plan(path).speech["pronunciations"] == mapping


@pytest.mark.parametrize("mapping", [
    None, [], "API", {"": "API"}, {" A": "API"}, {"A ": "API"},
    {"A": ""}, {"A": " API"}, {"A": "API "}, {"A": 1},
    {"A" * 65: "API"}, {"A": "B" * 129}, {"A\nP": "API"},
    {"A": "B\u200b"}, {"A": "one", "a": "two"},
    {str(i): "word" for i in range(65)},
])
def test_optional_pronunciations_refuse_invalid_mapping(tmp_path, mapping):
    path, _ = _plan(tmp_path)
    value = json.loads(path.read_text())
    value["speech"]["pronunciations"] = mapping
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="speech_pronunciations"):
        preview_cutover.load_plan(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("mapping", [None, {}, {"A P I": "API"}])
@pytest.mark.parametrize("supported", [True, False])
async def test_optional_pronunciations_host_forwarding(tmp_path, monkeypatch, mapping, supported):
    path, profile = _plan(tmp_path)
    plan = preview_cutover.load_plan(path)
    if mapping is not None:
        plan.speech["pronunciations"] = mapping
    plan.output.mkdir()
    (plan.output / "host.stop").touch()
    observed = []

    class Launcher:
        async def start(self):
            observed.append("start")
        async def close(self):
            observed.append("close")

    def supported_build(*, hermes_discover_tools=False, kokoro_pronunciations=None, **kwargs):
        assert kokoro_pronunciations == (mapping or None)
        return Launcher()

    def unsupported_build(*, hermes_discover_tools=False, **kwargs):
        assert "kokoro_pronunciations" not in kwargs
        return Launcher()

    host = SimpleNamespace(
        _load_api_bearer=lambda _: "synthetic-not-a-credential",
        _load_voice_companion=lambda *_: None,
        LoopbackPeerAuthorizer=object,
        build_local_host_launcher=supported_build if supported else unsupported_build,
    )
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda _: (plan, None, profile))
    monkeypatch.setattr(preview_cutover, "_runtime_modules", lambda _: (None, host))
    if mapping and not supported:
        with pytest.raises(ValueError, match="kokoro_pronunciations_host_unavailable"):
            await preview_cutover._host_child(tmp_path / "active")
        assert observed == []
    else:
        await preview_cutover._host_child(tmp_path / "active")
        assert observed == ["start", "close"]


@pytest.mark.parametrize("mapping", [None, {}, {"A P I": "API"}])
@pytest.mark.parametrize("supported", [True, False])
def test_optional_pronunciations_preflight_capability(tmp_path, monkeypatch, mapping, supported):
    path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(path)
    if mapping is not None:
        plan.speech["pronunciations"] = mapping
    plan.runtime_target.mkdir()

    def supported_build(*, hermes_discover_tools=False, kokoro_pronunciations=None):
        pass

    def unsupported_build(*, hermes_discover_tools=False):
        pass

    host = SimpleNamespace(
        __file__=str(plan.runtime_target / "hermes_realtime" / "host_launcher.py"),
        build_local_host_launcher=supported_build if supported else unsupported_build,
    )
    monkeypatch.setattr(preview_cutover.importlib, "import_module", lambda _: host)
    before = list(sys.path)
    if mapping and not supported:
        with pytest.raises(ValueError, match="kokoro_pronunciations_host_unavailable"):
            preview_cutover.verify_host_profile_mode(plan)
    else:
        preview_cutover.verify_host_profile_mode(plan)
    assert sys.path == before


_EXAMPLES = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.skipif(sys.platform != "win32", reason="the example plan uses Windows paths")
def test_shipped_native_examples_satisfy_the_offline_validators(tmp_path):
    from scripts.native_work_profile import StatusServer, build_native_work_profile

    plan = preview_cutover.load_plan(_EXAMPLES / "preview-plan.example.json")
    assert plan.policy_mode == "native"
    config = preview_cutover._profile_config(
        (_EXAMPLES / "native-profile-config.example.yaml").read_bytes()
    )
    preview_cutover._validate_native_model_binding(config, plan)
    launcher = tmp_path / "status.exe"
    launcher.write_bytes(b"synthetic launcher fixture")
    state = tmp_path / "state"
    state.mkdir()
    server = StatusServer(launcher, hashlib.sha256(launcher.read_bytes()).hexdigest(), state)
    build_native_work_profile(config, server)


@pytest.mark.parametrize(
    "edit",
    [
        lambda value: value.pop("native_status"),
        lambda value: value["models"].update(background="other-model"),
    ],
    ids=["plan_field_missing", "profile_model_mismatch"],
)
@pytest.mark.skipif(sys.platform != "win32", reason="the example plan uses Windows paths")
def test_shipped_native_examples_still_fail_closed(tmp_path, edit):
    value = json.loads((_EXAMPLES / "preview-plan.example.json").read_text(encoding="utf-8"))
    edit(value)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    config = preview_cutover._profile_config(
        (_EXAMPLES / "native-profile-config.example.yaml").read_bytes()
    )
    with pytest.raises(ValueError):
        preview_cutover._validate_native_model_binding(
            config, preview_cutover.load_plan(path)
        )


def test_make_speech_receipt_round_trips_through_the_consumer(tmp_path, capsys):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    receipt_path = plan.speech_assets["receipt"]
    expected = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_path.unlink()

    assert preview_cutover.main(["make-speech-receipt", "--plan", str(plan_path)]) == 0

    assert json.loads(receipt_path.read_text(encoding="utf-8")) == expected
    assert '"category":"speech_receipt_written"' in capsys.readouterr().out
    assert len(preview_cutover._speech_asset_manifest(plan)) == 3


def test_make_speech_receipt_never_overwrites_an_existing_receipt(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    before = plan.speech_assets["receipt"].read_bytes()

    with pytest.raises(ValueError, match="speech_receipt_exists"):
        preview_cutover.make_speech_receipt(plan)

    assert plan.speech_assets["receipt"].read_bytes() == before


def test_make_speech_receipt_refuses_a_kokoro_file_that_does_not_match_its_pin(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.speech_assets["receipt"].unlink()
    (plan.speech_assets["kokoro_cache"] / _FAKE_MODEL.filename).write_bytes(
        b"x" * len(_FAKE_MODEL_BYTES)
    )

    with pytest.raises(ValueError, match="kokoro_model_asset"):
        preview_cutover.make_speech_receipt(plan)

    assert not plan.speech_assets["receipt"].exists()


def test_make_speech_receipt_refuses_a_missing_pinned_voice(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.speech_assets["receipt"].unlink()
    (plan.speech_assets["kokoro_cache"] / _FAKE_VOICES.filename).unlink()

    with pytest.raises(ValueError, match="kokoro_voice_asset"):
        preview_cutover.make_speech_receipt(plan)


def test_make_speech_receipt_refuses_an_empty_moonshine_cache(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.speech_assets["receipt"].unlink()
    for item in plan.speech_assets["moonshine_cache"].rglob("*"):
        if item.is_file():
            item.unlink()

    with pytest.raises(ValueError, match="speech_cache_receipt_files"):
        preview_cutover.make_speech_receipt(plan)


def test_make_speech_receipt_refuses_nested_cache_roots(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.speech_assets["receipt"].unlink()
    # The Kokoro cache lives under the shared asset root.
    plan.speech_assets["moonshine_cache"] = plan.speech_assets["kokoro_cache"].parents[1]

    with pytest.raises(ValueError, match="speech_cache_roots"):
        preview_cutover.make_speech_receipt(plan)


@pytest.mark.skipif(sys.platform != "win32", reason="needs a case-insensitive filesystem")
def test_make_speech_receipt_refuses_a_pinned_name_with_different_case(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.speech_assets["receipt"].unlink()
    model = plan.speech_assets["kokoro_cache"] / _FAKE_MODEL.filename
    model.rename(model.with_name(_FAKE_MODEL.filename.upper()))

    with pytest.raises(ValueError, match="kokoro_model_asset"):
        preview_cutover.make_speech_receipt(plan)

    assert not plan.speech_assets["receipt"].exists()


def test_make_speech_receipt_refuses_a_linked_moonshine_file(tmp_path):
    plan_path, _ = _plan(tmp_path)
    plan = preview_cutover.load_plan(plan_path)
    plan.speech_assets["receipt"].unlink()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    link = plan.speech_assets["moonshine_cache"] / "linked.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(ValueError, match="speech_asset_link"):
        preview_cutover.make_speech_receipt(plan)
