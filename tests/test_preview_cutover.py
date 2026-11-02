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


def test_failure_metadata_keeps_operator_category_without_echoing_dependency_errors():
    try:
        preview_cutover.require(False, "runtime_import_origin")
    except ValueError as error:
        assert preview_cutover._failure_status(error)["failure_category"] == "runtime_import_origin"
    private = "synthetic_private_exception_value"
    assert private not in json.dumps(preview_cutover._failure_status(ValueError(private)))


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
    assert not (tmp_path / "preview" / "input" / "home" / ".env").exists()


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

    host_tree = ast.parse(inspect.getsource(preview_cutover._host_child))
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
    (previous / "state" / "voice-tail-v1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(preview_cutover, "_started_at", lambda _pid: None)
    monkeypatch.setattr(preview_cutover, "_listener_rows", lambda _port: [])

    def copy_tail(_previous, output, _runtime):
        (output / "state").mkdir()
        (output / "state" / "voice-tail-v1.json").write_text("{}", encoding="utf-8")
        return {"bytes": 2, "sha256": "e" * 64}

    monkeypatch.setattr(preview_cutover, "_copy_voice_tail", copy_tail)
    output = tmp_path / "preview"
    result = preview_cutover.finalize(output / "prepared.private.json", previous)

    assert result["category"] == "ready_to_launch"
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
    monkeypatch.setattr(preview_cutover, "_git", lambda path, *args:
                        plan.hermes_revision if args[-1] == "HEAD^{commit}" else
                        plan.hermes_tree if args[-1] == "HEAD^{tree}" else "")
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
