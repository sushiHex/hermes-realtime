from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_realtime.integration import codex_access_credentials as credentials
from scripts import preview_cutover


def configured(tmp_path):
    source = Path(__file__).parents[1] / "src"
    helper = source / "hermes_realtime/integration/codex_access_credentials.py"
    executable = Path(sys._base_executable).resolve()
    argv = credentials.command_argv(executable, helper, tmp_path / "auth.json",
                                    hashlib.sha256(b"synthetic-account").hexdigest())
    base = "https://chatgpt.com/backend-api/codex"
    config = {
        "model": {"default": "test-model", "provider": "custom:live"},
        "providers": {"live": {"base_url": base, "default_model": "test-model",
                                "api_mode": "codex_responses",
                                "key_cmd": credentials.shell_command(argv),
                                "extra_headers": {"ChatGPT-Account-ID": "synthetic-account"}}},
        "plugins": {"entries": {"hermes-realtime": {"settings": {"web_evidence": True,
            "presentation": {"provider": "custom", "base_url": base, "model": "test-model",
                             "api_mode": "codex_responses", "key_cmd_argv": argv}}}}},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    plan = SimpleNamespace(runtime_target=source, policy_mode="native", profile_source=tmp_path,
                           hermes_python=Path(sys.executable),
                           executables={"host_python": Path(sys.executable)})
    return plan, config, executable


def test_preflight_rejects_a_different_native_credential_command(tmp_path, monkeypatch):
    plan, config, executable = configured(tmp_path)
    config["providers"]["live"]["key_cmd"] += " shadow-command"
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    monkeypatch.setattr(preview_cutover, "_native_credential_python", lambda _: executable,
                        raising=False)
    with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
        preview_cutover.verify_host_profile_mode(plan)


def test_preflight_accepts_equal_reviewed_live_credential_roles(tmp_path, monkeypatch):
    plan, _, executable = configured(tmp_path)
    monkeypatch.setattr(preview_cutover, "_native_credential_python", lambda _: executable,
                        raising=False)
    preview_cutover.verify_host_profile_mode(plan)


def test_native_base_probe_uses_isolation_without_auth_material(tmp_path, monkeypatch):
    base = tmp_path / "python.exe"
    base.write_bytes(b"synthetic-base")
    observed = []

    def run(command, **options):
        observed.append((command, options))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"base": str(base)}), stderr="")

    monkeypatch.setattr(preview_cutover.subprocess, "run", run)
    plan = SimpleNamespace(hermes_python=tmp_path / "venv.exe")
    assert preview_cutover._native_credential_python(plan) == base
    assert observed[0][0][1:4] == ["-I", "-B", "-S"]
    assert set(observed[0][1]["env"]) <= {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
                                           "PROGRAMFILES"}


def test_live_credential_helper_runtime_origin_is_bound(tmp_path, monkeypatch):
    plan, config, executable = configured(tmp_path)
    monkeypatch.setattr(preview_cutover, "_native_credential_python", lambda _: executable)
    monkeypatch.setattr(credentials, "__file__", str(tmp_path / "unrelated-helper.py"))
    with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
        preview_cutover._verify_native_credentials(plan, config)


@pytest.mark.parametrize("fault", ["exit", "output_bound", "schema", "relative", "missing"])
def test_native_base_probe_refuses_invalid_reply(tmp_path, monkeypatch, fault):
    base = tmp_path / "python.exe"
    base.write_bytes(b"synthetic-base")
    monkeypatch.chdir(tmp_path)
    value = {"base": str(base)}
    returncode = 0
    if fault == "exit":
        returncode = 1
    elif fault == "schema":
        value["unexpected"] = True
    elif fault == "relative":
        value["base"] = "python.exe"
    elif fault == "missing":
        value["base"] = str(tmp_path / "absent.exe")
    stdout = json.dumps(value)
    if fault == "output_bound":
        stdout = " " * 4096 + stdout
    monkeypatch.setattr(preview_cutover.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=""))
    plan = SimpleNamespace(hermes_python=tmp_path / "venv.exe")
    with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
        preview_cutover._native_credential_python(plan)


@pytest.mark.parametrize("helper,field", [
    ("_native_credential_python", "base"), ("_native_credential_packages", "packages"),
])
def test_native_metadata_reply_refuses_nonempty_stderr(tmp_path, monkeypatch, helper, field):
    target = tmp_path / "native-target"
    if field == "base":
        target.write_bytes(b"synthetic-base")
    else:
        target.mkdir()
    monkeypatch.setattr(preview_cutover.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps({field: str(target)}), stderr="SYNTHETIC_PRIVATE_VALUE"))
    plan = SimpleNamespace(hermes_python=tmp_path / "venv.exe")
    with pytest.raises(preview_cutover.PreviewRefusal, match="^native_credential_binding$"):
        getattr(preview_cutover, helper)(plan)


def runtime_fixture(tmp_path, monkeypatch):
    from hermes_realtime.integration import native_credential_probe
    plan, config, executable = configured(tmp_path)
    plan.output = tmp_path / "output"
    home = plan.output / "input" / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(json.dumps(config))
    plan.hermes_source = tmp_path / "native-source"
    plan.ports = {"api": 12345}
    packages = tmp_path / "native-packages"
    packages.mkdir()
    environment = {"HERMES_HOME": str(home), "SYNTHETIC_ONLY": "value"}
    calls = []
    monkeypatch.setattr(preview_cutover, "_native_credential_python", lambda _: executable)
    monkeypatch.setattr(preview_cutover, "_native_credential_packages", lambda _: packages)
    def gateway(profile, port):
        assert profile == home and port == plan.ports["api"]
        return environment
    monkeypatch.setattr(preview_cutover, "_gateway_environment", gateway)
    monkeypatch.setattr(native_credential_probe, "verify_native_credential_resolution",
                        lambda **kwargs: calls.append(kwargs))
    return SimpleNamespace(plan=plan, config=config, executable=executable, packages=packages,
                           home=home, module=native_credential_probe, calls=calls,
                           environment=environment)


@pytest.mark.parametrize("home_kind", ["source", "unplanned"])
def test_runtime_resolution_never_runs_bookkeeping_in_source_or_unplanned_home(
    tmp_path, monkeypatch, home_kind,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    target = s.plan.profile_source if home_kind == "source" else tmp_path / "unplanned"
    target.mkdir(exist_ok=True)
    (target / "config.yaml").write_text(json.dumps(s.config))
    # A guard omission must reach a functioning synthetic probe, not fail a fixture assertion.
    monkeypatch.setattr(preview_cutover, "_gateway_environment", lambda *args: {})
    with pytest.raises(preview_cutover.PreviewRefusal, match="^native_credential_binding$"):
        preview_cutover._verify_native_credential_runtime(s.plan, target)
    assert s.calls == []


@pytest.mark.parametrize("helper,field", [
    ("_native_credential_python", "base"), ("_native_credential_packages", "packages"),
])
@pytest.mark.parametrize("fault", ["exit", "bound", "schema", "duplicate", "relative",
                                    "missing", "wrong_kind", "timeout", "spawn", "path_kind"])
def test_native_metadata_helpers_fail_closed_on_untrusted_reply(
    tmp_path, monkeypatch, helper, field, fault,
):
    import subprocess
    target = tmp_path / "target"
    if field == "base":
        target.write_bytes(b"synthetic")
    else:
        target.mkdir()
    value = {field: str(target)}
    if fault == "schema":
        value["extra"] = "SYNTHETIC_PRIVATE_VALUE"
    elif fault == "relative":
        monkeypatch.chdir(tmp_path)
        relative = tmp_path / "relative"
        if field == "base":
            relative.write_bytes(b"synthetic")
        else:
            relative.mkdir()
        value[field] = "relative"
    elif fault == "missing":
        value[field] = str(tmp_path / "absent")
    elif fault == "wrong_kind":
        value[field] = True
    elif fault == "path_kind":
        alternate = tmp_path / "opposite-kind"
        if field == "base":
            alternate.mkdir()
        else:
            alternate.write_bytes(b"synthetic")
        value[field] = str(alternate)
    raw = json.dumps(value)
    if fault == "bound":
        raw += " " * 4096
    elif fault == "duplicate":
        raw = raw.replace('"' + field + '":', '"' + field + '":"duplicate","' + field + '":')
    def run(*args, **kwargs):
        if fault == "timeout":
            raise subprocess.TimeoutExpired("synthetic", 15, output="SYNTHETIC_PRIVATE_VALUE")
        if fault == "spawn":
            raise OSError("SYNTHETIC_PRIVATE_VALUE")
        return SimpleNamespace(returncode=1 if fault == "exit" else 0, stdout=raw, stderr="")
    monkeypatch.setattr(preview_cutover.subprocess, "run", run)
    with pytest.raises(
        preview_cutover.PreviewRefusal, match="^native_credential_binding$",
    ) as error:
        getattr(preview_cutover, helper)(SimpleNamespace(hermes_python=tmp_path / "native-python"))
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(error.value)


@pytest.mark.parametrize("helper,field,isolated_flags", [
    ("_native_credential_python", "base", ["-I", "-B", "-S"]),
    ("_native_credential_packages", "packages", ["-I", "-B"]),
])
def test_native_metadata_helpers_use_only_pinned_interpreter_and_curated_environment(
    tmp_path, monkeypatch, helper, field, isolated_flags,
):
    import subprocess
    target = tmp_path / "target"
    if field == "base":
        target.write_bytes(b"synthetic")
    else:
        target.mkdir()
    observed = []
    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({field: str(target)}), stderr="")
    monkeypatch.setattr(preview_cutover.subprocess, "run", run)
    monkeypatch.setattr(preview_cutover, "_base_environment", lambda: {"SYNTHETIC_ONLY": "value"})
    native_python = tmp_path / "native-python"
    assert getattr(preview_cutover, helper)(SimpleNamespace(hermes_python=native_python)) == target
    argv, options = observed[0]
    assert argv[:1 + len(isolated_flags)] == [str(native_python), *isolated_flags]
    assert argv[1 + len(isolated_flags)] == "-c"
    assert options["env"] == {"SYNTHETIC_ONLY": "value"}
    assert options["stdin"] is subprocess.DEVNULL
    assert options["timeout"] == 15 and options["capture_output"]
    assert options["text"] is True and options["encoding"] == "utf-8"


@pytest.mark.parametrize("staged_env", [True, False])
def test_runtime_resolution_forwards_exact_selected_metadata_and_profile_environment(
    tmp_path, monkeypatch, staged_env,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    (s.plan.profile_source / ".env").write_text("SYNTHETIC_SOURCE=value\n")
    if staged_env:
        (s.home / ".env").write_text("SYNTHETIC_STAGED=value\n")
    preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    provider = s.config["providers"]["live"]
    assert s.calls == [{
        "executable": s.executable, "hermes_source": s.plan.hermes_source,
        "native_site_packages": s.packages, "home": s.home,
        "env_file": (s.home if staged_env else s.plan.profile_source) / ".env",
        "provider": "custom:live", "base_url": provider["base_url"],
        "model": provider["default_model"], "api_mode": provider["api_mode"],
        "command": provider["key_cmd"], "environment": s.environment,
    }]


@pytest.mark.parametrize("fault", ["wrong", "missing", "probe"])
def test_runtime_resolution_refuses_ambient_module_or_failed_probe(
    tmp_path, monkeypatch, capsys, fault,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    if fault == "probe":
        def fail(**kwargs):
            raise RuntimeError("SYNTHETIC_PRIVATE_VALUE")
        monkeypatch.setattr(s.module, "verify_native_credential_resolution", fail)
    else:
        monkeypatch.setattr(s.module, "__file__",
                            None if fault == "missing" else str(tmp_path / "ambient.py"))
    with pytest.raises(preview_cutover.PreviewRefusal, match="^native_credential_binding$"):
        preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert s.calls == []
    assert capsys.readouterr().err == ('[native-credential-refused] '
                                     '{"version":1,"category":"runtime_refused"}\n')


@pytest.mark.parametrize("mode", ["legacy", "static"])
def test_runtime_resolution_is_noop_for_unchanged_static_or_legacy_profiles(
    tmp_path, monkeypatch, mode,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    if mode == "legacy":
        s.plan.policy_mode = "legacy"
    else:
        presentation = s.config["plugins"]["entries"]["hermes-realtime"]["settings"]["presentation"]
        del presentation["key_cmd_argv"]
        presentation["api_key_env"] = "SYNTHETIC_KEY"
        (s.home / "config.yaml").write_text(json.dumps(s.config))
    preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert s.calls == []


def test_preflight_resolves_live_native_route_before_ready(tmp_path, monkeypatch):
    s = runtime_fixture(tmp_path, monkeypatch)
    s.plan.ports.update(companion=12346, livekit=12347, host=12348)
    for name in ("verify_realtime_runtime", "verify_profile", "verify_cpu_speech",
                 "verify_host_profile_mode", "verify_speech_assets",
                 "_verify_managed_kokoro_assets"):
        monkeypatch.setattr(preview_cutover, name, lambda *args: None)
    assert preview_cutover.preflight(s.plan, s.home)["category"] == "preflight_ready"
    assert len(s.calls) == 1


def test_native_base_probe_selects_native_not_foreground_python(tmp_path, monkeypatch):
    base = tmp_path / "base.exe"
    base.write_bytes(b"synthetic")
    native = tmp_path / "native.exe"
    foreground = tmp_path / "foreground.exe"
    observed = []
    def run(command, **options):
        observed.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"base": str(base)}), stderr="")
    monkeypatch.setattr(preview_cutover.subprocess, "run", run)
    plan = SimpleNamespace(hermes_python=native, executables={"host_python": foreground})
    assert preview_cutover._native_credential_python(plan) == base
    assert observed[0][0] == str(native)


def test_runtime_resolution_refuses_source_profile_even_at_planned_home(tmp_path, monkeypatch):
    s = runtime_fixture(tmp_path, monkeypatch)
    s.plan.profile_source = s.home
    with pytest.raises(preview_cutover.PreviewRefusal, match="^native_credential_binding$"):
        preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert s.calls == []


@pytest.mark.parametrize("probe_fails", [False, True])
def test_native_probe_removes_only_its_new_empty_directories_before_profile_copy(
    tmp_path, monkeypatch, probe_fails,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    def probe(**kwargs):
        for name in ("skills", "hooks"):
            (s.home / name).mkdir()
        if probe_fails:
            raise ValueError("synthetic_probe_refusal")
    monkeypatch.setattr(s.module, "verify_native_credential_resolution", probe)
    if probe_fails:
        with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
            preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    else:
        preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    for name in ("skills", "hooks"):
        assert not (s.home / name).exists()
        original = tmp_path / ("source-" + name)
        original.mkdir()
        (original / "synthetic.txt").write_text("preserve this synthetic file")
        assert preview_cutover._copy_tree(original, s.home / name)[0] == 1
        assert (s.home / name / "synthetic.txt").read_text() == "preserve this synthetic file"


def test_native_probe_preserves_preexisting_empty_directories_and_copy_collision_guard(
    tmp_path, monkeypatch,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    for name in ("skills", "hooks"):
        (s.home / name).mkdir()
    preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    for name in ("skills", "hooks"):
        assert (s.home / name).is_dir()
    source = tmp_path / "copy-source"
    source.mkdir()
    with pytest.raises(preview_cutover.PreviewRefusal, match="profile_copy_collision"):
        preview_cutover._copy_tree(source, s.home / "skills")


@pytest.mark.parametrize("kind", ["nonempty", "file", "reparse"])
def test_native_probe_preserves_unsafe_new_path_and_refuses_ready(
    tmp_path, monkeypatch, kind,
):
    s = runtime_fixture(tmp_path, monkeypatch)
    target = s.home / "skills"
    original_lstat = Path.lstat
    def lstat(path, *args, **kwargs):
        metadata = original_lstat(path, *args, **kwargs)
        if path == target and kind == "reparse":
            return SimpleNamespace(st_mode=metadata.st_mode, st_file_attributes=0x400)
        return metadata
    monkeypatch.setattr(Path, "lstat", lstat)
    def probe(**kwargs):
        if kind == "file":
            target.write_text("synthetic retained data")
        else:
            target.mkdir()
            if kind == "nonempty":
                (target / "data").write_text("synthetic retained data")
    monkeypatch.setattr(s.module, "verify_native_credential_resolution", probe)
    with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
        preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert target.exists()
    if kind == "nonempty":
        assert (target / "data").read_text() == "synthetic retained data"
    elif kind == "file":
        assert target.read_text() == "synthetic retained data"


def test_native_probe_cleanup_error_refuses_ready_without_echo(tmp_path, monkeypatch, capsys):
    s = runtime_fixture(tmp_path, monkeypatch)
    target = s.home / "skills"
    original_rmdir = Path.rmdir
    def rmdir(path):
        if path == target:
            raise PermissionError("SYNTHETIC_PRIVATE_PATH")
        return original_rmdir(path)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    monkeypatch.setattr(s.module, "verify_native_credential_resolution",
                        lambda **kwargs: target.mkdir())
    with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
        preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert target.is_dir()
    assert capsys.readouterr().err == ('[native-credential-refused] '
                                     '{"version":1,"category":"runtime_refused"}\n')


def test_native_probe_cleanup_rechecks_owned_parent_before_removal(tmp_path, monkeypatch):
    s = runtime_fixture(tmp_path, monkeypatch)
    target = s.home / "skills"
    original_resolve = Path.resolve
    changed = False
    def resolve(path, *args, **kwargs):
        if changed and path == s.home:
            return tmp_path / "outside"
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    def probe(**kwargs):
        nonlocal changed
        target.mkdir()
        changed = True
    monkeypatch.setattr(s.module, "verify_native_credential_resolution", probe)
    with pytest.raises(preview_cutover.PreviewRefusal, match="native_credential_binding"):
        preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert target.is_dir()


def test_native_probe_does_not_remove_other_new_directories(tmp_path, monkeypatch):
    s = runtime_fixture(tmp_path, monkeypatch)
    target = s.home / "keep"
    monkeypatch.setattr(s.module, "verify_native_credential_resolution",
                        lambda **kwargs: target.mkdir())
    preview_cutover._verify_native_credential_runtime(s.plan, s.home)
    assert target.is_dir()
