from types import SimpleNamespace

import pytest

from hermes_realtime.integration import native_web_evidence as native


def context(config, observed):
    return SimpleNamespace(
        get_config=lambda key, default=None: {"web_evidence": True,
                                             "presentation": config}.get(key, default),
        register_middleware=lambda *args: observed.append("middleware"),
        register_platform_handler=lambda *args: observed.append("platform"),
        on_unload=lambda *args: observed.append("unload"),
    )


def presentation(tmp_path, live=True):
    value = {"provider": "custom", "base_url": "https://chatgpt.com/backend-api/codex",
             "model": "synthetic-model", "api_mode": "codex_responses"}
    value.update({"key_cmd_argv": [str(tmp_path / "python.exe"), "helper"]}
                 if live else {"api_key_env": "SYNTHETIC_KEY"})
    return value


def test_live_registration_proves_native_source_before_advertising(tmp_path, monkeypatch):
    observed = []
    config = presentation(tmp_path)
    monkeypatch.setattr(native, "_verify_live_credential_runtime",
                        lambda value: observed.append(("verified", value)), raising=False)
    native.register_web_evidence(context(config, observed))
    assert observed[0] == ("verified", config)
    assert "platform" in observed


def test_unproven_live_source_cannot_advertise_web_evidence(tmp_path, monkeypatch):
    observed = []

    def refuse(value):
        raise RuntimeError("native_credential_binding")

    monkeypatch.setattr(native, "_verify_live_credential_runtime", refuse, raising=False)
    with pytest.raises(RuntimeError, match="native_credential_binding"):
        native.register_web_evidence(context(presentation(tmp_path), observed))
    assert observed == []


def test_static_registration_keeps_existing_behavior(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(native, "_verify_live_credential_runtime",
                        lambda value: pytest.fail("static path must not resolve credentials"),
                        raising=False)
    native.register_web_evidence(context(presentation(tmp_path, live=False), observed))
    assert observed == ["middleware", "middleware", "platform", "unload"]


@pytest.mark.parametrize("marker,category,expected", [
    ("native-credential-refused", "runtime_refused", "native_credential_runtime_refused"),
    ("presentation-credential-refused", "cleanup_unknown",
     "presentation_credential_cleanup_unknown"),
    ("native-credential-refused", "untrusted-value", "credential_failure_invalid"),
])
def test_credential_category_survives_raw_log_deletion(tmp_path, marker, category, expected):
    import json

    from scripts.lookup_backend_logs import CappedCapture

    path = tmp_path / "private.log"
    capture = CappedCapture(path)
    capture.feed(("[" + marker + "] " + json.dumps({"version": 1, "category": category})
                  + "\n").encode())
    result = capture.finish()
    path.unlink()
    assert set(result["categories"]) == {expected}


def test_credential_diagnostic_extra_fields_never_enter_retained_metadata(tmp_path):
    from scripts.lookup_backend_logs import CappedCapture

    capture = CappedCapture(tmp_path / "private.log")
    capture.feed(b'[native-credential-refused] {"version":1,"category":"runtime_refused",'
                 b'"secret":"synthetic-private"}\n')
    result = capture.finish()
    assert set(result["categories"]) == {"credential_failure_invalid"}
    assert "synthetic-private" not in str(result)


@pytest.mark.parametrize("fault", [None, "config", "origin", "probe"])
def test_live_startup_checks_real_loaded_environment_and_selected_source(
    tmp_path, monkeypatch, capsys, fault,
):
    import hashlib
    import sys
    from pathlib import Path

    from hermes_realtime.integration import codex_access_credentials as credentials

    helper = Path(native.__file__).with_name("codex_access_credentials.py")
    argv = credentials.command_argv(Path(sys._base_executable), helper,
                                    tmp_path / "auth.json",
                                    hashlib.sha256(b"synthetic-account").hexdigest())
    config = presentation(tmp_path)
    config["key_cmd_argv"] = argv
    provider = {"base_url": config["base_url"], "api_mode": "codex_responses",
                "default_model": config["model"], "key_cmd": credentials.shell_command(argv),
                "extra_headers": {"ChatGPT-Account-ID": "synthetic-account"}}
    loaded = {"model": {"provider": "custom:live", "default": config["model"]},
              "providers": {"live": provider}, "plugins": {"entries": {"hermes-realtime": {
                  "settings": {"presentation": dict(config)},
              }}}}
    if fault == "config":
        loaded["model"]["default"] = "different-model"
        loaded["providers"]["live"]["default_model"] = "different-model"
        loaded["plugins"]["entries"]["hermes-realtime"]["settings"]["presentation"][
            "model"] = "different-model"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SYNTHETIC_EXTERNAL_ENV", "present")
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        SimpleNamespace(load_config=lambda: loaded))
    monkeypatch.setitem(sys.modules, "hermes_cli", SimpleNamespace(
        __file__=None if fault == "origin" else str(tmp_path / "hermes_cli/__init__.py"),
    ))
    observed = []

    def verify(**kwargs):
        observed.append(kwargs)
        if fault == "probe":
            raise ValueError("synthetic-private-message")

    monkeypatch.setitem(sys.modules, "hermes_realtime.integration.native_credential_probe",
                        SimpleNamespace(verify_native_credential_resolution=verify))
    if fault is None:
        native._verify_live_credential_runtime(config)
        import sysconfig

        assert observed[0]["hermes_source"] == tmp_path
        assert observed[0]["executable"] == Path(sys._base_executable)
        assert observed[0]["native_site_packages"] == Path(sysconfig.get_path("purelib"))
        assert observed[0]["home"] == tmp_path
        assert observed[0]["env_file"] is None
        assert observed[0]["environment"]["SYNTHETIC_EXTERNAL_ENV"] == "present"
        assert observed[0]["command"] == provider["key_cmd"]
    else:
        with pytest.raises(RuntimeError, match="^native_credential_binding$"):
            native._verify_live_credential_runtime(config)
        captured = capsys.readouterr()
        assert "synthetic-private-message" not in captured.err
        assert '"category":"runtime_refused"' in captured.err
