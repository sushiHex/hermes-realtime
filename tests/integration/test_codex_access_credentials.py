from __future__ import annotations

import base64
import hashlib
import json

import pytest

from hermes_realtime.integration import codex_access_credentials as credentials


def jwt(account="synthetic-account", expiry=2000):
    payload = {"exp": expiry, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return "header." + encoded + ".signature"


def fixture(tmp_path, token=None):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"tokens": {
        "access_token": token or jwt(), "refresh_token": "NEVER-EXPORT",
    }}))
    return path, hashlib.sha256(b"synthetic-account").hexdigest()


def test_live_access_read_observes_owner_rotation_without_copying_refresh_token(tmp_path):
    path, account = fixture(tmp_path)
    first_bytes = path.read_bytes()
    first = credentials.read_access_credential(path, account, now=1000)
    assert first == {"access_token": jwt(), "expires_in": 1}
    assert path.read_bytes() == first_bytes
    second = jwt(expiry=3000)
    path.write_text(json.dumps({"tokens": {
        "access_token": second, "refresh_token": "NEW-PRIVATE",
    }}))
    assert credentials.read_access_credential(path, account, now=1000)["access_token"] == second
    assert "refresh_token" not in first and "NEVER-EXPORT" not in json.dumps(first)


@pytest.mark.parametrize("token,category", [
    (jwt(account="different-account"), "account_mismatch"),
    (jwt(expiry=1000), "credential_expired"),
    (jwt(expiry=1060), "credential_expired"),
    (jwt(expiry=True), "credential_invalid"),
    (jwt(expiry=float("inf")), "credential_invalid"),
    (jwt(account="\ud800"), "credential_invalid"),
    ("not-a-jwt", "credential_invalid"),
])
def test_invalid_or_expiring_credentials_are_refused_before_output(tmp_path, token, category):
    path, account = fixture(tmp_path, token)
    with pytest.raises(credentials.AccessCredentialError, match="^" + category + "$"):
        credentials.read_access_credential(path, account, now=1000)


def test_oversized_and_duplicate_stores_are_refused(tmp_path):
    path, account = fixture(tmp_path)
    for data in (b"x" * 65537, b'{"tokens":{},"tokens":{}}'):
        path.write_bytes(data)
        with pytest.raises(credentials.AccessCredentialError, match="credential_invalid"):
            credentials.read_access_credential(path, account, now=1000)


def test_symbolic_link_source_is_refused(tmp_path):
    path, account = fixture(tmp_path)
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("symbolic links unavailable")
    with pytest.raises(credentials.AccessCredentialError, match="credential_unavailable"):
        credentials.read_access_credential(link, account, now=1000)


def test_cli_refusal_never_prints_credentials_or_private_path(tmp_path, capsys, monkeypatch):
    path, account = fixture(tmp_path, jwt(account="different-account"))
    monkeypatch.setattr(credentials.time, "time", lambda: 1000)
    assert credentials.main(["--auth-file", str(path), "--account-sha256", account]) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.split("] ", 1)[1]) == {
        "version": 1, "category": "account_mismatch",
    }
    assert str(path) not in captured.err and "NEVER-EXPORT" not in captured.err


def test_command_builder_is_fixed_argv_and_quotes_shell_arguments(tmp_path):
    argv = credentials.command_argv(tmp_path / "host python.exe", tmp_path / "helper.py",
                                    tmp_path / "auth file.json", "a" * 64)
    assert argv[1:4] == ["-I", "-B", "-S"] and argv[5] == "--auth-file"
    assert argv[7:] == ["--account-sha256", "a" * 64]
    assert credentials.shell_command(argv, windows=True) == " ".join(
        '"' + item + '"' for item in argv
    )
    import shlex
    assert shlex.split(credentials.shell_command(argv, windows=False)) == argv
    with pytest.raises(ValueError, match="credential_command"):
        credentials.shell_command(argv[:-1] + ["%UNTRUSTED%"], windows=True)


def binding(tmp_path):
    executable, helper = tmp_path / "python.exe", tmp_path / "credential.py"
    pin = hashlib.sha256(b"synthetic-account").hexdigest()
    argv = credentials.command_argv(executable, helper, tmp_path / "auth.json", pin)
    base = "https://chatgpt.com/backend-api/codex"
    config = {
        "model": {"provider": "custom:live", "default": "test-model"},
        "providers": {"live": {"base_url": base, "default_model": "test-model",
                                "api_mode": "codex_responses",
                                "key_cmd": credentials.shell_command(argv),
                                "extra_headers": {"ChatGPT-Account-ID": "synthetic-account"}}},
        "plugins": {"entries": {"hermes-realtime": {"settings": {"presentation": {
            "provider": "custom", "base_url": base, "model": "test-model",
            "api_mode": "codex_responses", "key_cmd_argv": argv,
        }}}}},
    }
    return config, executable, helper


def test_native_and_presenter_use_exact_same_operator_command(tmp_path):
    config, executable, helper = binding(tmp_path)
    credentials.validate_native_binding(config, executable, helper)


@pytest.mark.parametrize("fault", [
    "command", "static", "provider", "route", "helper", "model", "base_override", "alias",
    "alias_collision", "trailing_slash",
    "header_missing", "header_mismatch", "header_extra", "header_control",
])
def test_live_credential_binding_rejects_cross_role_or_source_drift(tmp_path, fault):
    config, executable, helper = binding(tmp_path)
    native = config["providers"]["live"]
    presentation = config["plugins"]["entries"]["hermes-realtime"]["settings"]["presentation"]
    if fault == "command":
        native["key_cmd"] += " other-command"
    elif fault == "static":
        config["model"]["api_key"] = "stale-secret"
    elif fault == "provider":
        config["model"]["provider"] = "custom:other"
    elif fault == "route":
        native["base_url"] = presentation["base_url"] = "https://other.invalid/v1"
    elif fault == "helper":
        presentation["key_cmd_argv"][4] = str(tmp_path / "other-helper.py")
        native["key_cmd"] = credentials.shell_command(presentation["key_cmd_argv"])
    elif fault == "model":
        native["default_model"] = "other-model"
    elif fault == "base_override":
        config["model"]["base_url"] = "https://other.invalid/v1"
    elif fault == "alias":
        native["name"] = "different-provider"
    elif fault == "alias_collision":
        config["providers"] = {"earlier": {"name": "live"}, **config["providers"]}
    elif fault == "trailing_slash":
        native["base_url"] = presentation["base_url"] = native["base_url"] + "/"
    elif fault == "header_missing":
        del native["extra_headers"]
    elif fault == "header_mismatch":
        native["extra_headers"]["ChatGPT-Account-ID"] = "different-account"
    elif fault == "header_extra":
        native["extra_headers"]["Authorization"] = "not-allowed"
    else:
        native["extra_headers"]["ChatGPT-Account-ID"] = "synthetic-account\r\n"
    with pytest.raises(ValueError, match="credential_binding"):
        credentials.validate_native_binding(config, executable, helper)


@pytest.mark.parametrize("fault", ["oversized", "duplicate_store", "duplicate_payload"])
def test_valid_credential_cannot_bypass_store_or_duplicate_bounds(tmp_path, fault):
    path, account = fixture(tmp_path)
    tokens = json.dumps({"access_token": jwt()})
    if fault == "oversized":
        raw = json.dumps({"tokens": {"access_token": jwt()}}).encode()
        path.write_bytes(raw + b" " * (65537 - len(raw)))
        assert path.stat().st_size == 65537
    elif fault == "duplicate_store":
        path.write_text('{"tokens":' + tokens + ',"tokens":' + tokens + '}')
    else:
        payload = ('{"exp":2000,"exp":2000,"https://api.openai.com/auth":'
                   '{"chatgpt_account_id":"synthetic-account"}}')
        encoded = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        path.write_text(json.dumps({"tokens": {
            "access_token": "header." + encoded + ".signature",
        }}))
    with pytest.raises(credentials.AccessCredentialError, match="^credential_invalid$"):
        credentials.read_access_credential(path, account, now=1000)


@pytest.mark.parametrize("fault", ["link", "reparse"])
def test_source_link_metadata_is_refused_without_real_link_privilege(tmp_path, monkeypatch, fault):
    from pathlib import Path
    from types import SimpleNamespace

    path, account = fixture(tmp_path)
    if fault == "link":
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    else:
        original = Path.lstat
        def lstat(self, *args, **kwargs):
            actual = original(self, *args, **kwargs)
            if self == path:
                return SimpleNamespace(st_mode=actual.st_mode, st_file_attributes=0x400)
            return actual
        monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(credentials.AccessCredentialError, match="^credential_unavailable$"):
        credentials.read_access_credential(path, account, now=1000)


@pytest.mark.parametrize("token,now,category", [
    (jwt(expiry=True), -1000, "credential_invalid"),
    (jwt(expiry=float("nan")), 1000, "credential_invalid"),
    (jwt(), True, "credential_expired"),
    (jwt(), float("nan"), "credential_expired"),
])
def test_claim_and_clock_types_are_independently_refused(tmp_path, token, now, category):
    path, account = fixture(tmp_path, token)
    with pytest.raises(credentials.AccessCredentialError, match="^" + category + "$"):
        credentials.read_access_credential(path, account, now=now)


@pytest.mark.parametrize(
    "account", ["synthetic account", "synthetic-account\r\n", "synthetic-é"],
    ids=["space", "control", "unicode"],
)
def test_account_header_value_policy_is_independent_of_account_pin(tmp_path, account):
    config, executable, helper = binding(tmp_path)
    native = config["providers"]["live"]
    presentation = config["plugins"]["entries"]["hermes-realtime"]["settings"]["presentation"]
    argv = presentation["key_cmd_argv"]
    argv[8] = hashlib.sha256(account.encode("utf-8")).hexdigest()
    native["extra_headers"]["ChatGPT-Account-ID"] = account
    native["key_cmd"] = credentials.shell_command(argv)
    with pytest.raises(ValueError, match="credential_binding"):
        credentials.validate_native_binding(config, executable, helper)


def test_guard_provider_name_must_be_canonical(tmp_path):
    config, executable, helper = binding(tmp_path)
    config["model"]["provider"] = "custom:Live"
    config["providers"] = {"Live": config["providers"]["live"]}
    with pytest.raises(ValueError, match="credential_binding"):
        credentials.validate_native_binding(config, executable, helper)
