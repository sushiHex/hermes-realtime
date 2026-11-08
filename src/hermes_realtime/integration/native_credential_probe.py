"""Positive native-route evidence without minting credentials.

The probe does not call the resolved credential source. The specified dotenv file
is read only; native config/pool housekeeping may write in the selected private
home. It observes that profile using native dotenv parsing/precedence.
Normal gateway environment loading and final startup checks remain separate.
Trusted native imports may perform standard platform metadata discovery; network
is blocked throughout and all subprocess/network I/O is blocked during resolution.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

NATIVE_PROBE = r'''
import contextlib, importlib, io, json, os, pathlib, sys
original_stdout = sys.stdout
facts = {"version": 1, "accepted": False, "selected_command": False, "route": False,
         "no_io": False, "homes": False, "source": False}
attempted_io = False
phase = "import"
class Discard(io.TextIOBase):
    def write(self, text): return len(text)
def audit(event, arguments):
    global attempted_io
    if (event in {"socket.connect", "socket.getaddrinfo"}
            or phase == "resolve" and event in {"subprocess.Popen", "os.system",
                 "os.posix_spawn", "os.fork", "os.forkpty", "os.spawn"}):
        attempted_io = True
        raise RuntimeError("probe_io_forbidden")
try:
    with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
        expected = json.loads(sys.stdin.buffer.read(65537))
        if set(expected) != {"hermes_source", "native_site_packages", "home", "env_file",
                             "provider", "base_url",
                             "model", "api_mode", "command"}:
            raise ValueError("probe_input")
        if not (type(expected["provider"]) is str
                and expected["provider"].startswith("custom:")
                and len(expected["provider"]) > len("custom:")):
            raise ValueError("probe_provider")
        source, home = (pathlib.Path(expected[key]).resolve()
                        for key in ("hermes_source", "home"))
        env_file = (pathlib.Path(expected["env_file"]).resolve()
                    if expected["env_file"] is not None else None)
        home_keys = ("HOME", "HERMES_HOME", "CODEX_HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
        def homes_match():
            real_home = os.environ.get("HERMES_REAL_HOME")
            return (all(os.environ.get(key) and pathlib.Path(os.environ[key]).resolve() == home
                        for key in home_keys)
                    and (not real_home or pathlib.Path(real_home).resolve() == home))
        if not homes_match() or (env_file is not None
                                and (not env_file.is_file() or env_file.is_symlink())):
            raise ValueError("probe_profile")
        packages = pathlib.Path(expected["native_site_packages"]).resolve()
        sys.path[:0] = [str(source), str(packages)]
        sys.addaudithook(audit)
        env_loader = importlib.import_module("hermes_cli.env_loader")
        if pathlib.Path(env_loader.__file__).resolve() != source / "hermes_cli/env_loader.py":
            raise ValueError("probe_source")
        # Match normal gateway ordering: dotenv before the provider resolver imports.
        if env_file is not None:
            env_loader._load_dotenv_with_fallback(env_file, override=True)
            env_loader._clear_known_keys_missing_from_dotenv(env_file)
        facts["homes"] = bool(homes_match())
        if not facts["homes"]:
            raise ValueError("probe_profile")
        command_module = importlib.import_module("agent.command_token_source")
        resolver_module = importlib.import_module("hermes_cli.runtime_provider")
        constants = importlib.import_module("hermes_constants")
        facts["source"] = all(
            pathlib.Path(module.__file__).resolve() == source / relative
            for module, relative in (
                (env_loader, "hermes_cli/env_loader.py"),
                (command_module, "agent/command_token_source.py"),
                (resolver_module, "hermes_cli/runtime_provider.py"),
                (constants, "hermes_constants.py"),
            )
        )
        if not facts["source"]:
            raise ValueError("probe_source")
        facts["homes"] = (facts["homes"]
                          and constants.get_hermes_home().resolve() == home
                          and constants.get_default_hermes_root().resolve() == home)
        if not facts["homes"]:
            raise ValueError("probe_profile")
        phase = "resolve"
        runtime = resolver_module.resolve_runtime_provider(
            requested=expected["provider"], target_model=expected["model"],
        )
        key = runtime.get("api_key")
        facts["selected_command"] = (
            type(key) is command_module.CommandTokenSource
            and key._command == expected["command"]
        )
        facts["route"] = (
            runtime.get("provider") == "custom"
            and runtime.get("requested_provider") == expected["provider"]
            and runtime.get("base_url") == expected["base_url"].rstrip("/")
            and runtime.get("model") == expected["model"]
            and runtime.get("api_mode") == expected["api_mode"] == "codex_responses"
        )
        facts["no_io"] = not attempted_io
        facts["accepted"] = all(facts[key] for key in (
            "selected_command", "route", "no_io", "homes", "source",
        ))
except BaseException:
    facts["no_io"] = not attempted_io
finally:
    original_stdout.write(json.dumps(facts, separators=(",", ":")) + "\n")
    original_stdout.flush()
raise SystemExit(0 if facts["accepted"] else 1)
'''

_EXPECTED = {"version": 1, "accepted": True, "selected_command": True, "route": True,
             "no_io": True, "homes": True, "source": True}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("native_credential_resolution")
    return result


def verify_native_credential_resolution(
    *, executable: Path, hermes_source: Path, native_site_packages: Path,
    home: Path, env_file: Path | None,
    provider: str, base_url: str, model: str, api_mode: str, command: str,
    environment: Mapping[str, str],
) -> None:
    accepted = False
    try:
        payload = {"hermes_source": str(hermes_source),
                   "native_site_packages": str(native_site_packages), "home": str(home),
                   "env_file": str(env_file) if env_file is not None else None,
                   "provider": provider, "base_url": base_url,
                   "model": model, "api_mode": api_mode, "command": command}
        completed = subprocess.run(
            [str(executable), "-I", "-B", "-S", "-c", NATIVE_PROBE], env=dict(environment),
            input=json.dumps(payload), capture_output=True, text=True, timeout=15,
        )
        if (completed.returncode != 0 or type(completed.stdout) is not str
                or len(completed.stdout) > 4096 or completed.stderr != ""):
            raise ValueError("native_credential_resolution")
        value = json.loads(completed.stdout, object_pairs_hook=_unique)
        if (type(value) is not dict or set(value) != set(_EXPECTED)
                or type(value["version"]) is not int or value["version"] != 1
                or any(type(value[key]) is not bool or value[key] is not True
                       for key in set(_EXPECTED) - {"version"})):
            raise ValueError("native_credential_resolution")
        accepted = True
    except Exception:
        raise ValueError("native_credential_resolution") from None
    finally:
        if not accepted:
            stream = sys.stderr
            if stream is not None:
                try:
                    print('[native-credential-probe] {"version":1,"category":"resolution_refused"}',
                          file=stream, flush=True)
                except Exception:
                    if sys.stderr is stream:
                        sys.stderr = None
