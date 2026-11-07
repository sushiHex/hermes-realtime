"""Prepare and own one parameterized native voice-preview cutover."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.native_work_profile import StatusServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
_PREFIX = "[voice-preview] "
_STATUS_LIMIT = 8192
_JSON_LIMIT = 65536
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_PLUGIN = "ordinary-lookup-experimental"
_POLICY = "ordinary-lookup/v1"
_STOP_POLL_SECONDS = 0.5
_MAX_PROFILE_FILES = 4096
_MAX_PROFILE_BYTES = 64 * 1024**2
_MAX_SPEECH_RECEIPT_BYTES = 1024**2
_MAX_SPEECH_FILES = 128
_MAX_SPEECH_FILE_BYTES = 512 * 1024**2
_MAX_SPEECH_TOTAL_BYTES = 1024**3


class PreviewRefusal(ValueError):
    """A fixed operator category, distinct from arbitrary dependency exceptions."""


def require(condition: bool, category: str) -> None:
    if not condition:
        raise PreviewRefusal(category)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path, limit: int = _JSON_LIMIT) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "input_bound")
    return raw


def _json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "json_duplicate_field")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=pairs)


def _profile_config(raw: bytes) -> dict[str, Any]:
    """Parse the bounded Hermes profile config without weakening JSON checks."""
    try:
        value = _json(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            import yaml

            value = yaml.safe_load(raw.decode("utf-8"))
        except Exception:
            raise PreviewRefusal("profile_config_parse") from None
    require(type(value) is dict, "profile_config_parse")
    return value


def _validate_native_model_binding(config: dict[str, Any], plan: PreviewPlanV1) -> None:
    """Match the plan's native defaults; this plan forbids model-specific overrides.

    Hermes accepts provider-prefixed model aliases when resolving per-model
    reasoning overrides. To avoid duplicating that resolver here, native plans
    require an empty ``agent.reasoning_overrides`` map.
    """
    model_section = config.get("model")
    if type(model_section) is dict:
        model = model_section.get("default") or model_section.get("model")
    else:
        model = model_section
    agent = config.get("agent")
    require(type(model) is str and model == plan.models["background"], "native_model_binding")
    require(type(agent) is dict, "native_model_binding")
    effort = agent.get("reasoning_effort")
    overrides = agent.get("reasoning_overrides", {})
    require(type(overrides) is dict and not overrides, "native_model_binding")
    require(effort == plan.models["background_effort"], "native_model_binding")


@dataclass(frozen=True)
class PreviewPlanV1:
    output: Path
    realtime_source: Path
    realtime_revision: str
    realtime_tree: str
    wheel: Path
    wheel_sha256: str
    runtime_target: Path
    runtime_record_sha256: str
    git_executable: Path
    hermes_source: Path
    hermes_revision: str
    hermes_tree: str
    hermes_python: Path
    profile_source: Path
    native_status: StatusServer | None
    speech_assets: dict[str, Path]
    policy_mode: str
    ports: dict[str, Any]
    models: dict[str, str]
    speech: dict[str, Any]
    executables: dict[str, Path]
    host_options: dict[str, Any]
    readiness_policy: str = "kernel_listeners"


def load_plan(path: Path) -> PreviewPlanV1:
    value = _json(_read(path))
    require(
        type(value) is dict
        and set(value) - {"readiness_policy"}
        == {
            "version",
            "output",
            "realtime",
            "hermes",
            "profile_mode",
            "native_status",
            "speech_assets",
            "ports",
            "models",
            "speech",
            "executables",
            "host_options",
        },
        "plan_fields",
    )
    require(type(value["version"]) is int and value["version"] == 1, "plan_version")
    readiness_policy = value.get("readiness_policy", "kernel_listeners")
    require(type(readiness_policy) is str
            and readiness_policy in {"kernel_listeners", "dev_owned_services"},
            "readiness_policy")
    require(
        type(value["profile_mode"]) is str and value["profile_mode"] in {"lookup", "native"},
        "profile_mode",
    )

    def abs_path(raw: Any, label: str) -> Path:
        require(type(raw) is str and 0 < len(raw) <= 2048, label + "_path")
        path_value = Path(raw)
        require(path_value.is_absolute() and ".." not in path_value.parts, label + "_path")
        return path_value

    realtime = value["realtime"]
    hermes = value["hermes"]
    require(
        type(realtime) is dict
        and set(realtime)
        == {
            "source",
            "revision",
            "tree",
            "wheel",
            "wheel_sha256",
            "runtime_target",
            "record_sha256",
            "git_executable",
        },
        "realtime_fields",
    )
    require(
        type(hermes) is dict
        and set(hermes)
        == {
            "source",
            "revision",
            "tree",
            "python",
            "profile_source",
        },
        "hermes_fields",
    )
    native_status = value["native_status"]
    if value["profile_mode"] == "native":
        from scripts.native_work_profile import StatusServer

        require(
            type(native_status) is dict
            and set(native_status) == {"command", "sha256", "state_dir", "python_user_base"},
            "native_status_fields",
        )
        require(
            type(native_status["sha256"]) is str and _HEX64.fullmatch(native_status["sha256"]),
            "native_status_digest",
        )
        require(
            native_status["python_user_base"] is None
            or type(native_status["python_user_base"]) is str,
            "native_status_python_base",
        )
        native_status_value = StatusServer(
            command=abs_path(native_status["command"], "native_status_command"),
            sha256=native_status["sha256"],
            state_dir=abs_path(native_status["state_dir"], "native_status_state_dir"),
            python_user_base=(
                abs_path(native_status["python_user_base"], "native_status_python_base")
                if native_status["python_user_base"] is not None
                else None
            ),
        )
    else:
        require(native_status is None, "lookup_native_status_unexpected")
        native_status_value = None
    speech_assets = value["speech_assets"]
    require(
        type(speech_assets) is dict
        and set(speech_assets) == {"receipt", "moonshine_cache", "kokoro_cache"},
        "speech_assets_fields",
    )
    speech_assets_value = {
        key: abs_path(raw, "speech_asset_" + key) for key, raw in speech_assets.items()
    }
    for obj, fields in ((realtime, ("revision", "tree")), (hermes, ("revision", "tree"))):
        require(
            all(type(obj[name]) is str and _HEX40.fullmatch(obj[name]) for name in fields),
            "source_oid",
        )
    require(
        type(realtime["wheel_sha256"]) is str
        and _HEX64.fullmatch(realtime["wheel_sha256"])
        and type(realtime["record_sha256"]) is str
        and _HEX64.fullmatch(realtime["record_sha256"]),
        "wheel_digest",
    )
    ports = value["ports"]
    require(
        type(ports) is dict
        and set(ports)
        == {
            "api",
            "companion",
            "livekit",
            "host",
            "protected",
        },
        "ports_fields",
    )
    owned = [ports[name] for name in ("api", "companion", "livekit", "host")]
    protected = ports["protected"]
    require(
        all(type(port) is int and 1024 <= port <= 65535 for port in owned)
        and len(set(owned)) == len(owned)
        and type(protected) is list
        and len(protected) <= 8
        and all(type(port) is int and 1024 <= port <= 65535 for port in protected)
        and len(set(protected)) == len(protected)
        and not set(owned).intersection(protected),
        "ports_values",
    )
    models = value["models"]
    require(
        type(models) is dict
        and set(models)
        == {
            "foreground",
            "foreground_effort",
            "background",
            "background_effort",
        }
        and all(type(item) is str and _LABEL.fullmatch(item) for item in models.values()),
        "models_values",
    )
    speech = value["speech"]
    require(
        type(speech) is dict
        and set(speech)
        == {
            "stt_provider",
            "stt_tier",
            "tts_provider",
            "tts_voice",
            "tts_threads",
            "tts_chunk_chars",
            "moonshine_version",
            "kokoro_version",
            "onnxruntime_version",
        },
        "speech_fields",
    )
    require(
        speech["stt_provider"] == "moonshine"
        and type(speech["stt_tier"]) is str
        and speech["stt_tier"] in {"tiny", "base", "small", "medium"}
        and speech["tts_provider"] == "kokoro"
        and type(speech["tts_voice"]) is str
        and _LABEL.fullmatch(speech["tts_voice"])
        and type(speech["tts_threads"]) is int
        and 1 <= speech["tts_threads"] <= 32
        and type(speech["tts_chunk_chars"]) is int
        and 32 <= speech["tts_chunk_chars"] <= 4096
        and all(
            type(speech[name]) is str and _LABEL.fullmatch(speech[name])
            for name in ("moonshine_version", "kokoro_version", "onnxruntime_version")
        ),
        "speech_values",
    )
    executables = value["executables"]
    require(
        type(executables) is dict
        and set(executables)
        == {
            "codex",
            "livekit",
            "host_python",
        },
        "executable_fields",
    )
    require(all(type(raw) is str for raw in executables.values()), "executable_path")
    host_options = value["host_options"]
    mcp_fields = {"kokoro_mcp_config", "kokoro_mcp_config_sha256"}
    require(
        type(host_options) is dict
        and set(host_options) - mcp_fields == {
            "allow_unsandboxed_tasks",
            "natural_work_tools",
            "task_start_policy",
            "conversation_profile",
        },
        "host_options_fields",
    )
    require(
        type(host_options["allow_unsandboxed_tasks"]) is bool
        and type(host_options["natural_work_tools"]) is bool
        and type(host_options["task_start_policy"]) is str
        and host_options["task_start_policy"] in {"confirm", "automatic"}
        and type(host_options["conversation_profile"]) is str
        and _LABEL.fullmatch(host_options["conversation_profile"]),
        "host_options_values",
    )
    _kokoro_mcp_config(host_options)
    return PreviewPlanV1(
        output=abs_path(value["output"], "output"),
        realtime_source=abs_path(realtime["source"], "realtime_source"),
        realtime_revision=realtime["revision"],
        realtime_tree=realtime["tree"],
        wheel=abs_path(realtime["wheel"], "wheel"),
        wheel_sha256=realtime["wheel_sha256"],
        runtime_target=abs_path(realtime["runtime_target"], "runtime_target"),
        runtime_record_sha256=realtime["record_sha256"],
        git_executable=abs_path(realtime["git_executable"], "git_executable"),
        hermes_source=abs_path(hermes["source"], "hermes_source"),
        hermes_revision=hermes["revision"],
        hermes_tree=hermes["tree"],
        hermes_python=abs_path(hermes["python"], "hermes_python"),
        profile_source=abs_path(hermes["profile_source"], "profile_source"),
        native_status=native_status_value,
        speech_assets=speech_assets_value,
        policy_mode=value["profile_mode"],
        ports=dict(ports),
        models=dict(models),
        speech=dict(speech),
        executables={key: abs_path(raw, key) for key, raw in executables.items()},
        host_options=dict(host_options),
        readiness_policy=readiness_policy,
    )


def verify_realtime_runtime(plan: PreviewPlanV1) -> dict[str, Any]:
    """Bind current source, wheel and installed payload through existing verifiers."""
    require(
        plan.realtime_source.resolve(strict=True) == ROOT.resolve(strict=True),
        "realtime_source_root",
    )
    from scripts import setup_lookup_backend as setup
    from scripts.real_gate_support import installed_hermes_identity
    from scripts.task13_artifact_orchestrator import verify_clean_tracked_candidate

    identity = verify_clean_tracked_candidate(plan.realtime_source)
    require(
        identity.candidate_head_oid == plan.realtime_revision
        and identity.candidate_tree_oid == plan.realtime_tree,
        "realtime_source_binding",
    )
    # DEV preview needs exact source/wheel/installed bytes, not the release
    # archive oracle's per-object Windows child traversal. CI owns that gate.
    measured = setup.verify_runtime(
        {
            "candidate": plan.realtime_revision,
            "tree": plan.realtime_tree,
            "hermes_source": plan.hermes_source,
            "wheel": plan.wheel,
            "wheel_sha256": plan.wheel_sha256,
            "runtime_target": plan.runtime_target,
        },
        source_root=plan.realtime_source,
        hermes_revision=plan.hermes_revision,
    )
    require(measured["recordSha256"] == plan.runtime_record_sha256, "runtime_record_binding")
    after = verify_clean_tracked_candidate(plan.realtime_source)
    require(
        after.candidate_head_oid == identity.candidate_head_oid
        and after.candidate_tree_oid == identity.candidate_tree_oid,
        "realtime_source_changed",
    )
    # The plan carries no upstream version; only the verified commit is used.
    # Reuse the all-tracked verifier's exact blob proof for Windows case aliases.
    hermes_identity = installed_hermes_identity("", plan.hermes_source)
    require(
        hermes_identity["commit"] == plan.hermes_revision
        and setup.git(plan.hermes_source, "rev-parse", "HEAD^{tree}") == plan.hermes_tree,
        "hermes_source_binding",
    )
    require(
        plan.hermes_python.is_file()
        and not plan.hermes_python.is_symlink()
        and plan.executables["host_python"].is_file()
        and plan.executables["codex"].is_file()
        and plan.executables["livekit"].is_file(),
        "preview_executable_missing",
    )
    return {
        "candidate": identity.candidate_head_oid,
        "tree": identity.candidate_tree_oid,
        "wheel_sha256": plan.wheel_sha256,
        "record_sha256": plan.runtime_record_sha256,
        "package_files": measured["packageFiles"],
    }


def _safe_profile_inventory(
    root: Path,
    *,
    max_files: int = _MAX_PROFILE_FILES,
    max_bytes: int = _MAX_PROFILE_BYTES,
) -> list[Path]:
    require(root.is_dir() and not root.is_symlink(), "profile_directory")
    entries = [root, *root.rglob("*")]
    require(
        not any(
            entry.is_symlink() or getattr(entry.lstat(), "st_file_attributes", 0) & 0x400
            for entry in entries
        ),
        "profile_link",
    )
    require(all(entry.is_file() or entry.is_dir() for entry in entries[1:]), "profile_entry_type")
    files = [entry for entry in entries if entry.is_file()]
    require(
        len(files) <= max_files and sum(entry.stat().st_size for entry in files) <= max_bytes,
        "profile_bound",
    )
    return files


def verify_profile(plan: PreviewPlanV1, home: Path) -> dict[str, Any]:
    files = _safe_profile_inventory(
        home,
        max_files=_MAX_PROFILE_FILES + _MAX_SPEECH_FILES,
        max_bytes=_MAX_PROFILE_BYTES + _MAX_SPEECH_TOTAL_BYTES,
    )
    config_path = home / "config.yaml"
    require(config_path.is_file() and config_path.stat().st_size <= _JSON_LIMIT, "profile_config")
    raw = _read(config_path, _JSON_LIMIT)
    config = _profile_config(raw)
    if plan.policy_mode == "lookup":
        from scripts.lookup_backend_gateway import require_interactive_policy_profile

        require_interactive_policy_profile(config, home)
    else:
        from scripts.native_work_profile import validate_native_work_profile

        require(plan.native_status is not None, "native_status_missing")
        _validate_native_model_binding(config, plan)
        validate_native_work_profile(config, plan.native_status)
    return {
        "profile_mode": plan.policy_mode,
        "config_sha256": digest(config_path),
        "file_count": len(files),
    }


def _kokoro_mcp_config(options: dict[str, Any]) -> Path | None:
    fields = {"kokoro_mcp_config", "kokoro_mcp_config_sha256"}
    if not fields.intersection(options):
        return None
    accepted = False
    try:
        require(fields.issubset(options), "kokoro_mcp_config_fields")
        raw, expected = options["kokoro_mcp_config"], options["kokoro_mcp_config_sha256"]
        require(type(raw) is str and 0 < len(raw) <= 2048, "kokoro_mcp_config_path")
        config = Path(raw)
        require(config.is_absolute() and ".." not in config.parts,
                "kokoro_mcp_config_path")
        require(type(expected) is str and _HEX64.fullmatch(expected) is not None,
                "kokoro_mcp_config_digest")
        require(config.is_file() and not config.is_symlink()
                and config.stat().st_size <= _JSON_LIMIT
                and digest(config) == expected, "kokoro_mcp_config_binding")
        accepted = True
        return config
    except OSError:
        raise PreviewRefusal("kokoro_mcp_config_binding") from None
    finally:
        if not accepted:
            print('[kokoro-mcp-preview-refused] {"category":"configuration_refused",'
                  '"version":1}', file=sys.stderr, flush=True)


def _bound_kokoro_mcp_config(plan: PreviewPlanV1) -> Path | None:
    return _kokoro_mcp_config(plan.host_options)


def _verify_managed_kokoro_assets(plan: PreviewPlanV1, home: Path) -> None:
    """Exercise the existing factory binding without starting a worker or MCP scope."""
    config = _bound_kokoro_mcp_config(plan)
    if config is None:
        return
    accepted = False
    try:
        model, voices = _kokoro_asset_pins(plan)
        prefix = Path(".cache") / "hermes-realtime" / "kokoro" / model.sha256
        expected = [
            (prefix / model.filename, model.sha256, model.size),
            (prefix / voices.filename, voices.sha256, voices.size),
        ]
        planned = [
            (relative, sha, size)
            for _, relative, sha, size in _speech_asset_manifest(plan)
            if relative.is_relative_to(Path(".cache") / "hermes-realtime" / "kokoro")
        ]
        require(sorted(planned) == sorted(expected), "kokoro_mcp_asset_binding")
        _runtime_modules(plan)
        managed = importlib.import_module("hermes_realtime.providers.kokoro_mcp")
        expected_module = plan.runtime_target / "hermes_realtime" / "providers" / "kokoro_mcp.py"
        require(managed.__file__ is not None
                and Path(managed.__file__).resolve() == expected_module.resolve(),
                "kokoro_mcp_asset_binding")
        factory = managed.KokoroMcpWorkerFactory.from_config(config)
        try:
            factory.create(
                model_path=(home / prefix / model.filename).resolve(),
                voices_path=(home / prefix / voices.filename).resolve(),
                expected_model_sha256=model.sha256,
                expected_voices_sha256=voices.sha256,
                language="en-gb", max_pcm_bytes=16 * 1024 * 1024,
                attestation_voice=plan.speech["tts_voice"],
            )
        finally:
            factory.close()
        accepted = True
    except Exception:
        raise PreviewRefusal("kokoro_mcp_asset_binding") from None
    finally:
        if not accepted:
            stream = sys.stderr
            if stream is not None:
                try:
                    print('[kokoro-mcp-preview-refused] {"category":"asset_binding_refused",'
                          '"version":1}', file=stream, flush=True)
                except Exception:
                    if sys.stderr is stream:
                        sys.stderr = None


def verify_cpu_speech(plan: PreviewPlanV1) -> None:
    code = (
        "import importlib.metadata as m,json,sys; "
        "expected=json.loads(sys.argv[1]); "
        "actual={'moonshine_version':m.version('moonshine-voice'),"
        "'kokoro_version':m.version('kokoro-onnx'),"
        "'onnxruntime_version':m.version('onnxruntime')}; "
        "gpu=False; "
        "\ntry: m.distribution('onnxruntime-gpu'); gpu=True\n"
        "except m.PackageNotFoundError: pass\n"
        "print(json.dumps({'actual':actual,'gpu':gpu},sort_keys=True))"
    )
    expected = {
        name: plan.speech[name]
        for name in (
            "moonshine_version",
            "kokoro_version",
            "onnxruntime_version",
        )
    }
    completed = subprocess.run(
        [
            str(plan.executables["host_python"]),
            "-I",
            "-B",
            "-c",
            code,
            json.dumps(expected, sort_keys=True),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    require(
        completed.returncode == 0 and len(completed.stdout) <= 2048 and completed.stderr == "",
        "cpu_speech_probe",
    )
    facts = _json(completed.stdout.encode("utf-8"))
    require(
        type(facts) is dict
        and set(facts) == {"actual", "gpu"}
        and facts["actual"] == expected
        and facts["gpu"] is False,
        "cpu_speech_binding",
    )


def _kokoro_asset_pins(plan: PreviewPlanV1) -> tuple[Any, Any]:
    _runtime_modules(plan)
    kokoro = importlib.import_module("hermes_realtime.providers.kokoro")
    expected_module = plan.runtime_target / "hermes_realtime" / "providers" / "kokoro.py"
    require(Path(kokoro.__file__).resolve() == expected_module.resolve(), "kokoro_import_origin")
    return kokoro._MODEL_ASSET, kokoro._VOICES_ASSET


def _speech_asset_manifest(plan: PreviewPlanV1) -> list[tuple[Path, Path, str, int]]:
    model_pin, voice_pin = _kokoro_asset_pins(plan)

    receipt_path = plan.speech_assets["receipt"]
    require(
        receipt_path.is_file()
        and not receipt_path.is_symlink()
        and receipt_path.stat().st_size <= _MAX_SPEECH_RECEIPT_BYTES,
        "speech_cache_receipt_missing",
    )
    receipt = _json(_read(receipt_path, _MAX_SPEECH_RECEIPT_BYTES))
    require(type(receipt) is dict, "speech_cache_receipt_shape")
    files = receipt.get("files")
    sizes = receipt.get("file_sizes")
    require(
        type(files) is dict
        and 0 < len(files) <= _MAX_SPEECH_FILES
        and type(sizes) is dict
        and set(files) == set(sizes),
        "speech_cache_receipt_files",
    )
    moonshine_root = plan.speech_assets["moonshine_cache"]
    kokoro_root = plan.speech_assets["kokoro_cache"]
    require(
        moonshine_root.is_dir()
        and not moonshine_root.is_symlink()
        and kokoro_root.is_dir()
        and not kokoro_root.is_symlink()
        and kokoro_root.name == model_pin.sha256,
        "speech_cache_roots",
    )
    moonshine_root = moonshine_root.resolve(strict=True)
    kokoro_root = kokoro_root.resolve(strict=True)
    require(
        receipt.get("moonshine_cache_root") == str(moonshine_root)
        and receipt.get("kokoro_cache_root") == str(kokoro_root),
        "speech_cache_root_binding",
    )
    require(
        type(receipt.get("version")) is int
        and receipt.get("version") == 1
        and receipt.get("moonshine_tier") == plan.speech["stt_tier"]
        and receipt.get("moonshine_package_version") == plan.speech["moonshine_version"]
        and receipt.get("kokoro_model_sha256") == model_pin.sha256
        and receipt.get("kokoro_model_bytes") == model_pin.size
        and receipt.get("kokoro_voice") == plan.speech["tts_voice"]
        and receipt.get("kokoro_voice_sha256") == voice_pin.sha256
        and receipt.get("kokoro_voice_bytes") == voice_pin.size,
        "speech_asset_binding",
    )

    cache_roots = (moonshine_root, kokoro_root)
    for root in cache_roots:
        require(
            not root.is_symlink() and not (getattr(root.lstat(), "st_file_attributes", 0) & 0x400),
            "speech_asset_link",
        )

    result: list[tuple[Path, Path, str, int]] = []
    found_kokoro: set[str] = set()
    total = 0
    for raw_path, expected_sha in files.items():
        require(
            type(raw_path) is str
            and 0 < len(raw_path) <= 2048
            and type(expected_sha) is str
            and _HEX64.fullmatch(expected_sha) is not None,
            "speech_asset_entry",
        )
        source = Path(raw_path)
        require(
            source.is_absolute() and source.is_file() and not source.is_symlink(),
            "speech_asset_missing",
        )
        source = source.resolve(strict=True)
        roots = [root for root in cache_roots if source.is_relative_to(root)]
        require(len(roots) == 1, "speech_asset_outside_cache")
        root = roots[0]
        size = sizes[raw_path]
        require(
            type(size) is int
            and 0 < size <= _MAX_SPEECH_FILE_BYTES
            and source.stat().st_size == size,
            "speech_asset_size",
        )
        total += size
        require(total <= _MAX_SPEECH_TOTAL_BYTES, "speech_assets_total_bound")
        require(digest(source) == expected_sha, "speech_cache_integrity")
        relative = source.relative_to(root)
        if root == moonshine_root:
            destination = Path(".cache") / "moonshine" / relative
        else:
            destination = (
                Path(".cache") / "hermes-realtime" / "kokoro" / model_pin.sha256 / relative
            )
            if source.name == model_pin.filename:
                require(
                    (expected_sha, size) == (model_pin.sha256, model_pin.size),
                    "kokoro_model_asset",
                )
                found_kokoro.add("model")
            elif source.name == voice_pin.filename:
                require(
                    (expected_sha, size) == (voice_pin.sha256, voice_pin.size),
                    "kokoro_voice_asset",
                )
                found_kokoro.add("voice")
            else:
                raise ValueError("kokoro_asset_unexpected")
        result.append((source, destination, expected_sha, size))
    # The receipt is the admitted catalog. Unrelated shared-cache files are
    # never read or staged; verify_speech_assets requires exact staged equality.
    require(found_kokoro == {"model", "voice"}, "speech_asset_inventory")
    return result


def verify_speech_assets(plan: PreviewPlanV1, home: Path | None = None) -> dict[str, int | str]:
    assets = _speech_asset_manifest(plan)
    total = 0
    expected_staged: set[Path] = set()
    for _, relative, expected_sha, size in assets:
        total += size
        if home is not None:
            staged = home / relative
            expected_staged.add(staged.resolve())
            require(
                staged.is_file()
                and not staged.is_symlink()
                and staged.stat().st_size == size
                and digest(staged) == expected_sha,
                "speech_asset_stage_integrity",
            )
    if home is not None:
        roots = (
            home / ".cache" / "moonshine",
            home / ".cache" / "hermes-realtime" / "kokoro",
        )
        actual_staged: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            require(root.is_dir() and not root.is_symlink(), "speech_asset_stage_root")
            entries = [root, *root.rglob("*")]
            require(
                all(
                    not entry.is_symlink()
                    and not (getattr(entry.lstat(), "st_file_attributes", 0) & 0x400)
                    for entry in entries
                ),
                "speech_asset_stage_link",
            )
            actual_staged.update(entry.resolve() for entry in entries if entry.is_file())
        require(actual_staged == expected_staged, "speech_asset_stage_inventory")
    return {
        "file_count": len(assets),
        "bytes": total,
        "receipt_sha256": digest(plan.speech_assets["receipt"]),
    }


def native_web_evidence_contract(config: dict[str, Any]) -> str | None:
    from scripts.native_work_profile import web_evidence_selected

    return "hermes.web-evidence/v1" if web_evidence_selected(config) else None


def verify_host_profile_mode(plan: PreviewPlanV1, home: Path | None = None) -> None:
    import importlib
    import inspect

    runtime_target = plan.runtime_target.resolve(strict=True)
    sys.path.insert(0, str(runtime_target))
    try:
        module = importlib.import_module("hermes_realtime.host_launcher")
        expected = plan.runtime_target / "hermes_realtime" / "host_launcher.py"
        require(Path(module.__file__).resolve() == expected.resolve(), "host_launcher_origin")
        parameters = inspect.signature(module.build_local_host_launcher).parameters
        if plan.policy_mode == "native":
            require("hermes_discover_tools" in parameters, "native_discovery_unavailable")
            config = _profile_config(
                _read((home or plan.profile_source) / "config.yaml", _JSON_LIMIT))
            if native_web_evidence_contract(config) is not None:
                require(
                    "hermes_web_evidence_contract" in parameters,
                    "native_web_evidence_unavailable",
                )
                presentation = importlib.import_module(
                    "hermes_realtime.integration.presentation_transport"
                )
                expected_presentation = (
                    plan.runtime_target / "hermes_realtime/integration/presentation_transport.py"
                )
                require(
                    Path(presentation.__file__).resolve() == expected_presentation.resolve(),
                    "native_web_presentation_origin",
                )
                settings = config["plugins"]["entries"]["hermes-realtime"]["settings"]
                try:
                    presentation.Presentation(settings["presentation"])
                except (TypeError, ValueError):
                    raise PreviewRefusal("native_web_presentation_config") from None
        else:
            require("hermes_lookup_policy" in parameters, "lookup_policy_unavailable")
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(runtime_target))


def preflight(plan: PreviewPlanV1, home: Path | None = None) -> dict[str, Any]:
    verify_realtime_runtime(plan)
    verify_profile(plan, home or plan.profile_source)
    verify_cpu_speech(plan)
    verify_host_profile_mode(plan, home or plan.profile_source)
    verify_speech_assets(plan, home or plan.profile_source)
    _verify_managed_kokoro_assets(plan, home or plan.profile_source)
    return {
        "category": "preflight_ready",
        "launch_authorized": False,
        "ports": [plan.ports[name] for name in ("api", "companion", "livekit", "host")],
    }


def parse_preflight_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Accept only one bounded supervisor status line and its matching exit code."""
    if type(completed) is not subprocess.CompletedProcess:
        raise ValueError("preflight_result_type")
    stdout = completed.stdout
    if type(stdout) is not str or not 0 < len(stdout.encode("utf-8")) <= _STATUS_LIMIT:
        raise ValueError("preflight_output_bound")
    if not stdout.endswith("\n") or stdout.count("\n") != 1 or not stdout.startswith(_PREFIX):
        raise ValueError("preflight_output_shape")
    raw = stdout[len(_PREFIX) : -1]

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("preflight_duplicate_field")
            value[key] = item
        return value

    try:
        status = json.loads(raw, object_pairs_hook=unique)
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ValueError) and error.args == ("preflight_duplicate_field",):
            raise
        raise ValueError("preflight_json_invalid") from None
    if type(status) is not dict or type(status.get("category")) is not str:
        raise ValueError("preflight_status_schema")
    if status["category"] == "preflight_ready":
        if set(status) != {"category", "launch_authorized", "ports"}:
            raise ValueError("preflight_ready_schema")
        ports = status.get("ports")
        if (
            type(status.get("launch_authorized")) is not bool
            or status["launch_authorized"]
            or type(ports) is not list
            or len(ports) > 8
            or any(type(port) is not int or not 1024 <= port <= 65535 for port in ports)
            or len(set(ports)) != len(ports)
        ):
            raise ValueError("preflight_ready_schema")
        if type(completed.returncode) is not int or completed.returncode != 0:
            raise ValueError("preflight_exit_mismatch")
    elif status["category"] == "refused":
        if set(status) != {"category", "failure_kind", "failure_category"}:
            raise ValueError("preflight_refused_schema")
        if (
            type(status["failure_kind"]) is not str
            or status["failure_kind"]
            not in {
                "RuntimeError",
                "ValueError",
                "OSError",
                "TimeoutError",
                "TypeError",
                "ImportError",
                "PermissionError",
                "other",
            }
            or type(status["failure_category"]) is not str
            or not _LABEL.fullmatch(status["failure_category"])
            or type(completed.returncode) is not int
            or completed.returncode == 0
        ):
            raise ValueError("preflight_exit_mismatch")
    else:
        raise ValueError("preflight_status_schema")
    return status


def record_prepared_after_preflight(
    plan: dict[str, Any], completed: subprocess.CompletedProcess[str], destination: Path
) -> dict[str, Any]:
    """Write a completion receipt only after a real successful supervisor preflight."""
    status = parse_preflight_result(completed)
    if status["category"] != "preflight_ready":
        raise ValueError("preflight_refused")
    if type(plan) is not dict or plan.get("version") != 1:
        raise ValueError("plan_schema")
    receipt = dict(plan)
    receipt["preflight"] = "accepted"
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("prepared_receipt_bound")
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return receipt


def emit(status: Mapping[str, Any]) -> None:
    print(_PREFIX + json.dumps(dict(status), sort_keys=True, separators=(",", ":")), flush=True)


def _copy_bounded_file(source: Path, destination: Path, maximum: int) -> str:
    require(
        source.is_file()
        and not source.is_symlink()
        and not (getattr(source.lstat(), "st_file_attributes", 0) & 0x400)
        and source.stat().st_size <= maximum,
        "profile_file_refused",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        copied = 0
        hasher = hashlib.sha256()
        while True:
            block = reader.read(min(65536, maximum + 1 - copied))
            if not block:
                break
            copied += len(block)
            require(copied <= maximum, "profile_file_bound")
            hasher.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    require(copied == source.stat().st_size, "profile_copy_size")
    return hasher.hexdigest()


def stage_preflight_profile(plan: PreviewPlanV1) -> Path:
    """Stage the config and verified speech assets; private profile data waits for handoff."""
    profile = plan.output / "input" / "home"
    require(
        plan.output.is_dir() and not plan.output.is_symlink() and not any(plan.output.iterdir()),
        "output_not_fresh",
    )
    profile.mkdir(parents=True)
    source_config = plan.profile_source / "config.yaml"
    if plan.policy_mode == "native":
        from scripts.native_work_profile import build_native_work_profile

        require(plan.native_status is not None, "native_status_missing")
        base_config = _profile_config(_read(source_config, _JSON_LIMIT))
        _validate_native_model_binding(base_config, plan)
        staged_config = build_native_work_profile(base_config, plan.native_status)
        (profile / "config.yaml").write_text(
            json.dumps(staged_config, sort_keys=True), encoding="utf-8"
        )
    else:
        _copy_bounded_file(source_config, profile / "config.yaml", _JSON_LIMIT)
    if plan.policy_mode == "lookup":
        from scripts.stage_lookup_policy import stage_lookup_policy

        stub = plan.output / "policy-stub"
        stub.mkdir()
        result = stage_lookup_policy(
            stub, ROOT / "tests" / "fixtures" / "plugin_generation_candidate"
        )
        require(result.get("policy") == _POLICY, "lookup_policy_revision")
        source_plugin = stub / "plugins" / _PLUGIN
        target_plugin = profile / "plugins" / _PLUGIN
        target_plugin.parent.mkdir()
        shutil.copytree(source_plugin, target_plugin)
        expected = result.get("files_sha256")
        require(
            type(expected) is dict and set(expected) == {p.name for p in target_plugin.iterdir()},
            "lookup_policy_shape",
        )
        for name, expected_sha in expected.items():
            require(digest(target_plugin / name) == expected_sha, "lookup_policy_copy")
    for source, relative, expected_sha, size in _speech_asset_manifest(plan):
        target = profile / relative
        _copy_bounded_file(source, target, min(size, _MAX_SPEECH_FILE_BYTES))
        require(digest(target) == expected_sha, "speech_asset_stage_copy")
    verify_speech_assets(plan, profile)
    return profile


def run_preflight(plan_path: Path, home: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "preflight",
        "--plan",
        str(plan_path),
        "--home",
        str(home),
    ]
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def prepare(plan_path: Path) -> dict[str, Any]:
    raw_plan = _json(_read(plan_path))
    plan = load_plan(plan_path)
    require(not plan.output.exists() and not plan.output.is_symlink(), "output_already_exists")
    plan.output.mkdir(parents=False)
    try:
        profile = stage_preflight_profile(plan)
        staged_plan_path = plan.output / "plan.private.json"
        staged_plan_path.write_bytes(_read(plan_path))
        completed = run_preflight(staged_plan_path, profile)
        status = parse_preflight_result(completed)
    except ValueError as error:
        status = _failure_status(error) if type(error) is PreviewRefusal else {
            "category": "refused",
            "failure_kind": "RuntimeError",
            "failure_category": (
                "preflight_output_invalid"
                if error.args == ("preflight_output_shape",)
                else "preparation_failed"
            ),
        }
    except (OSError, subprocess.SubprocessError):
        status = {
            "category": "refused",
            "failure_kind": "OSError",
            "failure_category": "preparation_failed",
        }
    if status["category"] != "preflight_ready":
        refusal = {
            "version": 1,
            "category": "preparation_refused",
            "failure_kind": status.get("failure_kind", "RuntimeError"),
            "failure_category": status.get("failure_category", "preflight_refused"),
            "prepared": False,
        }
        with (plan.output / "preparation-refused.private.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(refusal, stream, sort_keys=True)
        raise ValueError("preflight_refused")
    prepared_plan = {
        "version": 1,
        "plan": raw_plan,
        "plan_sha256": digest(plan_path),
        "profile": str(profile),
        "config_sha256": digest(profile / "config.yaml"),
        "source_config_sha256": digest(plan.profile_source / "config.yaml"),
        "speech_asset_receipt_sha256": digest(plan.speech_assets["receipt"]),
        "plugin_files_sha256": (
            {path.name: digest(path) for path in (profile / "plugins" / _PLUGIN).iterdir()}
            if plan.policy_mode == "lookup"
            else {}
        ),
        "output": str(plan.output),
        "policy_mode": plan.policy_mode,
        "profile_complete": False,
        "preflight": "accepted",
    }
    destination = plan.output / "prepared.private.json"
    record_prepared_after_preflight(prepared_plan, completed, destination)
    return {"category": "prepared", "profile_complete": False, "output": str(plan.output)}


def _read_json_file(path: Path, limit: int = _JSON_LIMIT) -> dict[str, Any]:
    require(
        path.is_file()
        and not path.is_symlink()
        and not (getattr(path.lstat(), "st_file_attributes", 0) & 0x400),
        "receipt_file",
    )
    value = _json(_read(path, limit))
    require(type(value) is dict, "receipt_shape")
    return value


def _listener_rows(port: int) -> list[tuple[str, int]]:
    from scripts.lookup_backend_task import listener_rows

    return listener_rows(port)


def _dev_readiness(plan: PreviewPlanV1) -> bool:
    return plan.readiness_policy == "dev_owned_services"


def _tcp_accepting(port: int) -> bool:
    from scripts.preview_readiness import tcp_accepting

    return tcp_accepting(port)


def _endpoint_snapshot(plan: PreviewPlanV1, ports: Any) -> dict[int, Any]:
    observe = _tcp_accepting if _dev_readiness(plan) else _listener_rows
    return {port: observe(port) for port in ports}


def _endpoint_count(plan: PreviewPlanV1, snapshot: dict[int, Any]) -> int:
    return sum(snapshot.values()) if _dev_readiness(plan) else sum(map(len, snapshot.values()))


def _require_host_http_ready(port: int) -> None:
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        require(response.status == 200, "host_http_not_ready")
        response.close()
    finally:
        connection.close()


def _require_owned_endpoint_rows(
    rows: Any, expected_pid: int, members: set[int], expected_count: int,
    *, role_pids: set[int] | None = None,
) -> int:
    allowed = {expected_pid} if role_pids is None else role_pids
    require(
        type(expected_pid) is int
        and expected_pid > 0
        and expected_pid in members
        and type(allowed) is set
        and expected_pid in allowed
        and allowed <= members
        and type(rows) is list
        and len(rows) == expected_count
        and all(
            type(row) is tuple
            and len(row) == 2
            and row[0] == "127.0.0.1"
            and type(row[1]) is int
            and row[1] in allowed
            for row in rows
        ),
        "endpoint_owner",
    )
    owners = {row[1] for row in rows}
    require(len(owners) == 1, "endpoint_owner")
    return next(iter(owners))


def _owned_role_members(root_pid: int, members: set[int]) -> set[int]:
    from scripts.rehearse_desktop_mvp import descendants, process_table

    return descendants(process_table(), {root_pid}) & members


def _require_gateway_health_owner(health: Any, expected_pid: int, members: set[int]) -> None:
    require(
        type(health) is dict
        and type(health.get("pid")) is int
        and health["pid"] == expected_pid
        and health["pid"] in members,
        "gateway_health_owner",
    )


def _started_at(pid: int) -> int | None:
    from scripts.rehearse_desktop_mvp import started_at

    return started_at(pid)


def verify_normal_stop(previous: Path, plan: PreviewPlanV1) -> dict[str, Any]:
    result_path = previous / "result-receipt.private.json"
    result = _read_json_file(result_path)
    previous_policy = result.get("readiness_policy", "kernel_listeners")
    previous_dev = previous_policy == "dev_owned_services"
    require(previous_policy in {"kernel_listeners", "dev_owned_services"}, "previous_normal_stop")
    if previous_dev:
        require(
            _dev_readiness(plan)
            and result.get("listener_ownership") == "unavailable"
            and result.get("remaining_owned_listeners") is None
            and result.get("protected_listeners_unchanged") is None
            and type(result.get("remaining_accepting_endpoints")) is int
            and result["remaining_accepting_endpoints"] == 0
            and (result.get("protected_endpoints_unchanged") is None
                 or type(result["protected_endpoints_unchanged"]) is bool),
            "previous_normal_stop",
        )
    else:
        require(
            result.get("listener_ownership", "verified") == "verified"
            and type(result.get("remaining_owned_listeners")) is int
            and result["remaining_owned_listeners"] == 0
            and type(result.get("protected_listeners_unchanged")) is bool,
            "previous_normal_stop",
        )
    require(
        result.get("category") == "stopped"
        and result.get("cleanup_settled") is True
        and result.get("cleanup_failed") is False
        and type(result.get("owned_survivor_count")) is int
        and result["owned_survivor_count"] == 0,
        "previous_normal_stop",
    )
    previous_ports = result.get("ports")
    if "ports" not in result:
        # The existing persistent supervisor records these same four ports flat.
        previous_ports = [result.get(key) for key in (
            "api_port", "companion_port", "livekit_port", "browser_port",
        )]
    require(
        type(previous_ports) is list
        and len(previous_ports) == 4
        and all(type(port) is int and 1024 <= port <= 65535 for port in previous_ports)
        and len(set(previous_ports)) == 4,
        "previous_ports_shape",
    )
    pids = _read_json_file(previous / "pids.json")
    require(
        set(pids) == {"gateway", "livekit", "host"}
        and all(type(pid) is int and pid > 0 for pid in pids.values()),
        "previous_pid_shape",
    )
    supervisor_path = previous / "supervisor.pid.private.txt"
    require(
        supervisor_path.is_file()
        and not supervisor_path.is_symlink()
        and supervisor_path.stat().st_size <= 32,
        "previous_supervisor_pid_file",
    )
    supervisor_raw = _read(supervisor_path, 32).decode("ascii").strip()
    require(supervisor_raw.isdecimal() and int(supervisor_raw) > 0, "previous_supervisor_pid")
    require(
        all(_started_at(pid) is None for pid in (*pids.values(), int(supervisor_raw))),
        "previous_process_alive",
    )
    all_ports = sorted(
        set(previous_ports) | {plan.ports[name] for name in ("api", "companion", "livekit", "host")}
    )
    require(not any(_endpoint_snapshot(plan, all_ports).values()), "previous_listener_alive")
    journal = _read_json_file(previous / "state" / "hermes-runs-v1.json")
    require(
        set(journal) == {"version", "pending", "admitted"}
        and type(journal["version"]) is int
        and journal["version"] == 1
        and type(journal["pending"]) is list
        and journal["pending"] == []
        and type(journal["admitted"]) is list
        and journal["admitted"] == [],
        "previous_work_unsettled",
    )
    return {
        "category": "normal_stop_verified",
        "receipt_sha256": digest(result_path),
        "processes_absent": True,
        "listeners_absent": None if _dev_readiness(plan) else True,
        "listener_ownership": "unavailable" if _dev_readiness(plan) else "verified",
        "endpoints_not_accepting": True if _dev_readiness(plan) else None,
        "work_journal_empty": True,
        "protected_listeners_unchanged": (
            None if _dev_readiness(plan) else result["protected_listeners_unchanged"]
        ),
    }


def _copy_tree(source: Path, target: Path) -> tuple[int, int]:
    files = _safe_profile_inventory(source)
    require(not target.exists(), "profile_copy_collision")
    target.mkdir(parents=False)
    directories = sorted(
        (item for item in source.rglob("*") if item.is_dir()),
        key=lambda item: len(item.relative_to(source).parts),
    )
    for item in directories:
        (target / item.relative_to(source)).mkdir(parents=False)
    copied = 0
    total = 0
    for source_file in files:
        relative = source_file.relative_to(source)
        out = target / relative
        _copy_bounded_file(source_file, out, _MAX_PROFILE_BYTES)
        copied += 1
        total += out.stat().st_size
        require(copied <= _MAX_PROFILE_FILES and total <= _MAX_PROFILE_BYTES, "profile_copy_bound")
        require(digest(source_file) == digest(out), "profile_copy_hash")
    return copied, total


def _copy_profile_supplement(plan: PreviewPlanV1, prepared: dict[str, Any]) -> dict[str, Any]:
    home = Path(prepared["profile"])
    expected_config = prepared.get("config_sha256")
    expected_source_config = prepared.get("source_config_sha256")
    require(
        type(expected_config) is str
        and _HEX64.fullmatch(expected_config)
        and digest(home / "config.yaml") == expected_config,
        "prepared_config_drift",
    )
    require(
        type(expected_source_config) is str and _HEX64.fullmatch(expected_source_config),
        "source_config_binding",
    )
    expected_speech_receipt = prepared.get("speech_asset_receipt_sha256")
    require(
        type(expected_speech_receipt) is str
        and _HEX64.fullmatch(expected_speech_receipt)
        and digest(plan.speech_assets["receipt"]) == expected_speech_receipt,
        "prepared_speech_asset_receipt_drift",
    )
    if plan.policy_mode == "lookup":
        plugin = home / "plugins" / _PLUGIN
        expected_plugins = prepared.get("plugin_files_sha256")
        require(
            type(expected_plugins) is dict
            and expected_plugins
            and {path.name: digest(path) for path in plugin.iterdir() if path.is_file()}
            == expected_plugins,
            "prepared_policy_drift",
        )
    else:
        expected_plugins = {}

    source = plan.profile_source
    _safe_profile_inventory(source)
    allowed = {"config.yaml", ".env", "SOUL.md", "skills", "hooks", "plugins"}
    require(all(item.name in allowed for item in source.iterdir()), "profile_entry_not_allowed")
    counts = {"files": 0, "bytes": 0}

    def copy_file(source_file: Path, destination: Path, maximum: int) -> None:
        digest_source = _copy_bounded_file(source_file, destination, maximum)
        require(digest(destination) == digest_source, "profile_copy_hash")
        counts["files"] += 1
        counts["bytes"] += destination.stat().st_size

    for item in source.iterdir():
        if item.name == "config.yaml":
            require(digest(item) == expected_source_config, "profile_config_changed")
            continue
        if item.name == "plugins":
            require(item.is_dir() and not item.is_symlink(), "profile_plugins_shape")
            (home / "plugins").mkdir(exist_ok=True)
            for plugin_item in item.iterdir():
                if plugin_item.name == _PLUGIN:
                    continue
                destination = home / "plugins" / plugin_item.name
                if plugin_item.is_dir():
                    copied, total = _copy_tree(plugin_item, destination)
                    counts["files"] += copied
                    counts["bytes"] += total
                elif plugin_item.is_file():
                    copy_file(plugin_item, destination, _MAX_PROFILE_BYTES)
                else:
                    raise ValueError("profile_plugin_entry")
            continue
        destination = home / item.name
        if item.name == ".env":
            require(item.is_file() and item.stat().st_size <= 1024**2, "profile_auth_file")
            copy_file(item, destination, 1024**2)
        elif item.name == "SOUL.md":
            require(item.is_file(), "profile_soul_file")
            copy_file(item, destination, 2_000_000)
        else:
            require(item.is_dir(), "profile_tree_shape")
            copied, total = _copy_tree(item, destination)
            counts["files"] += copied
            counts["bytes"] += total

    require(any(item.name == ".env" for item in source.iterdir()), "profile_auth_missing")
    require(
        counts["files"] <= _MAX_PROFILE_FILES and counts["bytes"] <= _MAX_PROFILE_BYTES,
        "profile_copy_bound",
    )
    require(digest(home / "config.yaml") == expected_config, "profile_config_copy")
    if plan.policy_mode == "lookup":
        require(
            {path.name: digest(path) for path in plugin.iterdir() if path.is_file()}
            == expected_plugins,
            "profile_policy_copy",
        )
    verify_profile(plan, home)
    verify_speech_assets(plan, home)
    return {
        "config_sha256": expected_config,
        "plugin_files_sha256": expected_plugins,
        "copied_file_count": counts["files"],
        "copied_bytes": counts["bytes"],
        "auth_copied_opaque": True,
    }


def _copy_voice_tail(previous: Path, output: Path, runtime_target: Path) -> dict[str, Any]:
    source = previous / "state" / "voice-tail-v1.json"
    require(
        source.is_file()
        and not source.is_symlink()
        and source.resolve().parent == (previous / "state").resolve(),
        "saved_voice_tail_missing",
    )
    package_parent = runtime_target.resolve(strict=True)
    sys.path.insert(0, str(package_parent))
    try:
        import hermes_realtime.integration.voice_tail as voice_tail

        expected_module = runtime_target / "hermes_realtime" / "integration" / "voice_tail.py"
        require(
            Path(voice_tail.__file__).resolve() == expected_module.resolve(),
            "voice_tail_import_origin",
        )
        maximum = voice_tail.max_voice_tail_bytes(32, 1024)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(package_parent))
    require(source.stat().st_size <= maximum, "saved_voice_tail_bound")
    raw = _read(source, maximum)
    target_state = output / "state"
    target_state.mkdir(parents=False, exist_ok=False)
    target = target_state / "voice-tail-v1.json"
    target.write_bytes(raw)
    require(digest(source) == digest(target), "saved_voice_tail_copy")
    return {"bytes": len(raw), "sha256": digest(target)}


def finalize(prepared_path: Path, previous: Path) -> dict[str, Any]:
    prepared = _read_json_file(prepared_path)
    require(
        prepared.get("version") == 1
        and prepared.get("preflight") == "accepted"
        and prepared.get("profile_complete") is False,
        "prepared_receipt_state",
    )
    output = Path(prepared.get("output", ""))
    require(
        prepared_path.resolve() == (output / "prepared.private.json").resolve(),
        "prepared_receipt_path",
    )
    plan_path = output / "plan.private.json"
    require(digest(plan_path) == prepared.get("plan_sha256"), "prepared_plan_drift")
    plan = load_plan(plan_path)
    stop = verify_normal_stop(previous, plan)
    supplemental = _copy_profile_supplement(plan, prepared)
    tail = _copy_voice_tail(previous, output, plan.runtime_target)
    handoff = {
        "version": 1,
        "category": "normal_stop_handoff_verified",
        "previous_stop": stop,
        "profile": supplemental,
        "voice_tail": tail,
    }
    handoff_path = output / "handoff.private.json"
    with handoff_path.open("x", encoding="utf-8") as stream:
        json.dump(handoff, stream, sort_keys=True, separators=(",", ":"))
    active = {
        "version": 1,
        "category": "ready_to_launch",
        "prepared_sha256": digest(prepared_path),
        "handoff_sha256": digest(handoff_path),
        "profile_complete": True,
        "launch_authorized": False,
    }
    active_path = output / "active.private.json"
    with active_path.open("x", encoding="utf-8") as stream:
        json.dump(active, stream, sort_keys=True, separators=(",", ":"))
    return {
        "category": "ready_to_launch",
        "profile_complete": True,
        "launch_authorized": False,
        "output": str(output),
    }


def _active_plan(active_path: Path) -> tuple[PreviewPlanV1, dict[str, Any], Path]:
    active = _read_json_file(active_path)
    require(
        active.get("version") == 1
        and active.get("category") == "ready_to_launch"
        and active.get("profile_complete") is True
        and active.get("launch_authorized") is False,
        "active_receipt_state",
    )
    output = active_path.resolve().parent
    prepared_path = output / "prepared.private.json"
    handoff_path = output / "handoff.private.json"
    require(
        digest(prepared_path) == active.get("prepared_sha256")
        and digest(handoff_path) == active.get("handoff_sha256"),
        "active_receipt_binding",
    )
    prepared = _read_json_file(prepared_path)
    require(
        prepared.get("profile_complete") is False and prepared.get("preflight") == "accepted",
        "prepared_receipt_state",
    )
    plan_path = output / "plan.private.json"
    require(digest(plan_path) == prepared.get("plan_sha256"), "active_plan_binding")
    plan = load_plan(plan_path)
    profile = Path(prepared.get("profile", ""))
    require(
        profile.resolve() == (output / "input" / "home").resolve()
        and plan.output.resolve() == output,
        "active_path_binding",
    )
    require(
        prepared.get("speech_asset_receipt_sha256") == digest(plan.speech_assets["receipt"]),
        "active_speech_asset_receipt_binding",
    )
    return plan, prepared, profile


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(
        len(encoded) <= _JSON_LIMIT and not path.exists() and not path.is_symlink(),
        "output_receipt_refused",
    )
    fd, temporary_name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish a complete inode without replacing a receipt another writer owns.
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise PreviewRefusal("output_receipt_refused") from None
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_modules(plan: PreviewPlanV1) -> tuple[Any, Any]:
    runtime_target = plan.runtime_target.resolve(strict=True)
    sys.path.insert(0, str(runtime_target))
    sys.path.insert(1, str(plan.hermes_source))
    try:
        realtime = importlib.import_module("hermes_realtime")
        host = importlib.import_module("hermes_realtime.host_launcher")
        expected = plan.runtime_target / "hermes_realtime" / "__init__.py"
        require(Path(realtime.__file__).resolve() == expected.resolve(), "runtime_import_origin")
        return realtime, host
    except BaseException:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(runtime_target))
            sys.path.remove(str(plan.hermes_source))
        raise


def _gateway_child(active_path: Path) -> int:
    plan, _, profile = _active_plan(active_path)
    _runtime_modules(plan)
    sys.path[:0] = [str(plan.hermes_source), str(plan.runtime_target)]
    cli = importlib.import_module("hermes_cli.main")
    expected_prefix = plan.hermes_source.resolve()
    require(
        Path(cli.__file__).resolve().is_relative_to(expected_prefix), "hermes_cli_import_origin"
    )
    sys.argv = ["hermes", "gateway", "run"]
    result = cli.main()
    return result if type(result) is int else 0


def _livekit_command(executable: Path, port: int) -> list[str]:
    from scripts import local_livekit

    return local_livekit.server_command(executable) + ["--port", str(port)]


def _emit_host_failure(facts: dict[str, str], error: BaseException) -> None:
    stream = sys.stderr
    if stream is not None:
        try:
            print("[voice-preview-host-failure] " + json.dumps({
                "version": 1, "stage": facts["stage"],
                "failure_kind": _failure_status(error)["failure_kind"],
            }, separators=(",", ":")), file=stream, flush=True)
        except Exception:
            if sys.stderr is stream:
                sys.stderr = None


async def _host_child(active_path: Path) -> int:
    facts = {"stage": "configuration"}
    failure: BaseException | None = None
    try:
        return await _run_host_child(active_path, facts)
    except BaseException as error:
        failure = error
        raise
    finally:
        if failure is not None:
            _emit_host_failure(facts, failure)


async def _run_host_child(active_path: Path, facts: dict[str, str]) -> int:
    plan, _, profile = _active_plan(active_path)
    _, host = _runtime_modules(plan)
    env_file = profile / ".env"
    bearer = host._load_api_bearer(env_file)
    companion = host._load_voice_companion(env_file, os.environ)
    if companion is not None:
        require(companion.port == plan.ports["companion"], "companion_port_binding")
    options = dict(plan.host_options)
    parameters = inspect.signature(host.build_local_host_launcher).parameters
    if plan.policy_mode == "native":
        require("hermes_discover_tools" in parameters, "native_discovery_unavailable")
        discover = True
        lookup_policy = None
    else:
        discover = False
        lookup_policy = _POLICY
    kwargs: dict[str, Any] = {
        "hermes_api_bearer": bearer,
        "hermes_api_url": f"http://127.0.0.1:{plan.ports['api']}",
        "hermes_model_source": "gateway",
        "hermes_lookup_policy": lookup_policy,
        "livekit_url": f"ws://127.0.0.1:{plan.ports['livekit']}",
        "room_name": "hermes-preview-" + hashlib.sha256(str(active_path).encode()).hexdigest()[:16],
        "worker_identity": "worker_preview_"
        + hashlib.sha256(str(active_path).encode()).hexdigest()[:16],
        "browser_host": "127.0.0.1",
        "browser_port": plan.ports["host"],
        "inference_provider": "codex",
        "codex_model": plan.models["foreground"],
        "codex_effort": plan.models["foreground_effort"],
        "codex_executable": str(plan.executables["codex"]),
        "stt_provider": plan.speech["stt_provider"],
        "moonshine_model_tier": plan.speech["stt_tier"],
        "tts_provider": plan.speech["tts_provider"],
        "kokoro_voice": plan.speech["tts_voice"],
        "kokoro_intra_op_threads": plan.speech["tts_threads"],
        "kokoro_stream_chunk_chars": plan.speech["tts_chunk_chars"],
        "allow_unsandboxed_tasks": options["allow_unsandboxed_tasks"],
        "natural_work_tools": options["natural_work_tools"],
        "task_start_policy": options["task_start_policy"],
        "conversation_profile": options["conversation_profile"],
        "loopback_authorizer": host.LoopbackPeerAuthorizer(),
        "hermes_run_record": plan.output / "state" / "hermes-runs-v1.json",
        "voice_tail": plan.output / "state" / "voice-tail-v1.json",
        "voice_companion": companion,
    }
    if plan.policy_mode == "native":
        config = _profile_config(_read(profile / "config.yaml", _JSON_LIMIT))
        contract = native_web_evidence_contract(config)
        if contract is not None:
            require("hermes_web_evidence_contract" in parameters, "native_web_evidence_unavailable")
            kwargs["hermes_web_evidence_contract"] = contract
    if "hermes_discover_tools" in parameters:
        kwargs["hermes_discover_tools"] = discover
    mcp_config = _bound_kokoro_mcp_config(plan)
    if mcp_config is not None:
        require("kokoro_mcp_config" in parameters, "kokoro_mcp_host_unavailable")
        kwargs["kokoro_mcp_config"] = mcp_config
    facts["stage"] = "host_construction"
    launcher = host.build_local_host_launcher(**kwargs)
    ready = plan.output / "host-ready.private.json"
    try:
        facts["stage"] = "host_start"
        await launcher.start()
        _write_new(ready, {"version": 1, "ready": True, "pid": os.getpid()})
        facts["stage"] = "serving"
        while not (plan.output / "host.stop").exists():
            await asyncio.sleep(_STOP_POLL_SECONDS)
    finally:
        try:
            await launcher.close()
        except BaseException:
            facts["stage"] = "host_close"
            raise
    return 0


def _base_environment() -> dict[str, str]:
    names = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PROGRAMFILES")
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def _gateway_environment(profile: Path, api_port: int) -> dict[str, str]:
    env = _base_environment()
    env.update(
        {
            "HOME": str(profile),
            "USERPROFILE": str(profile),
            "APPDATA": str(profile),
            "LOCALAPPDATA": str(profile),
            "HERMES_HOME": str(profile),
            "CODEX_HOME": str(profile),
            "API_SERVER_HOST": "127.0.0.1",
            "API_SERVER_PORT": str(api_port),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
        }
    )
    return env


def _host_environment(plan: PreviewPlanV1, profile: Path) -> dict[str, str]:
    _, host = _runtime_modules(plan)
    from hermes_realtime.providers.codex_app_server import _subscription_environment

    env = _subscription_environment(os.environ)
    env.update(_base_environment())
    env.update(
        {
            "HERMES_HOME": str(profile),
            "HOME": str(profile),
            "USERPROFILE": str(profile),
            "MOONSHINE_VOICE_CACHE": str(profile / ".cache" / "moonshine"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
        }
    )
    original_home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if "CODEX_HOME" not in env and original_home is not None:
        env["CODEX_HOME"] = str(Path(original_home) / ".codex")
    return env


def _gateway_manifest(plan: PreviewPlanV1) -> Path:
    path = plan.output / "gateway-manifest.private.json"
    require(not path.exists(), "gateway_manifest_exists")
    records = list(plan.runtime_target.glob("*.dist-info/RECORD"))
    require(len(records) == 1, "runtime_record_set")
    payload = {
        "objective": "Interactive voice preview",
        "hermes_source": str(plan.hermes_source),
        "hermes_python": str(plan.hermes_python),
        "runtime_target": str(plan.runtime_target),
        "packageRoot": str(plan.runtime_target / "hermes_realtime"),
        "recordPath": str(records[0]),
        "recordSha256": plan.runtime_record_sha256,
        "baseUrl": f"http://127.0.0.1:{plan.ports['api']}",
        "api_port": plan.ports["api"],
        "companion_port": plan.ports["companion"],
    }
    _write_new(path, payload)
    return path


def launch_preview(active_path: Path) -> dict[str, Any]:
    """Request an independent owner; readiness remains the owner's receipt."""
    require(sys.platform == "win32", "windows_required")
    active_path = active_path.resolve()
    plan, _, _ = _active_plan(active_path)
    from scripts.preview_owner_launch import PreviewOwnerLaunchRefusal, launch_preview_owner

    try:
        launch_preview_owner(
            interpreter=plan.executables["host_python"], active=active_path, output=plan.output
        )
    except PreviewOwnerLaunchRefusal as error:
        raise PreviewRefusal(error.category) from None
    return {"category": "launch_requested", "launch_authorized": True, "ready": False}


def serve(active_path: Path) -> int:
    require(sys.platform == "win32", "windows_required")
    plan, _, profile = _active_plan(active_path)
    preflight(plan, profile)
    output = plan.output
    for name in (
        "stop.request",
        "host.stop",
        "ready-receipt.private.json",
        "result-receipt.private.json",
        "pids.json",
        "supervisor.pid.private.txt",
        "gateway-witness.private.json",
    ):
        require(not (output / name).exists(), "run_artifact_exists")
    ports = [plan.ports[name] for name in ("api", "companion", "livekit", "host")]
    from scripts.lookup_backend_task import request_json

    require(not any(_endpoint_snapshot(plan, ports).values()), "selected_port_occupied")
    try:
        protected_before = _endpoint_snapshot(plan, plan.ports["protected"])
    except BaseException:
        protected_before = None
    with (output / "supervisor.pid.private.txt").open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()) + "\n")
    from scripts import local_livekit
    from scripts.lookup_backend_logs import CappedProcesses
    from scripts.rehearse_desktop_mvp import Containment, Processes

    containment = None
    owned = None
    processes = None
    result: dict[str, Any] = {
        "category": "starting", "ready": False, "readiness_policy": plan.readiness_policy,
        "listener_ownership": "unavailable" if _dev_readiness(plan) else "not_established",
    }
    startup_error: str | None = None
    gateway_manifest = None
    try:
        from scripts.preview_owner_launch import (
            PreviewOwnerLaunchRefusal,
            assert_independent_owner,
        )

        try:
            assert_independent_owner(interpreter=plan.executables["host_python"])
        except PreviewOwnerLaunchRefusal as error:
            raise PreviewRefusal(error.category) from None
        containment = Containment()
        owned = Processes(output)
        processes = CappedProcesses(owned, output)
        if plan.policy_mode == "lookup":
            gateway_manifest = _gateway_manifest(plan)
            gateway_command = [
                str(plan.hermes_python),
                "-I",
                "-B",
                str(ROOT / "scripts" / "lookup_backend_gateway.py"),
                str(gateway_manifest),
                "--transport-profile",
                "interactive",
            ]
        else:
            gateway_command = [
                str(plan.hermes_python),
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "gateway-child",
                "--active",
                str(active_path),
            ]
        if processes is None:
            raise ValueError("process_owner_unavailable")
        gateway_process = processes.spawn(
            "gateway",
            gateway_command,
            env=_gateway_environment(profile, plan.ports["api"]),
            cwd=plan.hermes_source,
            log=Path(os.devnull),
        )
        livekit_path = local_livekit.verified_server()
        require(
            livekit_path.resolve() == plan.executables["livekit"].resolve(), "livekit_path_binding"
        )
        livekit_env = _base_environment() | {"LIVEKIT_KEYS": local_livekit.DEVELOPMENT_KEYS}
        livekit_process = processes.spawn(
            "livekit",
            _livekit_command(livekit_path, plan.ports["livekit"]),
            env=livekit_env,
            cwd=livekit_path.parent,
            log=Path(os.devnull),
        )
        host_command = [
            str(plan.executables["host_python"]),
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "host-child",
            "--active",
            str(active_path),
        ]
        base_url = f"http://127.0.0.1:{plan.ports['api']}"
        token = None
        _, host_module = _runtime_modules(plan)
        token = host_module._load_api_bearer(profile / ".env")
        deadline = time.monotonic() + 900
        gateway_ready = False
        while time.monotonic() < deadline:
            require(not (output / "stop.request").exists(), "stop_requested_during_start")
            require(
                all(processes.roots[name].poll() is None for name in ("gateway", "livekit")),
                "child_exited_during_start",
            )
            if _dev_readiness(plan):
                if all(_endpoint_snapshot(plan, [plan.ports[name] for name in
                                                ("api", "companion", "livekit")]).values()):
                    from urllib.error import HTTPError

                    from scripts.preview_readiness import require_gateway_health_owner

                    members = containment.members()
                    require(livekit_process.pid in members, "livekit_process_owner")
                    try:
                        health = request_json(base_url + "/health/detailed", token, timeout=3)
                    except HTTPError as error:
                        if error.code not in {502, 503, 504}:
                            raise PreviewRefusal("gateway_health_refused") from None
                    except ValueError:
                        raise PreviewRefusal("gateway_health_invalid") from None
                    except (OSError, TimeoutError):
                        pass
                    else:
                        require_gateway_health_owner(
                            health, root_pid=gateway_process.pid, members=members,
                        )
                        gateway_ready = True
                        break
            else:
                api_rows = _listener_rows(plan.ports["api"])
                companion_rows = _listener_rows(plan.ports["companion"])
                livekit_rows = _listener_rows(plan.ports["livekit"])
                if api_rows and companion_rows and livekit_rows:
                    members = containment.members()
                    from scripts.lookup_backend_task import require_owned_listeners

                    gateway_rows = api_rows + companion_rows
                    require_owned_listeners(gateway_rows, members)
                    gateway_pid = _require_owned_endpoint_rows(
                        gateway_rows, gateway_process.pid, members, 2,
                        role_pids=_owned_role_members(gateway_process.pid, members),
                    )
                    _require_owned_endpoint_rows(livekit_rows, livekit_process.pid, members, 1)
                    try:
                        health = request_json(base_url + "/health/detailed", token, timeout=3)
                    except (OSError, ValueError, TimeoutError):
                        pass
                    else:
                        _require_gateway_health_owner(health, gateway_pid, members)
                        gateway_ready = True
                        break
            time.sleep(_STOP_POLL_SECONDS)
        require(gateway_ready, "gateway_start_timeout")
        host_process = processes.spawn(
            "host",
            host_command,
            env=_host_environment(plan, profile),
            cwd=plan.hermes_source,
            log=Path(os.devnull),
        )
        host_ready_path = output / "host-ready.private.json"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not host_ready_path.exists():
            require(not (output / "stop.request").exists(), "stop_requested_during_start")
            require(processes.roots["host"].poll() is None, "host_exited_during_start")
            time.sleep(_STOP_POLL_SECONDS)
        require(host_ready_path.is_file(), "host_start_timeout")
        members = containment.members()
        host_ready = _read_json_file(host_ready_path)
        if _dev_readiness(plan):
            from scripts.preview_readiness import require_host_readiness

            require(livekit_process.poll() is None and livekit_process.pid in members,
                    "livekit_process_owner")
            require_host_readiness(host_ready, root_pid=host_process.pid, members=members)
            _require_host_http_ready(plan.ports["host"])
        else:
            host_pid = _require_owned_endpoint_rows(
                _listener_rows(plan.ports["host"]), host_process.pid, members, 1,
                role_pids=_owned_role_members(host_process.pid, members),
            )
            require(
                host_ready == {"version": 1, "ready": True, "pid": host_pid},
                "host_ready_receipt",
            )
        result.update(category="ready", ready=True, ports=ports,
                      listener_ownership="unavailable" if _dev_readiness(plan) else "verified")
        _write_new(output / "ready-receipt.private.json", result)
        emit({"category": "ready", "launch_authorized": True, "ports": ports})
        while not (output / "stop.request").exists():
            require(
                all(process.poll() is None for process in processes.roots.values()),
                "owned_child_exited",
            )
            time.sleep(_STOP_POLL_SECONDS)
        startup_error = None
    except BaseException as error:
        startup_error = (
            None
            if isinstance(error, KeyboardInterrupt) and result.get("ready") is True
            else "operator_stop"
            if isinstance(error, KeyboardInterrupt)
            else _failure_status(error)["failure_category"]
        )
        if isinstance(error, KeyboardInterrupt) and result.get("ready") is True:
            with contextlib.suppress(FileExistsError):
                (output / "stop.request").open("x").close()
        result.update(category="refused", failure_category=startup_error)
    finally:
        with contextlib.suppress(FileExistsError):
            (output / "host.stop").open("x").close()
        host_process = processes.roots.get("host") if processes is not None else None
        if host_process is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                host_process.wait(timeout=45)
        cleanup_failed = False
        child_facts = {}
        survivors = {}
        if processes is not None and owned is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    processes.stop_all(containment)
            except BaseException:
                cleanup_failed = True
            try:
                child_facts = processes.finish()
            except BaseException:
                cleanup_failed = True
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    survivors = owned.survivors(containment)
            except BaseException:
                cleanup_failed = True
                survivors = {"<unknown>": 1}
        survivor_count = sum(survivors.values())
        remaining: int | None
        try:
            remaining = _endpoint_count(plan, _endpoint_snapshot(plan, ports))
        except BaseException:
            cleanup_failed = True
            remaining = None if _dev_readiness(plan) else max(1, len(ports))
        try:
            protected_after = _endpoint_snapshot(plan, plan.ports["protected"])
            protected_unchanged = (
                None if protected_before is None and _dev_readiness(plan) else
                protected_before is not None and protected_after == protected_before
            )
        except BaseException:
            protected_unchanged = None if _dev_readiness(plan) else False
        journal_path = output / "state" / "hermes-runs-v1.json"
        try:
            journal = _read_json_file(journal_path)
            journal_empty = (
                set(journal) == {"version", "pending", "admitted"}
                and type(journal["version"]) is int
                and journal["version"] == 1
                and journal["pending"] == []
                and journal["admitted"] == []
            )
        except (OSError, ValueError, json.JSONDecodeError):
            journal_empty = False
        settled = not cleanup_failed and survivor_count == 0 and remaining == 0
        stopped = (
            result.get("ready") is True and startup_error is None and settled and journal_empty
        )
        receipt = {
            "version": 1,
            "category": "stopped" if stopped else "refused",
            "cleanup_settled": settled,
            "cleanup_failed": cleanup_failed,
            "owned_survivor_count": survivor_count,
            "readiness_policy": plan.readiness_policy,
            "listener_ownership": "unavailable" if _dev_readiness(plan) else "verified",
            "remaining_owned_listeners": None if _dev_readiness(plan) else remaining,
            "protected_listeners_unchanged": None if _dev_readiness(plan) else protected_unchanged,
            "remaining_accepting_endpoints": remaining if _dev_readiness(plan) else None,
            "protected_endpoints_unchanged": protected_unchanged if _dev_readiness(plan) else None,
            "run_journal_empty": journal_empty,
            "ports": ports,
            "child_output_metadata": child_facts,
            "failure_category": startup_error if not stopped else None,
        }
        _write_new(output / "result-receipt.private.json", receipt)
        for name in ("gateway.private.log", "livekit.private.log", "host.private.log"):
            with contextlib.suppress(FileNotFoundError):
                (output / name).unlink()
        if stopped:
            emit({"category": "stopped", "launch_authorized": False, "ports": ports})
        else:
            emit({"category": "refused", "launch_authorized": False, "ports": ports})
    return 0 if stopped else 1


def stop_preview(output: Path, timeout: float = 180.0) -> dict[str, Any]:
    require(output.is_dir() and not output.is_symlink(), "run_directory")
    result_path = output / "result-receipt.private.json"
    if result_path.is_file():
        result = _read_json_file(result_path)
        require(
            result.get("category") == "stopped" and result.get("cleanup_settled") is True,
            "run_not_cleanly_stopped",
        )
        return {"category": "stopped", "launch_authorized": False, "ports": result.get("ports", [])}
    marker = output / "stop.request"
    require(not marker.is_symlink(), "stop_marker_link")
    if not marker.exists():
        with marker.open("x") as stream:
            stream.write("")
    else:
        require(marker.is_file() and marker.stat().st_size == 0, "stop_marker_shape")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result_path.is_file():
            result = _read_json_file(result_path)
            require(
                result.get("category") == "stopped"
                and result.get("cleanup_settled") is True
                and result.get("cleanup_failed") is False,
                "run_not_cleanly_stopped",
            )
            return {
                "category": "stopped",
                "launch_authorized": False,
                "ports": result.get("ports", []),
            }
        time.sleep(_STOP_POLL_SECONDS)
    raise TimeoutError("stop_timeout")


def _failure_status(error: BaseException) -> dict[str, str]:
    from scripts.preview_readiness import PreviewReadinessRefusal

    known = {
        "ValueError",
        "RuntimeError",
        "OSError",
        "TimeoutError",
        "TypeError",
        "ImportError",
        "PermissionError",
    }
    kind = ("ValueError" if type(error) in (PreviewRefusal, PreviewReadinessRefusal)
            else type(error).__name__)
    kind = kind if kind in known else "other"
    category = "preflight_failed"
    if (type(error) in (PreviewRefusal, PreviewReadinessRefusal) and len(error.args) == 1
            and type(error.args[0]) is str and _LABEL.fullmatch(error.args[0])):
        category = error.args[0]
    return {"category": "refused", "failure_kind": kind, "failure_category": category}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--plan", type=Path, required=True)
    preflight_parser.add_argument("--home", type=Path, required=True)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--prepared", type=Path, required=True)
    finalize_parser.add_argument("--previous", type=Path, required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--active", type=Path, required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--active", type=Path, required=True)
    stop_parser = commands.add_parser("stop")
    stop_parser.add_argument("--output", type=Path, required=True)
    host_child_parser = commands.add_parser("host-child", help=argparse.SUPPRESS)
    host_child_parser.add_argument("--active", type=Path, required=True)
    gateway_child_parser = commands.add_parser("gateway-child", help=argparse.SUPPRESS)
    gateway_child_parser.add_argument("--active", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            plan = load_plan(args.plan)
            status = preflight(plan, args.home)
            emit(status)
            return 0
        if args.command == "prepare":
            result = prepare(args.plan)
            plan = load_plan(args.plan)
            emit(
                {
                    "category": result["category"],
                    "launch_authorized": False,
                    "ports": [plan.ports[name] for name in ("api", "companion", "livekit", "host")],
                }
            )
            return 0
        if args.command == "finalize":
            result = finalize(args.prepared, args.previous)
            plan = load_plan(Path(result["output"]) / "plan.private.json")
            emit(
                {
                    "category": result["category"],
                    "launch_authorized": False,
                    "ports": [plan.ports[name] for name in ("api", "companion", "livekit", "host")],
                }
            )
            return 0
        if args.command == "serve":
            return serve(args.active)
        if args.command == "launch":
            emit(launch_preview(args.active))
            return 0
        if args.command == "stop":
            emit(stop_preview(args.output))
            return 0
        if args.command == "gateway-child":
            return _gateway_child(args.active)
        if args.command == "host-child":
            return asyncio.run(_host_child(args.active))
        raise ValueError("command_unavailable")
    except BaseException as error:
        emit(_failure_status(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
