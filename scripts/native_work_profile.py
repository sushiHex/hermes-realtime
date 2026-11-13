"""Stage an explicit native-work profile with one reviewed read-only MCP tool.

The caller supplies the reviewed launcher identity; this helper never discovers
credentials, starts a process, or selects capabilities from a task objective.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StatusServer:
    command: Path
    sha256: str
    state_dir: Path
    python_user_base: Path | None = None


def web_evidence_selected(config: dict[str, Any]) -> bool:
    """Read an explicit deployment choice; never infer grounding from tool names."""
    settings = config.get("plugins", {}).get("entries", {}).get(
        "hermes-realtime", {}
    ).get("settings", {})
    selected = settings.get("web_evidence", False)
    if type(selected) is not bool:
        raise ValueError("native web evidence selection refused")
    if selected:
        presentation = settings.get("presentation")
        if type(presentation) is not dict:
            raise ValueError("native web evidence presentation refused")
    return selected


class NativeProfileRefusal(ValueError):
    """A fixed, content-free category the preview retains as refusal evidence."""


def memory_provider_inspection(config: dict[str, Any]) -> str:
    """Refuse an external memory provider; report the config-only result by category.

    At the pinned Hermes revision ``memory.provider`` alone selects the external
    provider, and its empty-string default means built-in only. This inspects the
    profile configuration; it is not a readback of a running gateway's memory state.
    """
    if "memory" not in config:
        return "absent"
    memory = config["memory"]
    if type(memory) is not dict:
        raise NativeProfileRefusal("memory_provider_refused")
    if "provider" not in memory:
        return "absent"
    provider = memory["provider"]
    if type(provider) is not str:
        raise NativeProfileRefusal("memory_provider_refused")
    if provider:
        raise NativeProfileRefusal("memory_provider_selected")
    return "builtin"


def _server_config(server: StatusServer) -> dict[str, Any]:
    if (
        not server.command.is_absolute()
        or not server.command.is_file()
        or re.fullmatch(r"[0-9a-f]{64}", server.sha256) is None
        or hashlib.sha256(server.command.read_bytes()).hexdigest() != server.sha256
    ):
        raise ValueError("native status launcher identity refused")
    directories = [server.state_dir]
    if server.python_user_base is not None:
        directories.append(server.python_user_base)
    if any(not path.is_absolute() or not path.is_dir() for path in directories):
        raise ValueError("native status installation refused")
    env = {"VRAM_MCP_AUDIT": "0", "VRAM_MCP_STATE_DIR": str(server.state_dir)}
    if server.python_user_base is not None:
        env["PYTHONUSERBASE"] = str(server.python_user_base)
    return {
        "command": str(server.command),
        "args": [],
        "env": env,
        "trust": "full",
        "tools": {"include": ["vram_status"], "resources": False, "prompts": False},
    }


def build_native_work_profile(base: dict[str, Any], server: StatusServer) -> dict[str, Any]:
    """Copy trusted profile settings, replacing its explicit capability selection."""
    config = copy.deepcopy(base)
    config["mcp_servers"] = {"vram": _server_config(server)}
    config.setdefault("platform_toolsets", {})["api_server"] = ["web", "mcp-vram"]
    plugins = config.setdefault("plugins", {})
    plugins["enabled"] = ["hermes-realtime"]
    entry = plugins.setdefault("entries", {}).setdefault("hermes-realtime", {})
    plugins["entries"] = {"hermes-realtime": entry}
    entry.setdefault("settings", {})["background_capabilities"] = True
    validate_native_work_profile(config, server)
    return config


def validate_native_work_profile(config: dict[str, Any], server: StatusServer) -> dict[str, Any]:
    """Fail closed on capability drift; return only non-secret staging metadata."""
    if config.get("mcp_servers") != {"vram": _server_config(server)}:
        raise ValueError("native status capability refused")
    if config.get("platform_toolsets", {}).get("api_server") != ["web", "mcp-vram"]:
        raise ValueError("native platform capability refused")
    approvals = config.get("approvals", {})
    if approvals.get("mode") != "manual" or approvals.get("unattended_mode") != "deny":
        raise ValueError("native approval policy refused")
    plugins = config.get("plugins", {})
    entries = plugins.get("entries", {})
    entry = entries.get("hermes-realtime", {})
    if (
        plugins.get("enabled") != ["hermes-realtime"]
        or set(entries) != {"hermes-realtime"}
        or entry.get("allow_tool_override") is not False
        or entry.get("settings", {}).get("background_capabilities") is not True
    ):
        raise ValueError("native plugin capability refused")
    web_evidence_selected(config)
    memory_provider_inspection(config)
    return {
        "version": 1,
        "mode": "native-work/v1",
        "lookup_policy": None,
        "toolsets": ["web", "mcp-vram"],
        "monitoring_tools": ["mcp__vram__vram_status"],
        "monitoring_scope": "gpu_and_ollama_residency",
        "launcher_sha256": server.sha256,
    }
