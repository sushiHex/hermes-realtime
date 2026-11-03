"""Native-work profile staging never selects capabilities from objective text."""

import copy
import hashlib

import pytest

from scripts.native_work_profile import (
    StatusServer,
    build_native_work_profile,
    validate_native_work_profile,
)


@pytest.fixture
def server(tmp_path):
    command = tmp_path / "vram-mcp.exe"
    command.write_bytes(b"synthetic-reviewed-launcher")
    return StatusServer(command, hashlib.sha256(command.read_bytes()).hexdigest(), tmp_path)


def base_config():
    return {
        "approvals": {"mode": "manual", "unattended_mode": "deny"},
        "platform_toolsets": {"api_server": ["web"]},
        "model": {"default": "synthetic-model"},
        "plugins": {
            "enabled": ["hermes-realtime"],
            "entries": {"hermes-realtime": {"allow_tool_override": False}},
        },
    }


def test_native_profile_enables_only_existing_status_and_preserves_approvals(server):
    original = base_config()
    before = copy.deepcopy(original)
    config = build_native_work_profile(original, server)
    assert original == before
    assert config["approvals"] == original["approvals"]
    assert config["model"] == original["model"]
    assert config["platform_toolsets"]["api_server"] == ["web", "mcp-vram"]
    assert config["mcp_servers"] == {
        "vram": {
            "command": str(server.command),
            "args": [],
            "env": {"VRAM_MCP_AUDIT": "0", "VRAM_MCP_STATE_DIR": str(server.state_dir)},
            "trust": "full",
            "tools": {"include": ["vram_status"], "resources": False, "prompts": False},
        }
    }
    assert (
        config["plugins"]["entries"]["hermes-realtime"]["settings"]["background_capabilities"]
        is True
    )
    assert validate_native_work_profile(config, server)["lookup_policy"] is None


def test_existing_python_installation_binding_is_explicit_and_private(server, tmp_path):
    bound = StatusServer(server.command, server.sha256, server.state_dir, tmp_path)
    config = build_native_work_profile(base_config(), bound)
    assert config["mcp_servers"]["vram"]["env"] == {
        "VRAM_MCP_AUDIT": "0",
        "VRAM_MCP_STATE_DIR": str(server.state_dir),
        "PYTHONUSERBASE": str(tmp_path),
    }
    manifest = validate_native_work_profile(config, bound)
    assert str(tmp_path) not in repr(manifest)
    missing = StatusServer(server.command, server.sha256, server.state_dir, tmp_path / "missing")
    with pytest.raises(ValueError, match="installation refused"):
        build_native_work_profile(base_config(), missing)
    missing_state = StatusServer(server.command, server.sha256, tmp_path / "missing-state")
    with pytest.raises(ValueError, match="installation refused"):
        build_native_work_profile(base_config(), missing_state)


def test_lookup_plugin_is_removed_only_from_the_new_profile_copy(server):
    original = base_config()
    original["plugins"]["enabled"].append("ordinary-lookup-experimental")
    original["plugins"]["entries"]["ordinary-lookup-experimental"] = {"settings": {}}
    config = build_native_work_profile(original, server)
    assert "ordinary-lookup-experimental" in original["plugins"]["enabled"]
    assert config["plugins"]["enabled"] == ["hermes-realtime"]
    assert set(config["plugins"]["entries"]) == {"hermes-realtime"}


@pytest.mark.parametrize("selection", ["true", 1, {}, None])
def test_native_web_evidence_requires_an_explicit_boolean(server, selection):
    original = base_config()
    original["plugins"]["entries"]["hermes-realtime"]["settings"] = {
        "web_evidence": selection
    }
    with pytest.raises(ValueError, match="evidence"):
        build_native_work_profile(original, server)


def test_native_web_evidence_requires_a_reviewed_presentation_route(server):
    original = base_config()
    original["plugins"]["entries"]["hermes-realtime"]["settings"] = {"web_evidence": True}
    with pytest.raises(ValueError, match="evidence"):
        build_native_work_profile(original, server)


def test_explicit_web_presentation_preserves_native_capabilities(server):
    original = base_config()
    settings = {
        "web_evidence": True,
        "presentation": {
            "provider": "synthetic", "base_url": "https://example.invalid/v1",
            "model": "synthetic-model", "api_key_env": "SYNTHETIC_KEY",
        },
    }
    original["plugins"]["entries"]["hermes-realtime"]["settings"] = settings
    before = copy.deepcopy(original)
    config = build_native_work_profile(original, server)
    assert original == before
    staged = config["plugins"]["entries"]["hermes-realtime"]["settings"]
    assert staged == settings | {"background_capabilities": True}
    assert config["approvals"] == original["approvals"]
    assert config["platform_toolsets"]["api_server"] == ["web", "mcp-vram"]
    assert validate_native_work_profile(config, server)["lookup_policy"] is None


@pytest.mark.parametrize(
    "change",
    [
        "extra_server",
        "extra_tool",
        "resources",
        "prompts",
        "terminal",
        "approval",
        "source",
        "plugin",
        "override",
        "env",
        "state_dir",
    ],
)
def test_profile_refuses_capability_or_source_drift(server, change):
    config = build_native_work_profile(base_config(), server)
    if change == "extra_server":
        config["mcp_servers"]["foreign"] = {}
    elif change == "extra_tool":
        config["mcp_servers"]["vram"]["tools"]["include"].append("warm")
    elif change in {"resources", "prompts"}:
        config["mcp_servers"]["vram"]["tools"][change] = True
    elif change == "terminal":
        config["platform_toolsets"]["api_server"].append("terminal")
    elif change == "approval":
        config["approvals"]["unattended_mode"] = "allow"
    elif change == "plugin":
        config["plugins"]["enabled"].append("ordinary-lookup-experimental")
    elif change == "override":
        config["plugins"]["entries"]["hermes-realtime"]["allow_tool_override"] = True
    elif change == "env":
        config["mcp_servers"]["vram"]["env"]["VRAM_MCP_AUDIT"] = "1"
    elif change == "state_dir":
        config["mcp_servers"]["vram"]["env"]["VRAM_MCP_STATE_DIR"] = "foreign-domain"
    else:
        server.command.write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_native_work_profile(config, server)
