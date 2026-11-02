"""Native tool inventory is descriptive data, never foreground execution authority."""

import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hermes_realtime.integration.api import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.integration.tool_catalog import TOOL_CATALOG_PATH, install_tool_catalog
from tests.integration.test_hermes_api_session import _capabilities_response


@pytest.mark.parametrize("change", [
    {"object": "other.catalog"}, {"platform": "cli"}, {"version": 2},
    {"version": True}, {"unexpected": "ignored must not be accepted"},
])
def test_catalog_identity_is_exact(change):
    from hermes_realtime.integration.tool_catalog import parse_tool_catalog

    payload = {"object": "hermes.realtime.tools", "platform": "api_server",
               "version": 1, "tools": ["web_search"]} | change
    with pytest.raises(ValueError, match="tool catalog"):
        parse_tool_catalog(payload)


def test_native_reader_uses_the_plain_runs_profile_and_resolver(monkeypatch):
    import sys
    from types import ModuleType

    from hermes_realtime.integration.tool_catalog import _native_tool_names

    config = object()
    for name in ("gateway", "gateway.run", "hermes_cli", "hermes_cli.tools_config", "model_tools"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["gateway.run"]._load_gateway_config = lambda: config

    def enabled(actual, platform):
        assert actual is config and platform == "api_server"
        return {"mcp-vram", "web"}

    def definitions(*, enabled_toolsets, quiet_mode, skip_tool_search_assembly):
        assert enabled_toolsets == ["mcp-vram", "web"]
        assert quiet_mode and skip_tool_search_assembly
        return [{"function": {"name": "mcp__vram__vram_status"}}]

    sys.modules["hermes_cli.tools_config"]._get_platform_tools = enabled
    sys.modules["model_tools"].get_tool_definitions = definitions
    assert _native_tool_names() == ("mcp__vram__vram_status",)


@pytest.mark.parametrize("enabled", [False, True, "true", 1])
def test_catalog_registration_requires_explicit_plugin_opt_in(enabled):
    from hermes_realtime.integration.tool_catalog import register_tool_catalog

    calls = []
    context = SimpleNamespace(
        get_config=lambda *_: enabled,
        register_platform_handler=lambda *args: calls.append(args),
    )
    register_tool_catalog(context)
    assert len(calls) == (1 if enabled is True else 0)
    if calls:
        assert calls[0] == ("api_server", install_tool_catalog)


@pytest.mark.asyncio
async def test_catalog_requires_native_auth_before_reading_tools():
    calls = []
    app = web.Application()
    adapter = SimpleNamespace(_check_auth=lambda _: web.Response(status=401))
    install_tool_catalog(app, adapter, reader=lambda: calls.append(True) or ("web_search",))
    async with TestClient(TestServer(app)) as client:
        response = await client.get(TOOL_CATALOG_PATH)
        assert response.status == 401
    assert calls == []


@pytest.mark.asyncio
async def test_session_discovers_resolved_tools_without_changing_run_selection():
    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    names = ("web_search", "mcp__vram__vram_status")
    install_tool_catalog(app, SimpleNamespace(_check_auth=lambda _: None), reader=lambda: names)
    async with TestServer(app) as server:
        config = HermesApiConfig(
            base_url=str(server.make_url("/")).rstrip("/"),
            bearer="synthetic-catalog-bearer-32-characters",
            discover_tools=True,
        )
        session = HermesApiTaskSession(config=config, session_id="session_catalog")
        assert session.configured_tools is None
        try:
            await session.start()
            assert session.configured_tools == tuple(sorted(names))
            assert config.lookup_policy is None
        finally:
            await session.close()
        assert session.configured_tools is None


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("web_search", "web_search"), ("private\ntext",), (1,)])
async def test_catalog_refuses_invalid_inventory_without_echoing_it(names, capsys):
    app = web.Application()
    install_tool_catalog(app, SimpleNamespace(_check_auth=lambda _: None), reader=lambda: names)
    async with TestClient(TestServer(app)) as client:
        response = await client.get(TOOL_CATALOG_PATH)
        assert response.status == 503
        assert await response.json() == {"error": "tool_catalog_unavailable"}
    output = capsys.readouterr().out
    assert "private" not in output
    assert json.loads(output.split(" ", 1)[1])["category"] == "discovery_refused"


@pytest.mark.asyncio
async def test_session_rejects_unbound_catalog_identity():
    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)

    async def wrong_identity(_):
        return web.json_response({
            "object": "other.catalog", "version": 1, "platform": "api_server",
            "tools": ["web_search"],
        })

    app.router.add_get(TOOL_CATALOG_PATH, wrong_identity)
    async with TestServer(app) as server:
        session = HermesApiTaskSession(config=HermesApiConfig(
            base_url=str(server.make_url("/")).rstrip("/"),
            bearer="synthetic-catalog-bearer-32-characters", discover_tools=True,
        ), session_id="session_catalog")
        try:
            with pytest.raises(RuntimeError, match="tool catalog"):
                await session.start()
            assert session.configured_tools is None
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_session_refuses_unsuccessful_catalog_response():
    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)

    async def unavailable(_):
        return web.json_response({
            "object": "hermes.realtime.tools", "version": 1, "platform": "api_server",
            "tools": ["web_search"],
        }, status=503)

    app.router.add_get(TOOL_CATALOG_PATH, unavailable)
    async with TestServer(app) as server:
        session = HermesApiTaskSession(config=HermesApiConfig(
            base_url=str(server.make_url("/")).rstrip("/"),
            bearer="synthetic-catalog-bearer-32-characters", discover_tools=True,
        ), session_id="session_catalog")
        try:
            with pytest.raises(RuntimeError, match="tool catalog"):
                await session.start()
            assert session.configured_tools is None
        finally:
            await session.close()
