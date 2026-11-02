"""Authenticated descriptive inventory of the native background tool surface."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any, cast

from aiohttp import web

TOOL_CATALOG_PATH = "/hermes-realtime/v1/tools"
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}\Z")


def validate_tool_names(value: object) -> tuple[str, ...]:
    if type(value) is not list and type(value) is not tuple:
        raise ValueError("invalid tool inventory")
    if len(value) > 512:
        raise ValueError("invalid tool inventory")
    if any(type(name) is not str or _NAME.fullmatch(name) is None for name in value):
        raise ValueError("invalid tool inventory")
    if len(set(value)) != len(value):
        raise ValueError("invalid tool inventory")
    return tuple(sorted(value))


def parse_tool_catalog(value: object) -> tuple[str, ...]:
    if (type(value) is not dict
            or set(value) != {"object", "version", "platform", "tools"}
            or value.get("object") != "hermes.realtime.tools"
            or type(value.get("version")) is not int or value["version"] != 1
            or value.get("platform") != "api_server"):
        raise ValueError("invalid tool catalog")
    return validate_tool_names(value["tools"])


def _native_tool_names() -> tuple[str, ...]:
    # Match the plain Runs constructor's native profile and tool resolution.
    # Imports remain inside Hermes; the realtime wheel adds no Hermes dependency.
    from gateway.run import _load_gateway_config  # type: ignore[import-not-found]
    from hermes_cli.tools_config import _get_platform_tools  # type: ignore[import-not-found]
    from model_tools import get_tool_definitions  # type: ignore[import-not-found]

    enabled = sorted(_get_platform_tools(_load_gateway_config(), "api_server"))
    definitions = get_tool_definitions(
        enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True,
    )
    return validate_tool_names([item["function"]["name"] for item in definitions])


def install_tool_catalog(
    app: web.Application,
    adapter: Any,
    *,
    reader: Callable[[], tuple[str, ...]] = _native_tool_names,
) -> None:
    authorize = getattr(adapter, "_check_auth", None)
    if not callable(authorize):
        raise ValueError("native tool catalog requires native authentication")

    async def catalog(request: web.Request) -> web.Response:
        category = None
        try:
            refused = authorize(request)
            if refused is not None:
                category = "authorization_refused"
                return cast(web.Response, refused)
            tools = validate_tool_names(await asyncio.to_thread(reader))
            return web.json_response({
                "object": "hermes.realtime.tools", "version": 1,
                "platform": "api_server", "tools": tools,
            })
        except Exception:
            category = "discovery_refused"
            return web.json_response({"error": "tool_catalog_unavailable"}, status=503)
        finally:
            if category is not None:
                print("[hermes-tool-catalog] " + json.dumps(
                    {"category": category, "version": 1}, separators=(",", ":")), flush=True)

    app.router.add_get(TOOL_CATALOG_PATH, catalog)


def register_tool_catalog(context: object) -> None:
    get_config = getattr(context, "get_config", None)
    if not callable(get_config) or get_config("background_capabilities", False) is not True:
        return
    register = getattr(context, "register_platform_handler", None)
    if not callable(register):
        raise RuntimeError("native tool catalog platform registration unavailable")
    register("api_server", install_tool_catalog)
