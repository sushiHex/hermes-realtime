"""Strict, bounded MCP warm callback for the owned lookup model.

The trusted launcher supplies an initialized, already-scoped async MCP session
factory and verifies the staged vram-mcp source bytes separately. This module
implements the public warm contract reviewed at vram-mcp revision
1f273fe41f2803df8c3887a86c4afa72f351b93c (PR #47). A revision string,
package name, or SDK name is not proof of the running server's source. The
factory must bind its transport, session initialization, and cleanup to the
passed absolute deadline; this callback never creates an unscoped client.

The factory yields a session with ``list_tools()`` and ``call_tool(name,
arguments)`` as in the Python MCP SDK. No SDK dependency or network fallback
is imported here. The owner still performs independent process, artifact, and
configured-context readback through ``start_model`` after this callback.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable
from typing import Any

from scripts.lookup_backend_model import MCP_CALLBACK_REFUSALS, StartupProfile

_OVERHEAD_SECONDS = 60.0
_MAX_WARM_SECONDS = 1800.0
_MIN_FIT_MARGIN_MB = 1024
_RESULT_BYTES = 65536
_ACTOR = "hermes-realtime"
_LOOPBACK_URL = re.compile(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})\Z")


def _warm_budget_seconds(deadline: float) -> float:
    # Compare absolute times first. Subtracting non-dyadic clocks can round an
    # exact 60-second reserve upward, then admit a warm with no real allowance.
    # Round the cutoff up and the derived wait down by one float step so neither
    # operation grants time beyond the caller's absolute deadline.
    cutoff = math.nextafter(time.monotonic() + _OVERHEAD_SECONDS, math.inf)
    if deadline <= cutoff:
        raise ValueError("mcp-budget")
    return math.nextafter(deadline - cutoff, 0.0)


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _payload(result: Any) -> dict[str, Any]:
    """Accept one bounded SDK text result; never interpret prose as an ack."""
    if _field(result, "isError") is not False:
        raise ValueError("mcp-result")
    content = _field(result, "content")
    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("mcp-result")
    item = content[0]
    body = _field(item, "text")
    if _field(item, "type") != "text" or type(body) is not str:
        raise ValueError("mcp-result")
    if len(body.encode("utf-8")) > _RESULT_BYTES:
        raise ValueError("mcp-result")
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        raise ValueError("mcp-result") from None
    if type(value) is not dict:
        raise ValueError("mcp-result")
    return value


def _allows(schema: Any, wanted: str) -> bool:
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind == wanted or (wanted == "number" and kind == "integer"):
        return True
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and any(_allows(row, wanted) for row in options):
            return True
    return False


def _check_tools(result: Any) -> None:
    entries = _field(result, "tools")
    if not isinstance(entries, list):
        raise ValueError("mcp-capability")
    names = [_field(entry, "name") for entry in entries]
    if names.count("list_loaded") != 1 or names.count("warm") != 1:
        raise ValueError("mcp-capability")
    listed = entries[names.index("list_loaded")]
    list_schema = _field(listed, "inputSchema")
    if (not isinstance(list_schema, dict) or list_schema.get("type") != "object"
            or list_schema.get("required")):
        raise ValueError("mcp-capability")
    warm = entries[names.index("warm")]
    shape = _field(warm, "inputSchema")
    if not isinstance(shape, dict) or shape.get("type") != "object":
        raise ValueError("mcp-capability")
    props = shape.get("properties")
    required = {"model": "string", "keep_alive": "string", "by": "string",
        "force": "boolean", "strict_fit": "boolean", "num_ctx": "integer",
        "num_parallel": "integer", "timeout_seconds": "number"}
    if not isinstance(props, dict) or any(
            not _allows(props.get(name), kind) for name, kind in required.items()):
        raise ValueError("mcp-capability")


def _scoped_observation(value: dict[str, Any], base_url: str) -> bool:
    observations = value.get("observations")
    if not isinstance(observations, dict):
        return False
    ollama = observations.get("ollama")
    return (isinstance(ollama, dict) and ollama.get("status") == "available"
        and ollama.get("source") == "ollama:/api/ps"
        and ollama.get("scope") == base_url)


def _check_preflight(value: dict[str, Any], base_url: str) -> None:
    if not _scoped_observation(value, base_url):
        raise ValueError("mcp-scope")
    # The caller has just started its own isolated model server. Existing
    # residency would make this a refresh with no fresh fit admission.
    if value.get("loaded") != []:
        raise ValueError("mcp-preflight")


def _check_warm(value: dict[str, Any], *, base_url: str, model: str,
                context: int, timeout: float) -> None:
    margin = value.get("fit_margin_mb")
    if (value.get("ok") is not True or value.get("outcome") != "succeeded"
            or value.get("resident") is not True or value.get("model") != model
            or value.get("keep_alive") != "-1"
            or type(value.get("num_ctx")) is not int or value["num_ctx"] != context
            or type(value.get("num_parallel")) is not int or value["num_parallel"] != 1
            or value.get("strict_fit") is not True
            or value.get("size_verified") is not True
            or type(margin) is not int or margin < _MIN_FIT_MARGIN_MB
            or value.get("coordination_warning") is not None
            or ("partial_fit" in value and value["partial_fit"] is not False)
            or (value.get("scope") is not None and value["scope"] != base_url)
            or not _scoped_observation(value, base_url)):
        raise ValueError("mcp-warm-ack")
    reported_timeout: Any = value.get("timeout_seconds")
    if (type(reported_timeout) not in (int, float)
            or not math.isfinite(reported_timeout)
            or reported_timeout != timeout):
        raise ValueError("mcp-warm-ack")


def make_strict_mcp_loader(session_factory: Callable[..., Any]) -> Callable[..., Any]:
    """Return the ``McpModelLoader`` callback for a trusted scoped factory.

    ``session_factory(base_url, deadline)`` is an async context manager. It
    must establish exactly that local Ollama endpoint in its own source-bound
    MCP process and use the deadline for transport and cleanup. The source
    factory sets a fit margin of at least 1024 MB; a lower reported margin is
    refused. Warm receives
    only the *remaining* budget less a fixed reserve for preflight, MCP
    response, configured-context readback, and cleanup. It never retries.
    """
    if not callable(session_factory):
        raise ValueError("mcp-session-factory")
    invoked = False

    async def perform(*, base_url: str, model: str, deadline: float,
                      startup_profile: StartupProfile, evidence: dict[str, Any]) -> bool:
        nonlocal invoked
        if invoked:
            raise ValueError("mcp-reused")
        port = _LOOPBACK_URL.fullmatch(base_url) if type(base_url) is str else None
        if (port is None or int(port.group(1)) > 65535 or int(port.group(1)) == 11434
                or type(model) is not str or not model
                or type(deadline) not in (int, float) or not math.isfinite(deadline)
                or type(startup_profile) is not StartupProfile):
            raise ValueError("mcp-input")
        invoked = True
        evidence["phase"] = "budget"
        _warm_budget_seconds(deadline)
        async with asyncio.timeout_at(deadline):
            async with session_factory(base_url, deadline) as session:
                evidence["phase"] = "capability"
                _check_tools(await session.list_tools())
                evidence["phase"] = "preflight"
                evidence["read_only_tool_calls"] += 1
                preflight = _payload(await session.call_tool("list_loaded", {}))
                _check_preflight(preflight, base_url)
                timeout = min(_MAX_WARM_SECONDS, _warm_budget_seconds(deadline))
                arguments = {"model": model, "keep_alive": "-1", "by": _ACTOR,
                    "force": False, "strict_fit": True,
                    "num_ctx": startup_profile.context_tokens,
                    "num_parallel": 1, "timeout_seconds": timeout}
                evidence["phase"] = "warm"
                evidence["warm_tool_calls"] += 1
                reply = await session.call_tool("warm", arguments)
                evidence["phase"] = "acknowledgment"
                result = _payload(reply)
                _check_warm(result, base_url=base_url, model=model,
                    context=startup_profile.context_tokens, timeout=timeout)
                return True

    async def load(*, base_url: str, model: str, deadline: float,
                   startup_profile: StartupProfile) -> bool:
        evidence: dict[str, Any] = {"phase": "input", "accepted": False,
            "failure_category": "none", "read_only_tool_calls": 0, "warm_tool_calls": 0}
        try:
            accepted = await perform(base_url=base_url, model=model, deadline=deadline,
                                     startup_profile=startup_profile, evidence=evidence)
            evidence["accepted"] = accepted
            return accepted
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                category = "mcp-cancelled"
            elif isinstance(error, TimeoutError):
                category = "mcp-timeout"
            elif (type(error) is ValueError and len(error.args) == 1
                    and type(error.args[0]) is str and error.args[0] in MCP_CALLBACK_REFUSALS):
                category = error.args[0]
            else:
                category = "mcp-other"
            evidence["failure_category"] = category
            raise
        finally:
            print("[lookup-mcp] " + json.dumps(evidence, separators=(",", ":")), flush=True)

    return load
