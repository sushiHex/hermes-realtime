"""Owned native-policy gateway; no provider/model action before main invocation.

Native plugin lifecycle owns tool policy. Harness binds Exa native search/extract
requests, independent transport counts, returned sources and response bounds.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from contextlib import nullcontext, redirect_stdout, suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
INSTALL: Path = ROOT

_DIAGNOSTIC_ENUMS = {
    "category": {"refused", "presented"},
    "stage": {
        "native-response", "source", "retrieval", "catalog", "route", "worker-loop",
        "provider", "response-envelope", "json", "fields", "content", "source-selection",
        "source-id-type", "source-id-range", "source-id-unique", "inline-links",
        "terminal-contract", "publication", "catalog-publication", "complete",
    },
    "rejection": {
        "native-policy-error", "generation-unavailable", "budget-unavailable", "budget-replaced",
        "batch-envelope", "call-envelope", "search-arguments", "url-membership", "batch-budget",
        "call-json", "retrieval-required", "other",
    },
    "failure_kind": {
        "ValueError", "RuntimeError", "JSONDecodeError", "TimeoutError", "CancelledError",
        "APIConnectionError", "APITimeoutError", "APIStatusError", "BadRequestError",
        "AuthenticationError", "PermissionDeniedError", "RateLimitError", "InternalServerError",
        "other",
    },
}
_DIAGNOSTIC_COUNTS = {
    "calls", "wire_requests", "catalog_count", "selected_source_count", "rendered_detail_length",
}
_DIAGNOSTIC_FLAGS = {"local_http_settled"}
for _prefix in ("source", "response"):
    for _name in ("kind", "reasoning_content_kind", "reasoning_kind"):
        _DIAGNOSTIC_ENUMS[f"{_prefix}_{_name}"] = {"string", "none", "other"}
    _DIAGNOSTIC_ENUMS[f"{_prefix}_finish_reason"] = {
        "stop", "length", "tool_calls", "content_filter", "function_call", "missing", "other",
    }
    _DIAGNOSTIC_COUNTS.update(f"{_prefix}_{name}" for name in (
        "choices_count", "length", "reasoning_content_length", "reasoning_length",
    ))
    _DIAGNOSTIC_FLAGS.update(f"{_prefix}_{name}" for name in (
        "tool_calls", "refusal", "parsed_present",
    ))


def _policy_diagnostic(value: Any) -> Any:
    """Project fixed scalar facts, never arbitrary native output or exception text."""
    if type(value) is not dict or not {"category", "stage"} <= value.keys():
        raise ValueError("diagnostic_schema")
    result = {}
    for key, allowed in _DIAGNOSTIC_ENUMS.items():
        if key not in value:
            continue
        item = value[key]
        if type(item) is not str:
            raise ValueError("diagnostic_schema")
        if item not in allowed:
            if key not in {"failure_kind", "rejection"}:
                raise ValueError("diagnostic_schema")
            item = "other"
        result[key] = item
    for key in _DIAGNOSTIC_COUNTS & value.keys():
        item = value[key]
        if type(item) is not int or not 0 <= item <= 1000000000:
            raise ValueError("diagnostic_schema")
        result[key] = item
    for key in _DIAGNOSTIC_FLAGS & value.keys():
        if type(value[key]) is not bool:
            raise ValueError("diagnostic_schema")
        result[key] = value[key]
    return result


def _diagnostic_pairs(items: Any) -> Any:
    value = dict(items)
    if len(value) != len(items):
        raise ValueError("diagnostic_duplicate_key")
    return value


class PolicyDiagnosticTee:
    """Retain at most sixteen scalar native markers; delegate ordinary stream behavior."""

    def __init__(self, delegate: Any, witness: Any, save: Any) -> None:
        self._delegate, self._save = delegate, save
        self._lock = threading.RLock()
        self._lines = threading.local()
        self._facts = {"version": 1, "events": [], "observed": 0, "invalid": 0,
                       "oversized": 0, "truncated": 0, "saveFailures": 0}
        witness["policyDiagnostics"] = self._facts

    def __getattr__(self, name: Any) -> Any:
        return getattr(self._delegate, name)

    def _increment(self, key: str) -> None:
        self._facts[key] = min(1000000, self._facts[key] + 1)

    def _persist(self) -> None:
        try:
            self._save()
        except Exception:
            self._increment("saveFailures")

    def _line(self, line: str) -> None:
        prefix = "[lookup-presentation] "
        if not line.startswith(prefix):
            return
        try:
            event = _policy_diagnostic(json.loads(
                line[len(prefix):], object_pairs_hook=_diagnostic_pairs,
            ))
        except (ValueError, TypeError, RecursionError):
            self._increment("invalid")
        else:
            self._increment("observed")
            if len(self._facts["events"]) < 16:
                self._facts["events"].append(event)
            else:
                self._increment("truncated")
        self._persist()

    def write(self, text: str) -> Any:
        with self._lock:
            result = self._delegate.write(text)
            if type(result) is not int or not 0 <= result <= len(text):
                return result
            pending = getattr(self._lines, "pending", "")
            dropping = getattr(self._lines, "dropping", False)
            start = 0
            while start < result:
                end = text.find("\n", start, result)
                newline = end != -1
                if not newline:
                    end = result
                if not dropping:
                    if len(pending) + end - start > 8192:
                        pending, dropping = "", True
                        self._increment("oversized")
                        self._persist()
                    else:
                        pending += text[start:end]
                if newline:
                    if not dropping:
                        self._line(pending)
                    pending, dropping = "", False
                start = end + 1
            self._lines.pending, self._lines.dropping = pending, dropping
            return result

    def flush(self) -> Any:
        with self._lock:
            return self._delegate.flush()


def read_bounded_decoded_response(response: Any) -> Any:
    try:
        body = response.raw.read(65537, decode_content=True)
    finally:
        response.close()
    if len(body) <= 65536:
        response._content = body
        response._content_consumed = True
    return body


def denial_caller_category() -> Any:
    """Bounded diagnostic frame labels only; never permission to forward a request."""
    categories = {
        str(INSTALL / "agent/model_metadata.py")
        .replace("\\", "/")
        .casefold(): "metadata_frame_seen",
        str(INSTALL / "agent/usage_pricing.py").replace("\\", "/").casefold(): "pricing_frame_seen",
        str(INSTALL / "hermes_cli/models.py").replace("\\", "/").casefold(): "pricing_frame_seen",
        str(INSTALL / "plugins/web/keyless_mcp.py")
        .replace("\\", "/")
        .casefold(): "native_web_frame_seen",
    }
    frame: Any = sys._getframe(1)
    for _ in range(16):
        if frame is None:
            break
        category = categories.get(frame.f_code.co_filename.replace("\\", "/").casefold())
        if category is not None:
            return category
        frame = frame.f_back
    return "unknown"


def metadata_urls_from_config(config: Any) -> Any:
    """Only the already selected native/presentation model route can supply metadata."""
    from urllib.parse import urlsplit

    model = config["model"]
    presentation = config["plugins"]["entries"]["ordinary-lookup-experimental"]["settings"][
        "presentation"
    ]
    base = model["base_url"]
    if (
        model.get("provider") != "custom"
        or presentation.get("provider") != model["provider"]
        or presentation.get("model") != model["default"]
        or presentation.get("base_url") != base
    ):
        raise RuntimeError("metadata_route_mismatch")
    parsed = urlsplit(base)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("metadata_route_refused")
    normalized = base.rstrip("/")
    alternate = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    return frozenset((normalized + "/models", alternate + "/models"))


def bounded_exa_send(
    original: Any,
    witness: Any,
    allowed_urls: Any,
    save: Any,
    state: Any,
    metadata_urls: Any = frozenset(),
) -> Any:
    lock = threading.RLock()

    def send(session: Any, request: Any, **kwargs: Any) -> Any:
        receipt: dict[str, Any] = {}
        forwarded = False
        phase = "request_decode"
        refusal: dict[str, Any] | None = None
        try:
            # Unknown traffic still refuses. This separate read-only lane is never active
            # within a provider invocation and never accepts the Exa endpoint itself.
            if (
                getattr(state, "search", None) is None
                and getattr(state, "extract", None) is None
                and request.url != "https://mcp.exa.ai/mcp"
            ):
                phase = "metadata_binding"
                with lock:
                    if (
                        request.url not in metadata_urls
                        or request.method != "GET"
                        or request.body is not None
                        or witness.get("metadataRequests", 0) >= 2
                    ):
                        raise RuntimeError("metadata_request_refused")
                    adapter = session.get_adapter(request.url)
                    if adapter.max_retries.total != 0:
                        raise RuntimeError("metadata_retry_refused")
                    witness["metadataRequests"] = witness.get("metadataRequests", 0) + 1
                    receipt["phase"] = "metadata"
                    forwarded = True
                kwargs.update(allow_redirects=False, stream=True, timeout=(5, 10), proxies={})
                phase = "metadata_send"
                response = original(session, request, **kwargs)
                phase = "metadata_read"
                raw = read_bounded_decoded_response(response)
                receipt.update(responseBytes=len(raw), httpStatus=response.status_code)
                if len(raw) > 65536:
                    raise RuntimeError("metadata_response_bound")
                if 300 <= response.status_code < 400:
                    raise RuntimeError("metadata_redirect_refused")
                return response
            body = json.loads(request.body)
            phase = "request_binding"
            params = body.get("params") or {}
            args = params.get("arguments") or {}
            with lock:
                search = getattr(state, "search", None)
                extract = getattr(state, "extract", None)
                if (
                    request.method != "POST"
                    or request.url != "https://mcp.exa.ai/mcp"
                    or body.get("jsonrpc") != "2.0"
                    or body.get("id") != 1
                    or body.get("method") != "tools/call"
                    or kwargs.get("timeout") != 30
                ):
                    raise RuntimeError("exa_request_binding_refused")
                if params.get("name") == "web_search_exa":
                    if (
                        search is None
                        or search["spent"]
                        or witness["searchRequests"] >= 5
                        or args != {"query": search["query"], "numResults": search["limit"]}
                    ):
                        raise RuntimeError("exa_search_binding_refused")
                    search["spent"] = True
                    witness["searchRequests"] += 1
                    receipt["phase"] = "search"
                elif params.get("name") == "web_fetch_exa":
                    urls = args.get("urls")
                    if (
                        extract is None
                        or witness["extractRequests"] >= 3
                        or type(urls) is not list
                        or len(urls) != 1
                        or type(urls[0]) is not str
                        or args != {"urls": urls}
                        or urls[0] not in allowed_urls
                        or urls[0] not in extract["urls"]
                        or urls[0] in extract["spent"]
                    ):
                        raise RuntimeError("exa_extract_binding_refused")
                    extract["spent"].add(urls[0])
                    witness["extractRequests"] += 1
                    receipt["phase"] = "extract"
                else:
                    raise RuntimeError("exa_request_binding_refused")
                forwarded = True
            kwargs.update(allow_redirects=False, stream=True)
            phase = "transport_send"
            response = original(session, request, **kwargs)
            phase = "response_read"
            raw = read_bounded_decoded_response(response)
            phase = "response_classify"
            receipt.update(responseBytes=len(raw), httpStatus=response.status_code)
            phase = "response_bound"
            if len(raw) > 65536:
                raise RuntimeError("exa_response_bound")
            phase = "response_status"
            if not 200 <= response.status_code < 300:
                raise RuntimeError("exa_http_refused")
            return response
        except BaseException as error:
            with lock:
                witness["transportRefused"] = True
                witness["transportDenialCount"] = min(9, witness.get("transportDenialCount", 0) + 1)
                denials = witness.setdefault("transportDenials", [])
                witness.setdefault("transportDenialsTruncated", False)
                if len(denials) < 8:
                    fixed = {
                        "exa_request_binding_refused",
                        "exa_search_binding_refused",
                        "exa_extract_binding_refused",
                        "exa_response_bound",
                        "exa_http_refused",
                        "metadata_request_refused",
                        "metadata_retry_refused",
                        "metadata_response_bound",
                        "metadata_redirect_refused",
                    }
                    category = (
                        str(error)
                        if type(error) is RuntimeError and str(error) in fixed
                        else "request_decode_error"
                        if phase == "request_decode"
                        else "transport_operation_error"
                    )
                    refusal = {
                        "phase": phase,
                        "category": category,
                        "callerCategory": denial_caller_category(),
                    }
                    denials.append(refusal)
                else:
                    witness["transportDenialsTruncated"] = True
                    refusal = {
                        "phase": phase,
                        "category": "denial_overflow",
                        "callerCategory": "unknown",
                    }
            raise
        finally:
            with lock:
                if forwarded:
                    witness["transportResponses"].append(receipt)
                try:
                    save()
                finally:
                    if refusal is not None:
                        marker = {
                            **refusal,
                            "denialCount": witness["transportDenialCount"],
                            "searchRequests": witness["searchRequests"],
                            "extractRequests": witness["extractRequests"],
                            "metadataRequests": witness.get("metadataRequests", 0),
                            "forwarded": forwarded,
                        }
                        with suppress(OSError):
                            print(
                                "[lookup-transport-refusal] "
                                + json.dumps(marker, separators=(",", ":"), sort_keys=True),
                                flush=True,
                            )

    return send


def _model_capture_module() -> Any:
    """Load the adjacent verified helper even under the gateway's -I launch."""
    spec = importlib.util.spec_from_file_location(
        "owned_model_capture", ROOT / "lookup_model_capture.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("capture_helper_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> Any:
    global INSTALL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--model-response-capture", default="off",
                        choices=("off", "metadata", "controlled-public"))
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    INSTALL = Path(manifest["hermes_source"]).resolve()
    output = manifest_path.parent
    package = Path(manifest["packageRoot"]).resolve()
    if package != (Path(manifest["runtime_target"]) / "hermes_realtime").resolve():
        raise RuntimeError("separate_payload_required")
    record = Path(manifest["recordPath"])
    if hashlib.sha256(record.read_bytes()).hexdigest() != manifest["recordSha256"]:
        raise RuntimeError("record_drift")
    sys.path[:0] = [str(package.parent), str(INSTALL)]
    import hermes_realtime

    if Path(hermes_realtime.__file__).resolve().parent != package:
        raise RuntimeError("import_origin_mismatch")
    sys.argv = ["hermes", "gateway", "run"]
    import tools.web_tools as web  # type: ignore[import-not-found]
    from agent.web_search_registry import get_provider  # type: ignore[import-not-found]
    from hermes_cli.main import main as gateway_main  # type: ignore[import-not-found]
    from plugins.web import keyless_mcp  # type: ignore[import-not-found]

    witness: dict[str, Any] = {
        "harnessToolBudgetEnforcement": False,
        "nativePolicyRequested": True,
        "searchRequests": 0,
        "extractRequests": 0,
        "transportResponses": [],
        "transportRefused": False,
        "globalPopenUnchanged": True,
    }
    target = output / "gateway-witness.private.json"
    save_lock = threading.Lock()
    original = subprocess.Popen

    def save() -> Any:
        with save_lock:
            target.write_text(json.dumps(witness), encoding="utf-8")

    web._ensure_web_plugins_loaded()
    search, extract = get_provider("exa"), get_provider("exa")
    if not (
        web._get_search_backend() == "exa"
        and search
        and search.supports_search()
        and search.is_keyless_available()
        and web._get_extract_backend() == "exa"
        and extract
        and extract.supports_extract()
        and extract.is_keyless_available()
        and keyless_mcp.use_keyless("exa", "")
        and keyless_mcp._ring_order("exa") == ["exa"]
        and not web._keyless_rescue_enabled()
        and web._load_web_config().get("cache_enabled") is False
    ):
        raise RuntimeError("web_config_refused")
    # Passive provider projections support the unchanged wire-level source binding.
    # No query count, tool call count or extraction batch policy is enforced here.
    import yaml  # type: ignore[import-untyped]

    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    with config_path.open("rb") as stream:
        config_raw = stream.read(65537)
    if len(config_raw) > 65536:
        raise RuntimeError("metadata_config_bound")
    config = yaml.safe_load(config_raw)
    metadata_urls = metadata_urls_from_config(config)
    capture_context: Any = nullcontext()
    if args.model_response_capture != "off":
        capture_module = _model_capture_module()

        def capture_updated(summary: Any) -> None:
            witness["modelResponseCapture"] = summary
            save()

        recorder = capture_module.ModelResponseCapture(
            output, args.model_response_capture, manifest, config, on_update=capture_updated,
        )
        capture_context = recorder.installed()
    returned_urls: set[str] = set()
    allowed_urls: set[str] = set()
    projection_lock = threading.Lock()
    original_search, original_extract = search.search, extract.extract
    wire_state = threading.local()

    def observe_search(query: Any, limit: Any = 5) -> Any:
        wire_state.search = {"query": query, "limit": max(1, int(limit)), "spent": False}
        try:
            result = original_search(query, limit)
        finally:
            wire_state.search = None
        if type(result) is dict and result.get("success") is True:
            data = result.get("data")
            rows = data.get("web") if type(data) is dict else None
            if type(rows) is list:
                with projection_lock:
                    returned_urls.update(
                        row["url"]
                        for row in rows
                        if type(row) is dict and type(row.get("url")) is str
                    )
        return result

    def observe_extract(urls: Any, **kwargs: Any) -> Any:
        if type(urls) is list:
            with projection_lock:
                allowed_urls.update(
                    url for url in urls if type(url) is str and url in returned_urls
                )
        wire_state.extract = {"urls": set(urls), "spent": set()}
        try:
            return original_extract(urls, **kwargs)
        finally:
            wire_state.extract = None

    search.search, extract.extract = observe_search, observe_extract
    import requests  # type: ignore[import-untyped]

    requests.sessions.Session.send = bounded_exa_send(
        requests.sessions.Session.send, witness, allowed_urls, save, wire_state, metadata_urls
    )
    witness.update(
        searchProviderReady=True,
        extractProviderReady=True,
        interpreterExact=Path(sys.executable).resolve()
        == Path(manifest["hermes_python"]).resolve(),
        packageExact=Path(hermes_realtime.__file__).resolve().parent == package,
        recordSha256=hashlib.sha256(record.read_bytes()).hexdigest(),
        configSha256=hashlib.sha256(
            (Path(os.environ["HERMES_HOME"]) / "config.yaml").read_bytes()
        ).hexdigest(),
        globalPopenUnchanged=subprocess.Popen is original,
    )
    diagnostics = PolicyDiagnosticTee(sys.stdout, witness, save)
    save()
    with redirect_stdout(diagnostics), capture_context:
        gateway_main()


if __name__ == "__main__":
    main()
