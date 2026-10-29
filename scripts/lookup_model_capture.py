"""Optional bounded observation of responses from one owned model endpoint.

No request text, headers or tool arguments are retained. Raw assistant content is
available only under the explicit controlled-public mode, never by environment.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_EVENTS = 8
MAX_CONTENT_BYTES = 16384
MAX_RESPONSE_BYTES = 65536
MAX_FILE_BYTES = 150000
_MODES = frozenset({"off", "metadata", "controlled-public"})
_FINISH = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_REASONS = frozenset({
    "artifact_bound", "write_failed", "witness_write_failed", "streaming_response",
    "redirected_response", "unread_response", "response_bound", "event_bound",
    "content_bound", "observation_failed", "wrapper_replaced",
})


def validate_mode(mode: Any) -> str:
    if type(mode) is not str or mode not in _MODES:
        raise ValueError("capture_mode")
    return mode


def capture_endpoint(mode: Any, manifest: Any, config: Any) -> str | None:
    """Bind only the configured native/presentation route to the owned model port."""
    if validate_mode(mode) == "off":
        return None
    try:
        port = manifest["model_port"]
        model = config["model"]
        presentation = config["plugins"]["entries"]["ordinary-lookup-experimental"][
            "settings"]["presentation"]
        base = model["base_url"]
        if (type(port) is not int or not 1 <= port <= 65535 or port == 11434
                or type(base) is not str or model.get("provider") != "custom"
                or presentation.get("provider") != model["provider"]
                or presentation.get("base_url") != base
                or presentation.get("model") != model["default"]):
            raise ValueError("capture_route")
        parsed = urlsplit(base)
        if (parsed.scheme != "http" or parsed.netloc != f"127.0.0.1:{port}"
                or parsed.query or parsed.fragment
                or re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*/?", parsed.path) is None):
            raise ValueError("capture_route")
        return base.rstrip("/") + "/chat/completions"
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("capture_route") from error


def _pairs(items: Any) -> Any:
    value = dict(items)
    if len(value) != len(items):
        raise ValueError("duplicate_key")
    return value


class ModelResponseCapture:
    """One process-owned artifact; failures affect observation, never model results."""

    def __init__(self, directory: Path, mode: str, manifest: Any, config: Any,
                 on_update: Any = None) -> None:
        self.mode = validate_mode(mode)
        self.endpoint = capture_endpoint(mode, manifest, config)
        self._native_scope = ({key: config["model"][field] for key, field in (
            ("provider", "provider"), ("model", "default"), ("base_url", "base_url"),
        )} if self.endpoint is not None else {})
        self.path = directory / "model-response-capture.private.json"
        self._lock = threading.RLock()
        self._on_update = on_update
        self.status: dict[str, Any] = {
            "version": 1, "mode": mode, "available": True, "events": [],
            "observed": 0, "truncated": False, "unavailable_reason": "none",
        }
        if self.mode != "off":
            if self.path.exists():
                raise ValueError("capture_artifact_exists")
            self._publish()

    def summary(self) -> dict[str, Any]:
        return {key: self.status[key] for key in (
            "mode", "available", "observed", "truncated", "unavailable_reason",
        )} | {"retained": len(self.status["events"])}

    def _fail(self, reason: str) -> None:
        self.status.update(available=False, unavailable_reason=(
            reason if reason in _REASONS else "observation_failed"))

    def _write(self) -> None:
        data = json.dumps(self.status, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            self._fail("artifact_bound")
            for event in self.status["events"]:
                event.pop("content", None)
            data = json.dumps(self.status, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("capture_artifact_bound")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent, prefix=".model-response-", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _publish(self) -> None:
        try:
            self._write()
        except Exception:
            self._fail("write_failed")
        if self._on_update is not None:
            try:
                self._on_update(self.summary())
            except Exception:
                self._fail("witness_write_failed")
        if not self.status["available"]:
            # A failed replace can leave an older artifact. Never let that imply
            # complete capture merely because it was the last successful write.
            with contextlib.suppress(Exception):
                print("[lookup-model-capture] " + json.dumps(self.summary(), separators=(",", ":")),
                      flush=True)

    def unavailable(self, reason: str) -> None:
        if self.mode == "off":
            return
        with self._lock:
            self._fail(reason)
            self._publish()

    def observe(self, request: Any, response: Any, *, streaming: bool = False) -> None:
        if self.mode == "off" or str(request.url) != self.endpoint:
            return
        with self._lock:
            try:
                self.status["observed"] = min(1000000, self.status["observed"] + 1)
                if streaming:
                    self._fail("streaming_response")
                elif (str(response.request.url) != self.endpoint or response.history):
                    self._fail("redirected_response")
                elif not response.is_stream_consumed:
                    self._fail("unread_response")
                elif len(response.content) > MAX_RESPONSE_BYTES:
                    self._fail("response_bound")
                elif len(self.status["events"]) >= MAX_EVENTS:
                    self._fail("event_bound")
                    self.status["truncated"] = True
                else:
                    # Both properties are cached bytes; never call read/aread here.
                    if len(request.content) > 262144:
                        raise ValueError("request_bound")
                    payload = json.loads(request.content, object_pairs_hook=_pairs)
                    body = json.loads(response.content, object_pairs_hook=_pairs)
                    choices = body["choices"]
                    if type(payload) is not dict or type(choices) is not list or len(choices) != 1:
                        raise ValueError("response_shape")
                    choice = choices[0]
                    content = choice["message"]["content"]
                    finish = choice.get("finish_reason")
                    self._record_content(content, finish, "response_format" in payload,
                                         "presentation")
            except Exception:
                self._fail("observation_failed")
            self._publish()

    def _record_content(self, content: Any, finish: Any, response_format: bool,
                        stage: str) -> None:
        event = {"index": len(self.status["events"]) + 1, "stage": stage,
                 "response_format": response_format,
                 "finish_reason": finish if finish in _FINISH else "other",
                 "content_kind": "string" if type(content) is str
                 else "none" if content is None else "other"}
        if type(content) is str:
            raw = content.encode("utf-8")
            event.update(content_chars=len(content), content_bytes=len(raw),
                         content_sha256=hashlib.sha256(raw).hexdigest())
            if self.mode == "controlled-public":
                if len(raw) <= MAX_CONTENT_BYTES:
                    event["content"] = content
                else:
                    self._fail("content_bound")
        self.status["events"].append(event)

    def observe_native(self, request: Any, response: Any, context: Any) -> None:
        if self.mode == "off" or any(
            context.get(key) != value for key, value in self._native_scope.items()
        ):
            return
        with self._lock:
            try:
                self.status["observed"] = min(1000000, self.status["observed"] + 1)
                if len(self.status["events"]) >= MAX_EVENTS:
                    self._fail("event_bound")
                    self.status["truncated"] = True
                else:
                    choices = response.choices
                    if len(choices) != 1 or type(request) is not dict:
                        raise ValueError("response_shape")
                    choice = choices[0]
                    self._record_content(choice.message.content, choice.finish_reason,
                                         "response_format" in request, "native")
            except Exception:
                self._fail("observation_failed")
            self._publish()

    @contextlib.contextmanager
    def installed(self, middleware: Any = None) -> Any:
        if self.mode == "off":
            yield
            return
        import httpx
        if middleware is None:
            import hermes_cli.middleware as middleware

        original_async = httpx.AsyncClient.send
        original_middleware = middleware.run_llm_execution_middleware

        def run_middleware(request: Any, next_call: Any, **context: Any) -> Any:
            def terminal(payload: Any) -> Any:
                response = next_call(payload)
                try:
                    self.observe_native(payload, response, context)
                except Exception:
                    self.unavailable("observation_failed")
                return response

            return original_middleware(request, terminal, **context)

        async def async_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            response = await original_async(client, request, *args, **kwargs)
            try:
                self.observe(request, response, streaming=kwargs.get("stream", False))
            except Exception:
                self.unavailable("observation_failed")
            return response

        middleware.run_llm_execution_middleware = run_middleware
        httpx.AsyncClient.send = async_send
        try:
            yield
        finally:
            if middleware.run_llm_execution_middleware is run_middleware:
                middleware.run_llm_execution_middleware = original_middleware
            else:
                self.unavailable("wrapper_replaced")
            if httpx.AsyncClient.send is async_send:
                httpx.AsyncClient.send = original_async
            else:
                self.unavailable("wrapper_replaced")
