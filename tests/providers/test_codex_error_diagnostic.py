"""Content-free qualification observations; no process or model is launched."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import pytest
from test_codex_app_server import FakeCodexTransport, _snapshot

from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from scripts import probe_confirmed_natural_work as probe


def notification(info: object = "usageLimitExceeded", **extra: object) -> dict[str, object]:
    return {"method": "error", "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1", "willRetry": False,
        "error": {"message": "SYNTHETIC_PRIVATE_message_path_url_token",
                  "additionalDetails": "SYNTHETIC_PRIVATE_details", "codexErrorInfo": info},
        **extra,
    }}


@pytest.mark.parametrize("code", [
    "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded", "serverOverloaded",
    "cyberPolicy", "internalServerError", "unauthorized", "badRequest", "threadRollbackFailed",
    "sandboxError", "other",
])
def test_known_string_codes_have_fixed_projection(code):
    assert probe.codex_error_diagnostic(notification(code)) == {
        "category": "provider_error", "code": code, "will_retry": False,
    }


@pytest.mark.parametrize("code", [
    "httpConnectionFailed", "responseStreamConnectionFailed", "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
])
@pytest.mark.parametrize("status", [None, 0, 429, 65535])
def test_supported_numeric_codes_follow_pinned_uint16_schema(code, status):
    expected = {"category": "provider_error", "code": code, "will_retry": False}
    if status is not None:
        expected["http_status_code"] = status
    message = notification({code: {"httpStatusCode": status}})
    assert probe.codex_error_diagnostic(message) == expected


@pytest.mark.parametrize("info", [
    None, "SYNTHETIC_PRIVATE_unknown_code", {}, {"SYNTHETIC_PRIVATE_unknown": {}},
    {"httpConnectionFailed": {}, "responseStreamDisconnected": {}},
    {"httpConnectionFailed": "SYNTHETIC_PRIVATE_payload"},
])
def test_unknown_or_malformed_code_stays_unknown(info):
    assert probe.codex_error_diagnostic(notification(info)) == {
        "category": "provider_error", "code": "unknown", "will_retry": False,
    }


@pytest.mark.parametrize("status", [True, False, -1, 65536, 429.0, "429"])
def test_invalid_http_status_is_not_a_supported_code(status):
    assert probe.codex_error_diagnostic(notification({
        "httpConnectionFailed": {"httpStatusCode": status},
    })) == {"category": "provider_error", "code": "unknown", "will_retry": False}


@pytest.mark.parametrize("kind", ["review", "compact"])
def test_nonsteerable_variant_keeps_only_fixed_code(kind):
    assert probe.codex_error_diagnostic(notification({
        "activeTurnNotSteerable": {"turnKind": kind},
    })) == {"category": "provider_error", "code": "activeTurnNotSteerable", "will_retry": False}


@pytest.mark.parametrize("kind", [None, "SYNTHETIC_PRIVATE_kind", 1])
def test_unknown_nonsteerable_kind_stays_unknown(kind):
    assert probe.codex_error_diagnostic(notification({
        "activeTurnNotSteerable": {"turnKind": kind},
    }))["code"] == "unknown"


@pytest.mark.parametrize("retry", [None, 0, 1, "SYNTHETIC_PRIVATE_retry", {}, []])
def test_retry_field_requires_exact_boolean(retry):
    assert probe.codex_error_diagnostic(notification(willRetry=retry)) == {
        "category": "provider_error", "code": "usageLimitExceeded",
    }


def test_retry_true_and_absent_http_status_follow_schema():
    assert probe.codex_error_diagnostic(notification(
        {"httpConnectionFailed": {}}, willRetry=True,
    )) == {"category": "provider_error", "code": "httpConnectionFailed", "will_retry": True}


def test_private_payload_fields_never_enter_projection():
    message = notification({"httpConnectionFailed": {
        "httpStatusCode": 429, "private": "SYNTHETIC_PRIVATE_nested",
    }}, private="SYNTHETIC_PRIVATE_arbitrary")
    message["private"] = "SYNTHETIC_PRIVATE_top"
    assert probe.codex_error_diagnostic(message) == {
        "category": "provider_error", "code": "httpConnectionFailed",
        "http_status_code": 429, "will_retry": False,
    }
    assert "SYNTHETIC_PRIVATE" not in json.dumps(probe.codex_error_diagnostic(message))


class StringSubclass(str):
    pass


class DictSubclass(dict):
    pass


def test_nonsteerable_kind_requires_exact_string():
    assert probe.codex_error_diagnostic(notification({
        "activeTurnNotSteerable": {"turnKind": StringSubclass("review")},
    }))["code"] == "unknown"


@pytest.mark.parametrize("info", [
    StringSubclass("usageLimitExceeded"), DictSubclass(httpConnectionFailed={}),
    {StringSubclass("httpConnectionFailed"): {}},
    {"httpConnectionFailed": DictSubclass(httpStatusCode=429)},
])
def test_code_projection_requires_exact_types(info):
    assert probe.codex_error_diagnostic(notification(info))["code"] == "unknown"


@pytest.mark.parametrize("message", [
    None, [], DictSubclass(method="error"),
    {"method": StringSubclass("error")}, {"method": "warning"},
    {"method": "account/rateLimits/updated", "params": {"error": {"code": 429}}},
    {"id": 1, "error": {"code": -32603, "message": "SYNTHETIC_PRIVATE_rpc"}},
])
def test_non_notification_messages_are_not_provider_error_evidence(message):
    assert probe.codex_error_diagnostic(message) is None


@pytest.mark.parametrize("params", [None, [], DictSubclass(error={})])
def test_malformed_notification_payload_is_unknown(params):
    assert probe.codex_error_diagnostic({"method": "error", "params": params}) == {
        "category": "provider_error", "code": "unknown",
    }


def test_nested_objects_require_exact_types():
    message = notification()
    message["params"] = DictSubclass(message["params"])
    assert probe.codex_error_diagnostic(message) == {
        "category": "provider_error", "code": "unknown",
    }
    message = notification()
    message["params"]["error"] = DictSubclass(message["params"]["error"])
    assert probe.codex_error_diagnostic(message) == {
        "category": "provider_error", "code": "unknown", "will_retry": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [False, True], ids=["healthy", "provider_error"])
async def test_observed_error_after_accepted_startup_remains_operational_without_retry(fault):
    class ObservedTransport(FakeCodexTransport):
        def __init__(self):
            super().__init__()
            self.observed = []
            self.accepted = []
            self.requests = {}

        async def send(self, message: Mapping[str, object]) -> None:
            self.requests[message.get("id")] = message.get("method")
            if fault and message.get("method") == "turn/start":
                self.sent.append(dict(message))
                await self.incoming.put({"id": message["id"], "result": {"turn": {
                    "id": "turn_server_1", "items": [], "status": "inProgress",
                }}})
                await self.incoming.put(notification())
            else:
                await super().send(message)

        async def receive(self):
            message = await super().receive()
            if "result" in message:
                self.accepted.append(self.requests.get(message.get("id")))
            observed = probe.codex_error_diagnostic(message)
            if observed is not None:
                self.observed.append(observed)
            return message

    transport = ObservedTransport()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
        request_timeout_seconds=1,
    )
    primary = None
    cleanup = []
    try:
        try:
            result = await asyncio.wait_for(_collect(inference), 2)
        except RuntimeError:
            primary = transport.observed[-1] if transport.observed else {"category": "unknown"}
    finally:
        try:
            await inference.close()
        except Exception:
            cleanup.append("cleanup_failed")
    assert transport.accepted.count("thread/start") == transport.accepted.count("turn/start") == 1
    assert sum(row.get("method") == "turn/start" for row in transport.sent) == 1
    assert transport.closed is True
    assert cleanup == []
    if fault:
        assert primary == {"category": "provider_error", "code": "usageLimitExceeded",
                           "will_retry": False}
        assert "SYNTHETIC_PRIVATE" not in json.dumps(primary)
    else:
        assert result == ["Four."]
        assert primary is None
        assert transport.observed == []


async def _collect(inference):
    return [part async for part in inference.stream(_snapshot(), turn_id="synthetic_diagnostic")]
