"""Adversarial acceptance tests for the source-only equivalence producer."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import pytest


def _module() -> Any:
    from scripts import deterministic_equivalence

    return deterministic_equivalence


def _arm(name: str, *, shutdown: bool = False) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if shutdown:
        records.append({"kind": "cancellation", "value": "host_shutdown"})
    else:
        for _ in range(2):
            records.extend(
                {"kind": kind, "value": str(index + 1) * 64}
                for index, kind in enumerate(
                    (
                        "committed_conversation_context_snapshot",
                        "generated_text",
                        "transport_confirmed_chunk",
                    )
                )
            )
    records.extend(
        (
            {"kind": "foreground_cleanup", "value": True},
            {"kind": "host_return", "value": "returned"},
        )
    )
    return {
        "arm": name,
        "complete": True,
        "records": records,
        "close": [
            {"stage": stage, "result": "succeeded"}
            for stage in ("browser_client", "foreground_close", "speech_loop", "launcher")
        ],
    }


def _observations() -> list[dict[str, Any]]:
    rows = [_arm(name) for name in ("disabled", "unconsented", "consented", "blocked", "faulted")]
    rows[4]["records"][-1]["value"] = "failed"
    rows[4]["close"][-1]["result"] = "failed"
    rows.extend(
        _arm(name, shutdown=True)
        for name in ("shutdown_disabled", "shutdown_unconsented", "shutdown_consented")
    )
    control = _arm("perturbed")
    control["records"][0]["value"] = "4" * 64
    rows.append(control)
    return rows


def test_validator_accepts_only_the_complete_ordered_comparison_with_negative_control() -> None:
    assert _module()._validate_trace_set(_observations()) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_arm",
        "duplicate_arm",
        "reordered_arm",
        "empty_trace",
        "incomplete",
        "equal_control",
        "missing_context",
        "missing_transport",
        "wrong_host_result",
        "changed_conversation",
        "missing_cancellation",
        "failed_cleanup",
        "unproven_close",
        "missing_close_stage",
        "unknown_field",
        "unkeyed_text",
        "bool_as_version",
        "fault_normalized",
    ],
)
def test_missing_disconnected_or_changed_observations_never_pass(mutation: str) -> None:
    rows = _observations()
    if mutation == "missing_arm":
        rows.pop()
    elif mutation == "duplicate_arm":
        rows[1] = deepcopy(rows[0])
    elif mutation == "reordered_arm":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "empty_trace":
        for row in rows:
            row["records"] = []
    elif mutation == "incomplete":
        rows[2]["complete"] = False
    elif mutation == "equal_control":
        rows[-1]["records"] = deepcopy(rows[0]["records"])
    elif mutation in {"missing_context", "missing_transport", "missing_cancellation"}:
        kind = {
            "missing_context": "committed_conversation_context_snapshot",
            "missing_transport": "transport_confirmed_chunk",
            "missing_cancellation": "cancellation",
        }[mutation]
        for row in rows:
            row["records"] = [record for record in row["records"] if record["kind"] != kind]
    elif mutation == "wrong_host_result":
        rows[2]["records"][-1]["value"] = "failed"
    elif mutation == "changed_conversation":
        rows[2]["records"][0]["value"] = "5" * 64
    elif mutation == "failed_cleanup":
        for row in rows:
            row["records"][-2]["value"] = False
    elif mutation == "unproven_close":
        for row in rows:
            row["close"] = []
    elif mutation == "missing_close_stage":
        for row in rows:
            row["close"] = [item for item in row["close"] if item["stage"] != "foreground_close"]
    elif mutation == "unknown_field":
        rows[0]["passed"] = True
    elif mutation == "unkeyed_text":
        rows[0]["records"][0]["value"] = "a transcript is not an observation digest"
    elif mutation == "bool_as_version":
        rows[0]["complete"] = 1
    elif mutation == "fault_normalized":
        rows[4]["records"][-1]["value"] = "returned"
    with pytest.raises(ValueError):
        _module()._validate_trace_set(rows)


def test_plain_data_cannot_be_promoted_to_verified_producer_authority() -> None:
    module = _module()
    with pytest.raises(TypeError):
        module.ObservedEquivalenceV1()
    with pytest.raises((TypeError, ValueError)):
        module.validate_deterministic_equivalence_v1(_observations())
    forged = object.__new__(module.ObservedEquivalenceV1)
    with pytest.raises(ValueError, match="unregistered"):
        module.validate_deterministic_equivalence_v1(forged)


@pytest.mark.skipif(os.name != "nt", reason="real Windows socket ownership")
def test_live_listener_requires_the_retained_owner_and_exact_loopback_port() -> None:
    import socket

    from scripts.equivalence_process import _require_owned_listener

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        _require_owned_listener(port, os.getpid())
        with pytest.raises(ValueError):
            _require_owned_listener(port, os.getpid() + 1)
    with pytest.raises(ValueError):
        _require_owned_listener(port, os.getpid())


def test_server_environment_preserves_windows_system_keys_without_inheriting_host_authority() -> (
    None
):
    from scripts.equivalence_worker import _server_environment

    values = {
        "SYSTEMROOT": "system",
        "SystemDrive": "drive",
        "TEMP": "temporary",
        "API_SERVER_KEY": "placeholder",
        "PYTHONPATH": "ambient",
    }
    result = _server_environment(values)
    assert result["SYSTEMROOT"] == "system"
    assert result["SystemDrive"] == "drive"
    assert result["TEMP"] == "temporary"
    assert "API_SERVER_KEY" not in result and "PYTHONPATH" not in result
