"""The real-Hermes M2 gate refuses missing or weakened observations."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import aiohttp
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qualify_voice_review.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("qualify_voice_review", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)

_BASELINE = {
    "version": "0.21.0", "commit": "29112bef099274229cadff79cdff7bf7b99c4b77",
    "baseline": True,
}


def _good() -> dict[str, object]:
    return {
        "coverage": {
            str(n): {
                "users": n, "admissions": n // 10 + int(n > 0), "covered_users": n,
                "closing_retained": 0, "max_messages": min(n * 2, 20),
                "digest": 0, "failures": 0,
            }
            for n in _SCRIPT._COUNTS
        },
        "busy_close": {"admitted": 1, "retained": 1, "lost": 0},
        "disconnect": {"admitted": 1, "retained": 1, "lost": 0},
        "restart": {"admitted": 1, "retained": 1, "lost": 0},
        "confinement": {
            f"{route}_{ordering}": {
                "attempted": 2, "outside_executed": 0, "denied": 2,
                "whitelist_equal": 1, "extras_empty": 1, "schema_restricted": 1,
            }
            for route in ("main", "routed")
            for ordering in ("serial", "parallel")
        },
        "boundary": {
            "refused": 1, "outside_executed": 0, "model_requests": 0,
        },
        "attribution": {
            route: {
                "cases": 6, "errors": 0, "digest": 0, "oversized_split_or_refused": 1,
            }
            for route in ("main", "routed")
        },
        "corrections": {"declared": 2, "persisted": 2, "fresh_applied": 6, "repeats": 3},
        "close_integrity": {
            "finished": 1, "fingerprint_equal": 1,
            "ended_at_null": 1, "append_after_close": 1,
        },
        "speech": {
            "summary_sink_calls": 1, "gateway_callback_bound": 0,
            "failure_sink_calls": 1, "outbound_summary_lines": 0,
            "work_dispatches": 0,
            "native_actions": 1, "native_failure": 1,
            "bridge_control": 1, "logger_control": 1,
            "sender_control": 1, "model_requests": 2,
        },
        "guards": {
            "warm_malformed": 1, "cold_malformed": 1, "extras": 1,
            "mutated_after_admission": 1,
            "credential_drift": 1, "model_drift": 1,
            "wrong_home_bound": 1, "bad_db": 1,
            "memory_construct": 1, "memory_load": 1, "token_rollback": 1,
            "model_leaks": 0,
        },
        "unattributed": 0,
    }


def test_only_complete_baseline_passes() -> None:
    assert _SCRIPT._passed(_good(), _BASELINE) is True
    assert _SCRIPT._passed(_good(), _BASELINE | {"baseline": False}) is False
    assert _SCRIPT._passed(_good(), _BASELINE | {"commit": "other"}) is False


@pytest.mark.parametrize("value", [True, 1.0])
def test_numeric_evidence_requires_exact_integers(value: object) -> None:
    observation = _good()
    observation["boundary"]["refused"] = value  # type: ignore[index]
    assert _SCRIPT._passed(observation, _BASELINE) is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("coverage", "0", "admissions"), 1),
        (("coverage", "10", "admissions"), 1),
        (("coverage", "11", "covered_users"), 10),
        (("coverage", "19", "closing_retained"), 1),
        (("coverage", "20", "max_messages"), 25),
        (("coverage", "9", "digest"), 1),
        (("coverage", "8", "failures"), 1),
        (("busy_close", "retained"), 0),
        (("disconnect", "lost"), 1),
        (("restart", "admitted"), 0),
        (("confinement", "main_serial", "outside_executed"), 1),
        (("confinement", "main_parallel", "denied"), 1),
        (("confinement", "routed_serial", "whitelist_equal"), 0),
        (("confinement", "routed_parallel", "extras_empty"), 0),
        (("confinement", "routed_parallel", "schema_restricted"), 0),
        (("boundary", "refused"), 0),
        (("boundary", "model_requests"), 1),
        (("attribution", "main", "errors"), 1),
        (("attribution", "routed", "digest"), 1),
        (("attribution", "routed", "oversized_split_or_refused"), 0),
        (("corrections", "persisted"), 1),
        (("corrections", "fresh_applied"), 5),
        (("close_integrity", "fingerprint_equal"), 0),
        (("close_integrity", "ended_at_null"), 0),
        (("close_integrity", "append_after_close"), 0),
        (("speech", "summary_sink_calls"), 0),
        (("speech", "gateway_callback_bound"), 1),
        (("speech", "failure_sink_calls"), 0),
        (("speech", "outbound_summary_lines"), 1),
        (("speech", "work_dispatches"), 1),
        (("speech", "sender_control"), 0),
        (("speech", "model_requests"), 1),
        (("speech", "bridge_control"), 0),
        (("speech", "logger_control"), 0),
        (("speech", "native_failure"), 0),
        (("guards", "warm_malformed"), 0),
        (("guards", "cold_malformed"), 0),
        (("guards", "extras"), 0),
        (("guards", "credential_drift"), 0),
        (("guards", "model_drift"), 0),
        (("guards", "mutated_after_admission"), 0),
        (("guards", "wrong_home_bound"), 0),
        (("guards", "memory_load"), 0),
        (("guards", "token_rollback"), 0),
        (("guards", "model_leaks"), 1),
        (("unattributed",), 1),
    ],
)
def test_one_weakened_observation_fails(path: tuple[str, ...], value: object) -> None:
    observation = _good()
    node = observation
    for name in path[:-1]:
        node = node[name]  # type: ignore[assignment]
    node[path[-1]] = value  # type: ignore[index]
    assert _SCRIPT._passed(observation, _BASELINE) is False


@pytest.mark.parametrize(
    "section", [
        "coverage", "busy_close", "confinement", "boundary", "corrections",
        "close_integrity", "speech", "guards",
    ]
)
def test_missing_section_fails(section: str) -> None:
    observation = _good()
    del observation[section]
    assert _SCRIPT._passed(observation, _BASELINE) is False


@pytest.mark.asyncio
async def test_stand_in_requires_one_synthetic_case_and_known_model() -> None:
    model = _SCRIPT._StandInModel()
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as http:
            for body in (
                {"model": "m2-main", "messages": [{"role": "user", "content": "none"}]},
                {"model": "other", "messages": [{"role": "user", "content": "m2case-one"}]},
                {
                    "model": "m2-main",
                    "messages": [{"role": "user", "content": "m2case-one m2case-two"}],
                },
            ):
                async with http.post(f"{url}/chat/completions", json=body) as response:
                    assert response.status == 400
            assert model.unattributed == 3
            assert model.calls == {}
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_stand_in_forces_serial_and_parallel_tool_calls() -> None:
    model = _SCRIPT._StandInModel()
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as http:
            for case, expected in (
                ("m2case-main_serial", ["terminal"]),
                ("m2case-main_parallel", ["terminal", "write_file"]),
            ):
                body = {"model": "m2-main", "messages": [{"role": "user", "content": case}]}
                async with http.post(f"{url}/chat/completions", json=body) as response:
                    assert response.status == 200
                    line = await response.content.readline()
                    chunk = json.loads(line.removeprefix(b"data: "))
                    calls = chunk["choices"][0]["delta"]["tool_calls"]
                    assert [call["function"]["name"] for call in calls] == expected
    finally:
        await model.close()
