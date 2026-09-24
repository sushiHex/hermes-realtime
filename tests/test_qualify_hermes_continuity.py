from __future__ import annotations

import contextlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import aiohttp
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qualify_hermes_continuity.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))  # The script imports its sibling gate support.
_SCRIPT_SPEC = importlib.util.spec_from_file_location("qualify_hermes_continuity", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = _SCRIPT
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

_NONCE = "continuity-0123456789abcdef"


def _completion(*texts: str) -> dict[str, object]:
    return {
        "model": "stand-in",
        "stream": True,
        "messages": [{"role": "user", "content": text} for text in texts],
    }


@pytest.mark.asyncio
async def test_the_stand_in_counts_work_while_it_streams_and_after_it_stops() -> None:
    model = _SCRIPT._StandInModel()
    url = await model.start()
    try:
        async with (
            aiohttp.ClientSession() as http,
            http.post(f"{url}/chat/completions", json=_completion(_NONCE)) as response,
        ):
            assert response.status == 200
            first = await response.content.readline()
            assert first.startswith(b"data: ")
            assert json.loads(first.removeprefix(b"data: "))["choices"][0]["delta"]
            assert model.work(_NONCE) == {"started": 1, "running": 1}
        await model.wait_until(_NONCE, running=0, timeout=10)
        assert model.work(_NONCE) == {"started": 1, "running": 0}
    finally:
        await model.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts",
    [("no nonce here",), (_NONCE, "continuity-fedcba9876543210")],
    ids=["none", "two"],
)
async def test_a_request_that_names_no_single_dispatch_is_refused_and_not_counted(
    texts: tuple[str, ...],
) -> None:
    model = _SCRIPT._StandInModel()
    url = await model.start()
    try:
        async with (
            aiohttp.ClientSession() as http,
            http.post(f"{url}/chat/completions", json=_completion(*texts)) as response,
        ):
            assert response.status == 400
        assert model.work(_NONCE) == {"started": 0, "running": 0}
        assert model.unattributed == 1
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_waiting_for_work_that_never_arrives_times_out() -> None:
    model = _SCRIPT._StandInModel()
    await model.start()
    try:
        with pytest.raises(TimeoutError):
            await model.wait_until(_NONCE, started=1, timeout=0.2)
    finally:
        await model.close()


def test_the_census_counts_record_entries_without_reading_their_content(tmp_path: Path) -> None:
    record = tmp_path / "runs.json"
    assert _SCRIPT._census(record) == {"admitted": 0, "pending": 0}
    record.write_text(
        json.dumps(
            {
                "admitted": ["run_a", "run_b"],
                "pending": [{"key": None, "minted_at": 1.0, "request": {}}],
                "version": 1,
            }
        ),
        encoding="utf-8",
    )
    assert _SCRIPT._census(record) == {"admitted": 2, "pending": 1}


def test_admissions_are_read_from_hermes_store_in_admission_order(tmp_path: Path) -> None:
    assert _SCRIPT._admissions(tmp_path) == []
    with contextlib.closing(sqlite3.connect(tmp_path / "runs_idempotency.db")) as db:
        db.execute("CREATE TABLE run_idempotency (status_json TEXT, created_at REAL)")
        db.executemany(
            "INSERT INTO run_idempotency VALUES (?, ?)",
            [
                (json.dumps({"status": "interrupted", "run_id": "run_b"}), 2.0),
                (json.dumps({"status": "cancelled", "run_id": "run_a"}), 1.0),
            ],
        )
        db.commit()
    assert _SCRIPT._admissions(tmp_path) == ["cancelled", "interrupted"]


def _observed() -> dict[str, dict[str, object]]:
    return json.loads(json.dumps(_SCRIPT._EXPECTED))


def test_the_expected_observations_pass() -> None:
    assert _SCRIPT._passed(_observed(), 0) is True


def test_an_unattributed_model_request_fails() -> None:
    assert _SCRIPT._passed(_observed(), 1) is False


@pytest.mark.parametrize(
    ("scenario", "field", "value"),
    [
        ("crash_before_acknowledgment", "recorded", {"admitted": 1, "pending": 0}),
        ("crash_after_acknowledgment", "recorded", {"admitted": 0, "pending": 1}),
        ("hermes_restart", "recorded", {"admitted": 0, "pending": 0}),
        ("crash_before_acknowledgment", "settlement", {"stopped": 0, "unknown": 1}),
        ("crash_after_acknowledgment", "settlement", {"stopped": 0, "unknown": 0}),
        ("hermes_restart", "settlement", {"stopped": 1, "unknown": 1}),
        ("crash_before_acknowledgment", "work", {"started": 2, "running": 0}),
        ("crash_after_acknowledgment", "work", {"started": 1, "running": 1}),
        ("hermes_restart", "work", {"started": 2, "running": 0}),
        ("hermes_restart", "remaining", {"admitted": 0, "pending": 1}),
        ("crash_after_acknowledgment", "remaining", {"admitted": 1, "pending": 0}),
        ("crash_before_acknowledgment", "admissions", ["cancelled", "cancelled"]),
        ("crash_after_acknowledgment", "admissions", ["failed"]),
        ("hermes_restart", "admissions", ["cancelled"]),
        ("hermes_restart", "admissions", ["interrupted", "cancelled"]),
    ],
)
def test_any_other_observation_fails(scenario: str, field: str, value: object) -> None:
    observed = _observed()
    observed[scenario][field] = value
    assert _SCRIPT._passed(observed, 0) is False


def test_a_missing_scenario_fails() -> None:
    observed = _observed()
    del observed["hermes_restart"]
    assert _SCRIPT._passed(observed, 0) is False
