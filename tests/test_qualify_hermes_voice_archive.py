from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "qualify_hermes_voice_archive.py"
)
sys.path.insert(0, str(_SCRIPT_PATH.parent))  # The script imports its sibling gate support.
_SPEC = importlib.util.spec_from_file_location("qualify_hermes_voice_archive", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)

_BASELINE: dict[str, object] = {"version": "0.21.0", "commit": "29112bef", "baseline": True}


def _observed() -> dict[str, list[dict[str, object]]]:
    return json.loads(json.dumps(_SCRIPT._EXPECTED))


def test_the_expected_observations_pass() -> None:
    assert _SCRIPT._passed(_observed(), _BASELINE) is True


def test_evidence_from_any_hermes_but_the_baseline_fails() -> None:
    assert _SCRIPT._passed(_observed(), _BASELINE | {"baseline": False}) is False
    assert _SCRIPT._passed(_observed(), {}) is False


def test_every_planned_scenario_has_an_expectation_step_for_step() -> None:
    assert set(_SCRIPT._PLAN) == set(_SCRIPT._EXPECTED)
    for scenario, steps in _SCRIPT._PLAN.items():
        assert len(steps) == len(_SCRIPT._EXPECTED[scenario])


def test_the_criteria_are_all_planned() -> None:
    plan = _SCRIPT._PLAN
    assert {f"foreign_{kind}" for kind in _SCRIPT._FOREIGN_KINDS} <= set(plan)
    assert {f"crash_{point}" for point in _SCRIPT._CRASH_POINTS} <= set(plan)
    assert {"surface", "dedup", "compaction_at_restart", "crash_ambiguous", "lease_false",
            "lease_raise", "lease_stolen", "cap", "creation_occupied"} <= set(plan)
    assert set(_SCRIPT._FOREIGN_KINDS) == {
        "compaction", "replace_content", "replace_reorder", "display_flag", "unleased_append",
        "delete", "rotation", "replace_archived",
    }


def _variations() -> list[tuple[str, int, str, object]]:
    variations: list[tuple[str, int, str, object]] = []
    for scenario, steps in _SCRIPT._EXPECTED.items():
        for index, step in enumerate(steps):
            variations.append((scenario, index, "exit", 1))
            variations.append((scenario, index, "markers", [*step["markers"], "archive:extra"]))
            for field in step["result"]:
                variations.append((scenario, index, f"result.{field}", "other"))
    return variations


@pytest.mark.parametrize(("scenario", "index", "field", "value"), _variations())
def test_any_other_observation_fails(scenario: str, index: int, field: str, value: object) -> None:
    observed = _observed()
    step = observed[scenario][index]
    if field.startswith("result."):
        step["result"][field.removeprefix("result.")] = value  # type: ignore[index]
    else:
        step[field] = value
    assert _SCRIPT._passed(observed, _BASELINE) is False


def test_a_missing_scenario_or_step_fails() -> None:
    observed = _observed()
    del observed["cap"]
    assert _SCRIPT._passed(observed, _BASELINE) is False
    observed = _observed()
    observed["crash_in_state"].pop()
    assert _SCRIPT._passed(observed, _BASELINE) is False


def test_markers_are_read_as_categories_only() -> None:
    stdout = "\n".join(
        [
            '[voice-archive] {"refusal":"conflict","rows":2,"version":1}',
            "unrelated output",
            '[voice-archive-lease] {"fence":"lost","version":1}',
            '[voice-archive-open] {"refusal":"quarantined","version":1}',
        ]
    )
    assert _SCRIPT._markers(stdout) == ["archive:conflict", "lease:lost", "open:quarantined"]
