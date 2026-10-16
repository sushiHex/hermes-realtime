"""Keep the resource-first scenario's entire safety cohort unchanged."""

import json
from pathlib import Path


def test_resource_scenario_preserves_every_original_negative() -> None:
    fixtures = Path(__file__).parent / "fixtures"
    original = json.loads((fixtures / "natural_work_intent_cases.json").read_bytes())
    scenario = json.loads((fixtures / "natural_work_resource_scenario/main.json").read_bytes())
    assert scenario["cases"][20:] == original["cases"][20:]
