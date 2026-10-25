"""Qualification controls remain distinct from real model/backend evidence."""

import asyncio
import copy
import json

import pytest

from scripts import qualify_confirmed_natural_work as qualification


def test_frozen_preregistration_rejects_changed_input(tmp_path):
    path = tmp_path / "changed.json"
    path.write_bytes(qualification.FIXTURE.read_bytes() + b" ")
    with pytest.raises(ValueError, match="preregistration_binding"):
        qualification.load_preregistration(path)


@pytest.mark.asyncio
async def test_deterministic_controls_do_not_claim_semantic_or_backend_pass():
    report = await qualification.run_controls()
    assert report["deterministic_confirmation"]["pass"] is True
    assert report["model_proposal_review"] == {
        "status": "not_run", "selection": "pending", "objective_fidelity": "pending",
        "lifecycle_claims": "pending", "observations": 0,
    }
    assert report["actual_source_proof"] == "not_run"
    assert report["natural_routing_qualified"] is False
    serialized = json.dumps(report)
    assert "Synthetic" not in serialized
    assert "Inspect **synthetic**" not in serialized
    assert "administrator credentials" not in serialized


@pytest.mark.parametrize("field,value", [
    ("dispatches", 1), ("cancellations", 1), ("presentation_equal", False),
    ("dispatch_equal", False), ("started_before_ack", True),
    ("unrelated_unchanged", False),
])
def test_control_oracle_rejects_each_independent_fault(field, value):
    expected = qualification.expected_control("unconfirmed")
    observed = copy.deepcopy(expected)
    observed[field] = value
    assert qualification.control_passes(observed, expected) is False


def test_control_oracle_rejects_stale_extra_observation():
    expected = qualification.expected_control("unconfirmed")
    observed = expected | {"stale": 1}
    assert qualification.control_passes(observed, expected) is False


@pytest.mark.asyncio
async def test_negative_effect_is_counted_after_delayed_owned_dispatch(monkeypatch):
    original_dispatch = qualification._Controller.dispatch

    async def delayed_dispatch(self, **kwargs):
        for _ in range(4):
            await asyncio.sleep(0)
        return await original_dispatch(self, **kwargs)

    async def broken_decline(self, **kwargs):
        return await self._surface.submit_start_command(
            objective=kwargs["objective"], invocation_id="broken_decline",
        )

    monkeypatch.setattr(qualification._Controller, "dispatch", delayed_dispatch)
    monkeypatch.setattr(qualification.ConversationTaskProposals, "decide", broken_decline)
    observed = await qualification._run_control("declined")
    assert observed["dispatches"] == 1
    expected = qualification.expected_control("declined")
    assert qualification.control_passes(observed, expected) is False
