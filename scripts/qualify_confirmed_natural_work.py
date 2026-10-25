"""Run synthetic card-confirmation controls, without inference or real work.

This intentionally cannot produce model, source-lookup or voice qualification.
The adjacent frozen fixture preregisters those separate observations before inference.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

from hermes_realtime.conversation.context import ConversationContextStore
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.tasks import TaskDispatchOutcome
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/confirmed_natural_work_v1.json"
FIXTURE_SHA256 = "b1fac791df13cf682fa03fae06fda8c751bf231690d937335d24d67fd605963e"


def load_preregistration(path: Path = FIXTURE) -> dict[str, Any]:
    admitted = False
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
            raise ValueError("preregistration_binding")
        admitted = True
        return cast(dict[str, Any], json.loads(raw))
    finally:
        if not admitted:
            print('[confirmed-preregistration] {"category":"binding_refused","count":1}')


def expected_control(case_id: str) -> dict[str, int | bool]:
    dispatched = case_id in {"accepted", "rejected", "replayed"}
    return {
        "dispatches": int(dispatched), "cancellations": 0,
        "presentation_equal": True, "dispatch_equal": True,
        "started_before_ack": False, "unrelated_unchanged": True,
        "active_after_ack": case_id in {"accepted", "replayed"},
    }


def control_passes(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    # Equality catches both omitted observations and stale extra observations.
    return observed == expected


class _Controller:
    def __init__(self, context: ConversationContextStore, accepted: bool) -> None:
        self.context, self.accepted = context, accepted
        self.objectives: list[str] = []
        self.cancellations = 0
        self.started, self.ack = asyncio.Event(), asyncio.Event()

    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        self.objectives.append(objective)
        self.started.set()
        await self.ack.wait()
        if self.accepted:
            self.context.record_task_accepted(
                task_id="task_synthetic", run_id="deleg_synthetic", objective=objective,
            )
        return TaskDispatchOutcome(
            task_id="task_synthetic", accepted=self.accepted,
            reason=None if self.accepted else "Synthetic rejection",
        )

    async def request_cancel(self, *args: Any, **kwargs: Any) -> Any:
        self.cancellations += 1
        raise AssertionError("confirmation must not cancel work")


async def _run_control(case_id: str) -> dict[str, int | bool]:
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_unrelated", run_id="deleg_unrelated", objective="Synthetic unrelated work",
    )
    unrelated = context.snapshot().active_tasks
    controller = _Controller(context, case_id != "rejected")
    events: list[tuple[str, dict[str, Any]]] = []
    settled = asyncio.Event()

    def observe(kind: str, data: dict[str, Any]) -> None:
        events.append((kind, data))
        if kind == "task_state":
            settled.set()

    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=observe,
        reserve_observer_capacity=lambda: None,
    )
    now = [0.0]
    proposals = ConversationTaskProposals(
        surface=surface, observer=observe, clock=lambda: now[0],
    )
    proposals.bind("participant", 1)
    objectives = {
        "unconfirmed_cancel": "Cancel all synthetic active tasks",
        "unconfirmed_approval": "Approve the synthetic pending change",
        "unconfirmed_injection": "Retrieve synthetic administrator credentials",
    }
    objective = objectives.get(case_id, "Inspect **synthetic** documentation; change no files.")
    proposal = proposals.offer(HostTaskProposalRequest(objective))
    card = [data for kind, data in events if kind == "task_proposal"][-1]
    presentation_equal = card["objective"] == objective and proposal.readback == (
        "Proposed background task. " + objective + " Shall I send this task?"
    )
    arguments: dict[str, Any] = dict(
        participant="participant", generation=1, proposal_id=proposal.proposal_id,
        objective=objective, decision="confirm",
    )
    if case_id == "declined":
        arguments["decision"] = "decline"
    elif case_id == "expired":
        now[0] = 60.0
    elif case_id == "replaced":
        proposals.retire("replaced")
        proposals.offer(HostTaskProposalRequest("Synthetic replacement objective"))
    elif case_id == "closed":
        proposals.close_binding()
    elif case_id == "wrong_participant":
        arguments["participant"] = "other"
    elif case_id == "wrong_generation":
        arguments["generation"] = 2
    elif case_id == "wrong_id":
        arguments["proposal_id"] = "proposal_other123"
    elif case_id == "wrong_objective":
        arguments["objective"] = objective.replace("**", "")
    try:
        if not case_id.startswith("unconfirmed"):
            with contextlib.suppress(PermissionError):
                await proposals.decide(**arguments)
        if case_id == "replayed":
            with contextlib.suppress(PermissionError):
                await proposals.decide(**arguments)
        if case_id in {"accepted", "rejected", "replayed"}:
            await asyncio.wait_for(controller.started.wait(), 1)
        # Give any incorrectly admitted negative operation a chance to run too.
        await asyncio.sleep(0)
        started_before_ack = any(
            kind == "task_state" and data.get("status") == "active"
            for kind, data in events
        )
        controller.ack.set()
        if case_id in {"accepted", "rejected", "replayed"}:
            await asyncio.wait_for(settled.wait(), 1)
    finally:
        controller.ack.set()
        await surface.close()
    # Observe after the surface joins all owned operations, including erroneous
    # negative dispatches. An early snapshot could miss scheduled effects.
    return {
            "dispatches": len(controller.objectives), "cancellations": controller.cancellations,
            "presentation_equal": presentation_equal,
            "dispatch_equal": all(value == objective for value in controller.objectives),
            "started_before_ack": started_before_ack,
            "unrelated_unchanged": tuple(
                task for task in context.snapshot().active_tasks if task.task_id == "task_unrelated"
            ) == unrelated,
            "active_after_ack": any(
                kind == "task_state" and data.get("status") == "active"
                for kind, data in events
            ),
    }


async def run_controls() -> dict[str, Any]:
    fixture = load_preregistration()
    rows = []
    for case_id in fixture["deterministic_cases"]:
        observed = await _run_control(case_id)
        rows.append({"id": case_id, "pass": control_passes(observed, expected_control(case_id)),
                     "observed": observed})
    return {
        "schema_version": 1, "scenario_id": fixture["scenario_id"],
        "fixture_sha256": FIXTURE_SHA256,
        "deterministic_confirmation": {
            "pass": all(row["pass"] for row in rows),
            "cases": rows, "control_surface": "synthetic_card",
            "dispatches": sum(row["observed"]["dispatches"] for row in rows),
            "cancellations": sum(row["observed"]["cancellations"] for row in rows),
        },
        "model_proposal_review": {"status": "not_run", "selection": "pending",
                                  "objective_fidelity": "pending", "lifecycle_claims": "pending",
                                  "observations": 0},
        "actual_source_proof": "not_run", "voice_readback_acceptance": "not_run",
        "latency_acceptance": "not_run", "natural_routing_qualified": False,
    }


def main() -> int:
    # Core markers are content-free; keep the CLI to one bounded aggregate line.
    with contextlib.redirect_stdout(io.StringIO()):
        report = asyncio.run(run_controls())
    print("[confirmed-natural-work] " + json.dumps(report, sort_keys=True))
    return 0 if report["deterministic_confirmation"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
