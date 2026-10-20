"""One frozen Phi proposal diagnostic; run only inside an externally owned controller.

Metadata/endpoint ownership must be attested by the caller before --live. This
probe never starts an Ollama server, changes its store, or confirms a proposal.
"""
# ruff: noqa: E402 -- bind candidate imports before production modules.

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hermes_realtime.conversation.task_proposals import HostTaskProposalRequest
from hermes_realtime.conversation.task_status import HostSpeechRequest
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.host_launcher import _build_streaming_inference as factory
from hermes_realtime.host_launcher import _run_full_host_preflight as preflight
from hermes_realtime.providers.ollama import OllamaStreamingInference
from scripts import evaluate_natural_work_routing as legacy
from scripts import evaluate_natural_work_scenario as harness
from scripts import qualify_confirmed_natural_work as qualification

_NUMERIC = frozenset({
    "load_duration_ns", "prompt_eval_duration_ns", "eval_duration_ns", "total_duration_ns",
    "prompt_eval_count", "num_ctx", "prompt_bytes", "prompt_chars", "messages",
    "timing_invalid_count",
})


class PromptTimingSink(io.TextIOBase):
    """Discard diagnostics except a bounded allowlist of numeric provider timings."""

    def __init__(self) -> None:
        self.pending = ""
        self.rows: list[dict[str, Any]] = []
        self.stage = "readiness"

    def write(self, value: str) -> int:
        self.pending += value
        if len(self.pending) > 65536:
            raise ValueError("timing_line_bound")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.startswith("[ollama-prompt] "):
                document = json.loads(line[len("[ollama-prompt] "):])
                row = {key: document[key] for key in _NUMERIC if key in document}
                if any(value is not None and (type(value) is not int or not 0 <= value < 2**63)
                       for value in row.values()):
                    raise ValueError("timing_numeric")
                self.rows.append({"stage": self.stage, **row})
        return len(value)


async def collect_decision(inference: Any, request: Any, turn_id: str,
                           row: dict[str, Any]) -> None:
    started = time.monotonic()
    terminal = False
    row.update(selection=None, speech=[], proposed_objective=None, first_speakable_ms=None)
    async for output in inference.stream(request, turn_id=turn_id):
        if terminal:
            raise ValueError("selection_terminal")
        if type(output) is HostTaskProposalRequest:
            if row["speech"]:
                raise ValueError("selection_terminal")
            row.update(selection="proposal", proposed_objective=output.objective)
            terminal = True
        elif type(output) is HostSpeechRequest:
            if row["speech"]:
                raise ValueError("selection_terminal")
            row["selection"] = "task_status"
            terminal = True
        elif type(output) is str:
            row["speech"].append(output)
            if row["first_speakable_ms"] is None and legacy._is_speakable(output):
                row["first_speakable_ms"] = (time.monotonic() - started) * 1000
            row["selection"] = "reply"
        else:
            raise ValueError("selection_type")
    if row["selection"] is None:
        raise ValueError("selection_empty")
    row["decision_elapsed_ms"] = (time.monotonic() - started) * 1000


async def run_cases(*, model: str, base_url: str, private_sink: Any,
                    model_digest: str, observations: list[dict[str, Any]] | None = None,
                    ) -> list[dict[str, Any]]:
    fixture = qualification.load_preregistration()
    rows = observations if observations is not None else []
    for revision, case in enumerate(fixture["model_cases"], 1):
        context, request = harness.make_context(
            legacy.IntentCase(case["id"], "negative", case["utterance"], None),
            revision, tuple(case.get("prior_user_turns", ())), active=False,
        )
        recorder = harness.Recorder()  # type: ignore[no-untyped-call]
        surface = ConversationWorkControlSurface(
            controller=recorder, context=context, observer=lambda *args: None,
            reserve_observer_capacity=lambda: None,
        )
        inference = factory(
            inference_provider="ollama", ollama_base_url=base_url, ollama_model=model,
            codex_model="", codex_effort="", codex_executable=None,
        )
        row = {"id": case["id"], "expected": case["expected"], "error": None,
               "status": "attempted",
               "readiness_attempted": False, "decision_attempted": False,
               "selection": None, "dispatches": 0, "timings": []}
        timing = PromptTimingSink()
        try:
            with contextlib.redirect_stdout(timing), contextlib.redirect_stderr(timing):
                if type(inference) is not OllamaStreamingInference:
                    raise ValueError("provider_refused")
                await inference.configure_context(4096)
                if inference.context_public_data()["modelDigest"] != model_digest:
                    raise ValueError("model_digest_refused")
                row["readiness_attempted"] = True
                await asyncio.wait_for(preflight(
                    task_session=harness.ReadinessDouble(), inference=inference,
                    synthesizer=harness.ReadinessDouble(), surface=surface, natural_work_tools=True,
                ), 60)
                timing.stage = "decision"
                row["decision_attempted"] = True
                await asyncio.wait_for(collect_decision(
                    inference, request, f"confirmed_probe_{revision}", row,
                ), 60)
        except Exception:
            row["error"] = "attempt_failed"
            raise
        finally:
            try:
                with contextlib.redirect_stdout(timing), contextlib.redirect_stderr(timing):
                    failures = []
                    for owner in (inference, surface):
                        try:
                            await asyncio.wait_for(owner.close(), 10)
                        except Exception as error:
                            failures.append(error)
                    if failures:
                        row["error"] = "cleanup_failed"
                        raise ExceptionGroup("probe cleanup failed", failures)
            finally:
                row["dispatches"] = len(recorder.objectives)
                row["timings"] = timing.rows
                rows.append(row)
                harness.write_private_row(private_sink, row)
    return rows


def record_unrun(rows: list[dict[str, Any]], private_sink: Any) -> None:
    """Preserve the frozen tail after an aborted attempt without executing it."""
    cases = qualification.load_preregistration()["model_cases"]
    for case in cases[len(rows):]:
        row = {"id": case["id"], "expected": case["expected"], "status": "not_run",
               "error": None, "selection": None, "dispatches": 0, "timings": []}
        rows.append(row)
        harness.write_private_row(private_sink, row)


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row["status"] == "attempted"]
    usable = [row for row in attempted if row["error"] is None]
    return {
        "schema_version": 1, "scenario_id": "confirmed-natural-work-v1",
        "fixture_sha256": qualification.FIXTURE_SHA256,
        "denominator": 8, "attempts": len(attempted), "unrun": 8 - len(attempted),
        "errors": len(attempted) - len(usable),
        "readiness_attempts": sum(row.get("readiness_attempted", False) for row in attempted),
        "decision_attempts": sum(row.get("decision_attempted", False) for row in attempted),
        "case_outcomes": [{"id": row["id"], "status": row["status"], "error": row["error"]}
                          for row in rows],
        "selections": dict(Counter(row["selection"] for row in usable)),
        "unwanted_proposals": sum(row["selection"] == "proposal" and row["expected"] != "proposal"
                                  for row in usable),
        "missed_proposals": sum(row["selection"] != "proposal" and row["expected"] == "proposal"
                                for row in usable),
        "dispatches": sum(row["dispatches"] for row in rows),
        "dispatch_mode": "rejecting_shadow", "real_effects": 0,
        "stage_timings": [{"id": row["id"], "timings": row["timings"],
                           "decision_elapsed_ms": row.get("decision_elapsed_ms"),
                           "first_speakable_ms": row.get("first_speakable_ms")} for row in rows],
        "objective_fidelity": "pending", "clarification_review": "pending",
        "lifecycle_claims": "pending", "actual_source_proof": "not_run",
        "voice_acceptance": "not_run", "natural_routing_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--model", choices=("phi4-mini:latest",), default="phi4-mini:latest")
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    before = None
    report: dict[str, Any] = {"status": "refused", "natural_routing_qualified": False}
    try:
        harness.verify_imports()
        before = harness.source_binding()
        before["probe_sha256"] = harness.digest(Path(__file__))
        harness.verify_execution_binding(before, args.source_head, args.live)
        if (before["tree"] != args.source_tree
                or harness.SHA256.fullmatch(args.model_digest) is None):
            raise ValueError("execution_binding_refused")
        qualification.load_preregistration()
        report = {"status": "prepared_only", "fixture_sha256": qualification.FIXTURE_SHA256}
        if args.live:
            with args.private_receipt.open("xb", buffering=0) as receipt:
                try:
                    asyncio.run(asyncio.wait_for(run_cases(
                        model=args.model, base_url=args.base_url, model_digest=args.model_digest,
                        private_sink=receipt, observations=rows,
                    ), 1000))
                finally:
                    record_unrun(rows, receipt)
            report = build_report(rows)
            report["status"] = (
                "completed" if report["attempts"] == 8 and report["errors"] == 0
                else "attempt_failed"
            )
        after = harness.source_binding()
        after["probe_sha256"] = harness.digest(Path(__file__))
        if after != before:
            raise ValueError("execution_input_drift")
    except Exception:
        report = build_report(rows) | {"status": "attempt_failed"}
    finally:
        report.update(source_binding=before, model=args.model, model_digest=args.model_digest,
                      natural_routing_qualified=False)
        print("[confirmed-natural-probe] " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") in {"prepared_only", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
