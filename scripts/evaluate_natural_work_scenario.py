"""Bounded, effect-free actual-adapter scenarios against an externally owned runtime."""
# ruff: noqa: E402 -- candidate source bootstrap must precede production imports.

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hermes_realtime.conversation.context import ConversationContextStore, ConversationMessage
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.task_status import HostSpeechRequest
from hermes_realtime.conversation.tasks import TaskDispatchOutcome
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.host_launcher import (
    _build_streaming_inference as factory,
)
from hermes_realtime.host_launcher import (
    _run_full_host_preflight as preflight,
)
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.speech import AudioFrame, SpeechChunk, Transcript
from scripts import evaluate_natural_work_routing as legacy

MAX_INPUT_BYTES = 256 * 1024
PUBLIC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DEVELOPMENT_SHA256 = "92bd45d83ac643fa2d154995eb81a2c3d0ada11fc41c311bb3872293e5365070"
DEVELOPMENT_IDS = (
    "research", "greeting", "explanation", "quotation", "negation",
    "cancel_model_authority", "approval_model_authority", "ambiguity", "host_status",
)


def verify_imports() -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "hermes_realtime" or name.startswith("hermes_realtime."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(ROOT / "src"):
                raise ValueError("candidate_import_refused")
    if Path(legacy.__file__).resolve() != ROOT / "scripts/evaluate_natural_work_routing.py":
        raise ValueError("scorer_import_refused")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, expected: str | None = None) -> tuple[dict[str, Any], str]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    observed = hashlib.sha256(raw).hexdigest()
    if not 1 <= len(raw) <= MAX_INPUT_BYTES or expected is not None and observed != expected:
        raise ValueError("input_binding_refused")
    value = json.loads(
        raw,
        object_pairs_hook=legacy._object_without_duplicates,
        parse_constant=legacy._reject_json_constant,
    )
    if type(value) is not dict:
        raise ValueError("input_object_required")
    return value, observed


def exact(value: dict[str, Any], fields: set[str]) -> None:
    if (
        set(value) != fields
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("scenario_schema_refused")


def prior_rows(value: object) -> tuple[str, ...]:
    if type(value) is not list or not 0 <= len(value) <= 8:
        raise ValueError("prior_context_refused")
    store = ConversationContextStore()
    for text in value:
        store.validate_user_text(text)
        if "{workspace}" in text:
            raise ValueError("unresolved_workspace_refused")
    return tuple(value)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    corpus: legacy.IntentCorpus
    prior_user_turns: tuple[str, ...]
    controls: tuple[dict[str, Any], ...]
    bindings: dict[str, str]


@dataclass(frozen=True)
class DevelopmentDiagnostic:
    cases: tuple[legacy.IntentCase, ...]
    bindings: dict[str, str]
    scenario_id: str = "natural-work-development-nine-v1"


def load_development(path: Path) -> DevelopmentDiagnostic:
    """Select unchanged development rows; never admit the separate holdout corpus."""
    document, observed = read_json(path, DEVELOPMENT_SHA256)
    rows = {row["id"]: row for row in document["development_cases"]}
    rows["ambiguity"] = document["clarification_control"]
    cases = tuple(legacy.IntentCase(
        identifier, rows[identifier]["category"], rows[identifier]["utterance"],
        rows[identifier]["expected_tool"],
    ) for identifier in DEVELOPMENT_IDS[:-1])
    status = legacy.IntentCase("host_status", "negative", "How is work?", None)
    return DevelopmentDiagnostic(cases + (status,), {str(path.resolve()): observed})


def load_scenario(path: Path) -> Scenario:
    document, descriptor_sha = read_json(path)
    exact(document, {"schema_version", "scenario_id", "main", "context", "controls"})
    if type(document["scenario_id"]) is not str or not PUBLIC_ID.fullmatch(document["scenario_id"]):
        raise ValueError("scenario_id_refused")
    bindings = {str(path.resolve()): descriptor_sha}

    def referenced(name: str) -> dict[str, Any]:
        ref = document[name]
        if (
            type(ref) is not dict
            or set(ref) != {"path", "sha256"}
            or type(ref["path"]) is not str
            or type(ref["sha256"]) is not str
            or SHA256.fullmatch(ref["sha256"]) is None
        ):
            raise ValueError("reference_shape_refused")
        target = (path.parent / ref["path"]).resolve()
        if Path(ref["path"]).is_absolute() or not target.is_relative_to(path.parent.resolve()):
            raise ValueError("reference_escape_refused")
        result, observed = read_json(target, ref["sha256"])
        bindings[str(target)] = observed
        return result

    corpus = legacy.parse_corpus(referenced("main"))
    context = referenced("context")
    exact(context, {"schema_version", "prior_user_turns"})
    prior = prior_rows(context["prior_user_turns"])
    if not prior:
        raise ValueError("resource_context_required")
    control_document = referenced("controls")
    exact(control_document, {"schema_version", "pairs"})
    pairs, controls, identifiers = control_document["pairs"], [], set()
    if type(pairs) is not list or len(pairs) not in (2, 4):
        raise ValueError("control_pair_count_refused")
    for pair in pairs:
        if type(pair) is not dict or set(pair) != {
            "id",
            "utterance",
            "sufficient_prior_user_turns",
            "missing_prior_user_turns",
            "manual_criteria",
        }:
            raise ValueError("control_shape_refused")
        if (
            type(pair["id"]) is not str
            or not PUBLIC_ID.fullmatch(pair["id"])
            or pair["id"] in identifiers
        ):
            raise ValueError("control_id_refused")
        identifiers.add(pair["id"])
        ConversationContextStore().validate_user_text(pair["utterance"])
        criteria = prior_rows(pair["manual_criteria"])
        if not criteria:
            raise ValueError("manual_criteria_required")
        for role in ("sufficient", "missing"):
            rows = prior_rows(pair[role + "_prior_user_turns"])
            controls.append(
                {
                    "id": pair["id"] + ":" + role,
                    "utterance": pair["utterance"],
                    "prior_user_turns": rows,
                    "expected_start": role == "sufficient",
                    "manual_criteria": criteria,
                }
            )
    return Scenario(document["scenario_id"], corpus, prior, tuple(controls), bindings)


def verify_access(path: Path, expected: str, scenario_sha: str, scenario_id: str) -> None:
    proof, _ = read_json(path, expected)
    if (
        proof.get("passed") is not True
        or proof.get("scenario_sha256") != scenario_sha
        or proof.get("scenario_id") != scenario_id
    ):
        raise ValueError("access_proof_refused")


def make_context(case: legacy.IntentCase, revision: int, prior: tuple[str, ...]):
    canonical = legacy._snapshot(case, revision)
    if case.case_id == "host_status":
        canonical = replace(canonical, active_tasks=(legacy._ACTIVE_TASK,))
    context = ConversationContextStore()
    for text in (*prior, case.utterance):
        context.record_user_transcript(Transcript(text=text, final=True))
    for index, task in enumerate(canonical.active_tasks):
        context.record_task_accepted(
            task_id=task.task_id, run_id=f"deleg_scenario_{index}", objective=task.objective
        )
    for index in range(canonical.terminal_task_count):
        context.record_task_accepted(
            task_id=f"task_ended_{index}",
            run_id=f"deleg_ended_{index}",
            objective="Synthetic ended work for qualification.",
        )
        context.record_task_completed(task_id=f"task_ended_{index}", run_id=f"deleg_ended_{index}")
    observed = context.snapshot()
    if observed.messages != tuple(ConversationMessage("user", x) for x in (*prior, case.utterance)):
        raise ValueError("context_retention_refused")
    if (observed.active_tasks, observed.terminal_task_count, observed.memory) != (
        canonical.active_tasks,
        canonical.terminal_task_count,
        canonical.memory,
    ):
        raise ValueError("context_authority_refused")
    return context, ConversationInferenceRequest(
        revision=revision,
        messages=observed.messages,
        active_tasks=observed.active_tasks,
        terminal_task_count=observed.terminal_task_count,
        memory=observed.memory,
        updates=(),
    )


class Recorder:
    def __init__(self):
        self.objectives, self.requests, self.latencies, self.events = [], [], [], []
        self.started = 0.0

    async def dispatch(self, *, objective, utterance_id):
        self.objectives.append(objective)
        return TaskDispatchOutcome(task_id="task_shadow", accepted=False, reason="shadow rejection")

    async def request_cancel(self, *args, **kwargs):
        raise RuntimeError("unexpected_cancellation")


class ReadinessDouble:
    async def start(self):
        return None

    async def synthesize(self, text, turn_id):
        yield SpeechChunk(
            text=text,
            turn_id=turn_id,
            chunk_id="chunk_preflight",
            audio=AudioFrame(pcm=b"\0\0", sample_rate_hz=24000, channels=1),
        )


def write_private_row(sink, row: dict[str, Any]) -> None:
    raw = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    if sink.tell() + len(raw) > 8 * 1024 * 1024:
        raise ValueError("private_receipt_bound")
    if sink.write(raw) != len(raw):
        raise OSError("private_receipt_write_failed")
    sink.flush()


async def run_cases(
    value: Scenario | DevelopmentDiagnostic,
    *,
    model: str,
    base_url: str,
    observations: list[dict[str, Any]] | None = None,
    private_sink=None,
) -> list[dict[str, Any]]:
    verify_imports()
    sequence: list[tuple[legacy.IntentCase, tuple[str, ...], str]]
    if isinstance(value, DevelopmentDiagnostic):
        sequence = [(case, (), "development") for case in value.cases]
    else:
        sequence = [(case, value.prior_user_turns, "main") for case in value.corpus.cases]
        sequence += [
            (
                legacy.IntentCase(row["id"], "negative", row["utterance"], None),
                row["prior_user_turns"],
                "control",
            )
            for row in value.controls
        ]
    results = observations if observations is not None else []
    for revision, (case, prior, cohort) in enumerate(sequence, 1):
        context, request = make_context(case, revision, prior)
        recorder = Recorder()
        surface = ConversationWorkControlSurface(
            controller=recorder,
            context=context,
            observer=lambda *args: None,
            reserve_observer_capacity=lambda: None,
        )

        async def observed_start(
            *, objective, invocation_id, original=surface.start_work, observer=recorder
        ):
            observer.requests.append(objective)
            observer.latencies.append((time.monotonic() - observer.started) * 1000)
            result = await original(objective=objective, invocation_id=invocation_id)
            observer.events.append("ack_accepted" if result.accepted else "ack_rejected")
            return result

        surface.start_work = observed_start
        inference = factory(
            inference_provider="ollama",
            ollama_base_url=base_url,
            ollama_model=model,
            codex_model="",
            codex_effort="",
            codex_executable=None,
        )
        row = {
            "id": case.case_id,
            "cohort": cohort,
            "usable_decision": False,
            "error": None,
            "speech": [],
            "objectives": recorder.requests,
            "shadow_dispatches": recorder.objectives,
            "first_speakable_ms": None,
            "host_status_request_ms": None,
            "latencies_ms": recorder.latencies,
            "readiness_attempted": False,
            "decision_attempted": False,
        }
        failures: list[BaseException] = []
        try:
            if type(inference) is not OllamaStreamingInference:
                raise ValueError("factory_provider_refused")
            await inference.configure_context(4096)
            row["readiness_attempted"] = True
            await asyncio.wait_for(
                preflight(
                    task_session=ReadinessDouble(),
                    inference=inference,
                    synthesizer=ReadinessDouble(),
                    surface=surface,
                    natural_work_tools=True,
                ),
                60,
            )
            recorder.started = time.monotonic()

            async def collect(
                inference=inference, request=request, revision=revision, row=row, recorder=recorder
            ):
                async for segment in inference.stream(request, turn_id=f"turn_scenario_{revision}"):
                    if row["host_status_request_ms"] is not None:
                        raise RuntimeError("host_selection_not_terminal")
                    if type(segment) is HostSpeechRequest:
                        if recorder.requests or row["speech"]:
                            raise RuntimeError("host_selection_mixed_output")
                        row["host_status_request_ms"] = (
                            time.monotonic() - recorder.started) * 1000
                        recorder.events.append("host_status_selected")
                        continue
                    if type(segment) is not str:
                        raise RuntimeError("unsupported_inference_output")
                    if row["first_speakable_ms"] is None and legacy._is_speakable(segment):
                        row["first_speakable_ms"] = (time.monotonic() - recorder.started) * 1000
                    recorder.events.append("speech")
                    row["speech"].append(segment)

            row["decision_attempted"] = True
            try:
                await asyncio.wait_for(collect(), 60)
            except ValueError as error:
                if str(error) not in {
                    "Ollama work refused: " + category
                    for category in (
                        "completion_reason",
                        "incomplete",
                        "calls_type",
                        "unexpected_native_calls",
                        "decision_unicode",
                        "unverified_claim",
                        "duplicate_keys",
                        "decision_json",
                        "decision_schema",
                        "unadvertised",
                        "objective_type",
                        "objective_blank",
                        "objective_length",
                        "objective_unicode",
                        "text_type",
                        "text_blank",
                        "text_length",
                        "text_unicode",
                        "private_authority",
                    )
                }:
                    raise
                row["error"] = "decision_refused"
            if recorder.requests and (
                row["speech"] != ["The background task was not accepted."]
                or recorder.events != ["ack_rejected", "speech"]
            ):
                raise RuntimeError("ack_witness_refused")
            row["usable_decision"] = row["error"] is None
        except BaseException as error:
            row["error"] = "operational_failure"
            failures.append(error)
        finally:
            for owner in (inference, surface):
                try:
                    await asyncio.wait_for(owner.close(), 10)
                except BaseException as error:
                    row["usable_decision"] = False
                    row["cleanup_failed"] = True
                    row["error"] = row["error"] or "cleanup_failure"
                    failures.append(error)
            results.append(row)
            if private_sink is not None:
                try:
                    write_private_row(private_sink, row)
                except BaseException as error:
                    failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("scenario decision and cleanup failed", failures)
    return results


def build_development_report(
    value: DevelopmentDiagnostic, rows: list[dict[str, Any]], *, model: str,
) -> dict[str, Any]:
    if [row["id"] for row in rows] != [case.case_id for case in value.cases]:
        raise ValueError("complete_order_required")
    observations, selections = [], []
    for case, row in zip(value.cases, rows, strict=True):
        expected = ("start_work" if case.expected_tool == "start_work" else (
            "task_status" if case.case_id in ("cancel_model_authority", "host_status")
            else "reply"))
        observed = ("task_status" if row["host_status_request_ms"] is not None else (
            "start_work" if row["objectives"] else (
                "reply" if any(legacy._is_speakable(text) for text in row["speech"])
                else "invalid")))
        observations.append({"id": case.case_id, "expected": expected, "observed": observed,
                             "matched": row["usable_decision"] and observed == expected})
        if row["host_status_request_ms"] is not None:
            selections.append(row["host_status_request_ms"])
    return {
        "schema_version": 1, "diagnostic_id": value.scenario_id, "diagnostic_only": True,
        "fixture_sha256": list(value.bindings.values()), "model": model,
        "cases": observations, "structural_pass": all(row["matched"] for row in observations),
        "objective_fidelity": "pending", "lifecycle_review": "pending",
        "clarification_review": "pending", "pass": False,
        "host_status_requests": {"scope": "selection_only", "count": len(selections),
                                 "latency_ms": legacy.latency_distribution(selections)},
        "first_speakable_ms": legacy.latency_distribution([
            row["first_speakable_ms"] for row in rows if row["first_speakable_ms"] is not None]),
        "start_proposal_ms": legacy.latency_distribution([
            latency for row in rows for latency in row["latencies_ms"]]),
        "real_effects": 0, "runtime_ownership": "external",
        "attempt_policy": "single_attempt_per_case", "context_tokens": 4096,
        "maximum_readiness": 9, "maximum_decisions": 9,
    }


def build_report(value: Scenario, rows: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    expected = [case.case_id for case in value.corpus.cases] + [row["id"] for row in value.controls]
    if [row["id"] for row in rows] != expected:
        raise ValueError("complete_order_required")
    observations = [
        legacy.CaseObservation(
            case_id=row["id"],
            observed_tools=tuple("start_work" for _ in row["objectives"]),
            tool_call_latencies_ms=tuple(row["latencies_ms"]),
            first_speakable_latency_ms=row["first_speakable_ms"],
        )
        for row in rows[:100]
    ]
    routing = legacy.routing_report(
        corpus=value.corpus,
        observations=observations,
        model=model,
        effort="none",
        codex_version="not_executed",
        retried_case_ids=(),
    )
    matched = all(
        bool(row["objectives"]) == control["expected_start"] and row["usable_decision"]
        for row, control in zip(rows[100:], value.controls, strict=True)
    )
    return {
        "schema_version": 1,
        "scenario_id": value.scenario_id,
        "input_sha256": list(value.bindings.values()),
        "routing": routing,
        "framing_pass": all(row["usable_decision"] for row in rows),
        "controls": {
            "total": len(value.controls),
            "structural_pass": matched,
            "manual_review": "pending",
        },
        "objective_fidelity": "pending",
        "pass": False,
        "errors": sum(row["error"] is not None for row in rows),
        "decisions": len(rows),
        "readiness_calls": len(rows),
        "real_effects": 0,
        "runtime_ownership": "external",
        "speech_preflight": "synthetic_pcm_not_voice_acceptance",
    }


def source_binding() -> dict[str, Any]:
    def git(*args):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args],
            text=True,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            timeout=10,
        ).strip()

    if git("remote", "get-url", "origin") != "https://github.com/sushiHex/hermes-realtime.git":
        raise ValueError("source_origin_refused")
    source = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in sorted((ROOT / "src").rglob("*.py"))
    }
    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "runner_sha256": digest(Path(__file__)),
        "scorer_sha256": digest(Path(legacy.__file__)),
        "dirty": bool(git("status", "--porcelain")),
    }


def verify_execution_binding(binding: dict[str, Any], expected_head: str, live: bool) -> None:
    if binding["head"] != expected_head or live and binding["dirty"]:
        raise ValueError("execution_binding_refused")


def verify_stability(value: Scenario | DevelopmentDiagnostic, before: dict[str, Any]) -> None:
    if source_binding() != before or any(digest(Path(p)) != h for p, h in value.bindings.items()):
        raise ValueError("execution_input_drift")


def main(argv: list[str] | None = None) -> int:
    parser = legacy._JsonArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scenario", type=Path)
    inputs.add_argument("--development-fixture", type=Path)
    parser.add_argument("--access-proof", type=Path)
    parser.add_argument("--access-proof-sha")
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--private-receipt", type=Path)
    parser.add_argument("--live", action="store_true")
    args = None
    before = None
    report: dict[str, Any]
    rows: list[dict[str, Any]]
    report, code, rows, receipt = {"schema_version": 1, "pass": False}, 2, [], None
    try:
        args = parser.parse_args(argv)
        verify_imports()
        before = source_binding()
        verify_execution_binding(before, args.source_head, args.live)
        if not PUBLIC_ID.fullmatch(args.model):
            raise ValueError("execution_binding_refused")
        value: Scenario | DevelopmentDiagnostic
        if args.development_fixture is not None:
            if args.access_proof is not None or args.access_proof_sha is not None:
                raise ValueError("development_resource_override_refused")
            value = load_development(args.development_fixture)
        else:
            if args.access_proof is None or args.access_proof_sha is None:
                raise ValueError("scenario_access_required")
            value = load_scenario(args.scenario)
            verify_access(
                args.access_proof, args.access_proof_sha,
                value.bindings[str(args.scenario.resolve())], value.scenario_id,
            )
            value.bindings[str(args.access_proof.resolve())] = args.access_proof_sha
        if not args.live:
            report.update(
                status="prepared_only", scenario_id=value.scenario_id, source_binding=before
            )
            code = 0
        else:
            if args.private_receipt is None:
                raise ValueError("private_receipt_required")
            receipt = args.private_receipt.open("xb", buffering=0)
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                try:
                    asyncio.run(
                        asyncio.wait_for(
                            run_cases(
                                value,
                                model=args.model,
                                base_url=args.base_url,
                                observations=rows,
                                private_sink=receipt,
                            ),
                            1300 if isinstance(value, DevelopmentDiagnostic)
                            else (100 + len(value.controls)) * 140 + 240,
                        )
                    )
                except BaseException:
                    try:
                        verify_stability(value, before)
                    except Exception:
                        report["secondary_categories"] = ["execution_input_drift"]
                    raise
            verify_stability(value, before)
            report = (build_development_report(value, rows, model=args.model)
                      if isinstance(value, DevelopmentDiagnostic)
                      else build_report(value, rows, model=args.model))
            report["source_binding"] = before
            code = 1  # Manual semantic verdicts remain pending.
    except Exception:
        report.update(status="error", error="scenario_evaluation_failed")
    finally:
        if before is not None:
            report["source_binding"] = before
        if receipt is not None:
            receipt.close()
        report.update(
            readiness_attempts=sum(row["readiness_attempted"] for row in rows),
            decision_attempts=sum(row["decision_attempted"] for row in rows),
        )
        print("[natural-work-scenario] " + json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
