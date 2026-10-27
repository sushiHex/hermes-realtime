"""One foreground proposal diagnostic; run only inside an externally owned controller.

Metadata/endpoint ownership must be attested by the caller before --live. This
probe never starts an Ollama server, changes its store, or confirms a proposal.
Ollama retains its existing defaults; Codex requires an explicit model, effort,
and byte-bound executable. Neither selection authorizes a live invocation.
"""
# ruff: noqa: E402 -- bind candidate imports before production modules.

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hermes_realtime.conversation.task_proposals import HostTaskProposalRequest
from hermes_realtime.conversation.task_status import HostSpeechRequest
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.host_launcher import _build_streaming_inference as factory
from hermes_realtime.host_launcher import _run_full_host_preflight as preflight
from hermes_realtime.providers.codex_app_server import (
    _REASONING_EFFORTS,
    CodexAppServerStreamingInference,
)
from hermes_realtime.providers.ollama import OllamaStreamingInference
from scripts import evaluate_natural_work_routing as legacy
from scripts import evaluate_natural_work_scenario as harness
from scripts import qualify_confirmed_natural_work as qualification

V2_FIXTURE = ROOT / "tests/fixtures/confirmed_natural_work_v2.json"
V2_FIXTURE_SHA256 = "6b6034f58ac5a3be65b80c257154e83186d0174ccefb3b94cc485eaed99eed53"


def load_fixture(version: str = "v1") -> dict[str, Any]:
    if version == "v1":
        return qualification.load_preregistration()
    admitted = False
    try:
        if version != "v2":
            raise ValueError("preregistration_binding")
        raw = V2_FIXTURE.read_bytes()
        if hashlib.sha256(raw).hexdigest() != V2_FIXTURE_SHA256:
            raise ValueError("preregistration_binding")
        result = cast(dict[str, Any], json.loads(raw))
        admitted = True
        return result
    finally:
        if not admitted:
            print('[confirmed-preregistration] {"category":"binding_refused","count":1}')


def fixture_metadata(version: str) -> dict[str, Any]:
    return {
        "scenario_id": "confirmed-natural-work-" + version,
        "fixture_sha256": V2_FIXTURE_SHA256 if version == "v2" else qualification.FIXTURE_SHA256,
        "evaluation_kind": (
            "exposed_development_comparison" if version == "v2" else "frozen_initial_diagnostic"
        ),
    }


_NUMERIC = frozenset({
    "load_duration_ns", "prompt_eval_duration_ns", "eval_duration_ns", "total_duration_ns",
    "prompt_eval_count", "num_ctx", "prompt_bytes", "prompt_chars", "messages",
    "timing_invalid_count",
})


def codex_error_diagnostic(message: object) -> dict[str, object] | None:
    """Project a Codex 0.145.0 ``error`` notification without retaining its text.

    The pinned app-server's generated v2/ErrorNotification schema defines these
    codexErrorInfo variants and nullable uint16 httpStatusCode fields. This is
    evidence only: it does not validate, suppress, retry or reinterpret provider
    events. Unknown/malformed codes stay unknown. Stage and cleanup ownership
    remain with the caller, separately from this content-free projection.
    """
    if (type(message) is not dict or type(message.get("method")) is not str
            or message.get("method") != "error"):
        return None
    result: dict[str, object] = {"category": "provider_error", "code": "unknown"}
    params = message.get("params")
    if type(params) is not dict:
        return result
    if type(params.get("willRetry")) is bool:
        result["will_retry"] = params["willRetry"]
    error = params.get("error")
    if type(error) is not dict:
        return result
    info = error.get("codexErrorInfo")
    if type(info) is str:
        if info in {
            "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
            "serverOverloaded", "cyberPolicy", "internalServerError", "unauthorized",
            "badRequest", "threadRollbackFailed", "sandboxError", "other",
        }:
            result["code"] = info
    elif type(info) is dict and len(info) == 1:
        code, detail = next(iter(info.items()))
        if type(code) is not str or type(detail) is not dict:
            return result
        if code in {
            "httpConnectionFailed", "responseStreamConnectionFailed",
            "responseStreamDisconnected", "responseTooManyFailedAttempts",
        }:
            status = detail.get("httpStatusCode")
            if status is not None and (type(status) is not int or not 0 <= status <= 65535):
                return result
            result["code"] = code
            if status is not None:
                result["http_status_code"] = status
        elif (code == "activeTurnNotSteerable" and type(detail.get("turnKind")) is str
              and detail.get("turnKind") in {"review", "compact"}):
            result["code"] = code
    return result


def failure_diagnostic(error: BaseException, *, stage: str, started: float) -> dict[str, Any]:
    """Classify exposed exception types only; never inspect text or chained causes.

    Preflight currently exposes readiness refusals as generic RuntimeError, so
    they remain `other` at stage `readiness`, rather than inferring a cause.
    """
    if isinstance(error, asyncio.CancelledError):
        category = "cancelled"
    elif isinstance(error, TimeoutError):
        category = "timeout"
    elif isinstance(error, (HTTPError, aiohttp.ClientResponseError)):
        category = "http"
    elif isinstance(error, (aiohttp.ClientError, URLError, ConnectionError)):
        category = "transport"
    else:
        category = "other"
    return {"stage": stage, "category": category,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}


def verify_provider_configuration(
    *, inference_provider: str, model: str | None, base_url: str | None,
    model_digest: str | None, codex_effort: str | None, codex_executable: str | None,
    codex_executable_sha256: str | None,
) -> None:
    """Refuse mixed selections and bind Codex bytes without running its executable."""
    category: str | None = "model_identifier_refused"
    try:
        if (type(model) is not str or len(model) > 256
                or re.fullmatch(
                    r"(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*"
                    r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?", model,
                ) is None):
            raise ValueError("model_identifier_refused")
        category = "provider_configuration_refused"
        if inference_provider == "ollama":
            if (not base_url or model_digest is None
                    or harness.SHA256.fullmatch(model_digest) is None
                    or any(value is not None for value in (
                        codex_effort, codex_executable, codex_executable_sha256,
                    ))):
                raise ValueError("provider_configuration_refused")
        elif inference_provider == "codex":
            if (base_url is not None or model_digest is not None
                    or codex_effort not in _REASONING_EFFORTS
                    or not codex_executable or codex_executable_sha256 is None
                    or harness.SHA256.fullmatch(codex_executable_sha256) is None):
                raise ValueError("provider_configuration_refused")
            category = "codex_executable_unreadable"
            actual_digest = harness.digest(Path(codex_executable))
            category = "codex_executable_binding_refused"
            if actual_digest != codex_executable_sha256:
                raise ValueError("codex_executable_binding_refused")
        else:
            category = "provider_refused"
            raise ValueError("provider_refused")
        category = None
    finally:
        if category is not None:
            print("[confirmed-provider-configuration] " + json.dumps(
                {"category": category, "count": 1}, sort_keys=True,
            ))


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
                           row: dict[str, Any], *,
                           clock: Callable[[], float] | None = None) -> None:
    """Consume production outputs through finalization, retaining first usable selection.

    ``decision_elapsed_ms`` remains the legacy generator-wide timing alias. Use
    ``first_usable_decision_ms`` for reply/proposal/status availability. A yielded
    decision never suppresses a subsequent provider/finalization failure.
    """
    now = clock or time.monotonic
    started = now()
    terminal = False
    row.update(selection=None, speech=[], proposed_objective=None, first_speakable_ms=None,
               first_usable_decision_ms=None, generator_elapsed_ms=None, decision_elapsed_ms=None)
    try:
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
                    row["first_speakable_ms"] = (now() - started) * 1000
                    row["first_usable_decision_ms"] = row["first_speakable_ms"]
                row["selection"] = "reply"
            else:
                raise ValueError("selection_type")
            if terminal:
                row["first_usable_decision_ms"] = (now() - started) * 1000
        if row["first_usable_decision_ms"] is None:
            raise ValueError("selection_empty")
    finally:
        row["generator_elapsed_ms"] = row["decision_elapsed_ms"] = (now() - started) * 1000


async def _close_owners(inference: Any, surface: ConversationWorkControlSurface,
                        timing: PromptTimingSink, row: dict[str, Any]) -> None:
    with contextlib.redirect_stdout(timing), contextlib.redirect_stderr(timing):
        failures = []
        for owner in (inference, surface):
            cleanup_started = time.monotonic()
            try:
                await asyncio.wait_for(owner.close(), 10)
            except (Exception, asyncio.CancelledError) as error:
                failures.append(RuntimeError("owner cleanup cancelled")
                                if isinstance(error, asyncio.CancelledError) else error)
                row["cleanup_failures"].append(failure_diagnostic(
                    error, stage="cleanup", started=cleanup_started,
                ))
        if failures:
            row["error"] = "cleanup_failed"
            raise ExceptionGroup("probe cleanup failed", failures)


async def run_cases(*, model: str, private_sink: Any, base_url: str | None = None,
                    model_digest: str | None = None,
                    inference_provider: str = "ollama", codex_effort: str | None = None,
                    codex_executable: str | None = None,
                    codex_executable_sha256: str | None = None,
                    observations: list[dict[str, Any]] | None = None,
                    fixture_version: str = "v1",
                    clock: Callable[[], float] | None = None,
                    ) -> list[dict[str, Any]]:
    verify_provider_configuration(
        inference_provider=inference_provider, model=model, base_url=base_url,
        model_digest=model_digest, codex_effort=codex_effort, codex_executable=codex_executable,
        codex_executable_sha256=codex_executable_sha256,
    )
    now = clock or time.monotonic
    fixture = load_fixture(fixture_version)
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
            inference_provider=inference_provider, ollama_base_url=base_url or "",
            ollama_model=model if inference_provider == "ollama" else "",
            codex_model=model if inference_provider == "codex" else "",
            codex_effort=codex_effort or "", codex_executable=codex_executable,
        )
        row = {"id": case["id"], "expected": case["expected"], "error": None,
               "status": "attempted",
               "readiness_attempted": False, "decision_attempted": False,
               "selection": None, "dispatches": 0, "timings": [],
               "operational_failure": None, "cleanup_failures": []}
        timing = PromptTimingSink()
        stage = "configuration"
        started = time.monotonic()
        try:
            with contextlib.redirect_stdout(timing), contextlib.redirect_stderr(timing):
                expected_type = (OllamaStreamingInference if inference_provider == "ollama"
                                 else CodexAppServerStreamingInference)
                if type(inference) is not expected_type:
                    raise ValueError("provider_refused")
                if inference_provider == "ollama":
                    await inference.configure_context(4096)
                    if inference.context_public_data()["modelDigest"] != model_digest:
                        raise ValueError("model_digest_refused")
                stage = "readiness"
                started = time.monotonic()
                row["readiness_attempted"] = True
                await asyncio.wait_for(preflight(
                    task_session=harness.ReadinessDouble(), inference=inference,
                    synthesizer=harness.ReadinessDouble(), surface=surface, natural_work_tools=True,
                ), 60)
                stage = "decision"
                started = time.monotonic()
                timing.stage = "decision"
                row["decision_attempted"] = True
                await asyncio.wait_for(collect_decision(
                    inference, request, f"confirmed_probe_{revision}", row, clock=now,
                ), 60)
        except (Exception, asyncio.CancelledError) as error:
            row["error"] = "attempt_failed"
            row["operational_failure"] = failure_diagnostic(error, stage=stage, started=started)
            raise
        finally:
            cleanup_started_all = now()
            cleanup_stage_started = time.monotonic()
            cleanup = asyncio.create_task(_close_owners(inference, surface, timing, row))
            cancelled = None
            try:
                # The owner operations retain their individual ten-second bounds.
                # Caller cancellation must wait for both to settle before returning.
                while True:
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as error:
                        cancelled = error
                        row["error"] = row["error"] or "attempt_failed"
                        if row["operational_failure"] is None:
                            row["operational_failure"] = failure_diagnostic(
                                error, stage="cleanup", started=cleanup_stage_started,
                            )
                        continue
                    except Exception:
                        if cancelled is None:
                            raise
                    break
                if cancelled is not None:
                    raise cancelled
            finally:
                row["cleanup_elapsed_ms"] = (now() - cleanup_started_all) * 1000
                row["dispatches"] = len(recorder.objectives)
                row["timings"] = timing.rows
                rows.append(row)
                harness.write_private_row(private_sink, row)
    return rows


def record_unrun(rows: list[dict[str, Any]], private_sink: Any,
                 *, fixture_version: str = "v1") -> None:
    """Preserve the frozen tail after an aborted attempt without executing it."""
    cases = load_fixture(fixture_version)["model_cases"]
    for case in cases[len(rows):]:
        row = {"id": case["id"], "expected": case["expected"], "status": "not_run",
               "error": None, "selection": None, "dispatches": 0, "timings": []}
        rows.append(row)
        harness.write_private_row(private_sink, row)


def build_report(rows: list[dict[str, Any]], *, fixture_version: str = "v1") -> dict[str, Any]:
    attempted = [row for row in rows if row["status"] == "attempted"]
    usable = [row for row in attempted if row["error"] is None]
    return {
        "schema_version": 1, **fixture_metadata(fixture_version),
        "denominator": 8, "attempts": len(attempted), "unrun": 8 - len(attempted),
        "errors": len(attempted) - len(usable),
        "readiness_attempts": sum(row.get("readiness_attempted", False) for row in attempted),
        "decision_attempts": sum(row.get("decision_attempted", False) for row in attempted),
        "case_outcomes": [{"id": row["id"], "status": row["status"], "error": row["error"],
                           "operational_failure": row.get("operational_failure"),
                           "cleanup_failures": row.get("cleanup_failures", [])}
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
                           "first_usable_decision_ms": row.get("first_usable_decision_ms"),
                           "generator_elapsed_ms": row.get("generator_elapsed_ms"),
                           "cleanup_elapsed_ms": row.get("cleanup_elapsed_ms"),
                           "first_speakable_ms": row.get("first_speakable_ms")} for row in rows],
        "objective_fidelity": "pending", "clarification_review": "pending",
        "lifecycle_claims": "pending", "actual_source_proof": "not_run",
        "voice_acceptance": "not_run", "natural_routing_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--inference-provider", default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--model-digest")
    parser.add_argument("--base-url")
    parser.add_argument("--codex-effort")
    parser.add_argument("--codex-executable")
    parser.add_argument("--codex-executable-sha256")
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--fixture-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    before = None
    report: dict[str, Any] = {"status": "refused", "natural_routing_qualified": False}
    configuration: dict[str, Any] = {}
    try:
        model = args.model
        if model is None and args.inference_provider == "ollama":
            model = "phi4-mini:latest"
        verify_provider_configuration(
            inference_provider=args.inference_provider, model=model, base_url=args.base_url,
            model_digest=args.model_digest, codex_effort=args.codex_effort,
            codex_executable=args.codex_executable,
            codex_executable_sha256=args.codex_executable_sha256,
        )
        configuration = {"inference_provider": args.inference_provider, "model": model,
                         "model_digest": args.model_digest, "codex_effort": args.codex_effort}
        harness.verify_imports()
        before = harness.source_binding()
        before["probe_sha256"] = harness.digest(Path(__file__))
        if args.inference_provider == "codex":
            before["codex_executable_sha256"] = args.codex_executable_sha256
        harness.verify_execution_binding(before, args.source_head, args.live)
        if before["tree"] != args.source_tree:
            raise ValueError("execution_binding_refused")
        load_fixture(args.fixture_version)
        report = {"status": "prepared_only", **fixture_metadata(args.fixture_version)}
        if args.live:
            with args.private_receipt.open("xb", buffering=0) as receipt:
                try:
                    asyncio.run(asyncio.wait_for(run_cases(
                        model=model, base_url=args.base_url, model_digest=args.model_digest,
                        inference_provider=args.inference_provider, codex_effort=args.codex_effort,
                        codex_executable=args.codex_executable,
                        codex_executable_sha256=args.codex_executable_sha256,
                        private_sink=receipt, observations=rows,
                        fixture_version=args.fixture_version,
                    ), 1000))
                finally:
                    record_unrun(rows, receipt, fixture_version=args.fixture_version)
            report = build_report(rows, fixture_version=args.fixture_version)
            report["status"] = (
                "completed" if report["attempts"] == 8 and report["errors"] == 0
                else "attempt_failed"
            )
        after = harness.source_binding()
        after["probe_sha256"] = harness.digest(Path(__file__))
        if args.inference_provider == "codex":
            after["codex_executable_sha256"] = harness.digest(Path(args.codex_executable))
        load_fixture(args.fixture_version)
        if after != before:
            raise ValueError("execution_input_drift")
    except Exception:
        report = build_report(rows, fixture_version=args.fixture_version) | {
            "status": "attempt_failed",
        }
    finally:
        report.update(configuration)
        report.update(source_binding=before,
                      natural_routing_qualified=False)
        print("[confirmed-natural-probe] " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") in {"prepared_only", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
