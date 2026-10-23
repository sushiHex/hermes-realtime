"""One opt-in #315 CPU readiness observation through the existing evaluator.

This fixed recipe is diagnostic, not a model, latency, or voice qualification.
The guardian alone owns descendant settlement. Runtime home/temp are private run
artifacts: this controller never removes them while descendants may remain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from hermes_realtime.providers import ollama_models  # noqa: E402
from hermes_realtime.providers.ollama import OllamaStreamingInference  # noqa: E402
from hermes_realtime.providers.ollama_models import OllamaModelClient  # noqa: E402
from scripts import equivalence_process as process_helpers  # noqa: E402
from scripts import evaluate_natural_work_routing as evaluator  # noqa: E402
from scripts import qualification_launch as launch  # noqa: E402
from scripts.candidate_source_archive_oracle import _trusted_windows_directories  # noqa: E402
from scripts.equivalence_process import _require_owned_listener  # noqa: E402
from scripts.qualification_tool_process import _system_environment  # noqa: E402

_MODEL = "phi-campaign-canonical:latest"
_MODEL_DIGEST = "aa87d3290530ffdac434309ccf7b9cafd7f186906d2c833d638c422f456183de"
_RUNTIME_MANIFEST_SHA256 = "b87d3612454a7382076a5aee3e421b966742877871882c122b88f241e8b5e96c"
_INPUT_FILES = sorted(
    [
        {
            "path": "manifests/registry.ollama.ai/library/phi-campaign-canonical/latest",
            "sha256": _MODEL_DIGEST,
            "size_bytes": 711,
        },
        {
            "path": "blobs/sha256-8c2539a423c4df2d9d2609fba56e7bad554696b66ed0e280c56e0ef4a32498fa",
            "sha256": "8c2539a423c4df2d9d2609fba56e7bad554696b66ed0e280c56e0ef4a32498fa",
            "size_bytes": 411,
        },
        {
            "path": "blobs/sha256-3c168af1dea0a414299c7d9077e100ac763370e5a98b3c53801a958a47f0a5db",
            "sha256": "3c168af1dea0a414299c7d9077e100ac763370e5a98b3c53801a958a47f0a5db",
            "size_bytes": 2491874624,
        },
        {
            "path": "blobs/sha256-1e8a63e92330684ba547df688fb7a80c6443a9a636309d948c7673441dcfd72d",
            "sha256": "1e8a63e92330684ba547df688fb7a80c6443a9a636309d948c7673441dcfd72d",
            "size_bytes": 655,
        },
        {
            "path": "blobs/sha256-fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d",
            "sha256": "fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d",
            "size_bytes": 1084,
        },
    ],
    key=lambda item: str(item["path"]),
)


def _load_resources(path: Path, digest: str, runtime_manifest: Path) -> dict[str, Any]:
    raw = launch._read(path, 65536)
    if type(digest) is not str or launch._sha(raw) != digest:
        raise ValueError("cpu_readiness_resource_digest")
    resources = cast(dict[str, Any], launch._validate_sealed_resources(launch._json(raw)))
    runtime_raw = launch._read(runtime_manifest, 65536)
    if launch._sha(runtime_raw) != _RUNTIME_MANIFEST_SHA256:
        raise ValueError("cpu_readiness_runtime_digest")
    manifest = launch._json(runtime_raw)
    expected = sorted(
        (
            {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["bytes"]}
            for item in manifest["files"]
        ),
        key=lambda item: item["path"],
    )
    if len(expected) != 44 or resources["execution"]["files"] != expected:
        raise ValueError("cpu_readiness_runtime_inventory")
    if len(resources["inputs"]) != 1 or resources["inputs"][0]["files"] != _INPUT_FILES:
        raise ValueError("cpu_readiness_input_inventory")
    return resources


def _verify_recipe_imports() -> None:
    if (
        Path(ollama_models.__file__).resolve()
        != _ROOT / "src/hermes_realtime/providers/ollama_models.py"
        or OllamaModelClient is not ollama_models.OllamaModelClient
        or Path(evaluator.__file__).resolve() != _ROOT / "scripts/evaluate_natural_work_routing.py"
        or Path(process_helpers.__file__).resolve() != _ROOT / "scripts/equivalence_process.py"
        or _require_owned_listener is not process_helpers._require_owned_listener
    ):
        raise ValueError("cpu_readiness_import_source")


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _runtime_environment(snapshot: Path, models: Path, work: Path, port: int) -> dict[str, str]:
    # Native Windows discovery, never ambient SystemRoot/PATH/proxy/model flags.
    environment = cast(dict[str, str], _system_environment())
    _, system = _trusted_windows_directories()
    home, temporary = work / "home", work / "temp"
    home.mkdir()
    temporary.mkdir()
    environment.update(
        {
            "PATH": ";".join((str(snapshot / "lib/ollama"), system)),
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": str(models),
            "OLLAMA_LLM_LIBRARY": "cpu",
            "OLLAMA_VULKAN": "0",
            "OLLAMA_GO_TEMPLATE": "1",
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_NOPRUNE": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
        }
    )
    return environment


async def _wait_server(
    process: subprocess.Popen[bytes],
    client: OllamaModelClient,
    port: int,
    deadline: float,
) -> None:
    async with asyncio.timeout_at(deadline):
        while True:
            if process.poll() is not None:
                raise RuntimeError("cpu_readiness_server_exited")
            try:
                _require_owned_listener(port, process.pid)
            except ValueError:
                await asyncio.sleep(0.05)
                continue
            version = (await client._request("version")).get("version")
            if version != "0.40.4":
                raise ValueError("cpu_readiness_runtime_version")
            _require_owned_listener(port, process.pid)
            if await client.identity(_MODEL) != _MODEL_DIGEST:
                raise ValueError("cpu_readiness_model_identity")
            return


async def _controller(models: Path, work: Path, head: str, tree: str) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + evaluator._READINESS_BUDGET_SECONDS
    stop_deadline = deadline + evaluator._CLOSE_BUDGET_SECONDS
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "cpu_readiness_only",
        "qualification": False,
        "outcome": "unavailable",
        "readiness_completed": False,
        "runtime_root_stopped": False,
        "timings_ms": {},
        "cleanup_failures": 0,
        "context": None,
    }
    process = None
    inference = None
    stage = "source"
    try:
        evaluator._verify_source_binding(head, tree)
        _verify_recipe_imports()
        if launch._outside_jobs():
            raise ValueError("cpu_readiness_requires_owned_job")
        root = os.environ.get(launch.EXECUTION_ROOT_ENV)
        if root is None or not Path(root).is_absolute():
            raise ValueError("cpu_readiness_requires_guardian_snapshot")
        snapshot = Path(root)
        port = _unused_port()
        environment = _runtime_environment(snapshot, models, work, port)
        if loop.time() >= deadline:
            raise TimeoutError("cpu_readiness_start_deadline")
        stage = "startup"
        started = loop.time()
        try:
            # No breakaway: the existing guardian Job owns this root and all
            # descendants. Keep Popen's process handle through bounded stop.
            process = subprocess.Popen(
                [str(snapshot / "ollama.exe"), "serve"],
                cwd=snapshot,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x8000000,
                close_fds=True,
            )
            client = OllamaModelClient(base_url=f"http://127.0.0.1:{port}")
            await _wait_server(process, client, port, deadline)
        finally:
            report["timings_ms"]["startup"] = round((loop.time() - started) * 1000, 3)
        stage = "readiness"
        started = loop.time()
        try:
            inference = cast(
                OllamaStreamingInference,
                await evaluator._open_qualified_inference(
                    provider="ollama",
                    model=_MODEL,
                    effort="",
                    codex_executable=None,
                    ollama_base_url=f"http://127.0.0.1:{port}",
                    ollama_num_ctx=4096,
                    ollama_num_gpu=0,
                    handler=evaluator.ShadowWorkHandler(can_cancel_work=False, accept_start=False),
                    deadline=deadline,
                ),
            )
            if process.poll() is not None:
                raise RuntimeError("cpu_readiness_server_exited")
            _require_owned_listener(port, process.pid)
            observed = inference.context_public_data()
            if (
                observed["modelDigest"] != _MODEL_DIGEST
                or observed["requestedContextTokens"] != 4096
            ):
                raise ValueError("cpu_readiness_context_identity")
            report["context"] = observed
            report["readiness_completed"] = True
        finally:
            report["timings_ms"]["readiness"] = round((loop.time() - started) * 1000, 3)
    except BaseException as error:
        report.update(
            stage=stage, category="deadline" if isinstance(error, TimeoutError) else "refused"
        )
    finally:
        if inference is not None:
            started = loop.time()
            try:
                await asyncio.wait_for(
                    inference.close(),
                    timeout=max(
                        0, min(evaluator._CLOSE_BUDGET_SECONDS, stop_deadline - loop.time())
                    ),
                )
            except BaseException:
                report["cleanup_failures"] += 1
            finally:
                report["timings_ms"]["adapter_close"] = round((loop.time() - started) * 1000, 3)
        if process is not None:
            started = loop.time()
            try:
                if process.poll() is None:
                    process.terminate()
                remaining = min(evaluator._CLOSE_BUDGET_SECONDS, stop_deadline - loop.time())
                if remaining <= 0:
                    raise TimeoutError("cpu_readiness_root_stop_deadline")
                process.wait(timeout=remaining)
                report["runtime_root_stopped"] = True
            except BaseException:
                report["cleanup_failures"] += 1
            finally:
                report["timings_ms"]["root_stop"] = round((loop.time() - started) * 1000, 3)
    if (
        report["readiness_completed"]
        and report["runtime_root_stopped"]
        and not report["cleanup_failures"]
    ):
        report["outcome"] = "ready"
    return report


def _controller_report(raw: bytes) -> dict[str, Any]:
    """Admit only this recipe's bounded diagnostics, without generated text."""
    value = launch._json(raw)
    required = {
        "schema_version",
        "scope",
        "qualification",
        "outcome",
        "readiness_completed",
        "runtime_root_stopped",
        "timings_ms",
        "cleanup_failures",
        "context",
    }
    phases = {"startup", "readiness", "adapter_close", "root_stop"}
    if (
        len(raw) > 4096
        or type(value) is not dict
        or not required <= set(value)
        or not set(value) <= required | {"stage", "category"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["scope"] != "cpu_readiness_only"
        or value["qualification"] is not False
        or value["outcome"] not in ("ready", "unavailable")
        or type(value["readiness_completed"]) is not bool
        or type(value["runtime_root_stopped"]) is not bool
        or type(value["cleanup_failures"]) is not int
        or not 0 <= value["cleanup_failures"] <= 2
        or type(value["timings_ms"]) is not dict
        or not set(value["timings_ms"]) <= phases
        or any(
            type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 120000
            for t in value["timings_ms"].values()
        )
        or ("stage" in value and value["stage"] not in {"source", "startup", "readiness"})
        or ("category" in value and value["category"] not in {"deadline", "refused"})
        or (
            value["outcome"] == "ready"
            and (
                not value["readiness_completed"]
                or not value["runtime_root_stopped"]
                or value["cleanup_failures"] != 0
                or set(value["timings_ms"]) != phases
            )
        )
    ):
        raise ValueError("cpu_readiness_controller_result")
    context = value["context"]
    if context is not None and (
        type(context) is not dict
        or set(context)
        != {
            "version",
            "requestedContextTokens",
            "advertisedContextTokens",
            "effectiveContextTokens",
            "modelDigest",
        }
        or type(context["version"]) is not int
        or context["version"] != 2
        or context["modelDigest"] != _MODEL_DIGEST
        or context["requestedContextTokens"] != 4096
        or any(
            t is not None and (type(t) is not int or not 0 < t <= 100_000_000)
            for t in (context["advertisedContextTokens"], context["effectiveContextTokens"])
        )
    ):
        raise ValueError("cpu_readiness_context_result")
    if value["outcome"] == "ready" and context is None:
        raise ValueError("cpu_readiness_context_result")
    return cast(dict[str, Any], value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-readiness", action="store_true")
    parser.add_argument("--controller", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-work", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--resources-sha256")
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--launch-binding", type=Path)
    parser.add_argument("--binding-sha256")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "cpu_readiness_only",
        "qualification": False,
        "outcome": "unavailable",
        "process_tree_settled": False,
    }
    controller_observed = False
    try:
        if args.controller:
            if args.input_root is None or args.run_work is None:
                raise ValueError("cpu_readiness_controller_arguments")
            with redirect_stdout(sys.stderr):
                report = asyncio.run(
                    _controller(args.input_root, args.run_work, args.source_head, args.source_tree)
                )
            # Preserve completed negative observations through the guardian.
            # Only the outer caller interprets outcome after Job settlement.
            controller_observed = True
        else:
            if not args.run_readiness or any(
                value is None
                for value in (
                    args.resources,
                    args.resources_sha256,
                    args.runtime_manifest,
                    args.launch_binding,
                    args.binding_sha256,
                    args.receipt,
                )
            ):
                raise ValueError("cpu_readiness_explicit_opt_in_required")
            evaluator._verify_source_binding(args.source_head, args.source_tree)
            _verify_recipe_imports()
            resources = _load_resources(
                args.resources, args.resources_sha256, args.runtime_manifest
            )
            work = args.receipt.with_name(args.receipt.name + ".runtime-work").resolve()
            work.mkdir()
            result = launch.run_owned_controller(
                Path(__file__),
                (
                    "--controller",
                    "--input-root",
                    resources["inputs"][0]["root"],
                    "--run-work",
                    str(work),
                    "--source-head",
                    args.source_head,
                    "--source-tree",
                    args.source_tree,
                ),
                binding_path=args.launch_binding,
                binding_sha256=args.binding_sha256,
                receipt=args.receipt,
                timeout_seconds=120,
                sealed_resources=resources,
            )
            report = _controller_report(result.stdout)
            report["process_tree_settled"] = True  # Only after guardian verified Job zero.
            evaluator._verify_source_binding(args.source_head, args.source_tree)
            if result.exit_code != 0:
                report["outcome"] = "unavailable"
    except BaseException:
        report.update(outcome="unavailable", category="launch_refused")
    finally:
        print(
            "[cpu-readiness] " + json.dumps({"version": 1, "outcome": report["outcome"]}),
            file=sys.stderr,
        )
        print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if controller_observed or report["outcome"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
