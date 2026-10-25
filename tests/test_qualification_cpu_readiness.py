"""Mocked #315 readiness recipe; no runtime, sockets, native calls, or inference."""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import subprocess
from pathlib import Path

import pytest

from scripts import qualification_cpu_readiness as recipe

_APPROVED_MANIFEST = (
    b'{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json",'
    b'"config":{"mediaType":"application/vnd.docker.container.image.v1+json",'
    b'"digest":"sha256:8c2539a423c4df2d9d2609fba56e7bad554696b66ed0e280c56e0ef4a32498fa",'
    b'"size":411},"layers":[{"mediaType":"application/vnd.ollama.image.model",'
    b'"digest":"sha256:3c168af1dea0a414299c7d9077e100ac763370e5a98b3c53801a958a47f0a5db",'
    b'"size":2491874624},{"mediaType":"application/vnd.ollama.image.template",'
    b'"digest":"sha256:1e8a63e92330684ba547df688fb7a80c6443a9a636309d948c7673441dcfd72d",'
    b'"size":655},{"mediaType":"application/vnd.ollama.image.license",'
    b'"digest":"sha256:fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d",'
    b'"size":1084}]}'
)


def _approved_inputs():
    manifest = json.loads(_APPROVED_MANIFEST)
    return sorted(
        [
            {
                "path": "manifests/registry.ollama.ai/library/phi-campaign-canonical/latest",
                "sha256": hashlib.sha256(_APPROVED_MANIFEST).hexdigest(),
                "size_bytes": len(_APPROVED_MANIFEST),
            },
            *(
                {
                    "path": "blobs/" + item["digest"].replace(":", "-"),
                    "sha256": item["digest"][7:],
                    "size_bytes": item["size"],
                }
                for item in [manifest["config"], *manifest["layers"]]
            ),
        ],
        key=lambda item: item["path"],
    )


def test_profile_pins_match_the_retained_small_manifest_without_reading_weights():
    assert recipe._MODEL == "phi-campaign-canonical:latest"
    assert hashlib.sha256(_APPROVED_MANIFEST).hexdigest() == recipe._MODEL_DIGEST
    assert _approved_inputs() == recipe._INPUT_FILES
    assert recipe._RUNTIME_MANIFEST_SHA256 == (
        "b87d3612454a7382076a5aee3e421b966742877871882c122b88f241e8b5e96c"
    )


def _documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime = {
        "files": [
            {"path": f"lib/ollama/{i:02}.dll", "sha256": "a" * 64, "bytes": 1} for i in range(43)
        ]
        + [{"path": "ollama.exe", "sha256": "b" * 64, "bytes": 1}]
    }
    raw = json.dumps(runtime).encode()
    monkeypatch.setattr(recipe, "_RUNTIME_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(raw)
    resources = {
        "schema_version": 1,
        "execution": {
            "root": str(tmp_path / "payload"),
            "files": [
                {"path": x["path"], "sha256": x["sha256"], "size_bytes": x["bytes"]}
                for x in runtime["files"]
            ],
        },
        "inputs": [{"root": str(tmp_path / "inputs"), "files": _approved_inputs()}],
    }
    return runtime_path, resources


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "resource_digest",
        "runtime_digest",
        "runtime_set",
        "input_digest",
        "input_path",
        "extra_input",
    ],
)
def test_recipe_binds_the_single_prepared_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    runtime_path, resources = _documents(tmp_path, monkeypatch)
    if fault == "runtime_set":
        resources["execution"]["files"].pop()
    elif fault == "input_digest":
        resources["inputs"][0]["files"][0]["sha256"] = "0" * 64
    elif fault == "input_path":
        resources["inputs"][0]["files"][-1]["path"] = "different-manifest"
    elif fault == "extra_input":
        resources["inputs"].append(copy.deepcopy(resources["inputs"][0]))
    elif fault == "runtime_digest":
        runtime_path.write_bytes(runtime_path.read_bytes() + b" ")
    path = tmp_path / "resources.json"
    raw = json.dumps(resources).encode()
    path.write_bytes(raw)
    digest = "0" * 64 if fault == "resource_digest" else hashlib.sha256(raw).hexdigest()
    if fault == "none":
        assert recipe._load_resources(path, digest, runtime_path) == resources
    else:
        with pytest.raises(ValueError):
            recipe._load_resources(path, digest, runtime_path)


def test_runtime_environment_is_closed_and_uses_only_the_guardian_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, models, work = (tmp_path / name for name in ("snapshot", "models", "work"))
    work.mkdir()
    monkeypatch.setattr(
        recipe,
        "_system_environment",
        lambda: {
            "SystemRoot": r"C:\Windows",
            "WINDIR": r"C:\Windows",
            "SystemDrive": "C:",
        },
    )
    monkeypatch.setattr(
        recipe, "_trusted_windows_directories", lambda: (r"C:\Windows", r"C:\Windows\System32")
    )
    for key in (
        "GGML_BACKEND_PATH",
        "LLAMA_ARG_MODEL",
        "OLLAMA_CONTEXT_LENGTH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "OLLAMA_HOST",
        "PATH",
        "HOME",
    ):
        monkeypatch.setenv(key, "AMBIENT_SENTINEL")
    env = recipe._runtime_environment(snapshot, models, work, 12345)
    assert "AMBIENT_SENTINEL" not in json.dumps(env)
    assert set(env) == {
        "SystemRoot",
        "WINDIR",
        "SystemDrive",
        "PATH",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "OLLAMA_HOST",
        "OLLAMA_MODELS",
        "OLLAMA_LLM_LIBRARY",
        "OLLAMA_VULKAN",
        "OLLAMA_GO_TEMPLATE",
        "OLLAMA_NO_CLOUD",
        "OLLAMA_NOPRUNE",
        "OLLAMA_NUM_PARALLEL",
        "OLLAMA_MAX_LOADED_MODELS",
    }
    assert env["OLLAMA_HOST"] == "127.0.0.1:12345"
    assert env["OLLAMA_MODELS"] == str(models)
    assert env["OLLAMA_LLM_LIBRARY"] == "cpu" and env["OLLAMA_VULKAN"] == "0"
    assert env["OLLAMA_GO_TEMPLATE"] == env["OLLAMA_NO_CLOUD"] == env["OLLAMA_NOPRUNE"] == "1"
    assert env["OLLAMA_NUM_PARALLEL"] == env["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert env["PATH"] == ";".join((str(snapshot / "lib/ollama"), r"C:\Windows\System32"))
    assert env["HOME"] == env["USERPROFILE"] == str(work / "home")
    assert env["TEMP"] == env["TMP"] == str(work / "temp")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "source",
        "outside_job",
        "binding",
        "version",
        "identity",
        "readiness",
        "adapter_close",
        "root_stop",
        "context",
        "deadline",
        "root_exits_after_preflight",
        "slow_identity",
    ],
)
async def test_controller_reuses_one_existing_preflight_and_leaves_tree_settlement_to_guardian(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    events = []
    snapshot = tmp_path / "guardian-snapshot"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv(recipe.launch.EXECUTION_ROOT_ENV, str(snapshot))
    if fault == "binding":
        monkeypatch.delenv(recipe.launch.EXECUTION_ROOT_ENV)
    monkeypatch.setattr(recipe.launch, "_outside_jobs", lambda: fault == "outside_job")

    def source(head, tree):
        events.append("source")
        if fault == "source":
            raise ValueError("synthetic source refusal")

    monkeypatch.setattr(recipe.evaluator, "_verify_source_binding", source)
    monkeypatch.setattr(recipe, "_verify_recipe_imports", lambda: None)
    monkeypatch.setattr(recipe, "_unused_port", lambda: 12345)
    monkeypatch.setattr(recipe, "_runtime_environment", lambda *a: {"CLOSED": "1"})
    if fault == "deadline":
        monkeypatch.setattr(recipe.evaluator, "_READINESS_BUDGET_SECONDS", -1)
    if fault == "slow_identity":
        monkeypatch.setattr(recipe.evaluator, "_READINESS_BUDGET_SECONDS", 0.01)

    class Process:
        pid = 42
        stopped = False

        def poll(self):
            return (
                0
                if self.stopped or (fault == "root_exits_after_preflight" and "preflight" in events)
                else None
            )

        def terminate(self):
            events.append("terminate")
            self.stopped = True

        def wait(self, timeout):
            assert 0 < timeout <= 10
            events.append("wait")
            if fault == "root_stop":
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return 0

    def popen(command, **kwargs):
        events.append("spawn")
        assert command == [str(snapshot / "ollama.exe"), "serve"]
        assert kwargs == {
            "cwd": snapshot,
            "env": {"CLOSED": "1"},
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "creationflags": 0x8000000,
            "close_fds": True,
        }
        return Process()

    class Metadata:
        def __init__(self, *, base_url):
            assert base_url == "http://127.0.0.1:12345"

        async def _request(self, endpoint):
            events.append(endpoint)
            assert events[-2] == "listener"
            return {"version": "other" if fault == "version" else "0.40.4"}

        async def identity(self, model):
            events.append("identity")
            assert model == "phi-campaign-canonical:latest"
            if fault == "slow_identity":
                await recipe.asyncio.sleep(0.05)
            return "0" * 64 if fault == "identity" else recipe._MODEL_DIGEST

    class Inference:
        async def close(self):
            events.append("adapter_close")
            if fault == "adapter_close":
                raise RuntimeError("synthetic adapter close")

        def context_public_data(self):
            return {
                "version": 2,
                "modelDigest": recipe._MODEL_DIGEST,
                "requestedContextTokens": 4097 if fault == "context" else 4096,
                "advertisedContextTokens": 131072,
                "effectiveContextTokens": 4096,
            }

    async def open_inference(**kwargs):
        events.append("preflight")
        assert kwargs["provider"] == "ollama" and kwargs["ollama_num_ctx"] == 4096
        assert kwargs["ollama_num_gpu"] == 0 and kwargs["model"] == recipe._MODEL
        assert kwargs["codex_executable"] is None
        assert type(kwargs["handler"]) is recipe.evaluator.ShadowWorkHandler
        assert kwargs["deadline"] > 0
        if fault == "readiness":
            raise TimeoutError("synthetic preflight bound")
        return Inference()

    monkeypatch.setattr(recipe.subprocess, "Popen", popen)
    monkeypatch.setattr(recipe, "OllamaModelClient", Metadata)
    monkeypatch.setattr(
        recipe, "_require_owned_listener", lambda port, pid: events.append("listener")
    )
    monkeypatch.setattr(recipe.evaluator, "_open_qualified_inference", open_inference)
    report = await recipe._controller(tmp_path / "models", work, "a" * 40, "b" * 40)
    assert report["qualification"] is False and "process_tree_settled" not in report
    assert report["outcome"] == ("ready" if fault == "none" else "unavailable")
    assert set(report["timings_ms"]) <= {"startup", "readiness", "adapter_close", "root_stop"}
    if fault in {"source", "outside_job", "binding", "deadline"}:
        assert "spawn" not in events
    else:
        assert events[-2:] == (
            ["adapter_close", "wait"]
            if fault == "root_exits_after_preflight"
            else ["terminate", "wait"]
        )
        assert report["runtime_root_stopped"] is (fault != "root_stop")
    if fault in {"version", "identity", "slow_identity"}:
        assert "preflight" not in events
    if fault == "none":
        assert events.count("preflight") == 1 and events.index("adapter_close") < events.index(
            "terminate"
        )
        assert report["readiness_completed"] is True
        assert report["context"]["effectiveContextTokens"] == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,category,cleanup_failures",
    [
        ("deadline_and_cleanup", "deadline", 1),
        ("refusal_and_cleanup", "refused", 1),
        ("deadline_only", "deadline", 0),
        ("deadline_cleanup_root_stop", "deadline", 2),
        ("unrelated_group", "refused", 0),
        ("wrong_shape", "refused", 0),
        ("startup_group", "refused", 0),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_controller_preserves_actual_preflight_failure_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    category: str,
    cleanup_failures: int,
) -> None:
    events = []
    group_message = "qualification preflight and adapter cleanup failed"
    primary = (
        TimeoutError("PRIVATE_TIMEOUT")
        if fault.startswith("deadline")
        else ValueError("PRIVATE_REFUSAL")
    )
    monkeypatch.setenv(recipe.launch.EXECUTION_ROOT_ENV, str(tmp_path / "snapshot"))
    monkeypatch.setattr(recipe.evaluator, "_verify_source_binding", lambda *a: None)
    monkeypatch.setattr(recipe, "_verify_recipe_imports", lambda: None)
    monkeypatch.setattr(recipe.launch, "_outside_jobs", lambda: False)
    monkeypatch.setattr(recipe, "_unused_port", lambda: 12345)
    monkeypatch.setattr(recipe, "_runtime_environment", lambda *a: {})
    monkeypatch.setattr(recipe, "OllamaModelClient", lambda **kwargs: object())

    class Process:
        pid = 42

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append("wait")
            if fault == "deadline_cleanup_root_stop":
                raise subprocess.TimeoutExpired("PRIVATE_ROOT", timeout)

    async def wait_server(*args):
        if fault == "startup_group":
            raise BaseExceptionGroup(group_message, [TimeoutError(), RuntimeError()])

    class Inference:
        async def configure_context(self, tokens):
            assert tokens == 4096

        async def stream(self, request, *, turn_id):
            events.append("preflight")
            yield ""
            raise primary

        async def close(self):
            events.append("adapter_close")
            if "cleanup" in fault:
                raise RuntimeError("PRIVATE_CLEANUP")

        def bind_work_tools(self, handler):
            pytest.fail("failed preflight admitted work tools")

    def factory(**kwargs):
        events.append("factory")
        if fault == "unrelated_group":
            raise BaseExceptionGroup("unrelated", [TimeoutError(), RuntimeError()])
        if fault == "wrong_shape":
            raise BaseExceptionGroup(group_message, [TimeoutError()])
        return Inference()

    monkeypatch.setattr(recipe.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(recipe, "_wait_server", wait_server)
    monkeypatch.setattr(recipe.evaluator, "_build_streaming_inference", factory)
    report = await recipe._controller(tmp_path / "models", tmp_path / "work", "a" * 40, "b" * 40)

    assert report["outcome"] == "unavailable" and report["readiness_completed"] is False
    assert report["context"] is None
    assert report["category"] == category
    assert report["cleanup_failures"] == cleanup_failures
    assert report["runtime_root_stopped"] is (fault != "deadline_cleanup_root_stop")
    assert report["stage"] == ("startup" if fault == "startup_group" else "readiness")
    expected_closes = 0 if fault in {"startup_group", "unrelated_group", "wrong_shape"} else 1
    assert events.count("adapter_close") == expected_closes
    assert events[-2:] == ["terminate", "wait"]
    assert recipe._controller_report(json.dumps(report).encode()) == report
    assert "PRIVATE" not in json.dumps(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["listener", "root_exited"])
async def test_metadata_is_never_contacted_without_the_retained_root_listener(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    class Process:
        pid = 42

        def poll(self):
            return 0 if fault == "root_exited" else None

    class Metadata:
        async def _request(self, endpoint):
            pytest.fail("metadata contacted an unowned listener")

    def refuse(port, pid):
        raise ValueError("synthetic listener not yet owned")

    monkeypatch.setattr(recipe, "_require_owned_listener", refuse)
    deadline = recipe.asyncio.get_running_loop().time() + 0.001
    with pytest.raises(TimeoutError if fault == "listener" else RuntimeError):
        await recipe._wait_server(Process(), Metadata(), 12345, deadline)


def test_unused_port_is_selected_on_literal_loopback_with_exclusive_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def setsockopt(self, *args):
            calls.append(args)

        def bind(self, address):
            calls.append(address)

        def getsockname(self):
            return "127.0.0.1", 12345

    monkeypatch.setattr(recipe.socket, "socket", lambda family, kind: Socket())
    monkeypatch.setattr(recipe.socket, "SO_EXCLUSIVEADDRUSE", 4, raising=False)
    assert recipe._unused_port() == 12345
    assert calls == [(socket.SOL_SOCKET, 4, 1), ("127.0.0.1", 0), "closed"]


@pytest.mark.parametrize("fault", ["none", "metadata_source", "evaluator_source", "helper_source"])
def test_recipe_rejects_wrong_loaded_source_origins(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    if fault == "metadata_source":
        monkeypatch.setattr(recipe.ollama_models, "__file__", "wrong.py")
    elif fault == "evaluator_source":
        monkeypatch.setattr(recipe.evaluator, "__file__", "wrong.py")
    elif fault == "helper_source":
        monkeypatch.setattr(recipe.process_helpers, "__file__", "wrong.py")
    if fault == "none":
        recipe._verify_recipe_imports()
    else:
        with pytest.raises(ValueError):
            recipe._verify_recipe_imports()


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "opt_in",
        "launch",
        "report",
        "root_not_stopped",
        "readiness_incomplete",
        "cleanup",
        "controller_exit",
        "preflight_failure",
    ],
)
def test_only_guardian_settlement_can_complete_the_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    runtime_path, resources = _documents(tmp_path, monkeypatch)
    resources_path = tmp_path / "resources.json"
    raw = json.dumps(resources).encode()
    resources_path.write_bytes(raw)
    receipt = tmp_path / "receipt.jsonl"
    called = []
    report = {
        "schema_version": 1,
        "scope": "cpu_readiness_only",
        "qualification": False,
        "outcome": "ready",
        "readiness_completed": True,
        "runtime_root_stopped": True,
        "timings_ms": {"startup": 1.0, "readiness": 2.0, "adapter_close": 1.0, "root_stop": 1.0},
        "cleanup_failures": 0,
        "context": {
            "version": 2,
            "modelDigest": recipe._MODEL_DIGEST,
            "requestedContextTokens": 4096,
            "advertisedContextTokens": 131072,
            "effectiveContextTokens": 4096,
        },
    }
    if fault == "report":
        report["private_output"] = "PRIVATE_SENTINEL"
    elif fault == "root_not_stopped":
        report["runtime_root_stopped"] = False
    elif fault == "readiness_incomplete":
        report["readiness_completed"] = False
    elif fault == "cleanup":
        report["cleanup_failures"] = 1
    elif fault == "preflight_failure":
        report.update(
            outcome="unavailable", readiness_completed=False, stage="readiness", category="deadline"
        )

    def run(script, args, **kwargs):
        called.append(True)
        assert script.resolve() == Path(recipe.__file__).resolve()
        assert args[0] == "--controller"
        assert args[args.index("--input-root") + 1] == resources["inputs"][0]["root"]
        assert kwargs["sealed_resources"] == resources and kwargs["timeout_seconds"] == 120
        assert kwargs["receipt"] == receipt
        if fault == "launch":
            raise RuntimeError("synthetic guardian failed settlement")
        return recipe.launch.LaunchResult(
            2 if fault == "controller_exit" else 0, json.dumps(report).encode()
        )

    monkeypatch.setattr(recipe.launch, "run_owned_controller", run)
    monkeypatch.setattr(recipe.evaluator, "_verify_source_binding", lambda *a: None)
    monkeypatch.setattr(recipe, "_verify_recipe_imports", lambda: None)
    args = [
        "--source-head",
        "a" * 40,
        "--source-tree",
        "b" * 40,
        "--resources",
        str(resources_path),
        "--resources-sha256",
        hashlib.sha256(raw).hexdigest(),
        "--runtime-manifest",
        str(runtime_path),
        "--launch-binding",
        str(tmp_path / "binding"),
        "--binding-sha256",
        "c" * 64,
        "--receipt",
        str(receipt),
    ]
    if fault != "opt_in":
        args.append("--run-readiness")
    code = recipe.main(args)
    captured = capsys.readouterr()
    actual = json.loads(captured.out)
    assert code == (0 if fault == "none" else 2)
    assert actual["qualification"] is False
    assert actual["outcome"] == ("ready" if fault == "none" else "unavailable")
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err
    if fault in {"opt_in", "launch"}:
        assert actual["process_tree_settled"] is False
    if fault == "none":
        assert actual["process_tree_settled"] is True
    if fault == "preflight_failure":
        assert actual["process_tree_settled"] is True
        assert actual["stage"] == "readiness" and actual["category"] == "deadline"
        assert actual["timings_ms"] == report["timings_ms"]
    if fault == "opt_in":
        assert not called


def test_completed_unavailable_controller_observation_exits_zero_to_retain_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def observe(*args):
        return {
            "schema_version": 1,
            "scope": "cpu_readiness_only",
            "qualification": False,
            "outcome": "unavailable",
            "readiness_completed": False,
            "runtime_root_stopped": True,
            "timings_ms": {"startup": 1.0, "readiness": 20.0, "root_stop": 1.0},
            "cleanup_failures": 0,
            "stage": "readiness",
            "category": "deadline",
            "context": None,
        }

    monkeypatch.setattr(recipe, "_controller", observe)
    code = recipe.main(
        [
            "--controller",
            "--input-root",
            str(tmp_path / "models"),
            "--run-work",
            str(tmp_path),
            "--source-head",
            "a" * 40,
            "--source-tree",
            "b" * 40,
        ]
    )
    actual = json.loads(capsys.readouterr().out)
    assert code == 0 and actual["outcome"] == "unavailable"
    assert actual["timings_ms"]["readiness"] == 20.0
