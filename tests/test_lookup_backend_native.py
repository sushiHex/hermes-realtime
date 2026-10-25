"""Opt-in native proof of the public execute function, without a model server.

Set LOOKUP_NATIVE_SOURCE and LOOKUP_NATIVE_PYTHON to a prepared pinned Hermes
checkout/interpreter. Reuse the existing disposable-profile/Job proof controller;
only its synthetic fixture and selected case are adapted in a private copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

FORMATTER = '''
        async def formatter(request):
            if not authorized(request):
                return web.json_response({}, status=401)
            payload = await request.json()
            assert payload["stream"] is False and payload["model"] == "synthetic-proof"
            assert payload["reasoning_effort"] == "none" and payload["max_tokens"] == 1024
            assert not any(key in payload for key in (
                "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"))
            source = json.loads(payload["messages"][-1]["content"])
            assert source["answer"] == "Synthetic completion."
            assert len(source["sources"]) == 4
            safety["formatter_requests"] = safety.get("formatter_requests", 0) + 1
            safety["formatter_catalog_bound"] = True
            return web.json_response({
                "id": "synthetic-formatter", "object": "chat.completion", "created": 0,
                "model": "synthetic-proof", "choices": [{"index": 0,
                    "finish_reason": "stop", "message": {"role": "assistant",
                        "content": json.dumps({"voice": "Synthetic answer.",
                            "detail": "Synthetic detail.", "source_ids": [0]})}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        app.router.add_post("/proof/v1/chat/completions", formatter)
'''

NATIVE_CASE = '''
import ast
import hashlib
import json
import os
import sys
from pathlib import Path


async def prove(port, key, witness, case):
    from aiohttp import ClientSession
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_cli.plugins import get_plugin_manager

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    # Execute the actual function AST, not a reconstruction. Loading the whole
    # product scripts package would collide with Hermes's own scripts namespace.
    task_path = project / "scripts/lookup_backend_task.py"
    raw = task_path.read_bytes()
    tree = ast.parse(raw)
    imports = [node for node in tree.body if isinstance(node, ast.Import) or
               isinstance(node, ast.ImportFrom) and node.module != "scripts.setup_lookup_backend"]
    functions = [node for node in tree.body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute"]
    assert len(functions) == 1
    setup_path = project / "scripts/setup_lookup_backend.py"
    setup_tree = ast.parse(setup_path.read_bytes())
    require = [node for node in setup_tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "require"]
    assert len(require) == 1
    scope = {"__name__": "_native_public_execute", "Any": object}
    exec(compile(ast.Module(require, type_ignores=[]), str(setup_path), "exec"), scope)
    exec(compile(ast.Module(imports + functions, type_ignores=[]), str(task_path), "exec"), scope)
    witness["public_execute_source_sha256"] = hashlib.sha256(raw).hexdigest()
    witness["public_execute_ast_exact"] = True
    witness["synthetic_provider_only"] = True
    get_plugin_manager().discover_and_load()
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": port, "key": key, "model_name": "synthetic-proof"}))
    home = Path(os.environ["HERMES_HOME"])
    (home / ".env").write_text("API_SERVER_KEY=" + key + "\\n", encoding="utf-8")
    base = "http://127.0.0.1:" + str(port)
    try:
        assert await adapter.connect(), "native-connect"
        async with ClientSession(headers={"Authorization": "Bearer " + key}) as client:
            async with client.post(base + "/generation-proof/control",
                                   json={"action": "release"}) as reply:
                assert reply.status == 200
            result = await scope["execute"]({"profile": str(home), "baseUrl": base,
                "objective": "Synthetic request."}, home / "public-task-evidence")
            witness["public_execute"] = result
            assert result["category"] == "completed", "public-task-completed"
            assert result["adapter_completed"] is True
            assert result["sse_completion_proven"] is True
            assert result["posts"] == 1 and result["accepted"] == 1
            assert result["cleanup_failure_kinds"] == []
            evidence = json.loads((home / "public-task-evidence" /
                                   "terminal-evidence.private.json").read_bytes())
            assert evidence["run"]["status"] == "completed"
            async with client.get(base + "/generation-proof/state") as reply:
                assert reply.status == 200
                state = await reply.json()
            rows = list(state["counts"].values())
            assert len(rows) == 1 and rows[0]["provider"] == 1
            assert rows[0]["search"] == 1 and rows[0]["extract"] == 0
            assert state["safety"]["host_initial_model_before_effect"] == 0
            assert state["safety"]["host_initial_native_result_seen"] is True
            assert state["safety"]["formatter_requests"] == 1
            assert state["safety"]["formatter_catalog_bound"] is True
            assert state["safety"]["fixture_failures"] == 0
            witness["public_execute_effects"] = {"initial_model_calls": 0,
                "native_search_calls": 1, "subsequent_stand_in_calls": 1,
                "formatter_sdk_calls": 1, "terminal_evidence_retained": True}
            assert witness["unclassified_network_attempts"] == 0
            return {"cases": 1, "native_completed": 1, "stand_in_calls": 1}
    finally:
        await adapter.disconnect()
'''


def replace_once(raw: str, old: str, new: str) -> str:
    assert raw.count(old) == 1, "native-fixture-source-anchor"
    return raw.replace(old, new)


def prepare_native_proof(directory: Path) -> None:
    tests = directory / "tests"
    tests.mkdir(parents=True)
    fixture = tests / "fixtures/plugin_generation_candidate"
    fixture.mkdir(parents=True)
    for name in ("offline_plugin_generation_proof.py", "generation_policy_mutations.py",
                 "generation_source.py", "generation_imports.py", "generation_policy_checks.py"):
        shutil.copyfile(PROJECT / "tests" / name, tests / name)
    for name in ("__init__.py", "policy.py", "entrypoint.py", "plugin.yaml"):
        shutil.copyfile(PROJECT / "tests/fixtures/plugin_generation_candidate" / name,
                        fixture / name)
    scripts = directory / "scripts"
    scripts.mkdir()
    for name in ("local_livekit.py", "real_gate_support.py", "rehearse_desktop_mvp.py",
                 "lookup_backend_task.py", "setup_lookup_backend.py"):
        shutil.copyfile(PROJECT / "scripts" / name, scripts / name)
    shutil.copytree(PROJECT / "src", directory / "src")
    (tests / "generation_native_cases.py").write_text(NATIVE_CASE, encoding="utf-8")
    runner = tests / "offline_plugin_generation_proof.py"
    raw = runner.read_text(encoding="utf-8")
    port = ("        with socket.socket() as listener:\n"
            "            listener.bind((\"127.0.0.1\", 0))\n"
            "            port = listener.getsockname()[1]\n")
    raw = replace_once(raw, port, "")
    raw = replace_once(raw, '        key = "synthetic-test-" + "a7" * 24\n',
                       port + '        key = "synthetic-test-" + "a7" * 24\n')
    raw = replace_once(raw, '"  base_url: http://127.0.0.1:9/v1\\n  api_key: synthetic-only\\n"',
                       'f"  base_url: http://127.0.0.1:{port}/proof/v1\\n'
                       '  api_key: synthetic-only\\n"')
    raw = replace_once(raw, '"      settings:\\n        proof_key: " + key + "\\n"',
        '"      settings:\\n        proof_key: " + key + "\\n"\n'
        '            f"        proof_base_url: http://127.0.0.1:{port}/proof/v1\\n"')
    runner.write_text(raw, encoding="utf-8")
    path = fixture / "__init__.py"
    raw = path.read_text(encoding="utf-8")
    raw = replace_once(raw, '"base_url": "http://127.0.0.1:9/v1"',
                       '"base_url": ctx.get_config("proof_base_url")')
    raw = replace_once(raw, 'request.headers.get("X-Proof-Case", "healthy")',
                       'request.headers.get("X-Proof-Case", "host-initial")')
    raw = replace_once(raw, '        app.router.add_get("/generation-proof/state", state)',
                       FORMATTER + '        app.router.add_get("/generation-proof/state", state)')
    path.write_text(raw, encoding="utf-8")
    for path in [*tests.glob("*.py"), *fixture.glob("*.py")]:
        compile(path.read_bytes(), str(path), "exec")


@pytest.mark.skipif(os.name != "nt" or not os.environ.get("LOOKUP_NATIVE_SOURCE")
                    or not os.environ.get("LOOKUP_NATIVE_PYTHON"),
                    reason="explicit prepared pinned Hermes environment required")
def test_public_execute_against_pinned_native_api(tmp_path):
    proof = tmp_path / "proof"
    prepare_native_proof(proof)
    receipt = Path(os.environ.get("LOOKUP_NATIVE_RECEIPT", str(tmp_path / "receipt.json")))
    assert not receipt.exists(), "fresh-native-receipt-required"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items()
           if not any(part in key.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))}
    env["SYNTHETIC_PRESENTATION_KEY"] = "synthetic-test-" + "a7" * 24
    result = subprocess.run([os.environ["LOOKUP_NATIVE_PYTHON"], "-I", "-S", "-B",
        str(proof / "tests/offline_plugin_generation_proof.py"), "--hermes-source",
        os.environ["LOOKUP_NATIVE_SOURCE"], "--receipt", str(receipt), "--case", "host-initial"],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
    assert receipt.exists(), "native-proof-receipt-missing"
    data = json.loads(receipt.read_bytes())
    assert result.returncode == 0 and data["ok"] is True, "native-proof-refused"
    assert data["child_settled"] is True and data["worker_survivor_count"] == 0
    assert data["profile_cleanup"] == "completed"
    assert data["public_execute_ast_exact"] is True
    assert data["public_execute"]["category"] == "completed"
    assert data["public_execute_source_sha256"] == hashlib.sha256(
        (PROJECT / "scripts/lookup_backend_task.py").read_bytes()).hexdigest()
