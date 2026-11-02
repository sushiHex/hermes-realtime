"""Effect-free pinned native admission-order proof; no provider/model calls."""
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOURCE = Path(sys.argv[1]).resolve()
EXPECTED = {
    "api_server.py": "6aec6687d47c81b3567e930882a499c5c5c041ecbe40e1d08f93038212401962",
    "api_server_runs.py": "048ae843592d701ff47437bd8edd47cd64ca6c0fdf88a71045bdb6fb337fbc63",
}
for name, digest in EXPECTED.items():
    assert hashlib.sha256((SOURCE / "gateway/platforms" / name).read_bytes()).hexdigest() == digest
sys.path.insert(0, str(SOURCE))
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402


async def main():
    assert asyncio.get_running_loop().get_task_factory() is None
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": ""}))
    captured = {}
    executed = []
    @web.middleware
    async def observe(request, handler):
        response = await handler(request)
        if response.status == 202:
            run_id = json.loads(response.body)["run_id"]
            task = adapter._active_run_tasks[run_id]
            assert type(task) is asyncio.Task and not task.done()
            assert not executed
            captured["task"] = task
            captured["run"] = run_id
        return response
    app = web.Application(middlewares=[observe])
    policy_path = Path(__file__).parent / "fixtures/plugin_generation_candidate/policy.py"
    spec = importlib.util.spec_from_file_location("lifecycle_policy", policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    generation = policy.Generation(lambda: None)
    generation.handles = [type("Handle", (), {"active": True})()]
    generation.wire(app, adapter)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    agent = MagicMock()
    def work(**kwargs):
        assert "task" in captured
        row = generation.members[captured["run"]][2]
        assert row["native_task"] is captured["task"]
        assert generation.scope() == (captured["run"], row)
        executed.append(True)
        return {"final_response": "synthetic result"}
    agent.run_conversation.side_effect = work
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/v1/runs", json={"input": "synthetic"},
                                             headers={"X-Lookup-Policy": policy.VERSION})
                assert response.status == 202
                await asyncio.wait_for(captured["task"], 15)
                assert executed == [True]
                assert adapter._run_statuses[captured["run"]]["status"] == "completed"
                assert captured["run"] not in adapter._active_run_tasks
                assert not generation.members and not generation.budgets
                assert captured["run"] in generation.terminal
    finally:
        for task in tuple(adapter._active_run_tasks.values()):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    print(json.dumps({"native_admission_bound_before_execution": True,
                      "native_task_settled": True, "provider_requests": 0}))


if __name__ == "__main__":
    asyncio.run(main())
