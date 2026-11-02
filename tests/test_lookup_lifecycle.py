"""Effect-free checks of persistent selected-run authority retirement."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web

from tests.test_lookup_presentation import setup as presentation_setup

setup = presentation_setup


class Native:
    def __init__(self):
        self._active_run_tasks = {}
        self._run_statuses = {}
        self.serial = 0
        self.started = []

    async def start(self, request):
        self.serial += 1
        run_id = f"run_{self.serial:032x}"
        release = asyncio.Event()
        async def work():
            self.started.append(run_id)
            await release.wait()
            self._run_statuses[run_id] = {"status": "completed"}
            self._active_run_tasks.pop(run_id, None)
        task = asyncio.create_task(work())
        self._active_run_tasks[run_id] = task
        self._run_statuses[run_id] = {"status": "running"}
        self.release = release
        self.task = task
        return web.json_response({"run_id": run_id, "replayed": False}, status=202)


def request(module, method="POST", path="/v1/runs", selected=True):
    async def body():
        return {"input": "synthetic lookup"}
    return SimpleNamespace(method=method, path=path, match_info={}, json=body,
        headers={"X-Lookup-Policy": module.VERSION} if selected else {})


@pytest.mark.asyncio
async def test_settled_runs_release_capacity_and_keep_bounded_authority(setup):
    module = setup.module
    generation = module.Generation(lambda: setup.refused)
    generation.handles = [SimpleNamespace(active=True)]
    native = Native()
    app = web.Application()
    def bind(adapter):
        generation.adapter = adapter
        generation.native_tasks = adapter._active_run_tasks
        generation.native_statuses = adapter._run_statuses
        generation.native_loop = asyncio.get_running_loop()
    generation.bind_native = bind
    generation.wire(app, native)
    admission = app.middlewares[-1]
    ids = []
    try:
        for _ in range(260):
            response = await admission(request(module), native.start)
            assert response.status == 202
            ids.append(json.loads(response.body)["run_id"])
            # The native task cannot start before middleware captures authority.
            assert ids[-1] not in native.started
            native.release.set()
            await native.task
            await asyncio.sleep(0)
        assert len(generation.members) == len(generation.budgets) == 0
        assert len(generation.terminal) <= 128
        assert len(generation.refused) == 0
        setup.session["run"] = ids[0]
        with pytest.raises(ValueError):
            generation.scope()
        async def status(_):
            return web.json_response({"status": "completed"})
        recent = await admission(request(module, "GET", "/v1/runs/" + ids[-1]), status)
        assert recent.headers["X-Lookup-Policy-Refused"] == "false"
        old = await admission(request(module, "GET", "/v1/runs/" + ids[0]), status)
        assert old.headers["X-Lookup-Policy-Outcome"] == "unavailable"
        async def replay(_):
            return web.json_response({"run_id": ids[0], "replayed": True}, status=202)
        assert (await admission(request(module), replay)).status == 409
    finally:
        native.release.set()
        await native.task


@pytest_asyncio.fixture
async def life(setup):
    generation = setup.module.Generation(lambda: setup.refused)
    generation.handles = [SimpleNamespace(active=True)]
    native = Native()
    def bind(adapter):
        generation.adapter = adapter
        generation.native_tasks = adapter._active_run_tasks
        generation.native_statuses = adapter._run_statuses
        generation.native_loop = asyncio.get_running_loop()
    generation.bind_native = bind
    app = web.Application()
    generation.wire(app, native)
    yield SimpleNamespace(g=generation, native=native, admission=app.middlewares[-1],
                          module=setup.module, session=setup.session, refused=setup.refused)
    tasks = tuple(native._active_run_tasks.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def start(life, *, selected=True):
    response = await life.admission(request(life.module, selected=selected), life.native.start)
    assert response.status == 202
    run_id = json.loads(response.body)["run_id"]
    life.session["run"] = run_id
    return run_id


@pytest.mark.asyncio
async def test_live_selected_capacity_stays_128(life):
    for _ in range(128):
        await start(life)
    response = await life.admission(request(life.module), life.native.start)
    assert response.status == 503
    assert life.native.serial == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("held", ["task", "terminal", "owned_work", "presentation"])
async def test_retirement_requires_all_settlement_authorities(life, held):
    run_id = await start(life)
    row = life.g.budgets[run_id]
    if held == "task":
        life.native._run_statuses[run_id] = {"status": "completed"}
    if held == "owned_work":
        row["owned_work"] = 1
    if held == "presentation":
        row["presentation_active"] = {"closing": True}
    if held != "task":
        # Hold the callback until the deliberately incomplete authority is installed.
        row["owned_work"] = 1
        life.native.release.set()
        await life.native.task
        if held == "terminal":
            life.native._run_statuses[run_id] = {"status": "running"}
        if held != "owned_work":
            row["owned_work"] = 0
    life.g.retire()
    assert run_id in life.g.members
    assert run_id not in life.g.terminal


@pytest.mark.asyncio
async def test_ordinary_authority_is_exact_live_task_without_selected_capacity(life):
    for _ in range(128):
        await start(life)
    ordinary = await start(life, selected=False)
    task = life.native.task
    assert life.g.scope() == (ordinary, None)
    assert life.g.model_gate(request={}, next_call=lambda _: "ordinary") == "ordinary"
    replacement = asyncio.create_task(asyncio.sleep(0))
    life.native._active_run_tasks[ordinary] = replacement
    with pytest.raises(ValueError):
        life.g.scope()
    life.native._active_run_tasks[ordinary] = task
    life.native.release.set()
    await task
    await asyncio.sleep(0)
    assert len(life.g.ordinary) == 0
    with pytest.raises(ValueError):
        life.g.scope()
    await replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["members", "terminal", "ordinary", "tasks", "statuses"])
async def test_replaced_authority_closes_generation(life, authority):
    await start(life)
    if authority == "tasks":
        life.native._active_run_tasks = dict(life.native._active_run_tasks)
    elif authority == "statuses":
        life.native._run_statuses = dict(life.native._run_statuses)
    else:
        setattr(life.g, authority, {})
    assert not life.g.available()


@pytest.mark.asyncio
async def test_unknown_native_key_never_gets_ordinary_effects(life, capsys):
    life.session["run"] = "run_" + "f" * 32
    calls = []
    assert life.g.model_gate(request={}, next_call=lambda _: calls.append(1)) is life.refused
    assert life.g.tool_gate(next_call=lambda _: calls.append(1), args={}) == life.module.REFUSAL
    assert life.g.wrap_search(lambda _: calls.append(1))({}) == life.module.REFUSAL
    async def extract(_):
        calls.append(1)
    assert await life.g.wrap_extract(extract)({}) == life.module.REFUSAL
    assert not calls
    assert '"category":"effect_refused"' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_custom_task_factory_refuses_before_native_admission(life):
    loop = asyncio.get_running_loop()
    loop.set_task_factory(lambda loop, coro, **kw: asyncio.Task(coro, loop=loop, **kw))
    try:
        response = await life.admission(request(life.module), life.native.start)
        assert response.status == 503
        assert life.native.serial == 0
    finally:
        loop.set_task_factory(None)


@pytest.mark.asyncio
async def test_missing_native_task_binding_refuses_202(life):
    async def missing(_):
        return web.json_response({"run_id": "run_" + "a" * 32}, status=202)
    response = await life.admission(request(life.module), missing)
    assert response.status == 503 and life.g.closed


@pytest.mark.asyncio
async def test_selected_task_identity_replacement_never_retires_or_allows_effect(life):
    run_id = await start(life)
    original = life.native.task
    replacement = asyncio.create_task(asyncio.sleep(0))
    life.native._active_run_tasks[run_id] = replacement
    with pytest.raises(ValueError):
        life.g.scope()
    life.g.retire()
    assert run_id in life.g.members
    assert run_id in life.g.refused
    life.native._active_run_tasks[run_id] = original
    await replacement


@pytest.mark.asyncio
async def test_refusal_survives_recent_terminal_attestation(life):
    run_id = await start(life)
    life.g.refuse(run_id)
    life.native.release.set()
    await life.native.task
    await asyncio.sleep(0)
    async def status(_):
        return web.json_response({"status": "completed"})
    response = await life.admission(request(life.module, "GET", "/v1/runs/" + run_id), status)
    assert response.headers["X-Lookup-Policy-Refused"] == "true"
    assert run_id not in life.g.refused and run_id not in life.g.budgets


@pytest.mark.asyncio
async def test_inflight_model_keeps_authority_until_cancelled_native_thread_settles(life):
    import threading

    from tests.test_lookup_presentation import completion
    run_id = await start(life)
    row = life.g.budgets[run_id]
    row["phase"] = "extract"
    row["urls"] = {"https://example.invalid/source"}
    entered, release = threading.Event(), threading.Event()
    def provider(_):
        entered.set()
        assert release.wait(5)
        return completion("synthetic")
    operation = asyncio.create_task(asyncio.to_thread(
        life.g.model_gate, api_mode="chat_completions", api_request_id="test-request",
        request={"tools": []}, next_call=provider))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        life.native.task.cancel()
        await asyncio.gather(life.native.task, return_exceptions=True)
        life.native._run_statuses[run_id] = {"status": "cancelled"}
        life.g.retire()
        assert run_id in life.g.members
        assert row["owned_work"] == 1
        release.set()
        assert await operation is life.refused
        assert run_id not in life.g.members and row["owned_work"] == 0
        assert life.g.terminal[run_id][2] is True
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_inflight_extract_keeps_authority_until_async_cleanup_settles(life):
    run_id = await start(life)
    row = life.g.budgets[run_id]
    row.update(request="request", spent={"call"}, admitted=set())
    args = {"urls": ["https://example.invalid/source"]}
    token = life.g.ticket.set((run_id, "request", "call", "web_extract", life.module.frozen(args)))
    entered, release = asyncio.Event(), asyncio.Event()
    async def original(_):
        entered.set()
        await release.wait()
        return "{}"
    operation = asyncio.create_task(life.g.wrap_extract(original)(args))
    life.g.ticket.reset(token)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        life.native.task.cancel()
        await asyncio.gather(life.native.task, return_exceptions=True)
        life.native._run_statuses[run_id] = {"status": "cancelled"}
        life.g.retire()
        assert run_id in life.g.members
        assert row["owned_work"] == 1
        release.set()
        assert await operation == life.module.REFUSAL
        assert run_id not in life.g.members and row["owned_work"] == 0
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_source_binding_rejects_unreviewed_bytes(setup, monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    for name in ("gateway", "gateway.platforms", "gateway.platforms.api_server",
                 "gateway.platforms.api_server_runs"):
        module = ModuleType(name)
        module.__file__ = str(tmp_path / (name + ".py"))
        from pathlib import Path
        Path(module.__file__).write_text("unreviewed", encoding="utf-8")
        monkeypatch.setitem(sys.modules, name, module)
    server = sys.modules["gateway.platforms.api_server"]
    server.APIServerAdapter = Native
    g = setup.module.Generation(lambda: None)
    with pytest.raises(ValueError, match="native-lifecycle-source"):
        g.bind_native(Native())


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_native_task_binding_rejects_completed_task(life, selected):
    completed = asyncio.create_task(asyncio.sleep(0))
    await completed
    async def native(_):
        run_id = "run_" + "a" * 32
        life.native._active_run_tasks[run_id] = completed
        return web.json_response({"run_id": run_id}, status=202)
    response = await life.admission(request(life.module, selected=selected), native)
    assert response.status == 503 and life.g.closed


@pytest.mark.asyncio
async def test_recent_terminal_replay_preserves_identity_without_readmission(life):
    run_id = await start(life)
    life.native.release.set()
    await life.native.task
    await asyncio.sleep(0)
    async def replay(_):
        return web.json_response({"run_id": run_id, "replayed": True}, status=202)
    response = await life.admission(request(life.module), replay)
    assert response.status == 202
    assert response.headers["X-Lookup-Run"] == run_id
    assert run_id not in life.g.members
    conflicting = request(life.module)
    async def changed_body():
        return {"input": "different"}
    conflicting.json = changed_body
    assert (await life.admission(conflicting, replay)).status == 409
    assert (await life.admission(request(life.module, selected=False), replay)).status == 409


@pytest.mark.asyncio
async def test_completed_selected_task_denies_new_callback_while_work_is_settling(life):
    run_id = await start(life)
    row = life.g.budgets[run_id]
    row["owned_work"] = 1
    life.native.release.set()
    await life.native.task
    with pytest.raises(ValueError):
        life.g.scope()
    row["owned_work"] = 0
    life.g.retire()


@pytest.mark.asyncio
async def test_retirement_refuses_replaced_task_even_after_original_done(life):
    run_id = await start(life)
    row = life.g.budgets[run_id]
    row["owned_work"] = 1
    life.native.release.set()
    await life.native.task
    replacement = asyncio.create_task(asyncio.sleep(0))
    life.native._active_run_tasks[run_id] = replacement
    row["owned_work"] = 0
    life.g.retire()
    assert run_id in life.g.members and run_id in life.g.refused
    await replacement


@pytest.mark.asyncio
async def test_retirement_refuses_replaced_budget(life):
    run_id = await start(life)
    row = life.g.budgets[run_id]
    row["owned_work"] = 1
    life.native.release.set()
    await life.native.task
    life.g.budgets[run_id] = dict(row)
    row["owned_work"] = 0
    life.g.retire()
    assert run_id in life.g.members and run_id in life.g.refused


@pytest.mark.asyncio
async def test_completed_ordinary_task_never_authorizes_callback(life):
    run_id = await start(life, selected=False)
    task = life.native.task
    life.native.release.set()
    await task
    await asyncio.sleep(0)
    # Even a retained task entry is not authority after completion.
    life.native._active_run_tasks[run_id] = task
    life.g.ordinary[task] = run_id
    with pytest.raises(ValueError):
        life.g.scope()


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_native_task_binding_rejects_task_shaped_object(life, selected):
    async def native(_):
        run_id = "run_" + "a" * 32
        class TaskShaped:
            def done(self):
                return False
            def add_done_callback(self, callback):
                pass
        life.native._active_run_tasks[run_id] = TaskShaped()
        return web.json_response({"run_id": run_id}, status=202)
    response = await life.admission(request(life.module, selected=selected), native)
    assert response.status == 503 and life.g.closed
    life.native._active_run_tasks.clear()


@pytest.mark.asyncio
async def test_ordinary_profile_alias_binds_local_native_task(life):
    alias = request(life.module, path="/p/other/v1/runs", selected=False)
    alias.match_info = {"profile": "other"}
    response = await life.admission(alias, life.native.start)
    assert response.status == 202
    run_id = json.loads(response.body)["run_id"]
    life.session["run"] = run_id
    assert life.g.scope() == (run_id, None)
    assert not life.g.closed
    selected = request(life.module, path="/p/other/v1/runs")
    selected.match_info = {"profile": "other"}
    assert (await life.admission(selected, life.native.start)).status == 400
    assert life.native.serial == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "none", "replaced"])
async def test_selected_row_task_tampering_refuses_callback(life, change):
    from tests.test_lookup_presentation import completion

    run_id = await start(life)
    row = life.g.budgets[run_id]
    replacement = None
    if change == "missing":
        row.pop("native_task")
    elif change == "none":
        row["native_task"] = None
    else:
        replacement = asyncio.create_task(asyncio.sleep(0))
        row["native_task"] = replacement
    calls = []
    row["phase"] = "extract"
    try:
        result = life.g.model_gate(
            api_mode="chat_completions", api_request_id="late-request", request={"tools": []},
            next_call=lambda _: calls.append(True) or completion("synthetic"))
        assert result is life.refused and not calls
        life.g.retire()
        assert run_id in life.g.members and run_id in life.g.refused
    finally:
        if replacement is not None:
            await replacement
