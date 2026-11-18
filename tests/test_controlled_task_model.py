"""Real loopback HTTP streams; no model, credentials, task events or native witness doubles."""
import asyncio
import json

import aiohttp
import pytest

from scripts.controlled_task_model import ControlledTaskModel

OBJECTIVES = {"a": "Compare rail ticket prices in detail.", "b": "Summarize ocean research."}


def body(text):
    return {"messages": [{"role": "user", "content": text}], "stream": True}


@pytest.mark.asyncio
async def test_model_discovery_does_not_admit_work():
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        assert {address[0] for address in model._runner.addresses} == {"127.0.0.1"}
        async with (aiohttp.ClientSession() as client,
                    client.get(url + "/models") as response):
            assert response.status == 200
            assert (await response.json())["data"][0]["id"] == "controlled"
        assert model.snapshot()["admitted"] == 0
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_held_stream_yields_model_chunks_without_completing():
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as client:
            response = await client.post(url + "/chat/completions", json=body(OBJECTIVES["a"]))
            line = await asyncio.wait_for(response.content.readline(), 2)
            assert line.startswith(b"data: ")
            chunk = json.loads(line.removeprefix(b"data: "))
            assert chunk["choices"][0]["delta"]["content"] == "."
            assert chunk["choices"][0]["finish_reason"] is None
            assert model.snapshot()["open"] == 1
            response.close()
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_independent_gates_hold_until_release_or_client_disconnect():
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as client:
            a = await client.post(url + "/chat/completions", json=body(OBJECTIVES["a"]))
            b = await client.post(url + "/chat/completions", json=body(OBJECTIVES["b"]))
            await asyncio.wait_for(model.wait_started("a"), 2)
            await asyncio.wait_for(model.wait_started("b"), 2)
            assert model.snapshot()["open"] == 2
            # A client disconnect must settle A without releasing B.
            a.close()
            await asyncio.wait_for(model.wait_closed("a"), 2)
            assert model.snapshot() == {"admitted": 2, "open": 1, "closed": 1, "refused": 0}
            model.release("b")
            assert b"[DONE]" in await b.read()
            await asyncio.wait_for(model.wait_closed("b"), 2)
            assert model.snapshot()["open"] == 0
    finally:
        await model.close()


@pytest.mark.parametrize("request_body", [
    body("prefix " + OBJECTIVES["a"]), body(OBJECTIVES["a"] + " suffix"),
    body("unknown"), {}, {**body(OBJECTIVES["a"]), "stream": False},
    {"messages": [{"role": "assistant", "content": OBJECTIVES["a"]}], "stream": True},
    {"messages": [{"role": "tool", "content": OBJECTIVES["a"]}], "stream": True},
    {"messages": [{"role": "user", "content": OBJECTIVES["a"]},
                  {"role": "user", "content": "unknown"}], "stream": True},
    {"messages": [{"role": "user", "content": OBJECTIVES["a"]},
                  {"role": "tool", "content": "result"}], "stream": True},
])
@pytest.mark.asyncio
async def test_unknown_stale_nonstream_and_tool_result_requests_are_refused(request_body):
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(url + "/chat/completions", json=request_body) as response:
                assert response.status == 400
            assert model.snapshot() == {"admitted": 0, "open": 0, "closed": 0, "refused": 1}
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_duplicate_admission_and_remote_release_are_refused():
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as client:
            a = await client.post(url + "/chat/completions", json=body(OBJECTIVES["a"]))
            async with client.post(url + "/chat/completions", json=body(OBJECTIVES["a"])) as r:
                assert r.status == 409
            async with client.post(url + "/release/a") as r:
                assert r.status == 404
            a.close()
            await asyncio.wait_for(model.wait_closed("a"), 2)
            async with client.post(url + "/chat/completions", json=body(OBJECTIVES["a"])) as r:
                assert r.status == 409
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_close_settles_every_stream_and_metadata_contains_no_objectives():
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    async with aiohttp.ClientSession() as client:
        responses = [await client.post(url + "/chat/completions", json=body(text))
                     for text in OBJECTIVES.values()]
        await asyncio.wait_for(model.close(), 2)
        assert model.snapshot()["closed"] == 2
        assert model.snapshot()["open"] == 0
        assert not any(text in json.dumps(model.snapshot()) for text in OBJECTIVES.values())
        for response in responses:
            response.close()
    await model.close()  # idempotent settlement


@pytest.mark.asyncio
async def test_release_of_a_does_not_release_b():
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        async with aiohttp.ClientSession() as client:
            a = await client.post(url + "/chat/completions", json=body(OBJECTIVES["a"]))
            b = await client.post(url + "/chat/completions", json=body(OBJECTIVES["b"]))
            model.release("a")
            assert b"[DONE]" in await a.read()
            assert model.snapshot()["open"] == 1
            assert not model._closed["b"].is_set()
            b.close()
            await asyncio.wait_for(model.wait_closed("b"), 2)
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_request_parsing_across_shutdown_cannot_admit_work():
    model = ControlledTaskModel(OBJECTIVES)
    parsing, proceed = asyncio.Event(), asyncio.Event()

    class Request:
        async def json(self):
            parsing.set()
            await proceed.wait()
            return body(OBJECTIVES["a"])

    pending = asyncio.create_task(model._complete(Request()))
    await parsing.wait()
    await model.close()
    proceed.set()
    response = await asyncio.wait_for(pending, 2)
    assert response.status == 400
    assert model.snapshot()["admitted"] == 0


@pytest.mark.asyncio
async def test_refusal_marker_contains_only_bounded_categories_and_counts(capsys):
    model = ControlledTaskModel(OBJECTIVES)
    url = await model.start()
    try:
        async with (aiohttp.ClientSession() as client,
                    client.post(url + "/chat/completions", json=body("unknown"))):
            pass
        line = capsys.readouterr().out.strip()
        assert line.startswith("[controlled-task-refusal] ")
        assert json.loads(line.split("] ", 1)[1]) == {"category": "request_refused", "count": 1}
    finally:
        await model.close()


@pytest.mark.parametrize("objectives", [
    {}, {"a": "same", "b": "same"}, {"a": ""}, {1: "value"}, {"a": 1},
    {str(i): str(i) for i in range(5)},
])
def test_ambiguous_or_empty_objectives_are_refused(objectives):
    with pytest.raises(ValueError):
        ControlledTaskModel(objectives)


@pytest.mark.asyncio
async def test_start_refuses_reuse_and_reopening_after_close():
    model = ControlledTaskModel(OBJECTIVES)
    await model.start()
    try:
        with pytest.raises(RuntimeError):
            await model.start()
    finally:
        await model.close()
    with pytest.raises(RuntimeError):
        await model.start()

