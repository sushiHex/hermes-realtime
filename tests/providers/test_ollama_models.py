from __future__ import annotations

import asyncio
import io
import json
import socket

import aiohttp
import pytest
from aiohttp import web

import hermes_realtime.providers.ollama_models as models_module
from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.client.bootstrap import BrowserTokenIssuer
from hermes_realtime.client.session import BrowserModelConfiguration, BrowserSessionDirector
from hermes_realtime.conversation import ConversationContextSnapshot, ConversationMessage
from hermes_realtime.host_launcher import _build_argument_parser, _build_browser_model_configuration
from hermes_realtime.livekit import LiveKitConnection
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import (
    OllamaContextConfiguration,
    OllamaModelClient,
    OllamaModelMetadata,
)

pytestmark = pytest.mark.asyncio

DIGEST = "a" * 64


class Backend:
    def __init__(self, *, maximum=131072, changed=False):
        self.maximum = maximum
        self.changed = changed
        self.requests = []
        self.responses = []

    async def __call__(self, path, body):
        self.requests.append((path, body))
        if path == "show":
            payload = {
                "model_info": {
                    "general.architecture": "synthetic",
                    "synthetic.context_length": self.maximum,
                },
                "capabilities": ["completion"],
            }
        elif path == "version":
            payload = {"version": "0.34.4"}
        else:
            digest = "b" * 64 if self.changed and len(self.requests) > 1 else DIGEST
            payload = {
                "models": [
                    {
                        "name": "test:latest",
                        "model": "test:latest",
                        "digest": digest,
                        "context_length": 4096,
                    }
                ]
            }
        return json.dumps(payload).encode()


async def test_metadata_only_identity_bound_discovery():
    backend = Backend()
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=backend)
    model = await client.discover("test:latest")
    assert (model.model, model.digest, model.advertised_context_tokens) == (
        "test:latest",
        DIGEST,
        131072,
    )
    assert [request[0] for request in backend.requests] == ["tags", "show", "tags"]
    assert backend.requests[1][1] == {"model": "test:latest"}
    assert await client.effective_context(model) == 4096


async def test_identity_change_during_discovery_refuses():
    client = OllamaModelClient(
        base_url="http://127.0.0.1:11434", read_request=Backend(changed=True)
    )
    with pytest.raises(ValueError, match="identity changed"):
        await client.discover("test:latest")


@pytest.mark.parametrize("maximum", [None, True, 0, -1, 100_000_001, "4096", 4096.0])
async def test_invalid_capacity_stays_unknown_requires_explicit_setting(maximum):
    client = OllamaModelClient(
        base_url="http://127.0.0.1:11434", read_request=Backend(maximum=maximum)
    )
    model = await client.discover("test:latest")
    assert model.advertised_context_tokens is None
    with pytest.raises(ValueError, match="explicit"):
        OllamaContextConfiguration(model=model)
    assert (
        OllamaContextConfiguration(
            model=model, requested_context_tokens=4096
        ).requested_context_tokens
        == 4096
    )


async def test_requested_context_does_not_allocate_advertised_maximum():
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=Backend())
    model = await client.discover("test:latest")
    assert OllamaContextConfiguration(model=model).requested_context_tokens == 16384
    assert (
        OllamaContextConfiguration(
            model=model, requested_context_tokens=4096
        ).requested_context_tokens
        == 4096
    )
    with pytest.raises(ValueError, match="advertised"):
        OllamaContextConfiguration(model=model, requested_context_tokens=262144)


async def test_metadata_read_bound_is_applied_before_parsing():
    async def oversized(endpoint, body):
        return b" " * 1_048_577

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=oversized)
    with pytest.raises(ValueError, match="size"):
        await client.discover("test:latest")


@pytest.mark.parametrize("status", [200, 204, 302])
async def test_metadata_transport_requires_http_200_before_reading(status, capsys):
    requests = []
    payload = b'{"version":"0.34.4"}'

    async def response(request):
        requests.append(request.path)
        return web.Response(status=status, body=payload, headers={"Location": "/redirect"})

    app = web.Application()
    app.router.add_get("/api/version", response)
    app.router.add_get("/redirect", response)
    runner = web.AppRunner(app)
    await runner.setup()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    try:
        await web.SockSite(runner, listener).start()
        client = OllamaModelClient(base_url=f"http://127.0.0.1:{port}")
        if status == 200:
            await client.require_context_contract()
        else:
            with pytest.raises(aiohttp.ClientResponseError) as refusal:
                await client.require_context_contract()
            assert refusal.value.status == status
        assert requests == ["/api/version"]
        reports = [
            json.loads(line.removeprefix("[ollama-metadata] "))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[ollama-metadata] ")
        ]
        assert reports == [
            {
                "version": 1,
                "kind": "version",
                "category": "accepted" if status == 200 else "unavailable",
                "bytes": len(payload) if status == 200 else 0,
            }
        ]
    finally:
        await runner.cleanup()
        listener.close()


@pytest.mark.parametrize(
    ("endpoint", "raw"),
    [
        ("version", b'{"version":"wrong","version":"0.34.4"}'),
        ("version", b'{"version":"0.34.4","version":"0.34.4"}'),
        ("tags", b'{"models":[false],"models":[]}'),
        ("tags", b'{"models":[{"name":"other","name":"test:latest"}]}'),
        ("ps", b'{"models":[{"digest":"same","digest":"same"}]}'),
        ("show", b'{"model_info":{"context_length":4096,"context_length":16384}}'),
    ],
    ids=[
        "root-conflict",
        "root-same",
        "root-models",
        "nested-name",
        "nested-same",
        "nested-capacity",
    ],
)
async def test_metadata_rejects_duplicate_json_keys_at_every_object(endpoint, raw, capsys):
    async def read(path, body):
        assert path == endpoint
        return raw

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ValueError, match="unique"):
        await client._request(endpoint)
    assert capsys.readouterr().out.splitlines() == [
        "[ollama-metadata] "
        + json.dumps(
            {"version": 1, "kind": endpoint, "category": "malformed", "bytes": len(raw)},
            separators=(",", ":"),
        )
    ]


async def test_effective_context_requires_exact_digest():
    backend = Backend()
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=backend)
    model = await client.discover("test:latest")
    backend.changed = True
    with pytest.raises(ValueError, match="identity"):
        await client.effective_context(model)


async def test_runtime_contract_rejects_unknown_versions():
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=Backend())
    await client.require_context_contract()

    async def old_runtime(endpoint, body):
        return b'{"version":"0.1.0"}'

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=old_runtime)
    with pytest.raises(ValueError, match="runtime"):
        await client.require_context_contract()


async def test_small_model_requires_explicit_smaller_context():
    client = OllamaModelClient(
        base_url="http://127.0.0.1:11434", read_request=Backend(maximum=4096)
    )
    model = await client.discover("test:latest")
    with pytest.raises(ValueError, match="advertised"):
        OllamaContextConfiguration(model=model)
    assert (
        OllamaContextConfiguration(
            model=model, requested_context_tokens=4096
        ).requested_context_tokens
        == 4096
    )


async def test_metadata_request_deadline_cancels_drip_reader(monkeypatch, capsys):
    cancelled = asyncio.Event()

    async def drip(endpoint, body):
        try:
            while True:
                await asyncio.sleep(0.001)
        finally:
            cancelled.set()

    monkeypatch.setattr(models_module, "_REQUEST_TIMEOUT_SECONDS", 0.02)
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=drip)
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(client.discover("test:latest"), 0.2)
    assert asyncio.get_running_loop().time() - started < 0.1
    assert cancelled.is_set()
    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[ollama-metadata] ")
    )
    report = json.loads(line.removeprefix("[ollama-metadata] "))
    assert report == {"version": 1, "kind": "tags", "category": "unavailable", "bytes": 0}


async def test_discovery_has_one_deadline_across_metadata_requests(monkeypatch):
    backend = Backend()

    async def slow(endpoint, body):
        await asyncio.sleep(0.02)
        return await backend(endpoint, body)

    monkeypatch.setattr(models_module, "_DISCOVERY_TIMEOUT_SECONDS", 0.03)
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=slow)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(client.discover("test:latest"), 0.2)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"name": "test:latest", "digest": DIGEST}] * 2,
        [{"name": "test:latest", "model": "different", "digest": DIGEST}],
        [{"name": "test:latest", "digest": "invalid"}],
        [{"name": "test:latest", "digest": DIGEST}]
        + [{"name": f"other_{index}", "digest": DIGEST} for index in range(256)],
        [False],
    ],
)
async def test_ambiguous_or_malformed_identity_refuses(rows):
    async def read(endpoint, body):
        return json.dumps({"models": rows}).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises((TypeError, ValueError)):
        await client.identity("test:latest")


@pytest.mark.parametrize(
    "capabilities", [None, ["embedding"], ["completion", True], ["completion"] * 33]
)
async def test_non_completion_metadata_refuses(capabilities):
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "show":
            payload["capabilities"] = capabilities
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ValueError, match="completion"):
        await client.discover("test:latest")


async def test_multiple_effective_allocations_are_refused():
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            payload["models"].append({**payload["models"][0], "context_length": 16384})
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ValueError, match="identity"):
        await client.effective_context(await client.discover("test:latest"))


@pytest.mark.parametrize("boundary", ["tags", "ps"])
@pytest.mark.parametrize(
    ("extra", "reverse", "expected"),
    [
        ({"name": "test:latest", "digest": "b" * 64}, False, None),
        ({"name": "test:latest", "digest": "b" * 64}, True, None),
        ({"name": "other", "model": "test:latest", "digest": DIGEST}, False, None),
        ({"name": "test:latest", "model": "other", "digest": DIGEST}, False, None),
        ({"name": "other", "model": "other", "digest": DIGEST}, False, 4096),
    ],
    ids=["stale-after", "stale-before", "conflicting-name", "conflicting-model", "unrelated"],
)
async def test_selected_model_counts_all_rows_referring_to_it(extra, reverse, expected, boundary):
    selected = {"name": "test:latest", "digest": DIGEST, "context_length": 4096}
    rows = [selected, {**extra, "context_length": 16384}]
    if reverse:
        rows.reverse()

    async def read(endpoint, body):
        assert (endpoint, body) == (boundary, None)
        return json.dumps({"models": rows}).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    if boundary == "tags":
        if expected is None:
            with pytest.raises(ValueError, match="identity"):
                await client.identity("test:latest")
        else:
            assert await client.identity("test:latest") == DIGEST
    else:
        if expected is None:
            with pytest.raises(ValueError, match="identity"):
                await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072))
        else:
            assert (
                await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072))
                == expected
            )


@pytest.mark.parametrize("boundary", ["tags", "ps"])
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"name": "test:latest"}, 4096),
        ({"name": "test:latest", "model": "test:latest"}, 4096),
        ({"name": "other", "model": "test:latest"}, None),
        ({"model": "test:latest"}, None),
        ({"name": "test:latest", "model": "other"}, None),
    ],
    ids=[
        "optional-model",
        "consistent",
        "conflicting-name",
        "missing-name",
        "conflicting-model",
    ],
)
async def test_selected_model_requires_one_consistent_row(row, expected, boundary):
    async def read(endpoint, body):
        assert (endpoint, body) == (boundary, None)
        return json.dumps({"models": [{"digest": DIGEST, "context_length": 4096, **row}]}).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    if boundary == "tags":
        if expected is None:
            with pytest.raises(ValueError, match="identity"):
                await client.identity("test:latest")
        else:
            assert await client.identity("test:latest") == DIGEST
    else:
        if expected is None:
            with pytest.raises(ValueError, match="identity"):
                await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072))
        else:
            assert (
                await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072))
                == expected
            )


async def test_effective_context_refuses_invalid_capacity():
    async def read(endpoint, body):
        return json.dumps(
            {"models": [{"name": "test:latest", "digest": DIGEST, "context_length": True}]}
        ).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert (
        await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072)) is None
    )


@pytest.mark.usefixtures("deny_uninjected_ollama_metadata")
async def test_session_setting_reaches_inference_and_freezes_identity():
    backend = Backend()
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=backend)
    bodies = []

    def generate(request, timeout):
        bodies.append(json.loads(request.data))
        return io.BytesIO(b'{"message":{"content":"Ready."},"done":true}\n')

    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test:latest", open_request=generate
    )
    await inference.configure_context(4096, client=client)
    with pytest.raises(RuntimeError, match="frozen"):
        await inference.configure_context(16384, client=client)
    initial = inference.context_public_data()
    assert initial == {
        "version": 2,
        "requestedContextTokens": 4096,
        "advertisedContextTokens": 131072,
        "effectiveContextTokens": None,
        "modelDigest": DIGEST,
    }
    snapshot = ConversationContextSnapshot(
        revision=0, messages=(ConversationMessage("user", "Synthetic question."),), active_tasks=()
    )
    assert [part async for part in inference.stream(snapshot, turn_id="turn_test")] == ["Ready."]
    assert bodies[0]["options"] == {"num_ctx": 4096}
    assert bodies[0]["messages"][1:] == [{"role": "user", "content": "Synthetic question."}]
    assert bodies[0]["truncate"] is False and bodies[0]["shift"] is False
    await inference.observe_context()
    assert inference.context_public_data() == {**initial, "effectiveContextTokens": 4096}
    backend.changed = True
    with pytest.raises(ValueError, match="identity changed"):
        await inference.observe_context()
    assert inference.context_public_data()["effectiveContextTokens"] is None
    await inference.close()


@pytest.mark.usefixtures("deny_uninjected_ollama_metadata")
async def test_inflight_configuration_and_request_share_the_admission_lock():
    entered = asyncio.Event()
    release = asyncio.Event()
    backend = Backend()
    bodies = []

    async def read(endpoint, body):
        if endpoint == "show":
            entered.set()
            await release.wait()
        return await backend(endpoint, body)

    def generate(request, timeout):
        bodies.append(json.loads(request.data))
        return io.BytesIO(b'{"message":{"content":"Ready."},"done":true}\n')

    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test:latest", open_request=generate
    )
    configuration = asyncio.create_task(
        inference.configure_context(
            4096, client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
        )
    )
    await entered.wait()

    async def consume():
        snapshot = ConversationContextSnapshot(
            revision=0,
            messages=(ConversationMessage("user", "Synthetic question."),),
            active_tasks=(),
        )
        return [part async for part in inference.stream(snapshot, turn_id="turn_race")]

    stream = asyncio.create_task(consume())
    await asyncio.sleep(0)
    assert not bodies
    release.set()
    await configuration
    assert await stream == ["Ready."]
    assert bodies[0]["options"] == {"num_ctx": 4096}
    await inference.close()


async def test_conflicting_effective_model_fields_are_refused():
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            payload["models"][0]["model"] = "different"
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ValueError, match="identity"):
        await client.effective_context(await client.discover("test:latest"))


@pytest.mark.parametrize("endpoint", ["tags", "show"])
@pytest.mark.parametrize("field", ["remote_model", "remote_host"])
@pytest.mark.parametrize("value", ["remote", True, None])
async def test_remote_backed_or_malformed_model_metadata_refuses(endpoint, field, value):
    backend = Backend()

    async def read(kind, body):
        payload = json.loads(await backend(kind, body))
        if kind == endpoint:
            target = payload["models"][0] if kind == "tags" else payload
            target[field] = value
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ValueError, match="local"):
        await client.discover("test:latest")


@pytest.mark.parametrize("field", ["remote_host", "remote_model"])
@pytest.mark.parametrize(
    "value",
    ["remote", True, None, False, 0, [], {}],
    ids=["nonempty", "true", "null", "false", "number", "array", "object"],
)
async def test_remote_or_malformed_allocation_is_refused_and_observed_as_unknown(
    field, value, capsys
):
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            payload["models"][0][field] = value
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    model = await client.discover("test:latest")
    with pytest.raises(ValueError, match="local"):
        await client.effective_context(model)
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test:latest", metadata_client=client
    )
    try:
        await inference.configure_context(4096)
        capsys.readouterr()
        await inference.observe_context()
        assert inference.context_public_data()["effectiveContextTokens"] is None
        reports = [
            json.loads(line.removeprefix("[ollama-context-observation] "))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[ollama-context-observation] ")
        ]
        assert reports == [{"version": 1, "kind": "allocation", "category": "metadata_refused"}]
    finally:
        await inference.close()


async def test_empty_remote_fields_preserve_local_allocation():
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            payload["models"][0].update(remote_host="", remote_model="")
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert await client.effective_context(await client.discover("test:latest")) == 4096


async def test_metadata_transport_reads_only_to_the_byte_cap_and_disables_redirects(monkeypatch):
    remaining = 2_000_000
    requested = []
    options = {}

    class Response:
        content = None
        status = 200

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            options["response_closed"] = True

        async def read(self, size):
            nonlocal remaining
            requested.append(size)
            remaining -= size
            return b" " * size

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            options["client_closed"] = True

        def request(self, method, url, **kwargs):
            options.update(kwargs)
            return Response()

    monkeypatch.setattr(models_module.aiohttp, "ClientSession", Client)
    client = OllamaModelClient(base_url="http://127.0.0.1:11434")
    with pytest.raises(ValueError, match="size"):
        await client.discover("test:latest")
    assert sum(requested) == 1_048_577
    assert max(requested) <= 65536
    assert options["trust_env"] is False and options["allow_redirects"] is False
    assert options["response_closed"] and options["client_closed"]


@pytest.mark.parametrize("value", [b"[]", b"false", "{}"])
async def test_metadata_response_requires_exact_bytes_and_object(value):
    async def read(endpoint, body):
        return value

    with pytest.raises(TypeError):
        await OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read).identity(
            "test:latest"
        )


@pytest.mark.parametrize(
    "value,error",
    [(True, TypeError), (4096.0, TypeError), (2047, ValueError), (262145, ValueError)],
)
async def test_explicit_setting_requires_supported_exact_integer(value, error):
    model = await OllamaModelClient(
        base_url="http://127.0.0.1:11434", read_request=Backend(maximum=1_048_576)
    ).discover("test:latest")
    with pytest.raises(error):
        OllamaContextConfiguration(model=model, requested_context_tokens=value)


@pytest.mark.parametrize(
    "origin",
    [
        "https://127.0.0.1:11434",
        "http://example.test:11434",
        "http://127.0.0.1",
        "http://user@127.0.0.1:11434",
        "http://127.0.0.1:11434/private",
        "http://127.0.0.1:11434?private=true",
    ],
)
async def test_metadata_client_requires_explicit_loopback_origin(origin):
    with pytest.raises(ValueError, match="loopback"):
        OllamaModelClient(base_url=origin)


async def test_allocation_observability_failure_stays_unknown():
    backend = Backend()

    async def read(endpoint, body):
        if endpoint == "ps":
            raise TimeoutError
        return await backend(endpoint, body)

    inference = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test:latest")
    await inference.configure_context(
        4096, client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    )
    await inference.observe_context()
    assert inference.context_public_data()["effectiveContextTokens"] is None
    await inference.close()


@pytest.mark.parametrize(
    "case,category",
    [
        ("stale", "metadata_refused"),
        ("missing_digest", "metadata_refused"),
        ("ambiguous", "metadata_refused"),
        ("inconsistent_name", "metadata_refused"),
        ("inconsistent_model", "metadata_refused"),
        ("missing_name", "metadata_refused"),
        ("absent", "unknown"),
        ("capacity_missing", "unknown"),
        ("capacity_null", "unknown"),
    ],
)
async def test_allocation_source_refusals_are_distinct_from_unavailable_capacity(
    case, category, capsys
):
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            row = payload["models"][0]
            if case == "stale":
                row["digest"] = "b" * 64
            elif case == "missing_digest":
                row.pop("digest")
            elif case == "ambiguous":
                payload["models"].append(row.copy())
            elif case == "inconsistent_name":
                row["name"] = "other"
            elif case == "inconsistent_model":
                row["model"] = "other"
            elif case == "missing_name":
                row.pop("name")
            elif case == "absent":
                payload["models"] = []
            elif case == "capacity_missing":
                row.pop("context_length")
            elif case == "capacity_null":
                row["context_length"] = None
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test:latest", metadata_client=client
    )
    try:
        await inference.configure_context(4096)
        capsys.readouterr()
        await inference.observe_context()
        assert inference.context_public_data()["effectiveContextTokens"] is None
        reports = [
            json.loads(line.removeprefix("[ollama-context-observation] "))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[ollama-context-observation] ")
        ]
        assert reports == [{"version": 1, "kind": "allocation", "category": category}]
    finally:
        await inference.close()


@pytest.mark.parametrize("outcome", ["observed", "unknown", "unavailable", "metadata_refused"])
async def test_allocation_observation_records_one_content_free_outcome(outcome, capsys):
    backend = Backend()

    async def read(endpoint, body):
        if endpoint == "ps":
            if outcome == "unavailable":
                raise TimeoutError
            if outcome == "metadata_refused":
                return b'{"models":[false]}'
            if outcome == "unknown":
                return b'{"models":[]}'
        return await backend(endpoint, body)

    inference = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test:latest")
    await inference.configure_context(
        4096, client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    )
    capsys.readouterr()
    await inference.observe_context()
    reports = [
        json.loads(line.removeprefix("[ollama-context-observation] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[ollama-context-observation] ")
    ]
    assert reports == [{"version": 1, "kind": "allocation", "category": outcome}]
    assert inference.context_public_data()["effectiveContextTokens"] == (
        4096 if outcome == "observed" else None
    )
    await inference.close()


@pytest.mark.parametrize("outcome", ["not_configured", "identity_refused", "identity_unavailable"])
async def test_observation_records_missing_configuration_and_identity_refusal(outcome, capsys):
    backend = Backend()
    unavailable = False

    async def read(endpoint, body):
        if unavailable:
            raise TimeoutError
        return await backend(endpoint, body)

    inference = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test:latest")
    if outcome != "not_configured":
        await inference.configure_context(
            4096, client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
        )
    backend.changed = outcome == "identity_refused"
    unavailable = outcome == "identity_unavailable"
    capsys.readouterr()
    if outcome == "not_configured":
        await inference.observe_context()
    else:
        with pytest.raises(ValueError if outcome == "identity_refused" else TimeoutError):
            await inference.observe_context()
        assert inference.context_public_data()["effectiveContextTokens"] is None
    reports = [
        json.loads(line.removeprefix("[ollama-context-observation] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[ollama-context-observation] ")
    ]
    assert reports == [{"version": 1, "kind": "allocation", "category": outcome}]
    await inference.close()


async def test_cancelled_allocation_observation_records_its_outcome(capsys):
    entered = asyncio.Event()
    backend = Backend()

    async def read(endpoint, body):
        if endpoint == "ps":
            entered.set()
            await asyncio.Event().wait()
        return await backend(endpoint, body)

    inference = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test:latest")
    await inference.configure_context(
        4096, client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    )
    capsys.readouterr()
    operation = asyncio.create_task(inference.observe_context())
    await entered.wait()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    reports = [
        json.loads(line.removeprefix("[ollama-context-observation] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[ollama-context-observation] ")
    ]
    assert reports == [{"version": 1, "kind": "allocation", "category": "cancelled"}]
    assert inference.context_public_data()["effectiveContextTokens"] is None
    await inference.close()


@pytest.mark.usefixtures("deny_uninjected_ollama_metadata")
async def test_configuring_after_first_inference_is_refused():
    inference = OllamaStreamingInference(
        metadata_client=OllamaModelClient(
            base_url="http://127.0.0.1:11434", read_request=Backend()
        ),
        base_url="http://127.0.0.1:11434",
        model="test:latest",
        open_request=lambda request, timeout: io.BytesIO(
            b'{"message":{"content":"Ready."},"done":true}\n'
        ),
    )
    snapshot = ConversationContextSnapshot(
        revision=0, messages=(ConversationMessage("user", "Synthetic question."),), active_tasks=()
    )
    assert [part async for part in inference.stream(snapshot, turn_id="turn_started")]
    with pytest.raises(RuntimeError, match="first inference"):
        await inference.configure_context(
            4096,
            client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=Backend()),
        )
    await inference.close()


async def test_host_context_cli_and_browser_configuration_are_opt_in():
    parser = _build_argument_parser()
    assert parser.parse_args([]).ollama_num_ctx is None
    assert parser.parse_args(["--ollama-num-ctx", "4096"]).ollama_num_ctx == 4096
    inference = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test:latest")
    await inference.configure_context(
        4096, client=OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=Backend())
    )
    model = _build_browser_model_configuration(
        inference_provider="ollama",
        ollama_model="test:latest",
        codex_model="unused",
        codex_effort="low",
        local_context=inference.context_observation,
    )
    assert model.public_data() == {
        "authentication": "local",
        "contextWindowTokens": None,
        "effort": None,
        "model": "test:latest",
        "provider": "ollama",
        "reportsTokenUsage": False,
        "transport": "local-http",
        **inference.context_public_data(),
    }
    other_model = BrowserModelConfiguration(
        authentication="local",
        provider="ollama",
        model="different",
        transport="local-http",
        effort=None,
        context_window_tokens=None,
        reports_token_usage=False,
        local_context=inference.context_observation,
    )
    with pytest.raises(ValueError, match="identity"):
        other_model.public_data()
    projection = BrowserEventProjection()

    async def provision(identity):
        return 1

    async def noop(*args):
        return None

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test", "test-key", "synthetic-browser-bootstrap-secret-32-bytes"
            ),
            room_name="test",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=noop,
        stop=noop,
        approval=noop,
        projection=projection,
        model_configuration=model,
    )
    await director.start()
    assert dict(projection.events_after(0)[1].data) == model.public_data()
    await inference.close()
