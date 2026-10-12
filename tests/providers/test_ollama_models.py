from __future__ import annotations

import asyncio
import io
import json

import pytest

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


async def test_effective_context_requires_exact_digest():
    backend = Backend()
    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=backend)
    model = await client.discover("test:latest")
    backend.changed = True
    assert await client.effective_context(model) is None


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


async def test_multiple_effective_allocations_stay_unknown():
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            payload["models"].append({**payload["models"][0], "context_length": 16384})
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert await client.effective_context(await client.discover("test:latest")) is None


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
async def test_effective_context_counts_all_rows_referring_to_selected_model(
    extra, reverse, expected
):
    selected = {"name": "test:latest", "digest": DIGEST, "context_length": 4096}
    rows = [selected, {**extra, "context_length": 16384}]
    if reverse:
        rows.reverse()

    async def read(endpoint, body):
        assert (endpoint, body) == ("ps", None)
        return json.dumps({"models": rows}).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert (
        await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072))
        == expected
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"name": "test:latest"}, 4096),
        ({"name": "test:latest", "model": "test:latest"}, 4096),
        ({"name": "other", "model": "test:latest"}, None),
        ({"model": "test:latest"}, None),
        ({"name": "test:latest", "model": "other"}, None),
        ({"name": "test:latest", "context_length": True}, None),
    ],
    ids=[
        "optional-model",
        "consistent",
        "conflicting-name",
        "missing-name",
        "conflicting-model",
        "invalid-capacity",
    ],
)
async def test_effective_context_requires_one_consistent_selected_row(row, expected):
    async def read(endpoint, body):
        assert (endpoint, body) == ("ps", None)
        return json.dumps({"models": [{"digest": DIGEST, "context_length": 4096, **row}]}).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert (
        await client.effective_context(OllamaModelMetadata("test:latest", DIGEST, 131072))
        == expected
    )


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


async def test_conflicting_effective_model_fields_stay_unknown():
    backend = Backend()

    async def read(endpoint, body):
        payload = json.loads(await backend(endpoint, body))
        if endpoint == "ps":
            payload["models"][0]["model"] = "different"
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert await client.effective_context(await client.discover("test:latest")) is None


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


async def test_metadata_transport_reads_only_to_the_byte_cap_and_disables_redirects(monkeypatch):
    remaining = 2_000_000
    requested = []
    options = {}

    class Response:
        content = None

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            options["response_closed"] = True

        def raise_for_status(self):
            pass

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


async def test_configuring_after_first_inference_is_refused():
    inference = OllamaStreamingInference(
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
