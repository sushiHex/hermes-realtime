from __future__ import annotations

import asyncio
import contextlib

import pytest
from test_ollama_catalog import CatalogBackend

from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import (
    OllamaContextConfiguration,
    OllamaContextObservation,
    OllamaModelMetadata,
)
from hermes_realtime.providers.ollama_session import OllamaSessionSelection

pytestmark = pytest.mark.asyncio


class Harness:
    def __init__(self, monkeypatch):
        self.backend = CatalogBackend()
        self.adapters = []
        self.closed = []
        self.ready = []
        self.block = None
        self.fail = False
        harness = self

        async def configure(adapter, requested, *, client):
            assert client is harness.client
            digest = "d" * 64 if harness.backend.changed else "a" * 64
            adapter._context_observation = OllamaContextObservation(
                OllamaContextConfiguration(OllamaModelMetadata("small", digest, 4096), requested)
            )

        async def observe(adapter):
            return adapter.context_observation()

        async def close(adapter):
            harness.closed.append(adapter)

        monkeypatch.setattr(OllamaStreamingInference, "configure_context", configure)
        monkeypatch.setattr(OllamaStreamingInference, "observe_context", observe)
        monkeypatch.setattr(OllamaStreamingInference, "close", close)
        self.client = self.backend.client()
        self.owner = OllamaSessionSelection(
            client=self.client,
            default_model="small",
            adapter_factory=self.factory,
            readiness=self.readiness,
        )

    def factory(self, model):
        adapter = OllamaStreamingInference(base_url="http://127.0.0.1:11434", model=model)
        self.adapters.append(adapter)
        return adapter

    async def readiness(self, adapter):
        self.ready.append(adapter)
        if self.block is not None:
            await self.block.wait()
        if self.fail:
            raise RuntimeError("synthetic readiness failure")

    async def prepare(self):
        return await self.owner.prepare("small", "a" * 64, 4096)


async def test_prepare_freezes_exact_adapter_and_context(monkeypatch):
    harness = Harness(monkeypatch)
    with pytest.raises(RuntimeError):
        harness.owner.require_prepared()
    catalog = await harness.owner.catalog()
    assert catalog["state"] == "configuring" and catalog["selection"] is None
    selection = await harness.prepare()
    assert selection == {
        "model": "small",
        "digest": "a" * 64,
        "requestedContextTokens": 4096,
        "advertisedContextTokens": 4096,
        "effectiveContextTokens": None,
    }
    configuration = harness.owner.require_prepared()
    assert configuration.model == "small"
    assert configuration.local_context() is harness.adapters[0].context_observation()
    assert await harness.prepare() == selection
    assert harness.ready == harness.adapters
    request_count = len(harness.backend.requests)
    assert (await harness.owner.catalog())["selection"] == selection
    assert len(harness.backend.requests) == request_count
    with pytest.raises(RuntimeError, match="frozen"):
        await harness.owner.prepare("small", "a" * 64, 2048)
    await harness.owner.close()
    assert harness.closed == harness.adapters
    with pytest.raises(RuntimeError):
        harness.owner.require_prepared()


async def test_stale_selection_never_runs_readiness(monkeypatch):
    harness = Harness(monkeypatch)
    await harness.owner.catalog()
    harness.backend.changed = True
    with pytest.raises(ValueError, match="identity"):
        await harness.prepare()
    assert harness.ready == []
    assert harness.closed == harness.adapters
    assert (await harness.owner.catalog())["state"] == "configuring"


async def test_failed_readiness_closes_candidate_and_can_retry(monkeypatch):
    harness = Harness(monkeypatch)
    harness.fail = True
    with pytest.raises(RuntimeError, match="readiness"):
        await harness.prepare()
    assert harness.closed == harness.adapters
    harness.fail = False
    await harness.prepare()
    assert len(harness.adapters) == 2
    await harness.owner.close()
    assert harness.closed == harness.adapters


async def test_pending_preparation_status_and_shutdown(monkeypatch):
    harness = Harness(monkeypatch)
    harness.block = asyncio.Event()
    preparation = asyncio.create_task(harness.prepare())
    while not harness.ready:
        await asyncio.sleep(0)
    status = await harness.owner.catalog()
    assert status["state"] == "preparing" and status["selection"] is None
    with pytest.raises(RuntimeError, match="preparing"):
        await asyncio.wait_for(harness.prepare(), timeout=0.1)
    await harness.owner.close()
    with pytest.raises(asyncio.CancelledError):
        await preparation
    harness.block.set()
    assert harness.closed == harness.adapters
    with pytest.raises(RuntimeError):
        harness.owner.require_prepared()


@pytest.mark.parametrize(
    "model,digest,context",
    [
        ("other", "a" * 64, 4096),
        ("small", "b" * 64, 4096),
        ("small", "a" * 64, 8192),
    ],
)
async def test_choice_must_match_catalog_and_bounded_capacity(monkeypatch, model, digest, context):
    harness = Harness(monkeypatch)
    with pytest.raises(ValueError):
        await harness.owner.prepare(model, digest, context)
    assert harness.adapters == []


async def test_concurrent_catalog_uses_one_owned_discovery(monkeypatch):
    harness = Harness(monkeypatch)
    first, second = await asyncio.gather(harness.owner.catalog(), harness.owner.catalog())
    assert first == second
    assert len([path for path, _ in harness.backend.requests if path == "tags"]) == 2


@pytest.mark.parametrize("field,value", [
    ("model", "alias"), ("requested", 2048), ("advertised", 8192), ("advertised", None),
])
async def test_configured_choice_must_equal_request(monkeypatch, field, value):
    harness = Harness(monkeypatch)

    async def configure(adapter, requested, *, client):
        adapter._context_observation = OllamaContextObservation(
            OllamaContextConfiguration(
                OllamaModelMetadata(
                    value if field == "model" else "small", "a" * 64,
                    value if field == "advertised" else 4096,
                ),
                value if field == "requested" else requested,
            )
        )

    monkeypatch.setattr(OllamaStreamingInference, "configure_context", configure)
    with pytest.raises(ValueError, match="identity"):
        await harness.prepare()
    assert harness.ready == []
    assert harness.closed == harness.adapters


@pytest.mark.parametrize("prepared", [False, True])
async def test_failed_cleanup_remains_owned_for_retry(monkeypatch, prepared):
    harness = Harness(monkeypatch)
    attempts = []

    async def close(adapter):
        attempts.append(adapter)
        if len(attempts) == 1:
            raise RuntimeError("synthetic close failure")
        harness.closed.append(adapter)

    monkeypatch.setattr(OllamaStreamingInference, "close", close)
    if prepared:
        await harness.prepare()
        with pytest.raises(RuntimeError, match="close failure"):
            await harness.owner.close()
    else:
        harness.fail = True
        with pytest.raises(RuntimeError, match="close failure"):
            await harness.prepare()
    await harness.owner.close()
    assert len(attempts) == 2
    assert harness.closed == harness.adapters


async def test_close_wins_even_if_readiness_suppresses_cancellation(monkeypatch):
    harness = Harness(monkeypatch)
    started = asyncio.Event()

    async def readiness(adapter):
        started.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()

    harness.owner._readiness = readiness
    preparation = asyncio.create_task(harness.prepare())
    await started.wait()
    await harness.owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        await preparation
    assert harness.owner._selection is None
    assert harness.closed == harness.adapters


async def test_preparation_deadline_closes_candidate(monkeypatch, capsys):
    import hermes_realtime.providers.ollama_session as module

    monkeypatch.setattr(module, "PREPARATION_TIMEOUT_SECONDS", 0.02)
    harness = Harness(monkeypatch)

    async def delayed_readiness(adapter):
        await asyncio.sleep(0.1)

    harness.owner._readiness = delayed_readiness
    with pytest.raises(TimeoutError):
        await harness.prepare()
    assert harness.closed == harness.adapters
    assert '"category":"deadline"' in capsys.readouterr().out


async def test_configuration_cannot_change_during_readiness(monkeypatch):
    harness = Harness(monkeypatch)

    async def readiness(adapter):
        adapter._context_observation = OllamaContextObservation(
            OllamaContextConfiguration(
                OllamaModelMetadata("small", "a" * 64, 4096),
                2048,
            )
        )

    harness.owner._readiness = readiness
    with pytest.raises(ValueError, match="during readiness"):
        await harness.prepare()
    assert harness.closed == harness.adapters


async def test_prepared_proxy_routes_stream_and_cancellation_to_selected_adapter(monkeypatch):
    harness = Harness(monkeypatch)
    routed = []

    async def stream(adapter, snapshot, *, turn_id):
        routed.append((adapter, snapshot, turn_id))
        yield "synthetic"

    async def cancel(adapter, turn_id):
        routed.append((adapter, turn_id))

    monkeypatch.setattr(OllamaStreamingInference, "stream", stream)
    monkeypatch.setattr(OllamaStreamingInference, "cancel", cancel)
    await harness.prepare()
    assert [part async for part in harness.owner.stream(None, turn_id="turn")] == ["synthetic"]
    await harness.owner.cancel("turn")
    assert routed == [(harness.adapters[0], None, "turn"), (harness.adapters[0], "turn")]


async def test_adapter_factory_cannot_replace_exact_runtime(monkeypatch):
    harness = Harness(monkeypatch)

    class Impostor:
        async def configure_context(self, *args, **kwargs):
            raise AssertionError("impostor accepted")

    harness.owner._factory = lambda model: Impostor()
    with pytest.raises(TypeError, match="exact"):
        await harness.prepare()


async def test_concurrent_close_does_not_repeat_adapter_cleanup(monkeypatch):
    harness = Harness(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def close(adapter):
        harness.closed.append(adapter)
        started.set()
        await release.wait()

    monkeypatch.setattr(OllamaStreamingInference, "close", close)
    await harness.prepare()
    first = asyncio.create_task(harness.owner.close())
    await started.wait()
    second = asyncio.create_task(harness.owner.close())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)
    assert harness.closed == harness.adapters


@pytest.mark.parametrize(
    "field,value",
    [
        ("client", object()),
        ("default_model", "bad\nmodel"),
        ("adapter_factory", None),
        ("readiness", None),
    ],
)
async def test_constructor_requires_exact_source_and_valid_callbacks(monkeypatch, field, value):
    harness = Harness(monkeypatch)
    arguments = {
        "client": harness.client,
        "default_model": "small",
        "adapter_factory": harness.factory,
        "readiness": harness.readiness,
    }
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        OllamaSessionSelection(**arguments)


async def test_failed_candidate_cleanup_fences_further_allocation(monkeypatch):
    harness = Harness(monkeypatch)
    harness.fail = True

    async def close(adapter):
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(OllamaStreamingInference, "close", close)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        await harness.prepare()
    with pytest.raises(RuntimeError, match="closed"):
        await harness.prepare()
    assert len(harness.adapters) == 1


async def test_cleanup_deadline_retains_task_and_refuses_late_admission(monkeypatch):
    import hermes_realtime.providers.ollama_session as module

    monkeypatch.setattr(module, "CLEANUP_TIMEOUT_SECONDS", 0.02)
    harness = Harness(monkeypatch)
    release = asyncio.Event()
    attempts = []

    async def close(adapter):
        attempts.append(adapter)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(release.wait(), timeout=0.1)
        harness.closed.append(adapter)

    monkeypatch.setattr(OllamaStreamingInference, "close", close)
    await harness.prepare()
    with pytest.raises(TimeoutError):
        await harness.owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        harness.owner.require_prepared()
    release.set()
    await harness.owner.close()
    assert harness.closed == harness.adapters
    assert attempts == harness.adapters


async def test_pending_prepare_join_is_bounded_and_keeps_cleanup_custody(monkeypatch):
    import hermes_realtime.providers.ollama_session as module

    monkeypatch.setattr(module, "CLEANUP_TIMEOUT_SECONDS", 0.02)
    harness = Harness(monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def readiness(adapter):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(release.wait(), timeout=0.1)

    harness.owner._readiness = readiness
    preparation = asyncio.create_task(harness.prepare())
    await entered.wait()
    with pytest.raises(TimeoutError):
        await harness.owner.close()
    assert not preparation.done()
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await preparation
    await harness.owner.close()
    assert harness.closed == harness.adapters


async def test_cancelled_catalog_waiter_keeps_discovery_owned(monkeypatch):
    harness = Harness(monkeypatch)
    harness.backend.block = True
    first = asyncio.create_task(harness.owner.catalog())
    while not harness.backend.active:
        await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await harness.owner.catalog()
    assert len([path for path, _ in harness.backend.requests if path == "tags"]) == 2
    await harness.owner.close()


async def test_old_catalog_finalizer_cannot_clear_new_discovery(monkeypatch):
    harness = Harness(monkeypatch)
    release_old = asyncio.Event()
    release_new = asyncio.Event()
    calls = errors = 0
    original_shield = asyncio.shield

    async def installed():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic first discovery failure")
        await release_new.wait()
        return (OllamaModelMetadata("small", "a" * 64, 4096),)

    async def shield(operation):
        nonlocal errors
        try:
            return await original_shield(operation)
        except ValueError:
            errors += 1
            if errors == 2:
                await release_old.wait()
            raise

    monkeypatch.setattr(harness.client, "installed_models", installed)
    monkeypatch.setattr(asyncio, "shield", shield)
    first = asyncio.create_task(harness.owner.catalog())
    second = asyncio.create_task(harness.owner.catalog())
    with pytest.raises(ValueError):
        await first
    third = asyncio.create_task(harness.owner.catalog())
    while calls < 2:
        await asyncio.sleep(0)
    release_old.set()
    with pytest.raises(ValueError):
        await second
    fourth = asyncio.create_task(harness.owner.catalog())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls == 2
    release_new.set()
    await asyncio.gather(third, fourth)


async def test_cancelled_adapter_cleanup_can_be_retried(monkeypatch):
    harness = Harness(monkeypatch)
    attempts = []

    async def close(adapter):
        attempts.append(adapter)
        if len(attempts) == 1:
            raise asyncio.CancelledError()
        harness.closed.append(adapter)

    monkeypatch.setattr(OllamaStreamingInference, "close", close)
    await harness.prepare()
    with pytest.raises(asyncio.CancelledError):
        await harness.owner.close()
    await harness.owner.close()
    assert harness.closed == harness.adapters
