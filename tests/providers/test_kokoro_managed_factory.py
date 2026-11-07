"""Managed worker seams with fake factories/processes only; no model imports."""
from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_realtime.providers import kokoro as kokoro_module
from hermes_realtime.providers import kokoro_worker as worker_module


@pytest.fixture
def managed(tmp_path, monkeypatch):
    events = []
    class Worker:
        warm_error = None
        close_error = None
        def warm(self):
            events.append("worker.warm")
            if self.warm_error:
                raise self.warm_error
        def close(self):
            events.append("worker.close")
            if self.close_error:
                raise self.close_error
        def synthesize_pcm(self, text, voice, speed):
            events.append("worker.synthesize")
            return b"\0\0" * 2400
    instance = Worker()
    class Factory:
        create_error = None
        close_error = None
        def create(self, **kwargs):
            events.append("factory.create")
            self.arguments = kwargs
            if self.create_error:
                raise self.create_error
            return instance
        def close(self):
            events.append("factory.close")
            if self.close_error:
                raise self.close_error
    factory = Factory()
    def asset(self, specification):
        path = tmp_path / specification.filename
        path.write_bytes(b"synthetic fixture, never loaded")
        return path
    monkeypatch.setattr(kokoro_module.KokoroSynthesizer, "_ensure_asset", asset)
    monkeypatch.setattr(kokoro_module.KokoroSynthesizer, "_upsample_pcm_for_transport",
                        staticmethod(lambda pcm: pcm))
    monkeypatch.setattr(kokoro_module, "importlib", SimpleNamespace(
        import_module=lambda name: pytest.fail("managed backend must not import model providers"),
    ))
    return factory, instance, events


@pytest.mark.asyncio
async def test_factory_owns_worker_warm_synthesis_and_ordered_close(managed):
    factory, worker, events = managed
    synth = kokoro_module.KokoroSynthesizer(worker_client_factory=factory)
    await synth.warm()
    assert events == ["factory.create", "worker.warm"]
    assert set(factory.arguments) == {
        "model_path", "voices_path", "expected_model_sha256", "expected_voices_sha256",
        "language", "max_pcm_bytes", "attestation_voice",
    }
    chunk = await anext(synth.synthesize("Synthetic speech.", "synthetic-turn"))
    assert chunk.audio.pcm
    await synth.close()
    assert events[-2:] == ["worker.close", "factory.close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "warm"])
async def test_factory_refusal_closes_owned_factory_without_cpu_fallback(managed, phase):
    factory, worker, events = managed
    if phase == "create":
        factory.create_error = RuntimeError("synthetic managed refusal")
    else:
        worker.warm_error = RuntimeError("synthetic managed refusal")
    synth = kokoro_module.KokoroSynthesizer(worker_client_factory=factory)
    with pytest.raises(RuntimeError, match="synthetic managed refusal"):
        await synth.warm()
    assert events[-1] == "factory.close"
    if phase == "warm":
        assert events[-2] == "worker.close"
    assert synth._engine is None
    await synth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["worker", "factory"])
async def test_factory_close_is_attempted_and_executor_closed_on_cleanup_error(
    managed, failure, capsys,
):
    factory, worker, events = managed
    synth = kokoro_module.KokoroSynthesizer(worker_client_factory=factory)
    await synth.warm()
    if failure == "worker":
        worker.close_error = RuntimeError("synthetic cleanup unknown")
    else:
        factory.close_error = RuntimeError("synthetic cleanup unknown")
    with pytest.raises(RuntimeError, match="synthetic cleanup unknown") as caught:
        await synth.close()
    with pytest.raises(RuntimeError, match="synthetic cleanup unknown") as repeated:
        await synth.close()
    assert repeated.value is caught.value
    assert events[-2:] == ["worker.close", "factory.close"]
    with pytest.raises(RuntimeError, match="shutdown"):
        synth._executor.submit(lambda: None)
    output = capsys.readouterr()
    assert output.out == "" and output.err.count("\n") == 2
    assert "synthetic cleanup unknown" not in output.err
    assert all('"category":"cleanup_unknown"' in line for line in output.err.splitlines())


@pytest.mark.parametrize("other", ["worker_python", "synthesize_pcm"])
def test_factory_cannot_be_combined_with_other_backend(managed, other, capsys):
    factory, worker, events = managed
    value = Path(sys.executable) if other == "worker_python" else lambda *args: b""
    with pytest.raises(ValueError, match="cannot be combined"):
        kokoro_module.KokoroSynthesizer(worker_client_factory=factory, **{other: value})
    assert events == []
    assert "schema_rejected" in capsys.readouterr().err


def test_factory_requires_both_owned_create_and_close(capsys):
    with pytest.raises(TypeError, match="create and close"):
        kokoro_module.KokoroSynthesizer(worker_client_factory=SimpleNamespace(create=lambda: None))
    assert "schema_rejected" in capsys.readouterr().err


@pytest.fixture
def client_paths(tmp_path):
    paths = [tmp_path / name for name in ("worker.py", "model", "voices")]
    for path in paths:
        path.write_bytes(b"synthetic")
    return {"python_executable": Path(sys.executable), "worker_script": paths[0],
            "model_path": paths[1], "voices_path": paths[2],
            "expected_model_sha256": hashlib.sha256(b"synthetic").hexdigest(),
            "expected_voices_sha256": hashlib.sha256(b"synthetic").hexdigest(),
            "language": "en-gb", "max_pcm_bytes": 1024}


def test_launcher_receives_only_port_and_token_without_direct_popen(client_paths, monkeypatch):
    calls = []
    class Launcher:
        def launch(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr(worker_module.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("direct fallback"))
    monkeypatch.setattr(worker_module.secrets, "token_urlsafe", lambda size: "synthetic-token")
    client = worker_module.KokoroWorkerClient(**client_paths, worker_launcher=Launcher())
    with pytest.raises(RuntimeError, match="startup with code 1"):
        client.warm()
    assert len(calls) == 1 and set(calls[0]) == {"connect_port", "token"}
    assert calls[0]["token"] == "synthetic-token" and type(calls[0]["connect_port"]) is int
    assert client.actual_providers == () and client._socket is None
    client.close()


def test_launcher_refusal_never_falls_back_to_direct_process(client_paths, monkeypatch):
    class Launcher:
        def launch(self, **kwargs):
            raise RuntimeError("synthetic launch refused")
    monkeypatch.setattr(worker_module.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("direct fallback"))
    client = worker_module.KokoroWorkerClient(**client_paths, worker_launcher=Launcher())
    with pytest.raises(RuntimeError, match="synthetic launch refused"):
        client.warm()
    assert client._process is None and client.actual_providers == ()
    client.close()


def test_launcher_must_expose_narrow_launch(client_paths, capsys):
    with pytest.raises(TypeError, match="worker_launcher must expose launch"):
        worker_module.KokoroWorkerClient(**client_paths, worker_launcher=object())
    assert "schema_rejected" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "warm", "synthesis"])
async def test_managed_close_interrupts_same_factory_before_executor_wait(managed, phase):
    factory, worker, events = managed
    started, stopped = threading.Event(), threading.Event()
    def interrupted():
        started.set()
        if not stopped.wait(2):
            raise RuntimeError("synthetic backend was not interrupted")
        raise RuntimeError("synthetic backend closed")
    def close_worker():
        events.append("worker.close")
        stopped.set()
    def close_factory():
        events.append("factory.close")
        stopped.set()
    worker.close = close_worker
    factory.close = close_factory
    if phase == "create":
        factory.create = lambda **kwargs: interrupted()
    elif phase == "warm":
        worker.warm = interrupted
    else:
        worker.synthesize_pcm = lambda *args: interrupted()
    synth = kokoro_module.KokoroSynthesizer(worker_client_factory=factory)
    if phase == "synthesis":
        await synth.warm()
        operation = asyncio.create_task(anext(synth.synthesize("Synthetic.", "synthetic-turn")))
    else:
        operation = asyncio.create_task(synth.warm())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.wait_for(synth.close(), timeout=1)
        if phase == "synthesis":
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(RuntimeError, match="synthetic backend closed"):
                await operation
        assert "factory.close" in events
        assert synth._engine is None
    finally:
        stopped.set()
        await synth.close()


@pytest.mark.asyncio
async def test_cancelled_close_drain_still_settles_managed_backend_and_retains_cancel(managed):
    factory, worker, events = managed
    synth = kokoro_module.KokoroSynthesizer(worker_client_factory=factory)
    await synth.warm()
    operation_started, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def operation():
        try:
            operation_started.set()
            await asyncio.Event().wait()
        finally:
            draining.set()
            await release.wait()
    owned = asyncio.create_task(operation())
    await operation_started.wait()
    synth._operations["synthetic-turn"] = owned
    closing = asyncio.create_task(synth.close())
    await asyncio.wait_for(draining.wait(), timeout=1)
    closing.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await closing
    await asyncio.gather(owned, return_exceptions=True)
    assert events[-2:] == ["worker.close", "factory.close"]
    assert synth._managed_close_error is caught.value
    with pytest.raises(asyncio.CancelledError) as repeated:
        await synth.close()
    assert repeated.value is caught.value
    with pytest.raises(RuntimeError, match="shutdown"):
        synth._executor.submit(lambda: None)


@pytest.mark.asyncio
async def test_concurrent_managed_close_refuses_pending_without_changing_owner_error(managed):
    factory, worker, events = managed
    synth = kokoro_module.KokoroSynthesizer(worker_client_factory=factory)
    await synth.warm()
    started, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def operation():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            draining.set()
            await release.wait()
    owned = asyncio.create_task(operation())
    await started.wait()
    synth._operations["synthetic-turn"] = owned
    closing = asyncio.create_task(synth.close())
    try:
        await asyncio.wait_for(draining.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="cleanup_pending"):
            await synth.close()
        assert not closing.done() and synth._managed_close_error is None
    finally:
        release.set()
        await closing
        await asyncio.gather(owned, return_exceptions=True)
    assert not synth._managed_close_pending and synth._managed_close_error is None
    await synth.close()
    assert events[-2:] == ["worker.close", "factory.close"]
