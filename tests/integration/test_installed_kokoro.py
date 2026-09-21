"""Opt-in smoke test for the installed CPU and CUDA Kokoro backends."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_realtime.providers.kokoro import (
    _MODEL_ASSET,
    _VOICES_ASSET,
    KokoroSynthesizer,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_INSTALLED_KOKORO") != "1",
        reason="set HERMES_REALTIME_INSTALLED_KOKORO=1 for the installed Kokoro smoke test",
    ),
]

_TEXT = "The synthetic Kokoro smoke test produced this short sentence."
_TRANSPORT_SAMPLE_RATE_HZ = 48_000
_TRANSPORT_CHANNELS = 1
_TRANSPORT_FRAME_BYTES = (_TRANSPORT_SAMPLE_RATE_HZ // 100) * 2


def _asset_cache() -> Path:
    configured = os.environ.get("HERMES_REALTIME_KOKORO_ASSET_CACHE")
    assert configured, "HERMES_REALTIME_KOKORO_ASSET_CACHE must name the provisioned cache"
    cache = Path(configured)
    assert cache.is_absolute(), "HERMES_REALTIME_KOKORO_ASSET_CACHE must be absolute"
    assert cache.is_dir(), "HERMES_REALTIME_KOKORO_ASSET_CACHE must be an existing directory"
    for asset in (_MODEL_ASSET, _VOICES_ASSET):
        path = cache / asset.filename
        assert path.is_file(), f"provisioned Kokoro asset is missing: {asset.filename}"
        KokoroSynthesizer._verify_asset(path, asset)
    return cache


def _cuda_python() -> Path:
    configured = os.environ.get("HERMES_REALTIME_KOKORO_CUDA_PYTHON")
    assert configured, "HERMES_REALTIME_KOKORO_CUDA_PYTHON must name the worker interpreter"
    python = Path(configured)
    assert python.is_absolute(), "HERMES_REALTIME_KOKORO_CUDA_PYTHON must be absolute"
    assert python.is_file(), "HERMES_REALTIME_KOKORO_CUDA_PYTHON must be an existing file"
    return python


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
async def test_installed_kokoro_synthesizes_transport_pcm_and_closes(backend: str) -> None:
    assert sys.platform == "win32", "the installed Kokoro smoke test requires Windows"
    synthesizer = KokoroSynthesizer(
        cache_dir=_asset_cache(),
        worker_python=_cuda_python() if backend == "cuda" else None,
    )
    owned_process: subprocess.Popen[bytes] | None = None
    try:
        await synthesizer.warm()
        if backend == "cuda":
            worker = synthesizer._worker_client
            assert worker is not None
            owned_process = worker._process
            assert owned_process is not None
            assert worker.actual_providers == (
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            assert owned_process.poll() is None
        else:
            assert synthesizer._worker_client is None

        chunks = [
            chunk
            async for chunk in synthesizer.synthesize(
                _TEXT,
                turn_id=f"installed_kokoro_{backend}",
            )
        ]

        assert chunks
        for chunk in chunks:
            audio = chunk.audio
            assert (
                audio.sample_rate_hz,
                audio.channels,
                len(audio.pcm) % _TRANSPORT_FRAME_BYTES,
            ) == (_TRANSPORT_SAMPLE_RATE_HZ, _TRANSPORT_CHANNELS, 0)
            assert audio.pcm
            samples = struct.unpack(f"<{len(audio.pcm) // 2}h", audio.pcm)
            assert any(sample != 0 for sample in samples)
    finally:
        try:
            await synthesizer.close()
            assert synthesizer._engine is None
            if owned_process is not None:
                assert owned_process.poll() is not None
        finally:
            if owned_process is not None and owned_process.poll() is None:
                owned_process.kill()
                owned_process.wait(timeout=2.0)
