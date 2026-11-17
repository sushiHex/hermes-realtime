"""MoonshineStreamingTranscriber.drain_partials: the endpoint partial capability (#576)."""

from __future__ import annotations

import asyncio

import pytest

from hermes_realtime.providers.moonshine import MoonshineStreamingTranscriber
from hermes_realtime.speech import AudioFrame, Transcript
from tests.providers.test_moonshine import ScriptedMoonshineBackend


def _frame() -> AudioFrame:
    return AudioFrame(pcm=b"\x01\x00" * 480, sample_rate_hz=48_000, channels=1)


@pytest.mark.asyncio
async def test_drain_returns_the_burst_partials_as_exact_non_finals_then_finish_finalizes() -> None:
    backend = ScriptedMoonshineBackend("Stop", final="Stop.")
    transcriber = MoonshineStreamingTranscriber(backend=backend)

    # A burst: the second push runs before the first push's decode can start.
    assert await transcriber.push(_frame()) == ()
    assert await transcriber.push(_frame()) == ()

    drained = await transcriber.drain_partials()

    assert type(drained) is tuple
    assert drained == (Transcript(text="Stop", final=False),)
    assert all(type(partial) is Transcript for partial in drained)
    assert backend.finish_calls == 0
    assert await transcriber.drain_partials() == ()
    assert await transcriber.finish_utterance() == Transcript(text="Stop.", final=True)
    assert backend.finish_calls == 1
    await transcriber.cancel()


@pytest.mark.asyncio
async def test_push_is_refused_while_a_drain_runs() -> None:
    backend = ScriptedMoonshineBackend("Stop")
    backend.release.clear()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    assert await transcriber.push(_frame()) == ()
    drain = asyncio.create_task(transcriber.drain_partials())
    assert await asyncio.to_thread(backend.added.wait, 1)

    with pytest.raises(RuntimeError, match="settling"):
        await transcriber.push(_frame())

    backend.release.set()
    assert await drain == (Transcript(text="Stop", final=False),)
    # The drain released the utterance: pushes are accepted again.
    assert await transcriber.push(_frame()) == ()
    await transcriber.cancel()


@pytest.mark.asyncio
async def test_drain_is_refused_once_closed() -> None:
    transcriber = MoonshineStreamingTranscriber(backend=ScriptedMoonshineBackend())
    await transcriber.close()

    with pytest.raises(RuntimeError, match="closed"):
        await transcriber.drain_partials()


@pytest.mark.asyncio
async def test_cancelled_drain_leaves_its_decode_in_custody_and_cancel_settles_it() -> None:
    backend = ScriptedMoonshineBackend("stale words", "fresh words")
    backend.release.clear()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    assert await transcriber.push(_frame()) == ()
    drain = asyncio.create_task(transcriber.drain_partials())
    assert await asyncio.to_thread(backend.added.wait, 1)

    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain
    # The decode still runs in the backend; it stays owned until cancel settles it.
    assert transcriber._operation is not None

    backend.release.set()
    await transcriber.cancel()
    assert transcriber._operation is None
    assert backend.cancel_calls == 1

    # The cancelled utterance's partial never reaches the next one.
    backend.completed.clear()
    assert await transcriber.push(_frame()) == ()
    assert await transcriber.drain_partials() == (Transcript(text="fresh words", final=False),)
    await transcriber.cancel()
