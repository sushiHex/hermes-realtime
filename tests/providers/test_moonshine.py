from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from hermes_realtime.providers import MoonshineStreamingTranscriber
from hermes_realtime.speech import AudioFrame, Transcript


def _frame(samples: int = 480) -> AudioFrame:
    return AudioFrame(
        pcm=b"\x00\x01" * samples,
        sample_rate_hz=48_000,
        channels=1,
    )


class ScriptedMoonshineBackend:
    def __init__(self, *partials: str | None, final: str = "Final words") -> None:
        self._partials = iter(partials)
        self._final = final
        self.added = threading.Event()
        self.completed = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.add_thread_ids: list[int] = []
        self.add_sizes: list[int] = []
        self.cancel_calls = 0
        self.close_calls = 0
        self.finish_calls = 0

    def add_pcm(self, pcm: bytes, sample_rate_hz: int, channels: int) -> str | None:
        assert pcm
        assert sample_rate_hz == 48_000
        assert channels == 1
        self.add_thread_ids.append(threading.get_ident())
        self.add_sizes.append(len(pcm))
        self.added.set()
        assert self.release.wait(timeout=2)
        result = next(self._partials, None)
        self.completed.set()
        return result

    def finish(self) -> str:
        self.finish_calls += 1
        return self._final

    def cancel(self) -> None:
        self.cancel_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("model_tier", "expected_architecture"),
    (
        ("tiny", "tiny-arch"),
        ("small", "small-arch"),
        ("medium", "medium-arch"),
    ),
)
@pytest.mark.asyncio
async def test_moonshine_selects_explicit_streaming_model_tier(
    monkeypatch: pytest.MonkeyPatch,
    model_tier: str,
    expected_architecture: str,
) -> None:
    selections: list[tuple[str, str]] = []
    transcribers: list[dict[str, object]] = []
    fake_moonshine = SimpleNamespace(
        ModelArch=SimpleNamespace(
            TINY_STREAMING="tiny-arch",
            SMALL_STREAMING="small-arch",
            MEDIUM_STREAMING="medium-arch",
        ),
        get_model_for_language=lambda wanted_language, wanted_model_arch: (
            selections.append((wanted_language, wanted_model_arch))
            or (f"model/{wanted_model_arch}", wanted_model_arch)
        ),
        Transcriber=lambda **kwargs: transcribers.append(kwargs) or ScriptedMoonshineBackend(),
    )
    monkeypatch.setitem(sys.modules, "moonshine_voice", fake_moonshine)

    transcriber = MoonshineStreamingTranscriber(model_tier=model_tier)
    await transcriber.close()

    assert selections == [("en", expected_architecture)]
    assert transcribers == [
        {
            "model_path": f"model/{expected_architecture}",
            "model_arch": expected_architecture,
            "update_interval": 0.2,
        }
    ]


def test_moonshine_requires_exact_streaming_backend_contract() -> None:
    with pytest.raises(TypeError, match="backend"):
        MoonshineStreamingTranscriber(backend=object())


@pytest.mark.parametrize(
    ("model_tier", "error"),
    (("small-streaming", ValueError), (None, TypeError)),
)
def test_moonshine_rejects_unknown_or_inexact_model_tier(
    model_tier: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="model_tier"):
        MoonshineStreamingTranscriber(
            model_tier=model_tier,  # type: ignore[arg-type]
            backend=ScriptedMoonshineBackend(),
        )


@pytest.mark.asyncio
async def test_moonshine_streams_changed_partial_off_loop_then_finalizes_once() -> None:
    backend = ScriptedMoonshineBackend("Can you give", "Can you give me")
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    loop_thread_id = threading.get_ident()

    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)
    partials = await transcriber.push(_frame())

    assert partials == (Transcript(text="Can you give", final=False),)
    assert backend.add_thread_ids[0] != loop_thread_id

    final = await transcriber.finish_utterance()

    assert final == Transcript(text="Final words", final=True)
    assert backend.finish_calls == 1
    await transcriber.cancel()


@pytest.mark.asyncio
async def test_moonshine_promotes_last_partial_when_native_final_is_empty() -> None:
    backend = ScriptedMoonshineBackend("Visible words", final="")
    transcriber = MoonshineStreamingTranscriber(backend=backend)

    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)
    assert await transcriber.push(_frame()) == (
        Transcript(text="Visible words", final=False),
    )

    assert await transcriber.finish_utterance() == Transcript(
        text="Visible words",
        final=True,
    )
    assert backend.finish_calls == 1
    await transcriber.cancel()


@pytest.mark.asyncio
async def test_moonshine_drops_last_partial_when_native_final_is_invalid() -> None:
    backend = ScriptedMoonshineBackend(
        "Do not leak me",
        final="x" * 4097,
    )
    transcriber = MoonshineStreamingTranscriber(backend=backend)

    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)
    assert await transcriber.push(_frame()) == (
        Transcript(text="Do not leak me", final=False),
    )
    with pytest.raises(ValueError, match="exceeds supported size"):
        await transcriber.finish_utterance()

    backend._final = ""
    assert await transcriber.push(_frame()) == ()
    assert await transcriber.finish_utterance() is None
    await transcriber.cancel()


@pytest.mark.asyncio
async def test_moonshine_deduplicates_partial_revisions() -> None:
    backend = ScriptedMoonshineBackend("same words", "same words")
    transcriber = MoonshineStreamingTranscriber(backend=backend)

    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)
    assert await transcriber.push(_frame()) == (
        Transcript(text="same words", final=False),
    )
    backend.completed.clear()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)
    assert await transcriber.push(_frame()) == ()

    await transcriber.cancel()


@pytest.mark.asyncio
async def test_moonshine_bounds_pending_pcm_while_native_inference_is_busy() -> None:
    backend = ScriptedMoonshineBackend(None)
    backend.release.clear()
    transcriber = MoonshineStreamingTranscriber(
        backend=backend,
        max_pending_bytes=len(_frame().pcm),
    )

    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.added.wait, 1)
    assert await transcriber.push(_frame()) == ()
    with pytest.raises(RuntimeError, match="capacity"):
        await transcriber.push(_frame())

    backend.release.set()
    await transcriber.cancel()


@pytest.mark.asyncio
async def test_moonshine_cancel_discards_stale_partial_and_recovers() -> None:
    backend = ScriptedMoonshineBackend("stale words", "fresh words")
    backend.release.clear()
    transcriber = MoonshineStreamingTranscriber(backend=backend)

    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.added.wait, 1)
    cancelled = asyncio.create_task(transcriber.cancel())
    await asyncio.sleep(0)
    backend.release.set()
    await cancelled

    backend.completed.clear()
    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)
    assert await transcriber.push(_frame()) == (
        Transcript(text="fresh words", final=False),
    )
    assert backend.cancel_calls == 1

    await transcriber.cancel()


@pytest.mark.asyncio
async def test_moonshine_cancel_waits_for_cancelled_native_finish_owner() -> None:
    class BlockingFinishBackend(ScriptedMoonshineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.finish_started = threading.Event()
            self.finish_released = threading.Event()
            self.finish_done = threading.Event()
            self.cancel_overlapped_finish = False

        def finish(self) -> str:
            self.finish_calls += 1
            self.finish_started.set()
            assert self.finish_released.wait(timeout=2)
            self.finish_done.set()
            return self._final

        def cancel(self) -> None:
            if not self.finish_done.is_set():
                self.cancel_overlapped_finish = True
            super().cancel()

    backend = BlockingFinishBackend()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    finishing = asyncio.create_task(transcriber.finish_utterance())
    assert await asyncio.to_thread(backend.finish_started.wait, 1)

    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing

    cleanup = asyncio.create_task(transcriber.cancel())
    await asyncio.sleep(0.05)
    assert not cleanup.done()
    assert backend.cancel_calls == 0

    backend.finish_released.set()
    await asyncio.wait_for(cleanup, timeout=1)

    assert backend.cancel_calls == 1
    assert not backend.cancel_overlapped_finish


@pytest.mark.asyncio
async def test_moonshine_caller_cancellation_cannot_orphan_native_cancel() -> None:
    class BlockingCancelBackend(ScriptedMoonshineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_started = threading.Event()
            self.cancel_released = threading.Event()

        def cancel(self) -> None:
            self.cancel_started.set()
            assert self.cancel_released.wait(timeout=2)
            super().cancel()

    backend = BlockingCancelBackend()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    cleanup = asyncio.create_task(transcriber.cancel())
    assert await asyncio.to_thread(backend.cancel_started.wait, 1)

    cleanup.cancel()
    await asyncio.sleep(0.05)
    assert not cleanup.done()
    with pytest.raises(RuntimeError, match="settling"):
        await transcriber.push(_frame())

    backend.cancel_released.set()
    with pytest.raises(BaseExceptionGroup, match="cancellation failed"):
        await asyncio.wait_for(cleanup, timeout=1)

    assert backend.cancel_calls == 1
    assert await transcriber.push(_frame()) == ()
    await transcriber.close()


@pytest.mark.asyncio
async def test_moonshine_cancel_closes_backend_after_native_add_failure() -> None:
    class FailingBackend(ScriptedMoonshineBackend):
        def add_pcm(self, pcm: bytes, sample_rate_hz: int, channels: int) -> str | None:
            del pcm, sample_rate_hz, channels
            self.completed.set()
            raise RuntimeError("native add failed")

    backend = FailingBackend()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    assert await transcriber.push(_frame()) == ()
    assert await asyncio.to_thread(backend.completed.wait, 1)
    await asyncio.sleep(0.01)

    with pytest.raises(BaseExceptionGroup, match="cancellation failed"):
        await transcriber.cancel()

    assert backend.cancel_calls == 1


@pytest.mark.asyncio
async def test_moonshine_drains_backlog_in_bounded_native_chunks() -> None:
    backend = ScriptedMoonshineBackend(None, None, None)
    backend.release.clear()
    frame = _frame()
    transcriber = MoonshineStreamingTranscriber(
        backend=backend,
        max_pending_bytes=len(frame.pcm) * 3,
        max_add_bytes=len(frame.pcm),
    )

    assert await transcriber.push(frame) == ()
    assert await asyncio.to_thread(backend.added.wait, 1)
    assert await transcriber.push(frame) == ()
    assert await transcriber.push(frame) == ()
    backend.release.set()

    assert await transcriber.finish_utterance() == Transcript(
        text="Final words",
        final=True,
    )
    assert len(backend.add_sizes) == 3
    assert max(backend.add_sizes) <= len(frame.pcm)
    await transcriber.close()


@pytest.mark.asyncio
async def test_moonshine_close_releases_native_model_once_and_rejects_reuse() -> None:
    backend = ScriptedMoonshineBackend()
    transcriber = MoonshineStreamingTranscriber(backend=backend)

    await transcriber.close()
    await transcriber.close()

    assert backend.cancel_calls == 1
    assert backend.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await transcriber.push(_frame())


@pytest.mark.asyncio
async def test_moonshine_cancelled_close_waits_for_inflight_finish_drain() -> None:
    class BlockingFinishBackend(ScriptedMoonshineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.finish_started = threading.Event()
            self.finish_released = threading.Event()
            self.close_started = threading.Event()
            self.close_overlapped_finish = False

        def finish(self) -> str:
            self.finish_started.set()
            assert self.finish_released.wait(timeout=2)
            return "final"

        def close(self) -> None:
            self.close_overlapped_finish = not self.finish_released.is_set()
            self.close_started.set()
            super().close()

    backend = BlockingFinishBackend()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    finishing = asyncio.create_task(transcriber.finish_utterance())
    assert await asyncio.to_thread(backend.finish_started.wait, 1)

    closing = asyncio.create_task(transcriber.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    await asyncio.sleep(0.05)
    close_started_before_finish_release = backend.close_started.is_set()
    close_finished_before_finish_release = closing.done()

    backend.finish_released.set()
    final = await asyncio.wait_for(finishing, timeout=1)
    assert final is not None
    assert final.text == "final"
    with pytest.raises(BaseExceptionGroup, match="close failed"):
        await asyncio.wait_for(closing, timeout=1)

    assert not close_started_before_finish_release
    assert not close_finished_before_finish_release
    assert not backend.close_overlapped_finish


@pytest.mark.asyncio
async def test_moonshine_caller_cancellation_cannot_orphan_native_close() -> None:
    class BlockingCloseBackend(ScriptedMoonshineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = threading.Event()
            self.close_released = threading.Event()

        def close(self) -> None:
            self.close_started.set()
            assert self.close_released.wait(timeout=2)
            super().close()

    backend = BlockingCloseBackend()
    transcriber = MoonshineStreamingTranscriber(backend=backend)
    closing = asyncio.create_task(transcriber.close())
    assert await asyncio.to_thread(backend.close_started.wait, 1)
    await transcriber.cancel()
    assert backend.cancel_calls == 1

    closing.cancel()
    await asyncio.sleep(0.05)
    assert not closing.done()
    with pytest.raises(RuntimeError, match="settling"):
        await transcriber.push(_frame())

    backend.close_released.set()
    with pytest.raises(BaseExceptionGroup, match="close failed"):
        await asyncio.wait_for(closing, timeout=1)

    await transcriber.close()
    assert backend.close_calls == 1
