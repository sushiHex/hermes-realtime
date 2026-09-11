"""Exercise retained close owners after real paired typed and PCM conversations."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _require
from scripts.equivalence_worker import _observation
from scripts.owned_close_faults import ARMS_V1


class _CloseFault:
    """Fault only the deterministic provider callback, preserving its real caller."""

    def __init__(self, arm: str) -> None:
        self.retry = arm.startswith("retry")
        self.cancelled = arm.startswith("cancelled")
        self.events: list[str] = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls = 0

    async def close(self) -> None:
        self.calls += 1
        self.events.append("provider_entered")
        self.entered.set()
        if self.retry and self.calls == 1:
            self.events.append("provider_failed")
            raise RuntimeError("qualification provider close fault")
        if self.cancelled:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.events.append("provider_cancelled")
                raise
        self.events.append("provider_returned")

    async def drive(self, composition: Any, running: Any) -> None:
        tasks: list[asyncio.Task[None]] = []
        try:
            if self.retry:
                try:
                    await composition.close_host(running)
                except RuntimeError as error:
                    _require(
                        str(error) == "qualification provider close fault", "wrong close fault"
                    )
                    self.events.append("caller_failed")
                else:
                    raise ValueError("provider close fault was not observed")
                previous = running._host._close_operation
                await composition.retry_owned_close(running)
                _require(
                    running._host._close_operation is not previous, "failed owner was not retried"
                )
                self.events.append("retry_returned")
            elif self.cancelled:
                first = asyncio.create_task(composition.close_host(running))
                tasks.append(first)
                entered = asyncio.create_task(self.entered.wait())
                try:
                    await asyncio.wait((first, entered), return_when=asyncio.FIRST_COMPLETED)
                    if first.done():
                        await first
                        raise ValueError("close returned before the held provider callback")
                finally:
                    entered.cancel()
                    await asyncio.gather(entered, return_exceptions=True)
                owner = running._host._close_operation
                join = asyncio.create_task(composition.retry_owned_close(running))
                tasks.append(join)
                await asyncio.sleep(0)
                _require(
                    owner is running._host._close_operation
                    and not owner.done() and not join.done(),
                    "second caller did not join the pending owner",
                )
                first.cancel()
                try:
                    await first
                except asyncio.CancelledError:
                    self.events.append("caller_cancelled")
                else:
                    raise ValueError("first close caller was not cancelled")
                _require(not owner.done() and not join.done(), "caller cancellation stopped close")
                self.events.append("provider_released")
                self.release.set()
                await join
                _require(owner is running._host._close_operation, "join replaced the owned close")
                self.events.append("join_returned")
            else:
                await composition.close_host(running)
                self.events.append("caller_returned")
            host = running._host
            _require(
                host._closed and host._runtime_closed and all(host._provider_closed)
                and host._close_operation.done() and not host._close_operation.cancelled()
                and host._close_operation.exception() is None
                and all(task.done() for task in tasks)
                and self.calls == (2 if self.retry else 1),
                "owned close retained unfinished stages or callers",
            )
        finally:
            self.release.set()
            # Even a failed observation must release its fault gate and join cleanup.
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if not running._host._closed:
                await composition.retry_owned_close(running)


async def _run_arm(name: str, workspace: Path, livekit_url: str, key: bytes) -> dict[str, Any]:
    from tests.integration.test_qualification_full_host_ingress import (
        _run_non_mutation_arm,
        _Synthesizer,
    )

    fault = _CloseFault(name)
    observed: list[dict[str, Any]] = []

    def synthesizer_factory() -> Any:
        synthesizer = _Synthesizer()
        synthesizer.close = fault.close
        return synthesizer

    def observe(
        raw: tuple[object, ...], metadata: tuple[object, ...], complete: bool,
    ) -> None:
        observed.append(_observation(name, raw, metadata, key, complete))

    await _run_non_mutation_arm(
        tmp_path=workspace,
        capture=name.endswith("consented"), consent=name.endswith("consented"),
        writer_fault=False, livekit_url=livekit_url,
        typed_stimulus="deliberately different typed stimulus"
        if name == "perturbed" else "paired typed stimulus",
        synthesizer_factory=synthesizer_factory, close_driver=fault.drive, observe=observe,
    )
    _require(len(observed) == 1, "close observation is disconnected or duplicated")
    observed[0]["owner_events"] = fault.events
    return observed[0]


async def observe_owned_close_faults(workspace: Path, livekit_url: str) -> dict[str, Any]:
    key = os.urandom(32)
    cases = [await _run_arm(name, workspace, livekit_url, key) for name in ARMS_V1]
    return {"arm": "owned_close_faults", "cases": cases}
