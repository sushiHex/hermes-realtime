"""Focused REDs for the opaque full-host qualification dependency capability."""

from __future__ import annotations

import inspect
from typing import Any

import pytest


def test_public_host_builder_has_no_qualification_dependency_or_mode_parameter() -> None:
    from hermes_realtime.host_launcher import build_local_host_launcher

    parameters = inspect.signature(build_local_host_launcher).parameters
    assert "qualification_no_hermes_tasks" not in parameters
    assert "qualification_dependencies" not in parameters
    assert "test_dependencies" not in parameters


@pytest.mark.asyncio
async def test_owner_bound_bundle_is_consumed_by_real_host_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime import host_launcher
    from hermes_realtime._qualification import (
        _new_qualification_full_host_dependencies,
        _new_qualification_owner_context_capability,
    )

    built: list[str] = []

    class Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    def factory(name: str) -> Any:
        def build() -> Provider:
            built.append(name)
            return Provider(name)

        return build

    def forbidden(**_kwargs: object) -> object:
        raise AssertionError("qualification composition loaded an ordinary provider")

    monkeypatch.setattr(host_launcher, "_build_streaming_inference", forbidden)
    monkeypatch.setattr(host_launcher, "_build_streaming_transcriber", forbidden)
    monkeypatch.setattr(host_launcher, "_build_synthesizer", forbidden)

    identity = object()
    collector = _Collector(identity)
    capability = _new_qualification_owner_context_capability(identity)
    bundle = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=factory("inference"),
        speech_presence_factory=factory("speech_presence"),
        transcriber_factory=factory("transcriber"),
        synthesizer_factory=factory("synthesizer"),
        vad_factory=factory("vad"),
        identity_factory=lambda: "qualification-identity",
    )
    with capability.wire(collector, full_host_dependencies=bundle):
        launcher = host_launcher.build_local_host_launcher(hermes_api_bearer=None)
    try:
        assert built == ["inference", "synthesizer", "transcriber", "speech_presence"]
    finally:
        await launcher.close()


def test_full_host_dependency_bundle_is_opaque_context_bound_and_one_shot() -> None:
    from hermes_realtime._qualification import (
        _current_qualification_full_host_dependencies,
        _new_qualification_full_host_dependencies,
        _new_qualification_owner_context_capability,
    )

    identity = object()
    collector = _Collector(identity)
    capability = _new_qualification_owner_context_capability(identity)
    bundle = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=lambda: object(),
        speech_presence_factory=lambda: object(),
        synthesizer_factory=lambda: object(),
        transcriber_factory=lambda: object(),
        vad_factory=lambda: object(),
        identity_factory=lambda: "qualification_identity",
    )

    assert _current_qualification_full_host_dependencies() is None
    with capability.wire(collector, full_host_dependencies=bundle):
        assert _current_qualification_full_host_dependencies() is bundle
        assert bundle.take() is bundle
        with pytest.raises(RuntimeError, match="one-shot"):
            bundle.take()
    assert _current_qualification_full_host_dependencies() is None


def test_full_host_dependency_bundle_retains_owner_bound_writer_factory() -> None:
    """The test-only capability may reach the real writer boundary only by factory."""

    from hermes_realtime._qualification import _new_qualification_full_host_dependencies

    identity = object()

    def writer_factory(runtime: object) -> object:
        return runtime

    bundle = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=lambda: object(),
        speech_presence_factory=lambda: object(),
        synthesizer_factory=lambda: object(),
        transcriber_factory=lambda: object(),
        vad_factory=lambda: object(),
        identity_factory=lambda: "qualification_identity",
        writer_transport_factory=writer_factory,
    )

    assert bundle.writer_transport_factory is writer_factory


def test_rollover_publication_fault_arm_is_owner_context_bound_single_use_and_optional() -> None:
    from hermes_realtime._qualification import (
        _new_qualification_full_host_dependencies,
        _new_qualification_owner_context_capability,
    )

    identity = object()
    foreign_identity = object()
    collector = _Collector(identity)
    capability = _new_qualification_owner_context_capability(identity)
    bundle = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=lambda: object(),
        speech_presence_factory=lambda: object(),
        synthesizer_factory=lambda: object(),
        transcriber_factory=lambda: object(),
        vad_factory=lambda: object(),
        identity_factory=lambda: "qualification_identity",
    )
    probe = bundle._capacity_probe

    assert probe.consume_rollover_publication_fault() is False
    assert probe.rollover_publication_fault_consumed() is False
    with pytest.raises(RuntimeError, match="active owner context"):
        probe.arm_rollover_publication_fault(identity)
    with capability.wire(collector, full_host_dependencies=bundle):
        with pytest.raises(RuntimeError, match="active owner context"):
            probe.arm_rollover_publication_fault(foreign_identity)
        probe.arm_rollover_publication_fault(identity)
        with pytest.raises(RuntimeError, match="single-use"):
            probe.arm_rollover_publication_fault(identity)
    assert probe.consume_rollover_publication_fault() is True
    assert probe.consume_rollover_publication_fault() is False
    assert probe.rollover_publication_fault_consumed() is True
    with (
        capability.wire(collector, full_host_dependencies=bundle),
        pytest.raises(RuntimeError, match="single-use"),
    ):
        probe.arm_rollover_publication_fault(identity)


def test_rollover_publication_fault_arm_has_no_public_reachability() -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import build_local_host_launcher

    for public in (HostEvidenceRuntimeV1, build_local_host_launcher):
        parameters = inspect.signature(public).parameters
        assert "rollover_publication_fault" not in parameters
        assert "qualification_capacity_probe" not in parameters


class _Collector:
    def __init__(self, identity: object) -> None:
        self._identity = identity

    def _qualification_owner_identity_matches(self, identity: object) -> bool:
        return identity is self._identity

    def record_committed_conversation_context_snapshot(self, value: bytes) -> None:
        del value

    def record_generated_text(self, value: bytes) -> None:
        del value

    def record_transport_confirmed_chunk(self, value: bytes) -> None:
        del value

    def record_cancellation(self, reason: object) -> None:
        del reason

    def record_foreground_cleanup(self, *, succeeded: bool) -> None:
        del succeeded
