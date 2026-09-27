"""The companion's hosting: the owned start, the archive service, and the plugin wiring."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from test_archive import FakeHermes

from hermes_realtime.companion.archive import VoiceArchive
from hermes_realtime.companion.host import (
    CompanionEndpoint,
    VoiceCompanionHost,
    VoiceCompanionService,
    companion_endpoint,
)
from hermes_realtime.companion.integrity import ArchiveRefusal
from hermes_realtime.companion.store import CompanionStore
from hermes_realtime.protocol import (
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceArchiveRow,
)

_HOST_MARKER = "[voice-companion] "
_PROBE = Path(__file__).resolve().parents[1] / "support" / "companion_lock_probe.py"
_FLAGS = (
    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    if os.name == "nt"
    else 0
)
_TOKEN = "t" * 32


def _row(seq: int, role: str = "user", **overrides: Any) -> VoiceArchiveRow:
    fields: dict[str, Any] = {
        "seq": seq,
        "role": role,
        "text": f"row {seq}",
        "interrupted": False,
        "ts": 1_700_000_000.0 + seq,
        "gap_before": None,
    }
    fields.update(overrides)
    return VoiceArchiveRow(**fields)


def _event(rows: list[VoiceArchiveRow], seq_from: int = 0, conversation: str = "conv") -> Any:
    return VoiceArchiveEvent(
        type="voice_archive",
        conversation_id=conversation,
        generation=0,
        seq_from=seq_from,
        seq_through=rows[-1].seq,
        rows=rows,
    )


def _service(tmp_path: Path, hermes: FakeHermes) -> tuple[VoiceCompanionService, CompanionStore]:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes, lease_ttl_seconds=30.0)
    return VoiceCompanionService(archive, store, hermes), store


def _markers(output: str, prefix: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(prefix))
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


# --- the service ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_checks_compatibility_and_durability_before_any_conversation(
    tmp_path: Path,
) -> None:
    hermes = FakeHermes()
    hermes.compat_failures = ("missing:SessionDB",)
    service, store = _service(tmp_path, hermes)
    try:
        with pytest.raises(ArchiveRefusal, match="incompatible"):
            await service.start()
        assert hermes.calls == ["check_compatibility"]
        hermes.compat_failures = ()
        hermes.durability = 1
        with pytest.raises(ArchiveRefusal, match="durability"):
            await service.start()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_start_opens_no_conversation_and_first_contact_runs_the_open_order(
    tmp_path: Path,
) -> None:
    hermes = FakeHermes()
    first, store = _service(tmp_path, hermes)
    await first.start()
    assert type(await first.archive(_event([_row(0)]))) is VoiceArchiveAckEvent
    await first.close()
    store.close()
    hermes.calls.clear()

    second, store = _service(tmp_path, hermes)
    try:
        await second.start()
        assert hermes.calls == ["check_compatibility", "durability_level"]
        assert hermes.lease == {}
        assert type(await second.archive(_event([_row(0)]))) is VoiceArchiveAckEvent
        # M0's open on first contact: compatibility and durability, the lease (the previous
        # holder released first), then verification, then the archive itself.
        assert hermes.calls[2:7] == [
            "check_compatibility",
            "durability_level",
            "release_lease",
            "acquire_lease",
            "read_projection",
        ]
        assert hermes.lease
    finally:
        await second.close()
        store.close()


@pytest.mark.asyncio
async def test_start_never_refuses_over_the_number_of_bound_conversations(
    tmp_path: Path,
) -> None:
    from hermes_realtime.companion.integrity import EXPECTED_HEADER, genesis
    from hermes_realtime.companion.store import Progress

    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    for index in range(40):
        store.bind(f"old{index}", f"voice_old{index}", Progress(genesis(EXPECTED_HEADER), None))
    try:
        await service.start()
        assert type(await service.archive(_event([_row(0)]))) is VoiceArchiveAckEvent
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_an_unknown_outcome_recovers_in_the_same_process(tmp_path: Path) -> None:
    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    await service.start()
    try:
        failures = [OSError("the commit's outcome is unknown")]

        def crash_once() -> None:
            if failures:
                raise failures.pop()

        hermes.after_insert = crash_once
        event = _event([_row(0)])
        assert await service.archive(event) is None
        # The resend re-opens the conversation, which settles the pending state first.
        assert type(await service.archive(event)) is VoiceArchiveAckEvent
        assert len(hermes.rows[hermes.only()]) == 1
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_a_lost_lease_recovers_in_the_same_process(tmp_path: Path) -> None:
    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    await service.start()
    try:
        assert type(await service.archive(_event([_row(0)]))) is VoiceArchiveAckEvent
        session_id = hermes.only()
        hermes.lease[session_id] = "a foreign holder that took the lease"
        refused = await service.archive(_event([_row(1)], seq_from=1))
        assert type(refused) is VoiceArchiveRefusedEvent
        assert refused.category == "lease_lost"
        del hermes.lease[session_id]
        assert type(await service.archive(_event([_row(1)], seq_from=1))) is VoiceArchiveAckEvent
    finally:
        await service.close()
        store.close()


def test_binding_is_bounded_and_fails_closed_at_the_limit(tmp_path: Path) -> None:
    from hermes_realtime.companion.integrity import EXPECTED_HEADER, genesis
    from hermes_realtime.companion.store import MAX_BOUND_CONVERSATIONS, Progress

    assert MAX_BOUND_CONVERSATIONS >= 1024
    store = CompanionStore(tmp_path / "companion.db")
    try:
        creation = Progress(genesis(EXPECTED_HEADER), None)
        store._connection.executemany(  # Fill to the limit without one call per row.
            "INSERT INTO voice_archive (conversation_id, session_id, pending_count, "
            "pending_chain) VALUES (?, ?, 0, ?)",
            [
                (f"c{index}", f"voice_c{index}", creation.fingerprint.chain)
                for index in range(MAX_BOUND_CONVERSATIONS - 1)
            ],
        )
        store.bind("last", "voice_last", creation)
        with pytest.raises(ArchiveRefusal, match="conversations"):
            store.bind("over", "voice_over", creation)
        assert store.read("over") is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_an_archive_is_committed_then_acknowledged_with_its_exact_range(
    tmp_path: Path,
) -> None:
    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    await service.start()
    try:
        event = _event(
            [_row(0), _row(1, "assistant", interrupted=True), _row(4, gap_before=(2, 3))]
        )
        reply = await service.archive(event)
        assert reply == VoiceArchiveAckEvent(
            type="voice_archive_ack", conversation_id="conv", generation=0, seq_from=0,
            seq_through=4,
        )
        stored = hermes.rows[hermes.only()]
        assert [row["platform_message_id"] for row in stored] == [
            "voice:conv:0:0", "voice:conv:0:1", "voice:conv:0:4",
        ]
        assert [row["timestamp"] for row in stored] == [1_700_000_000.0, 1_700_000_001.0,
                                                       1_700_000_004.0]
        # A resend of an acknowledged batch inserts nothing and is acknowledged again.
        assert await service.archive(event) == reply
        assert len(hermes.rows[hermes.only()]) == 3
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "seq_from", "category"),
    [
        pytest.param([_row(0, interrupted=True)], 0, "invalid", id="interrupted-user-row"),
        pytest.param([_row(1, "assistant", gap_before=(0, 0))], 0, "invalid",
                     id="gap-on-an-assistant-row"),
        pytest.param([_row(0, text=" ")], 0, "invalid", id="blank-text"),
        pytest.param([_row(0), _row(2)], 0, "partition", id="hole"),
        pytest.param([_row(1)], 0, "partition", id="range-starts-before-the-first-row"),
        pytest.param([_row(3, gap_before=(1, 2))], 0, "partition", id="gap-after-a-hole"),
    ],
)
async def test_a_batch_that_does_not_partition_its_range_is_refused_whole(
    tmp_path: Path,
    rows: list[VoiceArchiveRow],
    seq_from: int,
    category: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    await service.start()
    try:
        reply = await service.archive(_event(rows, seq_from))
        assert type(reply) is VoiceArchiveRefusedEvent
        assert reply.category == category
        assert all(not rows for rows in hermes.rows.values())
        # A row M0's types refuse leaves one bounded marker: category and count only.
        expected = [{"refusal": "invalid", "rows": len(rows), "version": 1}]
        output = capsys.readouterr().out
        assert _markers(output, _HOST_MARKER) == (expected if category == "invalid" else [])
        assert "row " not in output
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_an_archive_refusal_is_returned_by_category(tmp_path: Path) -> None:
    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    await service.start()
    try:
        assert type(await service.archive(_event([_row(0)]))) is VoiceArchiveAckEvent
        # Skipping seq 1 continues nothing the archive holds.
        reply = await service.archive(_event([_row(2)], seq_from=2))
        assert type(reply) is VoiceArchiveRefusedEvent
        assert reply.category == "identity"
    finally:
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_an_unknown_outcome_is_answered_with_nothing(tmp_path: Path) -> None:
    hermes = FakeHermes()
    service, store = _service(tmp_path, hermes)
    await service.start()
    try:
        def crash() -> None:
            raise OSError("disk went away")

        hermes.after_insert = crash
        assert await service.archive(_event([_row(0)])) is None
    finally:
        await service.close()
        store.close()


# --- the host --------------------------------------------------------------------------------


class _Bridge:
    def __init__(self, voice: VoiceCompanionService, log: list[str]) -> None:
        self.voice = voice
        self.log = log

    async def start(self) -> None:
        self.log.append("bridge:start")

    async def close(self) -> None:
        self.log.append("bridge:close")


def _host(tmp_path: Path, hermes: FakeHermes, log: list[str]) -> VoiceCompanionHost:
    def bridge(voice: VoiceCompanionService) -> _Bridge:
        log.append("bridge:build")
        return _Bridge(voice, log)

    return VoiceCompanionHost(
        store_path=tmp_path / "companion.db",
        open_port=lambda: hermes,
        bridge_factory=bridge,  # type: ignore[arg-type]
        lease_ttl_seconds=30.0,
    )


def test_an_owned_start_builds_the_bridge_only_after_the_service_is_ready(
    tmp_path: Path,
) -> None:
    hermes = FakeHermes()
    log: list[str] = []
    host = _host(tmp_path, hermes, log)
    try:
        host.start()
        assert host.wait_ready(5.0) is True
        assert hermes.calls[:2] == ["check_compatibility", "durability_level"]
        assert log == ["bridge:build", "bridge:start"]
    finally:
        host.close()
    assert log == ["bridge:build", "bridge:start", "bridge:close"]
    assert host.wait_ready(0.0) is False


def test_a_failed_start_never_starts_the_bridge_and_leaves_one_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hermes = FakeHermes()
    hermes.compat_failures = ("missing:SessionDB",)
    log: list[str] = []
    host = _host(tmp_path, hermes, log)
    host.start()
    assert host.wait_ready(0.5) is False
    host.close()

    assert log == []
    assert hermes.calls[-1] == "close"
    assert _markers(capsys.readouterr().out, _HOST_MARKER) == [
        {"refusal": "incompatible", "version": 1}
    ]


def test_multiplexing_is_refused_while_a_companion_is_owned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hermes = FakeHermes()
    first = _host(tmp_path / "a", hermes, [])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first.start()
    try:
        second = _host(tmp_path / "b", FakeHermes(), [])
        with pytest.raises(RuntimeError, match="multiplex"):
            second.start()
        assert _markers(capsys.readouterr().out, _HOST_MARKER) == [
            {"refusal": "multiplexed", "version": 1}
        ]
    finally:
        first.close()
    third = _host(tmp_path / "b", FakeHermes(), [])
    third.start()
    try:
        assert third.wait_ready(5.0) is True
    finally:
        third.close()


def _probe(mode: str, data_dir: Path, control: Path, port: int) -> subprocess.Popen[bytes]:
    environment = dict(os.environ) | {
        "HERMES_REALTIME_COMPANION_PORT": str(port),
        "HERMES_REALTIME_COMPANION_TOKEN": _TOKEN,
    }
    return subprocess.Popen(
        (sys.executable, str(_PROBE), mode, str(data_dir), str(control)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        creationflags=_FLAGS,
    )


def test_a_second_process_finding_the_profile_owned_stands_down(tmp_path: Path) -> None:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    data_dir, control = tmp_path / "plugin-data", tmp_path / "control"
    data_dir.mkdir()
    control.mkdir()
    holder = _probe("hold", data_dir, control, port)
    try:
        deadline = time.monotonic() + 30
        while not (control / "ready").exists():
            assert holder.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        contender = _probe("contend", data_dir, control, port)
        output = contender.communicate(timeout=60)[0].decode("utf-8", "replace")
    finally:
        (control / "stop").write_text("stop", encoding="utf-8")
        holder.wait(timeout=60)

    # The contender touched no lease, row, store or port, and left nothing to unload.
    assert _markers(output, "[probe] ") == [
        {"owned": False, "open_port": 0, "hermes_calls": 0, "unload_callbacks": 0}
    ]
    assert _markers(output, _HOST_MARKER) == [{"refusal": "held", "version": 1}]
    assert holder.returncode == 0
    # Once the owner unloads, the profile can be owned again.
    successor = _probe("contend", data_dir, control, port)
    later = successor.communicate(timeout=60)[0].decode("utf-8", "replace")
    assert _markers(later, "[probe] ") == [
        {"owned": True, "open_port": 1, "hermes_calls": 2, "unload_callbacks": 1}
    ]


def test_the_profile_lock_is_taken_before_anything_and_held_until_close(
    tmp_path: Path,
) -> None:
    from hermes_realtime.integration.run_record import lock_run_record, unlock_run_record

    host = _host(tmp_path, FakeHermes(), [])
    store_path = tmp_path / "companion.db"
    assert host.start() is True
    try:
        assert host.wait_ready(5.0)
        assert lock_run_record(store_path) is None
    finally:
        host.close()
    descriptor = lock_run_record(store_path)
    assert descriptor is not None
    unlock_run_record(descriptor)


def test_close_releases_every_lease_and_stops_the_loop_thread(tmp_path: Path) -> None:
    hermes = FakeHermes()
    host = _host(tmp_path, hermes, [])
    host.start()
    assert host.wait_ready(5.0)
    reply = host.submit(host.service.archive(_event([_row(0)])))
    assert type(reply) is VoiceArchiveAckEvent
    assert hermes.lease

    host.close()
    host.close()

    assert hermes.lease == {}
    # The database closes once, last, after the lease it held was released.
    assert hermes.calls.count("close") == 1
    assert hermes.calls[-1] == "close"
    assert "release_lease" in hermes.calls[: hermes.calls.index("close")]
    assert not any(thread.name == "voice-companion" for thread in threading.enumerate())


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, None),
        (
            {"HERMES_REALTIME_COMPANION_PORT": "8765", "HERMES_REALTIME_COMPANION_TOKEN": _TOKEN},
            CompanionEndpoint(port=8765, token=_TOKEN),
        ),
    ],
)
def test_the_endpoint_comes_from_the_environment(
    environ: dict[str, str], expected: CompanionEndpoint | None
) -> None:
    assert companion_endpoint(environ) == expected


@pytest.mark.parametrize(
    "environ",
    [
        {"HERMES_REALTIME_COMPANION_PORT": "8765"},
        {"HERMES_REALTIME_COMPANION_TOKEN": _TOKEN},
        {"HERMES_REALTIME_COMPANION_PORT": "0", "HERMES_REALTIME_COMPANION_TOKEN": _TOKEN},
        {"HERMES_REALTIME_COMPANION_PORT": "65536", "HERMES_REALTIME_COMPANION_TOKEN": _TOKEN},
        {"HERMES_REALTIME_COMPANION_PORT": " 8765", "HERMES_REALTIME_COMPANION_TOKEN": _TOKEN},
        {"HERMES_REALTIME_COMPANION_PORT": "8765", "HERMES_REALTIME_COMPANION_TOKEN": "short"},
    ],
)
def test_a_partial_or_malformed_endpoint_fails_closed(environ: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="companion"):
        companion_endpoint(environ)


def test_the_marker_never_carries_the_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError) as caught:
        companion_endpoint(
            {"HERMES_REALTIME_COMPANION_PORT": "x", "HERMES_REALTIME_COMPANION_TOKEN": _TOKEN}
        )
    assert _TOKEN not in str(caught.value)
    assert _TOKEN not in capsys.readouterr().out


def test_submit_requires_a_ready_host(tmp_path: Path) -> None:
    host = _host(tmp_path, FakeHermes(), [])

    async def nothing() -> None:
        return None

    coroutine = nothing()
    with pytest.raises(RuntimeError, match="ready"):
        host.submit(coroutine)
    coroutine.close()
    asyncio.run(asyncio.sleep(0))
