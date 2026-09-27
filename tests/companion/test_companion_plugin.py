"""Plugin registration builds the companion; an owned start serves it; unload closes it."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from test_archive import FakeHermes

from hermes_realtime import hermes_plugin
from hermes_realtime.integration import BridgeAuthenticationError, LocalHermesBridgeClient

_TOKEN = "companion-test-token-with-enough-entropy"
_MARKER = "[voice-companion] "


class _State:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class _Context:
    """The v0.21 PluginContext surface the plugin uses."""

    def __init__(self, data_dir: Path) -> None:
        self.state = _State(data_dir)
        self.unload: list[Callable[[], None]] = []

    @property
    def subagent_lifecycle(self) -> object:
        return object()

    @property
    def profile_name(self) -> str:
        return "default"

    def on_unload(self, callback: Callable[[], None]) -> object:
        self.unload.append(callback)
        return object()


class _LegacyContext:
    """A context without ``state`` or ``on_unload``: the companion cannot be owned."""

    @property
    def subagent_lifecycle(self) -> object:
        return object()


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def hermes(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHermes]:
    fake = FakeHermes()
    monkeypatch.setattr(hermes_plugin, "_open_archive_port", lambda: fake)
    yield fake
    companion = hermes_plugin._companion
    if companion is not None:
        companion.close()
        hermes_plugin._companion = None


def _configure(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    monkeypatch.setenv("HERMES_REALTIME_COMPANION_PORT", str(port))
    monkeypatch.setenv("HERMES_REALTIME_COMPANION_TOKEN", _TOKEN)


def _markers(output: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(_MARKER))
        for line in output.splitlines()
        if line.startswith(_MARKER)
    ]


async def _negotiate(port: int) -> frozenset[str]:
    client = await LocalHermesBridgeClient.connect(
        host="127.0.0.1", port=port, token=_TOKEN, participant_id="voice-archive",
        capabilities=("voice_archive",),
    )
    try:
        return client.capabilities
    finally:
        await client.close()


def test_registration_starts_an_owned_companion_that_unload_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hermes: FakeHermes
) -> None:
    port = _free_port()
    _configure(monkeypatch, port)
    context = _Context(tmp_path / "plugin-data")

    hermes_plugin.register(context)

    companion = hermes_plugin._companion
    assert companion is not None
    assert companion.wait_ready(5.0) is True
    assert (tmp_path / "plugin-data" / "voice-companion.db").exists()
    assert asyncio.run(_negotiate(port)) == frozenset({"voice_archive"})
    assert len(context.unload) == 1

    context.unload[0]()

    assert companion.wait_ready(0.0) is False
    with pytest.raises(OSError):
        asyncio.run(_negotiate(port))


def test_an_unconfigured_registration_starts_no_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hermes: FakeHermes
) -> None:
    monkeypatch.delenv("HERMES_REALTIME_COMPANION_PORT", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_COMPANION_TOKEN", raising=False)
    context = _Context(tmp_path)

    hermes_plugin.register(context)

    assert hermes_plugin._companion is None
    assert context.unload == []
    assert hermes.calls == []


@pytest.mark.parametrize(
    ("context_kind", "port_value", "category"),
    [
        pytest.param("legacy", None, "context", id="context-without-unload-or-state"),
        pytest.param("modern", "not-a-port", "endpoint", id="malformed-endpoint"),
    ],
)
def test_a_companion_that_cannot_be_owned_is_refused_and_dispatch_still_registers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermes: FakeHermes,
    context_kind: str,
    port_value: str | None,
    category: str,
) -> None:
    _configure(monkeypatch, _free_port())
    if port_value is not None:
        monkeypatch.setenv("HERMES_REALTIME_COMPANION_PORT", port_value)
    context = _LegacyContext() if context_kind == "legacy" else _Context(tmp_path)

    hermes_plugin.register(context)

    assert hermes_plugin._companion is None
    assert hermes_plugin.get_runtime() is not None
    output = capsys.readouterr().out
    assert _markers(output) == [{"refusal": category, "version": 1}]
    assert _TOKEN not in output
    assert hermes.calls == []


def test_a_second_registration_while_one_is_owned_refuses_to_multiplex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermes: FakeHermes,
) -> None:
    _configure(monkeypatch, _free_port())
    first = _Context(tmp_path / "one")
    hermes_plugin.register(first)
    owned = hermes_plugin._companion
    assert owned is not None and owned.wait_ready(5.0)
    capsys.readouterr()
    second = _Context(tmp_path / "two")

    hermes_plugin.register(second)

    assert hermes_plugin._companion is owned
    assert second.unload == []
    assert _markers(capsys.readouterr().out) == [{"refusal": "multiplexed", "version": 1}]


def test_the_companion_bridge_refuses_the_wrong_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hermes: FakeHermes
) -> None:
    port = _free_port()
    _configure(monkeypatch, port)
    hermes_plugin.register(_Context(tmp_path))
    companion = hermes_plugin._companion
    assert companion is not None and companion.wait_ready(5.0)

    async def wrong() -> None:
        await LocalHermesBridgeClient.connect(
            host="127.0.0.1", port=port, token="x" * 40, participant_id="voice-archive",
            capabilities=("voice_archive",),
        )

    with pytest.raises(BridgeAuthenticationError):
        asyncio.run(wrong())
