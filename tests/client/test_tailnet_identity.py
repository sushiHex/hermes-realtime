from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.client.tailnet import (
    TailnetAuthorizationError,
    TailnetPeerAddress,
    TailnetPeerAuthorizer,
    TailscaleCliWhoIsResolver,
)


@pytest.mark.asyncio
async def test_exact_stable_id_authorizes_a_valid_tailnet_socket_peer() -> None:
    calls: list[TailnetPeerAddress] = []

    async def resolve(peer: TailnetPeerAddress) -> bytes:
        calls.append(peer)
        return b'{"Node":{"StableID":"node-stable-id"}}'

    authorizer = TailnetPeerAuthorizer(
        allowed_stable_id="node-stable-id",
        resolver=resolve,
    )
    peer = TailnetPeerAddress.from_peername(("100.64.1.2", 54321))

    await authorizer.authorize(peer)

    assert calls == [peer]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b'{"Node":{}}',
        b'{"Node":{"StableID":"other"}}',
        b'{"Node":{"StableID":"node-stable-id","StableID":"node-stable-id"}}',
        b"[]",
        b"{",
        b"x" * 65_537,
    ],
    ids=["missing", "mismatch", "duplicate", "non-object", "malformed", "oversized"],
)
async def test_invalid_or_nonmatching_whois_fails_closed(payload: bytes) -> None:
    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        return payload

    authorizer = TailnetPeerAuthorizer(
        allowed_stable_id="node-stable-id",
        resolver=resolve,
    )
    with pytest.raises(TailnetAuthorizationError, match="tailnet peer authorization failed"):
        await authorizer.authorize(
            TailnetPeerAddress.from_peername(("100.64.1.2", 54321))
        )


@pytest.mark.parametrize(
    "peername",
    [
        None,
        ("not-an-ip", 1234),
        ("127.0.0.1", 1234),
        ("0.0.0.0", 1234),
        ("192.0.2.1", 1234),
        ("::1", 1234, 0, 0),
        ("::", 1234, 0, 0),
        ("2001:db8::1", 1234, 0, 0),
    ],
)
def test_non_tailnet_or_malformed_peer_is_rejected(peername: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TailnetPeerAddress.from_peername(peername)


@pytest.mark.asyncio
async def test_resolution_exceptions_are_redacted() -> None:
    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        raise RuntimeError(
            "100.64.1.2 private-login private-node node-stable-id raw-profile"
        )

    authorizer = TailnetPeerAuthorizer(
        allowed_stable_id="node-stable-id",
        resolver=resolve,
    )
    with pytest.raises(TailnetAuthorizationError) as raised:
        await authorizer.authorize(
            TailnetPeerAddress.from_peername(("100.64.1.2", 54321))
        )
    rendered = repr(raised.value)
    for secret in (
        "100.64.1.2",
        "private-login",
        "private-node",
        "node-stable-id",
        "raw-profile",
    ):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_concurrent_resolution_is_globally_bounded() -> None:
    active = 0
    maximum = 0
    release = asyncio.Event()

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await release.wait()
        active -= 1
        return b'{"Node":{"StableID":"node-stable-id"}}'

    authorizer = TailnetPeerAuthorizer(
        allowed_stable_id="node-stable-id",
        resolver=resolve,
        max_concurrency=2,
    )
    peers = [
        TailnetPeerAddress.from_peername((f"100.64.1.{index}", 5000 + index))
        for index in range(1, 5)
    ]
    tasks = [asyncio.create_task(authorizer.authorize(peer)) for peer in peers]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert maximum == 2
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_same_source_resolution_is_serialized() -> None:
    active = 0
    maximum = 0
    first_entered = asyncio.Event()
    release = asyncio.Event()

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        first_entered.set()
        await release.wait()
        active -= 1
        return b'{"Node":{"StableID":"node-stable-id"}}'

    authorizer = TailnetPeerAuthorizer(
        allowed_stable_id="node-stable-id", resolver=resolve
    )
    tasks = [
        asyncio.create_task(
            authorizer.authorize(
                TailnetPeerAddress.from_peername(("100.64.1.2", 5000 + index))
            )
        )
        for index in range(3)
    ]
    await first_entered.wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert maximum == 1
    release.set()
    await asyncio.gather(*tasks)
    assert maximum == 1


def test_resolver_rejects_relative_injected_executable() -> None:
    with pytest.raises(ValueError, match="executable"):
        TailscaleCliWhoIsResolver(executable="tailscale")


def test_resolver_finds_standard_windows_install_when_cli_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "Tailscale" / "tailscale.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
    monkeypatch.delenv("PROGRAMW6432", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_TAILSCALE_CLI", raising=False)

    resolver = TailscaleCliWhoIsResolver()

    assert Path(resolver._executable) == executable.resolve()


def test_resolver_fails_closed_when_cli_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.delenv("HERMES_REALTIME_TAILSCALE_CLI", raising=False)

    with pytest.raises(RuntimeError, match="unavailable"):
        TailscaleCliWhoIsResolver()


def test_windows_resolution_ignores_cwd_local_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "working"
    cwd.mkdir()
    (cwd / "tailscale.exe").write_bytes(b"untrusted")
    program_files = tmp_path / "trusted"
    trusted = program_files / "Tailscale" / "tailscale.exe"
    trusted.parent.mkdir(parents=True)
    trusted.write_bytes(b"trusted")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PROGRAMFILES", str(program_files))
    monkeypatch.delenv("PROGRAMW6432", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_TAILSCALE_CLI", raising=False)

    resolver = TailscaleCliWhoIsResolver()

    assert Path(resolver._executable) == trusted.resolve()


def test_tailnet_peer_normalizes_mapped_ipv4_and_brackets_ipv6() -> None:
    mapped = TailnetPeerAddress.from_peername(("::ffff:100.64.1.2", 54321, 0, 0))
    ipv6 = TailnetPeerAddress.from_peername(("fd7a:115c:a1e0::2", 54321, 0, 0))

    assert mapped.ip == "100.64.1.2"
    assert mapped.socket_argument == "100.64.1.2:54321"
    assert ipv6.socket_argument == "[fd7a:115c:a1e0::2]:54321"


class _BlockingProcess:
    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode: int | None = None
        self.killed = False
        self._done = asyncio.Event()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode


@pytest.mark.asyncio
async def test_cli_resolution_timeout_kills_child_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _BlockingProcess()

    async def create(*_args: object, **_kwargs: object) -> Any:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    resolver = TailscaleCliWhoIsResolver(
        executable="C:/synthetic/tailscale.exe", timeout_seconds=0.01
    )

    with pytest.raises(TailnetAuthorizationError):
        await resolver(TailnetPeerAddress.from_peername(("100.64.1.2", 54321)))

    assert process.killed


@pytest.mark.asyncio
async def test_cli_resolution_cancellation_kills_child_and_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _BlockingProcess()

    async def create(*_args: object, **_kwargs: object) -> Any:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    resolver = TailscaleCliWhoIsResolver(executable="C:/synthetic/tailscale.exe")
    task = asyncio.create_task(
        resolver(TailnetPeerAddress.from_peername(("100.64.1.2", 54321)))
    )
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed


class _ChunkReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def read(self, _limit: int) -> bytes:
        await asyncio.sleep(0)
        return self._chunks.pop(0) if self._chunks else b""


class _CompletedProcess:
    def __init__(self, stdout: list[bytes], stderr: list[bytes], returncode: int = 0) -> None:
        self.stdout = _ChunkReader(stdout)
        self.stderr = _ChunkReader(stderr)
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


@pytest.mark.asyncio
async def test_cli_reassembles_chunked_json_and_uses_exact_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _CompletedProcess(
        [b'{"Node":{"Sta', b'bleID":"node-stable-id"}}'], [b""]
    )
    captured: tuple[tuple[object, ...], dict[str, object]] | None = None

    async def create(*args: object, **kwargs: object) -> Any:
        nonlocal captured
        captured = (args, kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    resolver = TailscaleCliWhoIsResolver(executable="C:/trusted/tailscale.exe")
    peer = TailnetPeerAddress.from_peername(("100.64.1.2", 54321))

    raw = await resolver(peer)

    assert raw == b'{"Node":{"StableID":"node-stable-id"}}'
    assert captured is not None
    assert captured[0] == (
        "C:/trusted/tailscale.exe",
        "whois",
        "--json",
        "--proto",
        "tcp",
        "100.64.1.2:54321",
    )
    assert "shell" not in captured[1]


@pytest.mark.asyncio
async def test_cli_rejects_oversized_chunked_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _CompletedProcess([b"x" * 40_000, b"x" * 25_537], [b""])

    async def create(*_args: object, **_kwargs: object) -> Any:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    resolver = TailscaleCliWhoIsResolver(executable="C:/trusted/tailscale.exe")

    with pytest.raises(TailnetAuthorizationError):
        await resolver(TailnetPeerAddress.from_peername(("100.64.1.2", 54321)))
