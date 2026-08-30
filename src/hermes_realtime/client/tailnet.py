"""Fail-closed exact Tailnet TCP-peer authorization."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_MAX_WHOIS_BYTES = 65_536
_MAX_STABLE_ID_CHARS = 128
_STABLE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class TailnetAuthorizationError(PermissionError):
    """Generic public-safe Tailnet authorization failure."""

    def __init__(self) -> None:
        super().__init__("tailnet peer authorization failed")


@dataclass(frozen=True, slots=True)
class TailnetPeerAddress:
    """Normalized Tailnet TCP peer address obtained only from the socket."""

    ip: str
    port: int

    @classmethod
    def from_peername(cls, peername: object) -> TailnetPeerAddress:
        if type(peername) is not tuple or len(peername) not in {2, 4}:
            raise TypeError("socket peer metadata is invalid")
        raw_ip, raw_port = peername[0], peername[1]
        if type(raw_ip) is not str or type(raw_port) is not int:
            raise TypeError("socket peer metadata is invalid")
        if not 1 <= raw_port <= 65_535:
            raise ValueError("socket peer port is invalid")
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError:
            raise ValueError("socket peer address is invalid") from None
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if (
            address.is_unspecified
            or address.is_loopback
            or address not in (_TAILNET_V4 if address.version == 4 else _TAILNET_V6)
        ):
            raise ValueError("socket peer is not a Tailnet address")
        return cls(ip=address.compressed, port=raw_port)

    @property
    def socket_argument(self) -> str:
        return f"[{self.ip}]:{self.port}" if ":" in self.ip else f"{self.ip}:{self.port}"


WhoIsResolver = Callable[[TailnetPeerAddress], Awaitable[bytes]]


@dataclass(slots=True)
class _SourceGate:
    lock: asyncio.Lock
    users: int = 0


def tailscale_cli_executable() -> str:
    """Resolve the installed CLI without exposing installation details."""

    override = os.environ.get("HERMES_REALTIME_TAILSCALE_CLI")
    if override is not None:
        candidate = Path(override)
        if not candidate.is_absolute() or not candidate.is_file():
            raise RuntimeError("tailscale executable is unavailable")
        return str(candidate.resolve())
    if sys.platform == "win32":
        roots = tuple(
            value
            for value in (
                os.environ.get("PROGRAMFILES"),
                os.environ.get("PROGRAMW6432"),
                "C:/Program Files",
            )
            if value
        )
        for root in roots:
            candidate = Path(root) / "Tailscale" / "tailscale.exe"
            if candidate.is_file():
                return str(candidate.resolve())
    else:
        for candidate in (
            Path("/usr/bin/tailscale"),
            Path("/usr/local/bin/tailscale"),
            Path("/opt/homebrew/bin/tailscale"),
        ):
            if candidate.is_file():
                return str(candidate.resolve())
    raise RuntimeError("tailscale executable is unavailable")


class TailnetPeerAuthorizer:
    """Authorize only one exact immutable Tailscale node StableID."""

    def __init__(
        self,
        *,
        allowed_stable_id: str,
        resolver: WhoIsResolver,
        max_concurrency: int = 4,
    ) -> None:
        self.validate_allowed_stable_id(allowed_stable_id)
        if not callable(resolver):
            raise TypeError("resolver must be callable")
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= 16:
            raise ValueError("max_concurrency must be from 1 through 16")
        self._allowed_stable_id = allowed_stable_id
        self._resolver = resolver
        self._global_limit = asyncio.Semaphore(max_concurrency)
        self._source_gates: dict[str, _SourceGate] = {}

    @staticmethod
    def validate_allowed_stable_id(value: object) -> None:
        if type(value) is not str:
            raise TypeError("allowed_stable_id must be an exact string")
        if _STABLE_ID.fullmatch(value) is None:
            raise ValueError("allowed_stable_id is invalid")

    async def authorize(self, peer: TailnetPeerAddress) -> None:
        if type(peer) is not TailnetPeerAddress:
            raise TypeError("peer must be an exact TailnetPeerAddress")
        source_gate = self._source_gates.get(peer.ip)
        if source_gate is None:
            source_gate = _SourceGate(lock=asyncio.Lock())
            self._source_gates[peer.ip] = source_gate
        source_gate.users += 1
        try:
            async with source_gate.lock, self._global_limit:
                raw = await self._resolver(peer)
            stable_id = self._parse_stable_id(raw)
            if stable_id != self._allowed_stable_id:
                raise TailnetAuthorizationError()
        except asyncio.CancelledError:
            raise
        except TailnetAuthorizationError:
            raise
        except Exception:
            raise TailnetAuthorizationError() from None
        finally:
            source_gate.users -= 1
            if source_gate.users == 0 and self._source_gates.get(peer.ip) is source_gate:
                self._source_gates.pop(peer.ip, None)

    @staticmethod
    def _parse_stable_id(raw: object) -> str:
        if type(raw) is not bytes or not raw or len(raw) > _MAX_WHOIS_BYTES:
            raise TailnetAuthorizationError()

        def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if type(key) is not str or key in result:
                    raise ValueError
                result[key] = value
            return result

        try:
            decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise TailnetAuthorizationError() from None
        if type(decoded) is not dict:
            raise TailnetAuthorizationError()
        node = decoded.get("Node")
        if type(node) is not dict:
            raise TailnetAuthorizationError()
        stable_id = node.get("StableID")
        if (
            type(stable_id) is not str
            or len(stable_id) > _MAX_STABLE_ID_CHARS
            or _STABLE_ID.fullmatch(stable_id) is None
        ):
            raise TailnetAuthorizationError()
        return stable_id


class TailscaleCliWhoIsResolver:
    """Bounded argv-only adapter for ``tailscale whois --json``."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        if executable is None:
            executable = tailscale_cli_executable()
        if (
            type(executable) is not str
            or not executable
            or len(executable) > 260
            or not Path(executable).is_absolute()
        ):
            raise ValueError("tailscale executable is invalid")
        if type(timeout_seconds) not in {int, float}:
            raise TypeError("timeout_seconds must be an exact number")
        timeout = cast(int | float, timeout_seconds)
        if not math.isfinite(timeout) or not 0 < timeout <= 10:
            raise ValueError("timeout_seconds must be between 0 and 10")
        self._executable = executable
        self._timeout_seconds = float(timeout)

    @staticmethod
    async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except (Exception, asyncio.CancelledError):
            return

    @staticmethod
    async def _read_bounded(reader: asyncio.StreamReader) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await reader.read(min(8192, _MAX_WHOIS_BYTES + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_WHOIS_BYTES:
                raise TailnetAuthorizationError()

    async def __call__(self, peer: TailnetPeerAddress) -> bytes:
        if type(peer) is not TailnetPeerAddress:
            raise TypeError("peer must be an exact TailnetPeerAddress")
        process: asyncio.subprocess.Process | None = None
        try:
            async with asyncio.timeout(self._timeout_seconds):
                process = await asyncio.create_subprocess_exec(
                    self._executable,
                    "whois",
                    "--json",
                    "--proto",
                    "tcp",
                    peer.socket_argument,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                assert process.stdout is not None
                assert process.stderr is not None
                stdout, stderr, _ = await asyncio.gather(
                    self._read_bounded(process.stdout),
                    self._read_bounded(process.stderr),
                    process.wait(),
                )
            if (
                process.returncode != 0
                or len(stdout) > _MAX_WHOIS_BYTES
                or len(stderr) > _MAX_WHOIS_BYTES
            ):
                raise TailnetAuthorizationError()
            return stdout
        except asyncio.CancelledError:
            if process is not None:
                await self._kill_and_reap(process)
            raise
        except BaseException:
            if process is not None:
                await self._kill_and_reap(process)
            raise TailnetAuthorizationError() from None
