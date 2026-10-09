"""The bridge hello as docs/hermes-bridge.md specifies it, written independently of the code.

Tests use it to speak the protocol from either side, so a change to the implementation that
the document does not describe fails here instead of passing against itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import socket
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

AUTH = "mutual_auth"
# What only the final authenticated acceptance carries; the welcome carries none of it.
ACCEPTED_KEYS = ("capabilities", "review_interval", "runtime")


def proof(
    token: str,
    role: str,
    *,
    participant_id: str,
    client_nonce: str,
    server_nonce: str,
    requested: object,
    negotiated: object = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """HMAC-SHA256 with the token over the canonical JSON transcript of one role.

    The ``server`` and ``client`` proofs come before anything is negotiated, so their
    ``negotiated`` and ``metadata`` are null; only the ``accept`` proof binds them.
    """

    transcript = {
        "client_nonce": client_nonce,
        "metadata": None if metadata is None else dict(metadata),
        "negotiated": None if negotiated is None else sorted(negotiated),  # type: ignore[call-overload]
        "participant_id": participant_id,
        "protocol_version": "0.3",
        "requested": sorted(requested),  # type: ignore[call-overload]
        "role": role,
        "server_nonce": server_nonce,
    }
    message = json.dumps(transcript, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hmac.new(token.encode("utf-8"), message.encode("ascii"), hashlib.sha256).hexdigest()


def hello(
    participant_id: str = "voice-archive",
    capabilities: tuple[str, ...] = ("voice_archive",),
    **overrides: object,
) -> dict[str, object]:
    document: dict[str, object] = {
        "participant_id": participant_id,
        "protocol_version": "0.3",
        "capabilities": sorted({*capabilities, AUTH}),
        "client_nonce": secrets.token_hex(32),
    }
    document.update(overrides)
    return document


def reset(sock: socket.socket) -> None:
    """Close a raw socket with a reset rather than an orderly close.

    A stream transport shuts its socket down before closing it, so its peer reads an
    orderly end of stream; only a raw socket closed with a zero linger sends a reset.
    """
    linger = struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
    sock.close()


async def raw_connect(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    await asyncio.get_running_loop().sock_connect(sock, (host, port))
    return sock


async def raw_line(sock: socket.socket) -> bytes:
    """Read up to and including one newline from a raw socket."""
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = await asyncio.get_running_loop().sock_recv(sock, 1)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


async def _line(reader: asyncio.StreamReader) -> object:
    raw = await reader.readline()
    return json.loads(raw) if raw else None


async def authenticate(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    token: str,
    document: dict[str, object],
) -> dict[str, object]:
    """Complete the hello as a client.

    Returns the welcome merged with what the verified acceptance carried, without the
    acceptance's proof.
    """

    writer.write(json.dumps(document).encode("utf-8") + b"\n")
    await writer.drain()
    welcome = await _line(reader)
    assert type(welcome) is dict and welcome.get("ok") is True, welcome
    assert set(welcome) == {"ok", "protocol_version", "server_nonce", "proof"}, welcome
    opening = {
        "participant_id": document["participant_id"],
        "client_nonce": document["client_nonce"],
        "server_nonce": welcome["server_nonce"],
        "requested": document["capabilities"],
    }
    assert welcome["proof"] == proof(token, "server", **opening)  # type: ignore[arg-type]
    writer.write(json.dumps({"proof": proof(token, "client", **opening)}).encode() + b"\n")  # type: ignore[arg-type]
    await writer.drain()
    accepted = await _line(reader)
    assert type(accepted) is dict and "accepted" in accepted, accepted
    metadata = {key: accepted[key] for key in ("review_interval", "runtime") if key in accepted}
    assert accepted["accepted"] == proof(
        token, "accept", **opening,  # type: ignore[arg-type]
        negotiated=accepted["capabilities"], metadata=metadata,
    )
    return welcome | {key: value for key, value in accepted.items() if key != "accepted"}


@dataclass
class FakeCompanion:
    """A companion that answers the hello as a test shapes it, signed as specified.

    ``shape`` receives the client's hello and returns the whole answer without nonce or
    proofs: ``capabilities``, ``review_interval`` and ``runtime`` go into the final
    acceptance, everything else into the welcome. ``sign`` may alter what the server proof
    covers and ``sign_accept`` what the acceptance proof covers, to splice a valid proof
    onto another handshake; ``accept`` decides the final acceptance line, and ``accept_extra``
    adds fields to it that no proof covers.
    """

    token: str
    shape: Callable[[dict[str, object]], dict[str, object]]
    sign: Callable[[dict[str, object]], dict[str, object]] = lambda covered: covered
    sign_accept: Callable[[dict[str, object]], dict[str, object]] = lambda covered: covered
    accept: str = "valid"
    accept_extra: dict[str, object] = field(default_factory=dict)
    server_nonce: str | None = None
    received: list[object] = field(default_factory=list)
    server_nonces: list[str] = field(default_factory=list)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            document = await _line(reader)
            self.received.append(document)
            assert type(document) is dict
            answer = self.shape(document)
            welcome = {key: value for key, value in answer.items() if key not in ACCEPTED_KEYS}
            final = {key: value for key, value in answer.items() if key in ACCEPTED_KEYS}
            if welcome.get("ok") is True:
                server_nonce = self.server_nonce or secrets.token_hex(32)
                self.server_nonces.append(server_nonce)
                opening = {
                    "participant_id": document["participant_id"],
                    "client_nonce": document["client_nonce"],
                    "server_nonce": server_nonce,
                    "requested": document["capabilities"],
                }
                welcome = welcome | {
                    "server_nonce": server_nonce,
                    "proof": proof(self.token, "server", **self.sign(dict(opening))),  # type: ignore[arg-type]
                }
                writer.write(json.dumps(welcome).encode("utf-8") + b"\n")
                await writer.drain()
                if self.accept == "dropped":
                    return
                reply = await _line(reader)
                if reply is not None:
                    self.received.append(reply)
                covered = self.sign_accept(opening | {
                    "negotiated": final.get("capabilities", []),
                    "metadata": {
                        key: final[key] for key in ("review_interval", "runtime") if key in final
                    },
                })
                if self.accept == "valid":
                    writer.write(json.dumps(final | self.accept_extra | {
                        "accepted": proof(self.token, "accept", **covered),  # type: ignore[arg-type]
                    }).encode() + b"\n")
                elif self.accept == "forged":
                    writer.write(json.dumps(final | {
                        "accepted": proof("not-the-token-" * 3, "accept", **covered),  # type: ignore[arg-type]
                    }).encode() + b"\n")
                elif self.accept == "silent":
                    await asyncio.sleep(30)
                elif self.accept == "missing":
                    return
            else:
                writer.write(json.dumps(answer).encode("utf-8") + b"\n")
            await writer.drain()
            while line := await reader.readline():
                self.received.append(json.loads(line))
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    async def start(self) -> tuple[asyncio.Server, int]:
        server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return server, server.sockets[0].getsockname()[1]
