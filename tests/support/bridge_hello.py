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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

AUTH = "mutual_auth"


def proof(
    token: str,
    role: str,
    *,
    participant_id: str,
    client_nonce: str,
    server_nonce: str,
    requested: object,
    negotiated: object,
    metadata: Mapping[str, object],
) -> str:
    """HMAC-SHA256 with the token over the canonical JSON transcript of one role."""

    transcript = {
        "client_nonce": client_nonce,
        "metadata": dict(metadata),
        "negotiated": sorted(negotiated),  # type: ignore[call-overload]
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


async def _line(reader: asyncio.StreamReader) -> object:
    raw = await reader.readline()
    return json.loads(raw) if raw else None


async def authenticate(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    token: str,
    document: dict[str, object],
) -> dict[str, object]:
    """Complete the hello as a client; return the welcome once acceptance verified."""

    writer.write(json.dumps(document).encode("utf-8") + b"\n")
    await writer.drain()
    welcome = await _line(reader)
    assert type(welcome) is dict and welcome.get("ok") is True, welcome
    metadata = {key: welcome[key] for key in ("review_interval", "runtime") if key in welcome}
    signed = {
        "participant_id": document["participant_id"],
        "client_nonce": document["client_nonce"],
        "server_nonce": welcome["server_nonce"],
        "requested": document["capabilities"],
        "negotiated": welcome["capabilities"],
        "metadata": metadata,
    }
    assert welcome["proof"] == proof(token, "server", **signed)  # type: ignore[arg-type]
    writer.write(json.dumps({"proof": proof(token, "client", **signed)}).encode() + b"\n")  # type: ignore[arg-type]
    await writer.drain()
    accepted = await _line(reader)
    assert accepted == {"accepted": proof(token, "accept", **signed)}  # type: ignore[arg-type]
    return welcome


@dataclass
class FakeCompanion:
    """A companion that answers the hello with a welcome a test shapes, signed as specified.

    ``shape`` receives the client's hello and returns the welcome without its nonce and proof.
    ``sign`` may alter what the server proof covers, to splice a valid proof onto another
    welcome; ``accept`` decides the final acceptance line.
    """

    token: str
    shape: Callable[[dict[str, object]], dict[str, object]]
    sign: Callable[[dict[str, object]], dict[str, object]] = lambda covered: covered
    accept: str = "valid"
    server_nonce: str | None = None
    received: list[object] = field(default_factory=list)
    server_nonces: list[str] = field(default_factory=list)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            document = await _line(reader)
            self.received.append(document)
            assert type(document) is dict
            welcome = self.shape(document)
            if welcome.get("ok") is True:
                server_nonce = self.server_nonce or secrets.token_hex(32)
                self.server_nonces.append(server_nonce)
                covered = {
                    "participant_id": document["participant_id"],
                    "client_nonce": document["client_nonce"],
                    "server_nonce": server_nonce,
                    "requested": document["capabilities"],
                    "negotiated": welcome.get("capabilities", []),
                    "metadata": {
                        key: welcome[key] for key in ("review_interval", "runtime")
                        if key in welcome
                    },
                }
                signed = self.sign(dict(covered))
                welcome = welcome | {
                    "server_nonce": server_nonce,
                    "proof": proof(self.token, "server", **signed),  # type: ignore[arg-type]
                }
                writer.write(json.dumps(welcome).encode("utf-8") + b"\n")
                await writer.drain()
                answer = await _line(reader)
                if answer is not None:
                    self.received.append(answer)
                if self.accept == "valid":
                    writer.write(json.dumps({
                        "accepted": proof(self.token, "accept", **covered),  # type: ignore[arg-type]
                    }).encode() + b"\n")
                elif self.accept == "forged":
                    writer.write(json.dumps({
                        "accepted": proof("not-the-token-" * 3, "accept", **covered),  # type: ignore[arg-type]
                    }).encode() + b"\n")
                elif self.accept == "silent":
                    await asyncio.sleep(30)
                elif self.accept == "missing":
                    return
            else:
                writer.write(json.dumps(welcome).encode("utf-8") + b"\n")
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
