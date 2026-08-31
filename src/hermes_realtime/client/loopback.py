"""Exact loopback TCP-peer authorization for a persistent local browser entry."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass


class LoopbackAuthorizationError(PermissionError):
    """Generic public-safe loopback authorization failure."""

    def __init__(self) -> None:
        super().__init__("loopback peer authorization failed")


@dataclass(frozen=True, slots=True)
class LoopbackPeerAddress:
    """Normalized loopback TCP peer address obtained only from the socket."""

    ip: str
    port: int

    @classmethod
    def from_peername(cls, peername: object) -> LoopbackPeerAddress:
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
        if not address.is_loopback:
            raise ValueError("socket peer is not a loopback address")
        return cls(ip=address.compressed, port=raw_port)


class LoopbackPeerAuthorizer:
    """Authorize only an exact loopback peer captured from the accepted socket."""

    async def authorize(self, peer: LoopbackPeerAddress) -> None:
        if type(peer) is not LoopbackPeerAddress:
            raise LoopbackAuthorizationError()
        try:
            address = ipaddress.ip_address(peer.ip)
        except ValueError:
            raise LoopbackAuthorizationError() from None
        if not address.is_loopback or type(peer.port) is not int or not 1 <= peer.port <= 65_535:
            raise LoopbackAuthorizationError()
