from __future__ import annotations

import pytest

from hermes_realtime.client import (
    LoopbackAuthorizationError,
    LoopbackPeerAddress,
    LoopbackPeerAuthorizer,
)


@pytest.mark.parametrize(
    "peername,expected",
    [
        (("127.0.0.1", 8765), LoopbackPeerAddress("127.0.0.1", 8765)),
        (("::1", 8765, 0, 0), LoopbackPeerAddress("::1", 8765)),
        (("::ffff:127.0.0.1", 8765, 0, 0), LoopbackPeerAddress("127.0.0.1", 8765)),
    ],
)
def test_loopback_peer_address_normalizes_only_socket_loopback_peers(
    peername: object,
    expected: LoopbackPeerAddress,
) -> None:
    assert LoopbackPeerAddress.from_peername(peername) == expected


@pytest.mark.parametrize(
    "peername",
    [
        ("100.64.0.1", 8765),
        ("192.168.1.2", 8765),
        ("127.0.0.1", 0),
        ("127.0.0.1", 65536),
        ("127.0.0.1", True),
        ["127.0.0.1", 8765],
        ("invalid", 8765),
    ],
)
def test_loopback_peer_address_rejects_nonloopback_and_malformed_peers(
    peername: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        LoopbackPeerAddress.from_peername(peername)


@pytest.mark.asyncio
async def test_loopback_authorizer_accepts_only_the_exact_validated_address_type() -> None:
    authorizer = LoopbackPeerAuthorizer()
    await authorizer.authorize(LoopbackPeerAddress("127.0.0.1", 8765))

    class AddressSubclass(LoopbackPeerAddress):
        pass

    for peer in (
        AddressSubclass("127.0.0.1", 8765),
        LoopbackPeerAddress("100.64.0.1", 8765),
        LoopbackPeerAddress("127.0.0.1", 0),
    ):
        with pytest.raises(LoopbackAuthorizationError):
            await authorizer.authorize(peer)
