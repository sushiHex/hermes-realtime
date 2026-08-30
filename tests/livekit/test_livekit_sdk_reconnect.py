from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from livekit.rtc import Room
from livekit.rtc._proto import room_pb2


@pytest.mark.asyncio
async def test_livekit_duplicate_local_track_subscription_is_idempotent() -> None:
    room = Room()
    track_sid = "TR_DUPLICATE_RECONNECT"
    first_subscription: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    publications = {
        track_sid: SimpleNamespace(
            _first_subscription=first_subscription,
            track=None,
        )
    }
    object.__setattr__(
        room,
        "_local_participant",
        SimpleNamespace(
            track_publications=publications,
            _track_publications=publications,
        ),
    )
    event = room_pb2.RoomEvent()
    event.local_track_subscribed.track_sid = track_sid

    room._on_room_event(event)
    room._on_room_event(event)

    assert first_subscription.done()
