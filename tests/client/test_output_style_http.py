import json

import pytest

from hermes_realtime.client import (
    BrowserBootstrapApplication,
    BrowserEventProjection,
    BrowserSessionDirector,
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    LoopbackPeerAuthorizer,
)
from hermes_realtime.conversation.output_style import OutputStyleSelection
from hermes_realtime.livekit import LiveKitConnection


async def noop(*args):
    pass


def setup():
    selection = OutputStyleSelection()
    connection = LiveKitConnection("wss://livekit.test", "key", "synthetic-secret" * 3)

    async def provision(identity):
        return 1

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(connection=connection, room_name="room",
                                 identity_factory=lambda: "browser_0123456789abcdef"),
        provision=provision, submit=noop, stop=noop, approval=noop,
        projection=BrowserEventProjection(), select_output_style=selection.select,
    )
    app = BrowserBootstrapApplication(
        sessions=director, verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None, loopback_authorizer=LoopbackPeerAuthorizer(), allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
    )
    return selection, director, app


async def request(app, token, body, **changes):
    return await app.handle(method="POST", path="/api/v1/output-style", headers={
        "origin": "http://127.0.0.1:8765", "authorization": f"Bearer {token}",
        "content-type": "application/json", "content-length": str(len(body)), **changes,
    }, body=body)


@pytest.mark.asyncio
async def test_output_style_authenticated_selection_has_no_work_side_effect() -> None:
    selection, director, app = setup()
    credential = await director.start()
    reply = await request(app, credential.token, b'{"style":"learning"}')
    assert json.loads(reply.body) == {"version": 1, "selectedStyle": "learning"}
    assert selection.get() == "learning"
    assert director._active_generation == 1
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.parametrize("body", [b'{"style":"Unknown"}', b'{"style":true}',
    b'{"style":"concise","authority":true}', b'{"style":"default","style":"concise"}',
    b'{}', b'[]', b'\xff', b'x' * 257, b'', b' ' * 257 + b'{"style":"default"}'])
@pytest.mark.asyncio
async def test_output_style_refuses_invalid_requests(body) -> None:
    selection, director, app = setup()
    credential = await director.start()
    with pytest.raises((ValueError, TypeError)):
        await request(app, credential.token, body)
    assert selection.get() == "default"
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
async def test_output_style_requires_current_identity_and_generation() -> None:
    selection, director, app = setup()
    credential = await director.start()
    with pytest.raises(PermissionError):
        await director.change_output_style(
            participant_identity="browser_fedcba9876543210", style="concise",
        )
    director._active_generation = None
    with pytest.raises(Exception, match="no browser session"):
        await director.change_output_style(
            participant_identity=credential.participant_identity, style="concise",
        )
    assert selection.get() == "default"
    director._active_generation = 1
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
async def test_output_style_requires_exact_participant_identity() -> None:
    class Identity(str):
        pass

    selection, director, app = setup()
    credential = await director.start()
    with pytest.raises(TypeError):
        await director.change_output_style(
            participant_identity=Identity(credential.participant_identity), style="concise",
        )
    assert selection.get() == "default"
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.parametrize("changes", [{"origin":"http://other.test"},
                                     {"authorization":"Bearer invalid"},
                                     {"content-type":"text/plain"}])
@pytest.mark.asyncio
async def test_output_style_requires_authenticated_same_origin_json(changes) -> None:
    selection, director, app = setup()
    credential = await director.start()
    with pytest.raises((ValueError, PermissionError)):
        await request(app, credential.token, b'{"style":"concise"}', **changes)
    assert selection.get() == "default"
    await director.stop(participant_identity=credential.participant_identity)
