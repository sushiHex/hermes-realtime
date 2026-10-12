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
async def test_fresh_session_resets_style_before_provision() -> None:
    selection, director, _app = setup()
    observed = []

    async def provision(_identity):
        observed.append(selection.get())
        return len(observed)

    director._provision = provision
    selection.select("learning")
    first = await director.start()
    assert observed == ["default"]
    await director.change_output_style(participant_identity=first.participant_identity,
                                       style="learning")
    await director.stop(participant_identity=first.participant_identity)
    second = await director.start()
    assert observed == ["default", "default"]
    assert selection.get() == "default"
    await director.stop(participant_identity=second.participant_identity)


@pytest.mark.asyncio
async def test_active_start_refusal_and_ordinary_rebind_preserve_style() -> None:
    selection, director, _app = setup()
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    director._issuer._identity_factory = lambda: next(identities)

    async def reprovision(_identity):
        assert selection.get() == "learning"
        return 2

    director._reprovision = reprovision
    first = await director.start()
    await director.change_output_style(participant_identity=first.participant_identity,
                                       style="learning")
    with pytest.raises(RuntimeError, match="already active"):
        await director.start()
    assert selection.get() == "learning"
    rebound = await director.rebind(participant_identity=first.participant_identity,
                                    request_id="rebind_0123456789abcdef")
    assert selection.get() == "learning"
    await director.stop(participant_identity=rebound.participant_identity)


@pytest.mark.asyncio
async def test_fresh_style_reset_failure_refuses_before_worker_and_emits_category(capsys) -> None:
    selection, director, _app = setup()
    provisioned = []

    async def provision(identity):
        provisioned.append(identity)
        return 1

    def refuse(_style):
        raise ValueError("Synthetic private reset failure")

    director._provision = provision
    director._select_output_style = refuse
    with pytest.raises(ValueError):
        await director.start()
    assert provisioned == []
    assert director.active_identity is None
    assert director.active_generation is None
    assert capsys.readouterr().out.splitlines() == [
        '[output-style] {"accepted":false,"category":"unavailable_or_selection_failure",'
        '"version":1}'
    ]
    director._select_output_style = selection.select
    credential = await director.start()
    assert len(provisioned) == 1
    assert selection.get() == "default"
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
async def test_fresh_session_without_optional_style_selector_still_starts(capsys) -> None:
    _selection, director, _app = setup()
    director._select_output_style = None
    credential = await director.start()
    assert director.active_generation == 1
    assert capsys.readouterr().out == ""
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
async def test_output_style_authenticated_selection_has_no_work_side_effect(capsys) -> None:
    selection, director, app = setup()
    credential = await director.start()
    reply = await request(app, credential.token, b'{"style":"learning"}')
    assert json.loads(reply.body) == {"version": 1, "selectedStyle": "learning"}
    assert selection.get() == "learning"
    assert director._active_generation == 1
    assert capsys.readouterr().out.splitlines() == [
        '[output-style] {"accepted":true,"version":1}'
    ]
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.parametrize("body", [b'{"style":"Unknown"}', b'{"style":true}',
    b'{"style":"concise","authority":true}', b'{"style":"default","style":"concise"}',
    b'{}', b'[]', b'\xff', b'x' * 257, b'', b' ' * 257 + b'{"style":"default"}'])
@pytest.mark.asyncio
async def test_output_style_refuses_invalid_requests(body, capsys) -> None:
    selection, director, app = setup()
    credential = await director.start()
    with pytest.raises((ValueError, TypeError)):
        await request(app, credential.token, body)
    assert selection.get() == "default"
    assert capsys.readouterr().out.splitlines() == [
        '[output-style] {"accepted":false,"category":"malformed_input","version":1}'
    ]
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.parametrize("path", ["/api/v1/refresh", "/api/v1/voice"])
@pytest.mark.asyncio
async def test_output_style_evidence_does_not_mark_unrelated_requests(path, capsys) -> None:
    _selection, director, app = setup()
    credential = await director.start()
    headers = {
        "origin": "http://127.0.0.1:8765", "authorization": f"Bearer {credential.token}",
        "content-length": "0",
    }
    if path == "/api/v1/voice":
        with pytest.raises(ValueError):
            await app.handle(method="POST", path=path, headers=headers, body=b"")
    else:
        reply = await app.handle(method="POST", path=path, headers=headers, body=b"")
        assert reply.status == 200
    assert capsys.readouterr().out == ""
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


@pytest.mark.parametrize(("changes", "category"), [
    ({"origin":"http://other.test"}, "authentication_or_binding"),
    ({"authorization":"Bearer invalid"}, "authentication_or_binding"),
    ({"authorization":"invalid"}, "authentication_or_binding"),
    ({"content-type":"text/plain"}, "malformed_input"),
    ({"content-length":"999"}, "malformed_input"),
])
@pytest.mark.asyncio
async def test_output_style_requires_authenticated_same_origin_json(
    changes, category, capsys,
) -> None:
    selection, director, app = setup()
    credential = await director.start()
    with pytest.raises((ValueError, PermissionError)):
        await request(app, credential.token, b'{"style":"concise"}', **changes)
    assert selection.get() == "default"
    assert capsys.readouterr().out.splitlines() == [
        '[output-style] ' + json.dumps(
            {"accepted": False, "category": category, "version": 1}, separators=(",", ":"),
        )
    ]
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.parametrize(("failure", "category", "error"), [
    ("binding", "authentication_or_binding", PermissionError),
    ("absent_identity", "absent_session", RuntimeError),
    ("absent_generation", "absent_session", RuntimeError),
    ("unavailable", "unavailable_or_selection_failure", RuntimeError),
    ("selector_value_error", "unavailable_or_selection_failure", ValueError),
])
@pytest.mark.asyncio
async def test_output_style_session_refusals_emit_only_bounded_category(
    failure, category, error, capsys,
) -> None:
    selection, director, app = setup()
    credential = await director.start()
    identity, generation, selector = (
        director._active_identity, director._active_generation, director._select_output_style,
    )
    if failure == "binding":
        director._active_identity = "browser_fedcba9876543210"
    elif failure == "absent_identity":
        director._active_identity = None
    elif failure == "absent_generation":
        director._active_generation = None
    elif failure == "unavailable":
        director._select_output_style = None
    else:
        def refuse(_style):
            raise ValueError("Synthetic private reason must not be emitted")
        director._select_output_style = refuse
    with pytest.raises(error):
        await request(app, credential.token, b'{"style":"concise"}')
    assert selection.get() == "default"
    assert capsys.readouterr().out.splitlines() == [
        '[output-style] ' + json.dumps(
            {"accepted": False, "category": category, "version": 1}, separators=(",", ":"),
        )
    ]
    director._active_identity, director._active_generation, director._select_output_style = (
        identity, generation, selector,
    )
    await director.stop(participant_identity=credential.participant_identity)
