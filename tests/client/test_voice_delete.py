from __future__ import annotations

import json

import pytest

from hermes_realtime.client import (
    BrowserBootstrapApplication,
    BrowserEventProjection,
    BrowserSessionDirector,
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    OneTimeBootstrapCapability,
)
from hermes_realtime.livekit import LiveKitConnection


@pytest.mark.asyncio
async def test_delete_control_requires_current_bearer_and_reports_pending() -> None:
    connection = LiveKitConnection(
        "wss://livekit.test", "test-key", "synthetic-browser-bootstrap-secret-32-bytes"
    )
    observed: list[tuple[str, int]] = []
    state = "idle"

    async def provision(_identity: str) -> int:
        return 3

    async def submit(*_args: object) -> None:
        pass

    async def delete(identity: str, generation: int) -> str:
        nonlocal state
        observed.append((identity, generation))
        state = "pending"
        return "pending"

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=submit,
        stop=submit,
        approval=submit,
        projection=BrowserEventProjection(),
        delete_voice_conversation=delete,
        voice_delete_status=lambda: state,
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "z" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    headers = {
        "authorization": f"Bearer {credential.token}",
        "content-length": "0",
        "origin": "https://phone.test:8443",
    }
    response = await app.handle(
        method="POST", path="/api/v1/delete-voice-conversation", headers=headers, body=b""
    )
    assert response.status == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert json.loads(response.body) == {"state": "pending", "version": 1}
    assert observed == [(credential.participant_identity, 3)]
    assert [event.kind for event in director._projection._events].count(
        "voice_conversation_cleared"
    ) == 1
    again = await app.handle(
        method="POST", path="/api/v1/delete-voice-conversation", headers=headers, body=b""
    )
    assert json.loads(again.body)["state"] == "pending"
    assert len(observed) == 1
    for rejected_headers, rejected_body in (
        (headers | {"content-type": "application/json"}, b""),
        (headers | {"content-length": "2"}, b"{}"),
    ):
        with pytest.raises(ValueError, match="must be empty"):
            await app.handle(
                method="POST", path="/api/v1/delete-voice-conversation",
                headers=rejected_headers, body=rejected_body,
            )
    assert len(observed) == 1
    status_response = await app.handle(
        method="POST", path="/api/v1/voice-delete-status", headers=headers, body=b""
    )
    assert json.loads(status_response.body) == {"state": "pending", "version": 1}
    assert [event.kind for event in director._projection._events].count(
        "voice_conversation_cleared"
    ) == 1
    with pytest.raises(PermissionError):
        await app.handle(
            method="POST",
            path="/api/v1/delete-voice-conversation",
            headers=headers | {"authorization": "Bearer invalid"},
            body=b"",
        )
    assert len(observed) == 1
