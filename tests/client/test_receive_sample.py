"""The browser's content-free receive-stats sample around a speech chunk end (#556)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from hermes_realtime.client import (
    BrowserBootstrapApplication,
    BrowserEventProjection,
    BrowserSessionDirector,
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    OneTimeBootstrapCapability,
)
from hermes_realtime.client.session import BrowserReceiveSample
from hermes_realtime.livekit import LiveKitConnection

_ORIGIN = "https://phone.test:8443"
_FULL: dict[str, Any] = {
    "windowMs": 2_480,
    "removedSamplesForAcceleration": 480,
    "insertedSamplesForDeceleration": 0,
    "concealedSamples": 240,
    "silentConcealedSamples": 60,
    "concealmentEvents": 2,
    "meanTargetDelayMs": 60,
    "duckActive": True,
    "audioContext": "running",
    "pageUptimeS": 3_600,
    "roomUptimeS": 1_200,
}


async def _ignored(*args: object) -> None:
    del args


async def _application() -> tuple[BrowserBootstrapApplication, str]:
    connection = LiveKitConnection(
        "wss://livekit.test", "test-key", "synthetic-browser-bootstrap-secret-32-bytes"
    )

    async def provision(identity: str) -> int:
        del identity
        return 3

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_ignored,
        stop=_ignored,
        approval=_ignored,
        projection=BrowserEventProjection(),
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "c" * 43),
        allowed_origin=_ORIGIN,
        worker_identity="worker_hermes_browser",
    )
    return app, credential.token


async def _post(app: BrowserBootstrapApplication, token: str, receive: object) -> int:
    body = json.dumps({"version": 1, "receive": receive}).encode()
    response = await app.handle(
        method="POST",
        path="/api/v1/audio-diagnostic",
        headers={
            "authorization": f"Bearer {token}",
            "content-length": str(len(body)),
            "content-type": "application/json",
            "origin": _ORIGIN,
        },
        body=body,
    )
    return response.status


def _markers(output: str) -> list[dict[str, Any]]:
    prefix = "[browser-receive-sample] "
    return [
        json.loads(line[len(prefix) :]) for line in output.splitlines() if line.startswith(prefix)
    ]


@pytest.mark.asyncio
async def test_receive_sample_prints_one_marker_per_sample_and_omits_absent_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app, token = await _application()
    capsys.readouterr()

    assert await _post(app, token, _FULL) == 202
    minimal = {"windowMs": 500, "duckActive": False, "pageUptimeS": 9, "roomUptimeS": 4}
    assert await _post(app, token, minimal) == 202

    assert _markers(capsys.readouterr().out) == [
        {
            "version": 1,
            "generation": 3,
            "window_ms": 2_480,
            "removed_samples_for_acceleration": 480,
            "inserted_samples_for_deceleration": 0,
            "concealed_samples": 240,
            "silent_concealed_samples": 60,
            "concealment_events": 2,
            "mean_target_delay_ms": 60,
            "duck_active": True,
            "audio_context": "running",
            "page_uptime_s": 3_600,
            "room_uptime_s": 1_200,
        },
        {
            "version": 1,
            "generation": 3,
            "window_ms": 500,
            "duck_active": False,
            "page_uptime_s": 9,
            "room_uptime_s": 4,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"concealedSamples": "240"}, TypeError),
        ({"concealedSamples": 2.5}, TypeError),
        ({"concealedSamples": True}, TypeError),
        ({"duckActive": 1}, TypeError),
        ({"concealedSamples": -1}, ValueError),
        ({"concealedSamples": 1_000_000_001}, ValueError),
        ({"windowMs": 600_001}, ValueError),
        ({"meanTargetDelayMs": 60_001}, ValueError),
        ({"pageUptimeS": 31_536_001}, ValueError),
        ({"audioContext": "playing"}, ValueError),
        ({"audioContext": None}, ValueError),
        ({"trackId": "TR_abc"}, ValueError),
        ({"streamId": "hermes-speech-01234567"}, ValueError),
    ],
)
async def test_receive_sample_rejects_strings_out_of_range_and_unknown_fields(
    change: dict[str, object],
    error: type[Exception],
    capsys: pytest.CaptureFixture[str],
) -> None:
    app, token = await _application()
    capsys.readouterr()
    with pytest.raises(error):
        await _post(app, token, {**_FULL, **change})
    assert _markers(capsys.readouterr().out) == []


@pytest.mark.asyncio
async def test_receive_sample_rejects_a_missing_required_field_and_a_bad_envelope() -> None:
    app, token = await _application()
    for required in ("windowMs", "duckActive", "pageUptimeS", "roomUptimeS"):
        partial = {key: value for key, value in _FULL.items() if key != required}
        with pytest.raises(ValueError):
            await _post(app, token, partial)
    with pytest.raises(ValueError):
        await _post(app, token, [_FULL])
    body = json.dumps({"version": 2, "receive": _FULL}).encode()
    with pytest.raises(ValueError):
        await app.handle(
            method="POST",
            path="/api/v1/audio-diagnostic",
            headers={
                "authorization": f"Bearer {token}",
                "content-length": str(len(body)),
                "content-type": "application/json",
                "origin": _ORIGIN,
            },
            body=body,
        )


@pytest.mark.asyncio
async def test_receive_samples_are_capped_per_binding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app, token = await _application()
    minimal = {"windowMs": 500, "duckActive": False, "pageUptimeS": 9, "roomUptimeS": 4}
    for _ in range(BrowserReceiveSample.MAX_PER_BINDING):
        assert await _post(app, token, minimal) == 202
    capsys.readouterr()
    with pytest.raises(RuntimeError, match="capacity"):
        await _post(app, token, minimal)
    assert _markers(capsys.readouterr().out) == []


def test_receive_sample_validates_exact_types_directly() -> None:
    with pytest.raises(TypeError):
        BrowserReceiveSample(
            window_ms=500,
            duck_active=False,
            page_uptime_s=9,
            room_uptime_s=4,
            audio_context=None,
            removed_samples_for_acceleration=None,
            inserted_samples_for_deceleration=None,
            concealed_samples=None,
            silent_concealed_samples=None,
            concealment_events=False,
            mean_target_delay_ms=None,
        )
