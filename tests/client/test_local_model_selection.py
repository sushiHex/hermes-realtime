from __future__ import annotations

import json

import pytest

from hermes_realtime.client.bootstrap import (
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    OneTimeBootstrapCapability,
)
from hermes_realtime.client.http import BrowserBootstrapApplication
from hermes_realtime.client.projection import BrowserEventProjection
from hermes_realtime.client.session import BrowserSessionDirector
from hermes_realtime.livekit import LiveKitConnection
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import OllamaModelClient
from hermes_realtime.providers.ollama_session import OllamaSessionSelection


def test_capability_checks_do_not_consume_or_extend_expiry():
    now = [0.0]
    cap = OneTimeBootstrapCapability(token_factory=lambda: "a" * 43, clock=lambda: now[0])
    cap.validate(cap.token)
    cap.validate(cap.token)
    with pytest.raises(PermissionError):
        cap.validate("b" * 43)
    now[0] = 120.0
    with pytest.raises(RuntimeError, match="expired"):
        cap.validate(cap.token)


def make_application(*, admission=None, owner=None, bind_owner=True):
    if bind_owner and type(owner) is OllamaSessionSelection and admission is None:
        admission = owner.require_prepared
    issued = []
    provisioned = []
    connection = LiveKitConnection("wss://livekit.test", "key", "synthetic-long-enough-test-secret")

    async def provision(identity):
        provisioned.append(identity)
        return 1

    async def noop(*args):
        pass

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="test",
            identity_factory=lambda: issued.append(True) or "browser_" + "a" * 16,
        ),
        provision=provision,
        submit=noop,
        stop=noop,
        approval=noop,
        projection=BrowserEventProjection(),
        local_model_configuration=admission,
    )
    cap = OneTimeBootstrapCapability(token_factory=lambda: "a" * 43)
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="test"),
        capability=cap,
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_selection",
        local_model_selection=owner,
    )
    return app, director, cap, issued, provisioned


async def request(app, token, path, body=b""):
    return await app.handle(
        method="POST",
        path=path,
        body=body,
        headers={
            "origin": "http://127.0.0.1:8765",
            "authorization": f"Bearer {token}",
            "content-length": str(len(body)),
            **({"content-type": "application/json"} if body else {}),
        },
    )


@pytest.mark.asyncio
async def test_disabled_detection_is_authenticated_and_nonconsuming():
    app, _, cap, issued, _ = make_application()
    with pytest.raises(PermissionError):
        await request(app, "b" * 43, "/api/v1/local-models")
    result = await request(app, cap.token, "/api/v1/local-models")
    assert json.loads(result.body) == {"version": 1, "available": False}
    cap.consume(cap.token)
    assert issued == []


@pytest.mark.asyncio
async def test_unprepared_admission_precedes_capability_and_credentials():
    def refused():
        raise RuntimeError("not prepared")

    app, director, cap, issued, provisioned = make_application(admission=refused)
    with pytest.raises(RuntimeError, match="prepared"):
        await request(app, cap.token, "/api/v1/bootstrap")
    cap.validate(cap.token)
    with pytest.raises(RuntimeError, match="prepared"):
        await director.start()
    assert issued == provisioned == []


def selection_owner():
    async def read(endpoint, body):
        if endpoint == "version":
            payload = {"version": "0.34.4"}
        elif endpoint == "show":
            payload = {
                "capabilities": ["completion"],
                "model_info": {
                    "general.architecture": "generic",
                    "generic.context_length": 4096,
                },
            }
        else:
            payload = {"models": [{"name": "small", "digest": "a" * 64, "context_length": 4096}]}
        return json.dumps(payload).encode()

    async def ready(adapter):
        pass

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    return OllamaSessionSelection(
        client=client,
        default_model="small",
        readiness=ready,
        adapter_factory=lambda model: OllamaStreamingInference(
            base_url="http://127.0.0.1:11434",
            model=model,
            metadata_client=client,
        ),
    )


def preparation_body(**overrides):
    return json.dumps(
        {
            "version": 1,
            "model": "small",
            "digest": "a" * 64,
            "requestedContextTokens": 4096,
            **overrides,
        }
    ).encode()


@pytest.mark.asyncio
async def test_prepare_issues_no_credentials_and_bootstrap_publishes_frozen_context():
    owner = selection_owner()
    app, _, cap, issued, provisioned = make_application(
        admission=owner.require_prepared, owner=owner
    )
    catalog = json.loads((await request(app, cap.token, "/api/v1/local-models")).body)
    assert catalog["available"] is True and catalog["defaultRequestedContextTokens"] == 16384
    reply = json.loads(
        (
            await request(
                app,
                cap.token,
                "/api/v1/local-model-prepare",
                preparation_body(),
            )
        ).body
    )
    assert reply["selection"]["effectiveContextTokens"] == 4096
    assert issued == provisioned == []
    result = await request(app, cap.token, "/api/v1/bootstrap")
    assert result.status == 200 and len(issued) == len(provisioned) == 1
    with pytest.raises(RuntimeError, match="consumed"):
        cap.validate(cap.token)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        preparation_body(version=True),
        preparation_body(version=2),
        preparation_body(extra=True),
        preparation_body(model="small\n"),
        preparation_body(requestedContextTokens=True),
        preparation_body() + b" " * (1025 - len(preparation_body())),
        preparation_body().replace(b'"version": 1', b'"version": 1, "version": 1'),
    ],
    ids=[
        "array",
        "boolean-version",
        "wrong-version",
        "extra",
        "control-model",
        "boolean-context",
        "oversized",
        "duplicate",
    ],
)
async def test_prepare_refuses_strict_body_without_consuming_or_issuing(body, capsys):
    owner = selection_owner()
    app, _, cap, issued, provisioned = make_application(
        admission=owner.require_prepared, owner=owner
    )
    with pytest.raises((TypeError, ValueError)):
        await request(app, cap.token, "/api/v1/local-model-prepare", body)
    cap.validate(cap.token)
    assert issued == provisioned == []
    markers = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[local-model-http] ")
    ]
    assert markers == ['[local-model-http] {"version":1,"kind":"prepare","category":"refused"}']
    await owner.close()


@pytest.mark.asyncio
async def test_director_rejects_wrong_configuration_before_issue():
    _, director, _, issued, provisioned = make_application(admission=lambda: object())
    with pytest.raises(TypeError, match="exact model configuration"):
        await director.start()
    assert issued == provisioned == []


def test_http_owner_must_be_exact_selection_owner():
    with pytest.raises(TypeError, match="exact"):
        make_application(owner=object())


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["body", "content-type"])
async def test_catalog_requires_empty_untyped_body(field):
    app, _, cap, _, _ = make_application()
    body = b"x" if field == "body" else b""
    headers = {
        "origin": "http://127.0.0.1:8765",
        "authorization": f"Bearer {cap.token}",
        "content-length": str(len(body)),
    }
    if field == "content-type":
        headers[field] = "application/json"
    with pytest.raises(ValueError, match="body"):
        await app.handle(method="POST", path="/api/v1/local-models", body=body, headers=headers)


@pytest.mark.asyncio
async def test_preparation_requires_json_content_type():
    owner = selection_owner()
    app, _, cap, _, _ = make_application(owner=owner)
    body = preparation_body()
    with pytest.raises(ValueError, match="JSON"):
        await app.handle(
            method="POST",
            path="/api/v1/local-model-prepare",
            body=body,
            headers={
                "origin": "http://127.0.0.1:8765",
                "authorization": f"Bearer {cap.token}",
                "content-length": str(len(body)),
                "content-type": "text/plain",
            },
        )
    await owner.close()


@pytest.mark.parametrize("impostor", [False, True])
def test_http_owner_must_match_exact_director_admission(impostor):
    owner = selection_owner()

    class FalseAuthority:
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __call__(self):
            raise AssertionError("unbound authority accepted")

    with pytest.raises(ValueError, match="binding"):
        make_application(
            owner=owner, bind_owner=False, admission=FalseAuthority() if impostor else None
        )


def test_http_owner_cannot_use_another_owners_bound_admission():
    owner, other = selection_owner(), selection_owner()
    with pytest.raises(ValueError, match="binding"):
        make_application(owner=owner, admission=other.require_prepared)
