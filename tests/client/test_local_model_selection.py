from __future__ import annotations

import asyncio
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


def bootstrap_body(selection):
    return json.dumps({"version": 1, "localModelSelection": selection}).encode()


async def request(app, token, path, body=b"", *, bind_choice=True):
    if (
        path == "/api/v1/bootstrap"
        and not body
        and bind_choice
        and app._local_model_selection is not None
    ):
        body = bootstrap_body(app._local_model_selection._selection)
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
async def test_connected_or_network_dropped_session_cannot_reprepare():
    owner = selection_owner()
    app, director, cap, _, _ = make_application(owner=owner)
    await owner.prepare("small", "a" * 64, 4096)
    await director.start(expected_local_selection=owner._selection)
    # Loss of the browser's socket never clears the server's session ownership.
    with pytest.raises(RuntimeError, match="active"):
        await request(
            app,
            cap.token,
            "/api/v1/local-model-prepare",
            preparation_body(requestedContextTokens=2048),
        )
    assert owner.context_observation().configuration.requested_context_tokens == 4096
    await director.stop(participant_identity=director.active_identity)
    result = await request(
        app, cap.token, "/api/v1/local-model-prepare", preparation_body(requestedContextTokens=2048)
    )
    assert json.loads(result.body)["selection"]["requestedContextTokens"] == 2048
    await owner.close()


@pytest.mark.asyncio
async def test_preparation_blocks_concurrent_start_until_exact_swap(monkeypatch):
    owner = selection_owner()
    app, director, cap, issued, _ = make_application(owner=owner)
    await owner.prepare("small", "a" * 64, 4096)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def readiness(adapter):
        entered.set()
        await release.wait()

    owner._readiness = readiness
    preparation = asyncio.create_task(
        request(
            app,
            cap.token,
            "/api/v1/local-model-prepare",
            preparation_body(requestedContextTokens=2048),
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    start = asyncio.create_task(
        director.start(
            expected_local_selection={
                **owner._selection,
                "requestedContextTokens": 2048,
            }
        )
    )
    await asyncio.sleep(0)
    assert issued == [] and not start.done()
    release.set()
    await preparation
    await start
    assert director._model_configuration.public_data()["requestedContextTokens"] == 2048
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["finalizing", "failed"])
async def test_preparation_refuses_incomplete_session_recovery(state):
    owner = selection_owner()
    app, director, cap, issued, _ = make_application(owner=owner)
    if state == "finalizing":
        director._rebind_finalizer = asyncio.create_task(asyncio.sleep(0))
    else:
        director._rebind_finalization_failed = True
    with pytest.raises(RuntimeError, match="recovery"):
        await request(app, cap.token, "/api/v1/local-model-prepare", preparation_body())
    assert issued == [] and owner._adapter is None
    if director._rebind_finalizer is not None:
        await director._rebind_finalizer
    await owner.close()


@pytest.mark.asyncio
async def test_session_lock_wait_is_inside_preparation_deadline(monkeypatch, capsys):
    import hermes_realtime.providers.ollama_session as module

    monkeypatch.setattr(module, "PREPARATION_TIMEOUT_SECONDS", 0.01)
    owner = selection_owner()
    app, director, cap, issued, _ = make_application(owner=owner)
    async with director._start_lock:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                request(app, cap.token, "/api/v1/local-model-prepare", preparation_body()), 0.2
            )
    assert issued == [] and owner._adapter is None
    assert '"category":"deadline"' in capsys.readouterr().out
    await owner.close()


@pytest.mark.asyncio
async def test_stale_prepared_tab_cannot_bootstrap_a_different_choice_or_burn_capability():
    owner = selection_owner()
    app, director, cap, issued, provisioned = make_application(owner=owner)
    first = await owner.prepare("small", "a" * 64, 4096)
    second = await owner.prepare("small", "a" * 64, 2048)
    with pytest.raises(ValueError, match="selection"):
        await request(app, cap.token, "/api/v1/bootstrap", bootstrap_body(first))
    cap.validate(cap.token)
    assert issued == provisioned == []
    result = await request(app, cap.token, "/api/v1/bootstrap", bootstrap_body(second))
    assert result.status == 200 and len(issued) == len(provisioned) == 1
    await owner.close()


@pytest.mark.asyncio
async def test_queued_stale_bootstrap_does_not_consume_capability_before_choice_binding():
    owner = selection_owner()
    app, director, cap, issued, provisioned = make_application(owner=owner)
    first = await owner.prepare("small", "a" * 64, 4096)
    entered, release = asyncio.Event(), asyncio.Event()

    async def readiness(adapter):
        entered.set()
        await release.wait()

    owner._readiness = readiness
    replacing = asyncio.create_task(director.prepare_local_model("small", "a" * 64, 2048))
    await entered.wait()
    joining = asyncio.create_task(
        request(app, cap.token, "/api/v1/bootstrap", bootstrap_body(first))
    )
    await asyncio.sleep(0)
    cap.validate(cap.token)
    assert not joining.done()
    release.set()
    await replacing
    with pytest.raises(ValueError, match="selection"):
        await joining
    cap.validate(cap.token)
    assert issued == provisioned == []
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "model",
        "digest",
        "requestedContextTokens",
        "advertisedContextTokens",
        "effectiveContextTokens",
    ],
)
async def test_bootstrap_requires_full_prepared_selection_equality(field):
    owner = selection_owner()
    app, _, cap, issued, provisioned = make_application(owner=owner)
    selected = await owner.prepare("small", "a" * 64, 4096)
    changed = {
        "model": "alias",
        "digest": "b" * 64,
        "requestedContextTokens": 2048,
        "advertisedContextTokens": 8192,
        "effectiveContextTokens": None,
    }
    with pytest.raises(ValueError, match="selection"):
        await request(
            app,
            cap.token,
            "/api/v1/bootstrap",
            bootstrap_body(
                {
                    **selected,
                    field: changed[field],
                }
            ),
        )
    cap.validate(cap.token)
    assert issued == provisioned == []
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"",
        b'["version","localModelSelection"]',
        b'{"version":true,"localModelSelection":{}}',
        b'{"version":2,"localModelSelection":{}}',
        b'{"version":1,"localModelSelection":{},"extra":0}',
        b'{"version":1,"version":1,"localModelSelection":{}}',
        b" " * 1025,
        b'{"version":1,"localModelSelection":null}',
    ],
    ids=[
        "empty", "array", "boolean-version", "wrong-version", "extra-field", "duplicate",
        "oversized", "null-choice",
    ],
)
async def test_local_bootstrap_refuses_legacy_or_malformed_choice_without_consumption(body):
    owner = selection_owner()
    app, _, cap, issued, provisioned = make_application(owner=owner)
    selection = await owner.prepare("small", "a" * 64, 4096)
    valid = bootstrap_body(selection)
    if b'"version":true' in body:
        body = valid.replace(b'"version": 1', b'"version": true')
    elif b'"version":2' in body:
        body = valid.replace(b'"version": 1', b'"version": 2')
    elif b'"extra"' in body:
        body = json.dumps({"version": 1, "localModelSelection": selection, "extra": 0}).encode()
    elif len(body) > 1024:
        body = valid + b" " * (1025 - len(valid))
    with pytest.raises(ValueError):
        await request(app, cap.token, "/api/v1/bootstrap", body, bind_choice=False)
    cap.validate(cap.token)
    assert issued == provisioned == []
    await owner.close()


@pytest.mark.asyncio
async def test_bootstrap_lock_wait_expires_without_issuing_or_consuming(capsys):
    owner = selection_owner()
    app, director, cap, issued, provisioned = make_application(owner=owner)
    await owner.prepare("small", "a" * 64, 4096)
    director._rollback_timeout = 0.01
    async with director._start_lock:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(request(app, cap.token, "/api/v1/bootstrap"), 0.2)
    cap.validate(cap.token)
    assert issued == provisioned == []
    assert '[browser-start-lock] {"version":1,"category":"deadline"}' in capsys.readouterr().out
    await owner.close()


@pytest.mark.asyncio
async def test_bootstrap_cannot_accept_an_impostor_capability(capsys):
    _, director, _, issued, provisioned = make_application()

    class Impostor:
        def consume(self, bearer):
            return None

    with pytest.raises(TypeError, match="exact"):
        await director.start(bootstrap_capability=Impostor(), bootstrap_bearer="a" * 43)
    assert issued == provisioned == []
    assert '[bootstrap-capability] {"version":1,"category":"refused"}' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_successful_start_does_not_call_the_refusal_marker_writer(monkeypatch):
    import hermes_realtime.client.session as module

    _, director, _, _, _ = make_application()

    def fail(*args, **kwargs):
        raise OSError("synthetic marker write refusal")

    monkeypatch.setattr(module, "print", fail, raising=False)
    await director.start()
    assert not director._start_lock.locked()
    assert director.active_generation == 1


@pytest.mark.asyncio
async def test_failed_start_lock_keeps_foreign_ownership_if_its_refusal_writer_fails(monkeypatch):
    import hermes_realtime.client.session as module

    _, director, _, issued, provisioned = make_application()
    director._rollback_timeout = 0.01

    def fail(*args, **kwargs):
        raise OSError("synthetic marker write refusal")

    monkeypatch.setattr(module, "print", fail, raising=False)
    async with director._start_lock:
        with pytest.raises(OSError, match="marker"):
            await director.start()
        assert director._start_lock.locked()
    assert issued == provisioned == []
    assert not director._start_lock.locked()


@pytest.mark.asyncio
async def test_local_bootstrap_requires_json_content_type_for_valid_choice():
    owner = selection_owner()
    app, _, cap, issued, provisioned = make_application(owner=owner)
    selected = await owner.prepare("small", "a" * 64, 4096)
    body = bootstrap_body(selected)
    with pytest.raises(ValueError, match="JSON"):
        await app.handle(
            method="POST",
            path="/api/v1/bootstrap",
            body=body,
            headers={
                "origin": "http://127.0.0.1:8765",
                "authorization": f"Bearer {cap.token}",
                "content-length": str(len(body)),
                "content-type": "text/plain",
            },
        )
    assert issued == provisioned == []
    cap.validate(cap.token)
    await owner.close()


@pytest.mark.asyncio
async def test_expected_selection_rejects_forged_dictionary_equality_and_missing_fields():
    owner = selection_owner()
    selected = await owner.prepare("small", "a" * 64, 4096)

    class Forged(dict):
        def __ne__(self, other):
            return False

    with pytest.raises(ValueError, match="exact fields"):
        owner.require_choice(Forged({**selected, "model": "alias"}))
    with pytest.raises(ValueError, match="exact fields"):
        owner.require_choice({key: value for key, value in selected.items() if key != "model"})
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
        __self__ = owner

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


def test_http_owner_cannot_bind_another_method_on_the_same_owner():
    owner = selection_owner()
    with pytest.raises(ValueError, match="binding"):
        make_application(owner=owner, admission=owner.context_observation)


@pytest.mark.asyncio
async def test_direct_preparation_refuses_an_impostor_bound_owner():
    class OtherOwner:
        def require_prepared(self):
            return None

        async def prepare(self, *args):
            pytest.fail("impostor owner was dispatched")

    owner = OtherOwner()
    _, director, _, _, _ = make_application(admission=owner.require_prepared)
    with pytest.raises(ValueError, match="binding"):
        await director.prepare_local_model("small", "a" * 64, 4096)
