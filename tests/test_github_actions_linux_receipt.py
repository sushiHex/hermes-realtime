import hashlib
import io
import json
import os
import ssl
import traceback
import urllib.request
import zipfile
from dataclasses import replace
from email.message import Message
from pathlib import Path
from runpy import run_path
from urllib.error import HTTPError

import pytest


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload() -> bytes:
    value = {
        "version": 1,
        "service": {"repositoryId": 1351392603, "workflowId": 345893071, "runId": 42, "attempt": 1},
        "candidate": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "sourceArchiveSha256": "c" * 64,
            "workflowSha256": hashlib.sha256(b"name: release-gates\n").hexdigest(),
        },
        "directWheel": {
            "basename": "hermes_realtime-0.0.3-py3-none-any.whl",
            "sha256": "d" * 64,
            "bytes": 123,
        },
        "linuxWheelhouse": {
            "manifestSha256": "e" * 64,
            "sourceLockSha256": "a" * 64,
            "requirementsSha256": "f" * 64,
            "constraintsSha256": "0" * 64,
            "wheels": [
                {"basename": "aiohttp-3.11.0-py3-none-any.whl", "sha256": "1" * 64, "bytes": 456}
            ],
        },
        "image": {
            "reference": "docker.io/library/python@sha256:" + "2" * 64,
            "configSha256": "3" * 64,
            "layerSha256s": ["4" * 64],
            "layerDiffSha256s": ["5" * 64],
        },
        "runtime": {
            "platform": "linux_x86_64",
            "pythonVersion": "3.11.16",
            "soabi": "cpython-311-x86_64-linux-gnu",
            "extSuffix": ".cpython-311-x86_64-linux-gnu.so",
            "multiarch": "x86_64-linux-gnu",
            "glibc": "2.36",
            "interpreterSha256": "6" * 64,
            "stdlibInventorySha256": "7" * 64,
        },
        "installation": {
            "installedInventorySha256": "8" * 64,
            "installedFileCount": 9,
            "importOriginCount": 10,
            "nullCapturePassed": True,
            "cleanupRemoved": True,
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _zip(payload: bytes) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("linux-null-capture-receipt-v1.json", payload)
    return result.getvalue()


class _Transport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, headers):
        self.calls.append((url, headers))
        return self.responses[url]


@pytest.fixture
def facts():
    from scripts.github_actions_linux_receipt import _LinuxReceiptFacts

    return _LinuxReceiptFacts(
        candidate_commit="a" * 40,
        candidate_tree="b" * 40,
        source_archive_sha256="c" * 64,
        workflow_bytes=b"name: release-gates\n",
        direct_wheel_sha256="d" * 64,
        direct_wheel_bytes=123,
        direct_wheel_basename="hermes_realtime-0.0.3-py3-none-any.whl",
        wheelhouse_manifest_sha256="e" * 64,
        source_lock_sha256="a" * 64,
        requirements_sha256="f" * 64,
        constraints_sha256="0" * 64,
        wheels=(("aiohttp-3.11.0-py3-none-any.whl", "1" * 64, 456),),
        image_reference="docker.io/library/python@sha256:" + "2" * 64,
        image_config_sha256="3" * 64,
        image_layers=("4" * 64,),
        image_diff_ids=("5" * 64,),
        image_python_version="3.11.16",
    )


def _responses(payload):
    from scripts import github_actions_linux_receipt as receipts

    archive = _zip(payload)
    base = receipts._API + "/repos/sushiHex/hermes-realtime"
    return {
        base + "/actions/runs/42/attempts/1": {
            "id": 42,
            "run_attempt": 1,
            "workflow_id": 345893071,
            "repository": {"id": 1351392603},
            "head_sha": "a" * 40,
            "path": ".github/workflows/release-gates.yml",
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
        },
        base + "/contents/.github/workflows/release-gates.yml?ref=" + "a" * 40: {
            "type": "file",
            "path": ".github/workflows/release-gates.yml",
            "encoding": "base64",
            "content": "bmFtZTogcmVsZWFzZS1nYXRlcwo=",
            "size": 20,
        },
        base + "/actions/runs/42/attempts/1/jobs?per_page=100": {
            "total_count": 1,
            "jobs": [
                {
                    "name": "Linux null capture",
                    "head_sha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "labels": ["ubuntu-24.04"],
                    "started_at": "2026-01-01T00:00:01Z",
                    "completed_at": "2026-01-01T00:00:10Z",
                }
            ],
        },
        base + "/actions/runs/42/artifacts?per_page=100": {
            "total_count": 1,
            "artifacts": [
                {
                    "id": 99,
                    "name": "hermes-realtime-linux-null-capture-receipt-v1",
                    "expired": False,
                    "digest": "sha256:" + _sha(archive),
                    "created_at": "2026-01-01T00:00:02Z",
                    "updated_at": "2026-01-01T00:00:03Z",
                    "workflow_run": {
                        "id": 42,
                        "head_sha": "a" * 40,
                        "repository_id": 1351392603,
                        "head_repository_id": 1351392603,
                    },
                }
            ],
        },
        base + "/actions/artifacts/99/zip": archive,
    }


def test_verifier_mints_only_after_exact_service_observations_and_archive_binding(
    facts, monkeypatch
):
    from scripts.github_actions_linux_receipt import (
        _linux_receipt_payload_for_consumer,
        authenticate_linux_receipt,
        linux_receipt_metadata,
    )

    transport = _Transport(_responses(_payload()))
    monkeypatch.setattr(
        "scripts.github_actions_linux_receipt._facts_from_capabilities", lambda *args: facts
    )
    monkeypatch.setattr(
        "scripts.github_actions_linux_receipt._https_transport", lambda *_: transport
    )
    receipt = authenticate_linux_receipt(42, object(), object())

    assert linux_receipt_metadata(receipt).run_id == 42
    payload = _linux_receipt_payload_for_consumer(receipt)
    assert payload.interpreter_sha256 == "6" * 64
    assert payload.stdlib_inventory_sha256 == "7" * 64
    assert payload.installed_inventory_sha256 == "8" * 64
    assert payload.null_capture_passed is True
    assert all(call[1].get("Authorization") is None for call in transport.calls)


@pytest.mark.parametrize("fault", ["attempt", "job", "artifact", "zip", "payload", "workflow"])
def test_any_service_or_payload_mismatch_refuses(facts, fault, monkeypatch):
    from scripts.github_actions_linux_receipt import authenticate_linux_receipt

    responses = _responses(_payload())
    base = "https://api.github.com/repos/sushiHex/hermes-realtime"
    if fault == "attempt":
        responses[base + "/actions/runs/42/attempts/1"]["run_attempt"] = 2
    elif fault == "job":
        responses[base + "/actions/runs/42/attempts/1/jobs?per_page=100"]["jobs"][0]["labels"] = [
            "ubuntu-24.04",
            "ARM64",
        ]
    elif fault == "artifact":
        responses[base + "/actions/runs/42/artifacts?per_page=100"]["artifacts"][0][
            "updated_at"
        ] = "2026-01-01T00:02:00Z"
    elif fault == "zip":
        responses[base + "/actions/artifacts/99/zip"] = b"foreign"
    elif fault == "payload":
        value = json.loads(_payload())
        value["directWheel"]["sha256"] = "9" * 64
        responses = _responses(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
    else:
        facts = replace(facts, workflow_bytes=b"foreign workflow\n")
    transport = _Transport(responses)
    monkeypatch.setattr(
        "scripts.github_actions_linux_receipt._facts_from_capabilities", lambda *args: facts
    )
    monkeypatch.setattr(
        "scripts.github_actions_linux_receipt._https_transport", lambda *_: transport
    )
    with pytest.raises(ValueError):
        authenticate_linux_receipt(42, object(), object())


def test_public_api_does_not_accept_a_caller_success_dictionary():
    from scripts.github_actions_linux_receipt import AuthenticatedLinuxNullCaptureReceiptV1

    with pytest.raises(TypeError):
        AuthenticatedLinuxNullCaptureReceiptV1()


def test_receipt_metadata_rechecks_the_retained_input_authority(facts, monkeypatch):
    from scripts.github_actions_linux_receipt import (
        authenticate_linux_receipt,
        linux_receipt_metadata,
    )

    transport = _Transport(_responses(_payload()))
    observations = iter((facts, replace(facts, direct_wheel_sha256="9" * 64)))
    monkeypatch.setattr(
        "scripts.github_actions_linux_receipt._facts_from_capabilities",
        lambda *args: next(observations),
    )
    monkeypatch.setattr(
        "scripts.github_actions_linux_receipt._https_transport", lambda *_: transport
    )
    receipt = authenticate_linux_receipt(42, object(), object())

    with pytest.raises(ValueError, match="source authority"):
        linux_receipt_metadata(receipt)


def test_artifact_redirect_drops_the_api_bearer(monkeypatch):
    from scripts.github_actions_linux_receipt import _HttpsTransport

    class _Response:
        url = "https://artifactcache.actions.githubusercontent.com/object"
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return b"receipt"

    seen = []

    class _Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, dict(request.headers), timeout))
            if len(seen) == 1:
                headers = Message()
                headers["Location"] = _Response.url
                raise HTTPError(request.full_url, 302, "redirect", headers, None)
            return _Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
    raw = _HttpsTransport("existing-read-token").get(
        "https://api.github.com/repos/sushiHex/hermes-realtime/actions/artifacts/99/zip",
        {"Accept": "application/vnd.github+json"},
    )

    assert raw == b"receipt"
    assert seen[0][1]["Authorization"] == "Bearer existing-read-token"
    assert "Authorization" not in seen[1][1]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SSL_CERT_FILE", "controlled-unbound-trust"),
        ("SSL_CERT_FILE", ""),
        ("SSL_CERT_DIR", "controlled-unbound-trust"),
        ("SSL_CERT_DIR", ""),
        ("SSLKEYLOGFILE", "controlled-unbound-key-log"),
        ("SSLKEYLOGFILE", ""),
    ],
)
def test_transport_refuses_ambient_trust_store_before_opener_creation(
    name, value, monkeypatch
):
    from scripts.github_actions_linux_receipt import _HttpsTransport

    created = []
    for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(ssl, "create_default_context", lambda: created.append("context"))
    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *handlers: created.append("opener")
    )

    with pytest.raises(ValueError, match="ambient TLS"):
        _HttpsTransport("not-a-real-token")
    assert created == []


def test_transport_refuses_key_log_before_native_context_touches_path(tmp_path, monkeypatch):
    from scripts.github_actions_linux_receipt import _HttpsTransport

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    marker = tmp_path / "unbound-tls-keys.log"
    monkeypatch.setenv("SSLKEYLOGFILE", str(marker))
    refused = False
    try:
        _HttpsTransport(None)
    except ValueError:
        refused = True
    assert (refused, marker.exists()) == (True, False)


def test_transport_constructs_one_explicit_no_proxy_verified_opener(monkeypatch):
    from scripts.github_actions_linux_receipt import _HttpsTransport, _NoRedirect

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)

    def refuse_proxy_discovery():
        raise AssertionError("ambient proxy discovery ran")

    contexts = []
    create_default_context = ssl.create_default_context
    build_opener = urllib.request.build_opener
    opener_arguments = []

    def create_context():
        context = create_default_context()
        contexts.append(context)
        return context

    def build(*handlers):
        opener_arguments.append(handlers)
        return build_opener(*handlers)

    monkeypatch.setattr(urllib.request, "getproxies", refuse_proxy_discovery)
    monkeypatch.setattr(ssl, "create_default_context", create_context)
    monkeypatch.setattr(urllib.request, "build_opener", build)
    transport = _HttpsTransport(None)
    assert len(opener_arguments) == 1
    proxies = [
        handler
        for handler in opener_arguments[0]
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    https = [
        handler
        for handler in transport._opener.handlers
        if isinstance(handler, urllib.request.HTTPSHandler)
    ]
    redirects = [
        handler for handler in transport._opener.handlers if isinstance(handler, _NoRedirect)
    ]
    assert len(proxies) == len(https) == len(redirects) == 1
    assert len(contexts) == 1 and https[0]._context is contexts[0]
    assert proxies[0].proxies == {}
    assert isinstance(https[0]._context, ssl.SSLContext)
    assert https[0]._context.verify_mode == ssl.CERT_REQUIRED
    assert https[0]._context.check_hostname is True


def test_transport_suppresses_context_failure_details_before_opener_creation(monkeypatch):
    from scripts.github_actions_linux_receipt import _HttpsTransport

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    created = []

    def refuse_context():
        raise OSError("private trust location")

    monkeypatch.setattr(ssl, "create_default_context", refuse_context)
    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: created.append(handlers))
    token = "not-a-real-token"
    with pytest.raises(ValueError) as raised:
        _HttpsTransport(token)
    rendered = "".join(traceback.format_exception(raised.type, raised.value, raised.tb))
    assert "private trust location" not in rendered
    assert "not-a-real-token" not in rendered
    assert created == []


def test_transport_explicit_context_preserves_urllib_http11_alpn(monkeypatch):
    from scripts.github_actions_linux_receipt import _HttpsTransport

    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE"):
        monkeypatch.delenv(name, raising=False)

    class _Context:
        verify_mode = ssl.CERT_REQUIRED
        check_hostname = True

        def __init__(self):
            self.protocols = []

        def set_alpn_protocols(self, protocols):
            self.protocols.append(protocols)

    context = _Context()
    monkeypatch.setattr(ssl, "create_default_context", lambda: context)
    monkeypatch.setattr("urllib.request.build_opener", lambda *_: object())
    _HttpsTransport(None)
    assert context.protocols == [["http/1.1"]]


def test_signed_storage_failure_suppresses_signed_url_and_credential(monkeypatch):
    from scripts.github_actions_linux_receipt import _HttpsTransport

    signed = "https://artifactcache.actions.githubusercontent.com/object?sig=private-signed-value"
    seen = []

    class _Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, dict(request.headers), timeout))
            headers = Message()
            if len(seen) == 1:
                headers["Location"] = signed
                raise HTTPError(request.full_url, 302, "redirect", headers, None)
            raise HTTPError(request.full_url, 500, "storage failure", headers, None)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
    token = "existing-read-token"
    with pytest.raises(ValueError) as raised:
        _HttpsTransport(token).get(
            "https://api.github.com/repos/sushiHex/hermes-realtime/actions/artifacts/99/zip",
            {"Accept": "application/vnd.github+json"},
        )

    rendered = "".join(traceback.format_exception(raised.type, raised.value, raised.tb))
    assert "private-signed-value" not in rendered
    assert "existing-read-token" not in rendered
    assert "Authorization" not in seen[1][1]


@pytest.mark.skipif(os.name != "nt", reason="genuine retained input authority needs Windows")
def test_genuine_linux_dependency_capability_supplies_archived_workflow_facts(
    tmp_path, tmp_path_factory, monkeypatch
):
    """Source, dependency, and image capabilities supply the consumer's facts."""
    from scripts import qualification_linux_image as images
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
    from scripts.github_actions_linux_receipt import _facts_from_capabilities
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_purpose

    helpers = run_path(str(Path.cwd() / "tests/test_qualification_candidate_files.py"))
    source = helpers["source"].__wrapped__(tmp_path_factory)
    repository, _, prior_identity, wheel = source
    workflow = repository / ".github/workflows/release-gates.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_bytes(b"name: release-gates\n")
    archive_helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
    archive_helpers["_git"]("add", ".", cwd=repository)
    archive_helpers["_git"]("commit", "-qm", "add workflow", cwd=repository)
    identity = archive_helpers["_identity"](repository, prior_identity.canonical_baseline_oid)
    archive = capture_candidate_source_archive(repository, identity, archive_helpers["_pin"]())
    source = repository, archive, identity, wheel
    graph = helpers["bound_graph"].__wrapped__(tmp_path, source)
    dependency_graph = helpers["dependency_graph"].__wrapped__(graph)
    _, _, _, archive, identity = dependency_graph
    config = json.dumps(
        {
            "architecture": "amd64",
            "os": "linux",
            "config": {"Env": ["PYTHON_VERSION=3.11.16"]},
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "d" * 64]},
        }
    ).encode()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": "sha256:" + _sha(config),
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": "sha256:" + "c" * 64,
                    "size": 10,
                }
            ],
        }
    ).encode()
    monkeypatch.setattr(
        images,
        "_POLICY",
        replace(
            images._POLICY,
            manifest_sha256=_sha(manifest),
            manifest_bytes=len(manifest),
        ),
    )
    image = images.admit_linux_runtime_image(manifest, config)
    with helpers["freeze"](dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        runtime = bind_dependency_purpose(files, candidate, purpose="realtime_linux_runtime")
        facts = _facts_from_capabilities(runtime, image)

    assert facts.workflow_bytes == b"name: release-gates\n"
    assert facts.candidate_commit == identity.candidate_head_oid
