import json
from pathlib import Path
from runpy import run_path


def _renderer() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "scripts"
        / "render_livekit_logs.py"
    )
    return run_path(str(path))


def test_livekit_log_renderer_redacts_identifiers_credentials_and_private_urls(
    tmp_path: Path,
) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    jwt = ".".join(("headerpart", "payloadpart", "signaturepart"))
    credential_value = "-".join(("synthetic", "credential", "value"))
    private_url = "wss://" + "private.example/rtc?access_token=" + jwt
    path = tmp_path / "livekit.err"
    path.write_text(
        "\n".join(
            (
                f"level=error room=private-room participant=private-peer token={credential_value}",
                f"failed endpoint={private_url}",
                f"Authorization: Bearer {jwt}",
                "level=error status=500 msg=signal request failed",
            )
        ),
        encoding="utf-8",
    )

    rendered = render_log(path)

    assert "status=500" in rendered
    assert "signal request failed" in rendered
    for prohibited in ("private-room", "private-peer", credential_value, jwt, private_url):
        assert prohibited not in rendered
    assert rendered.count("[REDACTED]") >= 4
    assert "[REDACTED-URL]" in rendered


def test_livekit_log_renderer_redacts_structured_and_escaped_sensitive_forms(
    tmp_path: Path,
) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    short_credential = "-".join(("short", "credential"))
    structured = {
        "level": "error",
        "room": "private-room",
        "participant_identity": "private-peer",
        "nested": {"access_token": short_credential},
        "camel": {
            "roomID": "camel-room",
            "participantIdentity": "camel-peer",
            "accessToken": short_credential,
            "apiKey": short_credential,
            "livekitKeys": short_credential,
        },
        "error": "failed wss://private.example/rtc?access_token=" + short_credential,
    }
    path = tmp_path / "structured.err"
    path.write_text(
        "\n".join(
            (
                json.dumps(structured),
                f"Authorization: Bearer {short_credential}",
                f"Authorization=Bearer {short_credential}",
                f"LIVEKIT_KEYS=synthetic-key: {short_credential}",
                f"LIVEKIT_KEYS: synthetic-key: {short_credential}",
                "escaped=wss:\\/\\/private.example\\/rtc?access_token="
                + short_credential,
            )
        ),
        encoding="utf-8",
    )

    rendered = render_log(path)

    for prohibited in (
        "private-room",
        "private-peer",
        "camel-room",
        "camel-peer",
        short_credential,
        "private.example",
    ):
        assert prohibited not in rendered
    assert '"level":"error"' in rendered
    assert rendered.count("[REDACTED]") >= 4
    assert rendered.count("[REDACTED-URL]") >= 2


def test_livekit_log_renderer_redacts_authority_credentials_and_identifiers(
    tmp_path: Path,
) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    sensitive_lines = (
        "LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secretsecretsecret",
        "Authorization: Basic ZGV2a2V5OnNlY3JldA==",
        "authorization=Basic ZGV2a2V5OnNlY3JldA==",
        "id=521adaa098e88a551f16766f53cb934a",
        "521adaa098e88a551f16766f53cb934a",
        "PA_8cfhXMtrjNqE RM_nQrXkadP2t6u",
    )
    path = tmp_path / "authority.err"
    path.write_text("\n".join(sensitive_lines), encoding="utf-8")

    rendered = render_log(path)

    for value in (
        "devkey",
        "secretsecretsecret",
        "ZGV2a2V5OnNlY3JldA==",
        "521adaa098e88a551f16766f53cb934a",
        "PA_8cfhXMtrjNqE",
        "RM_nQrXkadP2t6u",
    ):
        assert value not in rendered


def test_livekit_log_sanitization_is_idempotent_for_existing_probe_corpus() -> None:
    renderer = _renderer()
    sanitize_text = renderer["_sanitize_text"]
    short_credential = "-".join(("short", "credential"))
    jwt = ".".join(("headerpart", "payloadpart", "signaturepart"))
    probes = (
        "level=error room=private-room participant=private-peer "
        "token=synthetic-credential-value",
        "failed endpoint=wss://private.example/rtc?access_token=" + jwt,
        f"Authorization: Bearer {short_credential}",
        f"Authorization=Bearer {short_credential}",
        f"LIVEKIT_KEYS=synthetic-key: {short_credential}",
        f"LIVEKIT_KEYS: synthetic-key: {short_credential}",
        "escaped=wss:\\/\\/private.example\\/rtc?access_token=" + short_credential,
        "LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secretsecretsecret",
        "Authorization: Basic ZGV2a2V5OnNlY3JldA==",
        "authorization=Basic ZGV2a2V5OnNlY3JldA==",
        "id=521adaa098e88a551f16766f53cb934a",
        "521adaa098e88a551f16766f53cb934a",
        "PA_8cfhXMtrjNqE RM_nQrXkadP2t6u",
        "endpoint=http://127.0.0.1:99999/rtc/v1?access_token=value",
        "endpoint=http://127.0.0.1:7880/rooms/private-room",
        "ordinary diagnostic line",
    )

    for line in probes:
        sanitized = sanitize_text(line)
        assert sanitize_text(sanitized) == sanitized


def test_livekit_log_renderer_emits_only_a_bounded_tail(tmp_path: Path) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    path = tmp_path / "livekit.out"
    path.write_text(
        "\n".join(f"line-{index:04d}-" + ("x " * 100) for index in range(500)),
        encoding="utf-8",
    )

    rendered = render_log(path, max_bytes=4_096, max_lines=12, max_line_chars=80)

    assert "[input truncated to final 4096 bytes]" in rendered
    retained = [line for line in rendered.splitlines() if line.startswith("line-")]
    assert len(retained) == 12
    assert all(len(line) <= 80 for line in retained)
    assert retained[-1].startswith("line-0499-")


def test_livekit_log_renderer_discards_partial_first_tail_line(tmp_path: Path) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    participant = "browser_sensitive_participant"
    exposed_suffix = participant[4:]
    path = tmp_path / "partial.err"
    path.write_text(f"participant={participant}", encoding="utf-8")

    rendered = render_log(path, max_bytes=len(exposed_suffix))

    assert exposed_suffix not in rendered


def test_livekit_log_renderer_handles_missing_and_invalid_bytes(tmp_path: Path) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    missing = tmp_path / "missing.err"
    invalid = tmp_path / "invalid.err"
    invalid.write_bytes(b"level=error msg=bad-byte-\xff\n")
    malformed_url = tmp_path / "malformed-url.err"
    malformed_url.write_text(
        "endpoint=http://127.0.0.1:99999/rtc/v1?access_token=value\n",
        encoding="utf-8",
    )
    private_path = tmp_path / "private-path.err"
    private_path.write_text(
        "endpoint=http://127.0.0.1:7880/rooms/private-room\n",
        encoding="utf-8",
    )

    assert render_log(missing) == "[log unavailable]"
    rendered = render_log(invalid)
    assert "level=error" in rendered
    assert "bad-byte-" in rendered
    assert "[REDACTED-URL]" in render_log(malformed_url)
    path_rendered = render_log(private_path)
    assert "private-room" not in path_rendered
    assert "/[REDACTED]" in path_rendered


def test_livekit_log_renderer_honors_tiny_line_bounds(tmp_path: Path) -> None:
    renderer = _renderer()
    render_log = renderer["render_log"]
    path = tmp_path / "tiny.err"
    path.write_text("ordinary diagnostic line", encoding="utf-8")

    rendered = render_log(path, max_line_chars=4)

    assert len(rendered) <= 4
