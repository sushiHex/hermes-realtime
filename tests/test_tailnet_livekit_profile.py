"""Pure first-slice contract for the separate LiveKit tailnet profile."""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from scripts import local_livekit, tailnet_livekit_profile


def test_remote_copy_is_separate_and_uses_the_same_verified_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = local_livekit.remote_path({"LOCALAPPDATA": str(tmp_path)})
    loopback = local_livekit.shared_path({"LOCALAPPDATA": str(tmp_path)})
    assert remote == (
        tmp_path / "hermes-realtime" / "tools" / "livekit-1.13.4-tailnet" / "livekit-server.exe"
    )
    assert remote != loopback
    assert not remote.is_relative_to(Path(__file__).resolve().parents[1])
    with pytest.raises(local_livekit.LiveKitUnavailable):
        local_livekit.remote_path({"LOCALAPPDATA": "relative"})

    binary = b"synthetic pinned binary"
    monkeypatch.setattr(local_livekit, "EXECUTABLE_SHA256", hashlib.sha256(binary).hexdigest())
    remote.parent.mkdir(parents=True)
    remote.write_bytes(binary + b"changed")
    with pytest.raises(local_livekit.LiveKitUnavailable, match="SHA-256"):
        local_livekit.verified_remote_server({"LOCALAPPDATA": str(tmp_path)})
    remote.write_bytes(binary)
    assert local_livekit.verified_remote_server({"LOCALAPPDATA": str(tmp_path)}) == remote
    with pytest.raises(local_livekit.LiveKitUnavailable, match="not installed"):
        local_livekit.verified_server({"LOCALAPPDATA": str(tmp_path)})


def test_remote_install_reuses_archive_inspection_without_touching_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = b"synthetic pinned binary"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("livekit-server.exe", binary)
        bundle.writestr("LICENSE", b"synthetic notice")
    archive = stream.getvalue()
    monkeypatch.setattr(local_livekit, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(local_livekit, "EXECUTABLE_SHA256", hashlib.sha256(binary).hexdigest())
    environ = {"LOCALAPPDATA": str(tmp_path)}

    remote = local_livekit.install_remote(environ, fetch=lambda url: archive)
    assert remote == local_livekit.remote_path(environ)
    assert remote.read_bytes() == binary
    assert not local_livekit.shared_path(environ).exists()
    assert local_livekit.install_remote(environ, fetch=lambda url: b"wrong") == remote


def test_remote_install_refuses_an_archive_pin_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_livekit, "ARCHIVE_SHA256", hashlib.sha256(b"expected").hexdigest())
    environ = {"LOCALAPPDATA": str(tmp_path)}
    with pytest.raises(local_livekit.LiveKitUnavailable, match="archive differs"):
        local_livekit.install_remote(environ, fetch=lambda url: b"changed")
    assert not local_livekit.remote_path(environ).parent.exists()


def test_remote_path_command_only_prints_a_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = b"synthetic pinned binary"
    monkeypatch.setattr(local_livekit, "EXECUTABLE_SHA256", hashlib.sha256(binary).hexdigest())
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    remote = local_livekit.remote_path()
    remote.parent.mkdir(parents=True)
    remote.write_bytes(binary + b"changed")
    assert local_livekit.main(["path-remote"]) == 1
    assert capsys.readouterr().out == ""
    remote.write_bytes(binary)
    assert local_livekit.main(["path-remote"]) == 0
    assert capsys.readouterr().out == f"{remote}\n"


def test_remote_config_has_only_the_fixed_network_fields() -> None:
    assert tailnet_livekit_profile.render_config("100.101.102.103") == (
        "port: 7880\n"
        "rtc:\n"
        "  use_external_ip: false\n"
        "  node_ip: 100.101.102.103\n"
        "  tcp_port: 7881\n"
        "  udp_port: 7882\n"
        "  ips:\n"
        "    includes:\n"
        "      - 100.101.102.103/32\n"
    )


@pytest.mark.parametrize(
    "address",
    [
        "100.63.0.1", "100.128.0.1", "127.0.0.1", "::1", "100.64.0.1/32",
        "100.64.0.1\nkeys: devkey", 7,
    ],
)
def test_remote_config_refuses_addresses_outside_the_ipv4_tailnet_scope(address: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tailnet_livekit_profile.render_config(address)  # type: ignore[arg-type]


def test_desired_rules_bind_every_required_filter_by_exact_value(tmp_path: Path) -> None:
    executable = local_livekit.remote_path({"LOCALAPPDATA": str(tmp_path)})
    rules = tailnet_livekit_profile.desired_firewall_rules(
        executable, "100.101.102.103", "Tailscale", "Private"
    )
    assert [rule.name for rule in rules] == [
        "HermesRealtime.Tailnet.LiveKit.TCP.v1",
        "HermesRealtime.Tailnet.LiveKit.UDP.v1",
    ]
    assert [rule.protocol for rule in rules] == ["TCP", "UDP"]
    assert [rule.local_port for rule in rules] == [7881, 7882]
    for rule in rules:
        assert rule.direction == "Inbound"
        assert rule.action == "Allow"
        assert rule.enabled is True
        assert rule.program == str(executable)
        assert rule.remote_address == "100.64.0.0/10"
        assert rule.local_address == "100.101.102.103"
        assert rule.interface_alias == "Tailscale"
        assert rule.profile == "Private"
        assert rule.policy_store == "PersistentStore"


def test_desired_rules_refuse_other_program_paths_and_addresses(tmp_path: Path) -> None:
    environ = {"LOCALAPPDATA": str(tmp_path)}
    loopback = local_livekit.shared_path(environ)
    with pytest.raises(ValueError, match="separate pinned profile"):
        tailnet_livekit_profile.desired_firewall_rules(
            loopback, "100.101.102.103", "Tailscale", "Private"
        )
    with pytest.raises(ValueError, match="outside"):
        tailnet_livekit_profile.desired_firewall_rules(
            local_livekit.remote_path(environ), "192.0.2.1", "Tailscale", "Private"
        )


def test_domain_authenticated_category_maps_to_one_firewall_profile(tmp_path: Path) -> None:
    rules = tailnet_livekit_profile.desired_firewall_rules(
        local_livekit.remote_path({"LOCALAPPDATA": str(tmp_path)}),
        "100.101.102.103",
        "Tailscale",
        "DomainAuthenticated",
    )
    assert [rule.profile for rule in rules] == ["Domain", "Domain"]


@pytest.mark.parametrize("alias", ["", "Any", "Tailscale\nUnexpected", 3])
def test_desired_rules_refuse_unbounded_interface_alias(tmp_path: Path, alias: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tailnet_livekit_profile.desired_firewall_rules(
            local_livekit.remote_path({"LOCALAPPDATA": str(tmp_path)}),
            "100.101.102.103",
            alias,  # type: ignore[arg-type]
            "Private",
        )


@pytest.mark.parametrize("profile", ["", "Any", "Private,Public", "Domain", "private", 3])
def test_desired_rules_refuse_unobserved_or_broad_profile(tmp_path: Path, profile: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tailnet_livekit_profile.desired_firewall_rules(
            local_livekit.remote_path({"LOCALAPPDATA": str(tmp_path)}),
            "100.101.102.103",
            "Tailscale",
            profile,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "key,secret",
    [
        ("devkey", "z" * 32),
        ("other", "local-" + "x" * 32),
        ("", "z" * 32),
        ("other", "short"),
    ],
)
def test_remote_credentials_refuse_development_and_weak_values(key: str, secret: str) -> None:
    with pytest.raises(ValueError, match="remote credentials"):
        tailnet_livekit_profile.validate_credentials(key, secret)


def test_remote_credentials_accept_bounded_synthetic_values() -> None:
    tailnet_livekit_profile.validate_credentials("synthetic-key", "z" * 32)


@pytest.mark.parametrize("key,secret", [("k" * 257, "z" * 32), ("key", "z" * 513)])
def test_remote_credentials_refuse_oversized_values(key: str, secret: str) -> None:
    with pytest.raises(ValueError, match="remote credentials"):
        tailnet_livekit_profile.validate_credentials(key, secret)
