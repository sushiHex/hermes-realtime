"""Pure network configuration and desired policy for the IPv4 tailnet LiveKit profile.

These values are inputs to a future owner-run setup. Nothing here observes an adapter,
changes Windows policy, writes a config, or starts a process.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass

from scripts import local_livekit

_TAILNET = ipaddress.IPv4Network("100.64.0.0/10")


@dataclass(frozen=True)
class FirewallRuleSpec:
    name: str
    direction: str
    action: str
    enabled: bool
    program: str
    protocol: str
    local_port: int
    remote_address: str
    local_address: str
    interface_alias: str
    profile: str
    policy_store: str


def _tailnet_ipv4(address: str) -> str:
    if type(address) is not str:
        raise TypeError("tailnet address must be an exact IPv4 string")
    try:
        parsed = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError as error:
        raise ValueError("tailnet address must be a single IPv4 address") from error
    if parsed not in _TAILNET:
        raise ValueError("tailnet address is outside the IPv4 tailnet range")
    return str(parsed)


def render_config(address: str) -> str:
    """Render only the pinned LiveKit network fields, using one checked address."""

    ipv4 = _tailnet_ipv4(address)
    return (
        "port: 7880\n"
        "rtc:\n"
        "  use_external_ip: false\n"
        f"  node_ip: {ipv4}\n"
        "  tcp_port: 7881\n"
        "  udp_port: 7882\n"
        "  ips:\n"
        "    includes:\n"
        f"      - {ipv4}/32\n"
    )


def validate_credentials(api_key: str, api_secret: str) -> None:
    """Apply the existing remote launcher bounds before a future setup uses credentials."""

    if type(api_key) is not str or type(api_secret) is not str:
        raise TypeError("remote credentials must be exact strings")
    if (
        not api_key
        or api_key == "devkey"
        or api_secret == "local-" + "x" * 32
        or len(api_key) > 256
        or not 32 <= len(api_secret) <= 512
    ):
        raise ValueError("remote credentials are missing, developmental, or outside bounds")


def desired_firewall_rules(
    environ: Mapping[str, str], address: str, interface_alias: str, observed_profile: str
) -> tuple[FirewallRuleSpec, FirewallRuleSpec]:
    """Specify rules for the one shared remote path; the caller must verify its bytes."""

    program = str(local_livekit.remote_path(environ))
    ipv4 = _tailnet_ipv4(address)
    if type(interface_alias) is not str:
        raise TypeError("interface alias must be an exact string")
    if (
        not interface_alias
        or len(interface_alias) > 256
        or interface_alias != interface_alias.strip()
        or interface_alias.lower() == "any"
        or any(character in interface_alias for character in "*?,\r\n\x00")
        or any(ord(character) < 32 for character in interface_alias)
    ):
        raise ValueError("interface alias must name one observed adapter")
    if type(observed_profile) is not str:
        raise TypeError("observed profile must be an exact string")
    profiles = {"Public": "Public", "Private": "Private", "DomainAuthenticated": "Domain"}
    if observed_profile not in profiles:
        raise ValueError("observed profile is not a single Windows network category")
    profile = profiles[observed_profile]

    def rule(name: str, protocol: str, port: int) -> FirewallRuleSpec:
        return FirewallRuleSpec(
            name=name,
            direction="Inbound",
            action="Allow",
            enabled=True,
            program=program,
            protocol=protocol,
            local_port=port,
            remote_address=str(_TAILNET),
            local_address=ipv4,
            interface_alias=interface_alias,
            profile=profile,
            policy_store="PersistentStore",
        )

    return (
        rule("HermesRealtime.Tailnet.LiveKit.TCP.v1", "TCP", 7881),
        rule("HermesRealtime.Tailnet.LiveKit.UDP.v1", "UDP", 7882),
    )
