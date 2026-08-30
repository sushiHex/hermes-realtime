"""Strict hostname normalization for remotely reachable transport origins."""

from __future__ import annotations

import ipaddress
import socket


def canonical_remote_hostname(value: str, *, boundary: str) -> str:
    """Return a canonical ASCII hostname that is not a loopback/unspecified alias."""

    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"{boundary} hostname must be ASCII") from None

    lowered = value.lower()
    if lowered == "localhost" or lowered.endswith(".localhost"):
        raise ValueError(f"{boundary} hostname must be remotely reachable")

    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(value)))
        except OSError:
            labels = value.split(".")
            if (
                len(value) > 253
                or any(
                    not label
                    or len(label) > 63
                    or not label[0].isalnum()
                    or not label[-1].isalnum()
                    or not label.replace("-", "a").isalnum()
                    for label in labels
                )
            ):
                raise ValueError(f"{boundary} hostname is invalid") from None
            return lowered

    if address.is_loopback or address.is_unspecified:
        raise ValueError(f"{boundary} hostname must be remotely reachable")
    return address.compressed
