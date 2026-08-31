"""Shared credential and loopback helpers for installed-boundary gates."""

from __future__ import annotations

import socket
from pathlib import Path


def load_api_key(env_file: Path) -> str:
    """Load exactly one strong API server bearer without logging it."""

    if not isinstance(env_file, Path):
        raise TypeError("Hermes env file must be a Path")
    matches: list[str] = []
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("API_SERVER_KEY="):
            matches.append(line.split("=", 1)[1].strip().strip('"').strip("'"))
    if len(matches) != 1 or len(matches[0]) < 32:
        raise RuntimeError("Hermes env file must contain one explicit strong API_SERVER_KEY")
    return matches[0]


def available_port() -> int:
    """Reserve and release one currently available loopback TCP port."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
