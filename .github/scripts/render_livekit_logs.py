"""Render bounded, sanitized tails of CI-owned LiveKit logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

_DEFAULT_MAX_BYTES = 65_536
_DEFAULT_MAX_LINES = 200
_DEFAULT_MAX_LINE_CHARS = 1_000
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_SAFE_LOOPBACK_PATHS = {"", "/", "/rtc/v1", "/rtc/v1/validate"}
_URL = re.compile(r"\b(?:https?|wss?)://[^\s\"'<>]+", re.IGNORECASE)
_ESCAPED_URL = re.compile(r"\b(?:https?|wss?):(?:\\/){2}[^\s\"'<>]+", re.IGNORECASE)
_JWT = re.compile(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_CREDENTIAL_LINE = re.compile(
    r"(?i)(\b(?:authorization|proxy-authorization|livekit[_-]?keys|"
    r"livekit[_-]?api[_-]?(?:key|secret))\s*[:=]\s*)[^\r\n]*"
)
_SENSITIVE_STEM = (
    r"(?:access[_-]?token|api[_-]?(?:key|secret|token)|authorization|id|identity|"
    r"livekit[_-]?keys|participant(?:[_-]?(?:id|identity))?|password|"
    r"room(?:[_-]?(?:id|identity|name))?|secret(?:[_-]?key)?|token|key)"
)
_SENSITIVE_NAME = re.compile(
    rf"(?:[A-Za-z0-9]+[_-])*{_SENSITIVE_STEM}\Z", re.IGNORECASE
)
_SENSITIVE_FIELD = re.compile(
    rf"(?i)([\"']?(?:[A-Za-z0-9]+[_-])*{_SENSITIVE_STEM}[\"']?\s*[:=]\s*)"
    r"(?:\"(?:\\.|[^\"\\\r\n])*\"|'(?:\\.|[^'\\\r\n])*'|[^\s,}\]\[]+)"
)
_LIVEKIT_SID = re.compile(r"\b(?:RM|PA|TR|PU|SU)_[A-Za-z0-9]{8,}\b")
_LONG_OPAQUE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")


def _sanitize_url(match: re.Match[str]) -> str:
    value = match.group(0)
    try:
        parsed = urlsplit(value)
        port_value = parsed.port
    except ValueError:
        return "[REDACTED-URL]"
    if (
        parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
    ):
        return "[REDACTED-URL]"
    host = parsed.hostname
    assert host is not None
    bracketed = f"[{host}]" if ":" in host else host
    port = "" if port_value is None else f":{port_value}"
    path = parsed.path if parsed.path in _SAFE_LOOPBACK_PATHS else "/[REDACTED]"
    query = "?[REDACTED]" if parsed.query else ""
    fragment = "#[REDACTED]" if parsed.fragment else ""
    return f"{parsed.scheme.lower()}://{bracketed}{port}{path}{query}{fragment}"


def _sanitize_text(value: str) -> str:
    value = _ESCAPED_URL.sub("[REDACTED-URL]", value)
    value = _URL.sub(_sanitize_url, value)
    value = _CREDENTIAL_LINE.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    value = _SENSITIVE_FIELD.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    value = _JWT.sub("[REDACTED]", value)
    value = _LIVEKIT_SID.sub("[REDACTED]", value)
    return _LONG_OPAQUE.sub("[REDACTED]", value)


def _sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_NAME.search(key) is not None else _sanitize_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _sanitize_line(value: str) -> str:
    printable = "".join(
        character if character >= " " or character == "\t" else "?" for character in value
    )
    if printable.startswith(("{", "[")):
        try:
            return json.dumps(_sanitize_json(json.loads(printable)), separators=(",", ":"))
        except (json.JSONDecodeError, RecursionError):
            pass
    return _sanitize_text(printable)


def _bound_line(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    marker = "...[truncated]"
    if maximum <= len(marker):
        return marker[:maximum]
    return value[: maximum - len(marker)] + marker


def render_log(
    path: Path,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_lines: int = _DEFAULT_MAX_LINES,
    max_line_chars: int = _DEFAULT_MAX_LINE_CHARS,
) -> str:
    """Return a bounded sanitized tail without exposing the host path."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib Path")
    for name, value in (
        ("max_bytes", max_bytes),
        ("max_lines", max_lines),
        ("max_line_chars", max_line_chars),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact integer")
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            payload = handle.read(max_bytes)
    except (OSError, ValueError):
        return "[log unavailable]"

    lines = payload.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes:
        lines = lines[1:]
    lines = lines[-max_lines:]
    rendered: list[str] = []
    if size > max_bytes:
        marker = f"[input truncated to final {max_bytes} bytes]"
        rendered.append(_bound_line(marker, max_line_chars))
    for raw_line in lines:
        rendered.append(_bound_line(_sanitize_line(raw_line), max_line_chars))
    return "\n".join(rendered) if rendered else "[log empty]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for path in args.paths:
        print(f"== {path.name} (sanitized tail) ==")
        print(render_log(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
